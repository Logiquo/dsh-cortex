import assert from 'node:assert/strict'
import { test } from 'node:test'
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir, homedir } from 'node:os'
import { join } from 'node:path'
import { Context } from '@deepseek-ai/cordis'
import { SessionStore, SessionId } from '@deepseek-ai/dsh-session'
import { SessionProjectionRegistry } from '@deepseek-ai/dsh-session-projection'
import { SessionQueryEngine } from '@deepseek-ai/dsh-session-query'
import { createUserMessage, createAssistantMessage } from '@deepseek-ai/dsh-llm'
import {
  Config,
  ContextGuardEngine,
  LlamaGuardRuntime,
  extractCardsFromEvents,
  buildS3Window,
} from '../lib/index.js'
import {
  DEFAULT_EMB_MODEL,
  DEFAULT_HF_MODEL_FILES,
  DEFAULT_JUDGE_HEAD,
  DEFAULT_JUDGE_MODEL,
  DEFAULT_LLAMA_BUILD,
  DEFAULT_SUM_MODEL,
  resolveModelPath,
  resolveStoreRoot,
} from '../lib/config.js'

class Query extends SessionQueryEngine {
  async searchSessions() { throw Error('No search index needed') }
  async searchEvents() { throw Error('No search index needed') }
}

/** Deterministic mock inference client for fast unit testing without GPU/llama-server. */
function makeMockInference() {
  const calls = { ensureReady: 0, summarize: 0, embed: 0, judge: 0 }
  const embedVec = text => {
    const v = new Float32Array(8)
    for (let i = 0; i < text.length; i++) v[i % 8] += text.charCodeAt(i) / 255
    let norm = 0
    for (let i = 0; i < 8; i++) norm += v[i] * v[i]
    norm = Math.sqrt(norm) || 1
    for (let i = 0; i < 8; i++) v[i] /= norm
    return v
  }
  return {
    calls,
    async ensureReady() {
      calls.ensureReady++
    },
    async summarize(card) {
      calls.summarize++
      if (card.kind === 'constraint') return `#${card.cardIdx} [USER] ${card.callText.slice(0, 60)}`
      if (card.kind === 'assistant') return `#${card.cardIdx} [ASSISTANT] ${card.callText.slice(0, 60)}`
      return `#${card.cardIdx} [${card.tool}] ${card.callText.slice(0, 40)} -> ${card.resText.slice(0, 40)}`
    },
    async embed(text) {
      calls.embed++
      return embedVec(text)
    },
    async judge(prompt) {
      calls.judge++
      return prompt.includes('SOFA mismatch') ? 0.72 : 0.12
    },
    status() { return { mock: true } },
    async dispose() {},
  }
}

async function fixture(t, config = {}) {
  const directory = await mkdtemp(join(tmpdir(), 'context-guard-test-'))
  const ctx = new Context()
  const fibers = []
  const mount = async (p, c = {}) => { const f = ctx.plugin(p, c); fibers.push(f); await f.await(); return f }
  await mount(SessionProjectionRegistry)
  await mount(SessionStore)
  await mount(Query)

  const agents = []
  ctx.provide('agents', { list: () => agents })

  const commands = new Map()
  ctx.provide('commands', {
    register: definition => {
      commands.set(definition.name, definition)
      return () => commands.delete(definition.name)
    },
  })

  const mock = makeMockInference()
  const guard = new ContextGuardEngine(ctx, { storeRoot: join(directory, 'cortex'), debounceMs: 5000, ...config }, { inference: mock })

  const session = ctx.sessions.create(SessionId(crypto.randomUUID()))
  const agent = { session }
  agents.push(agent)

  t.after(async () => {
    for (const f of fibers.reverse()) await f.dispose()
    guard.store.close()
    await rm(directory, { recursive: true, force: true })
  })

  return { directory, ctx, session, agent, commands, guard, mock }
}

test('default storeRoot points to ~/.dsh/context-cortex', () => {
  assert.equal(resolveStoreRoot(Config({})), join(homedir(), '.dsh', 'context-cortex'))
  assert.throws(() => resolveStoreRoot(Config({ storeRoot: 'relative' })), /absolute/)
})

test('extractCardsFromEvents filters strictly via dsh MessageSourceMap and pairs tool calls + final replies', async t => {
  const { ctx, session } = await fixture(t)

  // Real user message (source.kind === 'user', no form)
  session.append('user/message', createUserMessage({ content: [{ type: 'text', text: 'Fix SOFA mismatch in eICU.' }], source: { kind: 'user' } }), { surfaceOp: 'append' })

  // Harness-injected user/message (non-user source.kind -> MUST be ignored!)
  session.append('user/message', createUserMessage({ content: [{ type: 'text', text: 'The user changed the approval policy.' }], source: { kind: 'user-approval', form: 'notice', summary: 'approval' } }), { surfaceOp: 'append' })
  session.append('user/message', createUserMessage({ content: [{ type: 'text', text: '[model changed]' }], source: { kind: 'model-selection', form: 'notice', summary: 'model' } }), { surfaceOp: 'append' })

  // Intermediate narration before tool_call -> MUST be cleared when tool/call arrives!
  session.append('assistant/message', {
    turn: 1, step: 1, stream: [],
    message: createAssistantMessage({ source: { provider: 'p', model: 'm' }, content: [{ type: 'text', text: 'Let me check the file first.' }] }),
  }, { surfaceOp: 'append' })

  // Tool call + tool result -> merged into ONE action card with exact startEventId..endEventId
  session.append('tool/call', { turn: 1, step: 1, callId: 'call_1', name: 'run_code', arguments: '{"cmd":"python check_sofa.py"}' })
  session.append('tool/result', {
    turn: 1, step: 1,
    message: {
      id: 'msg_tr1',
      role: 'tool',
      toolCallId: 'call_1',
      source: { kind: 'tool', callId: 'call_1' },
      content: [{ type: 'text', text: 'SOFA mismatch: 14 stays dropped' }],
    },
  }, { surfaceOp: 'append' })

  // Final user-visible assistant reply (reasoning block ignored, text kept)
  session.append('assistant/message', {
    turn: 1, step: 2, stream: [],
    message: createAssistantMessage({
      source: { provider: 'p', model: 'm' },
      content: [
        { type: 'reasoning', text: 'secret internal CoT' },
        { type: 'text', text: 'Found 14 dropped stays due to missing FiO2.' },
      ],
    }),
  }, { surfaceOp: 'append' })
  session.append('turn/end', { turn: 1, reason: { kind: 'completed' } })

  const obs = await ctx.sessionQuery.observeSession(session.id, { projectionMode: 'none' })
  let evs
  try { evs = [...obs.events] } finally { obs[Symbol.dispose]() }

  const cards = extractCardsFromEvents(evs)
  assert.equal(cards.length, 3)
  assert.equal(cards[0].kind, 'constraint')
  assert.equal(cards[0].callText, 'Fix SOFA mismatch in eICU.')
  assert.equal(cards[1].kind, 'action')
  assert.equal(cards[1].tool, 'run_code')
  assert.match(cards[1].resText, /SOFA mismatch/)
  assert.ok(cards[1].endEventId > cards[1].startEventId)
  assert.equal(cards[2].kind, 'assistant')
  assert.equal(cards[2].callText, 'Found 14 dropped stays due to missing FiO2.')
  assert.ok(!cards[2].callText.includes('secret internal CoT'))
})

test('incremental sync, real-time p_degraded scoring, semanticSearch, timeline.txt, and /context-guard rebuild', async t => {
  const { ctx, session, agent, commands, guard, mock } = await fixture(t)

  // Turn 1: user + 2 actions (card_idx 2 is action -> triggers judge!)
  session.append('user/message', createUserMessage({ content: [{ type: 'text', text: 'Investigate SOFA mismatch.' }], source: { kind: 'user' } }), { surfaceOp: 'append' })
  session.append('tool/call', { turn: 1, step: 1, callId: 'c1', name: 'read_file', arguments: '{"path":"sofa.py"}' })
  session.append('tool/result', {
    turn: 1, step: 1,
    message: { id: 'tr1', role: 'tool', toolCallId: 'c1', source: { kind: 'tool', callId: 'c1' }, content: [{ type: 'text', text: 'def compute_sofa(): ...' }] },
  }, { surfaceOp: 'append' })
  session.append('tool/call', { turn: 1, step: 2, callId: 'c2', name: 'run_code', arguments: '{"cmd":"python sofa.py"}' })
  session.append('tool/result', {
    turn: 1, step: 2,
    message: { id: 'tr2', role: 'tool', toolCallId: 'c2', source: { kind: 'tool', callId: 'c2' }, content: [{ type: 'text', text: 'SOFA mismatch detected!' }] },
  }, { surfaceOp: 'append' })

  const st1 = await guard.syncSession(session)
  assert.equal(st1.cardCount, 3)
  assert.equal(st1.scoredCount, 1)
  assert.equal(st1.latestScores[0].pDegraded, 0.72)

  // Second sync with no new events must reuse 100% of cached cards (0 extra inference calls!)
  const beforeCalls = { ...mock.calls }
  await guard.syncSession(session)
  assert.deepEqual(mock.calls, beforeCalls)

  // timeline.txt written alongside guard.sqlite
  const tl = await readFile(join(st1.directory, 'timeline.txt'), 'utf8')
  assert.match(tl, /p=0\.720/)
  assert.match(tl, /\[run_code\]/)

  // semanticSearch, lexicalSearch, and listSummaries exposed on ctx.contextGuard
  const provider = ctx.get('contextGuard')
  assert.ok(provider)
  const hits = await provider.semanticSearch(session.id, 'SOFA mismatch', { topK: 2 })
  assert.equal(hits.length, 2)
  assert.ok(typeof hits[0].startId === 'number' && typeof hits[0].endId === 'number')

  const sumLex = await provider.lexicalSearch(session.id, { query: 'run_code.*SOFA', regex: true, limit: 5 })
  assert.equal(sumLex.length, 1)
  assert.match(sumLex[0].micro, /\[run_code\]/)

  const sumRange = await provider.listSummaries(session.id, { fromId: 0, toId: 100 })
  assert.equal(sumRange.cards.length, 3)
  assert.ok(sumRange.throughEventId >= 0)

  // /context-guard status & /context-guard rebuild commands
  const statusCmd = await commands.get('context-guard').handler({ agent, rawInput: 'status', signal: new AbortController().signal })
  assert.equal(statusCmd.kind, 'success')
  const parsed = JSON.parse(statusCmd.text)
  assert.equal(parsed.cardCount, 3)
  assert.equal(parsed.scoredCount, 1)

  const rebuildCmd = await commands.get('context-guard').handler({ agent, rawInput: 'rebuild', signal: new AbortController().signal })
  assert.equal(rebuildCmd.kind, 'success')
  assert.match(rebuildCmd.text, /Rebuilt guard\.sqlite/)
  assert.ok(mock.calls.summarize > beforeCalls.summarize)

  // Startup asset check must have been triggered automatically on ContextGuardEngine construction
  assert.equal(mock.calls.ensureReady, 1)

  // S3 window builder sanity check
  const { cards } = await guard.store.loadCards(session.id)
  const win = buildS3Window(cards, 2)
  assert.match(win, /\[ACTIVE CONSTRAINTS\]/)
  assert.match(win, /\[CURRENT ACTION ANCHOR \| run_code\]/)
})

test('LlamaGuardRuntime.ensureReady checks runtime and auto-downloads missing models from HF atomically', async t => {
  const dir = await mkdtemp(join(tmpdir(), 'context-guard-dl-'))
  const fakeBin = join(dir, 'llama-server.exe')
  await writeFile(fakeBin, 'stub-binary')

  const requestedUrls = []
  const baseEndpoint = 'https://mock-hf.local'
  const payloads = {
    [`${baseEndpoint}/Logic-z/dsh-cortex/resolve/main/${DEFAULT_HF_MODEL_FILES.embedding}`]: Buffer.from('FAKE-EMB-GGUF'),
    [`${baseEndpoint}/Logic-z/dsh-cortex/resolve/main/${DEFAULT_HF_MODEL_FILES.summarizer}`]: Buffer.from('FAKE-SUM-GGUF'),
    [`${baseEndpoint}/Logic-z/dsh-cortex/resolve/main/${DEFAULT_HF_MODEL_FILES.judge}`]: Buffer.from('FAKE-JUDGE-GGUF'),
    [`${baseEndpoint}/Logic-z/dsh-cortex/resolve/main/${DEFAULT_HF_MODEL_FILES.judgeHead}`]: Buffer.from(JSON.stringify({ hidden_size: 4, weight: [0.1, 0.2, 0.3, 0.4], bias: 0.02 })),
  }

  const origFetch = globalThis.fetch
  globalThis.fetch = async url => {
    const u = String(url)
    requestedUrls.push(u)
    const body = payloads[u]
    if (!body) return new Response('not found', { status: 404 })
    return new Response(body, {
      status: 200,
      headers: { 'content-length': String(body.length) },
    })
  }

  t.after(async () => {
    globalThis.fetch = origFetch
    await rm(dir, { recursive: true, force: true })
  })

  const storeRoot = join(dir, 'cortex')
  const runtime = new LlamaGuardRuntime({
    storeRoot,
    llamaServerPath: fakeBin,
    llamaBuild: DEFAULT_LLAMA_BUILD,
    autoDownloadRuntime: false,
    autoDownloadModels: true,
    hfRepo: 'Logic-z/dsh-cortex',
    hfEndpoint: baseEndpoint,
    gpuLayers: 0,
    summarizerModelPath: resolveModelPath(storeRoot, undefined, DEFAULT_SUM_MODEL),
    judgeModelPath: resolveModelPath(storeRoot, undefined, DEFAULT_JUDGE_MODEL),
    judgeHeadPath: resolveModelPath(storeRoot, undefined, DEFAULT_JUDGE_HEAD),
    embeddingModelPath: resolveModelPath(storeRoot, undefined, DEFAULT_EMB_MODEL),
  })

  // First ensureReady downloads all 4 missing model files from the mock HF endpoint
  await runtime.ensureReady()
  assert.equal(requestedUrls.length, 4)
  assert.equal(await readFile(resolveModelPath(storeRoot, undefined, DEFAULT_EMB_MODEL), 'utf8'), 'FAKE-EMB-GGUF')
  assert.equal(await readFile(resolveModelPath(storeRoot, undefined, DEFAULT_SUM_MODEL), 'utf8'), 'FAKE-SUM-GGUF')
  assert.equal(await readFile(resolveModelPath(storeRoot, undefined, DEFAULT_JUDGE_MODEL), 'utf8'), 'FAKE-JUDGE-GGUF')
  const head = JSON.parse(await readFile(resolveModelPath(storeRoot, undefined, DEFAULT_JUDGE_HEAD), 'utf8'))
  assert.equal(head.bias, 0.02)
  assert.equal(runtime.status().prepare.status, 'ready')

  // Second ensureReady finds all files already present and performs 0 downloads
  await runtime.ensureReady()
  assert.equal(requestedUrls.length, 4)

  // With autoDownloadModels disabled on an empty store, ensureReady fails cleanly
  const emptyRoot = join(dir, 'empty-cortex')
  const noDl = new LlamaGuardRuntime({
    storeRoot: emptyRoot,
    llamaServerPath: fakeBin,
    llamaBuild: DEFAULT_LLAMA_BUILD,
    autoDownloadRuntime: false,
    autoDownloadModels: false,
    gpuLayers: 0,
    summarizerModelPath: resolveModelPath(emptyRoot, undefined, DEFAULT_SUM_MODEL),
    judgeModelPath: resolveModelPath(emptyRoot, undefined, DEFAULT_JUDGE_MODEL),
    judgeHeadPath: resolveModelPath(emptyRoot, undefined, DEFAULT_JUDGE_HEAD),
    embeddingModelPath: resolveModelPath(emptyRoot, undefined, DEFAULT_EMB_MODEL),
  })
  await assert.rejects(() => noDl.ensureReady(), /autoDownloadModels is disabled/)
})
