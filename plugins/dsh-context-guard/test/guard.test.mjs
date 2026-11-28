import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { existsSync } from 'node:fs'
import { test } from 'node:test'
import { mkdtemp, readdir, readFile, rm, writeFile } from 'node:fs/promises'
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
  evaluateIntervention,
  buildInterventionPrompt,
} from '../lib/index.js'
import {
  DEFAULT_EMB_MODEL,
  DEFAULT_HF_MODEL_FILES,
  DEFAULT_HF_REVISION,
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

test('incremental sync, real-time p_degraded scoring, semanticSearch, timeline.txt, and /context-guard status', async t => {
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

  // /context-guard zero-arg status command (and rejects extra arguments)
  const statusCmd = await commands.get('context-guard').handler({ agent, rawInput: '', signal: new AbortController().signal })
  assert.equal(statusCmd.kind, 'success')
  assert.match(statusCmd.text, /^cards 3 \(scored 1, latest p=0\.72\)/)
  assert.match(statusCmd.text, /recent p_degraded: #2 run_code=0\.72/)

  const badArgCmd = await commands.get('context-guard').handler({ agent, rawInput: 'rebuild', signal: new AbortController().signal })
  assert.equal(badArgCmd.kind, 'error')
  assert.match(badArgCmd.text, /no arguments/)

  // Programmatic rebuildSession API still rebuilds guard.sqlite cleanly
  const rebuilt = await guard.rebuildSession(session)
  assert.equal(rebuilt.cardCount, 3)
  assert.equal(rebuilt.scoredCount, 1)
  assert.ok(mock.calls.summarize > beforeCalls.summarize)

  // Startup asset check must have been triggered automatically on ContextGuardEngine construction
  assert.equal(mock.calls.ensureReady, 1)

  // S3 window builder sanity check
  const { cards } = await guard.store.loadCards(session.id)
  const win = buildS3Window(cards, 2)
  assert.match(win, /\[ACTIVE CONSTRAINTS\]/)
  assert.match(win, /\[CURRENT ACTION ANCHOR \| run_code\]/)
})

test('LlamaGuardRuntime.ensureReady verifies SHA256 on HF models and llama.cpp archive downloads', async t => {
  const dir = await mkdtemp(join(tmpdir(), 'context-guard-dl-'))
  const fakeBin = join(dir, 'llama-server.exe')
  await writeFile(fakeBin, 'stub-binary')

  const sha256Of = buf => createHash('sha256').update(buf).digest('hex')
  const requestedUrls = []
  const baseEndpoint = 'https://mock-hf.local'
  const embBuf = Buffer.from('FAKE-EMB-GGUF')
  const sumBuf = Buffer.from('FAKE-SUM-GGUF')
  const judgeBuf = Buffer.from('FAKE-JUDGE-GGUF')
  const headBuf = Buffer.from(JSON.stringify({ hidden_size: 4, weight: [0.1, 0.2, 0.3, 0.4], bias: 0.02 }))
  const payloads = {
    [`${baseEndpoint}/Logic-z/dsh-cortex/resolve/${DEFAULT_HF_REVISION}/${DEFAULT_HF_MODEL_FILES.embedding}`]: embBuf,
    [`${baseEndpoint}/Logic-z/dsh-cortex/resolve/${DEFAULT_HF_REVISION}/${DEFAULT_HF_MODEL_FILES.summarizer}`]: sumBuf,
    [`${baseEndpoint}/Logic-z/dsh-cortex/resolve/${DEFAULT_HF_REVISION}/${DEFAULT_HF_MODEL_FILES.judge}`]: judgeBuf,
    [`${baseEndpoint}/Logic-z/dsh-cortex/resolve/${DEFAULT_HF_REVISION}/${DEFAULT_HF_MODEL_FILES.judgeHead}`]: headBuf,
  }

  const origFetch = globalThis.fetch
  globalThis.fetch = async url => {
    const u = String(url)
    requestedUrls.push(u)
    if (u.startsWith('https://github.com/ggml-org/llama.cpp/releases/download/')) {
      const tampered = Buffer.from('TAMPERED-ARCHIVE')
      return new Response(tampered, { status: 200, headers: { 'content-length': String(tampered.length) } })
    }
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
    modelSha256: {
      embedding: sha256Of(embBuf),
      summarizer: sha256Of(sumBuf),
      judge: sha256Of(judgeBuf),
      judgeHead: sha256Of(headBuf),
    },
    gpuLayers: 0,
    summarizerModelPath: resolveModelPath(storeRoot, undefined, DEFAULT_SUM_MODEL),
    judgeModelPath: resolveModelPath(storeRoot, undefined, DEFAULT_JUDGE_MODEL),
    judgeHeadPath: resolveModelPath(storeRoot, undefined, DEFAULT_JUDGE_HEAD),
    embeddingModelPath: resolveModelPath(storeRoot, undefined, DEFAULT_EMB_MODEL),
  })

  // First ensureReady downloads all 4 missing model files from the pinned revision and verifies SHA256
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

  // Tampered model payload fails pinned DEFAULT_HF_MODEL_SHA256 verification and cleans up temp file
  const tamperedModelRoot = join(dir, 'tampered-model')
  const tamperedModelRuntime = new LlamaGuardRuntime({
    storeRoot: tamperedModelRoot,
    llamaServerPath: fakeBin,
    llamaBuild: DEFAULT_LLAMA_BUILD,
    autoDownloadRuntime: false,
    autoDownloadModels: true,
    hfRepo: 'Logic-z/dsh-cortex',
    hfEndpoint: baseEndpoint,
    gpuLayers: 0,
    summarizerModelPath: resolveModelPath(tamperedModelRoot, undefined, DEFAULT_SUM_MODEL),
    judgeModelPath: resolveModelPath(tamperedModelRoot, undefined, DEFAULT_JUDGE_MODEL),
    judgeHeadPath: resolveModelPath(tamperedModelRoot, undefined, DEFAULT_JUDGE_HEAD),
    embeddingModelPath: resolveModelPath(tamperedModelRoot, undefined, DEFAULT_EMB_MODEL),
  })
  await assert.rejects(() => tamperedModelRuntime.ensureReady(), /SHA256 mismatch/)
  assert.equal(existsSync(resolveModelPath(tamperedModelRoot, undefined, DEFAULT_EMB_MODEL)), false)
  assert.deepEqual(await readdir(join(tamperedModelRoot, 'models')), [])

  // Tampered llama.cpp runtime archive fails pinned DEFAULT_LLAMA_ARCHIVE_SHA256 verification before extraction
  const tamperedBinRoot = join(dir, 'tampered-bin')
  const tamperedBinRuntime = new LlamaGuardRuntime({
    storeRoot: tamperedBinRoot,
    llamaBuild: DEFAULT_LLAMA_BUILD,
    autoDownloadRuntime: true,
    autoDownloadModels: false,
    gpuLayers: 0,
    summarizerModelPath: resolveModelPath(tamperedBinRoot, undefined, DEFAULT_SUM_MODEL),
    judgeModelPath: resolveModelPath(tamperedBinRoot, undefined, DEFAULT_JUDGE_MODEL),
    judgeHeadPath: resolveModelPath(tamperedBinRoot, undefined, DEFAULT_JUDGE_HEAD),
    embeddingModelPath: resolveModelPath(tamperedBinRoot, undefined, DEFAULT_EMB_MODEL),
  })
  await assert.rejects(() => tamperedBinRuntime.ensureReady(), /SHA256 mismatch/)
  assert.deepEqual(await readdir(join(tamperedBinRoot, 'runtime', DEFAULT_LLAMA_BUILD)), [])

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

test('evaluateIntervention enforces 8-step cooldown after start/compact/user/intervention and triggers on 2 hits in 16 steps', () => {
  const policy = { degradedThreshold: 0.5, hitWindowSteps: 16, hitsToTrigger: 2, cooldownSteps: 8 }
  const makeAction = (idx, p, compactBefore = false) => ({
    cardIdx: idx,
    startEventId: idx * 2,
    endEventId: idx * 2 + 1,
    kind: 'action',
    tool: 'run_code',
    callText: 'cmd',
    resText: 'out',
    compactBefore,
    micro: `#${idx} [run_code] cmd -> out`,
    embedding: new Float32Array(4),
    pDegraded: p,
    createdAt: 0,
  })
  const makeUser = idx => ({
    cardIdx: idx,
    startEventId: idx * 2,
    endEventId: idx * 2,
    kind: 'constraint',
    tool: '',
    callText: 'Fix SOFA mismatch without dropping stays.',
    resText: '',
    compactBefore: false,
    micro: `#${idx} [USER] Fix SOFA mismatch`,
    embedding: new Float32Array(4),
    pDegraded: null,
    createdAt: 0,
  })

  // 1. Cooldown after user message: 8 high-score actions right after user must NOT trigger
  const cards = [makeUser(0)]
  for (let i = 1; i <= 8; i++) cards.push(makeAction(i, 0.7))
  assert.equal(evaluateIntervention(cards, -1, policy).shouldIntervene, false)

  // 2. 9th action with p >= 0.50: window has >= 2 hits and cooldown elapsed -> triggers!
  cards.push(makeAction(9, 0.55))
  const ev1 = evaluateIntervention(cards, -1, policy)
  assert.equal(ev1.shouldIntervene, true)
  assert.equal(ev1.triggerCardIdx, 9)

  // 3. After recording intervention at cardIdx=9, next 8 actions must stay silent (cooldown)
  for (let i = 10; i <= 17; i++) cards.push(makeAction(i, 0.65))
  assert.equal(evaluateIntervention(cards, 9, policy).shouldIntervene, false)

  // 4. 9th action after intervention (cardIdx=18) with p < 0.50 must NOT trigger (current step not a hit)
  cards.push(makeAction(18, 0.30))
  assert.equal(evaluateIntervention(cards, 9, policy).shouldIntervene, false)

  // 5. 10th action after intervention (cardIdx=19) with p >= 0.50 -> triggers again
  cards.push(makeAction(19, 0.52))
  assert.equal(evaluateIntervention(cards, 9, policy).shouldIntervene, true)

  // 6. Compaction resets cooldown for 8 actions
  cards.push(makeAction(20, 0.70, true))
  for (let i = 21; i <= 27; i++) cards.push(makeAction(i, 0.70))
  assert.equal(evaluateIntervention(cards, 9, policy).shouldIntervene, false)
  cards.push(makeAction(28, 0.70))
  assert.equal(evaluateIntervention(cards, 9, policy).shouldIntervene, true)

  // 7. Intervention prompt is zero-pronoun and includes active constraints + observed actions
  const prompt = buildInterventionPrompt(cards, { hasHistoryTools: true })
  assert.match(prompt, /<system-reminder>/)
  assert.match(prompt, /<\/system-reminder>/)
  assert.doesNotMatch(prompt, /level=/)
  assert.match(prompt, /Fix SOFA mismatch/)
  assert.match(prompt, /Observed recent actions:/)
  assert.match(prompt, /history_search/)
  assert.doesNotMatch(prompt, /\b(you|your|yours|I|we|our)\b/i)
})

test('tools/post-execute appends checkpoint reminder onto tool result without creating a user message and cards.ts strips it', async t => {
  const { ctx, session, agent, guard } = await fixture(t, { cooldownSteps: 1, hitsToTrigger: 2, hitWindowSteps: 16 })

  session.append('user/message', createUserMessage({ content: [{ type: 'text', text: 'Investigate SOFA mismatch.' }], source: { kind: 'user' } }), { surfaceOp: 'append' })
  // Append 3 actions; mock judge returns 0.72 for SOFA mismatch (cardIdx 2 & 3 -> 2 hits, > 1 cooldown)
  for (let s = 1; s <= 3; s++) {
    session.append('tool/call', { turn: 1, step: s, callId: `c${s}`, name: 'run_code', arguments: '{"cmd":"python sofa.py"}' })
    session.append('tool/result', {
      turn: 1, step: s,
      message: { id: `tr${s}`, role: 'tool', toolCallId: `c${s}`, source: { kind: 'tool', callId: `c${s}` }, content: [{ type: 'text', text: 'SOFA mismatch!' }] },
    }, { surfaceOp: 'append' })
  }

  await guard.syncSession(session)

  // Simulate next root tool completion: tools/post-execute appends <system-reminder> onto result.content (no UserMessage!)
  const toolResult = { isError: false, value: 'ok', content: [{ type: 'text', text: 'still mismatching' }] }
  const postDecision = await ctx.waterfall(
    'tools/post-execute',
    { name: 'run_code', arguments: {}, agent, callId: 'c4', rootCallId: 'c4', signal: new AbortController().signal },
    toolResult,
    async () => ({ kind: 'accept' }),
  )

  assert.equal(postDecision.kind, 'accept')
  assert.equal(postDecision.additionalContexts, undefined)
  assert.equal(postDecision.content.length, 2)
  assert.equal(postDecision.content[0].text, 'still mismatching')
  assert.match(postDecision.content[1].text, /<system-reminder>/)
  assert.match(postDecision.content[1].text, /<\/system-reminder>/)

  // Persisting that enriched tool/result into session must NOT leak <system-reminder> into extracted card resText
  session.append('tool/call', { turn: 1, step: 4, callId: 'c4', name: 'run_code', arguments: '{"cmd":"python sofa.py"}' })
  session.append('tool/result', {
    turn: 1, step: 4,
    message: { id: 'tr4', role: 'tool', toolCallId: 'c4', source: { kind: 'tool', callId: 'c4' }, content: postDecision.content },
  }, { surfaceOp: 'append' })
  const obs = await ctx.sessionQuery.observeSession(session.id, { projectionMode: 'none' })
  let evs
  try { evs = [...obs.events] } finally { obs[Symbol.dispose]() }
  const cards = extractCardsFromEvents(evs)
  assert.equal(cards.filter(c => c.kind === 'constraint').length, 1)
  const lastAction = cards[cards.length - 1]
  assert.equal(lastAction.resText, 'still mismatching')
  assert.doesNotMatch(lastAction.resText, /<system-reminder>/)
})

test('escalates to full-prefix compactRegion on 3rd degradation hit after 2 reminders in one turn and continues turn', async t => {
  const { ctx, session, agent, commands, guard } = await fixture(t, {
    cooldownSteps: 1,
    hitsToTrigger: 2,
    hitWindowSteps: 16,
    enableAutoCompact: true,
    interventionsBeforeCompact: 2,
  })

  const compactCalls = []
  ctx.provide('compaction', {
    async compactRegion(start, end, targetAgent) {
      compactCalls.push({ start, end, sessionId: targetAgent.session.id })
      return { shadowedRange: { start, end }, shadowedSeqs: [], shadowedTokenCount: 100 }
    },
  })

  // Leading system/message (must be skipped by full-prefix compact range) + real user message
  session.append('system/message', { id: 'sys0', role: 'system', content: [{ type: 'text', text: 'System prompt' }], source: { kind: 'system-prompt' } }, { surfaceOp: 'append' })
  const userEv = session.append('user/message', createUserMessage({ content: [{ type: 'text', text: 'Investigate SOFA mismatch.' }], source: { kind: 'user' } }), { surfaceOp: 'append' })

  let stepNum = 0
  const addAction = () => {
    stepNum++
    const callId = `c${stepNum}`
    session.append('assistant/message', {
      turn: 1, step: stepNum, stream: [],
      message: createAssistantMessage({
        source: { provider: 'p', model: 'm' },
        content: [{ type: 'tool-call', id: callId, name: 'run_code', arguments: '{"cmd":"python sofa.py"}' }],
      }),
    }, { surfaceOp: 'append' })
    session.append('tool/call', { turn: 1, step: stepNum, callId, name: 'run_code', arguments: '{"cmd":"python sofa.py"}' })
    return session.append('tool/result', {
      turn: 1, step: stepNum,
      message: { id: `tr${stepNum}`, role: 'tool', toolCallId: callId, source: { kind: 'tool', callId }, content: [{ type: 'text', text: 'SOFA mismatch!' }] },
    }, { surfaceOp: 'append' })
  }

  const runPostExecute = async () => ctx.waterfall(
    'tools/post-execute',
    { name: 'run_code', arguments: {}, agent, callId: `c${stepNum}`, rootCallId: `c${stepNum}`, signal: new AbortController().signal },
    { isError: false, value: 'ok', content: [{ type: 'text', text: 'SOFA mismatch!' }] },
    async () => ({ kind: 'accept' }),
  )

  const runPreStep = async () => ctx.waterfall(
    'agent/pre-step',
    { agent, messages: [], turn: 1, step: stepNum + 1, signal: new AbortController().signal, modelSelection: { provider: 'p', model: 'm' } },
    async () => ({ kind: 'enter', messages: [] }),
  )

  // Hit #1 (3 actions -> 2 hits, > 1 cooldown) => Reminder #1
  addAction(); addAction(); addAction()
  await guard.syncSession(session)
  const d1 = await runPostExecute()
  assert.equal(d1.content.length, 2)
  assert.match(d1.content[1].text, /<system-reminder>/)
  assert.equal(compactCalls.length, 0)

  // Hit #2 (2 more actions after cooldown=1) => Reminder #2
  addAction(); addAction()
  await guard.syncSession(session)
  const d2 = await runPostExecute()
  assert.equal(d2.content.length, 2)
  assert.match(d2.content[1].text, /<system-reminder>/)
  assert.equal(compactCalls.length, 0)

  // Hit #3 (2 more actions after cooldown=1, already had 2 reminders in this turn) => Auto-Compact!
  addAction()
  const lastToolEv = addAction()
  await guard.syncSession(session)

  // Should NOT emit a 3rd reminder in tools/post-execute
  const d3 = await runPostExecute()
  assert.equal(d3.content, undefined)

  // agent/pre-step executes compactRegion(userEv.seq .. lastToolEv.seq) and still enters next step!
  const preDecision = await runPreStep()
  assert.equal(preDecision.kind, 'enter')
  assert.equal(compactCalls.length, 1)
  assert.equal(compactCalls[0].start, userEv.seq)
  assert.equal(compactCalls[0].end, lastToolEv.seq)

  // Status reflects autoCompactCount=1 and reset turnCount=0
  const statusCmd = await commands.get('context-guard').handler({ agent, rawInput: '', signal: new AbortController().signal })
  assert.equal(statusCmd.kind, 'success')
  assert.match(statusCmd.text, /interventions 2, auto-compact 1/)
  assert.match(statusCmd.text, /total 2 \(turn 0\), auto-compact 1/)
  assert.doesNotMatch(statusCmd.text, /auto-compact queued/)
})
