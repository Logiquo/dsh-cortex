import assert from 'node:assert/strict'
import { test } from 'node:test'
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir, homedir } from 'node:os'
import { join } from 'node:path'
import { Context } from '@deepseek-ai/cordis'
import { SessionStore, SessionId } from '@deepseek-ai/dsh-session'
import { SessionProjectionRegistry } from '@deepseek-ai/dsh-session-projection'
import { SessionQueryEngine } from '@deepseek-ai/dsh-session-query'
import { TokenMeter } from '@deepseek-ai/dsh-token-meter'
import { LlmRuntime, LlmAdapter, createUserMessage, createAssistantMessage, createSystemMessage } from '@deepseek-ai/dsh-llm'
import * as plugin from '../lib/index.js'
import { Config, archiveRoot, basicCompactionConfig } from '../lib/config.js'
import { ArchiveStore } from '../lib/archive.js'
import { briefTargetTokens } from '../lib/brief.js'

const brief = 'Task: Implement the feature while preserving violet-42. Next action: inspect the implementation and run the relevant checks. Completion requires those checks to pass.'
const user = text => createUserMessage({ content: [{ type: 'text', text }], source: { kind: 'user' } })

class Query extends SessionQueryEngine {
  async searchSessions() { throw Error('No search index needed') }
  async searchEvents() { throw Error('No search index needed') }
}

async function fixture(t, config = {}, behavior = {}) {
  const directory = await mkdtemp(join(tmpdir(), 'context-offload-test-'))
  const ctx = new Context()
  const fibers = []
  const mount = async (p, c = {}) => { const f = ctx.plugin(p, c); fibers.push(f); await f.await(); return f }
  await mount(SessionProjectionRegistry)
  await mount(SessionStore)
  await mount(LlmRuntime)
  await mount(TokenMeter)
  await mount(Query)

  const calls = []
  const modelWindows = behavior.modelWindows ?? { model: 100000 }
  class Fake extends LlmAdapter {
    async resolveModel(provider, model) {
      const contextWindow = modelWindows[model] ?? 100000
      return {
        provider,
        id: model,
        name: model,
        context: { contextWindow },
        defaultMaxTokens: behavior.defaultMaxTokens ?? 2048,
      }
    }
    async *stream(options) {
      calls.push(options)
      if (behavior.onCall) await behavior.onCall(options)
      if (behavior.reasoning !== undefined) yield { type: 'reasoning-delta', index: 0, text: behavior.reasoning }
      yield { type: 'text-delta', index: 1, text: behavior.text ?? brief }
      yield { type: 'finish', reason: { kind: behavior.finish ?? 'stop' } }
    }
  }
  ctx.llm.registerAdapter(['probe'], new Fake())

  const agents = []
  ctx.provide('agents', { list: () => agents })

  const commands = new Map()
  ctx.provide('commands', {
    register: definition => {
      commands.set(definition.name, definition)
      return () => commands.delete(definition.name)
    },
  })

  const registeredTools = new Map()
  ctx.provide('tools', {
    register: (definition, options) => {
      registeredTools.set(definition.name, { definition, options })
      return () => registeredTools.delete(definition.name)
    },
  })

  const fiber = await mount(plugin, { 'spill-root': join(directory, 'archive'), ...config })
  const session = ctx.sessions.create(SessionId(crypto.randomUUID()))
  const agent = {
    session,
    options: { provider: 'probe', model: behavior.model ?? 'model' },
    runMaintenance: operation => operation(new AbortController().signal),
  }
  agents.push(agent)

  t.after(async () => {
    for (const f of fibers.reverse()) await f.dispose()
    await rm(directory, { recursive: true, force: true })
  })

  return { directory, ctx, session, agent, calls, commands, registeredTools, fiber }
}

function conversation(session, turn, text = 'Details and context. '.repeat(4000), model = 'model') {
  session.append('turn/start', { turn })
  session.append('user/message', user(text), { surfaceOp: 'append' })
  session.append('step/start', { turn, step: 1 })
  session.append('request/header', { header: { config: { provider: 'probe', model, maxTokens: 2048 } }, reason: 'initial' })
  session.append(
    'assistant/message',
    {
      turn,
      step: 1,
      stream: [],
      message: createAssistantMessage({
        source: { provider: 'probe', model },
        content: [{ type: 'text', text: `Completed work for turn ${turn}.` }],
      }),
    },
    { surfaceOp: 'append' },
  )
  session.append('step/end', { turn, step: 1 })
  session.append('turn/end', { turn, reason: { kind: 'completed' } })
}

async function events(ctx, session) {
  const cut = await ctx.sessionQuery.observeSession(session.id, { projectionMode: 'none' })
  try { return [...cut.events] } finally { cut[Symbol.dispose]() }
}

test('portable defaults, BasicCompactionConfig pass-through, and brief target scaling', () => {
  const config = Config({})
  assert.equal(archiveRoot(config), join(homedir(), '.dsh', 'context-offload'))
  assert.throws(() => archiveRoot(Config({ 'spill-root': 'relative' })), /absolute/)
  assert.deepEqual(basicCompactionConfig({ 'spill-root': '/tmp/x', thresholdRatio: 0.7 }), { thresholdRatio: 0.7 })
  assert.equal(briefTargetTokens(undefined), 2048)
  assert.equal(briefTargetTokens(16000), 1024)
  assert.equal(briefTargetTokens(64000), 2560)
  assert.equal(briefTargetTokens(200000), 4096)
})

test('forward-looking brief commits with searchable history and tool guidance', async t => {
  const { ctx, agent, session, calls } = await fixture(t)
  conversation(session, 1, 'Original exact fact: violet-42.\n' + 'History data\n'.repeat(7000))
  const result = await ctx.compaction.compactNow(agent, new AbortController().signal)
  assert.ok(result)
  const summary = result.summary[0].text
  assert.match(summary, /Completion requires those checks to pass/)
  assert.match(summary, /history_search/)
  assert.match(summary, /history_read/)
  const catalog = await ctx.compaction.archives.verify(session.id)
  assert.ok(catalog.through >= Number(result.endSeq))
  assert.match(catalog.chunks[0].file, /context-offload-/)
  const cut = await ctx.compaction.archives.ensure(session.id, await events(ctx, session))
  const archivedText = (await Promise.all(cut.catalog.chunks.map(chunk => readFile(join(cut.directory, chunk.file), 'utf8')))).join('\n')
  assert.match(archivedText, /Original exact fact: violet-42/)
  assert.equal(calls.length, 1)
  assert.equal(calls[0].purpose, 'compaction')
  assert.equal(calls[0].maxTokens, 65536)
  const promptText = calls[0].messages.at(-1).content[0].text
  assert.match(promptText, /Dead Ends & Decisions/)
  assert.match(promptText, /Archive Pointers/)
  const logged = await events(ctx, session)
  const summaryIndex = logged.findIndex(e => e.type === 'compaction/summary')
  assert.equal(logged[summaryIndex + 1].type, 'user/message')
  assert.ok(session.surface.replaceGeneration > 0)
})

test('reasoning is neither charged to the brief nor retained in history', async t => {
  const reasoning = 'Deliberation that never enters history. '.repeat(400)
  const { ctx, agent, session, calls } = await fixture(t, {}, { reasoning })
  conversation(session, 1, 'Original exact fact: violet-42.\n' + 'History data\n'.repeat(7000))
  const result = await ctx.compaction.compactNow(agent, new AbortController().signal)
  assert.ok(result)
  assert.equal(calls.length, 1)
  assert.match(result.summary[0].text, /Completion requires those checks to pass/)
  const logged = await events(ctx, session)
  const summaryEvent = logged.find(e => e.type === 'compaction/summary')
  assert.deepEqual(summaryEvent.data.rawOutput.map(block => block.type), ['text'])
  assert.ok(!JSON.stringify(summaryEvent.data).includes('Deliberation that never enters history'))
})

test('multiple compactions and reopening the archive preserve canonical early records', async t => {
  const { ctx, agent, session } = await fixture(t)
  conversation(session, 1, 'EARLIEST-violet-42\n' + 'first long trace\n'.repeat(6000))
  await ctx.compaction.compactNow(agent, new AbortController().signal)
  conversation(session, 2, 'CORRECTION: violet-43 supersedes violet-42.\n' + 'second trace\n'.repeat(6000))
  await ctx.compaction.compactNow(agent, new AbortController().signal)
  const reopened = new ArchiveStore(ctx.compaction.archives.root)
  const cut = await reopened.ensure(session.id, await events(ctx, session))
  assert.equal(cut.compaction, 2)
  const all = (await Promise.all(cut.catalog.chunks.map(c => readFile(join(cut.directory, c.file), 'utf8')))).join('\n')
  assert.match(all, /EARLIEST-violet-42/)
  assert.match(all, /CORRECTION: violet-43/)
  const before = cut.catalog.chunks.length
  assert.equal((await reopened.ensure(session.id, await events(ctx, session))).catalog.chunks.length, before)
})

test('corrupt history prevents replacement even when cache is warm and leaves the surface intact', async t => {
  const { ctx, agent, session } = await fixture(t)
  conversation(session, 1)
  await ctx.sessions.flush(session)
  const store = ctx.compaction.archives
  const cut = await store.ensure(session.id, await events(ctx, session))
  await writeFile(join(cut.directory, cut.catalog.chunks[0].file), 'corrupted')
  const generation = session.surface.replaceGeneration
  await assert.rejects(
    ctx.compaction.compactNow(agent, new AbortController().signal),
    error => /checksum/.test(error.cause?.message ?? error.message),
  )
  assert.equal(session.surface.replaceGeneration, generation)
})

test('empty output fails on the first attempt without losing context', async t => {
  const { ctx, agent, session, calls } = await fixture(t, {}, { text: '  \n  ' })
  conversation(session, 1)
  await assert.rejects(
    ctx.compaction.compactNow(agent, new AbortController().signal),
    error => /brief is empty/.test(error.cause?.message ?? error.message),
  )
  assert.equal(calls.length, 1)
  assert.equal(session.surface.replaceGeneration, 0)
})

test('a tool-calls finish still commits usable text, as the host summarizer does', async t => {
  const { ctx, agent, session, calls } = await fixture(t, {}, { finish: 'tool-calls' })
  conversation(session, 1)
  const result = await ctx.compaction.compactNow(agent, new AbortController().signal)
  assert.ok(result)
  assert.equal(calls.length, 1)
  assert.match(result.summary[0].text, /Completion requires those checks to pass/)
})

test('a summary-error listener drives the only retry path, as in the host summarizer', async t => {
  const { ctx, agent, session, calls } = await fixture(t, {}, { text: '  \n  ' })
  conversation(session, 1)
  let recoveries = 0
  ctx.on('compaction/summary-error', () => ++recoveries === 1)
  await assert.rejects(
    ctx.compaction.compactNow(agent, new AbortController().signal),
    error => /brief is empty/.test(error.cause?.message ?? error.message),
  )
  assert.equal(calls.length, 2)
  assert.equal(recoveries, 2)
  assert.equal(session.surface.replaceGeneration, 0)
})

test('cancellation during brief generation preserves the original surface', async t => {
  const abort = new AbortController()
  const { ctx, agent, session } = await fixture(t, {}, { onCall: () => abort.abort(new Error('cancelled by test')) })
  conversation(session, 1)
  await assert.rejects(ctx.compaction.compactNow(agent, abort.signal), /cancelled by test/)
  assert.equal(session.surface.replaceGeneration, 0)
})

test('ordinary export and /context-offload status do not trigger inference and report dynamic contextWindow', async t => {
  const { ctx, agent, session, calls, commands } = await fixture(t, {}, { modelWindows: { model: 128000 } })
  conversation(session, 1, 'A short user request.')
  await ctx.sessions.flush(session)
  const status = await commands.get('context-offload').handler({ agent, rawInput: 'status', signal: new AbortController().signal })
  assert.equal(status.kind, 'success')
  const parsed = JSON.parse(status.text)
  assert.equal(parsed.provider, 'probe')
  assert.equal(parsed.model, 'model')
  assert.equal(parsed.contextWindow, 128000)
  assert.equal(parsed.thresholdRatio, 0.8)
  assert.equal(parsed.retainRatio, 0.16)
  assert.equal(calls.length, 0)
  assert.equal((await ctx.compaction.archives.verify(session.id)).sessionId, session.id)
})

test('uses adapter contextWindow and BasicCompactionEngine defaults for automatic pressure compaction', async t => {
  // Small model window (32,000) with headroomTokens=4096 so threshold = min(25600, 32000 - 2048 - 4096) = 25600.
  const { ctx, agent, session, calls } = await fixture(
    t,
    { headroomTokens: 4096 },
    { modelWindows: { small: 32000, large: 200000 }, model: 'small' },
  )
  session.append('system/message', { turn: 1, step: 1, message: createSystemMessage('') }, { surfaceOp: 'append' })
  conversation(session, 1, 'Historical material\n'.repeat(7000), 'small')
  conversation(session, 2, 'Recent material\n'.repeat(1500), 'small')
  session.append('turn/start', { turn: 3 })
  const original = [...session.surface.nodes]
  const before = ctx.tokenMeter.measure(session)
  assert.ok(before.totalTokens > 25600 && before.totalTokens < 100000)
  const result = await ctx.compaction.compactIfNeeded(agent, 'pressure', new AbortController().signal)
  assert.ok(result)
  assert.ok(result.shadowedSeqs.length < original.length)
  assert.equal(session.surface.nodes[0], original[0])
  assert.ok(!result.shadowedSeqs.includes(original[0]))
  assert.ok(session.surface.nodes.includes(original.at(-1)))
  assert.ok(ctx.tokenMeter.measure(session).totalTokens < before.totalTokens)
  assert.equal(calls.length, 1)
  session.append('turn/end', { turn: 3, reason: { kind: 'completed' } })
})

test('does not compact when adapter contextWindow is large enough for the current conversation', async t => {
  const { ctx, agent, session, calls } = await fixture(
    t,
    { headroomTokens: 4096 },
    { modelWindows: { large: 200000 }, model: 'large' },
  )
  conversation(session, 1, 'Historical material\n'.repeat(7000), 'large')
  session.append('turn/start', { turn: 2 })
  const result = await ctx.compaction.compactIfNeeded(agent, 'pressure', new AbortController().signal)
  assert.equal(result, null)
  assert.equal(calls.length, 0)
  session.append('turn/end', { turn: 2, reason: { kind: 'completed' } })
})

test('archive refuses diverging history and recovers a dead writer lock', async t => {
  const { ctx, session } = await fixture(t)
  conversation(session, 1, 'Canonical original')
  const store = ctx.compaction.archives
  const snapshot = await events(ctx, session)
  const cut = await store.ensure(session.id, snapshot)
  await writeFile(join(cut.directory, 'catalog.json.lock'), '2147483647\n')
  await store.ensure(session.id, snapshot)
  const changed = structuredClone(snapshot)
  changed[1].data = user('A different historical message')
  await assert.rejects(store.ensure(session.id, changed), /differs from archived prefix/)
})

test('provider truncation fails once and preserves the unmodified surface', async t => {
  const { ctx, session, agent, calls } = await fixture(t, {}, { finish: 'max-tokens' })
  conversation(session, 1)
  await assert.rejects(
    ctx.compaction.compactNow(agent, new AbortController().signal),
    error => /truncated at the token cap/.test(error.cause?.message ?? error.message),
  )
  assert.equal(calls.length, 1)
  assert.equal(session.surface.replaceGeneration, 0)
})

test('a main-session fork archives its inherited prefix under a different directory', async t => {
  const { ctx, session } = await fixture(t)
  conversation(session, 1, 'Inherited exact fact: violet-42.')
  await ctx.sessions.flush(session)
  const child = ctx.sessions.fork(session)
  await ctx.sessions.flush(child)
  const store = ctx.compaction.archives
  const source = await store.ensure(session.id, await events(ctx, session))
  const fork = await store.ensure(child.id, await events(ctx, child))
  assert.notEqual(fork.directory, source.directory)
  const text = (await Promise.all(fork.catalog.chunks.map(c => readFile(join(fork.directory, c.file), 'utf8')))).join('\n')
  assert.match(text, /Inherited exact fact: violet-42/)
})

test('history_search and history_read activate only after compaction and retrieve shadowed history', async t => {
  const { ctx, agent, session, registeredTools } = await fixture(t)
  const searchEntry = registeredTools.get('history_search')
  const readEntry = registeredTools.get('history_read')
  assert.ok(searchEntry && readEntry)
  assert.equal(searchEntry.definition.isConcurrencySafe({ query: 'violet' }), true)
  assert.equal(readEntry.definition.isConcurrencySafe({ fromSeq: 0 }), true)

  // Inactive before compaction.
  assert.equal(searchEntry.options.when({ agent }), false)
  assert.equal(readEntry.options.when({ agent }), false)

  conversation(session, 1, 'Early requirement: token=violet-42.\n' + 'padding line\n'.repeat(6000))
  await ctx.compaction.compactNow(agent, new AbortController().signal)
  conversation(session, 2, 'Later correction: token=violet-99 supersedes violet-42.')

  // Active after compaction for main-agent sessions, never for subagents.
  assert.equal(searchEntry.options.when({ agent }), true)
  assert.equal(readEntry.options.when({ agent }), true)
  const subSession = ctx.sessions.create(SessionId(crypto.randomUUID()), { origin: 'subagent' })
  assert.equal(searchEntry.options.when({ agent: { session: subSession } }), false)

  const exec = { agent, signal: new AbortController().signal }

  // Newest-first search surfaces the correction before the original mention.
  const searchRes = await searchEntry.definition.execute({ query: 'violet-' }, exec)
  assert.ok(searchRes.totalMatches >= 2)
  assert.match(searchRes.matches[0].snippet, /violet-99/)
  assert.ok(searchRes.matches.some(m => m.snippet.includes('violet-42')))

  // Oldest-first + eventTypes filter.
  const oldestRes = await searchEntry.definition.execute(
    { query: 'violet-\\d+', regex: true, order: 'oldest', eventTypes: ['user/message'], limit: 1 },
    exec,
  )
  assert.equal(oldestRes.matches.length, 1)
  assert.equal(oldestRes.hasMore, true)
  assert.match(oldestRes.matches[0].snippet, /violet-42/)

  // Invalid regex produces a clear error.
  await assert.rejects(
    searchEntry.definition.execute({ query: '[invalid', regex: true }, exec),
    /Invalid regex/,
  )

  // history_read reads the exact shadowed event with pagination.
  const hitSeq = oldestRes.matches[0].seq
  const page1 = await readEntry.definition.execute({ fromSeq: hitSeq, maxChars: 200, charOffset: 0 }, exec)
  assert.equal(page1.fromSeq, hitSeq)
  assert.equal(page1.toSeq, hitSeq)
  assert.equal(page1.truncated, true)
  assert.equal(page1.nextCharOffset, 200)
  assert.match(page1.content, /Early requirement: token=violet-42/)

  const page2 = await readEntry.definition.execute({ fromSeq: hitSeq, maxChars: 200, charOffset: page1.nextCharOffset }, exec)
  assert.equal(page2.content.length, 200)
  assert.notEqual(page2.content, page1.content)

  // Reversed range rejects cleanly.
  await assert.rejects(
    readEntry.definition.execute({ fromSeq: 5, toSeq: 2 }, exec),
    /toSeq/,
  )
})
