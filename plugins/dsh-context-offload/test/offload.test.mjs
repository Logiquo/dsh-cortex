import assert from 'node:assert/strict'
import { test } from 'node:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir, homedir } from 'node:os'
import { join } from 'node:path'
import { DatabaseSync } from 'node:sqlite'
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

  const fiber = await mount(plugin, { storeRoot: join(directory, 'cortex'), ...config })
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

test('portable defaults (~/.dsh/context-cortex), BasicCompactionConfig pass-through, and brief target scaling', () => {
  const config = Config({})
  assert.equal(archiveRoot(config), join(homedir(), '.dsh', 'context-cortex'))
  assert.throws(() => archiveRoot(Config({ storeRoot: 'relative' })), /absolute/)
  assert.deepEqual(basicCompactionConfig({ storeRoot: '/tmp/x', 'spill-root': '/tmp/y', thresholdRatio: 0.7 }), { thresholdRatio: 0.7 })
  assert.equal(briefTargetTokens(undefined), 2048)
  assert.equal(briefTargetTokens(16000), 1024)
  assert.equal(briefTargetTokens(64000), 2560)
  assert.equal(briefTargetTokens(200000), 4096)
})

test('forward-looking brief commits to offload.sqlite (events, compact) with tool guidance', async t => {
  const { ctx, agent, session, calls } = await fixture(t)
  conversation(session, 1, 'Original exact fact: violet-42.\n' + 'History data\n'.repeat(7000))
  const result = await ctx.compaction.compactNow(agent, new AbortController().signal)
  assert.ok(result)
  const summary = result.summary[0].text
  assert.match(summary, /Completion requires those checks to pass/)
  assert.match(summary, /history_search/)
  assert.match(summary, /history_read/)
  assert.match(summary, /offload\.sqlite/)

  const status = await ctx.compaction.archives.verify(session.id)
  assert.ok(status)
  assert.ok(status.throughId >= Number(result.endSeq))
  assert.equal(status.compactionCount, 1)

  // Inspect SQLite tables directly (events, compact)
  const db = new DatabaseSync(status.dbPath, { readOnly: true })
  try {
    const compactRows = db.prepare('SELECT id, through_id, provider, model, body, framed_text FROM compact').all()
    assert.equal(compactRows.length, 1)
    assert.equal(compactRows[0].provider, 'probe')
    assert.equal(compactRows[0].model, 'model')
    assert.match(compactRows[0].body, /violet-42/)
    const range = await ctx.compaction.archives.readEventRange(session.id, 0, 2)
    assert.equal(range.length, 3)
    assert.equal(range[0].id, 0)
    assert.match(range[1].text, /Original exact fact: violet-42/)
  } finally {
    db.close()
  }

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

test('multiple compactions and reopening offload.sqlite preserve canonical early records', async t => {
  const { ctx, agent, session } = await fixture(t)
  conversation(session, 1, 'EARLIEST-violet-42\n' + 'first long trace\n'.repeat(6000))
  await ctx.compaction.compactNow(agent, new AbortController().signal)
  conversation(session, 2, 'CORRECTION: violet-43 supersedes violet-42.\n' + 'second trace\n'.repeat(6000))
  await ctx.compaction.compactNow(agent, new AbortController().signal)

  const reopened = new ArchiveStore(ctx.compaction.archives.root)
  try {
    const cut = await reopened.ensure(session.id, await events(ctx, session))
    assert.equal(cut.compaction, 2)
    assert.equal(cut.compactionCount, 2)
    const rows = await reopened.readEventRange(session.id, 0, cut.throughId)
    const all = rows.map(r => r.text).join('\n')
    assert.match(all, /EARLIEST-violet-42/)
    assert.match(all, /CORRECTION: violet-43/)
    const again = await reopened.ensure(session.id, await events(ctx, session))
    assert.equal(again.eventCount, cut.eventCount)
  } finally {
    reopened.close()
  }
})

test('corrupt sqlite row prevents replacement even when cache is warm, and ArchiveStore.rebuild repairs it', async t => {
  const { ctx, agent, session } = await fixture(t)
  conversation(session, 1, 'Canonical violet-42 content')
  await ctx.sessions.flush(session)
  const store = ctx.compaction.archives
  const cut = await store.ensure(session.id, await events(ctx, session))

  // Tamper with an archived event row directly in SQLite
  const tamper = new DatabaseSync(cut.dbPath)
  try {
    tamper.prepare("UPDATE events SET raw_json = '{\"corrupted\":true}' WHERE id = 1").run()
  } finally {
    tamper.close()
  }

  const generation = session.surface.replaceGeneration
  await assert.rejects(
    ctx.compaction.compactNow(agent, new AbortController().signal),
    error => /checksum/.test(error.cause?.message ?? error.message),
  )
  assert.equal(session.surface.replaceGeneration, generation)

  // Rebuilding from canonical dsh history restores integrity
  const allEvents = await events(ctx, session)
  await store.rebuild(session.id, allEvents)
  const verified = await store.verify(session.id)
  assert.ok(verified)
  assert.equal(verified.throughId, allEvents.length - 1)
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

test('ordinary export and /context-offload do not trigger inference and report dynamic contextWindow', async t => {
  const { ctx, agent, session, calls, commands } = await fixture(t, {}, { modelWindows: { model: 128000 } })
  conversation(session, 1, 'A short user request.')
  await ctx.sessions.flush(session)
  const status = await commands.get('context-offload').handler({ agent, rawInput: '', signal: new AbortController().signal })
  assert.equal(status.kind, 'success')
  assert.match(status.text, /model \(128k, compact @ 80%\)/)
  assert.match(status.text, /model: probe\/model \(window 128000/)
  assert.match(status.text, /policy: threshold 80%, retain 16%/)
  assert.match(status.text, /db: .*offload\.sqlite/)
  const badArg = await commands.get('context-offload').handler({ agent, rawInput: 'status', signal: new AbortController().signal })
  assert.equal(badArg.kind, 'error')
  assert.match(badArg.text, /no arguments/)
  assert.equal(calls.length, 0)
  assert.equal((await ctx.compaction.archives.verify(session.id)).sessionId, session.id)
})

test('uses adapter contextWindow and BasicCompactionEngine defaults for automatic pressure compaction', async t => {
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

test('archive refuses diverging history on existing offload.sqlite', async t => {
  const { ctx, session } = await fixture(t)
  conversation(session, 1, 'Canonical original')
  const store = ctx.compaction.archives
  const snapshot = await events(ctx, session)
  await store.ensure(session.id, snapshot)
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
  const rows = await store.readEventRange(child.id, 0, fork.throughId)
  assert.match(rows.map(r => r.text).join('\n'), /Inherited exact fact: violet-42/)
})

test('history_search and history_read query offload.sqlite and support hybrid semantic fusion', async t => {
  const { ctx, agent, session, registeredTools } = await fixture(t)
  const searchEntry = registeredTools.get('history_search')
  const readEntry = registeredTools.get('history_read')
  assert.ok(searchEntry && readEntry)
  assert.equal(searchEntry.definition.isConcurrencySafe({ query: 'violet' }), true)
  assert.equal(readEntry.definition.isConcurrencySafe({ fromId: 0 }), true)

  // Tools are active on main-agent sessions from turn 1 (before compaction), and never on subagents.
  assert.equal(searchEntry.options.when({ agent }), true)
  assert.equal(readEntry.options.when({ agent }), true)
  const subSession = ctx.sessions.create(SessionId(crypto.randomUUID()), { meta: { origin: 'subagent' } })
  assert.equal(searchEntry.options.when({ agent: { session: subSession } }), false)
  assert.equal(readEntry.options.when({ agent: { session: subSession } }), false)

  conversation(session, 1, 'Early requirement: token=violet-42.\n' + 'padding line\n'.repeat(6000))
  await ctx.compaction.compactNow(agent, new AbortController().signal)
  conversation(session, 2, 'Later correction: token=violet-99 supersedes violet-42.')

  const exec = { agent, signal: new AbortController().signal }

  // Default hybrid mode gracefully falls back to lexical when dsh-context-guard is not mounted,
  // and tags compacted vs in-context hits accurately.
  const searchRes = await searchEntry.definition.execute({ query: 'violet-' }, exec)
  assert.equal(searchRes.modeUsed, 'lexical')
  assert.ok(searchRes.compactedThroughId >= 0)
  assert.ok(searchRes.totalLexicalMatches >= 2)
  assert.match(searchRes.lexicalMatches[0].snippet, /violet-99/)
  assert.equal(searchRes.lexicalMatches[0].scope, 'in-context')
  const earlyMatch = searchRes.lexicalMatches.find(m => m.snippet.includes('violet-42') && m.id <= searchRes.compactedThroughId)
  assert.ok(earlyMatch)
  assert.equal(earlyMatch.scope, 'compacted')
  assert.deepEqual(searchRes.summaryLexicalMatches, [])
  assert.deepEqual(searchRes.semanticMatches, [])

  // Explicit semantic mode errors cleanly when guard is absent.
  await assert.rejects(
    searchEntry.definition.execute({ query: 'violet', mode: 'semantic' }, exec),
    /dsh-context-guard/,
  )

  // Oldest-first + regex + eventTypes filter.
  const oldestRes = await searchEntry.definition.execute(
    { query: 'violet-\\d+', regex: true, order: 'oldest', eventTypes: ['user/message'], limit: 1 },
    exec,
  )
  assert.equal(oldestRes.lexicalMatches.length, 1)
  assert.equal(oldestRes.hasMoreLexical, true)
  assert.match(oldestRes.lexicalMatches[0].snippet, /violet-42/)
  assert.equal(oldestRes.lexicalMatches[0].scope, 'compacted')

  await assert.rejects(
    searchEntry.definition.execute({ query: '[invalid', regex: true }, exec),
    /Invalid regex/,
  )

  // history_read(detail: 'summary') without guard uses head+tail truncation and tags [compacted]/[in-context].
  const hitId = oldestRes.lexicalMatches[0].id
  const truncSummary = await readEntry.definition.execute({ fromId: hitId, toId: hitId, detail: 'summary' }, exec)
  assert.equal(truncSummary.detailUsed, 'summary')
  assert.equal(truncSummary.summarySource, 'truncated')
  assert.match(truncSummary.content, /\[compacted\]/)
  assert.match(truncSummary.content, /chars omitted/)

  // Mount a mock dsh-context-guard provider supporting semanticSearch + lexicalSearch + listSummaries.
  const capturedQueries = []
  const lastEventId = searchRes.totalEvents - 1
  ctx.provide('contextGuard', {
    async semanticSearch(_sessionId, q) {
      capturedQueries.push(q)
      return [
        // Overlaps lexical hitId -> should be deduplicated out in hybrid mode
        { cardIdx: 0, startId: hitId, endId: hitId, kind: 'constraint', score: 0.95, micro: '#0 [USER] Early violet-42' },
        // Overlaps summaryLexical hit -> should also be deduplicated out in hybrid mode
        { cardIdx: 1, startId: hitId + 1, endId: hitId + 2, kind: 'constraint', score: 0.90, micro: '#1 [USER] Summary-only violet-77' },
        // Distinct semantic hit -> should be included
        { cardIdx: 2, startId: hitId + 3, endId: hitId + 4, kind: 'assistant', score: 0.84, micro: '#2 [ASSISTANT] Completed turn 1.' },
      ]
    },
    async lexicalSearch(_sessionId, opts) {
      assert.equal(opts.regex, true)
      return [
        // Overlaps lexical hitId -> deduplicated
        { cardIdx: 0, startId: hitId, endId: hitId, kind: 'constraint', micro: '#0 [USER] Early violet-42' },
        // Distinct summary-only regex hit -> included
        { cardIdx: 1, startId: hitId + 1, endId: hitId + 2, kind: 'constraint', micro: '#1 [USER] Summary-only violet-77' },
      ]
    },
    async listSummaries(_sessionId, range) {
      // Simulate guard indexed through compactedThroughId (lagging behind latest events -> tests 'guard' & 'mixed')
      return {
        throughEventId: searchRes.compactedThroughId,
        cards: [
          { cardIdx: 0, startId: hitId, endId: hitId, kind: 'constraint', micro: '#0 [USER] Early violet-42' },
        ].filter(c => c.endId >= range.fromId && c.startId <= range.toId),
      }
    },
  })

  const hybridRes = await searchEntry.definition.execute(
    { query: 'violet-\\d+', regex: true, order: 'oldest', eventTypes: ['user/message'], limit: 1 },
    exec,
  )
  assert.equal(hybridRes.modeUsed, 'hybrid')
  assert.equal(capturedQueries[0], 'violet-')
  assert.equal(hybridRes.lexicalMatches.length, 1)
  assert.equal(hybridRes.summaryLexicalMatches.length, 1)
  assert.equal(hybridRes.summaryLexicalMatches[0].cardIdx, 1)
  assert.equal(hybridRes.semanticMatches.length, 1)
  assert.equal(hybridRes.semanticMatches[0].cardIdx, 2)

  // history_read(detail: 'summary') with guard: covered range uses 'guard', range extending past guard uses 'mixed'.
  const guardSummary = await readEntry.definition.execute({ fromId: hitId, toId: searchRes.compactedThroughId, detail: 'summary' }, exec)
  assert.equal(guardSummary.summarySource, 'guard')
  assert.match(guardSummary.content, /card #0/)
  assert.match(guardSummary.content, /compacted/)

  const mixedSummary = await readEntry.definition.execute({ fromId: hitId, toId: lastEventId, detail: 'summary' }, exec)
  assert.equal(mixedSummary.summarySource, 'mixed')
  assert.match(mixedSummary.content, /card #0/)
  assert.match(mixedSummary.content, /\[in-context\]/)

  // history_read(detail: 'full') reads the exact shadowed event directly from offload.sqlite with pagination + [compacted] tag.
  const page1 = await readEntry.definition.execute({ fromId: hitId, maxChars: 200, charOffset: 0 }, exec)
  assert.equal(page1.detailUsed, 'full')
  assert.equal(page1.summarySource, 'none')
  assert.equal(page1.fromId, hitId)
  assert.equal(page1.toId, hitId)
  assert.equal(page1.truncated, true)
  assert.equal(page1.nextCharOffset, 200)
  assert.match(page1.content, /\[compacted\]/)
  assert.match(page1.content, /Early requirement: token=violet-42/)

  const page2 = await readEntry.definition.execute({ fromId: hitId, maxChars: 200, charOffset: page1.nextCharOffset }, exec)
  assert.equal(page2.content.length, 200)
  assert.notEqual(page2.content, page1.content)

  await assert.rejects(
    readEntry.definition.execute({ fromId: 5, toId: 2 }, exec),
    /toId/,
  )
})
