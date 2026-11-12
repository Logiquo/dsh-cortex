import assert from 'node:assert/strict'
import { test } from 'node:test'
import { mkdtemp, readFile, readdir, rm, writeFile } from 'node:fs/promises'
import { tmpdir, homedir } from 'node:os'
import { join } from 'node:path'
import { Context } from '@deepseek-ai/cordis'
import { SessionStore, SessionId } from '@deepseek-ai/dsh-session'
import { SessionProjectionRegistry } from '@deepseek-ai/dsh-session-projection'
import { SessionQueryEngine } from '@deepseek-ai/dsh-session-query'
import { TokenMeter } from '@deepseek-ai/dsh-token-meter'
import { LlmRuntime, LlmAdapter, createUserMessage, createAssistantMessage, createSystemMessage } from '@deepseek-ai/dsh-llm'
import * as plugin from '../lib/index.js'
import { Config, archiveRoot } from '../lib/config.js'
import { ArchiveStore } from '../lib/archive.js'
import { budgets, selectRange } from '../lib/policy.js'
const brief = 'Implement the feature while preserving violet-42. The next action is to inspect the implementation and run the relevant checks. Completion requires those checks to pass.'
const user = text => createUserMessage({ content: [{ type: 'text', text }], source: { kind: 'user' } })
class Query extends SessionQueryEngine {
  async searchSessions() { throw Error('No search index needed') }
  async searchEvents() { throw Error('No search index needed') }
}
async function fixture(t, config = {}, behavior = {}) {
  const directory = await mkdtemp(join(tmpdir(), 'ctx-offload-test-'))
  const ctx = new Context()
  const fibers = []
  const mount = async (p, c = {}) => { const f = ctx.plugin(p, c); fibers.push(f); await f.await(); return f }
  await mount(SessionProjectionRegistry)
  await mount(SessionStore)
  await mount(LlmRuntime)
  await mount(TokenMeter)
  await mount(Query)
  const calls = []
  class Fake extends LlmAdapter {
    async resolveModel(provider, model) { return { provider, id: model, name: model, context: { contextWindow: 100000 }, defaultMaxTokens: 2048 } }
    async *stream(options) {
      calls.push(options)
      if (behavior.onCall) await behavior.onCall(options)
      yield { type: 'text-delta', index: 0, text: behavior.text ?? brief }
      yield { type: 'finish', reason: { kind: behavior.finish ?? 'stop' } }
    }
  }
  ctx.llm.registerAdapter(['probe'], new Fake())
  const agents = []
  ctx.provide('agents', { list: () => agents })
  const commands = new Map()
  ctx.provide('commands', { register: definition => { commands.set(definition.name, definition); return () => commands.delete(definition.name) } })
  const fiber = await mount(plugin, { 'context-limit': 32000, 'spill-root': join(directory, 'archive'), ...config })
  const session = ctx.sessions.create(SessionId(crypto.randomUUID()))
  const agent = { session, options: { provider: 'probe', model: 'model' }, runMaintenance: operation => operation(new AbortController().signal) }
  agents.push(agent)
  t.after(async () => { for (const f of fibers.reverse()) await f.dispose(); await rm(directory, { recursive: true, force: true }) })
  return { directory, ctx, session, agent, calls, commands, fiber }
}
function conversation(session, turn, text = 'Details and context. '.repeat(4000)) {
  session.append('turn/start', { turn })
  session.append('user/message', user(text), { surfaceOp: 'append' })
  session.append('step/start', { turn, step: 1 })
  session.append('request/header', { header: { config: { provider: 'probe', model: 'model', maxTokens: 2048 } }, reason: 'initial' })
  session.append('assistant/message', { turn, step: 1, stream: [], message: createAssistantMessage({ source: { provider: 'probe', model: 'model' }, content: [{ type: 'text', text: 'Completed current work.' }] }) }, { surfaceOp: 'append' })
  session.append('step/end', { turn, step: 1 })
  session.append('turn/end', { turn, reason: { kind: 'completed' } })
}
async function events(ctx, session) { const cut = await ctx.sessionQuery.observeSession(session.id, { projectionMode: 'none' }); try { return [...cut.events] } finally { cut[Symbol.dispose]() } }

test('portable defaults and configured context-budget formulas', () => {
  const config = Config({})
  assert.equal(archiveRoot(config), join(homedir(), '.dsh', 'ctx-offload'))
  assert.deepEqual(budgets(Config({ 'context-limit': 32000 })), { triggerTokens: 24000, briefMaxTokens: 2400, retainRecentTokens: 4800 })
  assert.deepEqual(budgets(Config({ 'context-limit': 16000 })), { triggerTokens: 12000, briefMaxTokens: 2048, retainRecentTokens: 4096 })
  assert.throws(() => archiveRoot(Config({ 'spill-root': 'relative' })), /absolute/)
  assert.equal(config['context-limit'], 100000)
  assert.deepEqual(budgets(config), { triggerTokens: 75000, briefMaxTokens: 7500, retainRecentTokens: 15000 })
})

test('a prose brief without prescribed headings commits with searchable original history', async t => {
  const { ctx, agent, session, calls } = await fixture(t)
  conversation(session, 1, 'Original exact fact: violet-42.\n' + 'History data\n'.repeat(7000))
  const result = await ctx.compaction.compactNow(agent, new AbortController().signal)
  assert.ok(result)
  const summary = result.summary[0].text
  assert.match(summary, /Completion requires those checks to pass/)
  assert.match(summary, /spawn a fresh search agent/i)
  assert.match(summary, /rg\/grep/)
  const catalog = await ctx.compaction.archives.verify(session.id)
  assert.ok(catalog.through >= Number(result.endSeq))
  const directory = summary.match(/History archive: (.+)/)[1]
  const archiveDir = JSON.parse(directory)
  const history = (await Promise.all(catalog.chunks.map(chunk => readFile(join(archiveDir, chunk.file), 'utf8')))).join('\n')
  assert.match(history, /Original exact fact: violet-42/)
  assert.equal(calls.length, 1)
  assert.equal(calls[0].purpose, 'compaction')
  assert.equal(calls[0].maxTokens, 2400)
  const logged = await events(ctx, session)
  const summaryIndex = logged.findIndex(e => e.type === 'compaction/summary')
  assert.equal(logged[summaryIndex + 1].type, 'user/message')
  assert.ok(session.surface.replaceGeneration > 0)
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

test('corrupt history prevents replacement and leaves the surface intact', async t => {
  const { ctx, agent, session } = await fixture(t)
  conversation(session, 1)
  await ctx.sessions.flush(session)
  const store = ctx.compaction.archives
  const cut = await store.ensure(session.id, await events(ctx, session))
  await writeFile(join(cut.directory, cut.catalog.chunks[0].file), 'corrupted')
  const generation = session.surface.replaceGeneration
  await assert.rejects(ctx.compaction.compactNow(agent, new AbortController().signal), error => /checksum/.test(error.cause?.message ?? error.message))
  assert.equal(session.surface.replaceGeneration, generation)
})

test('empty output fails without losing context', async t => {
  const { ctx, agent, session, calls } = await fixture(t, {}, { text: '  \n  ' })
  conversation(session, 1)
  await assert.rejects(ctx.compaction.compactNow(agent, new AbortController().signal), error => /brief is empty/.test(error.cause?.message ?? error.message))
  assert.equal(calls.length, 2)
  assert.equal(session.surface.replaceGeneration, 0)
})

test('cancellation during brief generation preserves the original surface', async t => {
  const abort = new AbortController()
  const { ctx, agent, session } = await fixture(t, {}, { onCall: () => abort.abort(new Error('cancelled by test')) })
  conversation(session, 1)
  await assert.rejects(ctx.compaction.compactNow(agent, abort.signal), /cancelled by test/)
  assert.equal(session.surface.replaceGeneration, 0)
})

test('ordinary export and status do not trigger inference or retrieval', async t => {
  const { ctx, agent, session, calls, commands } = await fixture(t)
  conversation(session, 1, 'A short user request.')
  await ctx.sessions.flush(session)
  const status = await commands.get('ctx-offload').handler({ agent, rawInput: 'status', signal: new AbortController().signal })
  assert.equal(status.kind, 'success')
  assert.equal(calls.length, 0)
  assert.equal((await ctx.compaction.archives.verify(session.id)).sessionId, session.id)
})

test('configured limit overrides advertised capacity and preserves the empty system head', async t => {
  const { ctx, agent, session, calls } = await fixture(t)
  session.append('system/message', { turn: 1, step: 1, message: createSystemMessage('') }, { surfaceOp: 'append' })
  conversation(session, 1, 'Historical material\n'.repeat(7000))
  conversation(session, 2, 'Recent material\n'.repeat(1500))
  session.append('turn/start', { turn: 3 })
  const original = [...session.surface.nodes]
  const before = ctx.tokenMeter.measure(session)
  assert.ok(before.totalTokens > 24000 && before.totalTokens < 75000)
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

test('archive refuses diverging history and recovers a dead writer lock', async t => {
  const { ctx, session } = await fixture(t)
  conversation(session, 1, 'Canonical original')
  const store = ctx.compaction.archives
  const snapshot = await events(ctx, session)
  const cut = await store.ensure(session.id, snapshot)
  // dsh's lock helper records a numeric PID. This PID cannot be live on supported hosts.
  await writeFile(join(cut.directory, 'catalog.json.lock'), '2147483647\n')
  await store.ensure(session.id, snapshot)
  const changed = structuredClone(snapshot)
  changed[1].data = user('A different historical message')
  await assert.rejects(store.ensure(session.id, changed), /differs from archived prefix/)
})

test('provider truncation and tiny reductions preserve the unmodified surface', async t => {
  const { ctx, session, agent } = await fixture(t, {}, { finish: 'max-tokens' })
  conversation(session, 1)
  await assert.rejects(ctx.compaction.compactNow(agent, new AbortController().signal), error => /max-tokens/.test(error.cause?.message ?? error.message))
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

test('recent history is an upper bound and can be empty when the last whole message is too large', async t => {
  const { ctx, session } = await fixture(t)
  conversation(session, 1, 'Prior history\n'.repeat(4000))
  session.append('user/message', user('A single large most recent message.\n'.repeat(2000)), { surfaceOp: 'append' })
  const measurement = ctx.tokenMeter.measure(session)
  const range = selectRange(session, measurement, 4096, false)
  assert.ok(range)
  assert.equal(range.end, session.surface.nodes.at(-1))
  const end = measurement.nodes.findIndex(node => node.seq === range.end)
  assert.equal(measurement.nodes.slice(end + 1).reduce((n, node) => n + node.tokens, 0), 0)
})

test('recent history never exceeds its limit when a larger adjacent message would overflow it', async t => {
  const { ctx, session } = await fixture(t)
  conversation(session, 1, 'Large historical message.\n'.repeat(4000))
  conversation(session, 2, 'Smaller recent request.\n'.repeat(200))
  const measurement = ctx.tokenMeter.measure(session)
  const range = selectRange(session, measurement, 4096, false)
  assert.ok(range)
  const end = measurement.nodes.findIndex(node => node.seq === range.end)
  const retained = measurement.nodes.slice(end + 1).reduce((n, node) => n + node.tokens, 0)
  assert.ok(retained > 0 && retained <= 4096)
})
