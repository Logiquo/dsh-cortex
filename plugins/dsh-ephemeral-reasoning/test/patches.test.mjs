import assert from 'node:assert/strict'
import { test } from 'node:test'
import { Context } from '@deepseek-ai/cordis'
import LlmRuntime, { createAssistantMessage, createUserMessage } from '@deepseek-ai/dsh-llm'
import { Session, SessionId } from '@deepseek-ai/dsh-session'
import { TokenMeter } from '@deepseek-ai/dsh-token-meter'
import { estimateContent, estimateMessage } from '@deepseek-ai/dsh-token-meter/estimate'
import { SessionProjectionRegistry } from '@deepseek-ai/dsh-session-projection'
import { interceptLlm, stripClosedTurnOptions, stripClosedTurnReasoning } from '../lib/outbound.js'
import { interceptMeter } from '../lib/meter.js'
import * as plugin from '../lib/index.js'

const make = () => Session.create(SessionId(crypto.randomUUID()))
const HEADER = { config: { provider: 'deepseek-account', model: 'test' } }
const ANSWER = [{ type: 'text', text: 'answer' }]
const REASONING = [{ type: 'reasoning', text: 'private-reasoning-trace-for-step' }]
const WITH_REASONING = [...REASONING, ...ANSWER]

function envelope(s, header = HEADER) {
  return s.append('request/header', { header, reason: 'initial' })
}

function settle(s, usage, content = WITH_REASONING) {
  return s.append('assistant/message', {
    turn: 1,
    step: 1,
    stream: [],
    ...(usage === undefined ? {} : { usage }),
    message: createAssistantMessage({
      content,
      source: { provider: 'deepseek-account', model: 'test', replayState: { private: true } },
    }),
  }, { surfaceOp: 'append' })
}

function prompt(s, text = 'next') {
  return s.append('user/message', createUserMessage({ source: { kind: 'user' }, content: [{ type: 'text', text }] }), { surfaceOp: 'append' })
}

function raw(nodes) {
  const surfaceTokens = nodes.reduce((total, node) => total + node.tokens, 0)
  return { logRevision: 0, baseline: { kind: 'estimated', tokens: surfaceTokens }, surfaceDeltaTokens: 0, totalTokens: surfaceTokens, surfaceTokens, nodes }
}

function fakeMeter(s) {
  return {
    measure: () => ({
      logRevision: s.seq,
      baseline: { kind: 'usage', tokens: 60000, usage: { inputTokens: 50000, outputTokens: 10000, reasoningTokens: 9000 } },
      surfaceDeltaTokens: 0,
      totalTokens: 60000,
      surfaceTokens: 133,
      nodes: [{ seq: 1, tokens: 133, heuristicTokens: 10 }],
    }),
    contextPressure: () => ({ pressureTokens: 50000, contextWindow: 100000 }),
  }
}

const price = content => estimateMessage(createAssistantMessage({ content, source: { provider: 'deepseek-account', model: 'test' } }))
const reasoningPrice = estimateContent(REASONING)
const visiblePrice = price(ANSWER)

async function setup() {
  const ctx = new Context()
  const projections = ctx.plugin(SessionProjectionRegistry)
  await projections.await()
  const meterFiber = ctx.plugin(TokenMeter)
  await meterFiber.await()
  const llm = new LlmRuntime(ctx)
  const fiber = ctx.plugin(plugin, {})
  await fiber.await()
  return { ctx, fiber, meter: ctx.tokenMeter, llm }
}

/* -------------------------------------------------------------------------- *
 * Outbound send-side stripping
 * -------------------------------------------------------------------------- */

test('stripClosedTurnReasoning strips closed turns and preserves open-turn tool-call chains', () => {
  const u1 = createUserMessage({ source: { kind: 'user' }, content: [{ type: 'text', text: 'turn 1' }] })
  const a1 = createAssistantMessage({
    content: WITH_REASONING,
    source: { provider: 'deepseek-account', model: 'deepseek-flash', replayState: { sig: 'turn1' } },
  })
  const u2 = createUserMessage({ source: { kind: 'user' }, content: [{ type: 'text', text: 'turn 2' }] })
  const a2 = createAssistantMessage({
    content: [...REASONING, { type: 'tool-call', id: 'c1', name: 'read', arguments: '{}' }],
    source: { provider: 'deepseek-account', model: 'deepseek-flash', replayState: { sig: 'turn2-step1' } },
  })
  const toolResult = { role: 'tool', source: { kind: 'tool', callId: 'c1' }, toolCallId: 'c1', content: [{ type: 'text', text: 'ok' }] }

  const out = stripClosedTurnReasoning([u1, a1, u2, a2, toolResult])
  assert.deepEqual(out[1].content, ANSWER, 'closed turn reasoning must be stripped')
  assert.deepEqual(out[1].source, { kind: 'model', provider: 'deepseek-account', model: 'deepseek-flash' }, 'replayState removed when reasoning stripped')
  assert.equal(out[3], a2, 'open turn assistant message and its replayState must stay untouched')
})

test('stripClosedTurnReasoning keeps replayState when a closed turn had no reasoning', () => {
  const u1 = createUserMessage({ source: { kind: 'user' }, content: [{ type: 'text', text: 'turn 1' }] })
  const a1 = createAssistantMessage({
    content: ANSWER,
    source: { provider: 'deepseek-account', model: 'deepseek-flash', replayState: { sig: 'keep' } },
  })
  const u2 = createUserMessage({ source: { kind: 'user' }, content: [{ type: 'text', text: 'turn 2' }] })
  const messages = [u1, a1, u2]
  assert.equal(stripClosedTurnReasoning(messages), messages)
})

test('stripClosedTurnOptions handles frozen options without mutating input', () => {
  const u1 = createUserMessage({ source: { kind: 'user' }, content: [{ type: 'text', text: 'turn 1' }] })
  const a1 = createAssistantMessage({
    content: WITH_REASONING,
    source: { provider: 'deepseek-account', model: 'deepseek-flash', replayState: { sig: 'old' } },
  })
  const u2 = createUserMessage({ source: { kind: 'user' }, content: [{ type: 'text', text: 'turn 2' }] })
  const options = Object.freeze({ provider: 'deepseek-account', model: 'deepseek-flash', messages: Object.freeze([u1, a1, u2]) })
  const rewritten = stripClosedTurnOptions(options)
  assert.notEqual(rewritten, options)
  assert.equal(Object.isFrozen(rewritten), true)
  assert.equal(a1.content.length, 2, 'original message must stay intact')
  assert.equal(rewritten.messages[1].content.length, 1)
})

test('interceptLlm patches forAdapter, deactivates beneath later wrappers, and restores on dispose', async () => {
  const { ctx, fiber, llm } = await setup()
  const u1 = createUserMessage({ source: { kind: 'user' }, content: [{ type: 'text', text: 't1' }] })
  const a1 = createAssistantMessage({ content: WITH_REASONING, source: { provider: 'deepseek-account', model: 'm', replayState: { s: 1 } } })
  const u2 = createUserMessage({ source: { kind: 'user' }, content: [{ type: 'text', text: 't2' }] })
  const forAdapter = llm.forAdapter.bind(llm)
  const res = forAdapter({ provider: 'deepseek-account', model: 'm', messages: [u1, a1, u2] }, undefined)
  assert.equal(res.messages[1].content.length, 1)

  await fiber.dispose()
  const afterDispose = llm.forAdapter({ provider: 'deepseek-account', model: 'm', messages: [u1, a1, u2] }, undefined)
  assert.equal(afterDispose.messages[1].content.length, 2)

  // Deactivate beneath a later wrapper
  const undo = interceptLlm(llm)
  const inner = llm.forAdapter
  const outer = function (options, adapter) { return inner.call(this, options, adapter) }
  llm.forAdapter = outer
  assert.equal(llm.forAdapter({ provider: 'deepseek-account', model: 'm', messages: [u1, a1, u2] }, undefined).messages[1].content.length, 1)
  undo()
  assert.equal(llm.forAdapter, outer)
  assert.equal(llm.forAdapter({ provider: 'deepseek-account', model: 'm', messages: [u1, a1, u2] }, undefined).messages[1].content.length, 2)

  // Disabled plugin leaves forAdapter untouched
  const disabled = ctx.plugin(plugin, { enabled: false })
  await disabled.await()
  assert.equal(llm.forAdapter({ provider: 'deepseek-account', model: 'm', messages: [u1, a1, u2] }, undefined).messages[1].content.length, 2)
  await disabled.dispose()
})

test('persisted session logs retain full reasoning and replayState', async () => {
  const { fiber } = await setup()
  const s = make()
  envelope(s)
  prompt(s, 'first')
  const a = settle(s, { inputTokens: 1000, outputTokens: 100 })
  assert.equal(a.data.message.content.length, 2)
  assert.deepEqual(a.data.message.source.replayState, { private: true })
  assert.equal(s.deriveMessages()[1].content.length, 2)
  await fiber.dispose()
})

/* -------------------------------------------------------------------------- *
 * TokenMeter & contextPressure counting
 * -------------------------------------------------------------------------- */

test('mid-turn: the provider total plus only the nodes appended after it', () => {
  const s = make(); envelope(s)
  const a = settle(s, { inputTokens: 1000, outputTokens: 100, cacheReadTokens: 5000 }, WITH_REASONING)
  const full = price(WITH_REASONING)
  const nodes = [{ seq: a.seq, tokens: full, heuristicTokens: full }, { seq: a.seq + 1, tokens: 40, heuristicTokens: 40 }]
  const meter = { measure: () => raw(nodes), contextPressure: () => ({ pressureTokens: 50000, contextWindow: 100000 }) }
  const undo = interceptMeter(meter)
  const m = meter.measure(s)
  assert.equal(m.baseline.kind, 'usage')
  assert.equal(m.baseline.tokens, 6100)
  assert.equal(m.surfaceDeltaTokens, 40)
  assert.equal(m.totalTokens, 6140)
  assert.equal(m.nodes, nodes, 'open-turn node prices are passed through untouched')
  assert.equal(meter.measure(make()).baseline.kind, 'estimated')
  undo()
})

test('a user message closes the turn and removes reasoning from both totalTokens and closed-turn nodes', () => {
  const s = make(); envelope(s)
  const a = settle(s, { inputTokens: 1000, outputTokens: 100 }, WITH_REASONING)
  const u = prompt(s)
  const expectedDropped = Math.round(100 * reasoningPrice / (reasoningPrice + visiblePrice))
  const full = price(WITH_REASONING)
  const nodes = [{ seq: a.seq, tokens: full, heuristicTokens: full }, { seq: u.seq, tokens: 40, heuristicTokens: 40 }]
  const meter = { measure: () => raw(nodes) }
  const undo = interceptMeter(meter)
  const m = meter.measure(s)
  assert.equal(m.baseline.kind, 'usage')
  assert.equal(m.baseline.tokens, 1100)
  assert.equal(m.surfaceDeltaTokens, 40 - expectedDropped)
  assert.equal(m.totalTokens, 1100 + 40 - expectedDropped)
  assert.equal(m.nodes[0].tokens, full - reasoningPrice, 'closed-turn node excludes stripped reasoning')
  assert.equal(m.surfaceTokens, full - reasoningPrice + 40)
  undo()
})

test('every settled step of the closed turn contributes its reasoning exactly once', () => {
  const s = make(); envelope(s)
  const first = settle(s, { inputTokens: 1000, outputTokens: 100 }, WITH_REASONING)
  const second = settle(s, { inputTokens: 1200, outputTokens: 100 }, WITH_REASONING)
  const u = prompt(s)
  const per = Math.round(100 * reasoningPrice / (reasoningPrice + visiblePrice))
  const full = price(WITH_REASONING)
  const nodes = [{ seq: first.seq, tokens: full, heuristicTokens: full }, { seq: second.seq, tokens: full, heuristicTokens: full }, { seq: u.seq, tokens: 7, heuristicTokens: 7 }]
  const meter = { measure: () => raw(nodes) }
  const undo = interceptMeter(meter)
  const m = meter.measure(s)
  assert.equal(m.totalTokens, 1200 + 100 + 7 - 2 * per)
  undo()
})

test('a reasoning-only settlement contributes its whole output as dropped reasoning', () => {
  const s = make(); envelope(s)
  const a = settle(s, { inputTokens: 1000, outputTokens: 50 }, REASONING)
  const u = prompt(s)
  const full = price(REASONING)
  const nodes = [{ seq: a.seq, tokens: full, heuristicTokens: full }, { seq: u.seq, tokens: 10, heuristicTokens: 10 }]
  const meter = { measure: () => raw(nodes) }
  const undo = interceptMeter(meter)
  assert.equal(meter.measure(s).totalTokens, 1000 + 10)
  undo()
})

test('an envelope change or a rewritten surface drops the anchor and excludes closed-turn reasoning', () => {
  const s = make(); envelope(s)
  const a = settle(s, { inputTokens: 5000, outputTokens: 100 }, WITH_REASONING)
  const u = prompt(s)
  envelope(s, { config: { provider: 'anthropic', model: 'other' } })
  const full = price(WITH_REASONING)
  const nodes = [{ seq: a.seq, tokens: full, heuristicTokens: full }, { seq: u.seq, tokens: 15, heuristicTokens: 15 }]
  const meter = { measure: () => raw(nodes) }
  const undo = interceptMeter(meter)
  const m = meter.measure(s)
  assert.equal(m.baseline.kind, 'estimated')
  assert.equal(m.totalTokens, full - reasoningPrice + 15)
  undo()
})

test('an effective header that was never logged still retires the anchor', () => {
  const s = make(); envelope(s)
  const a = settle(s, { inputTokens: 5000, outputTokens: 100 }, ANSWER)
  const nodes = [{ seq: a.seq, tokens: visiblePrice, heuristicTokens: visiblePrice }]
  const meter = { measure: () => raw(nodes) }
  const undo = interceptMeter(meter)
  const m = meter.measure(s, { config: { provider: 'anthropic', model: 'other' } })
  assert.equal(m.baseline.kind, 'estimated')
  assert.equal(m.totalTokens, visiblePrice)
  undo()
})

test('an identical envelope re-logged for a resumed series keeps the anchor', () => {
  const s = make(); envelope(s)
  const a = settle(s, { inputTokens: 5000, outputTokens: 100 }, ANSWER)
  envelope(s)
  const nodes = [{ seq: a.seq, tokens: visiblePrice, heuristicTokens: visiblePrice }]
  const meter = { measure: () => raw(nodes) }
  const undo = interceptMeter(meter)
  const m = meter.measure(s)
  assert.equal(m.baseline.kind, 'usage')
  assert.equal(m.totalTokens, 5100)
  undo()
})

test('a settlement without usage keeps the previous anchor; its node stays in the unpriced tail', () => {
  const s = make(); envelope(s)
  const a = settle(s, { inputTokens: 1000, outputTokens: 50 }, ANSWER)
  const aborted = settle(s, undefined, ANSWER)
  const nodes = [{ seq: a.seq, tokens: visiblePrice, heuristicTokens: visiblePrice }, { seq: aborted.seq, tokens: 20, heuristicTokens: 20 }]
  const meter = { measure: () => raw(nodes) }
  const undo = interceptMeter(meter)
  const m = meter.measure(s)
  assert.equal(m.baseline.kind, 'usage')
  assert.equal(m.totalTokens, 1070)
  undo()
})

test('meter wrapper deactivates beneath later extensions without overwriting them', () => {
  const s = make()
  const meter = fakeMeter(s)
  const unm = interceptMeter(meter)
  const inside = meter.measure
  const outer = function (...args) { return inside.apply(this, args) }
  meter.measure = outer
  unm()
  assert.equal(meter.measure, outer)
  assert.equal(meter.measure(s).baseline.kind, 'usage')
})

test('the occupancy ring reports the same corrected total as the meter', async () => {
  const s = make(); const { ctx, fiber, meter } = await setup()
  envelope(s)
  prompt(s, 'first')
  s.append('step/start', { turn: 1, step: 1 })
  settle(s, { inputTokens: 1000, outputTokens: 100 }, WITH_REASONING)
  s.append('step/end', { turn: 1, step: 1 })
  const ring = () => ctx.sessionProjections.snapshot(s, ['contextPressure']).values.contextPressure
  const midMeter = meter.measure(s).totalTokens
  assert.equal(ring().projectedTokens, midMeter, 'ring and meter must agree while the turn is open')
  prompt(s, 'second')
  const endMeter = meter.measure(s).totalTokens
  assert.equal(ring().projectedTokens, endMeter, 'ring and meter must agree after the turn closed')
  assert.ok(endMeter < midMeter, 'closing the turn must drop its reasoning')
  await fiber.dispose()
})

test('subagent occupancy is corrected exactly like a main session', async () => {
  const s = make(); const { ctx, fiber, meter } = await setup()
  const id = SessionId(crypto.randomUUID())
  const child = Session.create(id, [], { ...s.header, id, origin: 'subagent' })
  envelope(child)
  prompt(child, 'first')
  child.append('step/start', { turn: 1, step: 1 })
  settle(child, { inputTokens: 900, outputTokens: 90 }, WITH_REASONING)
  child.append('step/end', { turn: 1, step: 1 })
  const ring = () => ctx.sessionProjections.snapshot(child, ['contextPressure']).values.contextPressure
  assert.equal(ring().projectedTokens, meter.measure(child).totalTokens)
  prompt(child, 'second')
  assert.equal(ring().projectedTokens, meter.measure(child).totalTokens)
  await fiber.dispose()
})

test('a second user message in the same loop keeps the closed turn reasoning dropped', async () => {
  const s = make(); const { ctx, fiber, meter } = await setup()
  envelope(s); prompt(s, 'first')
  s.append('step/start', { turn: 1, step: 1 })
  settle(s, { inputTokens: 1000, outputTokens: 100 }, WITH_REASONING)
  s.append('step/end', { turn: 1, step: 1 })
  const ring = () => ctx.sessionProjections.snapshot(s, ['contextPressure']).values.contextPressure.projectedTokens
  prompt(s, 'second')
  assert.equal(ring(), meter.measure(s).totalTokens)
  const closed = ring()
  for (const text of ['third', 'fourth', 'fifth']) {
    prompt(s, text)
    assert.equal(ring(), meter.measure(s).totalTokens)
  }
  const delta = ['third', 'fourth', 'fifth'].reduce((total, text) => total + estimateMessage(createUserMessage({ source: { kind: 'user' }, content: [{ type: 'text', text }] })), 0)
  assert.equal(ring() - closed, delta)
  await fiber.dispose()
})

test('a replace mid-turn keeps the open turn reasoning and the two surfaces agree at close', async () => {
  const s = make(); const { ctx, fiber, meter } = await setup()
  envelope(s); prompt(s, 'first')
  s.append('step/start', { turn: 1, step: 1 })
  settle(s, { inputTokens: 1000, outputTokens: 100 }, WITH_REASONING)
  s.append('step/end', { turn: 1, step: 1 })
  const tool = s.append('tool/result', { message: { role: 'tool', source: { kind: 'tool', callId: 'c1' }, toolCallId: 'c1', content: [{ type: 'text', text: 'x'.repeat(400) }] } }, { surfaceOp: 'append' })
  s.append('compaction/prune', { shadowedRange: { start: tool.seq, end: tool.seq }, shadowedSeqs: [tool.seq], shadowedTokenCount: 100 })
  s.append('tool/result', { message: { role: 'tool', source: { kind: 'tool', callId: 'c1' }, toolCallId: 'c1', content: [{ type: 'text', text: 'pruned' }] } }, { surfaceOp: { op: 'replace', startSeq: tool.seq, endSeq: tool.seq }, sourceEventSeqs: [tool.seq] })
  s.append('step/start', { turn: 1, step: 2 })
  s.append('assistant/message', { turn: 1, step: 2, stream: [], usage: { inputTokens: 1100, outputTokens: 120 }, message: createAssistantMessage({ content: WITH_REASONING, source: { provider: 'deepseek-account', model: 'test', replayState: { private: true } } }) }, { surfaceOp: 'append' })
  s.append('step/end', { turn: 1, step: 2 })
  const ring = () => ctx.sessionProjections.snapshot(s, ['contextPressure']).values.contextPressure.projectedTokens
  prompt(s, 'close')
  assert.equal(ring(), meter.measure(s).totalTokens)
  assert.ok(ring() < 1220)
  await fiber.dispose()
})

test('dense scripts are not misread as reasoning, and no reasoning is inferred when none exists', async () => {
  const s = make(); const { ctx, fiber, meter } = await setup()
  const cjk = '这是一个需要仔细分析的问题，因为其中涉及多个相互依赖的约束条件。'.repeat(20)
  envelope(s); prompt(s, 'ask')
  s.append('step/start', { turn: 1, step: 1 })
  s.append('assistant/message', { turn: 1, step: 1, stream: [], usage: { inputTokens: 500, outputTokens: cjk.length }, message: createAssistantMessage({ content: [{ type: 'text', text: cjk }], source: { provider: 'deepseek-account', model: 'test' } }) }, { surfaceOp: 'append' })
  s.append('step/end', { turn: 1, step: 1 })
  const ring = () => ctx.sessionProjections.snapshot(s, ['contextPressure']).values.contextPressure.projectedTokens
  prompt(s, 'next')
  assert.equal(ring(), meter.measure(s).totalTokens)
  assert.equal(meter.measure(s).totalTokens, 500 + cjk.length + estimateMessage(createUserMessage({ source: { kind: 'user' }, content: [{ type: 'text', text: 'next' }] })))
  await fiber.dispose()
})

test('a finished turn stops resending its reasoning on turn/end, not only on the next prompt', async () => {
  const s = make(); const { ctx, fiber, meter } = await setup()
  envelope(s); prompt(s, 'first')
  s.append('step/start', { turn: 1, step: 1 })
  settle(s, { inputTokens: 1000, outputTokens: 100 }, WITH_REASONING)
  s.append('step/end', { turn: 1, step: 1 })
  const ring = () => ctx.sessionProjections.snapshot(s, ['contextPressure']).values.contextPressure.projectedTokens
  const open = meter.measure(s).totalTokens
  assert.equal(ring(), open)
  s.append('turn/end', { turn: 1, reason: { kind: 'completed' } })
  const closed = meter.measure(s).totalTokens
  assert.ok(closed < open)
  assert.equal(ring(), closed)
  prompt(s, 'next')
  assert.equal(ring(), meter.measure(s).totalTokens)
  await fiber.dispose()
})

test('a projection definition installed later is not clobbered on unload', async () => {
  const { ctx, fiber } = await setup()
  const registration = ctx.sessionProjections.registrations.get('contextPressure')
  const later = { ...registration.def }
  registration.def = later
  await fiber.dispose()
  assert.equal(registration.def, later)
})

test('the occupancy unit carries a version that discards rows from the old fold', async () => {
  const { ctx, fiber } = await setup()
  const def = ctx.sessionProjections.registrations.get('contextPressure').def
  assert.equal(def.stateVersion, 9)
  assert.throws(() => def.stateSchema.parse({ surfaceTokens: 0, appendedTokens: 0, turnReasoning: 0, droppedReasoning: 0 }))
  await fiber.dispose()
})

test('a refused occupancy override leaves no llm or meter patch and no owner marker behind', async () => {
  const ctx = new Context()
  const projections = ctx.plugin(SessionProjectionRegistry)
  await projections.await()
  const meterFiber = ctx.plugin(TokenMeter)
  await meterFiber.await()
  const llm = new LlmRuntime(ctx)
  const reg = ctx.sessionProjections.registrations.get('contextPressure')
  reg.def = { ...reg.def, stateVersion: 999 }
  const beforeMeasure = Object.getOwnPropertyDescriptor(ctx.tokenMeter, 'measure')
  const beforeForAdapter = Object.getOwnPropertyDescriptor(llm, 'forAdapter')
  const fiber = ctx.plugin(plugin, {})
  await assert.rejects(fiber.await(), /stateVersion 999/)
  assert.deepEqual(Object.getOwnPropertyDescriptor(ctx.tokenMeter, 'measure'), beforeMeasure)
  assert.deepEqual(Object.getOwnPropertyDescriptor(llm, 'forAdapter'), beforeForAdapter)
  assert.equal(Reflect.get(ctx.tokenMeter, Symbol.for('dsh-ephemeral-reasoning.owner')), undefined)
  await fiber.dispose()
  await meterFiber.dispose()
  await projections.dispose()
})
