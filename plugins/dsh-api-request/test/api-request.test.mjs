import assert from 'node:assert/strict'
import { test } from 'node:test'
import { Context } from '@deepseek-ai/cordis'
import LlmRuntime, { createAssistantMessage, createUserMessage } from '@deepseek-ai/dsh-llm'
import { Session, SessionId } from '@deepseek-ai/dsh-session'
import { TokenMeter } from '@deepseek-ai/dsh-token-meter'
import { estimateContent, estimateMessage } from '@deepseek-ai/dsh-token-meter/estimate'
import { SessionProjectionRegistry } from '@deepseek-ai/dsh-session-projection'
import * as apiRequestPlugin from '../lib/index.js'

const makeSession = () => Session.create(SessionId(crypto.randomUUID()))
const HEADER = { config: { provider: 'deepseek-account', model: 'test' } }
const ANSWER = [{ type: 'text', text: 'answer' }]
const REASONING = [{ type: 'reasoning', text: 'private-reasoning-trace-for-step' }]
const WITH_REASONING = [...REASONING, ...ANSWER]

function envelope(s, header = HEADER) {
  return s.append('request/header', { header, reason: 'initial' })
}

function settle(s, usage, content = WITH_REASONING) {
  s.append('step/start', { turn: 1, step: 1 })
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

async function setup() {
  const ctx = new Context()
  const projections = ctx.plugin(SessionProjectionRegistry)
  await projections.await()
  const meterFiber = ctx.plugin(TokenMeter)
  await meterFiber.await()
  const llm = new LlmRuntime(ctx)
  const apiFiber = ctx.plugin(apiRequestPlugin)
  await apiFiber.await()
  return {
    ctx,
    apiFiber,
    meter: ctx.tokenMeter,
    llm,
    api: ctx.apiRequest,
    dispose: async () => {
      await apiFiber.dispose()
      await meterFiber.dispose()
      await projections.dispose()
    },
  }
}

test('startup fails closed when llm.forAdapter is already wrapped', async () => {
  const ctx = new Context()
  const projections = ctx.plugin(SessionProjectionRegistry)
  await projections.await()
  const meter = ctx.plugin(TokenMeter)
  await meter.await()
  const llm = new LlmRuntime(ctx)
  Object.defineProperty(llm, 'forAdapter', { configurable: true, writable: true, value() { throw Error('foreign') } })
  const beforeMeasure = Object.getOwnPropertyDescriptor(ctx.tokenMeter, 'measure')

  const apiFiber = ctx.plugin(apiRequestPlugin)
  await assert.rejects(apiFiber.await(), /LlmRuntime\.forAdapter/)
  assert.deepEqual(Object.getOwnPropertyDescriptor(ctx.tokenMeter, 'measure'), beforeMeasure)

  await apiFiber.dispose()
  await meter.dispose()
  await projections.dispose()
})

test('pipeline executes by (order, id), tracks appliedIds, and respects scope', async () => {
  const { llm, api, dispose } = await setup()
  const trace = []

  const u1 = createUserMessage({ source: { kind: 'user' }, content: [{ type: 'text', text: 'hello' }] })

  // Register out of order to verify deterministic sorting
  const d1 = api.register({
    id: 'b-second',
    order: 200,
    apply(tCtx) {
      trace.push({ id: 'b-second', applied: [...tCtx.appliedIds] })
      if (tCtx.appliedIds.includes('a-first')) return undefined
      return [...tCtx.messages, { role: 'user', content: [{ type: 'text', text: 'fallback' }] }]
    },
  })
  const d2 = api.register({
    id: 'a-first',
    order: 100,
    apply(tCtx) {
      trace.push({ id: 'a-first', applied: [...tCtx.appliedIds] })
      return [...tCtx.messages, { role: 'user', content: [{ type: 'text', text: 'primary' }] }]
    },
  })

  const out = llm.forAdapter({ provider: 'deepseek-account', model: 'm', messages: [u1] }, undefined)
  assert.equal(out.messages.length, 2)
  assert.equal(out.messages[1].content[0].text, 'primary')
  assert.deepEqual(trace, [
    { id: 'a-first', applied: [] },
    { id: 'b-second', applied: ['a-first'] },
  ])

  // Auxiliary purpose call is skipped by default scope ('conversation')
  const aux = llm.forAdapter({ provider: 'deepseek-account', model: 'm', purpose: 'compaction', messages: [u1] }, undefined)
  assert.equal(aux.messages.length, 1)

  d1()
  d2()
  await dispose()
})

test('message diff metering handles closed-turn reasoning stripping and anchor settlement', async () => {
  const { meter, llm, api, dispose } = await setup()

  const undo = api.register({
    id: 'test/strip-reasoning',
    order: 200,
    apply({ messages, lastUserIndex }) {
      if (lastUserIndex <= 0) return undefined
      let changed = false
      const next = messages.map((m, idx) => {
        if (idx >= lastUserIndex || m.role !== 'assistant') return m
        if (!m.content.some(b => b.type === 'reasoning')) return m
        changed = true
        const { replayState: _r, ...source } = m.source
        return { ...m, source, content: m.content.filter(b => b.type !== 'reasoning') }
      })
      return changed ? next : undefined
    },
  })

  const s = makeSession()
  envelope(s)
  prompt(s, 'turn 1')
  settle(s, { inputTokens: 1000, outputTokens: 500, reasoningTokens: 400 })

  // Mid-turn: reasoning still resent, total == billed (1500)
  const mid = meter.measure(s)
  assert.equal(mid.totalTokens, 1500)

  // Close turn with user message
  const u2 = prompt(s, 'turn 2')
  const closed = meter.measure(s)
  const u2Price = estimateMessage(u2.data)
  // total = 1500 - 400 (reasoning) + u2Price
  assert.equal(closed.totalTokens, 1500 - 400 + u2Price)

  // Verify outbound stripping via forAdapter
  const out = llm.forAdapter({
    provider: 'deepseek-account',
    model: 'test',
    sessionId: s.id,
    messages: [
      createUserMessage({ source: { kind: 'user' }, content: [{ type: 'text', text: 'turn 1' }] }),
      createAssistantMessage({ content: WITH_REASONING, source: { provider: 'deepseek-account', model: 'test', replayState: { p: 1 } } }),
      u2.data,
    ],
  }, undefined)
  assert.equal(out.messages[1].content.length, 1)
  assert.equal('replayState' in out.messages[1].source, false)

  undo()
  await dispose()
})

test('message diff metering handles regex text reduction and ephemeral injection with anchor offset', async () => {
  const { meter, llm, api, dispose } = await setup()

  const boilerplate = 'BOILERPLATE_'.repeat(20) // ~240 chars -> ~60 tokens
  let reminderText = 'REMINDER_CONTENT_'.repeat(10)

  const undoStrip = api.register({
    id: 'test/regex-clean',
    order: 100,
    apply({ messages }) {
      let changed = false
      const next = messages.map((m) => {
        if (m.role !== 'user' || !('id' in m)) return m
        const text = m.content[0]?.type === 'text' ? m.content[0].text : ''
        if (!text.includes(boilerplate)) return m
        changed = true
        return { ...m, content: [{ type: 'text', text: text.replace(boilerplate, '') }] }
      })
      return changed ? next : undefined
    },
  })

  const undoRemind = api.register({
    id: 'test/reminder',
    order: 300,
    apply({ messages }) {
      if (!reminderText) return undefined
      return [...messages, { role: 'user', content: [{ type: 'text', text: reminderText }] }]
    },
  })

  const s = makeSession()
  envelope(s)
  const u1 = prompt(s, `${boilerplate}hello`)
  const rawU1Price = estimateMessage(u1.data)
  const cleanU1Price = estimateMessage(createUserMessage({ source: { kind: 'user' }, content: [{ type: 'text', text: 'hello' }] }))
  const reminderPrice = estimateContent([{ type: 'text', text: reminderText }]) + 4

  // Unanchored: surface reflects regex reduction + ephemeral reminder
  const beforeSettle = meter.measure(s)
  assert.equal(beforeSettle.nodes[0].tokens, cleanU1Price)
  assert.ok(beforeSettle.nodes[0].tokens < rawU1Price)
  assert.equal(beforeSettle.totalTokens, cleanU1Price + reminderPrice)

  // Dispatch via forAdapter (records attachedTokens = reminderPrice)
  llm.forAdapter({
    provider: 'deepseek-account',
    model: 'test',
    sessionId: s.id,
    messages: [u1.data],
  }, undefined)

  // Settle assistant message (billed includes the reminder that was sent)
  settle(s, { inputTokens: 1000, outputTokens: 100 }, ANSWER)

  // Same reminder still active: -anchorAttached + currentAttached = 0 -> total == 1100
  const afterSettleSame = meter.measure(s)
  assert.equal(afterSettleSame.totalTokens, 1100)

  // Reminder cleared on next step: -anchorAttached + 0 -> total == 1100 - reminderPrice
  reminderText = ''
  const afterClear = meter.measure(s)
  assert.equal(afterClear.totalTokens, 1100 - reminderPrice)

  undoStrip()
  undoRemind()
  await dispose()
})
