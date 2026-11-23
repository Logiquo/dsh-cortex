import assert from 'node:assert/strict'
import { test } from 'node:test'
import { Context } from '@deepseek-ai/cordis'
import LlmRuntime, { createAssistantMessage, createUserMessage } from '@deepseek-ai/dsh-llm'
import { Session, SessionId } from '@deepseek-ai/dsh-session'
import { TokenMeter } from '@deepseek-ai/dsh-token-meter'
import { estimateContent, estimateMessage } from '@deepseek-ai/dsh-token-meter/estimate'
import { SessionProjectionRegistry } from '@deepseek-ai/dsh-session-projection'
import * as apiRequestPlugin from 'dsh-api-request'
import { stripClosedTurnReasoning } from '../lib/outbound.js'
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

const reasoningPrice = estimateContent(REASONING)

async function setup(pluginConfig = {}) {
  const ctx = new Context()
  const projections = ctx.plugin(SessionProjectionRegistry)
  await projections.await()
  const meterFiber = ctx.plugin(TokenMeter)
  await meterFiber.await()
  const llm = new LlmRuntime(ctx)
  const apiFiber = ctx.plugin(apiRequestPlugin)
  await apiFiber.await()
  const fiber = ctx.plugin(plugin, pluginConfig)
  await fiber.await()
  return {
    ctx,
    fiber,
    apiFiber,
    meter: ctx.tokenMeter,
    llm,
    dispose: async () => {
      await fiber.dispose()
      await apiFiber.dispose()
      await meterFiber.dispose()
      await projections.dispose()
    },
  }
}

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

test('fails closed when apiRequest protocol version mismatches', async () => {
  const ctx = new Context()
  ctx.provide('apiRequest', { protocolVersion: 999, register: () => () => {} })
  const fiber = ctx.plugin(plugin, {})
  await assert.rejects(fiber.await(), /protocol v1; found v999/)
  await fiber.dispose()
})

test('disabled plugin leaves outbound messages untouched', async () => {
  const { llm, dispose } = await setup({ enabled: false })
  const u1 = createUserMessage({ source: { kind: 'user' }, content: [{ type: 'text', text: 't1' }] })
  const a1 = createAssistantMessage({ content: WITH_REASONING, source: { provider: 'deepseek-account', model: 'm', replayState: { s: 1 } } })
  const u2 = createUserMessage({ source: { kind: 'user' }, content: [{ type: 'text', text: 't2' }] })
  const res = llm.forAdapter({ provider: 'deepseek-account', model: 'm', messages: [u1, a1, u2] }, undefined)
  assert.equal(res.messages[1].content.length, 2)
  await dispose()
})

test('registers via dsh-api-request: strips outbound closed turns, corrects meter & ring, and cleans up on dispose', async () => {
  const { fiber, meter, llm, dispose } = await setup()

  const s = make()
  envelope(s)
  prompt(s, 'turn 1')
  const a1Event = settle(s, { inputTokens: 1000, outputTokens: 500, reasoningTokens: 400 })

  // Mid-turn: reasoning still resent
  assert.equal(meter.measure(s).totalTokens, 1500)
  assert.equal(meter.contextPressure(s).pressureTokens, 1500)

  // Close turn: reasoning dropped from totalTokens, closed node, and contextPressure
  const u2 = prompt(s, 'turn 2')
  const u2Price = estimateMessage(u2.data)
  const closed = meter.measure(s)
  assert.equal(closed.totalTokens, 1500 - 400 + u2Price)
  assert.equal(closed.nodes[1].tokens, estimateMessage(a1Event.data.message) - reasoningPrice)
  assert.equal(meter.contextPressure(s).pressureTokens, closed.totalTokens)

  // Outbound forAdapter strips closed-turn reasoning
  const u1Msg = createUserMessage({ source: { kind: 'user' }, content: [{ type: 'text', text: 'turn 1' }] })
  const a1Msg = createAssistantMessage({ content: WITH_REASONING, source: { provider: 'deepseek-account', model: 'test', replayState: { s: 1 } } })
  const out = llm.forAdapter({ provider: 'deepseek-account', model: 'test', messages: [u1Msg, a1Msg, u2.data] }, undefined)
  assert.equal(out.messages[1].content.length, 1)

  // Disposing ephemeral-reasoning unregisters only its transform while dsh-api-request stays active
  await fiber.dispose()
  const afterDispose = llm.forAdapter({ provider: 'deepseek-account', model: 'test', messages: [u1Msg, a1Msg, u2.data] }, undefined)
  assert.equal(afterDispose.messages[1].content.length, 2)

  await dispose()
})
