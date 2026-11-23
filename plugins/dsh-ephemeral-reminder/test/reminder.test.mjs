import assert from 'node:assert/strict'
import { test } from 'node:test'
import { Context } from '@deepseek-ai/cordis'
import LlmRuntime, { createAssistantMessage, createUserMessage } from '@deepseek-ai/dsh-llm'
import { Session, SessionId } from '@deepseek-ai/dsh-session'
import { TokenMeter } from '@deepseek-ai/dsh-token-meter'
import { estimateMessage } from '@deepseek-ai/dsh-token-meter/estimate'
import { SessionProjectionRegistry } from '@deepseek-ai/dsh-session-projection'
import * as apiRequestPlugin from 'dsh-api-request'
import * as ephemeralReasoningPlugin from 'dsh-ephemeral-reasoning'
import { DEFAULT_REMINDER_TEXT, appendReminderToLastUser } from '../lib/outbound.js'
import * as plugin from '../lib/index.js'

const make = () => Session.create(SessionId(crypto.randomUUID()))
const HEADER = { config: { provider: 'deepseek-account', model: 'test' } }

function envelope(s, header = HEADER) {
  return s.append('request/header', { header, reason: 'initial' })
}

function prompt(s, text = 'next') {
  return s.append(
    'user/message',
    createUserMessage({ source: { kind: 'user' }, content: [{ type: 'text', text }] }),
    { surfaceOp: 'append' },
  )
}

function settle(s, content = [{ type: 'text', text: 'answer' }]) {
  s.append('step/start', { turn: 1, step: 1 })
  return s.append(
    'assistant/message',
    {
      turn: 1,
      step: 1,
      stream: [],
      message: createAssistantMessage({
        content,
        source: { provider: 'deepseek-account', model: 'test' },
      }),
    },
    { surfaceOp: 'append' },
  )
}

async function setup(pluginConfig = {}, withReasoning = false) {
  const ctx = new Context()
  const projections = ctx.plugin(SessionProjectionRegistry)
  await projections.await()
  const meterFiber = ctx.plugin(TokenMeter)
  await meterFiber.await()
  const llm = new LlmRuntime(ctx)
  const apiFiber = ctx.plugin(apiRequestPlugin)
  await apiFiber.await()
  const reasoningFiber = withReasoning ? ctx.plugin(ephemeralReasoningPlugin) : undefined
  if (reasoningFiber) await reasoningFiber.await()
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
      if (reasoningFiber) await reasoningFiber.dispose()
      await apiFiber.dispose()
      await meterFiber.dispose()
      await projections.dispose()
    },
  }
}

test('appendReminderToLastUser appends only to tail user message and is idempotent', () => {
  const u1 = createUserMessage({ source: { kind: 'user' }, content: [{ type: 'text', text: 'turn 1' }] })
  const a1 = createAssistantMessage({
    content: [{ type: 'text', text: 'answer 1' }],
    source: { provider: 'deepseek-account', model: 'test' },
  })
  const u2 = createUserMessage({ source: { kind: 'user' }, content: [{ type: 'text', text: 'turn 2' }] })
  const toolResult = {
    role: 'tool',
    source: { kind: 'tool', callId: 'c1' },
    toolCallId: 'c1',
    content: [{ type: 'text', text: 'ok' }],
  }

  const out = appendReminderToLastUser([u1, a1, u2, toolResult], DEFAULT_REMINDER_TEXT)
  assert.equal(out[0], u1, 'earlier user message must remain untouched')
  assert.equal(out[1], a1)
  assert.equal(out[3], toolResult)
  assert.deepEqual(out[2].content, [
    { type: 'text', text: 'turn 2' },
    { type: 'text', text: DEFAULT_REMINDER_TEXT },
  ])
  assert.equal(out[2].id, u2.id)

  const secondPass = appendReminderToLastUser(out, DEFAULT_REMINDER_TEXT)
  assert.equal(secondPass, out, 'idempotent when reminder is already present')
})

test('fails closed when apiRequest protocol version mismatches', async () => {
  const ctx = new Context()
  ctx.provide('apiRequest', { protocolVersion: 999, register: () => () => {} })
  await assert.rejects(
    async () => {
      const fiber = ctx.plugin(plugin)
      await fiber.await()
    },
    /protocol v1/,
  )
})

test('forAdapter injects reminder on conversation calls, skips auxiliary purposes, and updates TokenMeter', async () => {
  const env = await setup()
  try {
    const adapter = { id: 'deepseek-account', maxInputChars: 100000 }
    const u1 = createUserMessage({ source: { kind: 'user' }, content: [{ type: 'text', text: 'first' }] })
    const a1 = createAssistantMessage({
      content: [{ type: 'text', text: 'reply' }],
      source: { provider: 'deepseek-account', model: 'test' },
    })
    const u2 = createUserMessage({ source: { kind: 'user' }, content: [{ type: 'text', text: 'second' }] })

    const convOut = env.llm.forAdapter(
      { provider: 'deepseek-account', model: 'test', messages: [u1, a1, u2] },
      adapter,
    )
    assert.equal(convOut.messages[0], u1, 'historical user message untouched')
    assert.deepEqual(convOut.messages[2].content, [
      { type: 'text', text: 'second' },
      { type: 'text', text: DEFAULT_REMINDER_TEXT },
    ])

    const auxOut = env.llm.forAdapter(
      { provider: 'deepseek-account', model: 'test', purpose: 'compaction', messages: [u1, a1, u2] },
      adapter,
    )
    assert.equal(auxOut.messages[2], u2, 'auxiliary compaction request must not be modified')

    // Verify TokenMeter reflects the added reminder tokens on the tail user message
    const s = make()
    envelope(s)
    prompt(s, 'first')
    settle(s)
    prompt(s, 'second')

    const u2Original = createUserMessage({ source: { kind: 'user' }, content: [{ type: 'text', text: 'second' }] })
    const u2WithReminder = {
      ...u2Original,
      content: [...u2Original.content, { type: 'text', text: DEFAULT_REMINDER_TEXT }],
    }
    const expectedDelta = estimateMessage(u2WithReminder) - estimateMessage(u2Original)

    const measured = env.meter.measure(s)
    const rawSurface = estimateMessage(u2Original) * 2 + estimateMessage(a1) - estimateMessage(u2Original) + estimateMessage(createUserMessage({ source: { kind: 'user' }, content: [{ type: 'text', text: 'first' }] }))
    assert.ok(expectedDelta > 0)
    assert.equal(measured.nodes.at(-1).tokens, estimateMessage(u2Original) + expectedDelta)
    assert.equal(measured.nodes[0].tokens, estimateMessage(createUserMessage({ source: { kind: 'user' }, content: [{ type: 'text', text: 'first' }] })))
    assert.ok(measured.totalTokens >= rawSurface)
  } finally {
    await env.dispose()
  }
})

test('composes cleanly with dsh-ephemeral-reasoning in TransformPipeline', async () => {
  const env = await setup({}, true)
  try {
    const adapter = { id: 'deepseek-account', maxInputChars: 100000 }
    const u1 = createUserMessage({ source: { kind: 'user' }, content: [{ type: 'text', text: 'turn 1' }] })
    const a1 = createAssistantMessage({
      content: [
        { type: 'reasoning', text: 'hidden-thought' },
        { type: 'text', text: 'visible-answer' },
      ],
      source: { provider: 'deepseek-account', model: 'test', replayState: { sig: 's1' } },
    })
    const u2 = createUserMessage({ source: { kind: 'user' }, content: [{ type: 'text', text: 'turn 2' }] })

    const out = env.llm.forAdapter(
      { provider: 'deepseek-account', model: 'test', messages: [u1, a1, u2] },
      adapter,
    )

    assert.equal(out.messages[0], u1)
    assert.deepEqual(out.messages[1].content, [{ type: 'text', text: 'visible-answer' }])
    assert.equal(out.messages[1].source.replayState, undefined)
    assert.deepEqual(out.messages[2].content, [
      { type: 'text', text: 'turn 2' },
      { type: 'text', text: DEFAULT_REMINDER_TEXT },
    ])
  } finally {
    await env.dispose()
  }
})
