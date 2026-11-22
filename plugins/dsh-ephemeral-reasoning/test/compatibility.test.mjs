import assert from 'node:assert/strict'
import { test } from 'node:test'
import { Context } from '@deepseek-ai/cordis'
import LlmRuntime from '@deepseek-ai/dsh-llm'
import { TokenMeter } from '@deepseek-ai/dsh-token-meter'
import { SessionProjectionRegistry } from '@deepseek-ai/dsh-session-projection'
import { assertVersions, assertLlmCompatible, assertMeterCompatible, SUPPORTED_DSH_VERSION } from '../lib/compatibility.js'
import * as plugin from '../lib/index.js'

const names = ['agent', 'session', 'llm', 'token-meter'].map(n => '@deepseek-ai/dsh-' + n)
const versions = Object.fromEntries(names.map(n => [n, SUPPORTED_DSH_VERSION]))

test('exact package versions are required, including every prerelease component', () => {
  assert.doesNotThrow(() => assertVersions())
  assert.doesNotThrow(() => assertVersions(versions))
  for (const name of names) {
    for (const version of ['0.2.2-rc.4', '0.2.2', '0.2.1-rc.1']) {
      assert.throws(() => assertVersions({ ...versions, [name]: version }), new RegExp(name.replaceAll('.', '\\.') + '@0\\.2\\.2-rc\\.5'))
    }
    const missing = { ...versions }
    delete missing[name]
    assert.throws(() => assertVersions(missing), /found missing/)
  }
})

test('actual incompatible or previously wrapped llm/meter implementations are rejected', () => {
  const ctx = new Context()
  const llm = new LlmRuntime(ctx)
  assertLlmCompatible(llm)
  Object.defineProperty(llm, 'forAdapter', { configurable: true, writable: true, value() { throw Error('unexpected') } })
  assert.throws(() => assertLlmCompatible(llm), /LlmRuntime\.forAdapter/)
  assert.throws(() => assertMeterCompatible({ measure() {} }), /TokenMeter\.measure/)
})

test('startup validates llm before patching the real Cordis meter', async () => {
  const ctx = new Context()
  const projections = ctx.plugin(SessionProjectionRegistry)
  await projections.await()
  const meter = ctx.plugin(TokenMeter)
  await meter.await()
  const llm = new LlmRuntime(ctx)
  Object.defineProperty(llm, 'forAdapter', { configurable: true, writable: true, value() { throw Error('foreign wrapper') } })
  const before = Object.getOwnPropertyDescriptor(ctx.tokenMeter, 'measure')
  const fiber = ctx.plugin(plugin, {})
  await assert.rejects(fiber.await(), /LlmRuntime\.forAdapter/)
  assert.deepEqual(Object.getOwnPropertyDescriptor(ctx.tokenMeter, 'measure'), before)
  assert.equal(Reflect.get(ctx.tokenMeter, Symbol.for('dsh-ephemeral-reasoning.owner')), undefined)
  await fiber.dispose()
  await meter.dispose()
  await projections.dispose()
})
