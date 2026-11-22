import type { Context } from '@deepseek-ai/cordis'
import type {} from '@deepseek-ai/dsh-llm'
import type {} from '@deepseek-ai/dsh-token-meter'
import Schema from '@deepseek-ai/schemastery'
import { interceptLlm } from './outbound.js'
import { assertPressureCompatible, interceptMeter, interceptPressure } from './meter.js'
import { assertVersions, assertMeterCompatible, assertLlmCompatible } from './compatibility.js'

export const name = 'dsh-ephemeral-reasoning'
export const inject = ['llm', 'tokenMeter', 'sessionProjections']
export interface Config { enabled: boolean }
export const Config: Schema<Config> = Schema.object({
  enabled: Schema.boolean().default(true),
})
const owner = Symbol.for('dsh-ephemeral-reasoning.owner')

/**
 * Wait for the host occupancy unit to be registered.
 * @param ctx - plugin context.
 * @returns the `sessionProjections` service once `contextPressure` exists, or
 *   undefined when it did not appear in time.
 */
async function pressureRegistry(ctx: Context): Promise<unknown> {
  const deadline = Date.now() + 30_000
  while (true) {
    const registry = ctx.get('sessionProjections') as unknown
    if (registry !== undefined && (registry as { registrations?: Map<string, unknown> }).registrations?.has('contextPressure') === true) return registry
    if (Date.now() >= deadline) {
      ctx.logger.warn('the contextPressure projection did not register; the composer occupancy ring keeps the host reading while the token meter is still corrected')
      return undefined
    }
    await new Promise(resolve => setTimeout(resolve, 4))
  }
}

export async function apply(ctx: Context, config: Config): Promise<void> {
  if (!config.enabled) return
  assertVersions()
  assertLlmCompatible(ctx.llm)
  assertMeterCompatible(ctx.tokenMeter)
  const registry = await pressureRegistry(ctx)
  if (registry !== undefined) assertPressureCompatible(registry)
  if (Reflect.get(ctx.tokenMeter, owner)) throw new Error('Ephemeral reasoning is already mounted for this token meter')
  Reflect.set(ctx.tokenMeter, owner, true)

  const undos: (() => void)[] = []
  const release = () => { for (const undo of undos.splice(0).reverse()) undo() }
  undos.push(() => Reflect.deleteProperty(ctx.tokenMeter, owner))
  try {
    ctx.effect(() => release)
    undos.push(interceptLlm(ctx.llm))
    undos.push(interceptMeter(ctx.tokenMeter))
    if (registry !== undefined) undos.push(interceptPressure(registry))
  } catch (error) {
    release()
    throw error
  }
}
