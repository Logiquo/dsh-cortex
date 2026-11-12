import type { Context } from '@deepseek-ai/cordis'
import type {} from '@deepseek-ai/dsh-agent'
import type {} from '@deepseek-ai/dsh-token-meter'
import type { Session } from '@deepseek-ai/dsh-session'
import Schema from '@deepseek-ai/schemastery'
import { interceptAppend } from './storage.js'
import { interceptMeter, interceptPressure } from './meter.js'
import { assertVersions, assertMeterCompatible, assertSessionCompatible } from './compatibility.js'

export const name = 'dsh-ephemeral-reasoning'
export const inject = ['agents', 'tokenMeter', 'sessionProjections']
export interface Config { enabled: boolean }
export const Config: Schema<Config> = Schema.object({
  enabled: Schema.boolean().default(true),
})
const owner = Symbol.for('dsh-ephemeral-reasoning.owner')

/**
 * Wait for the host occupancy unit to be registered.
 *
 * The registry is a service whose owner registers its units during an
 * asynchronous `init`, so an injected dependant can run before that settles.
 * The wait is bounded so a missing unit fails loudly here instead of silently
 * leaving the composer ring on the host's prompt-side number.
 * @param ctx - plugin context.
 * @returns the `sessionProjections` service once `contextPressure` exists.
 */
async function pressureRegistry(ctx: Context): Promise<unknown> {
  const deadline = Date.now() + 2000
  while (true) {
    const registry = ctx.get('sessionProjections') as unknown
    if (registry !== undefined && (registry as { registrations?: Map<string, unknown> }).registrations?.has('contextPressure') === true) return registry
    if (Date.now() >= deadline) throw new Error('Ephemeral reasoning: the contextPressure projection did not register; refusing to start with an uncorrected occupancy reading')
    await new Promise(resolve => setTimeout(resolve, 4))
  }
}

export async function apply(ctx: Context, config: Config): Promise<void> {
  if (!config.enabled) return
  if (Reflect.get(ctx.tokenMeter, owner)) throw new Error('Ephemeral reasoning is already mounted for this token meter')
  assertVersions()
  assertMeterCompatible(ctx.tokenMeter)
  const existing = ctx.agents.list()
  // Validate every existing target before installing any patch.
  for (const agent of existing) assertSessionCompatible(agent.session)
  const registry = await pressureRegistry(ctx)
  const sessions = new Map<Session, () => void>()
  const attach = (session: Session) => {
    if (sessions.has(session)) return
    assertSessionCompatible(session)
    sessions.set(session, interceptAppend(session))
  }
  ctx.effect(() => {
    Reflect.set(ctx.tokenMeter, owner, true)
    // Every session this plugin sanitizes is measured and reported the same way,
    // so the meter and the occupancy unit cannot disagree about any of them.
    const undoMeter = interceptMeter(ctx.tokenMeter, () => true)
    // Installed after the wait above, so a refused override aborts startup
    // before any other patch is left behind.
    const undoPressure = interceptPressure(registry)
    return () => {
      undoPressure()
      undoMeter()
      for (const undo of sessions.values()) undo()
      sessions.clear()
      Reflect.deleteProperty(ctx.tokenMeter, owner)
    }
  })
  ctx.on('agent/created', ({ agent }) => { attach(agent.session); return undefined })
  ctx.on('agent/disposed', ({ agent }) => {
    sessions.get(agent.session)?.()
    sessions.delete(agent.session)
  })
  for (const agent of existing) attach(agent.session)
}
