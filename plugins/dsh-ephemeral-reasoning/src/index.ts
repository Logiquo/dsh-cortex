import type { Context } from '@deepseek-ai/cordis'
import type {} from '@deepseek-ai/dsh-agent'
import type {} from '@deepseek-ai/dsh-token-meter'
import type { Session } from '@deepseek-ai/dsh-session'
import Schema from '@deepseek-ai/schemastery'
import { interceptAppend } from './storage.js'
import { assertPressureCompatible, interceptMeter, interceptPressure } from './meter.js'
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
 * The wait is bounded, and running out of it is not fatal: the composer ring is
 * one of two surfaces this plugin corrects, and the token meter it also
 * corrects is enough to keep billing honest. A missing unit therefore degrades
 * with a warning instead of refusing to start.
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
  assertMeterCompatible(ctx.tokenMeter)
  // The occupancy unit registers during the host's own asynchronous init, so
  // this wait is the install's only suspension. Every check below runs after it
  // and nothing awaits again: a list taken before the wait would miss sessions
  // created while it ran, because the creation listener is registered after it.
  const registry = await pressureRegistry(ctx)
  const existing = ctx.agents.list()
  // Validate every existing target before installing any patch.
  for (const agent of existing) assertSessionCompatible(agent.session)
  // Everything that can refuse is decided above, before the first mutation.
  if (registry !== undefined) assertPressureCompatible(registry)
  if (Reflect.get(ctx.tokenMeter, owner)) throw new Error('Ephemeral reasoning is already mounted for this token meter')
  Reflect.set(ctx.tokenMeter, owner, true)

  // Disposers run in reverse install order: sessions first, then the occupancy
  // unit, then the meter, and the owner marker last so a reload cannot race a
  // half-unloaded mount.
  const undos: (() => void)[] = []
  const release = () => { for (const undo of undos.splice(0).reverse()) undo() }
  undos.push(() => Reflect.deleteProperty(ctx.tokenMeter, owner))
  const sessions = new Map<Session, () => void>()
  try {
    // Registered first, so a disposal that lands between the installs below
    // still runs every disposer collected so far.
    ctx.effect(() => release)
    undos.push(interceptMeter(ctx.tokenMeter))
    // Installed after the validation above, so a refused override can never
    // leave the meter patched for the rest of the process.
    if (registry !== undefined) undos.push(interceptPressure(registry))
  } catch (error) {
    release()
    throw error
  }
  undos.push(() => {
    for (const undo of sessions.values()) undo()
    sessions.clear()
  })

  /**
   * Sanitize one session's writes.
   *
   * A session this plugin cannot wrap keeps host behaviour rather than
   * rejecting the agent that owns it: failing here would surface as a failed
   * `agent/created` for a session the harness has already built. The warning is
   * the record that this session's reasoning is not being removed.
   * @param session - session to wrap.
   */
  const attach = (session: Session) => {
    if (sessions.has(session)) return
    try {
      assertSessionCompatible(session)
      sessions.set(session, interceptAppend(session))
    } catch (error) {
      ctx.logger.warn(`session ${String(session.id)} was not sanitized: ${error instanceof Error ? error.message : String(error)}`)
    }
  }
  ctx.on('agent/created', ({ agent }) => { attach(agent.session); return undefined })
  ctx.on('agent/disposed', ({ agent }) => {
    sessions.get(agent.session)?.()
    sessions.delete(agent.session)
  })
  for (const agent of existing) attach(agent.session)
}
