import type { Context } from '@deepseek-ai/cordis'
import type {} from '@deepseek-ai/dsh-agent'
import type {} from '@deepseek-ai/dsh-token-meter'
import type { Session } from '@deepseek-ai/dsh-session'
import Schema from '@deepseek-ai/schemastery'
import { interceptAppend } from './storage.js'
import { interceptMeter } from './meter.js'
import { assertVersions, assertMeterCompatible, assertSessionCompatible } from './compatibility.js'

export const name = 'dsh-ephemeral-reasoning'
export const inject = ['agents', 'tokenMeter']
export interface Config { enabled: boolean }
export const Config: Schema<Config> = Schema.object({
  enabled: Schema.boolean().default(true),
})
const owner = Symbol.for('dsh-ephemeral-reasoning.owner')

export function apply(ctx: Context, config: Config): void {
  if (!config.enabled) return
  if (Reflect.get(ctx.tokenMeter, owner)) throw new Error('Ephemeral reasoning is already mounted for this token meter')
  assertVersions()
  assertMeterCompatible(ctx.tokenMeter)
  const existing = ctx.agents.list()
  // Validate every existing target before installing any patch.
  for (const agent of existing) {
    if (agent.session.header.origin !== 'subagent') assertSessionCompatible(agent.session)
  }
  const sessions = new Map<Session, () => void>()
  const attach = (session: Session) => {
    if (session.header.origin === 'subagent' || sessions.has(session)) return
    assertSessionCompatible(session)
    sessions.set(session, interceptAppend(session))
  }
  ctx.effect(() => {
    Reflect.set(ctx.tokenMeter, owner, true)
    const undoMeter = interceptMeter(ctx.tokenMeter, session => sessions.has(session))
    return () => {
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
