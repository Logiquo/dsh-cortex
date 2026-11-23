import type { Context } from '@deepseek-ai/cordis'
import Schema from '@deepseek-ai/schemastery'
import { API_REQUEST_PROTOCOL } from 'dsh-api-request'
import { stripClosedTurnReasoning } from './outbound.js'

export const name = 'dsh-ephemeral-reasoning'
export const inject = ['apiRequest']
export interface Config { enabled: boolean }
export const Config: Schema<Config> = Schema.object({
  enabled: Schema.boolean().default(true),
})

export function apply(ctx: Context, config: Config): void {
  if (!config.enabled) return
  if (ctx.apiRequest.protocolVersion !== API_REQUEST_PROTOCOL) {
    throw new Error(
      `Ephemeral reasoning requires dsh-api-request protocol v${API_REQUEST_PROTOCOL}; found v${ctx.apiRequest.protocolVersion}.`,
    )
  }

  ctx.effect(() =>
    ctx.apiRequest.register({
      id: 'dsh-ephemeral-reasoning/strip',
      order: 200,
      apply({ messages, lastUserIndex }) {
        if (lastUserIndex <= 0) return undefined
        const next = stripClosedTurnReasoning(messages)
        return next === messages ? undefined : next
      },
    }),
  )
}
