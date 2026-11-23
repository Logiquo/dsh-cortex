import type { Context } from '@deepseek-ai/cordis'
import Schema from '@deepseek-ai/schemastery'
import { API_REQUEST_PROTOCOL } from 'dsh-api-request'
import { DEFAULT_REMINDER_TEXT, appendReminderToLastUser } from './outbound.js'

export const name = 'dsh-ephemeral-reminder'
export const inject = ['apiRequest']

export interface Config {
  enabled: boolean
  reminderText: string
  order: number
}

export const Config: Schema<Config> = Schema.object({
  enabled: Schema.boolean().default(true),
  reminderText: Schema.string().role('textarea').default(DEFAULT_REMINDER_TEXT),
  order: Schema.number().default(300),
})

export function apply(ctx: Context, config: Config): void {
  if (!config.enabled) return
  if (ctx.apiRequest.protocolVersion !== API_REQUEST_PROTOCOL) {
    throw new Error(
      `Ephemeral reminder requires dsh-api-request protocol v${API_REQUEST_PROTOCOL}; found v${ctx.apiRequest.protocolVersion}.`,
    )
  }
  if (!config.reminderText.trim()) return

  ctx.effect(() =>
    ctx.apiRequest.register({
      id: 'dsh-ephemeral-reminder/append',
      order: config.order,
      apply({ messages, lastUserIndex }) {
        if (lastUserIndex < 0) return undefined
        const next = appendReminderToLastUser(messages, config.reminderText, lastUserIndex)
        return next === messages ? undefined : next
      },
    }),
  )
}
