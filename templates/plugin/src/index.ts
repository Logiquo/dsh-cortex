/**
 * __TITLE__ — __DESCRIPTION__
 *
 * A dsh plugin is a Cordis plugin: it exports `name`, a `Config` schema and an
 * `apply(ctx, config)` function. Declare every service you read from `ctx`
 * (e.g. `agents`, `settings`) in `inject`, or Cordis throws on access.
 */
import type { Context } from '@deepseek-ai/cordis'
import z from '@deepseek-ai/schemastery'

export interface Config {
  enabled: boolean
}

export const Config: z<Config> = z.object({
  enabled: z.boolean().default(true).description('Turn the plugin on or off.'),
})

export const name = '__NAME__'

export const inject: string[] = []

export function apply(ctx: Context, config: Config): void {
  if (!config.enabled) return
  ctx.logger.debug('__NAME__: loaded')
}
