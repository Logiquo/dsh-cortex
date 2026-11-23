import { Service } from '@deepseek-ai/cordis'
import type { Context } from '@deepseek-ai/cordis'
import type { RequestMessage } from '@deepseek-ai/dsh-llm'
import { deriveEventMessage } from '@deepseek-ai/dsh-session'
import type { SessionEvent } from '@deepseek-ai/dsh-session'
import { resolveHostDriver } from './drivers/registry.js'
import { TransformPipeline } from './pipeline.js'
import {
  API_REQUEST_PROTOCOL,
  type ApiRequestService,
  type HostDriverInfo,
  type RequestTransform,
} from './types.js'

export { API_REQUEST_PROTOCOL }
export type {
  ApiRequestService,
  DurableMessageDiff,
  HostDriverInfo,
  PipelineDiff,
  PolicyPurposeScope,
  RequestTransform,
  TransformContext,
} from './types.js'
export { TransformPipeline, diffMessages } from './pipeline.js'

export const name = 'dsh-api-request'
export const inject = ['llm', 'tokenMeter', 'sessionProjections']

export class ApiRequest extends Service implements ApiRequestService {
  static inject = ['llm', 'tokenMeter', 'sessionProjections']

  readonly protocolVersion = API_REQUEST_PROTOCOL
  hostDriver: HostDriverInfo = { id: 'uninitialized', dshVersion: 'unknown' }
  private readonly pipeline = new TransformPipeline()
  private uninstall: (() => void) | undefined

  constructor(ctx: Context) {
    super(ctx, 'apiRequest')
  }

  async [Service.init](): Promise<void> {
    const { driver, dshVersion } = await resolveHostDriver(this.ctx)
    this.hostDriver = { id: driver.id, dshVersion }
    const uninstall = await driver.install(this.ctx, this.pipeline)
    this.uninstall = uninstall
    this.ctx.effect(() => () => {
      this.uninstall?.()
      this.uninstall = undefined
    })
  }

  register(transform: RequestTransform): () => void {
    return this.pipeline.register(transform)
  }

  estimateHiddenTokens(event: SessionEvent, isClosedTurn: boolean): number {
    const msg = deriveEventMessage(event)
    if (msg === null) return 0
    const messages: RequestMessage[] = isClosedTurn
      ? [msg, { role: 'user', content: [{ type: 'text', text: '' }] }]
      : [msg]
    const diff = this.pipeline.run({ messages, purpose: undefined })
    const d = diff.durableDiffs.get(msg.id)
    if (d === undefined || d.deltaTokens >= 0) return 0
    return -d.deltaTokens
  }
}

export async function apply(ctx: Context): Promise<void> {
  await ctx.plugin(ApiRequest).await()
}
