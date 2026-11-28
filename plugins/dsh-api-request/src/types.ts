import type {
  LlmRequestPurpose,
  MessageId,
  RequestMessage,
} from '@deepseek-ai/dsh-llm'
import type { SessionId } from '@deepseek-ai/dsh-session'

export const API_REQUEST_PROTOCOL = 1 as const

declare module '@deepseek-ai/cordis' {
  interface Context {
    apiRequest: ApiRequestService
  }
}

export type PolicyPurposeScope =
  | 'conversation'
  | 'all'
  | readonly LlmRequestPurpose[]

export interface HostDriverInfo {
  readonly id: string
  readonly dshVersion: string
}

export interface TransformContext {
  readonly sessionId?: SessionId | undefined
  readonly purpose?: LlmRequestPurpose | undefined
  readonly provider?: string | undefined
  readonly model?: string | undefined
  readonly messages: readonly RequestMessage[]
  readonly appliedIds: readonly string[]
  readonly lastUserIndex: number
}

export interface RequestTransform {
  readonly id: string
  readonly order: number
  readonly scope?: PolicyPurposeScope
  apply(ctx: TransformContext): readonly RequestMessage[] | undefined
}

export interface DurableMessageDiff {
  readonly id: MessageId
  readonly beforeTokens: number
  readonly afterTokens: number
  /** Signed delta: afterTokens - beforeTokens. */
  readonly deltaTokens: number
  /** Heuristic tokens of reasoning blocks removed from this message. */
  readonly removedReasoningTokens: number
  /** Heuristic tokens of non-reasoning blocks removed from this message. */
  readonly removedOtherTokens: number
}

export interface PipelineDiff {
  readonly messages: readonly RequestMessage[]
  readonly appliedIds: readonly string[]
  readonly durableDiffs: ReadonlyMap<MessageId, DurableMessageDiff>
  readonly attachedTokens: number
}

export interface ApiRequestService {
  readonly protocolVersion: number
  readonly hostDriver: HostDriverInfo
  register(transform: RequestTransform): () => void
}
