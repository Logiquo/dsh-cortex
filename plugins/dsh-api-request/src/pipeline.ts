import { freezeMessage } from '@deepseek-ai/dsh-llm'
import type { LlmRequestPurpose, Message, MessageId, RequestMessage } from '@deepseek-ai/dsh-llm'
import type { SessionId } from '@deepseek-ai/dsh-session'
import { estimateContent, estimateMessage } from '@deepseek-ai/dsh-token-meter/estimate'
import type { DurableMessageDiff, PipelineDiff, PolicyPurposeScope, RequestTransform } from './types.js'

export interface PipelineInput {
  readonly messages: readonly RequestMessage[]
  readonly sessionId?: SessionId | undefined
  readonly purpose?: LlmRequestPurpose | undefined
  readonly provider?: string | undefined
  readonly model?: string | undefined
}

export function findLastUserIndex(messages: readonly RequestMessage[]): number {
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i]?.role === 'user') return i
  }
  return -1
}

function matchesScope(scope: PolicyPurposeScope | undefined, purpose: LlmRequestPurpose | undefined): boolean {
  if (scope === 'all') return true
  if (scope === undefined || scope === 'conversation') return purpose === undefined
  return purpose !== undefined && scope.includes(purpose)
}

function hasId(message: RequestMessage): message is Message {
  return 'id' in message && typeof message.id === 'string'
}

function messagePrice(message: RequestMessage): number {
  if (message.content.length === 0) return 0
  return estimateMessage(message as Message)
}

function reasoningHeuristic(message: RequestMessage): number {
  const reasoning = message.content.filter(block => block.type === 'reasoning')
  return reasoning.length === 0 ? 0 : estimateContent(reasoning)
}

export class TransformPipeline {
  private readonly transforms = new Map<string, RequestTransform>()
  private sorted: readonly RequestTransform[] = []

  register(transform: RequestTransform): () => void {
    if (!transform.id) throw new Error('dsh-api-request: transform id must be non-empty')
    if (!Number.isFinite(transform.order)) throw new Error(`dsh-api-request: transform "${transform.id}" order must be finite`)
    if (this.transforms.has(transform.id)) {
      throw new Error(`dsh-api-request: transform "${transform.id}" is already registered`)
    }
    this.transforms.set(transform.id, transform)
    this.rebuildSorted()
    return () => {
      if (this.transforms.get(transform.id) !== transform) return
      this.transforms.delete(transform.id)
      this.rebuildSorted()
    }
  }

  list(): readonly RequestTransform[] {
    return this.sorted
  }

  private rebuildSorted(): void {
    this.sorted = [...this.transforms.values()].sort((a, b) => {
      if (a.order !== b.order) return a.order - b.order
      return a.id.localeCompare(b.id)
    })
  }

  run(input: PipelineInput): PipelineDiff {
    const before = input.messages
    let current = before
    const appliedIds: string[] = []

    for (const transform of this.sorted) {
      if (!matchesScope(transform.scope, input.purpose)) continue
      const next = transform.apply({
        sessionId: input.sessionId,
        purpose: input.purpose,
        provider: input.provider,
        model: input.model,
        messages: current,
        appliedIds: [...appliedIds],
        lastUserIndex: findLastUserIndex(current),
      })
      if (next !== undefined && next !== current) {
        current = next
        appliedIds.push(transform.id)
      }
    }

    return diffMessages(before, current, appliedIds)
  }
}

export function diffMessages(
  before: readonly RequestMessage[],
  after: readonly RequestMessage[],
  appliedIds: readonly string[],
): PipelineDiff {
  if (after === before) {
    return {
      messages: after,
      appliedIds,
      durableDiffs: new Map(),
      attachedTokens: 0,
    }
  }

  const beforeById = new Map<MessageId, Message>()
  let beforeEphemeralTokens = 0
  for (const msg of before) {
    if (hasId(msg)) beforeById.set(msg.id, msg)
    else beforeEphemeralTokens += messagePrice(msg)
  }

  const afterById = new Map<MessageId, RequestMessage>()
  let afterEphemeralTokens = 0
  for (const msg of after) {
    if (hasId(msg)) afterById.set(msg.id, msg)
    else afterEphemeralTokens += messagePrice(msg)
  }

  const durableDiffs = new Map<MessageId, DurableMessageDiff>()

  for (const [id, beforeMsg] of beforeById) {
    const afterMsg = afterById.get(id)
    if (afterMsg === beforeMsg) continue

    const beforeTokens = messagePrice(beforeMsg)
    if (afterMsg === undefined) {
      const beforeReasoning = reasoningHeuristic(beforeMsg)
      durableDiffs.set(id, {
        id,
        beforeTokens,
        afterTokens: 0,
        deltaTokens: -beforeTokens,
        removedReasoningTokens: beforeReasoning,
        removedOtherTokens: Math.max(0, beforeTokens - beforeReasoning),
      })
      continue
    }

    const afterTokens = messagePrice(afterMsg)
    const deltaTokens = afterTokens - beforeTokens
    if (deltaTokens === 0) continue

    const beforeReasoning = reasoningHeuristic(beforeMsg)
    const afterReasoning = reasoningHeuristic(afterMsg)
    const removedReasoningTokens = Math.max(0, beforeReasoning - afterReasoning)
    const removedTotal = Math.max(0, beforeTokens - afterTokens)
    const removedOtherTokens = Math.max(0, removedTotal - removedReasoningTokens)

    durableDiffs.set(id, {
      id,
      beforeTokens,
      afterTokens,
      deltaTokens,
      removedReasoningTokens,
      removedOtherTokens,
    })
  }

  const attachedTokens = Math.max(0, afterEphemeralTokens - beforeEphemeralTokens)
  return {
    messages: after,
    appliedIds,
    durableDiffs,
    attachedTokens,
  }
}

export function freezeModifiedMessages(
  before: readonly RequestMessage[],
  after: readonly RequestMessage[],
  freezeResult: boolean,
): RequestMessage[] {
  if (after === before) return after as RequestMessage[]
  const beforeSet = new Set(before)
  const normalized = after.map(msg => (beforeSet.has(msg) || Object.isFrozen(msg) ? msg : freezeMessage(msg as Message)))
  return (freezeResult ? Object.freeze(normalized) : normalized) as RequestMessage[]
}
