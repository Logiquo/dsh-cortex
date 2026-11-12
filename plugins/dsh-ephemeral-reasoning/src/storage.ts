import { freezeMessage } from '@deepseek-ai/dsh-llm'
import type { AssistantStreamRecord } from '@deepseek-ai/dsh-llm'
import type { Session, SessionEventMap } from '@deepseek-ai/dsh-session'
import { estimateContent } from '@deepseek-ai/dsh-token-meter/estimate'
import { patchMethod } from './patch.js'

/**
 * Field sanitation adds to a settlement so the measurement can price the
 * reasoning it removed.
 *
 * The provider reports one exact `outputTokens` for reasoning and visible text
 * together, so the split has to be inferred. Inferring it from the visible text
 * alone divides by a fixed chars-per-token density, which is only right for
 * space-delimited scripts: Chinese, Japanese, Korean, and Thai run about one
 * token per character, so the visible price collapses to a quarter of the real
 * count and the shortfall looks like reasoning that was never there. Recording
 * the removed reasoning's own price turns the guess into a ratio of two prices
 * under the same heuristic, where that density error cancels.
 *
 * The field is additive: unknown payload keys are accepted and persisted, and
 * no reader of `assistant/message` treats it as content.
 */
export const REASONING_EVIDENCE = 'ephemeralReasoning'

/** Reasoning a sanitation pass removed from one settlement. */
interface ReasoningEvidence {
  /** Heuristic price of the removed reasoning blocks, in the host's own units. */
  readonly removedTokens: number
}

/**
 * Read the reasoning price a sanitation pass recorded for one settlement.
 * @param data - the logged event payload.
 * @returns the recorded price, or undefined for a settlement written before this
 *   evidence existed or by a session this plugin did not sanitize.
 */
export function removedReasoningTokens(data: unknown): number | undefined {
  if (typeof data !== 'object' || data === null) return undefined
  const evidence = (data as Record<string, unknown>)[REASONING_EVIDENCE]
  if (typeof evidence !== 'object' || evidence === null) return undefined
  const removed = (evidence as Record<string, unknown>)['removedTokens']
  if (typeof removed !== 'number' || !Number.isSafeInteger(removed) || removed < 0) return undefined
  return removed
}

/**
 * Price the reasoning blocks a sanitation pass removes from an assembled message.
 *
 * Recorded for every sanitized settlement, including the ones that carried no
 * reasoning at all. A zero here is a fact — "the whole output was visible text"
 * — and it is what stops a denser script from being read as reasoning.
 * @param content - the message content before sanitation.
 * @returns heuristic tokens of the reasoning blocks; zero when there are none.
 */
function reasoningEvidence(content: SessionEventMap['assistant/message']['message']['content']): ReasoningEvidence {
  const reasoning = content.filter(block => block.type === 'reasoning')
  return { removedTokens: reasoning.length === 0 ? 0 : estimateContent(reasoning) }
}

/**
 * Whether a recorded stream carries reasoning that sanitation removes.
 *
 * Used to decide the replay envelope, which is index-aligned with the blocks the
 * stream produced: it may only be dropped from a stream that actually lost
 * blocks, or the stored stream would no longer describe its own message.
 * @param stream - recorded stream entries.
 * @returns true when sanitation would remove at least one record.
 */
function streamHadReasoning(stream: readonly AssistantStreamRecord[]): boolean {
  for (const record of stream) {
    if (record.type === 'reasoning-chunks') return true
    if (record.type !== 'chunk') continue
    const chunk = record.chunk
    if (chunk.type === 'reasoning-delta') return true
    if (chunk.type === 'block-start' && chunk.blockType === 'reasoning') return true
    if (chunk.type === 'block-end' && chunk.block.type === 'reasoning') return true
  }
  return false
}

/**
 * Filter packed records directly: no token-by-token expansion of discarded reasoning.
 * @param stream - recorded stream entries.
 * @param stripReplayState - drop the terminal replay envelope because the blocks
 *   it describes went away. It must be decided by the caller that also decides
 *   the message's own replay state, so the two never disagree: dsh verifies that
 *   a stored settlement reassembles to its own message, usage and replay state.
 * @returns the entries a model would see, with reasoning removed and the block
 *   indices renumbered to stay dense.
 */
export function cleanStream(stream: readonly AssistantStreamRecord[], stripReplayState = false): AssistantStreamRecord[] {
  const result: AssistantStreamRecord[] = []
  const indices = new Map<number, number | null>()
  let nextIndex = 0
  const retainedIndex = (original: number): number | null => {
    if (!indices.has(original)) indices.set(original, nextIndex++)
    return indices.get(original) ?? null
  }
  for (const record of stream) {
    if (record.type === 'reasoning-chunks') continue
    if (record.type !== 'chunk') {
      const index = retainedIndex(record.index)
      if (index !== null) result.push({ ...record, index })
      continue
    }
    let chunk = record.chunk
    if (chunk.type === 'block-start' && chunk.blockType === 'reasoning') {
      indices.set(chunk.index, null)
      continue
    }
    if (chunk.type === 'reasoning-delta' || (chunk.type === 'block-end' && chunk.block.type === 'reasoning')) continue
    if ('index' in chunk) {
      const index = retainedIndex(chunk.index)
      if (index === null) continue
      chunk = { ...chunk, index }
    }
    if (chunk.type === 'finish' && stripReplayState) {
      const { replayState: _state, ...finish } = chunk
      chunk = finish
    }
    result.push({ ...record, chunk })
  }
  return result
}

/**
 * Sanitize one event payload.
 *
 * A settlement that removed reasoning also drops its provider replay state: that
 * envelope is index-aligned with the blocks it described, so keeping it after
 * the reasoning blocks are gone would answer the wrong block count. The same
 * decision governs the stream's terminal `finish` record, because dsh verifies
 * that a stored settlement still reassembles to its own message, usage and
 * replay state; dropping it from only one of the two makes that check fail on
 * every settlement that carried no reasoning at all. A settlement that removed
 * nothing keeps its replay state untouched in both places — the sanitizer
 * rewrites what it must and leaves every other field exactly as the producer
 * wrote it.
 * @param type - event type.
 * @param data - event payload as the producer wrote it.
 * @returns the payload to admit, untouched when it is not a settlement this
 *   sanitizer can read. Admission validates every payload; a sanitizer that
 *   threw its own `TypeError` on a malformed one would replace the host's
 *   precise rejection with a stack trace that says nothing about the caller's
 *   mistake.
 */
function cleanData(type: string, data: unknown): unknown {
  if (type === 'assistant/attempt') {
    if (typeof data !== 'object' || data === null) return data
    const attempt = data as SessionEventMap['assistant/attempt']
    if (!Array.isArray(attempt.stream)) return data
    // A settlement that committed no surface message has no stored message to
    // agree with, so only a stream that lost blocks gives up its envelope.
    return { ...attempt, stream: cleanStream(attempt.stream, streamHadReasoning(attempt.stream)) }
  }
  if (type !== 'assistant/message') return data
  if (typeof data !== 'object' || data === null) return data
  const response = data as SessionEventMap['assistant/message']
  const message = response.message
  if (typeof message !== 'object' || message === null || !Array.isArray(message.content)) return data
  const content = message.content.filter(block => block.type !== 'reasoning')
  const evidence = reasoningEvidence(message.content)
  const stripReplayState = evidence.removedTokens > 0
  // A settlement whose `stream` is absent is admitted unchanged; older writers
  // and third-party loops omit it, and a sanitizer must never be the reason a
  // write fails.
  const stream = Array.isArray(response.stream) ? { stream: cleanStream(response.stream, stripReplayState) } : {}
  const repaired = { ...response, ...stream, [REASONING_EVIDENCE]: evidence }
  if (!stripReplayState) return repaired
  const { replayState: _state, ...source } = message.source
  return {
    ...repaired,
    message: freezeMessage({ ...message, source, content }),
  }
}

/**
 * Sanitize before normal admission, observer publication, and persistence.
 *
 * Every session is sanitized, subagents included: reasoning kept in a subagent
 * log is re-sent on its later steps and across its own turn boundaries, where
 * the provider no longer restores it, so it is billed on every subsequent round
 * for the rest of that child's life. The parent never receives child reasoning
 * in any case, so retaining it buys nothing.
 * @param session - session whose `append` is wrapped.
 * @returns disposer restoring the unpatched method.
 */
export function interceptAppend(session: Session): () => void {
  const original = session.append
  let active = true
  // The reflection boundary preserves the generic append method and its overloads.
  const append = function (this: Session, ...args: unknown[]): unknown {
    if (!active) return Reflect.apply(original, this, args)
    const [type, data, ...opts] = args
    if (typeof type === 'string') return Reflect.apply(original, this, [type, cleanData(type, data), ...opts])
    // Retain compatibility with batch-capable builds; pinned rc.1 may reject batches.
    if (Array.isArray(type)) return Reflect.apply(original, this, [type.map(input => ({ ...input, data: cleanData(input.type, input.data) })), ...opts])
    return Reflect.apply(original, this, args)
  } as Session['append']
  const undo = patchMethod(session, 'append', append)
  return () => { active = false; undo() }
}
