import { freezeMessage } from '@deepseek-ai/dsh-llm'
import type { AssistantStreamRecord } from '@deepseek-ai/dsh-llm'
import type { Session, SessionEventMap } from '@deepseek-ai/dsh-session'
import { patchMethod } from './patch.js'

/** Filter packed records directly: no token-by-token expansion of discarded reasoning. */
export function cleanStream(stream: readonly AssistantStreamRecord[]): AssistantStreamRecord[] {
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
    if (chunk.type === 'finish') {
      const { replayState: _state, ...finish } = chunk
      chunk = finish
    }
    result.push({ ...record, chunk })
  }
  return result
}

function cleanData(type: string, data: unknown): unknown {
  if (type === 'assistant/attempt') {
    const attempt = data as SessionEventMap['assistant/attempt']
    return { ...attempt, stream: cleanStream(attempt.stream) }
  }
  if (type !== 'assistant/message') return data
  const response = data as SessionEventMap['assistant/message']
  const { replayState: _state, ...source } = response.message.source
  return {
    ...response, stream: cleanStream(response.stream),
    message: freezeMessage({ ...response.message, source, content: response.message.content.filter(block => block.type !== 'reasoning') }),
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
    if (Array.isArray(type)) return Reflect.apply(original, this, [type.map(input => ({ ...input, data: cleanData(input.type, input.data) }))])
    return Reflect.apply(original, this, args)
  } as Session['append']
  const undo = patchMethod(session, 'append', append)
  return () => { active = false; undo() }
}
