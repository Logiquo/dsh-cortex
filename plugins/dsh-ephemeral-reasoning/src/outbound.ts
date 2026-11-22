import { freezeMessage } from '@deepseek-ai/dsh-llm'
import type LlmRuntime from '@deepseek-ai/dsh-llm'
import type { GenerateOptions, RequestMessage } from '@deepseek-ai/dsh-llm'
import { patchMethod } from './patch.js'

/**
 * Strip reasoning blocks and replay state from assistant messages belonging to
 * completed turns (strictly before the latest user message). Messages in the
 * currently open turn are preserved verbatim so within-turn tool-call chains
 * retain their reasoning continuity.
 * @param messages - outbound conversation messages.
 * @returns messages with closed-turn reasoning removed, preserving original
 *   references when nothing needed stripping.
 */
export function stripClosedTurnReasoning(messages: readonly RequestMessage[]): RequestMessage[] {
  let lastUserIndex = -1
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i]?.role === 'user') {
      lastUserIndex = i
      break
    }
  }
  if (lastUserIndex <= 0) return messages as RequestMessage[]

  let changed = false
  const next = messages.map((message, index) => {
    if (index >= lastUserIndex || message.role !== 'assistant') return message
    if (!message.content.some(block => block.type === 'reasoning')) return message
    changed = true
    const content = message.content.filter(block => block.type !== 'reasoning')
    const { replayState: _state, ...source } = message.source
    return freezeMessage({ ...message, source, content })
  })
  return changed ? next : (messages as RequestMessage[])
}

/**
 * Apply {@link stripClosedTurnReasoning} to an outbound `GenerateOptions` object,
 * preserving frozen semantics when the input options are frozen.
 */
export function stripClosedTurnOptions(options: GenerateOptions): GenerateOptions {
  const messages = stripClosedTurnReasoning(options.messages)
  if (messages === options.messages) return options
  const frozenMessages = (Object.isFrozen(options.messages) ? Object.freeze(messages) : messages) as RequestMessage[]
  const rewritten = { ...options, messages: frozenMessages }
  return Object.isFrozen(options) ? Object.freeze(rewritten) : rewritten
}

/**
 * Patch `LlmRuntime.forAdapter` to strip reasoning from completed turns right
 * before adapter dispatch, leaving persisted session logs untouched.
 * @param llm - host `llm` service instance.
 * @returns disposer restoring the unpatched method.
 */
export function interceptLlm(llm: LlmRuntime): () => void {
  const target = llm as unknown as {
    forAdapter(options: GenerateOptions, adapter: unknown): GenerateOptions
  }
  const original = target.forAdapter
  let active = true
  const undo = patchMethod(target, 'forAdapter', function (this: LlmRuntime, options: GenerateOptions, adapter: unknown) {
    const base = original.call(this, options, adapter)
    return active ? stripClosedTurnOptions(base) : base
  })
  return () => { active = false; undo() }
}
