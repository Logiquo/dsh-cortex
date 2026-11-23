import { freezeMessage } from '@deepseek-ai/dsh-llm'
import type { RequestMessage } from '@deepseek-ai/dsh-llm'

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
