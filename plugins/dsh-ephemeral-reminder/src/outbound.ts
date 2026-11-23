import type { RequestMessage } from '@deepseek-ai/dsh-llm'

export const DEFAULT_REMINDER_TEXT = `<system-reminder>
Direct and signal-rich only. No introductory filler, no "not X, but Y" contrast formulas, no repeating prior caveats.
</system-reminder>`

function findLastUserIndex(messages: readonly RequestMessage[]): number {
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i]?.role === 'user') return i
  }
  return -1
}

/**
 * Append a text block containing `reminderText` to the latest user message only.
 * Historical user messages are left untouched so the reminder is sent only for the active tail turn.
 */
export function appendReminderToLastUser(
  messages: readonly RequestMessage[],
  reminderText: string,
  targetIndex?: number,
): readonly RequestMessage[] {
  if (!reminderText) return messages
  const idx = targetIndex ?? findLastUserIndex(messages)
  if (idx < 0 || idx >= messages.length) return messages

  const target = messages[idx]
  if (!target || target.role !== 'user') return messages

  const alreadyPresent = target.content.some(
    block => block.type === 'text' && block.text === reminderText,
  )
  if (alreadyPresent) return messages

  const next = [...messages]
  next[idx] = {
    ...target,
    content: [...target.content, { type: 'text', text: reminderText }],
  }
  return next
}
