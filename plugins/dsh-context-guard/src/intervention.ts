import type { StoredCard } from './cards.js'

export const DEFAULT_DEGRADED_THRESHOLD = 0.5
export const DEFAULT_HIT_WINDOW_STEPS = 16
export const DEFAULT_HITS_TO_TRIGGER = 2
export const DEFAULT_COOLDOWN_STEPS = 8

export interface InterventionPolicy {
  degradedThreshold: number
  hitWindowSteps: number
  hitsToTrigger: number
  cooldownSteps: number
}

export interface InterventionEvaluation {
  shouldIntervene: boolean
  triggerCardIdx: number | null
  currentScore: number | null
  windowHits: number
  actionsSinceReset: number
}

/**
 * Evaluate whether the current session timeline should trigger a context-guard intervention.
 *
 * All step counts are measured in `[ACTION]` cards (`tool/call + tool/result` pairs).
 * Reset boundaries (each enforcing `cooldownSteps` quiet actions and clearing the hit window):
 *   1. Session start
 *   2. Most recent `/compact` (`compactBefore === true`)
 *   3. Most recent real `[USER]` message (`kind === 'constraint'`)
 *   4. Most recent injected intervention (`lastInterventionCardIdx`)
 */
export function evaluateIntervention(
  cards: readonly StoredCard[],
  lastInterventionCardIdx: number,
  policy: InterventionPolicy,
): InterventionEvaluation {
  // Find the most recent scored action card.
  let latestActionPos = -1
  for (let i = cards.length - 1; i >= 0; i--) {
    const c = cards[i]!
    if (c.kind === 'action' && c.pDegraded !== null) {
      latestActionPos = i
      break
    }
  }
  if (latestActionPos === -1) {
    return { shouldIntervene: false, triggerCardIdx: null, currentScore: null, windowHits: 0, actionsSinceReset: 0 }
  }

  const latestAction = cards[latestActionPos]!
  const currentScore = latestAction.pDegraded!

  // Find the latest reset boundary at or before latestActionPos.
  // Actions strictly after resetBoundaryPos belong to the current active window.
  let resetBoundaryPos = -1
  for (let i = latestActionPos; i >= 0; i--) {
    const c = cards[i]!
    if (c.cardIdx <= lastInterventionCardIdx) {
      resetBoundaryPos = i
      break
    }
    if (c.kind === 'constraint') {
      resetBoundaryPos = i
      break
    }
    if (c.compactBefore) {
      // The card itself is the first card AFTER compaction, so compaction occurred right before `i`.
      resetBoundaryPos = i - 1
      break
    }
  }

  // Collect all action cards strictly after resetBoundaryPos up to latestActionPos.
  const actionsAfterReset: StoredCard[] = []
  for (let i = resetBoundaryPos + 1; i <= latestActionPos; i++) {
    const c = cards[i]!
    if (c.kind === 'action') actionsAfterReset.push(c)
  }

  const actionsSinceReset = actionsAfterReset.length
  if (actionsSinceReset <= policy.cooldownSteps) {
    return {
      shouldIntervene: false,
      triggerCardIdx: null,
      currentScore,
      windowHits: 0,
      actionsSinceReset,
    }
  }

  const windowSlice = actionsAfterReset.slice(-policy.hitWindowSteps)
  let windowHits = 0
  for (const c of windowSlice) {
    if (c.pDegraded !== null && c.pDegraded >= policy.degradedThreshold) {
      windowHits++
    }
  }

  const shouldIntervene = currentScore >= policy.degradedThreshold && windowHits >= policy.hitsToTrigger
  return {
    shouldIntervene,
    triggerCardIdx: shouldIntervene ? latestAction.cardIdx : null,
    currentScore,
    windowHits,
    actionsSinceReset,
  }
}

function clipOneLine(text: string, maxChars: number): string {
  const clean = text.replace(/\s+/g, ' ').trim()
  return clean.length <= maxChars ? clean : `${clean.slice(0, maxChars - 1)}…`
}

/**
 * Render the zero-pronoun, structural meta-instruction intervention reminder.
 * Uses externalized recent-action trace + active user constraints + evidence-conditioned decision gate.
 */
export function buildInterventionPrompt(
  cards: readonly StoredCard[],
  options: { hasHistoryTools?: boolean } = {},
): string {
  const constraints = cards
    .filter(c => c.kind === 'constraint' && c.callText.trim().length > 0)
    .slice(-3)
  const recentActions = cards
    .filter(c => c.kind === 'action')
    .slice(-4)

  const lines: string[] = [
    '<system-reminder>',
    'Recent execution shows signs of drift or repetition. Pause for a brief evidence check before the next action.',
  ]

  if (constraints.length > 0) {
    lines.push('', 'Active user constraints:')
    for (const c of constraints) {
      lines.push(`- [id ${c.startEventId}] ${clipOneLine(c.callText, 260)}`)
    }
  }

  if (recentActions.length > 0) {
    lines.push('', 'Observed recent actions:')
    for (const a of recentActions) {
      const microClean = a.micro.replace(/^#\d+\s*/, '')
      lines.push(`- [ids ${a.startEventId}..${a.endEventId}] ${clipOneLine(microClean, 220)}`)
    }
  }

  const historyHint = options.hasHistoryTools
    ? ' (Use `history_search` / `history_read` if an earlier fact needs verification.)'
    : ''

  lines.push(
    '',
    'Decision gate:',
    '1. Identify the core assumption behind the observed actions and check whether direct evidence supports it against the constraints above.',
    `2. If evidence is missing, contradictory, or the same approach has already failed, do not apply another incremental tweak—switch to an independent approach or verify the assumption directly first.${historyHint}`,
    '3. If blocked by missing requirements or external limits, stop iterating and report the blocker clearly.',
    '',
    'Omit this reminder from user-facing output.',
    '</system-reminder>',
  )

  return lines.join('\n')
}
