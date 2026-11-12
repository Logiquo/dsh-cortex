import type { Session, SessionSeq } from '@deepseek-ai/dsh-session'
import type { TokenMeasurement } from '@deepseek-ai/dsh-token-meter'
import { toolPairingBalancedBefore, toolPairingBalancedAfter } from '@deepseek-ai/dsh-compaction'
import type { Config } from './config.js'

/** Select the longest balanced recent suffix that fits the upper bound. */
export function selectRange(session: Session, measurement: TokenMeasurement, retainTokens: number, hasSystemHead: boolean) {
  const nodes = session.surface.nodes
  if (nodes.length !== measurement.nodes.length || nodes.some((seq, index) => seq !== measurement.nodes[index]?.seq)) {
    throw new Error('Context offload: token meter and session surface differ')
  }
  const first = hasSystemHead ? 1 : 0
  if (nodes.length <= first || !toolPairingBalancedAfter(session, nodes.at(-1)!)) return null
  let keep = nodes.length
  let retained = 0
  for (let index = nodes.length - 1; index >= first; index--) {
    retained += measurement.nodes[index]!.tokens
    if (retained > retainTokens) break
    if (toolPairingBalancedBefore(session, nodes[index]!)) keep = index
  }
  if (keep <= first) return null
  return { start: nodes[first] as SessionSeq, end: nodes[keep - 1] as SessionSeq }
}
export function budgets(config: Config) {
  const contextWindow = config['context-limit']
  if (!Number.isSafeInteger(contextWindow) || contextWindow <= 0) throw new Error('context-limit must be a positive integer')
  const triggerTokens = Math.floor(contextWindow * 0.75)
  return {
    triggerTokens,
    briefMaxTokens: Math.max(Math.floor(triggerTokens * 0.10), 2048),
    retainRecentTokens: Math.max(Math.floor(triggerTokens * 0.20), 4096),
  }
}
