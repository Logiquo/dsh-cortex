import type { Session } from '@deepseek-ai/dsh-session'
import type { TokenMeter } from '@deepseek-ai/dsh-token-meter'
import { estimateToolsTokens } from '@deepseek-ai/dsh-token-meter/estimate'
import { patchMethod } from './patch.js'

/** Preserve clean route prices, but never anchor retained context to billed output reasoning. */
export function interceptMeter(meter: TokenMeter, affected: (session: Session) => boolean): () => void {
  const measure = meter.measure
  let active = true
  const undo = patchMethod(meter, 'measure', function (this: TokenMeter, session, header) {
    const raw = measure.call(this, session, header)
    if (!active || !affected(session)) return raw
    const totalTokens = raw.surfaceTokens + estimateToolsTokens(header ?? session.requestHeader())
    return Object.freeze({
      ...raw, totalTokens,
      baseline: Object.freeze({ kind: 'estimated' as const, tokens: totalTokens }), surfaceDeltaTokens: 0,
    })
  })
  return () => { active = false; undo() }
}
