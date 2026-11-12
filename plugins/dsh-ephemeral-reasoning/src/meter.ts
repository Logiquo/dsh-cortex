import type { TokenUsage } from '@deepseek-ai/dsh-llm'
import { SessionSeq, deriveEventMessage, headerEquals } from '@deepseek-ai/dsh-session'
import type { EpochHeader, Session } from '@deepseek-ai/dsh-session'
import type { TokenMeter } from '@deepseek-ai/dsh-token-meter'
import { estimateMessage, estimateToolsTokens } from '@deepseek-ai/dsh-token-meter/estimate'
import { patchMethod } from './patch.js'

/**
 * Context pressure as the next request will bill it:
 *
 *     total = billed total + appended content − dropped reasoning
 *
 * **Billed total — EXACT.** The provider's count of the newest settled call:
 * `inputTokens + cacheReadTokens + cacheWriteTokens + outputTokens` of the
 * usage record the adapter stored on `assistant/message`. It comes from the
 * provider, not from us, so everything the provider accepted up to that call is
 * counted exactly, including reasoning this plugin removed from storage.
 *
 * **Appended content — ESTIMATE.** Price of the surface nodes added after the
 * anchor: tool results, user text, and any system or developer update. No server has seen
 * these tokens, so they are priced with dsh's fixed heuristic —
 * `ceil(chars / 4) + 4` per block plus `4` per message, route-priced for images
 * and files. Computed from the host meter's own `nodes`, so image and file
 * prices are the ones the route actually declares.
 *
 * **Dropped reasoning — ESTIMATE, zero unless the turn closed.** Reasoning the
 * provider stops resending once a `user/message` arrives. Per settled message it is
 * `outputTokens − heuristic(visible content)`: the exact output count minus a
 * price of the small part that will be resent, so estimator error lands on the
 * visible text rather than on the reasoning. A message sanitation emptied
 * derives to nothing, so its whole output counts as dropped.
 *
 * While the turn is open the provider resends its reasoning, so nothing is
 * dropped and the billed total alone is authoritative. Every settlement
 * refreshes it, so no term accumulates across the session.
 *
 * Billing is untouched: the recorded usage stays in the log and in every
 * projection that reads it, and only the derived pressure total changes.
 *
 * @module dsh-ephemeral-reasoning/meter
 */

/** The newest provider-recorded total that still describes the session's history. */
interface Anchor {
  /** Sequence of the `assistant/message` that reported this usage. */
  readonly seq: SessionSeq
  readonly usage: Readonly<TokenUsage>
  /** Whether a `user/message` has been appended after the anchor. */
  readonly closed: boolean
  /** Reasoning of the anchor's turn, still described by `usage` only while the turn is open. */
  readonly turnReasoning: number
}

interface Settled {
  readonly seq: SessionSeq
  readonly usage: Readonly<TokenUsage>
  readonly header: EpochHeader
  readonly turnReasoning: number
}

interface State {
  /** Next unread durable sequence number. */
  readonly consumed: number
  /** Envelope in force for the current request series, if one has been logged. */
  readonly header: EpochHeader | undefined
  /** Reasoning generated since the last `user/message`, over settled messages. */
  readonly chainReasoning: number
  readonly settled: Settled | undefined
  /** Whether a `user/message` landed after `settled`. */
  readonly closed: boolean
}

const states = new WeakMap<Session, State>()

/**
 * Sum disjoint provider buckets exactly as the host meter does, so a recorded
 * prompt plus response is never double-counted through its reasoning output.
 * @param usage - provider usage record for one settled call.
 * @returns the billed total for that call.
 */
function usageTotal(usage: TokenUsage): number {
  return usage.inputTokens + (usage.cacheReadTokens ?? 0) + (usage.cacheWriteTokens ?? 0) + usage.outputTokens
}

/**
 * Reasoning a settlement produced, as recorded output minus the heuristic price
 * of what the provider will resend as ordinary content.
 *
 * An assistant message that sanitation emptied derives to no message at all, so
 * its visible price is zero and its whole output is reasoning. That is the same
 * rule that removes it from the next request.
 * @param event - the settled `assistant/message`.
 * @param usage - its provider usage record.
 * @returns non-negative reasoning tokens for that message.
 */
function reasoningTokens(event: Parameters<typeof deriveEventMessage>[0], usage: TokenUsage): number {
  const message = deriveEventMessage(event)
  const visible = message === null ? 0 : estimateMessage(message)
  return Math.max(0, usage.outputTokens - visible)
}

/**
 * Compare two envelopes without letting an uncomparable pair keep an anchor.
 * @param billed - envelope the recorded total was billed under.
 * @param next - envelope now in force.
 * @returns whether both describe the same billable request shape.
 */
function sameEnvelope(billed: EpochHeader, next: EpochHeader): boolean {
  try {
    return headerEquals(billed, next)
  } catch {
    return false
  }
}

/**
 * Replay one session's unread durable tail for its newest usable anchor.
 *
 * A settlement's `usage` is the provider's own count of the prompt it received
 * and the response it produced, so it is exact for everything sent up to that
 * call. Two later events end that guarantee: a request envelope with different
 * content means the next call bills a different route or tool set, and a
 * replacing surface op means compaction rewrote the billed history. Re-logging
 * an identical envelope does not end it — the same messages are still billed.
 *
 * While the turn is open the provider resends its reasoning, so the billed
 * total stays authoritative. A `user/message` closes the turn: from then on the
 * reasoning counted in `chainReasoning` is no longer resent, and the measurement
 * subtracts it.
 * @param state - fold state before the unread tail.
 * @param session - session whose durable tail is read.
 * @returns fold state advanced through the current tail.
 */
function fold(state: State, session: Session): State {
  let consumed = state.consumed
  let header = state.header
  let chainReasoning = state.chainReasoning
  let settled = state.settled
  let closed = state.closed
  while (consumed < session.seq) {
    const event = session.eventAt(SessionSeq(consumed))
    consumed += 1
    if (event === undefined) continue
    // A replacing node rewrote the billed history; no earlier total describes it.
    if (event.surfaceOp !== undefined && event.surfaceOp !== 'append') settled = undefined
    switch (event.type) {
      case 'request/header': {
        const next = event.data.header
        header = next
        if (settled !== undefined && !sameEnvelope(settled.header, next)) settled = undefined
        break
      }
      case 'user/message':
        chainReasoning = 0
        if (settled !== undefined) closed = true
        break
      case 'assistant/message': {
        const usage = event.data.usage
        if (usage === undefined) break
        chainReasoning += reasoningTokens(event, usage)
        if (header !== undefined) {
          settled = { seq: event.seq, usage, header, turnReasoning: chainReasoning }
          closed = false
        }
        break
      }
      default:
        break
    }
  }
  return { consumed, header, chainReasoning, settled, closed }
}

/**
 * Read the session's newest usable anchor without mutating it.
 *
 * Reads only durable events, so a resumed session rebuilds the same anchor from
 * its log. Cost is O(events appended since the previous call).
 * @param session - main-agent session to fold.
 * @returns the newest usable anchor, or undefined before any settled call.
 */
function latestAnchor(session: Session): Anchor | undefined {
  const previous = states.get(session) ?? {
    consumed: 0,
    header: undefined,
    chainReasoning: 0,
    settled: undefined,
    closed: false,
  }
  const next = fold(previous, session)
  states.set(session, next)
  if (next.settled === undefined) return undefined
  return { ...next.settled, closed: next.closed }
}

/**
 * Report the billed total, plus appended content, minus dropped reasoning.
 *
 * The host meter already accepts a recorded usage anchor, but only when that
 * total reaches its fixed chars-per-token price of the same history; that
 * heuristic runs high on prose, so a real count is routinely rejected and the
 * whole history is repriced instead. Compaction re-logs an identical envelope,
 * which the host guard cannot tell from an unchanged one, so it would also
 * reuse a pre-compaction total against a much smaller surface. Without a usable
 * anchor this therefore prices the current surface instead of passing the host
 * result through.
 * @param meter - host token meter instance.
 * @param affected - whether this session's measurements are owned by the plugin.
 * @returns disposer restoring the unpatched method.
 */
export function interceptMeter(meter: TokenMeter, affected: (session: Session) => boolean): () => void {
  const measure = meter.measure
  let active = true
  const undo = patchMethod(meter, 'measure', function (this: TokenMeter, session, header) {
    const raw = measure.call(this, session, header)
    if (!active || !affected(session)) return raw
    const anchor = latestAnchor(session)
    if (anchor === undefined) {
      const totalTokens = raw.surfaceTokens + estimateToolsTokens(header ?? session.requestHeader())
      return Object.freeze({
        ...raw,
        baseline: Object.freeze({ kind: 'estimated' as const, tokens: totalTokens }),
        surfaceDeltaTokens: 0,
        totalTokens,
      })
    }
    // Every node at or before the anchor was inside the prompt the provider billed.
    let appended = 0
    for (const node of raw.nodes) if (node.seq > anchor.seq) appended += node.tokens
    const billed = usageTotal(anchor.usage)
    const dropped = anchor.closed ? anchor.turnReasoning : 0
    return Object.freeze({
      ...raw,
      baseline: Object.freeze({ kind: 'usage' as const, tokens: billed, usage: anchor.usage }),
      surfaceDeltaTokens: appended - dropped,
      totalTokens: Math.max(0, billed + appended - dropped),
    })
  })
  return () => { active = false; undo() }
}

/* -------------------------------------------------------------------------- *
 * The client-visible occupancy unit
 *
 * `measure()` is host-side. The composer ring and the stats panel read the
 * `contextPressure` projection, which the host computes from prompt-side usage
 * plus surface movement: it never includes reasoning the provider re-sends, so
 * mid-turn it reports less than the next request will bill.
 *
 * The registry cannot be overridden by registering the same key — a second
 * registration only increments the existing reference count and the first
 * definition keeps driving. This replaces the live definition in place and
 * drops its memoized cells so every session refolds with the corrected fold.
 * -------------------------------------------------------------------------- */

/** Key of the host occupancy unit this plugin corrects. */
const PRESSURE_KEY = 'contextPressure'
/** `stateVersion` the override was validated against. */
const HOST_PRESSURE_VERSION = 5
/** The override's own version, so a stale persisted row refolds instead of seeding. */
const PRESSURE_VERSION = 6

/** One pending surface replacement the host priced ahead of its shadow node. */
interface PressureClaim {
  readonly start: SessionSeq
  readonly end: SessionSeq
  readonly tokens: number
}

/** State of the corrected occupancy unit: the host's fields plus the anchor facts. */
interface PressureState {
  readonly contextWindow?: number
  readonly pressureTokens?: number
  readonly surfaceTokens: number
  readonly sampledSurfaceTokens?: number
  readonly claim?: PressureClaim
  readonly envelope?: EpochHeader
  readonly billedTotal?: number
  readonly appendedTokens: number
  /** Reasoning of the turn now open; the provider still resends it. */
  readonly turnReasoning: number
  /** Reasoning of the turn that closed; the provider no longer resends it. */
  readonly droppedReasoning: number
}

/** The registry's stored definition, as far as this override touches it. */
interface PressureUnit {
  def: {
    stateVersion: number
    stateSchema: { parse(value: unknown): PressureState }
    init(header: { origin?: string }, inheritedEventCount: number): PressureState
    apply(state: PressureState, event: never): PressureState
    wire?: {
      viewSchema: { parse(value: unknown): unknown }
      view(state: PressureState): Record<string, unknown>
      summary?: boolean
    }
  }
  cells: WeakMap<object, unknown>
}

/**
 * Validate one persisted occupancy row.
 *
 * Only this plugin's own checkpoints reach it, because the version above
 * rejects older rows; the check exists so a corrupt row refolds instead of
 * poisoning a fold.
 * @param value - persisted state from a checkpoint row.
 * @returns the normalized state.
 */
function parsePressureState(value: unknown): PressureState {
  if (typeof value !== 'object' || value === null) throw new Error('contextPressure state must be an object')
  const source = value as Record<string, unknown>
  const count = (key: string, required: boolean): number | undefined => {
    const raw = source[key]
    if (raw === undefined) {
      if (required) throw new Error(`contextPressure state is missing "${key}"`)
      return undefined
    }
    if (typeof raw !== 'number' || !Number.isSafeInteger(raw) || raw < 0) {
      throw new Error(`contextPressure state field "${key}" must be a non-negative safe integer`)
    }
    return raw
  }
  const contextWindow = count('contextWindow', false)
  const pressureTokens = count('pressureTokens', false)
  const sampledSurfaceTokens = count('sampledSurfaceTokens', false)
  const billedTotal = count('billedTotal', false)
  const claim = source['claim']
  const envelope = source['envelope']
  return {
    ...contextWindow === undefined ? {} : { contextWindow },
    ...pressureTokens === undefined ? {} : { pressureTokens },
    surfaceTokens: count('surfaceTokens', true)!,
    ...sampledSurfaceTokens === undefined ? {} : { sampledSurfaceTokens },
    ...claim === undefined ? {} : { claim: claim as PressureClaim },
    ...envelope === undefined ? {} : { envelope: envelope as EpochHeader },
    ...billedTotal === undefined ? {} : { billedTotal },
    appendedTokens: count('appendedTokens', true)!,
    turnReasoning: count('turnReasoning', true)!,
    droppedReasoning: count('droppedReasoning', true)!,
  }
}

/** Drop the anchor and everything counted against it. */
function clearPressureAnchor(state: PressureState): PressureState {
  const { billedTotal: _billed, envelope: _envelope, ...rest } = state
  return { ...rest, appendedTokens: 0, turnReasoning: 0, droppedReasoning: 0 }
}

/**
 * Install the corrected occupancy unit.
 * @param registry - the host `sessionProjections` service.
 * @returns disposer restoring the host definition.
 * @throws when the unit is absent or is not the version this override was built for.
 */
export function interceptPressure(registry: unknown): () => void {
  const registrations = (registry as { registrations?: Map<string, PressureUnit> }).registrations
  const registration = registrations?.get(PRESSURE_KEY)
  if (registrations === undefined || registration === undefined) {
    throw new Error(`Ephemeral reasoning: the ${PRESSURE_KEY} projection is not registered; refusing to install an unverified override`)
  }
  const original = registration.def
  const wire = original.wire
  if (original.stateVersion !== HOST_PRESSURE_VERSION || wire === undefined) {
    throw new Error(`Ephemeral reasoning: unrecognized ${PRESSURE_KEY} unit (stateVersion ${original.stateVersion}); refusing to patch an unverified implementation (tested dsh 0.2.2-rc.1)`)
  }
  const hostApply = original.apply as unknown as (state: PressureState, event: never) => PressureState
  const hostInit = original.init
  const hostView = wire.view
  const apply = (state: PressureState, event: never): PressureState => {
    const base = hostApply(state, event) as PressureState
    const source = event as unknown as { type: string; surfaceOp?: unknown; data: Record<string, unknown> }
    let next = base
    const appendedTokens = state.appendedTokens + (base.surfaceTokens - state.surfaceTokens)
    if (appendedTokens !== state.appendedTokens) next = { ...next, appendedTokens }
    const op = 'surfaceOp' in source ? source.surfaceOp : undefined
    // A replacing node — compaction, prune — rewrote the surface the total described.
    if (op !== undefined && op !== 'append') next = clearPressureAnchor(next)
    switch (source.type) {
      case 'request/header': {
        const envelope = source.data['header'] as EpochHeader
        if (next.envelope !== undefined && !sameEnvelope(next.envelope, envelope)) next = clearPressureAnchor(next)
        next = { ...next, envelope }
        break
      }
      case 'user/message': {
        // The turn that just ended is no longer resent; carry its reasoning over.
        next = { ...next, droppedReasoning: next.turnReasoning, turnReasoning: 0 }
        break
      }
      case 'assistant/message': {
        const usage = source.data['usage'] as TokenUsage | undefined
        if (usage === undefined) break
        next = {
          ...next,
          turnReasoning: next.turnReasoning + reasoningTokens(source as never, usage),
          droppedReasoning: 0,
          billedTotal: usageTotal(usage),
          appendedTokens: 0,
        }
        break
      }
      default:
        break
    }
    return next
  }
  const definition = {
    key: PRESSURE_KEY,
    stateVersion: PRESSURE_VERSION,
    stateSchema: { parse: parsePressureState },
    init: (header: { origin?: string }, inheritedEventCount: number): PressureState => ({
      ...hostInit(header as never, inheritedEventCount),
      appendedTokens: 0,
      turnReasoning: 0,
      droppedReasoning: 0,
    }),
    apply: apply as never,
    wire: {
      viewSchema: wire.viewSchema,
      summary: wire.summary,
      view: (state: PressureState): Record<string, unknown> => {
        const host = hostView(state)
        if (state.billedTotal === undefined) return host
        return { ...host, projectedTokens: Math.max(0, state.billedTotal + state.appendedTokens - state.droppedReasoning) }
      },
    },
  }
  const installed = definition as unknown as PressureUnit['def']
  registration.def = installed
  registration.cells = new WeakMap()
  return () => {
    // A definition installed after this one owns the unit now; leave it in place.
    if (registration.def !== installed) return
    registration.def = original
    registration.cells = new WeakMap()
  }
}
