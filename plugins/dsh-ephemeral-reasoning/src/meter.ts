import type { TokenUsage } from '@deepseek-ai/dsh-llm'
import { SessionSeq, canonicalHeader, deriveEventMessage, headerEquals } from '@deepseek-ai/dsh-session'
import type { EpochHeader, Session } from '@deepseek-ai/dsh-session'
import type { TokenMeter } from '@deepseek-ai/dsh-token-meter'
import { estimateContent, estimateMessage, estimateToolsTokens } from '@deepseek-ai/dsh-token-meter/estimate'
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
 * counted exactly.
 *
 * **Appended content — ESTIMATE.** Price of the surface nodes added after the
 * anchor: tool results, user text, and any system or developer update. No server
 * has seen these tokens, so they are priced with dsh's fixed heuristic —
 * `ceil(chars / 4) + 4` per block plus `4` per message, route-priced for images
 * and files. Reasoning on completed turns is excluded because outbound
 * stripping omits it before dispatch.
 *
 * **Dropped reasoning — ESTIMATE, zero unless the turn closed.** Reasoning the
 * provider stops resending once a `user/message` or `turn/end` closes the turn.
 * Per settled message it is the share of `outputTokens` that its reasoning
 * blocks accounted for.
 *
 * While the turn is open the provider keeps resending its reasoning, so nothing
 * is dropped and the billed total alone is authoritative.
 *
 * @module dsh-ephemeral-reasoning/meter
 */

/** The newest provider-recorded total that still describes the session's history. */
interface Anchor {
  /** Sequence of the `assistant/message` that reported this usage. */
  readonly seq: SessionSeq
  readonly usage: Readonly<TokenUsage>
  /** Envelope the recorded total was billed under. */
  readonly header: EpochHeader
  /** Whether a `user/message` or `turn/end` has been appended after the anchor. */
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
  /** Reasoning generated since the last `user/message` or `turn/end`, over settled messages. */
  readonly chainReasoning: number
  readonly settled: Settled | undefined
  /** Whether a `user/message` or `turn/end` landed after `settled`. */
  readonly closed: boolean
  /** Heuristic reasoning prices of `assistant/message` nodes in the currently open turn. */
  readonly openTurnNodes: ReadonlyMap<SessionSeq, number>
  /** Heuristic reasoning prices of `assistant/message` nodes in closed turns. */
  readonly closedReasoningBySeq: ReadonlyMap<SessionSeq, number>
}

const states = new WeakMap<Session, State>()

/**
 * Sum disjoint provider buckets exactly as the host meter does, so a recorded
 * prompt plus response is never double-counted through its reasoning output.
 */
function usageTotal(usage: TokenUsage): number {
  return usage.inputTokens + (usage.cacheReadTokens ?? 0) + (usage.cacheWriteTokens ?? 0) + usage.outputTokens
}

/**
 * Split the provider's exact output count between reasoning and visible text.
 */
function splitReasoning(outputTokens: number, reasoning: number, visible: number): number {
  const total = reasoning + visible
  if (outputTokens <= 0 || reasoning <= 0) return 0
  if (total <= 0) return outputTokens
  return Math.min(outputTokens, Math.round(outputTokens * reasoning / total))
}

/**
 * Heuristic token price of the reasoning blocks on an `assistant/message` event.
 */
function heuristicReasoningTokens(event: { readonly data?: unknown }): number {
  const data = event.data as { readonly message?: { readonly content?: unknown } } | undefined
  const content = data?.message?.content
  if (!Array.isArray(content)) return 0
  const reasoning = content.filter(block => typeof block === 'object' && block !== null && (block as { type?: unknown }).type === 'reasoning')
  return reasoning.length === 0 ? 0 : estimateContent(reasoning as never)
}

/**
 * Reasoning a settlement produced, read directly from the stored message content
 * when the adapter does not report `usage.reasoningTokens`.
 */
function reasoningTokens(event: Parameters<typeof deriveEventMessage>[0], usage: TokenUsage): number {
  if (usage.reasoningTokens !== undefined && usage.reasoningTokens > 0) return usage.reasoningTokens
  const message = deriveEventMessage(event)
  if (message === null) return usage.outputTokens > 0 ? usage.outputTokens : 0
  const reasoning = message.content.filter(block => block.type === 'reasoning')
  const reasoningPrice = reasoning.length === 0 ? 0 : estimateContent(reasoning)
  const visibleBlocks = message.content.filter(block => block.type !== 'reasoning')
  const visiblePrice = visibleBlocks.length === 0 ? 0 : estimateMessage({ ...message, content: visibleBlocks })
  return splitReasoning(usage.outputTokens, reasoningPrice, visiblePrice)
}

/** Compare two envelopes without letting an uncomparable pair keep an anchor. */
function sameEnvelope(billed: EpochHeader, next: EpochHeader): boolean {
  try {
    return headerEquals(billed, next)
  } catch {
    return false
  }
}

/**
 * Replay one session's unread durable tail for its newest usable anchor and
 * per-node closed-turn reasoning prices.
 */
function fold(state: State, session: Session): State {
  if (state.consumed >= session.seq) return state
  let consumed = state.consumed
  let header = state.header
  let chainReasoning = state.chainReasoning
  let settled = state.settled
  let closed = state.closed
  const openTurnNodes = new Map(state.openTurnNodes)
  const closedReasoningBySeq = new Map(state.closedReasoningBySeq)

  while (consumed < session.seq) {
    const event = session.eventAt(SessionSeq(consumed))
    consumed += 1
    if (event === undefined) continue
    if (event.surfaceOp !== undefined && event.surfaceOp !== 'append') settled = undefined
    switch (event.type) {
      case 'request/header': {
        const next = event.data.header
        header = next
        if (settled !== undefined && !sameEnvelope(settled.header, next)) settled = undefined
        break
      }
      case 'user/message':
      case 'turn/end':
        chainReasoning = 0
        if (settled !== undefined) closed = true
        for (const [seq, tokens] of openTurnNodes) closedReasoningBySeq.set(seq, tokens)
        openTurnNodes.clear()
        break
      case 'assistant/message': {
        const hReasoning = heuristicReasoningTokens(event)
        if (hReasoning > 0) openTurnNodes.set(event.seq, hReasoning)
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
  return { consumed, header, chainReasoning, settled, closed, openTurnNodes, closedReasoningBySeq }
}

/**
 * Read the session's newest usable anchor and closed-turn reasoning map.
 */
function latestState(session: Session): { anchor: Anchor | undefined; closedReasoningBySeq: ReadonlyMap<SessionSeq, number> } {
  const previous = states.get(session) ?? {
    consumed: 0,
    header: undefined,
    chainReasoning: 0,
    settled: undefined,
    closed: false,
    openTurnNodes: new Map<SessionSeq, number>(),
    closedReasoningBySeq: new Map<SessionSeq, number>(),
  }
  const next = fold(previous, session)
  states.set(session, next)
  return {
    anchor: next.settled === undefined ? undefined : { ...next.settled, closed: next.closed },
    closedReasoningBySeq: next.closedReasoningBySeq,
  }
}

/**
 * Report the billed total, plus appended content (excluding closed-turn reasoning),
 * minus dropped reasoning, and adjust `nodes` / `surfaceTokens` so persisted
 * closed-turn reasoning does not inflate downstream compaction budgets.
 */
export function interceptMeter(meter: TokenMeter): () => void {
  const measure = meter.measure
  let active = true
  const undo = patchMethod(meter, 'measure', function (this: TokenMeter, session, header) {
    const raw = measure.call(this, session, header)
    if (!active) return raw
    const { anchor, closedReasoningBySeq } = latestState(session)
    let nodesChanged = false
    const nodes = raw.nodes.map((node) => {
      const removed = closedReasoningBySeq.get(node.seq) ?? 0
      if (removed <= 0) return node
      nodesChanged = true
      return Object.freeze({
        ...node,
        tokens: Math.max(0, node.tokens - removed),
        heuristicTokens: Math.max(0, node.heuristicTokens - removed),
      })
    })
    const adjustedNodes = nodesChanged ? Object.freeze(nodes) : raw.nodes
    const surfaceTokens = nodesChanged ? adjustedNodes.reduce((sum, node) => sum + node.tokens, 0) : raw.surfaceTokens
    const effective = header === undefined ? session.requestHeader() : canonicalHeader(header)

    if (anchor === undefined || (effective !== undefined && !sameEnvelope(anchor.header, effective))) {
      const totalTokens = surfaceTokens + estimateToolsTokens(effective ?? session.requestHeader())
      return Object.freeze({
        ...raw,
        nodes: adjustedNodes,
        surfaceTokens,
        baseline: Object.freeze({ kind: 'estimated' as const, tokens: totalTokens }),
        surfaceDeltaTokens: 0,
        totalTokens,
      })
    }
    let appended = 0
    for (const node of adjustedNodes) if (node.seq > anchor.seq) appended += node.tokens
    const billed = usageTotal(anchor.usage)
    const dropped = anchor.closed ? anchor.turnReasoning : 0
    return Object.freeze({
      ...raw,
      nodes: adjustedNodes,
      surfaceTokens,
      baseline: Object.freeze({ kind: 'usage' as const, tokens: billed, usage: anchor.usage }),
      surfaceDeltaTokens: appended - dropped,
      totalTokens: Math.max(0, billed + appended - dropped),
    })
  })
  return () => { active = false; undo() }
}

/* -------------------------------------------------------------------------- *
 * The client-visible occupancy unit
 * -------------------------------------------------------------------------- */

const PRESSURE_KEY = 'contextPressure'
const HOST_PRESSURE_VERSION = 5
const PRESSURE_VERSION = 9

interface PressureClaim {
  readonly start: SessionSeq
  readonly end: SessionSeq
  readonly tokens: number
}

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
  /** Heuristic reasoning of the turn now open (included in host surfaceTokens). */
  readonly openHeuristicReasoning: number
  /** Unanchored heuristic reasoning added since the latest usage anchor. */
  readonly unanchoredOpenReasoning: number
  /** Heuristic reasoning of closed turns still retained in host surfaceTokens. */
  readonly closedHeuristicReasoning: number
}

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
  refs?: number
}

function parsePressureState(value: unknown): PressureState {
  if (typeof value !== 'object' || value === null) throw new Error('contextPressure state must be an object')
  const source = value as Record<string, unknown>
  const isCount = (raw: unknown): raw is number => typeof raw === 'number' && Number.isSafeInteger(raw) && raw >= 0
  const count = (key: string, required: boolean): number | undefined => {
    const raw = source[key]
    if (raw === undefined) {
      if (required) throw new Error(`contextPressure state is missing "${key}"`)
      return undefined
    }
    if (!isCount(raw)) {
      throw new Error(`contextPressure state field "${key}" must be a non-negative safe integer`)
    }
    return raw
  }
  const contextWindow = count('contextWindow', false)
  const pressureTokens = count('pressureTokens', false)
  const sampledSurfaceTokens = count('sampledSurfaceTokens', false)
  const billedTotal = count('billedTotal', false)
  const parseClaim = (raw: unknown): PressureClaim | undefined => {
    if (typeof raw !== 'object' || raw === null) return undefined
    const fields = raw as Record<string, unknown>
    const start = fields['start']
    const end = fields['end']
    const tokens = fields['tokens']
    if (!isCount(start) || !isCount(end) || !isCount(tokens)) return undefined
    return { start: SessionSeq(start), end: SessionSeq(end), tokens }
  }
  const claim = parseClaim(source['claim'])
  const stored = source['envelope']
  const envelope = typeof stored === 'object' && stored !== null && !Array.isArray(stored)
    ? stored as EpochHeader
    : undefined
  return {
    ...contextWindow === undefined ? {} : { contextWindow },
    ...pressureTokens === undefined ? {} : { pressureTokens },
    surfaceTokens: count('surfaceTokens', true)!,
    ...sampledSurfaceTokens === undefined ? {} : { sampledSurfaceTokens },
    ...claim === undefined ? {} : { claim },
    ...envelope === undefined ? {} : { envelope },
    ...envelope === undefined || billedTotal === undefined ? {} : { billedTotal },
    appendedTokens: count('appendedTokens', true)!,
    turnReasoning: count('turnReasoning', true)!,
    droppedReasoning: count('droppedReasoning', true)!,
    openHeuristicReasoning: count('openHeuristicReasoning', true)!,
    unanchoredOpenReasoning: count('unanchoredOpenReasoning', true)!,
    closedHeuristicReasoning: count('closedHeuristicReasoning', true)!,
  }
}

function clearPressureAnchor(state: PressureState): PressureState {
  const { billedTotal: _billed, ...rest } = state
  return { ...rest, appendedTokens: 0, unanchoredOpenReasoning: 0, closedHeuristicReasoning: 0 }
}

export function assertPressureCompatible(registry: unknown): void {
  pressureRegistration(registry)
}

function pressureRegistration(registry: unknown): { registration: PressureUnit; original: PressureUnit['def'] } {
  const registrations = (registry as { registrations?: Map<string, PressureUnit> }).registrations
  const registration = registrations?.get(PRESSURE_KEY)
  if (registrations === undefined || registration === undefined) {
    throw new Error(`Ephemeral reasoning: the ${PRESSURE_KEY} projection is not registered; refusing to install an unverified override`)
  }
  const original = registration.def
  if (registration.refs === 0) {
    throw new Error(`Ephemeral reasoning: the ${PRESSURE_KEY} projection is a released predecessor; refusing to patch a unit the live drive no longer serves (tested dsh 0.2.2-rc.5)`)
  }
  if (original.stateVersion !== HOST_PRESSURE_VERSION || original.wire === undefined) {
    throw new Error(`Ephemeral reasoning: unrecognized ${PRESSURE_KEY} unit (stateVersion ${original.stateVersion}); refusing to patch an unverified implementation (tested dsh 0.2.2-rc.5)`)
  }
  return { registration, original }
}

export function interceptPressure(registry: unknown): () => void {
  const { registration, original } = pressureRegistration(registry)
  const wire = original.wire!
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
    if (op !== undefined && op !== 'append') next = clearPressureAnchor(next)
    switch (source.type) {
      case 'request/header': {
        const envelope = source.data['header'] as EpochHeader
        if (next.envelope !== undefined && !sameEnvelope(next.envelope, envelope)) next = clearPressureAnchor(next)
        next = { ...next, envelope }
        break
      }
      case 'user/message':
      case 'turn/end': {
        next = {
          ...next,
          droppedReasoning: next.droppedReasoning + next.turnReasoning,
          turnReasoning: 0,
          closedHeuristicReasoning: next.closedHeuristicReasoning + next.openHeuristicReasoning,
          openHeuristicReasoning: 0,
          appendedTokens: Math.max(0, next.appendedTokens - state.unanchoredOpenReasoning),
          unanchoredOpenReasoning: 0,
        }
        break
      }
      case 'assistant/message': {
        const hReasoning = heuristicReasoningTokens(source)
        const usage = source.data['usage'] as TokenUsage | undefined
        if (usage === undefined) {
          next = {
            ...next,
            openHeuristicReasoning: next.openHeuristicReasoning + hReasoning,
            unanchoredOpenReasoning: next.unanchoredOpenReasoning + hReasoning,
          }
          break
        }
        next = {
          ...next,
          turnReasoning: next.turnReasoning + reasoningTokens(source as never, usage),
          droppedReasoning: 0,
          ...next.envelope === undefined ? {} : { billedTotal: usageTotal(usage) },
          appendedTokens: 0,
          openHeuristicReasoning: next.openHeuristicReasoning + hReasoning,
          unanchoredOpenReasoning: 0,
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
      openHeuristicReasoning: 0,
      unanchoredOpenReasoning: 0,
      closedHeuristicReasoning: 0,
    }),
    apply: apply as never,
    wire: {
      viewSchema: wire.viewSchema,
      summary: wire.summary,
      view: (state: PressureState): Record<string, unknown> => {
        const host = hostView(state)
        if (state.billedTotal !== undefined) {
          return { ...host, projectedTokens: Math.max(0, state.billedTotal + state.appendedTokens - state.droppedReasoning) }
        }
        return {
          ...host,
          projectedTokens: Math.max(0, state.surfaceTokens - state.closedHeuristicReasoning + estimateToolsTokens(state.envelope)),
        }
      },
    },
  }
  const installed = definition as unknown as PressureUnit['def']
  registration.def = installed
  registration.cells = new WeakMap()
  return () => {
    if (registration.def !== installed) return
    registration.def = original
    registration.cells = new WeakMap()
  }
}
