import type { Context } from '@deepseek-ai/cordis'
import LlmRuntime from '@deepseek-ai/dsh-llm'
import type { GenerateOptions, Message, MessageId, RequestMessage, TokenUsage } from '@deepseek-ai/dsh-llm'
import { SessionSeq, canonicalHeader, deriveEventMessage, headerEquals } from '@deepseek-ai/dsh-session'
import type { EpochHeader, Session, SessionEvent, SessionId } from '@deepseek-ai/dsh-session'
import { TokenMeter } from '@deepseek-ai/dsh-token-meter'
import { estimateContent, estimateMessage, estimateToolsTokens } from '@deepseek-ai/dsh-token-meter/estimate'
import agentPackage from '@deepseek-ai/dsh-agent/package.json' with { type: 'json' }
import sessionPackage from '@deepseek-ai/dsh-session/package.json' with { type: 'json' }
import llmPackage from '@deepseek-ai/dsh-llm/package.json' with { type: 'json' }
import meterPackage from '@deepseek-ai/dsh-token-meter/package.json' with { type: 'json' }
import { patchMethod } from '../patch.js'
import { freezeModifiedMessages, type TransformPipeline } from '../pipeline.js'
import type { DurableMessageDiff } from '../types.js'
import type { HostDriver, ProbeResult } from './types.js'

const SUPPORTED_VERSIONS = ['0.2.2-rc.5'] as const
const PRESSURE_KEY = 'contextPressure'
const HOST_PRESSURE_VERSION = 5
const PRESSURE_VERSION = 10

function assertMethod(actual: unknown, expected: unknown, label: string): string | undefined {
  if (
    typeof actual !== 'function' ||
    typeof expected !== 'function' ||
    Function.prototype.toString.call(actual) !== Function.prototype.toString.call(expected)
  ) {
    return `unrecognized or already wrapped ${label}`
  }
  return undefined
}

function checkVersions(): string | undefined {
  const actual: Record<string, string> = {
    '@deepseek-ai/dsh-agent': agentPackage.version,
    '@deepseek-ai/dsh-session': sessionPackage.version,
    '@deepseek-ai/dsh-llm': llmPackage.version,
    '@deepseek-ai/dsh-token-meter': meterPackage.version,
  }
  const target = llmPackage.version
  if (!SUPPORTED_VERSIONS.includes(target as (typeof SUPPORTED_VERSIONS)[number])) {
    return `unsupported @deepseek-ai/dsh-llm@${target}`
  }
  for (const [pkg, ver] of Object.entries(actual)) {
    if (ver !== target) {
      return `requires ${pkg}@${target}, found ${ver}`
    }
  }
  return undefined
}

async function waitForPressureRegistry(ctx: Context): Promise<unknown> {
  const deadline = Date.now() + 30_000
  while (true) {
    const registry = ctx.get('sessionProjections') as unknown
    if (
      registry !== undefined &&
      (registry as { registrations?: Map<string, unknown> }).registrations?.has(PRESSURE_KEY) === true
    ) {
      return registry
    }
    if (Date.now() >= deadline) return undefined
    await new Promise(resolve => setTimeout(resolve, 4))
  }
}

/* -------------------------------------------------------------------------- *
 * Anchor & settlement helpers
 * -------------------------------------------------------------------------- */

interface DispatchSnapshot {
  readonly attachedTokens: number
  readonly durableDiffs: ReadonlyMap<MessageId, DurableMessageDiff>
}

interface SettledNodeInfo {
  readonly seq: SessionSeq
  readonly messageId: MessageId | undefined
  readonly usage: Readonly<TokenUsage>
  readonly heuristicTotal: number
  readonly heuristicReasoning: number
}

interface SettledAnchor {
  readonly seq: SessionSeq
  readonly usage: Readonly<TokenUsage>
  readonly header: EpochHeader
  readonly attachedTokens: number
  readonly turnSettledNodes: readonly SettledNodeInfo[]
  readonly dispatchDiffs: ReadonlyMap<MessageId, DurableMessageDiff> | undefined
}

interface MeterFoldState {
  readonly consumed: number
  readonly header: EpochHeader | undefined
  readonly turnSettledNodes: readonly SettledNodeInfo[]
  readonly settled: SettledAnchor | undefined
  readonly closed: boolean
}

function usageTotal(usage: TokenUsage): number {
  return usage.inputTokens + (usage.cacheReadTokens ?? 0) + (usage.cacheWriteTokens ?? 0) + usage.outputTokens
}

function splitReasoning(outputTokens: number, reasoning: number, visible: number): number {
  const total = reasoning + visible
  if (outputTokens <= 0 || reasoning <= 0) return 0
  if (total <= 0) return outputTokens
  return Math.min(outputTokens, Math.round(outputTokens * reasoning / total))
}

function sameEnvelope(billed: EpochHeader, next: EpochHeader): boolean {
  try {
    return headerEquals(billed, next)
  } catch {
    return false
  }
}

function heuristicMessageReasoning(message: Message | null): { total: number; reasoning: number } {
  if (message === null || message.content.length === 0) return { total: 0, reasoning: 0 }
  const total = estimateMessage(message)
  const reasoningBlocks = message.content.filter(b => b.type === 'reasoning')
  const reasoning = reasoningBlocks.length === 0 ? 0 : estimateContent(reasoningBlocks)
  return { total, reasoning }
}

function settledReasoningShare(node: SettledNodeInfo, diff: DurableMessageDiff): number {
  const { usage } = node
  if (diff.removedOtherTokens === 0 && diff.removedReasoningTokens > 0) {
    if (usage.reasoningTokens !== undefined && usage.reasoningTokens > 0) {
      return usage.reasoningTokens
    }
    const visible = Math.max(0, node.heuristicTotal - node.heuristicReasoning)
    return splitReasoning(usage.outputTokens, node.heuristicReasoning, visible)
  }
  if (diff.removedReasoningTokens > 0) {
    const visible = Math.max(0, node.heuristicTotal - node.heuristicReasoning)
    const reasoningPart =
      usage.reasoningTokens !== undefined && usage.reasoningTokens > 0
        ? usage.reasoningTokens
        : splitReasoning(usage.outputTokens, node.heuristicReasoning, visible)
    return reasoningPart + diff.removedOtherTokens
  }
  return Math.max(0, -diff.deltaTokens)
}

/* -------------------------------------------------------------------------- *
 * Driver implementation for DSH 0.2.2-rc.5
 * -------------------------------------------------------------------------- */

export const dsh022Rc5Driver: HostDriver = {
  id: 'dsh-0.2.2',
  dshVersions: SUPPORTED_VERSIONS,

  probe(ctx: Context): ProbeResult {
    const versionErr = checkVersions()
    if (versionErr) return { ok: false, reason: versionErr }

    const llm = ctx.get('llm') as LlmRuntime | undefined
    if (!llm) return { ok: false, reason: 'llm service is missing' }
    if (!Object.isExtensible(llm)) return { ok: false, reason: 'llm service is not extensible' }
    const llmDesc =
      Object.getOwnPropertyDescriptor(llm, 'forAdapter') ??
      Object.getOwnPropertyDescriptor(Object.getPrototypeOf(llm), 'forAdapter')
    const llmErr = assertMethod(llmDesc?.value, Reflect.get(LlmRuntime.prototype, 'forAdapter'), 'LlmRuntime.forAdapter')
    if (llmErr) return { ok: false, reason: llmErr }

    const meter = ctx.get('tokenMeter') as TokenMeter | undefined
    if (!meter) return { ok: false, reason: 'tokenMeter service is missing' }
    if (!Object.isExtensible(meter)) return { ok: false, reason: 'tokenMeter service is not extensible' }
    const meterDesc =
      Object.getOwnPropertyDescriptor(meter, 'measure') ??
      Object.getOwnPropertyDescriptor(Object.getPrototypeOf(meter), 'measure')
    const meterErr = assertMethod(meterDesc?.value, TokenMeter.prototype.measure, 'TokenMeter.measure')
    if (meterErr) return { ok: false, reason: meterErr }

    return { ok: true }
  },

  async install(ctx: Context, pipeline: TransformPipeline): Promise<() => void> {
    const llm = ctx.get('llm') as LlmRuntime
    const meter = ctx.get('tokenMeter') as TokenMeter
    const undos: (() => void)[] = []

    const lastDispatchBySession = new Map<SessionId, DispatchSnapshot>()
    const foldStates = new WeakMap<Session, MeterFoldState>()

    // 1. Patch LlmRuntime.forAdapter
    const llmTarget = llm as unknown as {
      forAdapter(options: GenerateOptions, adapter: unknown): GenerateOptions
    }
    const originalForAdapter = llmTarget.forAdapter
    let activeLlm = true
    undos.push(
      patchMethod(llmTarget, 'forAdapter', function (this: LlmRuntime, options: GenerateOptions, adapter: unknown) {
        const base = originalForAdapter.call(this, options, adapter)
        if (!activeLlm) return base

        const diff = pipeline.run({
          messages: base.messages,
          sessionId: base.sessionId,
          purpose: base.purpose,
          provider: base.provider,
          model: base.model,
        })

        if (base.sessionId !== undefined && base.purpose === undefined) {
          lastDispatchBySession.set(base.sessionId, {
            attachedTokens: diff.attachedTokens,
            durableDiffs: diff.durableDiffs,
          })
        }

        if (diff.messages === base.messages) return base
        const frozenMessages = freezeModifiedMessages(base.messages, diff.messages, Object.isFrozen(base.messages))
        const rewritten = { ...base, messages: frozenMessages }
        return Object.isFrozen(base) ? Object.freeze(rewritten) : rewritten
      }),
    )
    undos.push(() => { activeLlm = false })

    // 2. Patch TokenMeter.measure
    function latestAnchor(session: Session): { anchor: SettledAnchor | undefined; closed: boolean } {
      const prev: MeterFoldState = foldStates.get(session) ?? {
        consumed: 0,
        header: undefined,
        turnSettledNodes: [],
        settled: undefined,
        closed: false,
      }
      if (prev.consumed >= session.seq) {
        return { anchor: prev.settled, closed: prev.closed }
      }

      let consumed = prev.consumed
      let header = prev.header
      let turnSettledNodes = [...prev.turnSettledNodes]
      let settled = prev.settled
      let closed = prev.closed

      while (consumed < session.seq) {
        const event = session.eventAt(SessionSeq(consumed))
        consumed += 1
        if (event === undefined) continue
        if (event.surfaceOp !== undefined && event.surfaceOp !== 'append') {
          settled = undefined
        }
        switch (event.type) {
          case 'request/header': {
            const next = event.data.header
            header = next
            if (settled !== undefined && !sameEnvelope(settled.header, next)) settled = undefined
            break
          }
          case 'user/message':
          case 'turn/end': {
            turnSettledNodes = []
            if (settled !== undefined) closed = true
            break
          }
          case 'assistant/message': {
            const usage = event.data.usage
            if (usage === undefined) break
            const derived = deriveEventMessage(event)
            const { total, reasoning } = heuristicMessageReasoning(derived)
            turnSettledNodes.push({
              seq: event.seq,
              messageId: derived?.id,
              usage,
              heuristicTotal: total,
              heuristicReasoning: reasoning,
            })
            if (header !== undefined) {
              const dispatch = lastDispatchBySession.get(session.id)
              settled = {
                seq: event.seq,
                usage,
                header,
                attachedTokens: dispatch?.attachedTokens ?? 0,
                turnSettledNodes: [...turnSettledNodes],
                dispatchDiffs: dispatch?.durableDiffs,
              }
              closed = false
            }
            break
          }
          default:
            break
        }
      }

      const nextState: MeterFoldState = { consumed, header, turnSettledNodes, settled, closed }
      foldStates.set(session, nextState)
      return { anchor: settled, closed }
    }

    const originalMeasure = meter.measure
    let activeMeter = true
    undos.push(
      patchMethod(meter, 'measure', function (this: TokenMeter, session: Session, header?: EpochHeader) {
        const raw = originalMeasure.call(this, session, header)
        if (!activeMeter) return raw

        const surfaceMessages: RequestMessage[] = []
        const seqByMessageId = new Map<MessageId, SessionSeq>()
        const messageIdBySeq = new Map<SessionSeq, MessageId>()

        for (const node of raw.nodes) {
          const event = session.eventAt(node.seq)
          if (event === undefined) continue
          const msg = deriveEventMessage(event)
          if (msg === null) continue
          surfaceMessages.push(msg)
          seqByMessageId.set(msg.id, node.seq)
          messageIdBySeq.set(node.seq, msg.id)
        }

        const diff = pipeline.run({
          messages: surfaceMessages,
          sessionId: session.id,
          purpose: undefined,
        })

        let nodesChanged = false
        const adjustedNodes = raw.nodes.map((node) => {
          const msgId = messageIdBySeq.get(node.seq)
          if (msgId === undefined) return node
          const d = diff.durableDiffs.get(msgId)
          if (d === undefined || d.deltaTokens === 0) return node
          nodesChanged = true
          return Object.freeze({
            ...node,
            tokens: Math.max(0, node.tokens + d.deltaTokens),
            heuristicTokens: Math.max(0, node.heuristicTokens + d.deltaTokens),
          })
        })
        const finalNodes = nodesChanged ? Object.freeze(adjustedNodes) : raw.nodes
        const surfaceTokens = nodesChanged
          ? finalNodes.reduce((sum, node) => sum + node.tokens, 0)
          : raw.surfaceTokens

        const effective = header === undefined ? session.requestHeader() : canonicalHeader(header)
        const { anchor } = latestAnchor(session)

        if (anchor === undefined || (effective !== undefined && !sameEnvelope(anchor.header, effective))) {
          const totalTokens = Math.max(
            0,
            surfaceTokens + estimateToolsTokens(effective ?? session.requestHeader()) + diff.attachedTokens,
          )
          return Object.freeze({
            ...raw,
            nodes: finalNodes,
            surfaceTokens,
            baseline: Object.freeze({ kind: 'estimated' as const, tokens: totalTokens }),
            surfaceDeltaTokens: diff.attachedTokens,
            totalTokens,
          })
        }

        let appended = 0
        for (const node of finalNodes) {
          if (node.seq > anchor.seq) appended += node.tokens
        }

        // Deduct content that was present in anchor's billed usage but is now omitted/reduced.
        let dropped = 0
        const turnNodeBySeq = new Map<SessionSeq, SettledNodeInfo>()
        for (const tn of anchor.turnSettledNodes) turnNodeBySeq.set(tn.seq, tn)

        for (const node of raw.nodes) {
          if (node.seq > anchor.seq) continue
          const msgId = messageIdBySeq.get(node.seq)
          if (msgId === undefined) continue
          const nowDiff = diff.durableDiffs.get(msgId)
          if (nowDiff === undefined || nowDiff.deltaTokens >= 0) continue

          const turnNode = turnNodeBySeq.get(node.seq)
          if (turnNode !== undefined) {
            const atDispatch = anchor.dispatchDiffs?.get(msgId)
            const alreadyDeducted = atDispatch !== undefined && atDispatch.deltaTokens < 0
              ? settledReasoningShare(turnNode, atDispatch)
              : 0
            dropped += Math.max(0, settledReasoningShare(turnNode, nowDiff) - alreadyDeducted)
          } else if (anchor.dispatchDiffs !== undefined) {
            const atDispatch = anchor.dispatchDiffs.get(msgId)
            const alreadyDeducted = atDispatch !== undefined && atDispatch.deltaTokens < 0
              ? -atDispatch.deltaTokens
              : 0
            dropped += Math.max(0, -nowDiff.deltaTokens - alreadyDeducted)
          }
        }

        const billed = usageTotal(anchor.usage)
        const netDelta = appended - dropped - anchor.attachedTokens + diff.attachedTokens
        return Object.freeze({
          ...raw,
          nodes: finalNodes,
          surfaceTokens,
          baseline: Object.freeze({ kind: 'usage' as const, tokens: billed, usage: anchor.usage }),
          surfaceDeltaTokens: netDelta,
          totalTokens: Math.max(0, billed + netDelta),
        })
      }),
    )
    undos.push(() => { activeMeter = false })

    // 3. Patch contextPressure projection if present
    const registry = await waitForPressureRegistry(ctx)
    if (registry !== undefined) {
      const pressureUndo = installPressureProjection(registry, pipeline)
      if (pressureUndo) undos.push(pressureUndo)
    }

    return () => {
      for (const undo of undos.splice(0).reverse()) undo()
    }
  },
}

/* -------------------------------------------------------------------------- *
 * contextPressure projection adapter (pure fold using pipeline transforms)
 * -------------------------------------------------------------------------- */

interface PressureUnit {
  def: {
    stateVersion: number
    stateSchema: { parse(value: unknown): Record<string, unknown> }
    init(header: { origin?: string }, inheritedEventCount: number): Record<string, unknown>
    apply(state: Record<string, unknown>, event: SessionEvent): Record<string, unknown>
    wire?: {
      viewSchema: { parse(value: unknown): unknown }
      view(state: Record<string, unknown>): Record<string, unknown>
      summary?: boolean | undefined
    }
  }
  cells: WeakMap<object, unknown>
  refs?: number
}

interface PressureState extends Record<string, unknown> {
  readonly surfaceTokens: number
  readonly envelope?: EpochHeader
  readonly billedTotal?: number
  readonly appendedTokens: number
  readonly turnReasoning: number
  readonly droppedReasoning: number
  readonly openHeuristicReasoning: number
  readonly unanchoredOpenReasoning: number
  readonly closedHeuristicReasoning: number
}

function installPressureProjection(registry: unknown, pipeline: TransformPipeline): (() => void) | undefined {
  const registrations = (registry as { registrations?: Map<string, PressureUnit> }).registrations
  const registration = registrations?.get(PRESSURE_KEY)
  if (!registration || registration.refs === 0) return undefined
  const original = registration.def
  if (original.stateVersion !== HOST_PRESSURE_VERSION || original.wire === undefined) return undefined

  const wire = original.wire
  const hostApply = original.apply
  const hostInit = original.init
  const hostView = wire.view

  const clearAnchor = (state: PressureState): PressureState => {
    const { billedTotal: _billed, ...rest } = state
    return { ...rest, appendedTokens: 0, unanchoredOpenReasoning: 0, closedHeuristicReasoning: 0 }
  }

  // Estimate closed-turn hidden tokens for a single assistant message via the registered pipeline
  const hiddenForClosedMessage = (event: SessionEvent): { total: number; reasoning: number } => {
    const derived = deriveEventMessage(event)
    if (derived === null) return { total: 0, reasoning: 0 }
    // Simulate a closed turn by placing a dummy user message after this message
    const dummyUser: RequestMessage = { role: 'user', content: [{ type: 'text', text: '' }] }
    const diff = pipeline.run({ messages: [derived, dummyUser], purpose: undefined })
    const d = diff.durableDiffs.get(derived.id)
    if (d === undefined || d.deltaTokens >= 0) return { total: 0, reasoning: 0 }
    return { total: -d.deltaTokens, reasoning: d.removedReasoningTokens }
  }

  const apply = (state: PressureState, event: SessionEvent): PressureState => {
    const base = hostApply(state, event) as PressureState
    let next = base
    const appendedTokens = state.appendedTokens + (base.surfaceTokens - state.surfaceTokens)
    if (appendedTokens !== state.appendedTokens) next = { ...next, appendedTokens }
    if (event.surfaceOp !== undefined && event.surfaceOp !== 'append') next = clearAnchor(next)

    switch (event.type) {
      case 'request/header': {
        const envelope = event.data.header
        if (next.envelope !== undefined && !sameEnvelope(next.envelope, envelope)) next = clearAnchor(next)
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
        const derived = deriveEventMessage(event)
        const { total: msgTotal } = heuristicMessageReasoning(derived)
        const hidden = hiddenForClosedMessage(event)
        const usage = event.data.usage
        if (usage === undefined) {
          next = {
            ...next,
            openHeuristicReasoning: next.openHeuristicReasoning + hidden.total,
            unanchoredOpenReasoning: next.unanchoredOpenReasoning + hidden.total,
          }
          break
        }
        const turnDrop =
          hidden.total === 0
            ? 0
            : hidden.reasoning > 0 && usage.reasoningTokens !== undefined && usage.reasoningTokens > 0
              ? usage.reasoningTokens
              : splitReasoning(usage.outputTokens, hidden.reasoning, Math.max(0, msgTotal - hidden.reasoning)) +
                Math.max(0, hidden.total - hidden.reasoning)

        next = {
          ...next,
          turnReasoning: next.turnReasoning + turnDrop,
          droppedReasoning: 0,
          ...next.envelope === undefined ? {} : { billedTotal: usageTotal(usage) },
          appendedTokens: 0,
          openHeuristicReasoning: next.openHeuristicReasoning + hidden.total,
          unanchoredOpenReasoning: 0,
        }
        break
      }
      default:
        break
    }
    if (next.billedTotal !== undefined) {
      next = { ...next, pressureTokens: Math.max(0, next.billedTotal + next.appendedTokens - next.droppedReasoning) }
    } else if (next.envelope !== undefined) {
      next = { ...next, pressureTokens: Math.max(0, next.surfaceTokens - next.closedHeuristicReasoning + estimateToolsTokens(next.envelope)) }
    }
    return next
  }

  const installed: PressureUnit['def'] = {
    ...original,
    stateVersion: PRESSURE_VERSION,
    init: (header, inheritedEventCount) => ({
      ...hostInit(header, inheritedEventCount),
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
      view: (state) => {
        const p = state as PressureState
        const host = hostView(state)
        if (p.billedTotal !== undefined) {
          return { ...host, projectedTokens: Math.max(0, p.billedTotal + p.appendedTokens - p.droppedReasoning) }
        }
        return {
          ...host,
          projectedTokens: Math.max(0, p.surfaceTokens - p.closedHeuristicReasoning + estimateToolsTokens(p.envelope)),
        }
      },
    },
  }

  registration.def = installed
  registration.cells = new WeakMap()
  return () => {
    if (registration.def !== installed) return
    registration.def = original
    registration.cells = new WeakMap()
  }
}
