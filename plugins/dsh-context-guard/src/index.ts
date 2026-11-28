import type { Context } from '@deepseek-ai/cordis'
import type { Session } from '@deepseek-ai/dsh-session'
import type {} from '@deepseek-ai/dsh-agent'
import type { CompactionEngine } from '@deepseek-ai/dsh-compaction'
import type {} from '@deepseek-ai/dsh-session-query'
import type {} from '@deepseek-ai/dsh-commands'
import type { ContentBlock } from '@deepseek-ai/dsh-llm'
import type { PostToolDecision } from '@deepseek-ai/dsh-tools'
import {
  Config,
  DEFAULT_EMB_MODEL,
  DEFAULT_HF_ENDPOINT,
  DEFAULT_HF_REPO,
  DEFAULT_JUDGE_HEAD,
  DEFAULT_JUDGE_MODEL,
  DEFAULT_LLAMA_BUILD,
  DEFAULT_SUM_MODEL,
  resolveModelPath,
  resolveStoreRoot,
} from './config.js'
import { buildS3Window, extractCardsFromEvents, isCompactionEvent } from './cards.js'
import type { ExtractedCard, StoredCard } from './cards.js'
import {
  DEFAULT_COOLDOWN_STEPS,
  DEFAULT_DEGRADED_THRESHOLD,
  DEFAULT_HIT_WINDOW_STEPS,
  DEFAULT_HITS_TO_TRIGGER,
  DEFAULT_INTERVENTIONS_BEFORE_COMPACT,
  buildInterventionPrompt,
  evaluateIntervention,
  selectFullPrefixCompactRange,
} from './intervention.js'
import type { InterventionEvaluation, InterventionPolicy } from './intervention.js'
import { GuardStore } from './store.js'
import type {
  GuardSessionStatus,
  SemanticSearchHit,
  SummaryCardItem,
  SummaryLexicalSearchOptions,
  SummaryRangeResult,
} from './store.js'
import { LlamaGuardRuntime } from './runtime.js'
import type { GuardInferenceClient } from './runtime.js'

export {
  Config,
  GuardStore,
  LlamaGuardRuntime,
  extractCardsFromEvents,
  buildS3Window,
  evaluateIntervention,
  buildInterventionPrompt,
  selectFullPrefixCompactRange,
}
export type {
  GuardInferenceClient,
  GuardSessionStatus,
  SemanticSearchHit,
  SummaryCardItem,
  SummaryLexicalSearchOptions,
  SummaryRangeResult,
  ExtractedCard,
  StoredCard,
  InterventionPolicy,
  InterventionEvaluation,
}

export const name = 'dsh-context-guard'
export const inject = ['sessions', 'sessionQuery', 'agents', 'commands']

export interface GuardServiceOptions {
  inference?: GuardInferenceClient
}

export interface ContextGuardApi {
  readonly store: GuardStore
  syncSession(session: Session, signal?: AbortSignal): Promise<GuardSessionStatus>
  rebuildSession(session: Session, signal?: AbortSignal): Promise<GuardSessionStatus>
  semanticSearch(
    sessionId: string,
    query: string,
    options: { topK: number; fromId?: number; toId?: number; signal?: AbortSignal },
  ): Promise<SemanticSearchHit[]>
  lexicalSearch(sessionId: string, options: SummaryLexicalSearchOptions): Promise<SummaryCardItem[]>
  listSummaries(sessionId: string, options?: { fromId?: number; toId?: number }): Promise<SummaryRangeResult>
  status(sessionId: string): Promise<GuardSessionStatus>
}

function cardsEqualPrefix(stored: StoredCard, extracted: ExtractedCard): boolean {
  return (
    stored.cardIdx === extracted.cardIdx &&
    stored.kind === extracted.kind &&
    stored.tool === extracted.tool &&
    stored.startEventId === extracted.startEventId &&
    stored.endEventId === extracted.endEventId &&
    stored.callText === extracted.callText &&
    stored.resText === extracted.resText &&
    stored.compactBefore === extracted.compactBefore
  )
}

/**
 * Real-time recorder, degradation scorer, semantic history search provider,
 * and mid-tool-chain checkpoint injector (`tools/post-execute`).
 * Writes session-scoped `guard.sqlite` (+ optional `timeline.txt`) under `~/.dsh/context-cortex/session-<hash>/`.
 */
export class ContextGuardEngine implements ContextGuardApi {
  readonly store: GuardStore
  readonly inference: GuardInferenceClient
  private readonly timers = new Map<Session, ReturnType<typeof setTimeout>>()
  private readonly pending = new Set<Promise<unknown>>()
  private readonly failures = new Map<string, string>()
  private readonly lastInterventionCardIdx = new Map<string, number>()
  private readonly pendingInterventions = new Map<string, { triggerCardIdx: number; text: string }>()
  private readonly pendingCompactions = new Map<string, { triggerCardIdx: number }>()
  private readonly interventionCounts = new Map<string, number>()
  private readonly turnInterventionCounts = new Map<string, number>()
  private readonly autoCompactCounts = new Map<string, number>()
  private readonly shutdownController = new AbortController()
  private readonly debounceMs: number
  private readonly enableIntervention: boolean
  private readonly enableAutoCompact: boolean
  private readonly interventionsBeforeCompact: number
  private readonly interventionPolicy: InterventionPolicy
  private startupError: string | null = null
  private stopping = false

  constructor(private readonly ctx: Context, config: Config = {}, opts: GuardServiceOptions = {}) {
    const storeRoot = resolveStoreRoot(config)
    this.debounceMs = config.debounceMs ?? 300
    this.enableIntervention = config.enableIntervention ?? true
    this.enableAutoCompact = config.enableAutoCompact ?? true
    this.interventionsBeforeCompact = Math.max(0, Math.floor(config.interventionsBeforeCompact ?? DEFAULT_INTERVENTIONS_BEFORE_COMPACT))
    this.interventionPolicy = {
      degradedThreshold: config.degradedThreshold ?? DEFAULT_DEGRADED_THRESHOLD,
      hitWindowSteps: config.hitWindowSteps ?? DEFAULT_HIT_WINDOW_STEPS,
      hitsToTrigger: config.hitsToTrigger ?? DEFAULT_HITS_TO_TRIGGER,
      cooldownSteps: config.cooldownSteps ?? DEFAULT_COOLDOWN_STEPS,
    }
    this.store = new GuardStore(storeRoot, config.writeTimelineTxt ?? true)
    this.inference = opts.inference ?? new LlamaGuardRuntime({
      storeRoot,
      ...(config.llamaServerPath ? { llamaServerPath: config.llamaServerPath } : {}),
      llamaBuild: config.llamaBuild ?? DEFAULT_LLAMA_BUILD,
      autoDownloadRuntime: config.autoDownloadRuntime ?? true,
      autoDownloadModels: config.autoDownloadModels ?? true,
      hfRepo: config.hfRepo ?? DEFAULT_HF_REPO,
      hfEndpoint: config.hfEndpoint ?? process.env.HF_ENDPOINT ?? DEFAULT_HF_ENDPOINT,
      gpuLayers: config.gpuLayers ?? 99,
      summarizerModelPath: resolveModelPath(storeRoot, config.summarizerModel, DEFAULT_SUM_MODEL),
      summarizerModelCustom: config.summarizerModel !== undefined,
      judgeModelPath: resolveModelPath(storeRoot, config.judgeModel, DEFAULT_JUDGE_MODEL),
      judgeModelCustom: config.judgeModel !== undefined,
      judgeHeadPath: resolveModelPath(storeRoot, config.judgeHead, DEFAULT_JUDGE_HEAD),
      judgeHeadCustom: config.judgeHead !== undefined,
      embeddingModelPath: resolveModelPath(storeRoot, config.embeddingModel, DEFAULT_EMB_MODEL),
      embeddingModelCustom: config.embeddingModel !== undefined,
    })

    // Non-blocking startup asset check: verify llama-server runtime + 3 model weights and auto-download if missing.
    if ((config.checkAssetsOnStartup ?? true) && this.inference.ensureReady) {
      const bootOp = this.inference.ensureReady(this.shutdownController.signal).catch(err => {
        if (!this.stopping) {
          this.startupError = err instanceof Error ? err.message : String(err)
        }
      })
      this.pending.add(bootOp)
      void bootOp.finally(() => this.pending.delete(bootOp))
    }

    const schedule = (session: Session) => {
      if (this.stopping || session.header.origin === 'subagent' || this.timers.has(session)) return
      const timer = setTimeout(() => {
        this.timers.delete(session)
        const op = this.syncSession(session).catch(() => { /* recorded in failures map */ })
        this.pending.add(op)
        void op.finally(() => this.pending.delete(op))
      }, this.debounceMs)
      timer.unref()
      this.timers.set(session, timer)
    }

    ctx.on('session/created', schedule)
    ctx.on('session/event', (session, event) => {
      if (event.type === 'turn/end' || isCompactionEvent(event)) {
        this.resetTurnState(session.id)
      }
      schedule(session)
    })
    ctx.on('session/flush', async session => {
      if (session.header.origin !== 'subagent') {
        await this.syncSession(session).catch(() => { /* recorded in failures map */ })
      }
    })
    ctx.on('session/disposed', session => {
      const timer = this.timers.get(session)
      if (timer) clearTimeout(timer)
      this.timers.delete(session)
      this.resetTurnState(session.id)
      this.lastInterventionCardIdx.delete(session.id)
      this.interventionCounts.delete(session.id)
      this.autoCompactCounts.delete(session.id)
      this.store.close(session.id)
    })

    // Mid-tool-chain intervention delivery via tools/post-execute:
    // Appends `<system-reminder>` persistently onto the latest root `role: 'tool'` result content
    // instead of injecting a new `role: 'user'` message, preserving `lastUserIndex` so
    // `dsh-ephemeral-reasoning` never strips mid-chain reasoning and `dsh-ephemeral-reminder`
    // never shifts mid-turn (0 KV-Cache invalidation).
    ctx.on('tools/post-execute', async (exec, result, next): Promise<PostToolDecision> => {
      const downstream = await next()
      if (!this.enableIntervention || !exec.agent || exec.parent !== undefined) return downstream
      if (exec.agent.session.header.origin === 'subagent') return downstream

      const sessionId = exec.agent.session.id
      const queued = this.pendingInterventions.get(sessionId)
      if (!queued) return downstream

      const reminderBlock: ContentBlock = { type: 'text', text: `\n${queued.text}` }
      if (downstream.kind === 'accept') {
        if (downstream.value !== undefined) {
          // Do not clobber another policy's structured value replacement; retry on next tool result.
          return downstream
        }
        this.recordReminderDelivered(sessionId, queued.triggerCardIdx)
        const baseContent = downstream.content ?? result.content
        return {
          ...downstream,
          kind: 'accept',
          content: [...baseContent, reminderBlock],
        }
      }

      this.recordReminderDelivered(sessionId, queued.triggerCardIdx)
      return {
        ...downstream,
        feedback: [...downstream.feedback, reminderBlock],
      }
    })

    // Reset turn state on real user input; otherwise execute queued full-prefix auto-compact
    // mid-turn when a tool-call chain keeps degrading after `interventionsBeforeCompact` reminders.
    ctx.on('agent/pre-step', async ({ agent, messages, signal }, next) => {
      const hasRealUserMsg = messages.some(m => {
        const src = m.source as { kind?: string; form?: unknown } | undefined
        return src?.kind === 'user' && src.form === undefined
      })
      if (hasRealUserMsg) {
        this.resetTurnState(agent.session.id)
        return next()
      }

      if (!signal.aborted && this.enableIntervention && this.enableAutoCompact && agent.session.header.origin !== 'subagent') {
        const queuedCompact = this.pendingCompactions.get(agent.session.id)
        if (queuedCompact) {
          this.pendingCompactions.delete(agent.session.id)
          const compaction = this.getCompactionEngine()
          if (compaction) {
            const range = selectFullPrefixCompactRange(agent.session)
            if (range) {
              try {
                await compaction.compactRegion(range.start, range.end, agent, signal)
                this.resetTurnState(agent.session.id)
                this.lastInterventionCardIdx.set(agent.session.id, queuedCompact.triggerCardIdx)
                this.autoCompactCounts.set(agent.session.id, (this.autoCompactCounts.get(agent.session.id) ?? 0) + 1)
              } catch (err) {
                // Never abort the active turn if compaction fails; enter cooldown before retrying.
                this.lastInterventionCardIdx.set(agent.session.id, queuedCompact.triggerCardIdx)
                if (!signal.aborted) {
                  this.failures.set(agent.session.id, `auto-compact failed: ${err instanceof Error ? err.message : String(err)}`)
                }
              }
            }
          }
        }
      }
      return next()
    })

    for (const agent of ctx.agents.list()) schedule(agent.session)

    // Provide optional Cordis service so dsh-context-offload's hybrid history_search activates automatically.
    ctx.provide('contextGuard', this)

    ctx.commands.register({
      name: 'context-guard',
      description: 'Inspect or rebuild the session context-guard index (guard.sqlite)',
      handler: async ({ agent, rawInput, signal }) => {
        const action = rawInput.trim() || 'status'
        if (agent.session.header.origin === 'subagent') {
          return { kind: 'error', text: 'Context guard targets main-agent sessions.' }
        }
        if (action === 'rebuild') {
          const st = await this.rebuildSession(agent.session, signal)
          return {
            kind: 'success',
            text: `Rebuilt guard.sqlite through event ${st.throughEventId} (${st.cardCount} cards, ${st.scoredCount} scored actions).`,
          }
        }
        if (action !== 'status') {
          return { kind: 'error', text: 'Usage: /context-guard [status|rebuild]' }
        }
        const st = await this.status(agent.session.id)
        return {
          kind: 'success',
          text: JSON.stringify({
            ...st,
            intervention: {
              enabled: this.enableIntervention,
              policy: this.interventionPolicy,
              count: this.interventionCounts.get(agent.session.id) ?? 0,
              turnCount: this.turnInterventionCounts.get(agent.session.id) ?? 0,
              lastInterventionCardIdx: this.lastInterventionCardIdx.get(agent.session.id) ?? null,
              pending: this.pendingInterventions.has(agent.session.id),
              enableAutoCompact: this.enableAutoCompact,
              interventionsBeforeCompact: this.interventionsBeforeCompact,
              autoCompactCount: this.autoCompactCounts.get(agent.session.id) ?? 0,
              pendingCompact: this.pendingCompactions.has(agent.session.id),
            },
            runtime: this.inference.status(),
            startupError: this.startupError,
            error: this.failures.get(agent.session.id) ?? null,
          }, null, 2),
        }
      },
    })

    ctx.effect(() => async () => {
      this.stopping = true
      this.shutdownController.abort()
      for (const timer of this.timers.values()) clearTimeout(timer)
      this.timers.clear()
      await Promise.allSettled([...this.pending])
      await this.store.drain()
      this.store.close()
      await this.inference.dispose()
    })
  }

  async syncSession(session: Session, signal?: AbortSignal): Promise<GuardSessionStatus> {
    return this.processSession(session, false, signal)
  }

  async rebuildSession(session: Session, signal?: AbortSignal): Promise<GuardSessionStatus> {
    return this.processSession(session, true, signal)
  }

  private async processSession(session: Session, forceRebuild: boolean, signal?: AbortSignal): Promise<GuardSessionStatus> {
    try {
      const observation = await this.ctx.sessionQuery.observeSession(session.id, {
        projectionMode: 'none',
        ...(signal ? { signal } : {}),
      })
      let eventsSnapshot
      try {
        eventsSnapshot = [...observation.events]
      } finally {
        observation[Symbol.dispose]()
      }

      const throughEventId = eventsSnapshot.length - 1
      const extracted = extractCardsFromEvents(eventsSnapshot)
      const prev = forceRebuild ? { throughEventId: -1, cards: [] as StoredCard[] } : await this.store.loadCards(session.id)

      // Reuse unchanged prefix cards so we never re-summarize, re-embed, or re-score prior turns.
      let reuseLen = 0
      if (!forceRebuild) {
        const minLen = Math.min(prev.cards.length, extracted.length)
        while (reuseLen < minLen && cardsEqualPrefix(prev.cards[reuseLen]!, extracted[reuseLen]!)) {
          reuseLen++
        }
      }

      if (!forceRebuild && reuseLen === extracted.length && prev.cards.length === extracted.length && prev.throughEventId === throughEventId) {
        this.failures.delete(session.id)
        return this.store.getStatus(session.id)
      }

      const working: StoredCard[] = prev.cards.slice(0, reuseLen)
      for (let i = reuseLen; i < extracted.length; i++) {
        signal?.throwIfAborted()
        const card = extracted[i]!
        const micro = await this.inference.summarize(card, signal)
        const embText = card.kind === 'constraint'
          ? card.callText.replace(/\s+/g, ' ').trim().slice(0, 512)
          : micro
        const embedding = await this.inference.embed(embText, signal)

        const stored: StoredCard = {
          ...card,
          micro,
          embedding,
          pDegraded: null,
          createdAt: Date.now(),
        }
        working.push(stored)

        // Score every action card (t >= 2) using exact V2.4 S3_32+32 window (passive recording only).
        if (card.kind === 'action' && i >= 2) {
          const prompt = buildS3Window(working, i)
          stored.pDegraded = await this.inference.judge(prompt, signal)
        }
      }

      const replaceAll = forceRebuild || reuseLen < prev.cards.length
      await this.store.saveCards(session.id, throughEventId, working, replaceAll)
      if (this.enableIntervention && !forceRebuild) {
        this.updatePendingIntervention(session.id, working)
      }
      this.failures.delete(session.id)
      return this.store.getStatus(session.id)
    } catch (error) {
      this.failures.set(session.id, error instanceof Error ? error.message : String(error))
      throw error
    }
  }

  private getCompactionEngine(): CompactionEngine | undefined {
    const getter = (this.ctx as unknown as { get?: (name: string) => unknown }).get
    return typeof getter === 'function' ? (getter.call(this.ctx, 'compaction') as CompactionEngine | undefined) : undefined
  }

  private hasHistoryTools(): boolean {
    return this.getCompactionEngine() !== undefined
  }

  private resetTurnState(sessionId: string): void {
    this.turnInterventionCounts.delete(sessionId)
    this.pendingInterventions.delete(sessionId)
    this.pendingCompactions.delete(sessionId)
  }

  private recordReminderDelivered(sessionId: string, triggerCardIdx: number): void {
    this.pendingInterventions.delete(sessionId)
    this.lastInterventionCardIdx.set(sessionId, triggerCardIdx)
    this.interventionCounts.set(sessionId, (this.interventionCounts.get(sessionId) ?? 0) + 1)
    this.turnInterventionCounts.set(sessionId, (this.turnInterventionCounts.get(sessionId) ?? 0) + 1)
  }

  private updatePendingIntervention(sessionId: string, cards: readonly StoredCard[]): void {
    const lastIdx = this.lastInterventionCardIdx.get(sessionId) ?? -1
    const ev = evaluateIntervention(cards, lastIdx, this.interventionPolicy)
    if (ev.shouldIntervene && ev.triggerCardIdx !== null) {
      const turnCount = this.turnInterventionCounts.get(sessionId) ?? 0
      if (this.enableAutoCompact && turnCount >= this.interventionsBeforeCompact && this.getCompactionEngine() !== undefined) {
        this.pendingInterventions.delete(sessionId)
        this.pendingCompactions.set(sessionId, { triggerCardIdx: ev.triggerCardIdx })
      } else {
        const text = buildInterventionPrompt(cards, { hasHistoryTools: this.hasHistoryTools() })
        this.pendingInterventions.set(sessionId, { triggerCardIdx: ev.triggerCardIdx, text })
      }
    } else if (ev.actionsSinceReset <= this.interventionPolicy.cooldownSteps) {
      // A reset boundary (new user message / compact) occurred; clear any stale queued intervention/compaction.
      this.pendingInterventions.delete(sessionId)
      this.pendingCompactions.delete(sessionId)
    }
  }

  async semanticSearch(
    sessionId: string,
    query: string,
    options: { topK: number; fromId?: number; toId?: number; signal?: AbortSignal },
  ): Promise<SemanticSearchHit[]> {
    const queryVec = await this.inference.embed(query, options.signal)
    return this.store.semanticSearch(sessionId, queryVec, {
      topK: options.topK,
      ...(options.fromId !== undefined ? { fromId: options.fromId } : {}),
      ...(options.toId !== undefined ? { toId: options.toId } : {}),
    })
  }

  lexicalSearch(sessionId: string, options: SummaryLexicalSearchOptions): Promise<SummaryCardItem[]> {
    return this.store.lexicalSearch(sessionId, options)
  }

  listSummaries(sessionId: string, options?: { fromId?: number; toId?: number }): Promise<SummaryRangeResult> {
    return this.store.listSummaries(sessionId, options)
  }

  status(sessionId: string): Promise<GuardSessionStatus> {
    return this.store.getStatus(sessionId)
  }
}

export async function apply(ctx: Context, config: Config = {}): Promise<void> {
  new ContextGuardEngine(ctx, config)
}
