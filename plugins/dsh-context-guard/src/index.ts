import type { Context } from '@deepseek-ai/cordis'
import type { Session } from '@deepseek-ai/dsh-session'
import type {} from '@deepseek-ai/dsh-agent'
import type {} from '@deepseek-ai/dsh-session-query'
import type {} from '@deepseek-ai/dsh-commands'
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
import { buildS3Window, extractCardsFromEvents } from './cards.js'
import type { ExtractedCard, StoredCard } from './cards.js'
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

export { Config, GuardStore, LlamaGuardRuntime, extractCardsFromEvents, buildS3Window }
export type {
  GuardInferenceClient,
  GuardSessionStatus,
  SemanticSearchHit,
  SummaryCardItem,
  SummaryLexicalSearchOptions,
  SummaryRangeResult,
  ExtractedCard,
  StoredCard,
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
 * Passive real-time recorder, degradation scorer, and semantic history search provider.
 * Writes session-scoped `guard.sqlite` (+ optional `timeline.txt`) under `~/.dsh/context-cortex/session-<hash>/`.
 * Does NOT inject reminders or trigger compaction in this phase.
 */
export class ContextGuardEngine implements ContextGuardApi {
  readonly store: GuardStore
  readonly inference: GuardInferenceClient
  private readonly timers = new Map<Session, ReturnType<typeof setTimeout>>()
  private readonly pending = new Set<Promise<unknown>>()
  private readonly failures = new Map<string, string>()
  private readonly shutdownController = new AbortController()
  private readonly debounceMs: number
  private startupError: string | null = null
  private stopping = false

  constructor(private readonly ctx: Context, config: Config = {}, opts: GuardServiceOptions = {}) {
    const storeRoot = resolveStoreRoot(config)
    this.debounceMs = config.debounceMs ?? 300
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
    ctx.on('session/event', session => schedule(session))
    ctx.on('session/flush', async session => {
      if (session.header.origin !== 'subagent') {
        await this.syncSession(session).catch(() => { /* recorded in failures map */ })
      }
    })
    ctx.on('session/disposed', session => {
      const timer = this.timers.get(session)
      if (timer) clearTimeout(timer)
      this.timers.delete(session)
      this.store.close(session.id)
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
      this.failures.delete(session.id)
      return this.store.getStatus(session.id)
    } catch (error) {
      this.failures.set(session.id, error instanceof Error ? error.message : String(error))
      throw error
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
