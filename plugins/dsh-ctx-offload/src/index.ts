import type { Context } from '@deepseek-ai/cordis'
import type { Agent } from '@deepseek-ai/dsh-agent'
import { BasicCompactionEngine } from '@deepseek-ai/dsh-compaction-basic'
import type { CompactionTrigger, CompactionResult } from '@deepseek-ai/dsh-compaction'
import type { Session, SessionSeq } from '@deepseek-ai/dsh-session'
import { BlockAssembler, LlmError, contentHasImage, createUserMessage, CONTEXT_WINDOW_EXCEEDED_CODE } from '@deepseek-ai/dsh-llm'
import type { Message, ToolSchema, ContentBlock, TokenUsage } from '@deepseek-ai/dsh-llm'
import type {} from '@deepseek-ai/dsh-session-query'
import type {} from '@deepseek-ai/dsh-commands'
import basicPackage from '@deepseek-ai/dsh-compaction-basic/package.json' with { type: 'json' }
import queryPackage from '@deepseek-ai/dsh-session-query/package.json' with { type: 'json' }
import spillPackage from '@deepseek-ai/dsh-spill-local/package.json' with { type: 'json' }
import { Config, archiveRoot } from './config.js'
import { ArchiveStore } from './archive.js'
import { briefInstruction, frameBrief, validateBrief } from './brief.js'
import { selectRange, budgets } from './policy.js'

export { Config }
export const name = 'dsh-ctx-offload'
export const inject = ['llm', 'tokenMeter', 'sessions', 'sessionQuery', 'agents', 'commands']

type BriefResult = {
  summary: ContentBlock[]; provider: string; model: string; maxTokens?: number; usage?: TokenUsage
} & ({ rawOutput: ContentBlock[]; llmStreamCall: true } | { rawOutput?: ContentBlock[]; llmStreamCall?: never })

/** Uses the host transaction implementation; only policy and brief generation differ. */
export class ContextOffloadEngine extends BasicCompactionEngine {
  static override inject = inject
  readonly offload: Config
  readonly archives: ArchiveStore
  private readonly timers = new Map<Session, ReturnType<typeof setTimeout>>()
  private readonly pending = new Set<Promise<unknown>>()
  private readonly failures = new Map<string, string>()
  private stopping = false

  constructor(ctx: Context, config: Config) {
    for (const pkg of [basicPackage, queryPackage, spillPackage]) {
      if (pkg.version !== '0.2.2-rc.1') throw new Error(`Context offload requires ${pkg.name}@0.2.2-rc.1; found ${pkg.version}`)
    }
    const root = archiveRoot(config)
    super(ctx, {
      // Use the configured-budget hooks below rather than the host model-window policy.
      auto: false,
      thresholdRatio: 0.75,
      headroomTokens: 0,
      retainTokens: 0,
      maxTokens: budgets(config).briefMaxTokens,
      compactionRetries: 0,
    })
    this.offload = config
    this.archives = new ArchiveStore(root)
    const overflowAttempts = new WeakSet<Agent>()
    ctx.on('agent/status', ({ agent, status }) => { if (status === 'idle') overflowAttempts.delete(agent) })
    ctx.on('agent/pre-step', async ({ agent, signal }, next) => {
      if (!signal.aborted) {
        try { await this.compactIfNeeded(agent, 'pressure', signal) }
        catch (error) {
          signal.throwIfAborted()
          ctx.logger.warn(`Context offload compaction failed: ${error instanceof Error ? error.message : String(error)}`)
        }
      }
      return next()
    })
    ctx.on('agent/request-error', async ({ agent, failure, signal }, next) => {
      if (failure.code !== CONTEXT_WINDOW_EXCEEDED_CODE || signal.aborted || overflowAttempts.has(agent)) return next()
      overflowAttempts.add(agent)
      try {
        const result = await this.compactIfNeeded(agent, 'context-overflow', signal)
        return result ? { kind: 'retry' } : next()
      } catch (error) {
        signal.throwIfAborted()
        ctx.logger.warn(`Context offload overflow recovery failed: ${error instanceof Error ? error.message : String(error)}`)
        return next()
      }
    })
    const schedule = (session: Session) => {
      if (this.stopping || session.header.origin === 'subagent' || this.timers.has(session)) return
      const timer = setTimeout(() => {
        this.timers.delete(session)
        const operation = this.capture(session).catch(() => { /* status records the failure; barriers retry */ })
        this.pending.add(operation)
        void operation.finally(() => this.pending.delete(operation))
      }, 25)
      timer.unref()
      this.timers.set(session, timer)
    }
    ctx.on('session/created', schedule)
    ctx.on('session/event', session => schedule(session))
    ctx.on('session/flush', async session => {
      if (session.header.origin !== 'subagent') await this.capture(session)
    })
    ctx.on('session/disposed', session => {
      const timer = this.timers.get(session)
      if (timer) clearTimeout(timer)
      this.timers.delete(session)
    })
    ctx.effect(() => async () => {
      this.stopping = true
      const sessions = [...this.timers.keys()]
      for (const timer of this.timers.values()) clearTimeout(timer)
      this.timers.clear()
      await Promise.allSettled([...this.pending, ...sessions.map(session => this.capture(session))])
      await this.archives.drain()
    })
    for (const agent of ctx.agents.list()) schedule(agent.session)
    ctx.commands.register({
      name: 'ctx-offload', description: 'Inspect, verify, or compact the context archive',
      handler: async ({ agent, rawInput, signal, commandId }) => {
        const action = rawInput.trim() || 'status'
        if (agent.session.header.origin === 'subagent') return { kind: 'error', text: 'Context offload targets main-agent sessions.' }
        if (action === 'compact') {
          const result = await this.compactNow(agent, signal, commandId)
          return { kind: 'success', text: result ? `Compacted ${result.shadowedSeqs.length} history items.` : 'No compactable history.' }
        }
        if (action === 'verify') {
          const catalog = await this.archives.verify(agent.session.id)
          return { kind: 'success', text: catalog ? `Archive verified through event ${catalog.through} (${catalog.chunks.length} chunks).` : 'No archive yet.' }
        }
        if (action !== 'status') return { kind: 'error', text: 'Usage: /ctx-offload [status|verify|compact]' }
        const catalog = await this.archives.verify(agent.session.id)
        return { kind: 'success', text: JSON.stringify({ root: this.archives.root, contextLimit: config['context-limit'], ...budgets(config), archivedThrough: catalog?.through ?? -1, chunks: catalog?.chunks.length ?? 0, error: this.failures.get(agent.session.id) ?? null }, null, 2) }
      },
    })
  }

  private async capture(session: Session, signal?: AbortSignal) {
    try {
      const observation = await this.ctx.sessionQuery.observeSession(session.id, { projectionMode: 'none', ...(signal ? { signal } : {}) })
      try {
        const cut = await this.archives.ensure(session.id, observation.events, signal)
        this.failures.delete(session.id)
        return { ...cut, events: observation.events }
      } finally { observation[Symbol.dispose]() }
    } catch (error) {
      this.failures.set(session.id, error instanceof Error ? error.message : String(error))
      throw error
    }
  }

  override async compactIfNeeded(agent: Agent, trigger: CompactionTrigger, signal: AbortSignal): Promise<CompactionResult | null> {
    if (agent.session.header.origin === 'subagent') return null
    const header = agent.session.requestHeader()
    if (!header) return null
    const budget = budgets(this.offload)
    const observation = await this.ctx.sessionQuery.observeSession(agent.session.id, { projectionMode: 'none', signal })
    let hasSystemHead: boolean
    try {
      const head = agent.session.surface.nodes[0]
      hasSystemHead = head !== undefined && observation.events[Number(head)]?.type === 'system/message'
    } finally { observation[Symbol.dispose]() }
    const measurement = this.ctx.tokenMeter.measure(agent.session)
    if (trigger === 'pressure' && measurement.totalTokens < budget.triggerTokens) return null
    const range = selectRange(agent.session, measurement, budget.retainRecentTokens, hasSystemHead)
    if (!range) return null
    return this.compactRegion(range.start, range.end, agent, signal)
  }

  override async compactRegion(start: SessionSeq, end: SessionSeq, agent: Agent, signal?: AbortSignal): Promise<CompactionResult> {
    const result = await super.compactRegion(start, end, agent, signal)
    await this.ctx.sessions.flush(agent.session)
    return result
  }

  protected override async summarize(input: { readonly messages: readonly Message[]; readonly tools?: readonly ToolSchema[] }, agent: Agent, signal?: AbortSignal): Promise<BriefResult> {
    if (agent.session.header.origin === 'subagent') return super.summarize(input, agent, signal)
    const cut = await this.capture(agent.session, signal)
    const target = agent.session.requestHeader()?.config ?? agent.options
    if (!target.provider || !target.model) throw new Error('No provider/model available for execution brief')
    const budget = budgets(this.offload)
    const selected = new Set(input.messages.map(message => message.id))
    const measurement = this.ctx.tokenMeter.measure(agent.session)
    const retained = measurement.nodes.reduce((total, node) => {
      const event = cut.events[Number(node.seq)]
      if (!event || event.type === 'system/message') return total
      const message = agent.session.deriveEventMessage(event)
      return message && !selected.has(message.id) ? total + node.tokens : total
    }, 0)
    if (retained > budget.retainRecentTokens) throw new Error('The host-selected recent tail exceeds its upper limit; use automatic compaction or a smaller last message')
    const guidanceTokens = this.ctx.tokenMeter.estimateMessage(createUserMessage({ source: { kind: 'user' }, content: [{ type: 'text', text: frameBrief('', cut) }] }))
    const bodyBudget = budget.briefMaxTokens - guidanceTokens
    if (bodyBudget <= 0) throw new Error('Archive guidance exceeds the execution brief upper limit')
    signal?.throwIfAborted()
    const assembler = new BlockAssembler()
    for await (const chunk of this.ctx.llm.stream({
      provider: target.provider, model: target.model,
      messages: [...input.messages, { role: 'user', content: [{ type: 'text', text: briefInstruction(bodyBudget) }] }],
      ...(input.tools === undefined ? {} : { tools: [...input.tools] }),
      toolHistory: agent.session.toolHistory(),
      // dsh's summarization headroom. Its own compaction uses this same value as the
      // summarize call cap (BasicCompactionConfig.headroomTokens), because that reply
      // carries reasoning as well as the text summary.
      maxTokens: 65536,
      sessionId: agent.session.id, purpose: 'compaction',
      ...(signal ? { signal } : {}),
    })) assembler.push(chunk)
    signal?.throwIfAborted()
    // Fail the brief on the first attempt, like the host summarizer. Recovery is its
    // `compaction/summary-error` waterfall: a listener there may change the input and
    // re-run this hook. A content verdict or a truncation cannot be re-asked, so
    // retrying the same request would only pay for the same reply twice.
    const finish = assembler.finish
    if (finish.kind === 'error' || finish.kind === 'aborted') throw new LlmError(finish.failure.message, finish.failure.code, finish.failure)
    if (finish.kind === 'max-tokens') throw Object.assign(new Error('Execution brief truncated at the token cap (incomplete checkpoint)'), { code: 'MAX_TOKENS' })
    const blocks = assembler.blocks()
    if (contentHasImage(blocks)) throw new LlmError('Execution brief cannot contain image output', 'UNSUPPORTED_CONTENT')
    // Only the text survives into history, so reasoning is neither counted nor retained.
    const rawOutput = blocks.filter(block => block.type === 'text')
    const body = rawOutput.map(block => block.text).join('\n')
    validateBrief(body)
    const text = frameBrief(body, cut)
    const tokens = this.ctx.tokenMeter.estimateMessage(createUserMessage({ source: { kind: 'user' }, content: [{ type: 'text', text }] }))
    if (tokens > budget.briefMaxTokens) throw new Error('Execution brief including archive guidance exceeds briefMaxTokens')
    await this.archives.saveBrief(cut, text)
    return { summary: [{ type: 'text' as const, text }], rawOutput, llmStreamCall: true as const, provider: target.provider, model: target.model, maxTokens: 65536, ...(assembler.usage === undefined ? {} : { usage: assembler.usage }) }
  }
}

export async function apply(ctx: Context, config: Config): Promise<void> {
  // The wrapper's Config is resolved once; the engine receives the host backend schema.
  const fiber = ctx.plugin(class extends ContextOffloadEngine {
    constructor(child: Context) { super(child, config) }
  })
  await fiber.await()
}
