import type { Context } from '@deepseek-ai/cordis'
import type { Agent } from '@deepseek-ai/dsh-agent'
import { BasicCompactionEngine } from '@deepseek-ai/dsh-compaction-basic'
import type { CompactionResult } from '@deepseek-ai/dsh-compaction'
import type { Session, SessionSeq } from '@deepseek-ai/dsh-session'
import { BlockAssembler, LlmError, contentHasImage } from '@deepseek-ai/dsh-llm'
import type { Message, ToolSchema, ContentBlock, TokenUsage } from '@deepseek-ai/dsh-llm'
import type {} from '@deepseek-ai/dsh-session-query'
import type {} from '@deepseek-ai/dsh-commands'
import type {} from '@deepseek-ai/dsh-tools'
import { Config, archiveRoot, basicCompactionConfig } from './config.js'
import { ArchiveStore } from './archive.js'
import { briefInstruction, briefTargetTokens, frameBrief, validateBrief } from './brief.js'
import { registerHistoryTools } from './tools.js'

export { Config }
export const name = 'dsh-context-offload'
export const inject = ['llm', 'tokenMeter', 'sessions', 'sessionQuery', 'agents', 'commands', 'tools']

const BACKGROUND_DEBOUNCE_MS = 250

type BriefResult = {
  summary: ContentBlock[]; provider: string; model: string; maxTokens?: number; usage?: TokenUsage
} & ({ rawOutput: ContentBlock[]; llmStreamCall: true } | { rawOutput?: ContentBlock[]; llmStreamCall?: never })

function resolveRouteTarget(agent: Agent): { provider: string; model: string } | undefined {
  const routed = agent.session.requestHeader()?.config
  if (routed && routed.provider.length > 0 && routed.model.length > 0) {
    return { provider: routed.provider, model: routed.model }
  }
  const selected = agent.model?.current?.()
  if (selected && selected.provider.length > 0 && selected.model.length > 0) {
    return { provider: selected.provider, model: selected.model }
  }
  if (agent.options.provider && agent.options.model) {
    return { provider: agent.options.provider, model: agent.options.model }
  }
  return undefined
}

/**
 * Subclasses BasicCompactionEngine using its native dynamic contextWindow policy,
 * automatic pressure / model-switch / overflow handling, and transaction machinery.
 * Persists canonical session history into session-scoped SQLite (`offload.sqlite`),
 * writes forward-looking execution briefs, and exposes history search/read tools.
 */
export class ContextOffloadEngine extends BasicCompactionEngine {
  static override inject = inject
  readonly offload: Config
  readonly archives: ArchiveStore
  private readonly timers = new Map<Session, ReturnType<typeof setTimeout>>()
  private readonly pending = new Set<Promise<unknown>>()
  private readonly failures = new Map<string, string>()
  private stopping = false

  constructor(ctx: Context, config: Config = {}) {
    const root = archiveRoot(config)
    super(ctx, basicCompactionConfig(config))
    this.offload = config
    this.archives = new ArchiveStore(root)

    const schedule = (session: Session) => {
      if (this.stopping || session.header.origin === 'subagent' || this.timers.has(session)) return
      const timer = setTimeout(() => {
        this.timers.delete(session)
        const operation = this.capture(session).catch(() => { /* status records the failure; barriers retry */ })
        this.pending.add(operation)
        void operation.finally(() => this.pending.delete(operation))
      }, BACKGROUND_DEBOUNCE_MS)
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
      this.archives.close(session.id)
    })

    const disposeTools = registerHistoryTools(ctx, {
      archives: this.archives,
      syncSession: (session, signal) => this.capture(session, signal),
    })

    ctx.effect(() => async () => {
      this.stopping = true
      disposeTools()
      const sessions = [...this.timers.keys()]
      for (const timer of this.timers.values()) clearTimeout(timer)
      this.timers.clear()
      await Promise.allSettled([...this.pending, ...sessions.map(session => this.capture(session))])
      await this.archives.drain()
      this.archives.close()
    })

    for (const agent of ctx.agents.list()) schedule(agent.session)

    ctx.commands.register({
      name: 'context-offload',
      description: 'Show session context archive and compaction status',
      handler: async ({ agent, rawInput, signal }) => {
        if (rawInput.trim().length > 0) {
          return { kind: 'error', text: 'Usage: /context-offload (no arguments)' }
        }
        if (agent.session.header.origin === 'subagent') {
          return { kind: 'error', text: 'Context offload targets main-agent sessions.' }
        }
        const status = await this.archives.verify(agent.session.id)
        const target = resolveRouteTarget(agent)
        let contextWindow: number | null = null
        if (target) {
          try {
            const resolved = await this.ctx.llm.resolveModelInfo(target.provider, target.model, signal)
            contextWindow = resolved.context?.contextWindow ?? null
          } catch {
            // Leave contextWindow null when the adapter cannot resolve the route yet.
          }
        }
        const thresholdPct = `${Math.round(this.config.thresholdRatio * 100)}%`
        const retainDesc = 'retainRatio' in this.config && typeof this.config.retainRatio === 'number'
          ? `${Math.round(this.config.retainRatio * 100)}%`
          : ('retainTokens' in this.config && typeof this.config.retainTokens === 'number' ? `${this.config.retainTokens} tok` : 'default')
        const windowShort = contextWindow ? `${Math.round(contextWindow / 1000)}k` : 'unknown window'
        const archivePart = status
          ? `archived ${status.eventCount} events (#${status.throughId}), ${status.compactionCount} compact`
          : 'no archive yet'
        const modelPart = target ? `${target.model} (${windowShort}, compact @ ${thresholdPct})` : `compact @ ${thresholdPct}`
        const err = this.failures.get(agent.session.id) ?? null

        const summary = [
          archivePart,
          modelPart,
          ...(err ? [`error: ${err}`] : []),
        ].join(' | ')

        const lines = [
          summary,
          `archive: ${status ? `${status.eventCount} events (through #${status.throughId}), ${status.compactionCount} compact` : 'not initialized'}`,
          `model: ${target ? `${target.provider}/${target.model}` : 'unresolved'} (window ${contextWindow ?? 'unknown'}, brief ~${briefTargetTokens(contextWindow ?? undefined)} tok)`,
          `policy: threshold ${thresholdPct}, retain ${retainDesc}, headroom ${this.config.headroomTokens} tok`,
          `db: ${status?.dbPath ?? this.archives.dbPathFor(agent.session.id)}`,
        ]
        if (err) lines.push(`error: ${err}`)

        return { kind: 'success', text: lines.join('\n') }
      },
    })
  }

  private async capture(session: Session, signal?: AbortSignal, verifyIntegrity = false) {
    try {
      const observation = await this.ctx.sessionQuery.observeSession(session.id, {
        projectionMode: 'none',
        ...(signal ? { signal } : {}),
      })
      try {
        const cut = await this.archives.ensure(session.id, observation.events, signal, { verifyIntegrity })
        this.failures.delete(session.id)
        return { ...cut, events: observation.events }
      } finally {
        observation[Symbol.dispose]()
      }
    } catch (error) {
      this.failures.set(session.id, error instanceof Error ? error.message : String(error))
      throw error
    }
  }

  override async compactRegion(start: SessionSeq, end: SessionSeq, agent: Agent, signal?: AbortSignal): Promise<CompactionResult> {
    const result = await super.compactRegion(start, end, agent, signal)
    await this.ctx.sessions.flush(agent.session)
    return result
  }

  protected override async summarize(
    input: { readonly messages: readonly Message[]; readonly tools?: readonly ToolSchema[] },
    agent: Agent,
    signal?: AbortSignal,
  ): Promise<BriefResult> {
    if (agent.session.header.origin === 'subagent') return super.summarize(input, agent, signal)

    // Full SQLite integrity & hash-chain verification before committing any history replacement.
    const cut = await this.capture(agent.session, signal, true)

    const configuredTarget = this.config.summarizationProvider.length > 0 && this.config.summarizationModel.length > 0
      ? { provider: this.config.summarizationProvider, model: this.config.summarizationModel }
      : undefined
    const target = configuredTarget ?? resolveRouteTarget(agent)
    if (!target) throw new Error('No provider/model available for execution brief')

    let contextWindow: number | undefined
    try {
      const info = await this.ctx.llm.resolveModelInfo(target.provider, target.model, signal)
      contextWindow = info.context?.contextWindow
    } catch {
      // Fall back to default soft target if adapter metadata is unavailable during manual compaction.
    }

    const targetTokens = briefTargetTokens(contextWindow)
    const callMaxTokens = this.config.maxTokens
    signal?.throwIfAborted()

    const assembler = new BlockAssembler()
    for await (const chunk of this.ctx.llm.stream({
      provider: target.provider,
      model: target.model,
      messages: [
        ...input.messages,
        { role: 'user', content: [{ type: 'text', text: briefInstruction(targetTokens) }] },
      ],
      ...(input.tools === undefined ? {} : { tools: [...input.tools] }),
      toolHistory: agent.session.toolHistory(),
      maxTokens: callMaxTokens,
      sessionId: agent.session.id,
      purpose: 'compaction',
      ...(signal ? { signal } : {}),
    })) assembler.push(chunk)

    signal?.throwIfAborted()

    const finish = assembler.finish
    if (finish.kind === 'error' || finish.kind === 'aborted') {
      throw LlmError.fromFailure(finish.failure)
    }
    if (finish.kind === 'max-tokens') {
      throw Object.assign(new Error('Execution brief truncated at the token cap (incomplete checkpoint)'), { code: 'MAX_TOKENS' })
    }

    const blocks = assembler.blocks()
    if (contentHasImage(blocks)) {
      throw new LlmError('Execution brief cannot contain image output', 'UNSUPPORTED_CONTENT')
    }

    // Only the text survives into history, so reasoning is neither counted nor retained.
    const rawOutput = blocks.filter(block => block.type === 'text')
    const body = rawOutput.map(block => block.text).join('\n')
    validateBrief(body)

    const text = frameBrief(body, cut)
    await this.archives.saveCompact(cut, {
      provider: target.provider,
      model: target.model,
      body,
      framedText: text,
    })

    return {
      summary: [{ type: 'text' as const, text }],
      rawOutput,
      llmStreamCall: true as const,
      provider: target.provider,
      model: target.model,
      maxTokens: callMaxTokens,
      ...(assembler.usage === undefined ? {} : { usage: assembler.usage }),
    }
  }
}

export async function apply(ctx: Context, config: Config = {}): Promise<void> {
  const fiber = ctx.plugin(class extends ContextOffloadEngine {
    constructor(child: Context) { super(child, config) }
  })
  await fiber.await()
}
