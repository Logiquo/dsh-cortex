import type { Context } from '@deepseek-ai/cordis'
import type { Session } from '@deepseek-ai/dsh-session'
import { defineTool } from '@deepseek-ai/dsh-tools'
import type { ToolActivationContext } from '@deepseek-ai/dsh-tools'
import type { ArchiveStore, LexicalMatch, StoredEventRow } from './archive.js'

const MAX_SEARCH_LIMIT = 30
const DEFAULT_SEARCH_LIMIT = 8
const MIN_SNIPPET_CHARS = 80
const MAX_SNIPPET_CHARS = 1000
const DEFAULT_SNIPPET_CHARS = 300

const MAX_READ_EVENTS_FULL = 20
const MAX_READ_EVENTS_SUMMARY = 200
const MIN_READ_CHARS = 200
const MAX_READ_CHARS = 20000
const DEFAULT_READ_CHARS = 6000
const SUMMARY_EDGE_CHARS = 150

export type HistoryScope = 'compacted' | 'in-context' | 'mixed'

export interface SemanticSearchHit {
  cardIdx: number
  startId: number
  endId: number
  kind: string
  score: number
  micro: string
}

export interface SummaryCardItem {
  cardIdx: number
  startId: number
  endId: number
  kind: string
  micro: string
}

export interface SummaryRangeResult {
  throughEventId: number
  cards: SummaryCardItem[]
}

export interface SummaryLexicalSearchOptions {
  query: string
  regex?: boolean
  caseSensitive?: boolean
  kinds?: string[]
  fromId?: number
  toId?: number
  order?: 'newest' | 'oldest'
  limit: number
  signal?: AbortSignal
}

export interface ContextGuardProvider {
  semanticSearch(
    sessionId: string,
    query: string,
    options: { topK: number; fromId?: number; toId?: number; signal?: AbortSignal },
  ): Promise<SemanticSearchHit[]>
  lexicalSearch?(sessionId: string, options: SummaryLexicalSearchOptions): Promise<SummaryCardItem[]>
  listSummaries?(sessionId: string, options?: { fromId?: number; toId?: number }): Promise<SummaryRangeResult>
}

export interface HistoryToolsDeps {
  archives: ArchiveStore
  syncSession: (session: Session, signal?: AbortSignal) => Promise<unknown>
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max)
}

function stripRegexMeta(pattern: string): string {
  return pattern
    .replace(/\\[dwsbBDWS]/g, ' ')
    .replace(/[\\^$.*+?()[\]{}|]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

function getGuardProvider(ctx: Context): ContextGuardProvider | undefined {
  const getter = (ctx as unknown as { get?: (name: string) => unknown }).get
  const candidate = (typeof getter === 'function' ? getter.call(ctx, 'contextGuard') : undefined) as
    | Partial<ContextGuardProvider>
    | undefined
  if (candidate && typeof candidate.semanticSearch === 'function') {
    return candidate as ContextGuardProvider
  }
  return undefined
}

/** Always expose history tools on main-agent sessions (covers both compacted and in-context history). */
function shouldActivate({ agent }: ToolActivationContext): boolean {
  return agent.session.header.origin !== 'subagent'
}

function scopeForId(id: number, compactedThroughId: number): 'compacted' | 'in-context' {
  return id <= compactedThroughId ? 'compacted' : 'in-context'
}

function scopeForRange(startId: number, endId: number, compactedThroughId: number): HistoryScope {
  if (endId <= compactedThroughId) return 'compacted'
  if (startId > compactedThroughId) return 'in-context'
  return 'mixed'
}

function formatScopeTag(scope: HistoryScope): string {
  return scope === 'mixed' ? 'compacted+in-context' : scope
}

function formatCompactedHeader(compactedThroughId: number): string {
  return compactedThroughId >= 0 ? `compacted<=${compactedThroughId}` : 'no compaction yet'
}

function formatHeadTailSummary(text: string, edgeChars = SUMMARY_EDGE_CHARS): string {
  if (text.length === 0) return '(no text content)'
  if (text.length <= edgeChars * 2 + 40) return text
  const omitted = text.length - edgeChars * 2
  return `${text.slice(0, edgeChars)}\n... [${omitted} chars omitted] ...\n${text.slice(-edgeChars)}`
}

function formatTruncatedEvent(row: StoredEventRow, compactedThroughId: number): string {
  const scope = scopeForId(row.id, compactedThroughId)
  return `=== event ${row.id} [${scope}]: ${row.type} (${row.textLen} chars) ===\n${formatHeadTailSummary(row.text)}`
}

function formatGuardSummaryCard(card: SummaryCardItem, compactedThroughId: number): string {
  const scope = formatScopeTag(scopeForRange(card.startId, card.endId, compactedThroughId))
  return `[ids ${card.startId}..${card.endId} | ${scope} | card #${card.cardIdx} | ${card.kind}] ${card.micro}`
}

function mapEventTypesToCardKinds(eventTypes: readonly string[] | undefined): string[] | undefined {
  if (!eventTypes || eventTypes.length === 0) return undefined
  const kinds = new Set<string>()
  for (const type of eventTypes) {
    if (type === 'user/message') kinds.add('constraint')
    else if (type === 'tool/call' || type === 'tool/result') kinds.add('action')
    else if (type === 'assistant/message') kinds.add('assistant')
  }
  return [...kinds]
}

function overlapsRange(startA: number, endA: number, startB: number, endB: number): boolean {
  return startA <= endB && startB <= endA
}

export function registerHistoryTools(ctx: Context, deps: HistoryToolsDeps): () => void {
  const disposeSearch = ctx.tools.register(
    defineTool({
      name: 'history_search',
      description:
        'Search canonical history of the current session across both [compacted] and [in-context] turns. Defaults to `hybrid` mode: exact/regex match on raw full-text events, plus exact/regex match and semantic similarity on step summaries when dsh-context-guard is active. Call `history_read` to inspect full text or browse summaries around any matched id.',
      parameters: {
        query: {
          type: 'string',
          required: true,
          description: 'Keyword, natural-language concept, or JS regex (when `regex` is true).',
        },
        mode: {
          type: 'string',
          enum: ['hybrid', 'lexical', 'semantic'],
          description:
            'Search mode: "hybrid" (default, combines full-text lexical/regex + summary lexical/regex + semantic when available), "lexical" (exact/regex over full-text and summaries when available), or "semantic" (vector similarity only).',
        },
        regex: {
          type: 'boolean',
          description:
            'Treat `query` as a JS regular expression for lexical matching (metacharacters are stripped automatically for the semantic pass in hybrid mode). Defaults to false.',
        },
        caseSensitive: {
          type: 'boolean',
          description: 'Case-sensitive lexical matching. Defaults to false.',
        },
        eventTypes: {
          type: 'array',
          items: { type: 'string' },
          description: 'Optional canonical event types to filter lexical matches (e.g. "user/message", "assistant/message", "tool/result").',
        },
        fromId: {
          type: 'integer',
          description: 'Optional inclusive lower bound on event id.',
        },
        toId: {
          type: 'integer',
          description: 'Optional inclusive upper bound on event id.',
        },
        order: {
          type: 'string',
          enum: ['newest', 'oldest'],
          description: 'Ordering for lexical matches: "newest" (default) or "oldest". Semantic matches are always ranked by descending similarity score.',
        },
        limit: {
          type: 'integer',
          description: `Max lexical matches to return (1..${MAX_SEARCH_LIMIT}, default ${DEFAULT_SEARCH_LIMIT}). In hybrid mode, also returns up to min(limit, 8) non-overlapping summary/semantic matches.`,
        },
        snippetChars: {
          type: 'integer',
          description: `Max characters per lexical snippet (${MIN_SNIPPET_CHARS}..${MAX_SNIPPET_CHARS}, default ${DEFAULT_SNIPPET_CHARS}).`,
        },
      },
      output: {
        schema: {
          type: 'object',
          additionalProperties: false,
          properties: {
            modeUsed: { type: 'string', enum: ['hybrid', 'lexical', 'semantic'], required: true },
            compactedThroughId: { type: 'integer', required: true },
            lexicalMatches: {
              type: 'array',
              required: true,
              items: {
                type: 'object',
                additionalProperties: false,
                properties: {
                  id: { type: 'integer', required: true },
                  type: { type: 'string', required: true },
                  scope: { type: 'string', enum: ['compacted', 'in-context'], required: true },
                  snippet: { type: 'string', required: true },
                  textLength: { type: 'integer', required: true },
                },
              },
            },
            summaryLexicalMatches: {
              type: 'array',
              required: true,
              items: {
                type: 'object',
                additionalProperties: false,
                properties: {
                  cardIdx: { type: 'integer', required: true },
                  startId: { type: 'integer', required: true },
                  endId: { type: 'integer', required: true },
                  kind: { type: 'string', required: true },
                  scope: { type: 'string', enum: ['compacted', 'in-context', 'mixed'], required: true },
                  micro: { type: 'string', required: true },
                },
              },
            },
            semanticMatches: {
              type: 'array',
              required: true,
              items: {
                type: 'object',
                additionalProperties: false,
                properties: {
                  cardIdx: { type: 'integer', required: true },
                  startId: { type: 'integer', required: true },
                  endId: { type: 'integer', required: true },
                  kind: { type: 'string', required: true },
                  scope: { type: 'string', enum: ['compacted', 'in-context', 'mixed'], required: true },
                  score: { type: 'number', required: true },
                  micro: { type: 'string', required: true },
                },
              },
            },
            totalLexicalMatches: { type: 'integer', required: true },
            hasMoreLexical: { type: 'boolean', required: true },
            scannedFrom: { type: 'integer', required: true },
            scannedTo: { type: 'integer', required: true },
            totalEvents: { type: 'integer', required: true },
          },
        },
        render: (args, value) => {
          const compactInfo = formatCompactedHeader(value.compactedThroughId)
          if (
            value.lexicalMatches.length === 0 &&
            value.summaryLexicalMatches.length === 0 &&
            value.semanticMatches.length === 0
          ) {
            return [{
              type: 'text',
              text: `No matches for ${JSON.stringify(args.query)} (mode=${value.modeUsed}, events ${value.scannedFrom}..${value.scannedTo}, ${compactInfo}, ${value.totalEvents} total events).`,
            }]
          }
          const sections: string[] = [
            `Search ${JSON.stringify(args.query)} (mode=${value.modeUsed}, events ${value.scannedFrom}..${value.scannedTo}, ${compactInfo}):`,
          ]
          if (value.lexicalMatches.length > 0) {
            const head = `Exact matches in full text (${value.lexicalMatches.length} of ${value.totalLexicalMatches}${value.hasMoreLexical ? ', more available' : ''}):`
            const body = value.lexicalMatches
              .map(m => `[id ${m.id} | ${m.scope} | ${m.type} | ${m.textLength} chars]\n${m.snippet}`)
              .join('\n\n')
            sections.push(`${head}\n${body}`)
          }
          if (value.summaryLexicalMatches.length > 0) {
            const n = value.summaryLexicalMatches.length
            const head = `Exact matches in summaries (${n} additional step${n === 1 ? '' : 's'}):`
            const body = value.summaryLexicalMatches
              .map(s => `[ids ${s.startId}..${s.endId} | ${formatScopeTag(s.scope)} | card #${s.cardIdx} | ${s.kind}] ${s.micro}`)
              .join('\n')
            sections.push(`${head}\n${body}`)
          }
          if (value.semanticMatches.length > 0) {
            const n = value.semanticMatches.length
            const head = `Semantic matches (${n} related step${n === 1 ? '' : 's'}):`
            const body = value.semanticMatches
              .map(s => `[ids ${s.startId}..${s.endId} | ${formatScopeTag(s.scope)} | card #${s.cardIdx} | ${s.kind} | score=${s.score.toFixed(2)}] ${s.micro}`)
              .join('\n')
            sections.push(`${head}\n${body}`)
          }
          return [{ type: 'text', text: sections.join('\n\n') }]
        },
      },
      isConcurrencySafe: () => true,
      async execute(args, exec) {
        if (!exec.agent) throw new Error('history_search requires an owning agent session')
        const query = args.query
        if (query.length === 0) throw new Error('history_search requires a non-empty query')

        const requestedMode = args.mode ?? 'hybrid'
        const provider = getGuardProvider(ctx)
        if (requestedMode === 'semantic' && !provider) {
          throw new Error('Semantic history search requires dsh-context-guard to be installed and active')
        }
        const modeUsed: 'hybrid' | 'lexical' | 'semantic' =
          requestedMode === 'hybrid' && !provider ? 'lexical' : requestedMode

        const limit = clamp(args.limit ?? DEFAULT_SEARCH_LIMIT, 1, MAX_SEARCH_LIMIT)
        const snippetChars = clamp(args.snippetChars ?? DEFAULT_SNIPPET_CHARS, MIN_SNIPPET_CHARS, MAX_SNIPPET_CHARS)

        await deps.syncSession(exec.agent.session, exec.signal)
        const sessionId = exec.agent.session.id
        const compactedThroughId = await deps.archives.getCompactedThroughId(sessionId)

        let rawLexicalMatches: LexicalMatch[] = []
        let totalLexicalMatches = 0
        let scannedFrom = 0
        let scannedTo = -1
        let totalEvents = 0

        if (modeUsed === 'lexical' || modeUsed === 'hybrid') {
          const lex = await deps.archives.searchEvents(sessionId, {
            query,
            regex: args.regex === true,
            caseSensitive: args.caseSensitive === true,
            ...(args.eventTypes ? { eventTypes: args.eventTypes } : {}),
            ...(args.fromId !== undefined ? { fromId: args.fromId } : {}),
            ...(args.toId !== undefined ? { toId: args.toId } : {}),
            order: args.order ?? 'newest',
            limit,
            snippetChars,
            signal: exec.signal,
          })
          rawLexicalMatches = lex.matches
          totalLexicalMatches = lex.totalMatches
          scannedFrom = lex.scannedFrom
          scannedTo = lex.scannedTo
          totalEvents = lex.totalEvents
        } else {
          totalEvents = await deps.archives.getEventCount(sessionId)
          if (totalEvents > 0) {
            scannedFrom = clamp(args.fromId ?? 0, 0, totalEvents - 1)
            scannedTo = clamp(args.toId ?? totalEvents - 1, 0, totalEvents - 1)
          }
        }

        const lexicalMatches = rawLexicalMatches.map(m => ({
          ...m,
          scope: scopeForId(m.id, compactedThroughId),
        }))

        const lexIds = new Set(lexicalMatches.map(m => m.id))

        let summaryLexicalMatches: Array<SummaryCardItem & { scope: HistoryScope }> = []
        if ((modeUsed === 'lexical' || modeUsed === 'hybrid') && provider?.lexicalSearch) {
          const mappedKinds = mapEventTypesToCardKinds(args.eventTypes)
          if (!args.eventTypes || (mappedKinds && mappedKinds.length > 0)) {
            const summaryCap = Math.min(limit, 8)
            const rawSumHits = await provider.lexicalSearch(sessionId, {
              query,
              regex: args.regex === true,
              caseSensitive: args.caseSensitive === true,
              ...(mappedKinds ? { kinds: mappedKinds } : {}),
              ...(args.fromId !== undefined ? { fromId: args.fromId } : {}),
              ...(args.toId !== undefined ? { toId: args.toId } : {}),
              order: args.order ?? 'newest',
              limit: summaryCap + lexicalMatches.length,
              ...(exec.signal ? { signal: exec.signal } : {}),
            })
            summaryLexicalMatches = rawSumHits
              .filter(hit => {
                for (const id of lexIds) {
                  if (id >= hit.startId && id <= hit.endId) return false
                }
                return true
              })
              .slice(0, summaryCap)
              .map(hit => ({
                ...hit,
                scope: scopeForRange(hit.startId, hit.endId, compactedThroughId),
              }))
          }
        }

        let semanticMatches: Array<SemanticSearchHit & { scope: HistoryScope }> = []
        if ((modeUsed === 'semantic' || modeUsed === 'hybrid') && provider) {
          const semQuery = args.regex === true ? stripRegexMeta(query) : query.trim()
          if (semQuery.length > 0) {
            const topK = modeUsed === 'hybrid' ? Math.min(limit, 8) : limit
            const rawHits = await provider.semanticSearch(sessionId, semQuery, {
              topK: topK + lexicalMatches.length + summaryLexicalMatches.length,
              ...(args.fromId !== undefined ? { fromId: args.fromId } : {}),
              ...(args.toId !== undefined ? { toId: args.toId } : {}),
              signal: exec.signal,
            })
            semanticMatches = rawHits
              .filter(hit => {
                if (modeUsed !== 'hybrid') return true
                for (const id of lexIds) {
                  if (id >= hit.startId && id <= hit.endId) return false
                }
                for (const s of summaryLexicalMatches) {
                  if (overlapsRange(hit.startId, hit.endId, s.startId, s.endId)) return false
                }
                return true
              })
              .slice(0, topK)
              .map(hit => ({
                ...hit,
                scope: scopeForRange(hit.startId, hit.endId, compactedThroughId),
              }))
          }
        }

        return {
          modeUsed,
          compactedThroughId,
          lexicalMatches,
          summaryLexicalMatches,
          semanticMatches,
          totalLexicalMatches,
          hasMoreLexical: totalLexicalMatches > lexicalMatches.length,
          scannedFrom,
          scannedTo,
          totalEvents,
        }
      },
      presentCall: args => ({
        card: 'generic',
        title: `Search history: ${args.query}`,
        kind: 'other',
        rawInput: args,
      }),
    }),
    { when: shouldActivate },
  )

  const disposeRead = ctx.tools.register(
    defineTool({
      name: 'history_read',
      description:
        'Read canonical history from the current session by exact event id range (tagged [compacted] or [in-context]). Supports `detail: "full"` (default, up to 20 events with full verbatim text) or `detail: "summary"` (up to 200 events, returning guard micro-card summaries when dsh-context-guard is active or head+tail truncated previews otherwise).',
      parameters: {
        fromId: {
          type: 'integer',
          required: true,
          description: 'Inclusive starting event id.',
        },
        toId: {
          type: 'integer',
          description: `Optional inclusive ending event id (defaults to fromId; capped at ${MAX_READ_EVENTS_FULL} events in "full" mode or ${MAX_READ_EVENTS_SUMMARY} events in "summary" mode).`,
        },
        detail: {
          type: 'string',
          enum: ['full', 'summary'],
          description:
            'Level of detail: "full" (default, verbatim event text) or "summary" (compact timeline summaries for browsing a larger block of history).',
        },
        maxChars: {
          type: 'integer',
          description: `Maximum characters to return (${MIN_READ_CHARS}..${MAX_READ_CHARS}, default ${DEFAULT_READ_CHARS}).`,
        },
        charOffset: {
          type: 'integer',
          description: 'Character offset into the formatted range for paginating long outputs. Defaults to 0.',
        },
      },
      output: {
        schema: {
          type: 'object',
          additionalProperties: false,
          properties: {
            detailUsed: { type: 'string', enum: ['full', 'summary'], required: true },
            summarySource: { type: 'string', enum: ['none', 'guard', 'truncated', 'mixed'], required: true },
            compactedThroughId: { type: 'integer', required: true },
            content: { type: 'string', required: true },
            fromId: { type: 'integer', required: true },
            toId: { type: 'integer', required: true },
            totalChars: { type: 'integer', required: true },
            truncated: { type: 'boolean', required: true },
            nextCharOffset: {
              required: true,
              oneOf: [{ type: 'integer' }, { type: 'null' }],
            },
          },
        },
        render: (_args, value) => {
          const compactInfo = formatCompactedHeader(value.compactedThroughId)
          const meta = value.detailUsed === 'summary'
            ? `detail=summary, source=${value.summarySource}, ${compactInfo}, ${value.totalChars} chars`
            : `detail=full, ${compactInfo}, ${value.totalChars} chars`
          const footer = value.truncated
            ? `\n\n[Truncated at ${value.nextCharOffset} / ${value.totalChars} chars; pass charOffset=${value.nextCharOffset} to continue.]`
            : ''
          return [{
            type: 'text',
            text: `Events ${value.fromId}..${value.toId} (${meta}):\n\n${value.content}${footer}`,
          }]
        },
      },
      isConcurrencySafe: () => true,
      async execute(args, exec) {
        if (!exec.agent) throw new Error('history_read requires an owning agent session')
        const detailUsed: 'full' | 'summary' = args.detail ?? 'full'
        const maxEvents = detailUsed === 'summary' ? MAX_READ_EVENTS_SUMMARY : MAX_READ_EVENTS_FULL
        const maxChars = clamp(args.maxChars ?? DEFAULT_READ_CHARS, MIN_READ_CHARS, MAX_READ_CHARS)
        const charOffset = Math.max(args.charOffset ?? 0, 0)

        await deps.syncSession(exec.agent.session, exec.signal)
        const sessionId = exec.agent.session.id
        const [totalEvents, compactedThroughId] = await Promise.all([
          deps.archives.getEventCount(sessionId),
          deps.archives.getCompactedThroughId(sessionId),
        ])
        if (totalEvents <= 0) {
          return {
            detailUsed,
            summarySource: 'none' as const,
            compactedThroughId,
            content: '',
            fromId: 0,
            toId: -1,
            totalChars: 0,
            truncated: false,
            nextCharOffset: null,
          }
        }

        const fromId = clamp(args.fromId, 0, totalEvents - 1)
        const requestedTo = args.toId ?? fromId
        if (requestedTo < fromId) {
          throw new Error(`toId (${requestedTo}) must be >= fromId (${fromId})`)
        }
        const toId = clamp(Math.min(requestedTo, fromId + maxEvents - 1), fromId, totalEvents - 1)

        let full = ''
        let summarySource: 'none' | 'guard' | 'truncated' | 'mixed' = 'none'

        if (detailUsed === 'full') {
          const rows = await deps.archives.readEventRange(sessionId, fromId, toId)
          full = rows
            .map(row => {
              const scope = scopeForId(row.id, compactedThroughId)
              return `=== event ${row.id} [${scope}]: ${row.type} ===\n${row.text.length > 0 ? row.text : '(no text content)'}`
            })
            .join('\n\n')
        } else {
          const provider = getGuardProvider(ctx)
          if (provider?.listSummaries) {
            const { throughEventId, cards } = await provider.listSummaries(sessionId, { fromId, toId })
            if (throughEventId >= fromId) {
              const parts: string[] = []
              if (cards.length > 0) {
                parts.push(cards.map(c => formatGuardSummaryCard(c, compactedThroughId)).join('\n'))
              }
              if (toId > throughEventId) {
                const tailRows = await deps.archives.readEventRange(sessionId, throughEventId + 1, toId)
                if (tailRows.length > 0) {
                  parts.push(tailRows.map(r => formatTruncatedEvent(r, compactedThroughId)).join('\n\n'))
                  summarySource = cards.length > 0 ? 'mixed' : 'truncated'
                } else {
                  summarySource = 'guard'
                }
              } else {
                summarySource = 'guard'
              }
              full = parts.join('\n\n')
            } else {
              const rows = await deps.archives.readEventRange(sessionId, fromId, toId)
              full = rows.map(r => formatTruncatedEvent(r, compactedThroughId)).join('\n\n')
              summarySource = 'truncated'
            }
          } else {
            const rows = await deps.archives.readEventRange(sessionId, fromId, toId)
            full = rows.map(r => formatTruncatedEvent(r, compactedThroughId)).join('\n\n')
            summarySource = 'truncated'
          }
        }

        const totalChars = full.length
        const slice = full.slice(charOffset, charOffset + maxChars)
        const endOffset = charOffset + slice.length
        const truncated = endOffset < totalChars

        return {
          detailUsed,
          summarySource,
          compactedThroughId,
          content: slice,
          fromId,
          toId,
          totalChars,
          truncated,
          nextCharOffset: truncated ? endOffset : null,
        }
      },
      presentCall: args => ({
        card: 'generic',
        title: args.toId !== undefined && args.toId !== args.fromId
          ? `Read history events ${args.fromId}..${args.toId}${args.detail === 'summary' ? ' (summary)' : ''}`
          : `Read history event ${args.fromId}${args.detail === 'summary' ? ' (summary)' : ''}`,
        kind: 'other',
        rawInput: args,
      }),
    }),
    { when: shouldActivate },
  )

  return () => {
    disposeSearch()
    disposeRead()
  }
}
