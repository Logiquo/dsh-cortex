import type { Context } from '@deepseek-ai/cordis'
import type { Session } from '@deepseek-ai/dsh-session'
import { defineTool } from '@deepseek-ai/dsh-tools'
import type { ToolActivationContext } from '@deepseek-ai/dsh-tools'
import type { ArchiveStore, LexicalMatch } from './archive.js'

const MAX_SEARCH_LIMIT = 30
const DEFAULT_SEARCH_LIMIT = 8
const MIN_SNIPPET_CHARS = 80
const MAX_SNIPPET_CHARS = 1000
const DEFAULT_SNIPPET_CHARS = 300

const MAX_READ_EVENTS = 20
const MIN_READ_CHARS = 200
const MAX_READ_CHARS = 20000
const DEFAULT_READ_CHARS = 6000

export interface SemanticSearchHit {
  cardIdx: number
  startId: number
  endId: number
  kind: string
  score: number
  micro: string
}

export interface SemanticSearchProvider {
  semanticSearch(
    sessionId: string,
    query: string,
    options: { topK: number; fromId?: number; toId?: number; signal?: AbortSignal },
  ): Promise<SemanticSearchHit[]>
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

function getSemanticProvider(ctx: Context): SemanticSearchProvider | undefined {
  const getter = (ctx as unknown as { get?: (name: string) => unknown }).get
  const candidate = (typeof getter === 'function' ? getter.call(ctx, 'contextGuard') : undefined) as
    | Partial<SemanticSearchProvider>
    | undefined
  if (candidate && typeof candidate.semanticSearch === 'function') {
    return candidate as SemanticSearchProvider
  }
  return undefined
}

function shouldActivate({ agent }: ToolActivationContext): boolean {
  return agent.session.header.origin !== 'subagent' && agent.session.surface.replaceGeneration > 0
}

export function registerHistoryTools(ctx: Context, deps: HistoryToolsDeps): () => void {
  const disposeSearch = ctx.tools.register(
    defineTool({
      name: 'history_search',
      description:
        'Search earlier canonical history of the current session (including compacted turns). Defaults to `hybrid` mode (exact/regex match on raw events + semantic similarity on micro-cards when dsh-context-guard is active). Returns event ids and concise snippets/summaries; call `history_read` to inspect full text around any matched id.',
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
            'Search mode: "hybrid" (default, combines exact/regex + semantic when available), "lexical" (exact/regex only), or "semantic" (vector similarity only).',
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
          description: `Max lexical matches to return (1..${MAX_SEARCH_LIMIT}, default ${DEFAULT_SEARCH_LIMIT}). In hybrid mode, also returns up to min(limit, 8) non-overlapping semantic matches.`,
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
            lexicalMatches: {
              type: 'array',
              required: true,
              items: {
                type: 'object',
                additionalProperties: false,
                properties: {
                  id: { type: 'integer', required: true },
                  type: { type: 'string', required: true },
                  snippet: { type: 'string', required: true },
                  textLength: { type: 'integer', required: true },
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
          if (value.lexicalMatches.length === 0 && value.semanticMatches.length === 0) {
            return [{
              type: 'text',
              text: `No matches for ${JSON.stringify(args.query)} (mode=${value.modeUsed}, events ${value.scannedFrom}..${value.scannedTo}, ${value.totalEvents} total events).`,
            }]
          }
          const sections: string[] = [
            `Search ${JSON.stringify(args.query)} (mode=${value.modeUsed}, events ${value.scannedFrom}..${value.scannedTo}):`,
          ]
          if (value.lexicalMatches.length > 0) {
            const head = `Exact matches (${value.lexicalMatches.length} of ${value.totalLexicalMatches}${value.hasMoreLexical ? ', more available' : ''}):`
            const body = value.lexicalMatches
              .map(m => `[id ${m.id} | ${m.type} | ${m.textLength} chars]\n${m.snippet}`)
              .join('\n\n')
            sections.push(`${head}\n${body}`)
          }
          if (value.semanticMatches.length > 0) {
            const head = `Semantic matches (${value.semanticMatches.length} related step${value.semanticMatches.length === 1 ? '' : 's'}):`
            const body = value.semanticMatches
              .map(s => `[ids ${s.startId}..${s.endId} | card #${s.cardIdx} | ${s.kind} | score=${s.score.toFixed(2)}] ${s.micro}`)
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
        const provider = getSemanticProvider(ctx)
        if (requestedMode === 'semantic' && !provider) {
          throw new Error('Semantic history search requires dsh-context-guard to be installed and active')
        }
        const modeUsed: 'hybrid' | 'lexical' | 'semantic' =
          requestedMode === 'hybrid' && !provider ? 'lexical' : requestedMode

        const limit = clamp(args.limit ?? DEFAULT_SEARCH_LIMIT, 1, MAX_SEARCH_LIMIT)
        const snippetChars = clamp(args.snippetChars ?? DEFAULT_SNIPPET_CHARS, MIN_SNIPPET_CHARS, MAX_SNIPPET_CHARS)

        await deps.syncSession(exec.agent.session, exec.signal)
        const sessionId = exec.agent.session.id

        let lexicalMatches: LexicalMatch[] = []
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
          lexicalMatches = lex.matches
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

        let semanticMatches: SemanticSearchHit[] = []
        if ((modeUsed === 'semantic' || modeUsed === 'hybrid') && provider) {
          const semQuery = args.regex === true ? stripRegexMeta(query) : query.trim()
          if (semQuery.length > 0) {
            const topK = modeUsed === 'hybrid' ? Math.min(limit, 8) : limit
            const rawHits = await provider.semanticSearch(sessionId, semQuery, {
              topK: topK + lexicalMatches.length,
              ...(args.fromId !== undefined ? { fromId: args.fromId } : {}),
              ...(args.toId !== undefined ? { toId: args.toId } : {}),
              signal: exec.signal,
            })
            const lexIds = new Set(lexicalMatches.map(m => m.id))
            semanticMatches = rawHits
              .filter(hit => {
                if (modeUsed !== 'hybrid') return true
                for (const id of lexIds) {
                  if (id >= hit.startId && id <= hit.endId) return false
                }
                return true
              })
              .slice(0, topK)
          }
        }

        return {
          modeUsed,
          lexicalMatches,
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
        'Read canonical event text from the current session by exact event id range (up to 20 events per call, bounded by maxChars with charOffset pagination). Use after `history_search` to inspect full context around a hit.',
      parameters: {
        fromId: {
          type: 'integer',
          required: true,
          description: 'Inclusive starting event id.',
        },
        toId: {
          type: 'integer',
          description: `Optional inclusive ending event id (defaults to fromId; capped at ${MAX_READ_EVENTS} events per call).`,
        },
        maxChars: {
          type: 'integer',
          description: `Maximum characters to return (${MIN_READ_CHARS}..${MAX_READ_CHARS}, default ${DEFAULT_READ_CHARS}).`,
        },
        charOffset: {
          type: 'integer',
          description: 'Character offset into the formatted range for paginating long events. Defaults to 0.',
        },
      },
      output: {
        schema: {
          type: 'object',
          additionalProperties: false,
          properties: {
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
          const footer = value.truncated
            ? `\n\n[Truncated at ${value.nextCharOffset} / ${value.totalChars} chars; pass charOffset=${value.nextCharOffset} to continue.]`
            : ''
          return [{
            type: 'text',
            text: `Events ${value.fromId}..${value.toId} (${value.totalChars} chars):\n\n${value.content}${footer}`,
          }]
        },
      },
      isConcurrencySafe: () => true,
      async execute(args, exec) {
        if (!exec.agent) throw new Error('history_read requires an owning agent session')
        const maxChars = clamp(args.maxChars ?? DEFAULT_READ_CHARS, MIN_READ_CHARS, MAX_READ_CHARS)
        const charOffset = Math.max(args.charOffset ?? 0, 0)

        await deps.syncSession(exec.agent.session, exec.signal)
        const sessionId = exec.agent.session.id
        const totalEvents = await deps.archives.getEventCount(sessionId)
        if (totalEvents <= 0) {
          return { content: '', fromId: 0, toId: -1, totalChars: 0, truncated: false, nextCharOffset: null }
        }

        const fromId = clamp(args.fromId, 0, totalEvents - 1)
        const requestedTo = args.toId ?? fromId
        if (requestedTo < fromId) {
          throw new Error(`toId (${requestedTo}) must be >= fromId (${fromId})`)
        }
        const toId = clamp(Math.min(requestedTo, fromId + MAX_READ_EVENTS - 1), fromId, totalEvents - 1)

        const rows = await deps.archives.readEventRange(sessionId, fromId, toId)
        const full = rows
          .map(row => `=== event ${row.id}: ${row.type} ===\n${row.text.length > 0 ? row.text : '(no text content)'}`)
          .join('\n\n')
        const totalChars = full.length
        const slice = full.slice(charOffset, charOffset + maxChars)
        const endOffset = charOffset + slice.length
        const truncated = endOffset < totalChars

        return {
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
          ? `Read history events ${args.fromId}..${args.toId}`
          : `Read history event ${args.fromId}`,
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
