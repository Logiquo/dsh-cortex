import type { Context } from '@deepseek-ai/cordis'
import { extractSessionEventText } from '@deepseek-ai/dsh-session-query'
import { defineTool } from '@deepseek-ai/dsh-tools'
import type { ToolActivationContext } from '@deepseek-ai/dsh-tools'

const MAX_SEARCH_LIMIT = 30
const DEFAULT_SEARCH_LIMIT = 10
const MIN_SNIPPET_CHARS = 80
const MAX_SNIPPET_CHARS = 1000
const DEFAULT_SNIPPET_CHARS = 300

const MAX_READ_EVENTS = 20
const MIN_READ_CHARS = 200
const MAX_READ_CHARS = 20000
const DEFAULT_READ_CHARS = 6000

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max)
}

function buildSnippet(text: string, matchIndex: number, matchLength: number, maxChars: number): string {
  if (text.length <= maxChars) return text
  const half = Math.max(Math.floor((maxChars - matchLength) / 2), 0)
  let start = Math.max(0, matchIndex - half)
  let end = Math.min(text.length, start + maxChars)
  if (end - start < maxChars) start = Math.max(0, end - maxChars)
  const prefix = start > 0 ? '…' : ''
  const suffix = end < text.length ? '…' : ''
  return `${prefix}${text.slice(start, end)}${suffix}`
}

function shouldActivate({ agent }: ToolActivationContext): boolean {
  return agent.session.header.origin !== 'subagent' && agent.session.surface.replaceGeneration > 0
}

export function registerHistoryTools(ctx: Context): () => void {
  const disposeSearch = ctx.tools.register(
    defineTool({
      name: 'history_search',
      description:
        'Search earlier canonical events of the current session (including history compacted out of the active context). Returns matching event seqs and bounded snippets, newest-first by default. Use `history_read` to inspect full text around a matched seq.',
      parameters: {
        query: {
          type: 'string',
          required: true,
          description: 'Keyword substring or regular expression (when `regex` is true) to search for.',
        },
        regex: {
          type: 'boolean',
          description: 'Treat `query` as a JavaScript regular expression instead of a literal substring. Defaults to false.',
        },
        caseSensitive: {
          type: 'boolean',
          description: 'Whether matching is case-sensitive. Defaults to false.',
        },
        eventTypes: {
          type: 'array',
          items: { type: 'string' },
          description: 'Optional event types to include (e.g. "user/message", "assistant/message", "tool/result").',
        },
        fromSeq: {
          type: 'integer',
          description: 'Optional inclusive lower bound on event seq.',
        },
        toSeq: {
          type: 'integer',
          description: 'Optional inclusive upper bound on event seq.',
        },
        order: {
          type: 'string',
          enum: ['newest', 'oldest'],
          description: 'Result ordering: "newest" (default, surfaces latest corrections first) or "oldest".',
        },
        limit: {
          type: 'integer',
          description: `Maximum matches to return (1..${MAX_SEARCH_LIMIT}, default ${DEFAULT_SEARCH_LIMIT}).`,
        },
        snippetChars: {
          type: 'integer',
          description: `Maximum characters per match snippet (${MIN_SNIPPET_CHARS}..${MAX_SNIPPET_CHARS}, default ${DEFAULT_SNIPPET_CHARS}).`,
        },
      },
      output: {
        schema: {
          type: 'object',
          additionalProperties: false,
          properties: {
            matches: {
              type: 'array',
              required: true,
              items: {
                type: 'object',
                additionalProperties: false,
                properties: {
                  seq: { type: 'integer', required: true },
                  type: { type: 'string', required: true },
                  snippet: { type: 'string', required: true },
                  textLength: { type: 'integer', required: true },
                },
              },
            },
            totalMatches: { type: 'integer', required: true },
            hasMore: { type: 'boolean', required: true },
            scannedFrom: { type: 'integer', required: true },
            scannedTo: { type: 'integer', required: true },
            totalEvents: { type: 'integer', required: true },
          },
        },
        render: (args, value) => {
          if (value.matches.length === 0) {
            return [{
              type: 'text',
              text: `No matches for ${JSON.stringify(args.query)} in events ${value.scannedFrom}..${value.scannedTo} (${value.totalEvents} total events).`,
            }]
          }
          const header = `Found ${value.totalMatches} match${value.totalMatches === 1 ? '' : 'es'} for ${JSON.stringify(args.query)} in events ${value.scannedFrom}..${value.scannedTo} (showing ${value.matches.length}${value.hasMore ? ', more available' : ''}):`
          const lines = value.matches.map(
            m => `[seq ${m.seq} | ${m.type} | ${m.textLength} chars]\n${m.snippet}`,
          )
          return [{ type: 'text', text: `${header}\n\n${lines.join('\n\n')}` }]
        },
      },
      isConcurrencySafe: () => true,
      async execute(args, exec) {
        if (!exec.agent) throw new Error('history_search requires an owning agent session')
        const query = args.query
        if (query.length === 0) throw new Error('history_search requires a non-empty query')

        const isRegex = args.regex === true
        const caseSensitive = args.caseSensitive === true
        const order = args.order ?? 'newest'
        const limit = clamp(args.limit ?? DEFAULT_SEARCH_LIMIT, 1, MAX_SEARCH_LIMIT)
        const snippetChars = clamp(args.snippetChars ?? DEFAULT_SNIPPET_CHARS, MIN_SNIPPET_CHARS, MAX_SNIPPET_CHARS)
        const typeFilter = args.eventTypes && args.eventTypes.length > 0 ? new Set(args.eventTypes) : null

        let pattern: RegExp | null = null
        let needle = ''
        if (isRegex) {
          try {
            pattern = new RegExp(query, caseSensitive ? '' : 'i')
          } catch (error) {
            throw new Error(`Invalid regex pattern: ${error instanceof Error ? error.message : String(error)}`)
          }
        } else {
          needle = caseSensitive ? query : query.toLowerCase()
        }

        const observation = await ctx.sessionQuery.observeSession(exec.agent.session.id, {
          projectionMode: 'none',
          signal: exec.signal,
        })
        try {
          const allEvents = observation.events
          if (allEvents.length === 0) {
            return { matches: [], totalMatches: 0, hasMore: false, scannedFrom: 0, scannedTo: -1, totalEvents: 0 }
          }
          const minSeq = clamp(args.fromSeq ?? 0, 0, allEvents.length - 1)
          const maxSeq = clamp(args.toSeq ?? allEvents.length - 1, 0, allEvents.length - 1)
          if (minSeq > maxSeq) {
            return { matches: [], totalMatches: 0, hasMore: false, scannedFrom: minSeq, scannedTo: maxSeq, totalEvents: allEvents.length }
          }

          const matches: { seq: number; type: string; snippet: string; textLength: number }[] = []
          let totalMatches = 0

          const step = order === 'newest' ? -1 : 1
          const start = order === 'newest' ? maxSeq : minSeq
          const end = order === 'newest' ? minSeq - 1 : maxSeq + 1

          for (let index = start; index !== end; index += step) {
            exec.signal.throwIfAborted()
            const event = allEvents[index]!
            if (typeFilter !== null && !typeFilter.has(event.type)) continue
            const text = extractSessionEventText(event)
            if (text.length === 0) continue

            let matchIndex = -1
            let matchLength = 0
            if (pattern !== null) {
              const found = pattern.exec(text)
              if (found !== null) {
                matchIndex = found.index
                matchLength = found[0].length
              }
            } else {
              const haystack = caseSensitive ? text : text.toLowerCase()
              matchIndex = haystack.indexOf(needle)
              matchLength = needle.length
            }

            if (matchIndex !== -1) {
              totalMatches += 1
              if (matches.length < limit) {
                matches.push({
                  seq: Number(event.seq),
                  type: event.type,
                  snippet: buildSnippet(text, matchIndex, matchLength, snippetChars),
                  textLength: text.length,
                })
              }
            }
          }

          return {
            matches,
            totalMatches,
            hasMore: totalMatches > matches.length,
            scannedFrom: minSeq,
            scannedTo: maxSeq,
            totalEvents: allEvents.length,
          }
        } finally {
          observation[Symbol.dispose]()
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
        'Read canonical event text from the current session by exact seq range (up to 20 events per call, bounded by maxChars with charOffset pagination). Use after `history_search` to inspect full context around a hit.',
      parameters: {
        fromSeq: {
          type: 'integer',
          required: true,
          description: 'Inclusive starting event seq.',
        },
        toSeq: {
          type: 'integer',
          description: `Optional inclusive ending event seq (defaults to fromSeq; capped at ${MAX_READ_EVENTS} events per call).`,
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
            fromSeq: { type: 'integer', required: true },
            toSeq: { type: 'integer', required: true },
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
            text: `Events ${value.fromSeq}..${value.toSeq} (${value.totalChars} chars):\n\n${value.content}${footer}`,
          }]
        },
      },
      isConcurrencySafe: () => true,
      async execute(args, exec) {
        if (!exec.agent) throw new Error('history_read requires an owning agent session')
        const maxChars = clamp(args.maxChars ?? DEFAULT_READ_CHARS, MIN_READ_CHARS, MAX_READ_CHARS)
        const charOffset = Math.max(args.charOffset ?? 0, 0)

        const observation = await ctx.sessionQuery.observeSession(exec.agent.session.id, {
          projectionMode: 'none',
          signal: exec.signal,
        })
        try {
          const allEvents = observation.events
          if (allEvents.length === 0) {
            return { content: '', fromSeq: 0, toSeq: -1, totalChars: 0, truncated: false, nextCharOffset: null }
          }
          const fromSeq = clamp(args.fromSeq, 0, allEvents.length - 1)
          const requestedTo = args.toSeq ?? fromSeq
          if (requestedTo < fromSeq) {
            throw new Error(`toSeq (${requestedTo}) must be >= fromSeq (${fromSeq})`)
          }
          const toSeq = clamp(Math.min(requestedTo, fromSeq + MAX_READ_EVENTS - 1), fromSeq, allEvents.length - 1)

          const parts: string[] = []
          for (let index = fromSeq; index <= toSeq; index++) {
            exec.signal.throwIfAborted()
            const event = allEvents[index]!
            const text = extractSessionEventText(event)
            parts.push(`=== event ${event.seq}: ${event.type} ===\n${text.length > 0 ? text : '(no text content)'}`)
          }
          const full = parts.join('\n\n')
          const totalChars = full.length
          const slice = full.slice(charOffset, charOffset + maxChars)
          const endOffset = charOffset + slice.length
          const truncated = endOffset < totalChars

          return {
            content: slice,
            fromSeq,
            toSeq,
            totalChars,
            truncated,
            nextCharOffset: truncated ? endOffset : null,
          }
        } finally {
          observation[Symbol.dispose]()
        }
      },
      presentCall: args => ({
        card: 'generic',
        title: args.toSeq !== undefined && args.toSeq !== args.fromSeq
          ? `Read history events ${args.fromSeq}..${args.toSeq}`
          : `Read history event ${args.fromSeq}`,
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
