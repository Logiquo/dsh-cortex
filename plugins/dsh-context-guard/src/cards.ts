import type { SessionEvent } from '@deepseek-ai/dsh-session'

export type CardKind = 'constraint' | 'action' | 'assistant'

export interface ExtractedCard {
  cardIdx: number
  startEventId: number
  endEventId: number
  kind: CardKind
  tool: string
  callText: string
  resText: string
  compactBefore: boolean
}

export interface StoredCard extends ExtractedCard {
  micro: string
  embedding: Float32Array
  pDegraded: number | null
  createdAt: number
}

export const JUDGE_PREFIX =
  'Judge whether the coding agent is degraded (looping, ignoring constraints, or evasive) at the latest action:\n\n'

export const SUM_ACTION_SYS =
  "Compress one AI coding-agent tool step into ONE concise single-line summary (<=25 words). " +
  "State specifically what action was taken, key file/script/command names, and concrete outcome, numbers, mismatches, or errors observed. " +
  "NEVER write generic filler like 'OK', 'executed successfully', or 'no error reported'. " +
  'Output ONLY the single line, no markdown, no prefix.'

export const SUM_ASSISTANT_SYS =
  "Compress one AI coding-agent final reply to the user into ONE concise single-line summary (<=30 words). " +
  "State specifically what conclusion, answer, deliverable, file/code change, recommendation, or refusal the agent gave the user. " +
  "NEVER write generic filler like 'responded to the user'. " +
  'Output ONLY the single line, no markdown, no prefix.'

export function headTailBytes(text: string, headChars: number, tailChars: number): string {
  if (!text) return ''
  if (text.length <= headChars + tailChars + 48) return text
  const mid = text.slice(headChars, text.length - tailChars)
  const nBytes = Buffer.byteLength(mid, 'utf8')
  return `${text.slice(0, headChars)}\n...[skipped ${nBytes} bytes]...\n${text.slice(text.length - tailChars)}`
}

function stripSystemReminderBlocks(text: string): string {
  return text.replace(/<system-reminder>[\s\S]*?<\/system-reminder>/g, '').trim()
}

function extractTextBlocks(content: unknown): string {
  if (typeof content === 'string') return content.trim()
  if (!Array.isArray(content)) return ''
  const parts: string[] = []
  for (const block of content) {
    if (typeof block === 'string') {
      if (block.trim()) parts.push(block.trim())
    } else if (block && typeof block === 'object') {
      const b = block as Record<string, unknown>
      if (b.type === 'text' && typeof b.text === 'string' && b.text.trim()) {
        parts.push(b.text.trim())
      }
    }
  }
  return parts.join('\n').trim()
}

function hasToolCallBlock(content: unknown): boolean {
  if (!Array.isArray(content)) return false
  return content.some(b => {
    if (!b || typeof b !== 'object') return false
    const t = (b as Record<string, unknown>).type
    return t === 'tool-call' || t === 'tool_use' || t === 'tool_call'
  })
}

/**
 * Pure structural check based directly on dsh's MessageSourceMap:
 * Real human user inputs always carry `source.kind === 'user'` with no `form`.
 * Harness/plugin injections declare their own `source.kind` (e.g. 'compact-checkpoint',
 * 'model-selection', 'user-approval', 'tool-registry', etc.) or a `source.form`.
 */
export function isRealUserMessageEvent(event: SessionEvent): boolean {
  if (event.type !== 'user/message') return false
  const data = event.data as { source?: { kind?: string; form?: unknown } } | undefined
  const src = data?.source
  if (!src || src.kind !== 'user') return false
  if (src.form !== undefined) return false
  return true
}

export function isCompactionEvent(event: SessionEvent): boolean {
  if ((event.type as string) === 'compaction/start') return true
  if (event.type === 'user/message') {
    const src = (event.data as { source?: { kind?: string } } | undefined)?.source
    return src?.kind === 'compact-checkpoint'
  }
  return false
}

/**
 * Extract V2.4 chronological cards (`constraint`, `action`, `assistant`) directly from
 * canonical dsh SessionEvents, preserving exact `(startEventId, endEventId)` pointers.
 */
export function extractCardsFromEvents(events: readonly SessionEvent[]): ExtractedCard[] {
  const cards: ExtractedCard[] = []
  interface PendingAction {
    startEventId: number
    endEventId: number
    tool: string
    callText: string
    resText: string
    compactBefore: boolean
    flushed: boolean
  }
  const pendingById = new Map<string, PendingAction>()
  const pendingQueue: PendingAction[] = []
  let pendingAsst: { startEventId: number; endEventId: number; texts: string[] } | null = null
  let compactBefore = false

  const flushActions = () => {
    for (const item of pendingQueue) {
      if (!item.flushed) {
        item.flushed = true
        cards.push({
          cardIdx: cards.length,
          startEventId: item.startEventId,
          endEventId: item.endEventId,
          kind: 'action',
          tool: item.tool,
          callText: item.callText,
          resText: item.resText,
          compactBefore: item.compactBefore,
        })
      }
    }
    pendingById.clear()
    pendingQueue.length = 0
  }

  const flushAssistant = () => {
    if (pendingAsst && pendingAsst.texts.length > 0) {
      const merged = pendingAsst.texts.join('\n\n').trim()
      if (merged) {
        cards.push({
          cardIdx: cards.length,
          startEventId: pendingAsst.startEventId,
          endEventId: pendingAsst.endEventId,
          kind: 'assistant',
          tool: 'assistant',
          callText: headTailBytes(merged, 6000, 6000),
          resText: '',
          compactBefore: false,
        })
      }
    }
    pendingAsst = null
  }

  for (const event of events) {
    const seq = Number(event.seq)

    if (isCompactionEvent(event)) {
      flushActions()
      flushAssistant()
      compactBefore = true
      continue
    }

    if (isRealUserMessageEvent(event)) {
      const data = event.data as { content?: unknown }
      const txt = extractTextBlocks(data.content)
      if (!txt) continue
      flushActions()
      flushAssistant()
      cards.push({
        cardIdx: cards.length,
        startEventId: seq,
        endEventId: seq,
        kind: 'constraint',
        tool: 'user',
        callText: headTailBytes(txt, 6000, 6000),
        resText: '',
        compactBefore,
      })
      compactBefore = false
      continue
    }

    if (event.type === 'tool/call') {
      pendingAsst = null
      const data = event.data as { callId?: string; name?: string; arguments?: unknown }
      const rawArgs = typeof data.arguments === 'string'
        ? data.arguments
        : JSON.stringify(data.arguments ?? {})
      const item: PendingAction = {
        startEventId: seq,
        endEventId: seq,
        tool: String(data.name || 'tool'),
        callText: headTailBytes(rawArgs, 12000, 12000),
        resText: '',
        compactBefore,
        flushed: false,
      }
      compactBefore = false
      const callId = String(data.callId || '')
      if (callId) pendingById.set(callId, item)
      pendingQueue.push(item)
      continue
    }

    if (event.type === 'tool/result') {
      const data = event.data as {
        message?: { toolCallId?: string; source?: { callId?: string }; content?: unknown }
      }
      const msg = data.message
      const callId = String(msg?.toolCallId || msg?.source?.callId || '')
      let target = callId ? pendingById.get(callId) : undefined
      if (target) {
        pendingById.delete(callId)
      } else {
        target = pendingQueue.find(c => !c.flushed && !c.resText)
      }
      const resTxt = headTailBytes(stripSystemReminderBlocks(extractTextBlocks(msg?.content)), 12000, 12000)
      if (target) {
        target.resText = target.resText ? `${target.resText}\n${resTxt}`.trim() : resTxt
        target.endEventId = seq
        if (!target.flushed) {
          target.flushed = true
          cards.push({
            cardIdx: cards.length,
            startEventId: target.startEventId,
            endEventId: target.endEventId,
            kind: 'action',
            tool: target.tool,
            callText: target.callText,
            resText: target.resText,
            compactBefore: target.compactBefore,
          })
        }
      }
      continue
    }

    if (event.type === 'assistant/message') {
      const msg = (event.data as { message?: { content?: unknown } } | undefined)?.message
      if (!msg || hasToolCallBlock(msg.content)) continue
      const txt = extractTextBlocks(msg.content)
      if (!txt) continue
      if (!pendingAsst) {
        pendingAsst = { startEventId: seq, endEventId: seq, texts: [txt] }
      } else {
        pendingAsst.endEventId = seq
        pendingAsst.texts.push(txt)
      }
      continue
    }

    if (event.type === 'turn/end') {
      flushActions()
      flushAssistant()
    }
  }

  flushActions()
  flushAssistant()
  return cards
}

export function buildSummarizerInput(card: ExtractedCard): { system: string; user: string; maxTokens: number } {
  if (card.kind === 'assistant') {
    return {
      system: SUM_ASSISTANT_SYS,
      user: `ASSISTANT FINAL REPLY:\n${headTailBytes(card.callText, 1800, 1800)}`,
      maxTokens: 56,
    }
  }
  const cTxt = headTailBytes(card.callText, 1500, 1500)
  const rTxt = headTailBytes(card.resText, 1200, 1800)
  const user = rTxt ? `TOOL: ${card.tool}\nCALL:\n${cTxt}\nRESULT:\n${rTxt}` : `TOOL: ${card.tool}\nCALL:\n${cTxt}`
  return { system: SUM_ACTION_SYS, user, maxTokens: 44 }
}

export function formatMicroCard(card: ExtractedCard, rawSummary: string): string {
  if (card.kind === 'constraint') {
    const oneLine = card.callText.replace(/\s+/g, ' ').trim()
    return `#${card.cardIdx} [USER] ${headTailBytes(oneLine, 100, 100)}`
  }
  let line = rawSummary.replace(/<think>[\s\S]*?<\/think>/g, '').replace(/\s+/g, ' ').trim()
  line = line.replace(/^#?\d+\s*\[[^\]]+\]\s*/, '').trim()
  if (card.kind === 'assistant') {
    return `#${card.cardIdx} [ASSISTANT] ${line.slice(0, 220)}`
  }
  return `#${card.cardIdx} [${card.tool}] ${line.slice(0, 180)}`
}

export function dotProduct(a: Float32Array, b: Float32Array): number {
  const n = Math.min(a.length, b.length)
  let s = 0
  for (let i = 0; i < n; i++) s += a[i]! * b[i]!
  return s
}

/**
 * Exact V2.4 S3_32+32 window builder (100% parity with training & offline evaluation).
 */
export function buildS3Window(cards: readonly StoredCard[], targetIdx: number): string {
  const vc: number[] = []
  for (let i = 0; i < targetIdx; i++) {
    if (cards[i]!.kind === 'constraint') vc.push(i)
  }
  const rc = vc.slice(-4)
  const cn = rc.length > 0 ? rc[rc.length - 1]! : null
  const rcs = new Set(rc)

  const pool: number[] = []
  for (let i = 0; i < targetIdx; i++) {
    if (!rcs.has(i)) pool.push(i)
  }

  let picked: Set<number>
  if (pool.length <= 64) {
    picked = new Set(pool)
  } else {
    const r32 = new Set(pool.slice(-32))
    const embT = cards[targetIdx]!.embedding
    const embCn = cn !== null ? cards[cn]!.embedding : null
    const scored: Array<{ sim: number; idx: number }> = []
    for (const idx of pool) {
      if (r32.has(idx)) continue
      const e = cards[idx]!.embedding
      const sa = dotProduct(e, embT)
      const sim = embCn !== null ? Math.max(sa, dotProduct(e, embCn)) : sa
      scored.push({ sim, idx })
    }
    scored.sort((a, b) => b.sim - a.sim)
    picked = new Set(r32)
    for (let i = 0; i < Math.min(32, scored.length); i++) {
      picked.add(scored[i]!.idx)
    }
  }

  const tl = [...picked, targetIdx].sort((a, b) => a - b)
  const out: string[] = ['[ACTIVE CONSTRAINTS]']
  for (const idx of rc) {
    const label = idx === cn ? 'USER (latest)' : 'USER'
    out.push(`#${idx} [${label}]\n${headTailBytes(cards[idx]!.callText.trim(), 400, 400)}`)
  }
  out.push(`\n[TIMELINE (up to 64 micro-cards: recent 32 + nearest 32, plus current action #${targetIdx})]`)

  let prev: number | null = null
  for (const idx of tl) {
    const lo = prev !== null ? prev + 1 : 0
    const cmps: number[] = []
    for (let k = lo; k <= idx; k++) {
      if (cards[k]!.compactBefore) cmps.push(k)
    }
    if (prev !== null && idx > prev + 1) {
      out.push(`... (omitted ${idx - prev - 1} steps) ...`)
    }
    for (const cp of cmps) {
      out.push(`--- [COMPACT before #${cp}] ---`)
    }
    const c = cards[idx]!
    if (idx === targetIdx) {
      const cFmt = headTailBytes(c.callText.trim(), 600, 600)
      const rRaw = c.resText.trim()
      const rFmt = rRaw ? headTailBytes(rRaw, 500, 500) : '(empty)'
      out.push(`#${idx} [CURRENT ACTION ANCHOR | ${c.tool}]\nCALL:\n${cFmt}\nRESULT:\n${rFmt}`)
    } else if (c.kind === 'constraint') {
      const oneLn = c.callText.replace(/\s+/g, ' ').trim()
      out.push(`#${idx} [USER] ${headTailBytes(oneLn, 100, 100)}`)
    } else {
      out.push(c.micro)
    }
    prev = idx
  }
  return JUDGE_PREFIX + out.join('\n')
}
