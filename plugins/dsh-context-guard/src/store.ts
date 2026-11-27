import { mkdir, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import { DatabaseSync } from 'node:sqlite'
import { sessionDir } from '@deepseek-ai/dsh-spill-local'
import type { CardKind, StoredCard } from './cards.js'
import { dotProduct } from './cards.js'

export const GUARD_DB_FILE = 'guard.sqlite'
export const TIMELINE_TXT_FILE = 'timeline.txt'
const SCHEMA_VERSION = '1'

export interface GuardSessionStatus {
  sessionId: string
  directory: string
  dbPath: string
  throughEventId: number
  cardCount: number
  actionCount: number
  scoredCount: number
  latestScores: Array<{ cardIdx: number; startEventId: number; endEventId: number; tool: string; pDegraded: number }>
}

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

interface CachedGuardDb {
  db: DatabaseSync
  dbPath: string
  directory: string
}

function initSchema(db: DatabaseSync, sessionId: string): void {
  db.exec(`
    PRAGMA journal_mode = WAL;
    PRAGMA synchronous = NORMAL;
    PRAGMA busy_timeout = 5000;

    CREATE TABLE IF NOT EXISTS meta (
      key   TEXT PRIMARY KEY,
      value TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS cards (
      card_idx       INTEGER PRIMARY KEY,
      start_event_id INTEGER NOT NULL,
      end_event_id   INTEGER NOT NULL,
      kind           TEXT    NOT NULL,
      tool           TEXT    NOT NULL,
      call_text      TEXT    NOT NULL,
      res_text       TEXT    NOT NULL,
      compact_before INTEGER NOT NULL DEFAULT 0,
      micro          TEXT    NOT NULL,
      embedding      BLOB    NOT NULL,
      p_degraded     REAL,
      created_at     INTEGER NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_cards_seq  ON cards(start_event_id, end_event_id);
    CREATE INDEX IF NOT EXISTS idx_cards_kind ON cards(kind, card_idx);
  `)
  const ins = db.prepare('INSERT OR IGNORE INTO meta (key, value) VALUES (?, ?)')
  ins.run('schema_version', SCHEMA_VERSION)
  ins.run('session_id', sessionId)
  ins.run('through_event_id', '-1')
  ins.run('updated_at', String(Date.now()))
}

function setMeta(db: DatabaseSync, key: string, value: string): void {
  db.prepare('INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value').run(key, value)
}

function float32ToBlob(vec: Float32Array): Uint8Array {
  return new Uint8Array(vec.buffer, vec.byteOffset, vec.byteLength)
}

function blobToFloat32(blob: Uint8Array): Float32Array {
  const copy = new Uint8Array(blob.byteLength)
  copy.set(blob)
  return new Float32Array(copy.buffer)
}

function formatTimelineLine(c: StoredCard): string {
  const pStr = c.pDegraded !== null ? `p=${c.pDegraded.toFixed(3)}` : 'p=---'
  const cmp = c.compactBefore ? '[COMPACT] ' : ''
  return `${cmp}[ids ${c.startEventId}..${c.endEventId} | ${pStr}] ${c.micro}`
}

export class GuardStore {
  private readonly cache = new Map<string, CachedGuardDb>()
  private readonly queues = new Map<string, Promise<unknown>>()

  constructor(readonly root: string, readonly writeTimelineTxt = true) {}

  sessionDirectory(sessionId: string): string {
    return sessionDir(this.root, sessionId)
  }

  dbPathFor(sessionId: string): string {
    return join(this.sessionDirectory(sessionId), GUARD_DB_FILE)
  }

  private serial<T>(sessionId: string, task: () => Promise<T>): Promise<T> {
    const prev = this.queues.get(sessionId) ?? Promise.resolve()
    const next = prev.catch(() => undefined).then(task)
    this.queues.set(sessionId, next)
    void next.finally(() => {
      if (this.queues.get(sessionId) === next) this.queues.delete(sessionId)
    }).catch(() => undefined)
    return next
  }

  async drain(): Promise<void> {
    await Promise.all([...this.queues.values()])
  }

  close(sessionId?: string): void {
    if (sessionId !== undefined) {
      const entry = this.cache.get(sessionId)
      if (entry) {
        try { entry.db.close() } catch { /* ignore */ }
        this.cache.delete(sessionId)
      }
      return
    }
    for (const entry of this.cache.values()) {
      try { entry.db.close() } catch { /* ignore */ }
    }
    this.cache.clear()
  }

  private async openDb(sessionId: string): Promise<CachedGuardDb> {
    const existing = this.cache.get(sessionId)
    if (existing) return existing
    const directory = this.sessionDirectory(sessionId)
    await mkdir(directory, { recursive: true, mode: 0o700 })
    const dbPath = join(directory, GUARD_DB_FILE)
    const db = new DatabaseSync(dbPath)
    initSchema(db, sessionId)
    const entry: CachedGuardDb = { db, dbPath, directory }
    this.cache.set(sessionId, entry)
    return entry
  }

  loadCards(sessionId: string): Promise<{ throughEventId: number; cards: StoredCard[] }> {
    return this.serial(sessionId, async () => {
      const { db } = await this.openDb(sessionId)
      const mRow = db.prepare("SELECT value FROM meta WHERE key = 'through_event_id'").get() as { value: string } | undefined
      const throughEventId = Number(mRow?.value ?? '-1')
      const rows = db.prepare(
        'SELECT card_idx, start_event_id, end_event_id, kind, tool, call_text, res_text, compact_before, micro, embedding, p_degraded, created_at FROM cards ORDER BY card_idx ASC',
      ).all() as Array<{
        card_idx: number
        start_event_id: number
        end_event_id: number
        kind: CardKind
        tool: string
        call_text: string
        res_text: string
        compact_before: number
        micro: string
        embedding: Uint8Array
        p_degraded: number | null
        created_at: number
      }>
      const cards: StoredCard[] = rows.map(r => ({
        cardIdx: r.card_idx,
        startEventId: r.start_event_id,
        endEventId: r.end_event_id,
        kind: r.kind,
        tool: r.tool,
        callText: r.call_text,
        resText: r.res_text,
        compactBefore: r.compact_before !== 0,
        micro: r.micro,
        embedding: blobToFloat32(r.embedding),
        pDegraded: r.p_degraded,
        createdAt: r.created_at,
      }))
      return { throughEventId, cards }
    })
  }

  saveCards(sessionId: string, throughEventId: number, allCards: readonly StoredCard[], replaceAll = false): Promise<void> {
    return this.serial(sessionId, async () => {
      const { db, directory } = await this.openDb(sessionId)
      const stmt = db.prepare(
        `INSERT INTO cards (card_idx, start_event_id, end_event_id, kind, tool, call_text, res_text, compact_before, micro, embedding, p_degraded, created_at)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
         ON CONFLICT(card_idx) DO UPDATE SET
           start_event_id = excluded.start_event_id,
           end_event_id   = excluded.end_event_id,
           kind           = excluded.kind,
           tool           = excluded.tool,
           call_text      = excluded.call_text,
           res_text       = excluded.res_text,
           compact_before = excluded.compact_before,
           micro          = excluded.micro,
           embedding      = excluded.embedding,
           p_degraded     = excluded.p_degraded,
           created_at     = excluded.created_at`,
      )
      db.exec('BEGIN IMMEDIATE')
      try {
        if (replaceAll) db.exec('DELETE FROM cards')
        for (const c of allCards) {
          stmt.run(
            c.cardIdx,
            c.startEventId,
            c.endEventId,
            c.kind,
            c.tool,
            c.callText,
            c.resText,
            c.compactBefore ? 1 : 0,
            c.micro,
            float32ToBlob(c.embedding),
            c.pDegraded,
            c.createdAt,
          )
        }
        setMeta(db, 'through_event_id', String(throughEventId))
        setMeta(db, 'updated_at', String(Date.now()))
        db.exec('COMMIT')
      } catch (err) {
        try { db.exec('ROLLBACK') } catch { /* ignore */ }
        throw err
      }

      if (this.writeTimelineTxt) {
        const text = allCards.map(formatTimelineLine).join('\n') + (allCards.length > 0 ? '\n' : '')
        await writeFile(join(directory, TIMELINE_TXT_FILE), text, 'utf8')
      }
    })
  }

  semanticSearch(
    sessionId: string,
    queryVec: Float32Array,
    options: { topK: number; fromId?: number; toId?: number },
  ): Promise<SemanticSearchHit[]> {
    return this.serial(sessionId, async () => {
      const { db } = await this.openDb(sessionId)
      const params: number[] = []
      const clauses: string[] = []
      if (options.fromId !== undefined) {
        clauses.push('end_event_id >= ?')
        params.push(options.fromId)
      }
      if (options.toId !== undefined) {
        clauses.push('start_event_id <= ?')
        params.push(options.toId)
      }
      const where = clauses.length > 0 ? `WHERE ${clauses.join(' AND ')}` : ''
      const rows = db.prepare(
        `SELECT card_idx, start_event_id, end_event_id, kind, micro, embedding FROM cards ${where}`,
      ).all(...params) as Array<{
        card_idx: number
        start_event_id: number
        end_event_id: number
        kind: string
        micro: string
        embedding: Uint8Array
      }>

      const scored: SemanticSearchHit[] = rows.map(r => ({
        cardIdx: r.card_idx,
        startId: r.start_event_id,
        endId: r.end_event_id,
        kind: r.kind,
        score: dotProduct(queryVec, blobToFloat32(r.embedding)),
        micro: r.micro,
      }))
      scored.sort((a, b) => b.score - a.score)
      return scored.slice(0, options.topK)
    })
  }

  listSummaries(
    sessionId: string,
    options: { fromId?: number; toId?: number } = {},
  ): Promise<SummaryRangeResult> {
    return this.serial(sessionId, async () => {
      const { db } = await this.openDb(sessionId)
      const mRow = db.prepare("SELECT value FROM meta WHERE key = 'through_event_id'").get() as { value: string } | undefined
      const throughEventId = Number(mRow?.value ?? '-1')
      const params: number[] = []
      const clauses: string[] = []
      if (options.fromId !== undefined) {
        clauses.push('end_event_id >= ?')
        params.push(options.fromId)
      }
      if (options.toId !== undefined) {
        clauses.push('start_event_id <= ?')
        params.push(options.toId)
      }
      const where = clauses.length > 0 ? `WHERE ${clauses.join(' AND ')}` : ''
      const rows = db.prepare(
        `SELECT card_idx, start_event_id, end_event_id, kind, micro FROM cards ${where} ORDER BY card_idx ASC`,
      ).all(...params) as Array<{
        card_idx: number
        start_event_id: number
        end_event_id: number
        kind: string
        micro: string
      }>
      return {
        throughEventId,
        cards: rows.map(r => ({
          cardIdx: r.card_idx,
          startId: r.start_event_id,
          endId: r.end_event_id,
          kind: r.kind,
          micro: r.micro,
        })),
      }
    })
  }

  lexicalSearch(sessionId: string, options: SummaryLexicalSearchOptions): Promise<SummaryCardItem[]> {
    let pattern: RegExp | null = null
    let needle = ''
    if (options.regex) {
      try {
        pattern = new RegExp(options.query, options.caseSensitive ? '' : 'i')
      } catch (error) {
        throw new Error(`Invalid regex pattern: ${error instanceof Error ? error.message : String(error)}`)
      }
    } else {
      needle = options.caseSensitive ? options.query : options.query.toLowerCase()
    }

    return this.serial(sessionId, async () => {
      const { db } = await this.openDb(sessionId)
      const params: Array<number | string> = []
      const clauses: string[] = []
      if (options.fromId !== undefined) {
        clauses.push('end_event_id >= ?')
        params.push(options.fromId)
      }
      if (options.toId !== undefined) {
        clauses.push('start_event_id <= ?')
        params.push(options.toId)
      }
      if (options.kinds && options.kinds.length > 0) {
        clauses.push(`kind IN (${options.kinds.map(() => '?').join(', ')})`)
        params.push(...options.kinds)
      }
      const where = clauses.length > 0 ? `WHERE ${clauses.join(' AND ')}` : ''
      const dir = options.order === 'oldest' ? 'ASC' : 'DESC'
      const stmt = db.prepare(
        `SELECT card_idx, start_event_id, end_event_id, kind, micro FROM cards ${where} ORDER BY card_idx ${dir}`,
      )
      const hits: SummaryCardItem[] = []
      for (const raw of stmt.iterate(...params)) {
        options.signal?.throwIfAborted()
        const r = raw as { card_idx: number; start_event_id: number; end_event_id: number; kind: string; micro: string }
        const matched = pattern !== null
          ? pattern.test(r.micro)
          : (options.caseSensitive ? r.micro : r.micro.toLowerCase()).includes(needle)
        if (matched) {
          hits.push({
            cardIdx: r.card_idx,
            startId: r.start_event_id,
            endId: r.end_event_id,
            kind: r.kind,
            micro: r.micro,
          })
          if (hits.length >= options.limit) break
        }
      }
      return hits
    })
  }

  getStatus(sessionId: string): Promise<GuardSessionStatus> {
    return this.serial(sessionId, async () => {
      const { db, directory, dbPath } = await this.openDb(sessionId)
      const mRow = db.prepare("SELECT value FROM meta WHERE key = 'through_event_id'").get() as { value: string } | undefined
      const counts = db.prepare(
        `SELECT COUNT(*) AS total,
                SUM(CASE WHEN kind = 'action' THEN 1 ELSE 0 END) AS actions,
                SUM(CASE WHEN p_degraded IS NOT NULL THEN 1 ELSE 0 END) AS scored
         FROM cards`,
      ).get() as { total: number; actions: number | null; scored: number | null }
      const latest = db.prepare(
        'SELECT card_idx, start_event_id, end_event_id, tool, p_degraded FROM cards WHERE p_degraded IS NOT NULL ORDER BY card_idx DESC LIMIT 10',
      ).all() as Array<{ card_idx: number; start_event_id: number; end_event_id: number; tool: string; p_degraded: number }>

      return {
        sessionId,
        directory,
        dbPath,
        throughEventId: Number(mRow?.value ?? '-1'),
        cardCount: Number(counts.total),
        actionCount: Number(counts.actions ?? 0),
        scoredCount: Number(counts.scored ?? 0),
        latestScores: latest.reverse().map(r => ({
          cardIdx: r.card_idx,
          startEventId: r.start_event_id,
          endEventId: r.end_event_id,
          tool: r.tool,
          pDegraded: Number(r.p_degraded.toFixed(4)),
        })),
      }
    })
  }
}
