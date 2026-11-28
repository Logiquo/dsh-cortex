import { createHash } from 'node:crypto'
import { existsSync } from 'node:fs'
import { mkdir } from 'node:fs/promises'
import { join } from 'node:path'
import { DatabaseSync } from 'node:sqlite'
import type { SessionEvent } from '@deepseek-ai/dsh-session'
import { extractSessionEventText } from '@deepseek-ai/dsh-session-query'
import { sessionDir } from '@deepseek-ai/dsh-spill-local'

export const OFFLOAD_DB_FILE = 'offload.sqlite'
const SCHEMA_VERSION = '1'

export interface ArchiveStatus {
  sessionId: string
  directory: string
  dbPath: string
  throughId: number
  eventCount: number
  compactionCount: number
  chainSha256: string
}

export interface ArchiveCut {
  directory: string
  dbPath: string
  sessionId: string
  throughId: number
  eventCount: number
  compactionCount: number
  compaction: number
  turn: number | null
  compactionId: string
}

export interface EnsureOptions {
  verifyIntegrity?: boolean
}

export interface CompactRecordInput {
  provider: string
  model: string
  body: string
  framedText: string
}

export interface StoredEventRow {
  id: number
  type: string
  text: string
  textLen: number
}

export interface LexicalMatch {
  id: number
  type: string
  snippet: string
  textLength: number
}

export interface LexicalSearchOptions {
  query: string
  regex: boolean
  caseSensitive: boolean
  eventTypes?: readonly string[]
  fromId?: number
  toId?: number
  order: 'newest' | 'oldest'
  limit: number
  snippetChars: number
  signal?: AbortSignal
}

export interface LexicalSearchResult {
  matches: LexicalMatch[]
  totalMatches: number
  scannedFrom: number
  scannedTo: number
  totalEvents: number
}

interface CachedSession {
  db: DatabaseSync
  dbPath: string
  throughId: number
  chainSha256: string
  eventRefs: readonly SessionEvent[]
}

const digest = (text: string) => createHash('sha256').update(text).digest('hex')

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

function extractNumericField(event: SessionEvent, field: 'turn' | 'step'): number | null {
  const data = (event as { data?: unknown }).data
  if (data !== null && typeof data === 'object' && field in data) {
    const val = (data as Record<string, unknown>)[field]
    if (typeof val === 'number' && Number.isFinite(val)) return Math.trunc(val)
  }
  return null
}

function extractTimeMs(event: SessionEvent): number | null {
  const raw = (event as { time?: unknown }).time
  return typeof raw === 'number' && Number.isFinite(raw) ? Math.trunc(raw) : null
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

    CREATE TABLE IF NOT EXISTS events (
      id       INTEGER PRIMARY KEY,
      type     TEXT    NOT NULL,
      turn     INTEGER,
      step     INTEGER,
      time_ms  INTEGER,
      text     TEXT    NOT NULL,
      text_len INTEGER NOT NULL,
      raw_json TEXT    NOT NULL,
      sha256   TEXT    NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_events_type_id ON events(type, id);
    CREATE INDEX IF NOT EXISTS idx_events_turn    ON events(turn) WHERE turn IS NOT NULL;

    CREATE TABLE IF NOT EXISTS compact (
      id             INTEGER PRIMARY KEY AUTOINCREMENT,
      compaction_id  TEXT    NOT NULL UNIQUE,
      through_id     INTEGER NOT NULL,
      turn           INTEGER,
      provider       TEXT    NOT NULL,
      model          TEXT    NOT NULL,
      body           TEXT    NOT NULL,
      framed_text    TEXT    NOT NULL,
      created_at     INTEGER NOT NULL
    );
  `)

  const insMeta = db.prepare('INSERT OR IGNORE INTO meta (key, value) VALUES (?, ?)')
  insMeta.run('schema_version', SCHEMA_VERSION)
  insMeta.run('session_id', sessionId)
  insMeta.run('through_id', '-1')
  insMeta.run('chain_sha256', '')
  insMeta.run('updated_at', String(Date.now()))
}

function readMeta(db: DatabaseSync): Map<string, string> {
  const rows = db.prepare('SELECT key, value FROM meta').all() as Array<{ key: string; value: string }>
  const map = new Map<string, string>()
  for (const row of rows) map.set(row.key, row.value)
  return map
}

function setMeta(db: DatabaseSync, key: string, value: string): void {
  db.prepare('INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value').run(key, value)
}

function verifyDbIntegrity(db: DatabaseSync, sessionId: string, directory: string, dbPath: string): ArchiveStatus {
  const qc = db.prepare('PRAGMA quick_check').get() as Record<string, string> | undefined
  const qcVal = qc ? Object.values(qc)[0] : undefined
  if (qcVal !== 'ok') throw new Error(`SQLite integrity check failed: ${qcVal ?? 'unknown'}`)

  const meta = readMeta(db)
  if (meta.get('schema_version') !== SCHEMA_VERSION) {
    throw new Error(`Unsupported offload schema_version: ${meta.get('schema_version') ?? 'missing'}`)
  }
  if (meta.get('session_id') !== sessionId) {
    throw new Error(`Archive session_id mismatch: expected ${sessionId}, got ${meta.get('session_id') ?? 'missing'}`)
  }

  const throughId = Number(meta.get('through_id') ?? '-1')
  const expectedChain = meta.get('chain_sha256') ?? ''
  if (!Number.isInteger(throughId) || throughId < -1) {
    throw new Error(`Invalid archive through_id: ${meta.get('through_id')}`)
  }

  const rows = db.prepare('SELECT id, raw_json, sha256 FROM events ORDER BY id ASC').all() as Array<{
    id: number
    raw_json: string
    sha256: string
  }>

  let chain = ''
  for (let i = 0; i < rows.length; i++) {
    const row = rows[i]!
    if (row.id !== i) throw new Error(`Non-contiguous archive events: expected id ${i}, got ${row.id}`)
    chain = digest(`${chain}\n${row.raw_json}`)
    if (row.sha256 !== chain) {
      throw new Error(`Archive checksum mismatch at event id ${row.id}`)
    }
  }

  if (rows.length - 1 !== throughId) {
    throw new Error(`Archive coverage mismatch: through_id=${throughId}, count=${rows.length}`)
  }
  if (chain !== expectedChain) {
    throw new Error('Archive chain checksum mismatch')
  }

  const compactRow = db.prepare('SELECT COUNT(*) AS cnt FROM compact').get() as { cnt: number }
  return {
    sessionId,
    directory,
    dbPath,
    throughId,
    eventCount: rows.length,
    compactionCount: Number(compactRow.cnt),
    chainSha256: chain,
  }
}

/** Dedicated SQLite archive ownership (`offload.sqlite`); never registers or changes ctx.spillStore. */
export class ArchiveStore {
  private readonly queues = new Map<string, Promise<unknown>>()
  private readonly cache = new Map<string, CachedSession>()

  constructor(readonly root: string) {}

  sessionDirectory(sessionId: string): string {
    return sessionDir(this.root, sessionId)
  }

  dbPathFor(sessionId: string): string {
    return join(this.sessionDirectory(sessionId), OFFLOAD_DB_FILE)
  }

  private serial<T>(sessionId: string, task: () => Promise<T>): Promise<T> {
    const previous = this.queues.get(sessionId) ?? Promise.resolve()
    const next = previous.catch(() => undefined).then(task)
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

  private async openDb(sessionId: string): Promise<CachedSession> {
    const existing = this.cache.get(sessionId)
    if (existing) return existing

    const directory = this.sessionDirectory(sessionId)
    await mkdir(directory, { recursive: true, mode: 0o700 })
    const dbPath = join(directory, OFFLOAD_DB_FILE)
    const db = new DatabaseSync(dbPath)
    let meta: Map<string, string>
    try {
      initSchema(db, sessionId)
      meta = readMeta(db)
      if (meta.get('schema_version') !== SCHEMA_VERSION) {
        throw new Error(`Unsupported offload schema_version: ${meta.get('schema_version') ?? 'missing'}`)
      }
    } catch (err) {
      try { db.close() } catch { /* ignore */ }
      throw err
    }
    const entry: CachedSession = {
      db,
      dbPath,
      throughId: Number(meta.get('through_id') ?? '-1'),
      chainSha256: meta.get('chain_sha256') ?? '',
      eventRefs: [],
    }
    this.cache.set(sessionId, entry)
    return entry
  }

  async verify(sessionId: string): Promise<ArchiveStatus | undefined> {
    return this.serial(sessionId, async () => {
      const directory = this.sessionDirectory(sessionId)
      const dbPath = join(directory, OFFLOAD_DB_FILE)
      if (!existsSync(dbPath)) return undefined
      const entry = this.cache.get(sessionId)
      if (entry) {
        return verifyDbIntegrity(entry.db, sessionId, directory, dbPath)
      }
      const db = new DatabaseSync(dbPath, { readOnly: true })
      try {
        return verifyDbIntegrity(db, sessionId, directory, dbPath)
      } finally {
        db.close()
      }
    })
  }

  /** Capture canonical history incrementally into offload.sqlite (or initialize from event 0). */
  ensure(sessionId: string, events: readonly SessionEvent[], signal?: AbortSignal, options?: EnsureOptions): Promise<ArchiveCut> {
    return this.serial(sessionId, async () => {
      signal?.throwIfAborted()
      const directory = this.sessionDirectory(sessionId)
      const entry = await this.openDb(sessionId)
      const { db, dbPath } = entry

      if (options?.verifyIntegrity === true) {
        const verified = verifyDbIntegrity(db, sessionId, directory, dbPath)
        entry.throughId = verified.throughId
        entry.chainSha256 = verified.chainSha256
      }

      const prevThrough = entry.throughId
      const prevChain = entry.chainSha256

      let chain = ''
      let prefixOk = prevThrough === -1
      let startIndex = 0

      if (
        options?.verifyIntegrity !== true &&
        prevThrough >= 0 &&
        events.length > prevThrough &&
        entry.eventRefs.length > prevThrough
      ) {
        let identicalRefs = true
        for (let i = 0; i <= prevThrough; i++) {
          if (events[i] !== entry.eventRefs[i]) {
            identicalRefs = false
            break
          }
        }
        if (identicalRefs) {
          chain = prevChain
          prefixOk = true
          startIndex = prevThrough + 1
        }
      }

      const pendingRows: Array<{
        id: number
        type: string
        turn: number | null
        step: number | null
        timeMs: number | null
        text: string
        rawJson: string
        sha256: string
      }> = []

      for (let index = startIndex; index < events.length; index++) {
        signal?.throwIfAborted()
        const event = events[index]!
        if (Number(event.seq) !== index) throw new Error('Canonical history is not contiguous from event 0')
        const rawJson = JSON.stringify(event)
        chain = digest(`${chain}\n${rawJson}`)
        if (index === prevThrough) {
          if (chain !== prevChain) throw new Error('Canonical history differs from archived prefix')
          prefixOk = true
        } else if (index > prevThrough) {
          const text = extractSessionEventText(event)
          pendingRows.push({
            id: index,
            type: event.type,
            turn: extractNumericField(event, 'turn'),
            step: extractNumericField(event, 'step'),
            timeMs: extractTimeMs(event),
            text,
            rawJson,
            sha256: chain,
          })
        }
      }
      if (!prefixOk) throw new Error('Canonical history is shorter than archived history')

      if (pendingRows.length > 0) {
        const ins = db.prepare(
          'INSERT INTO events (id, type, turn, step, time_ms, text, text_len, raw_json, sha256) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
        )
        db.exec('BEGIN IMMEDIATE')
        try {
          for (const row of pendingRows) {
            ins.run(row.id, row.type, row.turn, row.step, row.timeMs, row.text, row.text.length, row.rawJson, row.sha256)
          }
          const newThrough = events.length - 1
          setMeta(db, 'through_id', String(newThrough))
          setMeta(db, 'chain_sha256', chain)
          setMeta(db, 'updated_at', String(Date.now()))
          db.exec('COMMIT')
          entry.throughId = newThrough
          entry.chainSha256 = chain
        } catch (err) {
          try { db.exec('ROLLBACK') } catch { /* ignore */ }
          throw err
        }
      }

      entry.eventRefs = events.slice(0, entry.throughId + 1)

      const starts = events.filter(event => event.type === 'compaction/start')
      const latest = starts.at(-1)
      const compaction = starts.length
      const turn = latest?.type === 'compaction/start' ? latest.data.turn : null
      const compactionId = latest?.type === 'compaction/start' ? String(latest.data.compactionId) : 'none'
      const compactRow = db.prepare('SELECT COUNT(*) AS cnt FROM compact').get() as { cnt: number }

      return {
        directory,
        dbPath,
        sessionId,
        throughId: entry.throughId,
        eventCount: entry.throughId + 1,
        compactionCount: Number(compactRow.cnt),
        compaction,
        turn,
        compactionId,
      }
    })
  }

  /** Rebuild this session's events table from canonical dsh history (preserving recorded compact briefs). */
  rebuild(sessionId: string, events: readonly SessionEvent[], signal?: AbortSignal): Promise<ArchiveCut> {
    return this.serial(sessionId, async () => {
      signal?.throwIfAborted()
      const directory = this.sessionDirectory(sessionId)
      const entry = await this.openDb(sessionId)
      const { db, dbPath } = entry

      const rows: Array<{
        id: number
        type: string
        turn: number | null
        step: number | null
        timeMs: number | null
        text: string
        rawJson: string
        sha256: string
      }> = []
      let chain = ''
      for (let index = 0; index < events.length; index++) {
        signal?.throwIfAborted()
        const event = events[index]!
        if (Number(event.seq) !== index) throw new Error('Canonical history is not contiguous from event 0')
        const rawJson = JSON.stringify(event)
        chain = digest(`${chain}\n${rawJson}`)
        rows.push({
          id: index,
          type: event.type,
          turn: extractNumericField(event, 'turn'),
          step: extractNumericField(event, 'step'),
          timeMs: extractTimeMs(event),
          text: extractSessionEventText(event),
          rawJson,
          sha256: chain,
        })
      }

      const ins = db.prepare(
        'INSERT INTO events (id, type, turn, step, time_ms, text, text_len, raw_json, sha256) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
      )
      db.exec('BEGIN IMMEDIATE')
      try {
        db.exec('DELETE FROM events')
        for (const row of rows) {
          ins.run(row.id, row.type, row.turn, row.step, row.timeMs, row.text, row.text.length, row.rawJson, row.sha256)
        }
        const newThrough = events.length - 1
        setMeta(db, 'schema_version', SCHEMA_VERSION)
        setMeta(db, 'session_id', sessionId)
        setMeta(db, 'through_id', String(newThrough))
        setMeta(db, 'chain_sha256', chain)
        setMeta(db, 'updated_at', String(Date.now()))
        db.exec('COMMIT')
        entry.throughId = newThrough
        entry.chainSha256 = chain
        entry.eventRefs = events.slice(0, events.length)
      } catch (err) {
        try { db.exec('ROLLBACK') } catch { /* ignore */ }
        throw err
      }

      const starts = events.filter(event => event.type === 'compaction/start')
      const latest = starts.at(-1)
      const compaction = starts.length
      const turn = latest?.type === 'compaction/start' ? latest.data.turn : null
      const compactionId = latest?.type === 'compaction/start' ? String(latest.data.compactionId) : 'none'
      const compactRow = db.prepare('SELECT COUNT(*) AS cnt FROM compact').get() as { cnt: number }

      return {
        directory,
        dbPath,
        sessionId,
        throughId: entry.throughId,
        eventCount: entry.throughId + 1,
        compactionCount: Number(compactRow.cnt),
        compaction,
        turn,
        compactionId,
      }
    })
  }

  saveCompact(cut: ArchiveCut, record: CompactRecordInput): Promise<void> {
    return this.serial(cut.sessionId, async () => {
      const entry = await this.openDb(cut.sessionId)
      entry.db
        .prepare(
          `INSERT INTO compact (compaction_id, through_id, turn, provider, model, body, framed_text, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(compaction_id) DO UPDATE SET
             through_id = excluded.through_id,
             turn = excluded.turn,
             provider = excluded.provider,
             model = excluded.model,
             body = excluded.body,
             framed_text = excluded.framed_text,
             created_at = excluded.created_at`,
        )
        .run(
          cut.compactionId,
          cut.throughId,
          cut.turn,
          record.provider,
          record.model,
          record.body,
          record.framedText,
          Date.now(),
        )
    })
  }

  async getEventCount(sessionId: string): Promise<number> {
    return this.serial(sessionId, async () => {
      const entry = await this.openDb(sessionId)
      return entry.throughId + 1
    })
  }

  /** O(1) lookup of the highest compacted event id (`-1` if no compaction has occurred). */
  async getCompactedThroughId(sessionId: string): Promise<number> {
    return this.serial(sessionId, async () => {
      const entry = await this.openDb(sessionId)
      const row = entry.db.prepare('SELECT MAX(through_id) AS max_id FROM compact').get() as { max_id: number | null } | undefined
      return row?.max_id ?? -1
    })
  }

  /** O(1) indexed range read directly from offload.sqlite. */
  async readEventRange(sessionId: string, fromId: number, toId: number): Promise<StoredEventRow[]> {
    return this.serial(sessionId, async () => {
      const entry = await this.openDb(sessionId)
      const rows = entry.db
        .prepare('SELECT id, type, text, text_len FROM events WHERE id BETWEEN ? AND ? ORDER BY id ASC')
        .all(fromId, toId) as Array<{ id: number; type: string; text: string; text_len: number }>
      return rows.map(r => ({ id: r.id, type: r.type, text: r.text, textLen: r.text_len }))
    })
  }

  /** Streaming lexical/regex search directly over indexed events in offload.sqlite. */
  async searchEvents(sessionId: string, opts: LexicalSearchOptions): Promise<LexicalSearchResult> {
    let pattern: RegExp | null = null
    let needle = ''
    if (opts.regex) {
      try {
        pattern = new RegExp(opts.query, opts.caseSensitive ? '' : 'i')
      } catch (error) {
        throw new Error(`Invalid regex pattern: ${error instanceof Error ? error.message : String(error)}`)
      }
    } else {
      needle = opts.caseSensitive ? opts.query : opts.query.toLowerCase()
    }

    return this.serial(sessionId, async () => {
      const entry = await this.openDb(sessionId)
      const totalEvents = entry.throughId + 1
      if (totalEvents <= 0) {
        return { matches: [], totalMatches: 0, scannedFrom: 0, scannedTo: -1, totalEvents: 0 }
      }
      const minId = clamp(opts.fromId ?? 0, 0, totalEvents - 1)
      const maxId = clamp(opts.toId ?? totalEvents - 1, 0, totalEvents - 1)
      if (minId > maxId) {
        return { matches: [], totalMatches: 0, scannedFrom: minId, scannedTo: maxId, totalEvents }
      }

      const dir = opts.order === 'newest' ? 'DESC' : 'ASC'
      const params: Array<number | string> = [minId, maxId]
      let where = 'id BETWEEN ? AND ? AND text_len > 0'
      if (opts.eventTypes && opts.eventTypes.length > 0) {
        where += ` AND type IN (${opts.eventTypes.map(() => '?').join(', ')})`
        params.push(...opts.eventTypes)
      }

      const stmt = entry.db.prepare(`SELECT id, type, text, text_len FROM events WHERE ${where} ORDER BY id ${dir}`)
      const matches: LexicalMatch[] = []
      let totalMatches = 0

      for (const raw of stmt.iterate(...params)) {
        opts.signal?.throwIfAborted()
        const row = raw as { id: number; type: string; text: string; text_len: number }
        let matchIndex = -1
        let matchLength = 0
        if (pattern !== null) {
          const found = pattern.exec(row.text)
          if (found !== null) {
            matchIndex = found.index
            matchLength = found[0].length
          }
        } else {
          const haystack = opts.caseSensitive ? row.text : row.text.toLowerCase()
          matchIndex = haystack.indexOf(needle)
          matchLength = needle.length
        }

        if (matchIndex !== -1) {
          totalMatches += 1
          if (matches.length < opts.limit) {
            matches.push({
              id: row.id,
              type: row.type,
              snippet: buildSnippet(row.text, matchIndex, matchLength, opts.snippetChars),
              textLength: row.text_len,
            })
          }
        }
      }

      return { matches, totalMatches, scannedFrom: minId, scannedTo: maxId, totalEvents }
    })
  }
}
