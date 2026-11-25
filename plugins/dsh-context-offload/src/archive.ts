import { createHash } from 'node:crypto'
import { mkdir, open, readFile, rename, stat } from 'node:fs/promises'
import { basename, join } from 'node:path'
import type { SessionEvent } from '@deepseek-ai/dsh-session'
import { extractSessionEventText } from '@deepseek-ai/dsh-session-query'
import { withFileLock, writeFileAtomic } from '@deepseek-ai/dsh-atomic-write'
import { saveTextFile, sessionDir } from '@deepseek-ai/dsh-spill-local'

interface Chunk { file: string; from: number; through: number; sha256: string }
export interface Catalog {
  version: 1
  sessionId: string
  through: number
  chain: string
  chunks: Chunk[]
}
export interface ArchiveCut { directory: string; catalog: Catalog; compaction: number; turn: number | null; compactionId: string }
export interface EnsureOptions { verifyChunks?: boolean }

interface CachedSession {
  catalog: Catalog
  catalogMtimeMs: number
  catalogSize: number
  eventRefs: readonly SessionEvent[]
}

const digest = (text: string) => createHash('sha256').update(text).digest('hex')
const pad = (n: number) => String(n).padStart(6, '0')

async function syncFile(path: string): Promise<void> {
  const handle = await open(path, 'r+')
  try { await handle.sync() } finally { await handle.close() }
}
async function syncDirectory(path: string): Promise<void> {
  if (process.platform === 'win32') return // Windows does not expose portable directory fsync.
  const handle = await open(path, 'r')
  try { await handle.sync() } finally { await handle.close() }
}
async function publish(path: string, text: string): Promise<void> {
  await writeFileAtomic(path, text, { mode: 0o600, dirMode: 0o700 })
  await syncFile(path)
}
export function renderEvents(events: readonly SessionEvent[]): string {
  const sections = events.map(event => {
    // Prefix every body line so user text cannot forge a record boundary.
    const text = extractSessionEventText(event).split('\n').map(line => `| ${line}`).join('\n')
    return `=== event ${event.seq}: ${event.type} ===\n${text}\nCanonical JSON:\n${JSON.stringify(event, null, 2)}\n`
  })
  return sections.join('\n')
}

function cloneCatalog(catalog: Catalog): Catalog {
  return {
    version: catalog.version,
    sessionId: catalog.sessionId,
    through: catalog.through,
    chain: catalog.chain,
    chunks: catalog.chunks.map(chunk => ({ ...chunk })),
  }
}

/** Dedicated archive ownership; never registers or changes ctx.spillStore. */
export class ArchiveStore {
  private readonly queues = new Map<string, Promise<unknown>>()
  private readonly cache = new Map<string, CachedSession>()
  constructor(readonly root: string) {}

  private serial<T>(sessionId: string, task: () => Promise<T>): Promise<T> {
    const previous = this.queues.get(sessionId) ?? Promise.resolve()
    const next = previous.catch(() => undefined).then(task)
    this.queues.set(sessionId, next)
    void next.finally(() => { if (this.queues.get(sessionId) === next) this.queues.delete(sessionId) }).catch(() => undefined)
    return next
  }
  async drain(): Promise<void> { await Promise.all([...this.queues.values()]) }

  /** Fast synchronous check whether this store has already archived events for a session. */
  hasArchived(sessionId: string): boolean {
    const entry = this.cache.get(sessionId)
    return entry !== undefined && entry.catalog.through >= 0
  }

  async verify(sessionId: string): Promise<Catalog | undefined> {
    const directory = sessionDir(this.root, sessionId)
    const catalogPath = join(directory, 'catalog.json')
    let raw: string
    try { raw = await readFile(catalogPath, 'utf8') }
    catch (error) { if ((error as NodeJS.ErrnoException).code === 'ENOENT') return undefined; throw error }
    const catalog = JSON.parse(raw) as Catalog
    if (catalog.version !== 1 || catalog.sessionId !== sessionId || !Array.isArray(catalog.chunks)) throw new Error('Invalid archive catalog')
    let next = 0
    for (const chunk of catalog.chunks) {
      if (basename(chunk.file) !== chunk.file || chunk.from !== next || chunk.through < chunk.from) throw new Error('Non-contiguous archive catalog')
      const text = await readFile(join(directory, chunk.file), 'utf8')
      if (digest(text) !== chunk.sha256) throw new Error(`Archive checksum mismatch: ${chunk.file}`)
      next = chunk.through + 1
    }
    if (catalog.through !== next - 1) throw new Error('Archive coverage mismatch')
    return catalog
  }

  /** Capture canonical history, including records shadowed by previous compactions. */
  ensure(sessionId: string, events: readonly SessionEvent[], signal?: AbortSignal, options?: EnsureOptions): Promise<ArchiveCut> {
    return this.serial(sessionId, async () => {
      signal?.throwIfAborted()
      const directory = sessionDir(this.root, sessionId)
      const catalogPath = join(directory, 'catalog.json')
      await mkdir(directory, { recursive: true, mode: 0o700 })
      return withFileLock(catalogPath, async () => {
        const cached = this.cache.get(sessionId)
        let previous: Catalog | undefined
        let cacheHit = false

        if (cached !== undefined && options?.verifyChunks !== true) {
          try {
            const info = await stat(catalogPath)
            if (info.mtimeMs === cached.catalogMtimeMs && info.size === cached.catalogSize) {
              previous = cloneCatalog(cached.catalog)
              cacheHit = true
            }
          } catch (error) {
            if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error
          }
        }

        if (!cacheHit) {
          previous = await this.verify(sessionId)
        }

        let chain = ''
        let prefix = previous === undefined || previous.through === -1
        let startIndex = 0

        // Fast path: when cached event object references match the archived prefix identically,
        // skip re-serializing and re-hashing the already-verified prefix.
        if (
          cacheHit &&
          cached !== undefined &&
          previous !== undefined &&
          previous.through >= 0 &&
          events.length > previous.through &&
          cached.eventRefs.length > previous.through
        ) {
          let identicalRefs = true
          for (let i = 0; i <= previous.through; i++) {
            if (events[i] !== cached.eventRefs[i]) {
              identicalRefs = false
              break
            }
          }
          if (identicalRefs) {
            chain = previous.chain
            prefix = true
            startIndex = previous.through + 1
          }
        }

        for (let index = startIndex; index < events.length; index++) {
          const event = events[index]!
          if (Number(event.seq) !== index) throw new Error('Canonical history is not contiguous from event 0')
          chain = digest(`${chain}\n${JSON.stringify(event)}`)
          if (index === previous?.through) {
            if (chain !== previous.chain) throw new Error('Canonical history differs from archived prefix')
            prefix = true
          }
        }
        if (!prefix) throw new Error('Canonical history is shorter than archived history')

        const starts = events.filter(event => event.type === 'compaction/start')
        const latest = starts.at(-1)
        const compaction = starts.length
        const turn = latest?.type === 'compaction/start' ? latest.data.turn : null
        const compactionId = latest?.type === 'compaction/start' ? String(latest.data.compactionId) : 'none'
        const catalog: Catalog = previous ?? { version: 1, sessionId, through: -1, chain: '', chunks: [] }
        const pending = events.slice(catalog.through + 1)
        if (pending.length > 0) {
          const from = catalog.through + 1
          const through = events.length - 1
          const name = `context-offload-c${pad(compaction)}-${turn === null ? 'idle' : `turn${pad(turn)}`}-events${pad(from)}-${pad(through)}.txt`
          const text = renderEvents(pending)
          const saved = await saveTextFile({ root: this.root, sessionId, suggestedName: `${name}.pending`, content: text })
          await syncFile(saved.path)
          signal?.throwIfAborted()
          const path = saved.path.slice(0, -'.pending'.length)
          await rename(saved.path, path)
          await syncDirectory(directory)
          catalog.chunks.push({ file: basename(path), from, through, sha256: digest(text) })
          catalog.through = through
          catalog.chain = chain
          await publish(catalogPath, JSON.stringify(catalog, null, 2) + '\n')
          await syncDirectory(directory)
        }
        if (catalog.through >= 0) {
          const info = await stat(catalogPath)
          this.cache.set(sessionId, {
            catalog: cloneCatalog(catalog),
            catalogMtimeMs: info.mtimeMs,
            catalogSize: info.size,
            eventRefs: events.slice(0, catalog.through + 1),
          })
        }
        return { directory, catalog, compaction, turn, compactionId }
      }, signal ? { signal } : {})
    })
  }

  saveBrief(cut: ArchiveCut, text: string): Promise<void> {
    return this.serial(cut.catalog.sessionId, async () => {
      await publish(join(cut.directory, `brief-${digest(cut.compactionId)}.md`), text)
      await syncDirectory(cut.directory)
    })
  }
}
