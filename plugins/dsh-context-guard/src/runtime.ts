import { spawn, execFile } from 'node:child_process'
import type { ChildProcess } from 'node:child_process'
import { createHash } from 'node:crypto'
import { createWriteStream, existsSync, statSync } from 'node:fs'
import { chmod, mkdir, readFile, rename, unlink } from 'node:fs/promises'
import { createServer } from 'node:net'
import { dirname, join } from 'node:path'
import { Readable, Transform } from 'node:stream'
import { pipeline } from 'node:stream/promises'
import type { ReadableStream as WebReadableStream } from 'node:stream/web'
import { promisify } from 'node:util'
import type { ExtractedCard } from './cards.js'
import { buildSummarizerInput, formatMicroCard } from './cards.js'
import {
  DEFAULT_HF_ENDPOINT,
  DEFAULT_HF_MODEL_FILES,
  DEFAULT_HF_MODEL_SHA256,
  DEFAULT_HF_REPO,
  DEFAULT_HF_REVISION,
  DEFAULT_LLAMA_ARCHIVE_SHA256,
  DEFAULT_LLAMA_BUILD,
} from './config.js'

const execFileAsync = promisify(execFile)

export interface GuardInferenceClient {
  ensureReady?(signal?: AbortSignal): Promise<void>
  summarize(card: ExtractedCard, signal?: AbortSignal): Promise<string>
  embed(text: string, signal?: AbortSignal): Promise<Float32Array>
  judge(prompt: string, signal?: AbortSignal): Promise<number>
  status(): Record<string, unknown>
  dispose(): Promise<void>
}

export const DEFAULT_INFERENCE_TIMEOUT_MS = 60_000

export interface LlamaRuntimeOptions {
  storeRoot: string
  llamaServerPath?: string
  llamaBuild: string
  autoDownloadRuntime: boolean
  autoDownloadModels?: boolean
  hfRepo?: string
  hfRevision?: string
  hfEndpoint?: string
  /** Optional SHA256 overrides for downloaded model assets (defaults to pinned DEFAULT_HF_MODEL_SHA256). */
  modelSha256?: Partial<Record<ModelAssetKey, string>>
  /** Optional SHA256 override for downloaded llama.cpp release archive (defaults to pinned DEFAULT_LLAMA_ARCHIVE_SHA256). */
  runtimeArchiveSha256?: string
  /** Per-request timeout in ms for summarize, embed, and judge calls (defaults to 60_000). */
  inferenceTimeoutMs?: number
  gpuLayers: number
  summarizerModelPath: string
  summarizerModelCustom?: boolean
  judgeModelPath: string
  judgeModelCustom?: boolean
  judgeHeadPath: string
  judgeHeadCustom?: boolean
  embeddingModelPath: string
  embeddingModelCustom?: boolean
}

type ModelAssetKey = keyof typeof DEFAULT_HF_MODEL_FILES

interface ModelAssetSpec {
  key: ModelAssetKey
  label: string
  destPath: string
  hfRelPath: string
  customOverride: boolean
}

interface PrepareState {
  status: 'idle' | 'checking' | 'downloading' | 'ready' | 'error'
  activeAsset: string | null
  lastError: string | null
}

function fileExistsNonEmpty(path: string): boolean {
  try {
    return existsSync(path) && statSync(path).size > 0
  } catch {
    return false
  }
}

interface DownloadOptions {
  signal?: AbortSignal
  expectedSha256?: string
}

async function downloadFileAtomic(url: string, destPath: string, options: DownloadOptions = {}): Promise<void> {
  const { signal, expectedSha256 } = options
  await mkdir(dirname(destPath), { recursive: true, mode: 0o700 })
  const tmpPath = `${destPath}.part-${process.pid}-${Date.now()}`
  try {
    const res = await fetch(url, { redirect: 'follow', ...(signal ? { signal } : {}) })
    if (!res.ok) throw new Error(`Failed to download ${url}: HTTP ${res.status}`)
    if (!res.body) throw new Error(`Empty response body for ${url}`)
    const expectedLen = Number(res.headers.get('content-length') ?? '0')
    const nodeStream = Readable.fromWeb(res.body as unknown as WebReadableStream)
    const hash = createHash('sha256')
    const hasher = new Transform({
      transform(chunk, _encoding, callback) {
        hash.update(chunk)
        callback(null, chunk)
      },
    })
    await pipeline(nodeStream, hasher, createWriteStream(tmpPath), ...(signal ? [{ signal }] : []))
    const writtenSize = statSync(tmpPath).size
    if (writtenSize === 0 || (expectedLen > 0 && writtenSize !== expectedLen)) {
      throw new Error(`Incomplete download for ${url}: got ${writtenSize} bytes, expected ${expectedLen}`)
    }
    if (expectedSha256 !== undefined) {
      const actualSha256 = hash.digest('hex')
      const normalizedExpected = expectedSha256.trim().toLowerCase()
      if (actualSha256 !== normalizedExpected) {
        throw new Error(`SHA256 mismatch for ${url}: expected ${normalizedExpected}, got ${actualSha256}`)
      }
    }
    await rename(tmpPath, destPath)
  } catch (err) {
    await unlink(tmpPath).catch(() => {})
    throw err
  }
}

interface JudgeHead {
  hidden_size: number
  weight: number[]
  bias: number
}

interface ReleaseAsset {
  archiveName: string
  binaryRelCandidates: string[]
}

function selectReleaseAsset(build: string): ReleaseAsset {
  const plat = process.platform
  const arch = process.arch
  if (plat === 'win32' && arch === 'x64') {
    return {
      archiveName: `llama-${build}-bin-win-vulkan-x64.zip`,
      binaryRelCandidates: ['llama-server.exe', 'build/bin/llama-server.exe'],
    }
  }
  if (plat === 'win32' && arch === 'arm64') {
    return {
      archiveName: `llama-${build}-bin-win-cpu-arm64.zip`,
      binaryRelCandidates: ['llama-server.exe', 'build/bin/llama-server.exe'],
    }
  }
  if (plat === 'darwin' && arch === 'arm64') {
    return {
      archiveName: `llama-${build}-bin-macos-arm64.tar.gz`,
      binaryRelCandidates: ['llama-server', 'build/bin/llama-server'],
    }
  }
  if (plat === 'darwin' && arch === 'x64') {
    return {
      archiveName: `llama-${build}-bin-macos-x64.tar.gz`,
      binaryRelCandidates: ['llama-server', 'build/bin/llama-server'],
    }
  }
  if (plat === 'linux' && arch === 'x64') {
    return {
      archiveName: `llama-${build}-bin-ubuntu-vulkan-x64.tar.gz`,
      binaryRelCandidates: ['llama-server', 'build/bin/llama-server'],
    }
  }
  if (plat === 'linux' && arch === 'arm64') {
    return {
      archiveName: `llama-${build}-bin-ubuntu-arm64.tar.gz`,
      binaryRelCandidates: ['llama-server', 'build/bin/llama-server'],
    }
  }
  throw new Error(`Unsupported platform/arch for prebuilt llama.cpp: ${plat}/${arch}`)
}

function findFreePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const srv = createServer()
    srv.unref()
    srv.on('error', reject)
    srv.listen(0, '127.0.0.1', () => {
      const addr = srv.address()
      const port = typeof addr === 'object' && addr ? addr.port : 0
      srv.close(() => resolve(port))
    })
  })
}

function normalizeL2(arr: number[]): Float32Array {
  const out = new Float32Array(arr.length)
  let sumSq = 0
  for (let i = 0; i < arr.length; i++) {
    const v = arr[i]!
    out[i] = v
    sumSq += v * v
  }
  const norm = Math.sqrt(sumSq)
  if (norm > 0) {
    for (let i = 0; i < out.length; i++) out[i] = out[i]! / norm
  }
  return out
}

class ManagedLlamaServer {
  private proc: ChildProcess | null = null
  private port: number | null = null
  private startPromise: Promise<string> | null = null
  private activeNgl: number

  constructor(
    readonly role: 'summarizer' | 'embedding' | 'judge',
    private readonly getBinary: (signal?: AbortSignal) => Promise<string>,
    private readonly ensureModel: (signal?: AbortSignal) => Promise<string>,
    private readonly modelPath: string,
    private readonly baseArgs: string[],
    private readonly preferredNgl: number,
  ) {
    this.activeNgl = preferredNgl
  }

  isRunning(): boolean {
    return this.proc !== null && this.proc.exitCode === null && this.port !== null
  }

  info(): { running: boolean; port: number | null; ngl: number; modelPath: string; present: boolean } {
    return {
      running: this.isRunning(),
      port: this.port,
      ngl: this.activeNgl,
      modelPath: this.modelPath,
      present: fileExistsNonEmpty(this.modelPath),
    }
  }

  async ensureUrl(signal?: AbortSignal): Promise<string> {
    signal?.throwIfAborted()
    if (this.isRunning()) return `http://127.0.0.1:${this.port}`
    if (!this.startPromise) {
      this.startPromise = this.launch(signal).finally(() => { this.startPromise = null })
    }
    return this.startPromise
  }

  private async launch(signal?: AbortSignal): Promise<string> {
    const [bin, resolvedModel] = await Promise.all([this.getBinary(signal), this.ensureModel(signal)])
    if (!fileExistsNonEmpty(resolvedModel)) {
      throw new Error(`Missing ${this.role} GGUF model at ${resolvedModel}`)
    }
    const nglAttempts = this.preferredNgl > 0 ? [this.preferredNgl, 0] : [0]
    let lastErr: unknown
    for (const ngl of nglAttempts) {
      try {
        const port = await findFreePort()
        const args = ['-m', resolvedModel, ...this.baseArgs, '-ngl', String(ngl), '--port', String(port), '--host', '127.0.0.1']
        const proc = spawn(bin, args, { stdio: ['ignore', 'ignore', 'pipe'], windowsHide: true })
        let errTail = ''
        proc.stderr?.on('data', chunk => { errTail = (errTail + chunk.toString()).slice(-3000) })

        let healthy = false
        for (let i = 0; i < 100; i++) {
          if (signal?.aborted) {
            try { proc.kill('SIGKILL') } catch { /* ignore */ }
            signal.throwIfAborted()
          }
          await new Promise(r => setTimeout(r, 250))
          if (signal?.aborted) {
            try { proc.kill('SIGKILL') } catch { /* ignore */ }
            signal.throwIfAborted()
          }
          if (proc.exitCode !== null) {
            throw new Error(`llama-server (${this.role}, ngl=${ngl}) exited ${proc.exitCode}: ${errTail}`)
          }
          try {
            const res = await fetch(`http://127.0.0.1:${port}/health`)
            if (res.ok) {
              const j = (await res.json()) as { status?: string }
              if (j.status === 'ok') { healthy = true; break }
            }
          } catch { /* keep polling */ }
        }
        if (!healthy) {
          proc.kill('SIGKILL')
          throw new Error(`llama-server (${this.role}, ngl=${ngl}) health timeout: ${errTail}`)
        }
        this.proc = proc
        this.port = port
        this.activeNgl = ngl
        return `http://127.0.0.1:${port}`
      } catch (err) {
        lastErr = err
      }
    }
    throw lastErr instanceof Error ? lastErr : new Error(String(lastErr))
  }

  async stop(): Promise<void> {
    const proc = this.proc
    this.proc = null
    this.port = null
    if (!proc || proc.exitCode !== null) return
    try { proc.kill('SIGTERM') } catch { /* ignore */ }
    await new Promise<void>(resolve => {
      const timer = setTimeout(() => {
        try { proc.kill('SIGKILL') } catch { /* ignore */ }
        resolve()
      }, 2000)
      timer.unref()
      proc.once('exit', () => { clearTimeout(timer); resolve() })
    })
  }
}

export class LlamaGuardRuntime implements GuardInferenceClient {
  private binaryPath: string | null = null
  private binaryPromise: Promise<string> | null = null
  private readyPromise: Promise<void> | null = null
  private readonly modelPromises = new Map<ModelAssetKey, Promise<string>>()
  private readonly modelAssets: Record<ModelAssetKey, ModelAssetSpec>
  private readonly prepareState: PrepareState = { status: 'idle', activeAsset: null, lastError: null }
  private head: JudgeHead | null = null
  private readonly sumServer: ManagedLlamaServer
  private readonly embServer: ManagedLlamaServer
  private readonly judgeServer: ManagedLlamaServer
  private readonly inferenceTimeoutMs: number

  constructor(private readonly opts: LlamaRuntimeOptions) {
    this.inferenceTimeoutMs = opts.inferenceTimeoutMs ?? DEFAULT_INFERENCE_TIMEOUT_MS
    this.modelAssets = {
      embedding: {
        key: 'embedding',
        label: 'embeddinggemma-2',
        destPath: opts.embeddingModelPath,
        hfRelPath: DEFAULT_HF_MODEL_FILES.embedding,
        customOverride: opts.embeddingModelCustom ?? false,
      },
      summarizer: {
        key: 'summarizer',
        label: 'qwen3.5-0.8b-agentic-summarizer',
        destPath: opts.summarizerModelPath,
        hfRelPath: DEFAULT_HF_MODEL_FILES.summarizer,
        customOverride: opts.summarizerModelCustom ?? false,
      },
      judge: {
        key: 'judge',
        label: 'qwen3.5-0.8b-agentic-monitor',
        destPath: opts.judgeModelPath,
        hfRelPath: DEFAULT_HF_MODEL_FILES.judge,
        customOverride: opts.judgeModelCustom ?? false,
      },
      judgeHead: {
        key: 'judgeHead',
        label: 'qwen3.5-0.8b-agentic-monitor/head.json',
        destPath: opts.judgeHeadPath,
        hfRelPath: DEFAULT_HF_MODEL_FILES.judgeHead,
        customOverride: opts.judgeHeadCustom ?? false,
      },
    }

    const getBin = (signal?: AbortSignal) => this.ensureBinary(signal)
    this.sumServer = new ManagedLlamaServer(
      'summarizer',
      getBin,
      signal => this.ensureModelAsset('summarizer', signal),
      opts.summarizerModelPath,
      ['-c', '2048'],
      opts.gpuLayers,
    )
    this.embServer = new ManagedLlamaServer(
      'embedding',
      getBin,
      signal => this.ensureModelAsset('embedding', signal),
      opts.embeddingModelPath,
      ['--embeddings', '--pooling', 'mean', '-c', '2048', '-b', '2048', '-ub', '2048'],
      opts.gpuLayers,
    )
    this.judgeServer = new ManagedLlamaServer(
      'judge',
      getBin,
      signal => this.ensureModelAsset('judge', signal),
      opts.judgeModelPath,
      ['--embeddings', '--pooling', 'last', '--embd-normalize', '-1', '-c', '8192', '-b', '8192', '-ub', '4096'],
      opts.gpuLayers,
    )
  }

  /**
   * Check that the llama-server runtime binary and all 3 model weights (+ judge head.json)
   * exist locally, auto-downloading any missing asset. Deduplicated across concurrent callers.
   */
  async ensureReady(signal?: AbortSignal): Promise<void> {
    signal?.throwIfAborted()
    if (!this.readyPromise) {
      this.readyPromise = this.prepareAllAssets(signal).finally(() => { this.readyPromise = null })
    }
    return this.readyPromise
  }

  private async prepareAllAssets(signal?: AbortSignal): Promise<void> {
    this.prepareState.status = 'checking'
    this.prepareState.lastError = null
    try {
      await this.ensureBinary(signal)
      const order: ModelAssetKey[] = ['embedding', 'summarizer', 'judge', 'judgeHead']
      for (const key of order) {
        signal?.throwIfAborted()
        await this.ensureModelAsset(key, signal)
      }
      this.prepareState.status = 'ready'
      this.prepareState.activeAsset = null
    } catch (err) {
      this.prepareState.status = 'error'
      this.prepareState.activeAsset = null
      this.prepareState.lastError = err instanceof Error ? err.message : String(err)
      throw err
    }
  }

  private resolveModelDownloadUrl(hfRelPath: string): string {
    const base = (this.opts.hfEndpoint ?? DEFAULT_HF_ENDPOINT).replace(/\/+$/, '')
    const repo = (this.opts.hfRepo ?? DEFAULT_HF_REPO).replace(/^\/+|\/+$/g, '')
    const revision = (this.opts.hfRevision ?? DEFAULT_HF_REVISION).replace(/^\/+|\/+$/g, '')
    return `${base}/${repo}/resolve/${revision}/${hfRelPath}`
  }

  private resolveExpectedModelSha256(key: ModelAssetKey): string | undefined {
    if (this.opts.modelSha256?.[key]) return this.opts.modelSha256[key]
    const repo = (this.opts.hfRepo ?? DEFAULT_HF_REPO).replace(/^\/+|\/+$/g, '')
    const revision = (this.opts.hfRevision ?? DEFAULT_HF_REVISION).replace(/^\/+|\/+$/g, '')
    if (repo === DEFAULT_HF_REPO && revision === DEFAULT_HF_REVISION) {
      return DEFAULT_HF_MODEL_SHA256[key]
    }
    return undefined
  }

  private async ensureModelAsset(key: ModelAssetKey, signal?: AbortSignal): Promise<string> {
    const spec = this.modelAssets[key]
    if (fileExistsNonEmpty(spec.destPath)) return spec.destPath
    const inflight = this.modelPromises.get(key)
    if (inflight) return inflight

    const task = (async () => {
      if (fileExistsNonEmpty(spec.destPath)) return spec.destPath
      if (spec.customOverride) {
        throw new Error(`Configured ${spec.label} file does not exist: ${spec.destPath}`)
      }
      if (this.opts.autoDownloadModels === false) {
        throw new Error(`Missing ${spec.label} at ${spec.destPath} and autoDownloadModels is disabled`)
      }
      this.prepareState.status = 'downloading'
      this.prepareState.activeAsset = spec.label
      const url = this.resolveModelDownloadUrl(spec.hfRelPath)
      const expectedSha256 = this.resolveExpectedModelSha256(key)
      await downloadFileAtomic(url, spec.destPath, {
        ...(signal ? { signal } : {}),
        ...(expectedSha256 ? { expectedSha256 } : {}),
      })
      return spec.destPath
    })().finally(() => { this.modelPromises.delete(key) })

    this.modelPromises.set(key, task)
    return task
  }

  private async ensureBinary(signal?: AbortSignal): Promise<string> {
    if (this.binaryPath && existsSync(this.binaryPath)) return this.binaryPath
    if (this.opts.llamaServerPath) {
      if (!existsSync(this.opts.llamaServerPath)) {
        throw new Error(`Configured llamaServerPath does not exist: ${this.opts.llamaServerPath}`)
      }
      this.binaryPath = this.opts.llamaServerPath
      return this.binaryPath
    }
    if (!this.binaryPromise) {
      this.binaryPromise = this.resolveOrDownloadBinary(signal).finally(() => { this.binaryPromise = null })
    }
    return this.binaryPromise
  }

  private async resolveOrDownloadBinary(signal?: AbortSignal): Promise<string> {
    const asset = selectReleaseAsset(this.opts.llamaBuild)
    const runtimeDir = join(this.opts.storeRoot, 'runtime', this.opts.llamaBuild)
    for (const rel of asset.binaryRelCandidates) {
      const cand = join(runtimeDir, rel)
      if (existsSync(cand)) {
        this.binaryPath = cand
        return cand
      }
    }
    if (!this.opts.autoDownloadRuntime) {
      throw new Error(`llama-server binary not found in ${runtimeDir} and autoDownloadRuntime is disabled`)
    }

    this.prepareState.status = 'downloading'
    this.prepareState.activeAsset = `llama.cpp (${this.opts.llamaBuild})`
    await mkdir(runtimeDir, { recursive: true, mode: 0o700 })
    const url = `https://github.com/ggml-org/llama.cpp/releases/download/${this.opts.llamaBuild}/${asset.archiveName}`
    const archivePath = join(runtimeDir, asset.archiveName)
    const expectedSha256 = this.opts.runtimeArchiveSha256 ?? DEFAULT_LLAMA_ARCHIVE_SHA256[asset.archiveName]
    if (this.opts.llamaBuild === DEFAULT_LLAMA_BUILD && !expectedSha256) {
      throw new Error(`Missing pinned SHA256 checksum for llama.cpp archive ${asset.archiveName}`)
    }
    await downloadFileAtomic(url, archivePath, {
      ...(signal ? { signal } : {}),
      ...(expectedSha256 ? { expectedSha256 } : {}),
    })

    if (asset.archiveName.endsWith('.zip')) {
      await execFileAsync('powershell.exe', [
        '-NoProfile',
        '-Command',
        `Expand-Archive -Force -Path '${archivePath.replace(/'/g, "''")}' -DestinationPath '${runtimeDir.replace(/'/g, "''")}'`,
      ])
    } else {
      await execFileAsync('tar', ['-xzf', archivePath, '-C', runtimeDir])
    }
    await unlink(archivePath).catch(() => {})

    for (const rel of asset.binaryRelCandidates) {
      const cand = join(runtimeDir, rel)
      if (existsSync(cand)) {
        if (process.platform !== 'win32') await chmod(cand, 0o755)
        this.binaryPath = cand
        return cand
      }
    }
    throw new Error(`Extracted ${asset.archiveName} to ${runtimeDir}, but llama-server binary was not found`)
  }

  private async loadHead(signal?: AbortSignal): Promise<JudgeHead> {
    if (this.head) return this.head
    const headPath = await this.ensureModelAsset('judgeHead', signal)
    const raw = await readFile(headPath, 'utf8')
    const parsed = JSON.parse(raw) as JudgeHead
    if (!Array.isArray(parsed.weight) || typeof parsed.bias !== 'number') {
      throw new Error(`Invalid judge head file: ${headPath}`)
    }
    this.head = parsed
    return parsed
  }

  private async postJson(
    server: ManagedLlamaServer,
    url: string,
    body: unknown,
    label: string,
    signal?: AbortSignal,
  ): Promise<unknown> {
    const timeoutMs = this.inferenceTimeoutMs
    const timeoutSignal = AbortSignal.timeout(timeoutMs)
    const combinedSignal = signal ? AbortSignal.any([signal, timeoutSignal]) : timeoutSignal
    try {
      const res = await fetch(url, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        signal: combinedSignal,
        body: JSON.stringify(body),
      })
      if (!res.ok) throw new Error(`${label} HTTP ${res.status}: ${await res.text()}`)
      return await res.json()
    } catch (err) {
      if (timeoutSignal.aborted && !signal?.aborted) {
        await server.stop()
        throw new Error(`${label} inference timed out after ${timeoutMs}ms`)
      }
      throw err
    }
  }

  async summarize(card: ExtractedCard, signal?: AbortSignal): Promise<string> {
    if (card.kind === 'constraint') return formatMicroCard(card, '')
    const base = await this.sumServer.ensureUrl(signal)
    const spec = buildSummarizerInput(card)
    const j = (await this.postJson(
      this.sumServer,
      `${base}/v1/chat/completions`,
      {
        messages: [
          { role: 'system', content: spec.system },
          { role: 'user', content: spec.user },
        ],
        max_tokens: spec.maxTokens,
        temperature: 0,
        chat_template_kwargs: { enable_thinking: false },
      },
      'Summarizer',
      signal,
    )) as { choices?: Array<{ message?: { content?: string } }> }
    const content = j.choices?.[0]?.message?.content ?? ''
    return formatMicroCard(card, content)
  }

  async embed(text: string, signal?: AbortSignal): Promise<Float32Array> {
    const base = await this.embServer.ensureUrl(signal)
    const j = (await this.postJson(
      this.embServer,
      `${base}/v1/embeddings`,
      { input: [text.slice(0, 1024)] },
      'Embedding',
      signal,
    )) as { data?: Array<{ embedding?: number[] }> }
    const vec = j.data?.[0]?.embedding
    if (!Array.isArray(vec) || vec.length === 0) throw new Error('Empty embedding vector returned')
    return normalizeL2(vec)
  }

  async judge(prompt: string, signal?: AbortSignal): Promise<number> {
    const [base, head] = await Promise.all([this.judgeServer.ensureUrl(signal), this.loadHead(signal)])
    const j = await this.postJson(
      this.judgeServer,
      `${base}/embedding`,
      { content: prompt },
      'Judge',
      signal,
    )
    let vec: number[] | undefined
    if (Array.isArray(j) && j[0]?.embedding) {
      vec = Array.isArray(j[0].embedding[0]) ? j[0].embedding[0] : j[0].embedding
    } else if (j && typeof j === 'object' && 'embedding' in j) {
      const e = (j as { embedding: unknown }).embedding
      vec = Array.isArray(e) ? (Array.isArray(e[0]) ? (e[0] as number[]) : (e as number[])) : undefined
    }
    if (!vec || vec.length !== head.weight.length) {
      throw new Error(`Judge hidden state dim mismatch: got ${vec?.length ?? 0}, expected ${head.weight.length}`)
    }
    let logit = head.bias
    for (let i = 0; i < head.weight.length; i++) {
      logit += vec[i]! * head.weight[i]!
    }
    return 1 / (1 + Math.exp(-logit))
  }

  status(): Record<string, unknown> {
    return {
      prepare: { ...this.prepareState },
      binaryPath: this.binaryPath,
      llamaBuild: this.opts.llamaBuild,
      hfRepo: this.opts.hfRepo ?? DEFAULT_HF_REPO,
      judgeHeadPresent: fileExistsNonEmpty(this.opts.judgeHeadPath),
      summarizer: this.sumServer.info(),
      embedding: this.embServer.info(),
      judge: this.judgeServer.info(),
    }
  }

  async dispose(): Promise<void> {
    await Promise.allSettled([this.sumServer.stop(), this.embServer.stop(), this.judgeServer.stop()])
  }
}
