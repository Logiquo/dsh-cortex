import { homedir } from 'node:os'
import { isAbsolute, join, normalize } from 'node:path'
import z from '@deepseek-ai/schemastery'

export const DEFAULT_LLAMA_BUILD = 'b12651'
export const DEFAULT_HF_REPO = 'Logic-z/dsh-cortex'
export const DEFAULT_HF_REVISION = 'f037ae5864df3c25d7d3db1f691893c5d743dd9d'
export const DEFAULT_HF_ENDPOINT = 'https://huggingface.co'
export const DEFAULT_SUM_MODEL = 'sum08b-Q4_K_M.gguf'
export const DEFAULT_JUDGE_MODEL = 'judge08b-Q4_K_M.gguf'
export const DEFAULT_JUDGE_HEAD = 'judge08b-head.json'
export const DEFAULT_EMB_MODEL = 'embeddinggemma-2-Q8_0.gguf'

export const DEFAULT_HF_MODEL_FILES = {
  summarizer: 'qwen3.5-0.8b-agentic-summarizer/qwen3.5-0.8b-agentic-summarizer-Q4_K_M.gguf',
  judge: 'qwen3.5-0.8b-agentic-monitor/qwen3.5-0.8b-agentic-monitor-Q4_K_M.gguf',
  judgeHead: 'qwen3.5-0.8b-agentic-monitor/head.json',
  embedding: 'embeddinggemma-2/embeddinggemma-2-Q8_0.gguf',
} as const

export const DEFAULT_HF_MODEL_SHA256 = {
  summarizer: 'dd7c04b8bb617b8e50dc61cce5f766b0aa3ce0fdd7693e656444d9542557c9ba',
  judge: '098a86e5744193c7499adf3e4e5f26f939720553ab5ceb7e167e8d5298627ab2',
  judgeHead: 'fcb63f178df1e43f68a38818efc024607d70203c38558e88bfbae7760e966e28',
  embedding: '2188ac1deca4b77dffefd603c2776a9d76d9d74ec01841392982ebb840b09135',
} as const

export const DEFAULT_LLAMA_ARCHIVE_SHA256: Readonly<Record<string, string>> = {
  'llama-b12651-bin-win-vulkan-x64.zip': 'e58fa9a137d8046c69334005ad6fb3e904ddc7e56f2a61ad1b844335826ef3f4',
  'llama-b12651-bin-win-cpu-arm64.zip': '756a62761e476536f007c866d46fa14125f6ca32d98cee6af14685659bd9f244',
  'llama-b12651-bin-macos-arm64.tar.gz': '264fd872de61679c45ad57e1e21c2a2a54e29b35734ecf3d50f24166447ab8d2',
  'llama-b12651-bin-macos-x64.tar.gz': '34c19771545479a783d3b68c446383ab703033890c052d3ff0b860a5c9f59e56',
  'llama-b12651-bin-ubuntu-vulkan-x64.tar.gz': '0bbcf2093f15c64c8a9bf54b809c9b2c812b982992dc2fbc9b36720c1d32cf13',
  'llama-b12651-bin-ubuntu-arm64.tar.gz': 'b4a8d17e16a1b42f6a1395dc5cd6967ec0d98005eb3be85b7eb893b082306ed4',
}

export interface Config {
  /** Shared cortex root directory (defaults to ~/.dsh/context-cortex). */
  storeRoot?: string
  /** Optional explicit path to an existing llama-server binary (skips runtime auto-download). */
  llamaServerPath?: string
  /** Pinned ggml-org/llama.cpp release tag used for auto-downloading prebuilt binaries. */
  llamaBuild?: string
  /** Auto-download prebuilt llama-server binary when missing. Defaults to true. */
  autoDownloadRuntime?: boolean
  /** Auto-download missing default GGUF weights and judge head from Hugging Face. Defaults to true. */
  autoDownloadModels?: boolean
  /** Check runtime and model assets immediately on plugin startup (non-blocking). Defaults to true. */
  checkAssetsOnStartup?: boolean
  /** Hugging Face repo hosting the dsh-cortex weights. Defaults to Logic-z/dsh-cortex. */
  hfRepo?: string
  /** Pinned Hugging Face commit SHA or revision. Defaults to pinned immutable commit SHA. */
  hfRevision?: string
  /** Hugging Face base URL (defaults to process.env.HF_ENDPOINT or https://huggingface.co). */
  hfEndpoint?: string
  /** Number of GPU layers to offload (-ngl). Defaults to 99 (auto-falls back to 0 CPU on failure). */
  gpuLayers?: number
  /** Summarizer GGUF filename (relative to <storeRoot>/models) or absolute path. */
  summarizerModel?: string
  /** Judge GGUF filename (relative to <storeRoot>/models) or absolute path. */
  judgeModel?: string
  /** Judge linear head JSON filename (relative to <storeRoot>/models) or absolute path. */
  judgeHead?: string
  /** Embedding GGUF filename (relative to <storeRoot>/models) or absolute path. */
  embeddingModel?: string
  /** Write human-readable timeline.txt alongside guard.sqlite. Defaults to true. */
  writeTimelineTxt?: boolean
  /** Background debounce interval in ms. Defaults to 300. */
  debounceMs?: number
  /** Enable mid-tool-chain context-guard intervention reminders. Defaults to true. */
  enableIntervention?: boolean
  /** Per-action p_degraded threshold counted as one hit. Defaults to 0.50. */
  degradedThreshold?: number
  /** Sliding window of recent action steps for counting hits. Defaults to 16. */
  hitWindowSteps?: number
  /** Number of hits in the window (with current action hitting) required to trigger an intervention. Defaults to 2. */
  hitsToTrigger?: number
  /** Quiet action-step cooldown after session start, /compact, a real user message, or an intervention. Defaults to 8. */
  cooldownSteps?: number
  /** Escalate to full-prefix compaction via ctx.compaction.compactRegion after repeated interventions in one turn. Defaults to true. */
  enableAutoCompact?: boolean
  /** Number of reminders issued within a single tool-call chain before the next degradation hit triggers auto-compact. Defaults to 2. */
  interventionsBeforeCompact?: number
}

export const Config: z<Config> = z.object({
  storeRoot: z.string(),
  llamaServerPath: z.string(),
  llamaBuild: z.string(),
  autoDownloadRuntime: z.boolean(),
  autoDownloadModels: z.boolean(),
  checkAssetsOnStartup: z.boolean(),
  hfRepo: z.string(),
  hfRevision: z.string(),
  hfEndpoint: z.string(),
  gpuLayers: z.number(),
  summarizerModel: z.string(),
  judgeModel: z.string(),
  judgeHead: z.string(),
  embeddingModel: z.string(),
  writeTimelineTxt: z.boolean(),
  debounceMs: z.number(),
  enableIntervention: z.boolean(),
  degradedThreshold: z.number(),
  hitWindowSteps: z.number(),
  hitsToTrigger: z.number(),
  cooldownSteps: z.number(),
  enableAutoCompact: z.boolean(),
  interventionsBeforeCompact: z.number(),
})

export function resolveStoreRoot(config: Config): string {
  const root = config.storeRoot === undefined
    ? join(homedir(), '.dsh', 'context-cortex')
    : normalize(config.storeRoot)
  if (!isAbsolute(root)) throw new Error('storeRoot must be an absolute path')
  return root
}

export function resolveModelPath(storeRoot: string, configured: string | undefined, defaultFile: string): string {
  const value = configured ?? defaultFile
  return isAbsolute(value) ? normalize(value) : join(storeRoot, 'models', value)
}
