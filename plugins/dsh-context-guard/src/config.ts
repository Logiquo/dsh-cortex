import { homedir } from 'node:os'
import { isAbsolute, join, normalize } from 'node:path'
import z from '@deepseek-ai/schemastery'

export const DEFAULT_LLAMA_BUILD = 'b12651'
export const DEFAULT_HF_REPO = 'Logic-z/dsh-cortex'
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
  /** Enable mid-tool-chain context-guard checkpoint injection. Defaults to true. */
  enableNudge?: boolean
  /** Per-action p_degraded threshold counted as one hit. Defaults to 0.50. */
  degradedThreshold?: number
  /** Sliding window of recent action steps for counting hits. Defaults to 16. */
  hitWindowSteps?: number
  /** Number of hits in the window (with current action hitting) required to trigger a checkpoint. Defaults to 2. */
  hitsToTrigger?: number
  /** Quiet action-step cooldown after session start, /compact, a real user message, or a checkpoint. Defaults to 8. */
  cooldownSteps?: number
}

export const Config: z<Config> = z.object({
  storeRoot: z.string(),
  llamaServerPath: z.string(),
  llamaBuild: z.string(),
  autoDownloadRuntime: z.boolean(),
  autoDownloadModels: z.boolean(),
  checkAssetsOnStartup: z.boolean(),
  hfRepo: z.string(),
  hfEndpoint: z.string(),
  gpuLayers: z.number(),
  summarizerModel: z.string(),
  judgeModel: z.string(),
  judgeHead: z.string(),
  embeddingModel: z.string(),
  writeTimelineTxt: z.boolean(),
  debounceMs: z.number(),
  enableNudge: z.boolean(),
  degradedThreshold: z.number(),
  hitWindowSteps: z.number(),
  hitsToTrigger: z.number(),
  cooldownSteps: z.number(),
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
