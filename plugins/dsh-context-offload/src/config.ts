import { homedir } from 'node:os'
import { isAbsolute, join, normalize } from 'node:path'
import z from '@deepseek-ai/schemastery'
import { BasicCompactionEngine } from '@deepseek-ai/dsh-compaction-basic'
import type { BasicCompactionConfig } from '@deepseek-ai/dsh-compaction-basic'

export interface Config extends BasicCompactionConfig {
  storeRoot?: string
  'spill-root'?: string
}

export const Config: z<Config> = z.intersect([
  BasicCompactionEngine.Config,
  z.object({
    storeRoot: z.string(),
    'spill-root': z.string(),
  }),
])

export function archiveRoot(config: Config): string {
  const configured = config.storeRoot ?? config['spill-root']
  const root = configured === undefined
    ? join(homedir(), '.dsh', 'context-cortex')
    : normalize(configured)
  if (!isAbsolute(root)) throw new Error('storeRoot must be an absolute path')
  return root
}

/** Extract only standard BasicCompactionConfig keys so BasicCompactionEngine's strict key validation accepts it. */
export function basicCompactionConfig(config: Config): BasicCompactionConfig {
  const { storeRoot: _storeRoot, 'spill-root': _spillRoot, ...rest } = config
  const out: Record<string, unknown> = {}
  for (const [key, value] of Object.entries(rest)) {
    if (value !== undefined) out[key] = value
  }
  return out as BasicCompactionConfig
}
