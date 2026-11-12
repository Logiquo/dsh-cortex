import { homedir } from 'node:os'
import { isAbsolute, join, normalize } from 'node:path'
import z from '@deepseek-ai/schemastery'

export interface Config {
  'context-limit': number
  'spill-root'?: string
}
export const Config: z<Config> = z.object({
  'context-limit': z.number().step(1).min(1).default(100000),
  'spill-root': z.string(),
})
export function archiveRoot(config: Config): string {
  const root = config['spill-root'] === undefined
    ? join(homedir(), '.dsh', 'ctx-offload') : normalize(config['spill-root'])
  if (!isAbsolute(root)) throw new Error('spill-root must be an absolute path')
  return root
}
