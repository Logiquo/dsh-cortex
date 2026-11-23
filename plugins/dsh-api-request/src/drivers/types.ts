import type { Context } from '@deepseek-ai/cordis'
import type { TransformPipeline } from '../pipeline.js'

export interface ProbeResult {
  readonly ok: boolean
  readonly reason?: string
}

export interface HostDriver {
  readonly id: string
  readonly dshVersions: readonly string[]
  probe(ctx: Context): Promise<ProbeResult> | ProbeResult
  install(ctx: Context, pipeline: TransformPipeline): Promise<() => void> | (() => void)
}
