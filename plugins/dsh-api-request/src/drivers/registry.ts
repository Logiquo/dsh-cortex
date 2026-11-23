import type { Context } from '@deepseek-ai/cordis'
import llmPackage from '@deepseek-ai/dsh-llm/package.json' with { type: 'json' }
import { dsh022Rc5Driver } from './dsh-0.2.2-rc5.js'
import type { HostDriver } from './types.js'

const DRIVERS: readonly HostDriver[] = [
  dsh022Rc5Driver,
]

export async function resolveHostDriver(ctx: Context): Promise<{ driver: HostDriver; dshVersion: string }> {
  const dshVersion = llmPackage.version

  // Exact version match first
  const candidates = DRIVERS.filter(d => d.dshVersions.includes(dshVersion))
  if (candidates.length === 0) {
    const supported = DRIVERS.flatMap(d => d.dshVersions).join(', ')
    throw new Error(
      `dsh-api-request: unsupported DSH version "${dshVersion}". Supported versions: [${supported}]. No unofficial API patches were installed.`,
    )
  }

  const reasons: string[] = []
  for (const driver of candidates) {
    const probe = await driver.probe(ctx)
    if (probe.ok) return { driver, dshVersion }
    reasons.push(`${driver.id}: ${probe.reason ?? 'probe failed'}`)
  }

  throw new Error(
    `dsh-api-request: host compatibility probe failed for DSH ${dshVersion}: ${reasons.join('; ')}. Refusing to install unverified patches.`,
  )
}
