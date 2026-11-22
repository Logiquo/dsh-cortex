import LlmRuntime from '@deepseek-ai/dsh-llm'
import { TokenMeter } from '@deepseek-ai/dsh-token-meter'
import agentPackage from '@deepseek-ai/dsh-agent/package.json' with { type: 'json' }
import sessionPackage from '@deepseek-ai/dsh-session/package.json' with { type: 'json' }
import llmPackage from '@deepseek-ai/dsh-llm/package.json' with { type: 'json' }
import meterPackage from '@deepseek-ai/dsh-token-meter/package.json' with { type: 'json' }

export const SUPPORTED_DSH_VERSION = '0.2.2-rc.5'
const versions = {
  '@deepseek-ai/dsh-agent': agentPackage.version,
  '@deepseek-ai/dsh-session': sessionPackage.version,
  '@deepseek-ai/dsh-llm': llmPackage.version,
  '@deepseek-ai/dsh-token-meter': meterPackage.version,
}

/** Peer/engine declarations are also checked here; direct loading can bypass installers. */
export function assertVersions(actual: Readonly<Record<string, string>> = versions): void {
  for (const name of Object.keys(versions)) {
    if (actual[name] !== SUPPORTED_DSH_VERSION) {
      throw new Error(`Ephemeral reasoning requires ${name}@${SUPPORTED_DSH_VERSION}; found ${actual[name] ?? 'missing'}. No unofficial API patches were installed. Revalidate this plugin before upgrading dsh.`)
    }
  }
}

function assertMethod(actual: unknown, expected: unknown, label: string): void {
  // Separate identical installations can legitimately have different function identities.
  if (typeof actual !== 'function' || typeof expected !== 'function' || Function.prototype.toString.call(actual) !== Function.prototype.toString.call(expected)) {
    throw new Error(`Ephemeral reasoning: unrecognized or already wrapped ${label}; refusing to patch an unverified implementation (tested dsh ${SUPPORTED_DSH_VERSION}).`)
  }
}

export function assertMeterCompatible(meter: TokenMeter): void {
  // Cordis binds service methods on property access; inspect their unbound descriptor.
  const own = Object.getOwnPropertyDescriptor(meter, 'measure')
  const descriptor = own ?? Object.getOwnPropertyDescriptor(Object.getPrototypeOf(meter), 'measure')
  assertMethod(descriptor?.value, TokenMeter.prototype.measure, 'TokenMeter.measure')
  if (!Object.isExtensible(meter)) throw new Error('Ephemeral reasoning: token meter is not extensible')
}

export function assertLlmCompatible(llm: LlmRuntime): void {
  const own = Object.getOwnPropertyDescriptor(llm, 'forAdapter')
  const descriptor = own ?? Object.getOwnPropertyDescriptor(Object.getPrototypeOf(llm), 'forAdapter')
  assertMethod(descriptor?.value, Reflect.get(LlmRuntime.prototype, 'forAdapter'), 'LlmRuntime.forAdapter')
  if (!Object.isExtensible(llm)) throw new Error('Ephemeral reasoning: llm service is not extensible')
}
