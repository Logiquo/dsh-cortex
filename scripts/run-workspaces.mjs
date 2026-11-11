#!/usr/bin/env node
import { readdirSync, existsSync } from 'node:fs'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { spawnSync } from 'node:child_process'

const root = fileURLToPath(new URL('../', import.meta.url))
const task = process.argv[2]
if (!['build', 'typecheck'].includes(task)) throw new Error('Expected build or typecheck')
const plugins = readdirSync(join(root, 'plugins'), { withFileTypes: true })
if (!plugins.some((entry) => entry.isDirectory() && existsSync(join(root, 'plugins', entry.name, 'package.json')))) {
  console.log('No plugins yet. Create one with: npm run new -- hello')
} else {
  // Reuse npm's JS entry point to avoid platform-specific shell invocation.
  const result = spawnSync(process.execPath, [process.env.npm_execpath, 'run', task, '--workspaces', '--if-present'], {
    cwd: root,
    stdio: 'inherit',
  })
  if (result.error) throw result.error
  process.exitCode = result.status ?? 1
}
