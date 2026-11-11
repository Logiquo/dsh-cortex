#!/usr/bin/env node
// Scaffold a new plugin from templates/plugin.
//
//   npm run new -- <name> [--scope @you] [--title "Title"] [--description "..."]
//
// <name> is the Cordis plugin id (kebab-case, e.g. "loop-guard").
// The package is published as <scope>/dsh-<name>, or dsh-<name> without a scope.
import { cpSync, existsSync, readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { parseArgs } from 'node:util'

const root = join(dirname(fileURLToPath(import.meta.url)), '..')

const { values, positionals } = parseArgs({
  allowPositionals: true,
  options: {
    scope: { type: 'string' },
    title: { type: 'string' },
    description: { type: 'string' },
  },
})

const name = positionals[0]
if (positionals.length !== 1 || !name || !/^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$/.test(name)) {
  console.error('usage: npm run new -- <kebab-case-name> [--scope @you] [--title "..."] [--description "..."]')
  process.exit(1)
}

const scope = values.scope ? values.scope.replace(/^@?/, '@').replace(/\/$/, '') : ''
if (scope && !/^@[a-z0-9][a-z0-9-]*$/.test(scope)) {
  console.error('scope must be an npm scope, e.g. @you')
  process.exit(1)
}
const pkg = scope ? `${scope}/dsh-${name}` : `dsh-${name}`
const title = values.title ?? name.split('-').map((w) => w[0].toUpperCase() + w.slice(1)).join(' ')
const description = values.description ?? `${title} plugin for dsh.`

const dest = join(root, 'plugins', name)
if (existsSync(dest)) {
  console.error(`plugins/${name} already exists`)
  process.exit(1)
}

cpSync(join(root, 'templates', 'plugin'), dest, { recursive: true })
cpSync(join(root, 'LICENSE'), join(dest, 'LICENSE'))

const vars = { __NAME__: name, __PACKAGE__: pkg, __TITLE__: title, __DESCRIPTION__: description }
const fill = (dir) => {
  for (const entry of readdirSync(dir)) {
    const path = join(dir, entry)
    if (statSync(path).isDirectory()) { fill(path); continue }
    let text = readFileSync(path, 'utf8')
    text = text.replace(/__(?:NAME|PACKAGE|TITLE|DESCRIPTION)__/g, (key) => {
      const value = vars[key]
      return path.endsWith('.json') ? JSON.stringify(value).slice(1, -1) : value
    })
    writeFileSync(path, text)
  }
}
fill(dest)

console.log(`created plugins/${name} (${pkg})`)
console.log('next: npm install && npm run build')
