# dsh-cortex

Minimal workspace for DeepSeek Harness host plugins. Each plugin is an independent
npm package and a dsh bundle; the root is private.

Requires Node.js **24.9+** and npm. This small workspace uses npm; the upstream
Harness source repository uses pnpm and has its own build pipeline.

## Create a plugin

```sh
npm install
npm run new -- dsh-hello --scope @you --title "Hello"
npm install
npm run typecheck
npm run build
```

The generator creates `plugins/dsh-hello` from `templates/plugin` and copies the MIT
license. The folder matches the supplied package basename (`dsh-hello`). Scoped
packages use that same basename as their folder because a scope contains a path
separator. Omit `--scope` to name the package `dsh-hello`. Optional `--description`
sets the package description.

```text
plugins/<package-name>/
  src/index.ts       # Cordis name, Config schema, inject, apply
  package.json       # ESM exports and dsh.bundle manifest
  cordis.patch.yml   # inserts the plugin by package name
  tsconfig.json      # extends the root TypeScript settings
  README.md
  LICENSE
```

Add behavior in `apply`. Declare consumed services in `inject`, and add their
packages to both peerDependencies and devDependencies when they must share the
host's instance. Use `ctx.effect` for resources that need cleanup. The starter
only needs Cordis and Schemastery; add agent, tool, or LLM APIs when needed.

## Load locally

With the dsh CLI installed, build first and link the plugin into a profile:

```sh
dsh plugin --profile demo add ./plugins/dsh-hello
dsh --profile demo --dump-config
dsh --profile demo
```

The bundle's `cordis.patch.yml` activates the package. Rebuild after source edits;
restart dsh to load the new output.

## Package

```sh
npm pack --workspace @you/dsh-hello
```

`prepack` builds the plugin before creating the tarball. Its published files
include JavaScript, declarations, the bundle patch, README, and license.
This template targets local links and built tarballs or registry packages.
Git installs of individual monorepo subdirectories are not supported.

## Planned plugins

- [`dsh-ephemeral-reasoning`](plugins/dsh-ephemeral-reasoning/README.md):
  version-pinned append-time sanitation and meter correction, with real-provider tests.

## References

- [Harness development guide](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/development.md)
- [Your first plugin](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/user/develop/basic/index.md)
- [Configuration](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/user/develop/basic/config.md)
- [Packaging and installation](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/user/develop/basic/publish.md)

Local reference material belongs in the ignored `garbage-do-not-search/` directory.

## License

MIT
