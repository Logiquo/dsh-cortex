# __TITLE__

__DESCRIPTION__

## Develop

From the workspace root:

```sh
npm run typecheck -w __PACKAGE__
npm run build -w __PACKAGE__
```

Implement behavior in `src/index.ts`. Declare consumed services in `inject` and
use `ctx.effect` for resources requiring cleanup.

## Install

After building, run from the workspace root with dsh installed:

```sh
dsh plugin --profile demo add ./plugins/__NAME__
dsh --profile demo --dump-config
dsh --profile demo
```

The bundled `cordis.patch.yml` activates the plugin with schema defaults.

## Configuration

| Key | Default | Meaning |
| --- | --- | --- |
| `enabled` | `true` | Turn the plugin on or off. |

## Package

```sh
npm pack -w __PACKAGE__
```

The prepack hook builds the published entry points. Install the resulting tarball
with `dsh plugin --profile demo add ./path/to/package.tgz`.

## License

MIT
