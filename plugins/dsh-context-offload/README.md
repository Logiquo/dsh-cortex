# dsh-context-offload

Forward-looking compaction briefs, session-scoped SQLite history archives (`offload.sqlite`), and built-in history search/read tools for DeepSeek Harness (`>=0.2.2-rc.1 <0.3.0`).

The plugin subclasses `BasicCompactionEngine`, inheriting its dynamic model-`contextWindow` trigger policy, `toolResultPruner` pre-pruning, model-switch compaction, `context-overflow` recovery, manual `/compact`, tool-pairing balance, and checkpoint transaction format.

## Install

Requires Node.js 24.9+ (uses built-in `node:sqlite`) and the standard `dsh` composition (including session query, commands, and tools).

```sh
npm install
npm run build

dsh plugin --profile demo add ./plugins/dsh-context-offload
dsh --profile demo --dump-config
dsh --profile demo
```

The bundle disables the standard `compaction-basic` entry and mounts this backend. The host `/compact` command continues to work directly through this plugin.

## Configuration

All compaction thresholds and retention ratios default directly to `dsh`'s `BasicCompactionEngine` policy (derived dynamically from each model's `contextWindow` via `ctx.llm.resolveModelInfo`). Standard `BasicCompactionConfig` options (such as `thresholdRatio`, `retainRatio`, `headroomTokens`, `modelPolicies`) may optionally be passed through.

```yaml
- id: dsh-context-offload
  config:
    # storeRoot: optional absolute persistent directory (defaults to ~/.dsh/context-cortex)
    # thresholdRatio: 0.8
    # retainRatio: 0.16
```

## SQLite archive storage (`offload.sqlite`)

The default shared cortex root is `join(homedir(), '.dsh', 'context-cortex')`:

```text
<store-root>/session-<hash>/
  offload.sqlite    # owned by dsh-context-offload (WAL mode)
```

Tables inside `offload.sqlite`:
- `meta`: `schema_version`, `session_id`, `through_id`, `chain_sha256`, `updated_at`.
- `events`: `id INTEGER PRIMARY KEY` (1:1 with canonical `dsh` `event.seq`), `type`, `turn`, `step`, `time_ms`, `text`, `text_len`, `raw_json`, `sha256`.
- `compact`: recorded compaction briefs (`id`, `compaction_id`, `through_id`, `turn`, `provider`, `model`, `body`, `framed_text`, `created_at`).

When opened on an existing session that has not yet been indexed (or after `/context-offload rebuild`), `offload.sqlite` is built directly from `dsh`'s canonical session history (`sessionQuery.observeSession(..., { projectionMode: 'none' })`).

## Built-in history tools

Once a main-agent session has been compacted (`replaceGeneration > 0`), two read-only, concurrency-safe tools activate automatically (consuming zero tool schema tokens prior to compaction):

- `history_search`: unified search over `offload.sqlite` (`mode`: `"hybrid"` default, `"lexical"`, or `"semantic"`). Runs streaming exact/regex search over `events` and, when `dsh-context-guard` is mounted (`ctx.contextGuard.semanticSearch`), automatically fuses non-overlapping semantic micro-card matches in the same call.
- `history_read`: `O(1)` primary-key range read (`fromId`..`toId`, up to 20 events per call) with `maxChars` and `charOffset` pagination.

## Commands

- `/compact`: manually compact an idle session (handled natively by `dsh-context-offload`).
- `/context-offload status`: show resolved model `contextWindow`, active policy ratios, `offload.sqlite` path, `archivedThrough`, `eventCount`, `compactions`, and latest export error.
- `/context-offload verify`: verify SQLite integrity (`PRAGMA quick_check`), contiguous `events.id`, and full `sha256` hash-chain continuity.
- `/context-offload rebuild`: rebuild the current session's `events` table from canonical `dsh` history.

## Validation

```sh
npm run typecheck
npm test
```
