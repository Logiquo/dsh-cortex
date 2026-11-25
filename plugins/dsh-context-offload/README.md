# dsh-context-offload

Forward-looking compaction briefs, session-scoped history archives, and built-in
history search/read tools for DeepSeek Harness (`>=0.2.2-rc.1 <0.3.0`).

The plugin subclasses `BasicCompactionEngine`, inheriting its dynamic
model-`contextWindow` trigger policy, `toolResultPruner` pre-pruning,
model-switch compaction, `context-overflow` recovery, manual `/compact`,
tool-pairing balance, and checkpoint transaction format.

## Install

Requires Node.js 24.9+ and the standard `dsh` composition (including session
query, commands, and tools).

```sh
npm install
npm run build

dsh plugin --profile demo add ./plugins/dsh-context-offload
dsh --profile demo --dump-config
dsh --profile demo
```

The bundle disables the standard `compaction-basic` entry and mounts this
backend. The host `/compact` command continues to work.

## Configuration

All compaction thresholds and retention ratios default directly to `dsh`'s
`BasicCompactionEngine` policy (derived dynamically from each model's
`contextWindow` via `ctx.llm.resolveModelInfo`). Standard
`BasicCompactionConfig` options (such as `thresholdRatio`, `retainRatio`,
`headroomTokens`, `modelPolicies`) may optionally be passed through.

```yaml
- id: dsh-context-offload
  config:
    # spill-root: optional absolute persistent directory (defaults to ~/.dsh/context-offload)
    # thresholdRatio: 0.8
    # retainRatio: 0.16
```

## Forward-looking execution briefs

When compacting a main-agent session, `dsh-context-offload` archives the full
canonical event log (with chunk integrity verification) and generates a
forward-looking execution brief focused on:

1. **Task & Constraints** — governing request, active constraints, permissions, and latest user corrections.
2. **Current State** — files modified/created, exact paths/symbols/values, unresolved failures, and observed vs. unverified status.
3. **Dead Ends & Decisions** — ruled-out approaches and key decisions so the fresh instance avoids repeating failed work.
4. **Next Action & Plan** — immediate next step, remaining steps in dependency order, and completion checks.
5. **Archive Pointers** — optional search keywords for bulky offloaded details.

Prior checkpoints are consolidated into a single fresh brief rather than nested.
Reasoning blocks are excluded from the persisted brief.

## Built-in history tools

Once a main-agent session has been compacted (`replaceGeneration > 0`), two
read-only, concurrency-safe tools activate automatically (consuming zero tool
schema tokens prior to compaction):

- `history_search`: keyword or regex search over the current session's full
  canonical event log (newest-first by default, filterable by `eventTypes` and
  `fromSeq`/`toSeq`, returning bounded snippets and event `seq`s).
- `history_read`: read exact canonical event ranges (`fromSeq`..`toSeq`, up to
  20 events per call) with `maxChars` and `charOffset` pagination.

## Archive storage

The default archive root is `join(homedir(), '.dsh', 'context-offload')`:

```text
<archive-root>/session-<hash>/
  catalog.json
  <random>-context-offload-c000001-turn000010-events000000-000150.txt
  brief-<transaction-hash>.md
```

Background exports use an in-memory verified catalog cache and incremental
hashing for low overhead; compaction commits and `/context-offload verify`
perform full chunk checksum and hash-chain verification.

## Commands

- `/compact` or `/context-offload compact`: manually compact an idle session.
- `/context-offload status`: show resolved model `contextWindow`, active policy ratios, archive coverage, and latest export error.
- `/context-offload verify`: verify catalog continuity and chunk checksums.

## Validation

```sh
npm run typecheck
npm test
```
