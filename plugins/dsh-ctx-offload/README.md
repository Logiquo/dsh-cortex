# dsh-ctx-offload

Early compaction for DeepSeek Harness **0.2.2-rc.1**, with forward-looking execution
briefs and searchable session-scoped history. The plugin subclasses dsh's basic
compactor, preserving its transactions, tool balancing, manual maintenance, and
checkpoint format.

## Install

Requires Node.js 24.9+ and the standard dsh composition, including session query,
commands, and ordinary subagent/file tools.

```sh
npm install
npm run build

dsh plugin --profile demo add ./plugins/dsh-ctx-offload
dsh --profile demo --dump-config
dsh --profile demo
```

The bundle disables the standard `compaction-basic` entry and mounts this backend.
The host `/compact` command continues to work. Custom compositions must mount one
compaction backend and provide the dependencies declared in `package.json`.

Profile configuration:

```yaml
- id: dsh-ctx-offload
  config:
    context-limit: 100000
    # spill-root: optional absolute persistent directory
```

Compaction is automatic while the plugin is loaded. `context-limit` is this plugin's
context budget and defaults to 100,000 tokens, independently of model-advertised
capacity. `spill-root` defaults to `join(homedir(), '.dsh', 'ctx-offload')`.

The internal budget formulas are:

```text
triggerTokens      = floor(context-limit * 0.75)
briefMaxTokens     = min(max(floor(triggerTokens * 0.05), 1024), triggerTokens)
retainRecentTokens = min(max(floor(triggerTokens * 0.10), 4096), triggerTokens)
```

The default produces a 75,000-token trigger, a 3,750-token brief limit, and a
7,500-token recent-history limit. Both content budgets are upper limits, not fill
targets; the clamp keeps a small `context-limit` from reserving more than the trigger
it is carved from. The brief includes archive guidance, which is subtracted from the
brief limit before generation. `briefMaxTokens` bounds only the text that replaces
history; the generation call takes the route's own response ceiling, so the model's
configured reasoning effort does not consume the brief budget or truncate the reply.
Reasoning blocks are neither counted nor retained. Automatic and overflow selection retain
the longest balanced recent suffix that fits, possibly none; larger complete
messages/tool pairs enter the compacted region. Manual compaction uses the host's
range selection and rejects a retained last message that exceeds the recent-history
limit. Malformed briefs are retried once. dsh's native shrinkage check requires the
framed replacement to be smaller than its source region.

Provider request validation and actual model capacity remain governed by dsh; this
setting controls the plugin's compaction policy.

## History and retrieval

The default archive root is `join(homedir(), '.dsh', 'ctx-offload')`, using the current
user's home and native Windows/macOS/Linux paths. Each session has a stable directory:

```text
<archive-root>/session-<hash>/
  catalog.json
  <random>-ctx-offload-c000001-turn000010-events000000-000150.txt
  <random>-ctx-offload-c000002-turn000025-events000151-000370.txt
  brief-<transaction-hash>.md
```

The plugin calls the public `saveTextFile` helper with its own root. The shared
`ctx.spillStore` configuration and cleanup policy remain unchanged.

Archive chunks contain readable text and canonical accepted events, including records
shadowed by earlier compactions. `catalog.json` tracks committed chunks, checksums,
coverage, and a canonical-prefix hash. Background export batches events, session
flushes await archive coverage, and each compaction archives its canonical input
before generating the replacement brief. File publication uses host atomic-write
and cross-process locking helpers, plus file sync and POSIX directory sync.

Each brief includes the actual session directory and instructions to delegate missing
historical questions to a fresh search agent. That agent uses normal `rg`/`grep` and
bounded reads, checks corrections, and returns short findings with file/line citations.
Retrieval is chosen by the main model through ordinary delegation. The plugin launches
no retrieval calls itself. Host permissions and normal delegation limits apply.

With `dsh-ephemeral-reasoning`, the archive exports accepted records after sanitation;
it does not collect live reasoning streams. Attachments remain references and metadata
in the text archive. Keep the original attachment store for access to their bytes.

## Commands

- `/compact` or `/ctx-offload compact`: manually compact an idle session.
- `/ctx-offload status`: show settings, archive coverage, and the latest export error.
- `/ctx-offload verify`: verify catalog continuity and chunk checksums.

## Validation and current limits

```sh
npm run typecheck
npm test
```

- Real-model brief quality and voluntary retrieval behavior require provider evaluation.
  Validation checks nonempty text and size; semantic completeness requires model evaluation.
- Windows/macOS paths use portable APIs; execution and crash durability were exercised
  on Linux. Windows directory fsync is unavailable through the portable Node API.
- Archives remain across disposal and resume. The inspected host exposes no public
  session-deletion event, so automatic archive deletion is not implemented. Remove
  the dedicated session directory when permanently deleting its session.
- Main-session forks backfill their inherited canonical prefix into their own archive.
  Existing inherited brief paths continue to reference the source archive until a new
  brief is generated. Retain source archives while such references are in use.
- A changed or shorter canonical prefix is rejected. After an unclean host-persistence
  failure, preserve the archive and reconcile it with the host log before retrying;
  the plugin does not silently discard an archived prefix.
- Archive snapshots/checksums currently scan full history. Large-session performance
  needs measurement before broad rollout.
- Session-private means user-private files grouped by session. Access uses dsh's
  existing filesystem permissions. Host-code writes need no model approval; shell
  retrieval follows the deployment's normal shell policy.

See the [design](../../DESIGN.md) for the intended architecture and acceptance criteria.
