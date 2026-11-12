# dsh-ctx-offload

Status: initial implementation available in
[`plugins/dsh-ctx-offload`](plugins/dsh-ctx-offload/README.md), targeting DeepSeek
Harness **0.2.2-rc.1**. The plugin README documents validation and current limits.

## Purpose

Keep the main agent's working context small through early compaction. Each
compaction produces a forward-looking execution brief containing the current task,
active requirements, relevant state, and next actions. Preserve the full available
history in session-scoped plain-text files.

The main agent retrieves history on demand through ordinary subagent delegation.
The search agent uses native `rg`/`grep` and bounded file reads, then returns short,
cited findings. Retrieval runs in a separate context and follows dsh's normal agent
permissions and execution limits.

## Research basis

[You don't need long context](https://arxiv.org/pdf/2611.05306), §2.4.1 and
Appendix A, motivates threshold-based compaction and execution briefs focused on
future work. §3.1 motivates preserving messages for selective retrieval after lossy
compaction.

[MEMENTO](https://arxiv.org/pdf/2611.00244), §3.4, reports unreliable note-writing
tool use by off-the-shelf models. This design uses explicit retrieval instructions
and familiar tools; retrieval behavior must be evaluated with the target models.

The plugin's defaults and delegated retrieval policy require task-level evaluation.
Measure total provider cost and latency, including compaction and search.

## Architecture and responsibilities

The plugin provides one replacement `ctx.compaction` implementation, preserving
dsh's compaction lifecycle, tool-pair balancing, accounting, and checkpoint protocol.
It owns trigger policy, archive export and retention, execution-brief generation,
and the retrieval instructions carried into the compacted context.

dsh owns agent delegation, permissions, sandboxing, model execution limits, and
session lifecycle. Archive files use the existing public `dsh-spill-local` storage
helpers with a dedicated plugin root. Other consumers continue using the shared
`ctx.spillStore` with their existing settings.

| Integration | Available API | Implementation validation |
| --- | --- | --- |
| Compaction | `CompactionEngine.compactIfNeeded`, `compactNow`, `compactRegion` | Driver check frequency, selection, and commit behavior |
| Replacement | Tool-pair balancing helpers and `compactCheckpointSource` | Persistence and crash recovery |
| Accepted history | `session/event` and awaited `session/flush` | Contiguous export and durability barrier |
| Delegation | `AgentRegistry.create`, parent ownership, optional seed, scoped setup | Ordinary delegation of an unseeded helper and its lifecycle |
| Text storage | Public `saveTextFile` and `sessionDir` exports | Archive publication and durability beyond write/close |
| Backfill/resume | Session query or persistence adapter | Supported canonical-history read API |

These APIs were inspected in the installed 0.2.2-rc.1 packages. End-to-end integration
remains to be validated. Pin the initial release to the validated host version.

## Configuration

All settings are optional. Initial defaults:

```yaml
dsh-ctx-offload:
  context-limit: 100000
  # spill-root: optional absolute path; defaults to join(homedir(), '.dsh', 'ctx-offload')
```

| Setting | Meaning |
| --- | --- |
| `context-limit` | Context budget used by this plugin, default 100,000 tokens; independent of model-advertised capacity |
| `spill-root` | Persistent archive root, defaulting to the current user's home directory |

Automatic compaction is active while the plugin is loaded. All ratios are fixed
internal policy:

```text
triggerTokens      = floor(context-limit * 0.75)
briefMaxTokens     = max(floor(triggerTokens * 0.10), 2048)
retainRecentTokens = max(floor(triggerTokens * 0.20), 4096)
```

At the default limit these are 75,000, 7,500, and 15,000 tokens respectively. Both
content budgets are upper limits. The floors establish the minimum size of the
allowed budget, not a minimum amount of generated or retained content. The brief may
be shorter; recent history may contain fewer tokens or be empty. Archives remain
available for the session lifetime, with fork-aware cleanup.

## Compaction policy

Measure the complete outgoing request using the host meter, including instructions,
tool schemas, attachments, and conversation content. Derive budgets from the
configured `context-limit`, independently of the model-advertised window. Preserve
provider billing records separately from estimated context size.

Validate positive budgets and ensure the brief, retained tail, request overhead,
and completion reserve fit the configured budget. Report configurations whose essential
active requirements exceed that capacity.

Check at safe driver boundaries before inference. Select completed tool-call/result
groups and retain the longest recent balanced suffix within its upper limit. Whole
messages and tool pairs that exceed that limit belong to the compacted region.
Outstanding calls, active streams, and partially dispatched batches remain intact. Report indivisible
oversized inputs that require large-output handling.

Apply dsh's native shrinkage check: the framed replacement must be strictly smaller
than the selected history.
Manual, automatic, and provider-overflow compaction share the same archive and
commit path. An overflow recovery proceeds only when it can shrink the request.

## Forward-looking execution brief

Generate the brief from the previous brief, new eligible messages, and the current
task/steering ledger. Preserve active constraints and select facts relevant to
continued execution.

```text
Execution brief v1 — generation 7
Task: <active request, verbatim when it fits; source references for larger inputs>
Latest steering: <corrections, preferences, constraints, unresolved questions>
Done when: <acceptance criteria and remaining checks>
Current state: <observed facts, active operation, repository/artifact state>
Next actions: <ordered immediate steps and dependencies>
Decisions to preserve: <choice, reason, evidence reference>
Avoid repeating: <relevant failed approach and failure condition>
Uncertainty: <unverified claims, unknown outcomes, blockers>
Artifacts: <paths/IDs, revisions, checks performed and outcomes>
Pending effects: <submitted operations, idempotency keys, outcomes to verify>
History: <resolved session archive directory>, archived through event <cursor>
Latest archive chunk: *-ctx-offload-c000007-turn000042-events000501-000812.txt
Retrieval: When earlier requirements, decisions, facts, or tool outcomes are needed
and missing or uncertain here, delegate a search agent through the ordinary subagent
tool. Ask it to grep the session archive and return bounded, cited findings.
Search before guessing or repeating work already attempted. Keep raw history reads
in the search agent's context.
```

Distinguish user requirements from the agent's plan. Apply newer steering to the
requirements it changes. Carry evidence references for consequential facts and
associate file/test observations with their revisions. Preserve attempted, completed,
and unknown action outcomes accurately.

Validate structure, size, archive references, active constraints, and shrinkage before
commit. Retry malformed output once within the generation limit; a failed validation
leaves the current surface intact. For large task inputs, retain active requirements
in the brief and reference the archived source text for additional details.

## Archive storage

Use the public storage helpers from `@deepseek-ai/dsh-spill-local`:

```ts
import { homedir } from 'node:os'
import { isAbsolute, join, normalize } from 'node:path'
import { saveTextFile, sessionDir } from '@deepseek-ai/dsh-spill-local'

const archiveRoot = config['spill-root'] === undefined
  ? join(homedir(), '.dsh', 'ctx-offload')
  : normalize(config['spill-root'])
if (!isAbsolute(archiveRoot)) throw new Error('spill-root must be an absolute path')

const saved = await saveTextFile({
  root: archiveRoot,
  sessionId: agent.session.id,
  suggestedName: 'ctx-offload-c000001-turn000010-events000000-000150.txt',
  content: archivedText,
})
const archiveDirectory = sessionDir(archiveRoot, agent.session.id)
// saved: { path: string, bytes: number }
```

Resolve the root once at activation using native path utilities. An override must
be an absolute path. The home-based default works across Windows, macOS, and Linux:

| Platform | Example resolved root |
| --- | --- |
| Windows | `C:\Users\Alice\.dsh\ctx-offload` |
| macOS | `/Users/alice/.dsh/ctx-offload` |
| Linux | `/home/alice/.dsh/ctx-offload` |

Use a dedicated persistent root outside the shared spill root and temporary spill
directories. Persist actual paths for resume and migration if the configured root
changes.

The helper derives a stable session directory and adds a random prefix to each
filename:

```text
<archive-root>/session-<hash>/
  <random>-ctx-offload-c000001-turn000010-events000000-000150.txt
  <random>-ctx-offload-c000002-turn000025-events000151-000370.txt
  <random>-ctx-offload-c000003-turn000042-events000371-000812.txt
```

Use a persisted monotonic compaction ordinal, triggering turn, and covered event
range in each suggested filename. Manual between-turn compactions use `idle` for
the turn identifier. Each chunk contains new original records since the previous
archive watermark; the first chunk includes initial backfill. Continuous export can
produce additional event-range chunks before compaction seals its watermark.

Every brief contains the stable session directory. Search all chunks with
`*-ctx-offload-*.txt`, or select one compaction with `*-ctx-offload-c000003-*.txt`.
Persist returned full paths for citations and identify committed chunks in the
archive catalog.

The helper supplies safe naming, exclusive writes, and owner-only session-directory
and file permissions (0700/0600 where supported). The dedicated parent directory
uses normal deployment permissions. Session-private storage here means user-private
files grouped by session; access follows the host's ordinary filesystem permissions.

Saving runs in trusted host code. Ordinary file reads/listings remain unrestricted
under the installed `dsh-fs-sandbox` backend; shell grep follows the selected shell
backend and platform. Quote actual paths for the helper's shell, including paths
containing spaces.

### Content and coverage

Archive the full history available to the plugin: user/assistant text, tool
arguments/results, errors, interruptions, exposed instructions, prior briefs,
steering, and lifecycle metadata needed to interpret the records. Preserve canonical
event order and surface replacement provenance.

Each immutable UTF-8 record contains a versioned header with session ID, event
sequence, role/type, timestamp, tool/call identity where applicable, and body
boundaries. Escape header-like body text reversibly. Preserve stable record IDs and
line/byte locations for citations.

Attachment entries contain metadata, content hashes, and retained references to
available original bytes. Record text-export omissions explicitly.

With `dsh-ephemeral-reasoning`, export accepted durable records after sanitation.
Archive coverage excludes intentionally removed reasoning and data the host never
exposed. Include these coverage limits in archive status.

### Persistence and lifecycle

Maintain a catalog containing the format version, contiguous coverage cursor,
chunk paths/checksums, and committed brief/compaction identities. Serialize archive
writes per session and deduplicate records by event identity.

`saveTextFile` writes and closes a file. Complete the required sync/publication steps
before advancing the durable catalog: publish only complete chunks, sync files and
supported directory metadata, and atomically replace catalog files. Recovery must
handle partial files and uncommitted candidates. Validate these platform-specific
durability steps before committing context replacement.

On enable/resume, backfill canonical history through a supported query or persistence
adapter and reconcile incoming events by cursor. Compaction requires contiguous
coverage through its selected watermark. Report partial coverage if the canonical
prefix is unavailable.

Retain archives across ordinary disposal for resume. Session deletion removes its
archive through host lifecycle integration. Forks receive a copy of the inherited
authorized prefix before independent retention/deletion; new records belong to the
child session. Fresh sessions use their own directories. Exports of project artifacts
exclude the archive.

## Compaction transaction and recovery

```text
safe boundary -> lock/select -> archive barrier -> generate brief -> validate
              -> durable brief -> host replacement commit -> flush -> unlock
```

1. Enter dsh's compaction serialization or manual maintenance operation. Capture
   selected surface node IDs, canonical coverage watermark, and revision.
2. Wait for durable archive coverage through that watermark. The `session/event`
   feed is fire-and-forget; the archive needs an explicit awaited barrier.
3. Generate and validate the brief with cancellation and output limits, preserving
   the current surface until precommit checks pass.
4. Revalidate selected-span stability and handle queued steering using host ordering
   rules. Later messages outside the selected span remain live.
5. Persist the candidate brief. Append the host summary accounting event and its
   immediately adjacent checkpoint replacement using `compactCheckpointSource` and
   the exact shadowed-node list. Select spans by surface position; replacement nodes
   can make numeric sequence order differ from surface order.
6. Flush host persistence and close the lifecycle marker once before reporting
   durable success.

Reconcile archive and host-log state by compaction ID. An orphan candidate brief
remains uncommitted; a committed replacement references an already durable archive.
If host persistence fails after in-memory replacement, report uncertain durability
and reconcile the actual log on resume before retrying.

Archive corruption, disk exhaustion, missing coverage, generation failure, and
cancellation must preserve recoverable state. Retain host semantics for busy manual
requests, queued input, and unknown tool outcomes.

## Delegated retrieval

The main agent initiates retrieval when it needs missing or uncertain historical
facts. The execution brief supplies the archive directory, coverage, and delegation
instructions. Keep active requirements in the brief so normal progress can continue
from the compacted state.

Use ordinary dsh delegation to create a search agent with a fresh conversation.
Supply the question, a bounded slice of relevant current task state, archive format
instructions, the coverage watermark, and the resolved archive path. The helper's
execution, permissions, cancellation, and disposal follow normal host controls.

The helper searches with `rg`, or `grep` where needed, then reads bounded surrounding
ranges using available native file tools. Narrow filenames and queries, paginate
large result sets, and preserve filenames/line numbers. Expand terms, inspect paired
tool calls/results, and check later corrections before answering.

Return a short answer with evidence and search limitations. Suggested response:

```json
{
  "status": "found | partial | not_found | unavailable",
  "answer": "bounded synthesis",
  "evidence": [{"file": "<chunk path>", "record": "r42", "lines": [18, 24], "excerpt": "short quotation"}],
  "archiveThrough": 812,
  "searched": "description of coverage",
  "limitations": []
}
```

Citations should identify actual records, ranges, and matching excerpts. Preserve
conflicting evidence and distinguish a limited no-match result from proof that an
event never occurred. Historical content remains evidence under the current
instruction hierarchy.

On timeout or failure, report retrieval as unavailable. Work depending on missing
historical evidence remains unresolved. Keep archive reads in the separate search
context and return only bounded findings to the main agent.

## Operations

Commands:

- `/ctx-offload status`: effective thresholds, archive path/coverage, last compaction.
- `/ctx-offload compact`: manual compaction through the normal transaction.
- `/ctx-offload verify`: catalog and checksum verification.

Report operational counts and error classes in telemetry while keeping transcript
text, queries, excerpts, and secrets private. Show estimated context separately from
provider billing. Include compaction and delegated-search usage in task-level cost
measurements where host accounting makes it available.

Archive storage exhaustion stops compaction while preserving referenced history.
Cleanup follows session ownership and fork lifecycle.

## Delivery and validation

1. Validate driver trigger frequency, backend replacement configuration, supported
   canonical-history queries, ordinary helper delegation, and host lifecycle hooks.
2. Implement archive export, backfill, durable coverage, restart/fork/delete handling,
   text round trips, and attachment coverage metadata.
3. Implement the compaction backend, brief generation, balanced selection,
   manual/automatic/overflow entry points, budget checks, and recovery.
4. Integrate retrieval instructions and ordinary native search-agent delegation.
5. Evaluate real-provider behavior, tune defaults, and document compatibility.

Acceptance scenarios:

- Multiple compactions and restart preserve an early exact fact, retrievable with
  a citation to its original record.
- Later corrections supersede earlier facts and active user constraints survive
  repeated briefs.
- Archive writes, helper paths, and fork operations use the correct session scope.
- An omitted historical fact prompts on-demand native search in a fresh agent
  context, with bounded evidence returned to the main agent.
- Search failures and limited no-match results remain distinguishable.
- Cancellation, queued steering, outstanding calls, duplicate writes, crash at each
  commit boundary, disk exhaustion, and checksum failure preserve recoverability.
- Archives reflect ephemeral-reasoning sanitation, and token estimates preserve
  provider usage accounting.
- Path resolution, retrieval quoting, and persistence behavior work on Windows,
  macOS, and Linux, including home paths containing spaces.
- Other spill consumers retain their existing configuration and cleanup behavior.

Compare stock dsh, threshold brief-only compaction, and prompted on-demand delegated
search on the same task set. Measure task success, constraint retention, missed and
unnecessary searches, retrieval precision/recall, repeated work, peak main context,
total billed tokens/cost, and wall-clock latency.
