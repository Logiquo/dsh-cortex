# dsh-ephemeral-reasoning

Show reasoning while generating, but save only cleared main-agent responses.
Answers, tool calls/results, IDs, roles, timestamps, and provider billing usage
are retained. Structured reasoning is removed from both the saved assistant
message and its embedded stream, including failed and interrupted attempts;
replay state is removed only from settlements that actually carried reasoning,
since that envelope is index-aligned with the blocks it describes. Subagents are
cleaned too: reasoning kept in a child's
log is re-sent on its later steps and across its own turn boundaries, where the
provider no longer restores it, so it is billed on every subsequent round.

Supports **dsh 0.2.2-rc.1** across all provider and model routes through its
shared message/stream protocol. Real-provider validation covers
`deepseek-account / deepseek-flash`; other routes are enabled but not individually
verified.
Three reversible overrides are installed: `Session.append()`,
`TokenMeter.measure()`, and the live `contextPressure` projection definition.
No provider, event-validation, global-prototype, or display-code override is used.

## Compatibility guard

`engines.dsh` and all four dsh peer dependencies require exactly `0.2.2-rc.1`.
Activation also checks the resolved agent/session/LLM/token-meter package versions
at runtime, so direct module loading cannot silently bypass the manifest checks.
It verifies the actual unbound `Session.append` and `TokenMeter.measure` methods
against those tested implementations, and checks the `contextPressure` unit's
`stateVersion`, before installing overrides. Unknown builds, mixed versions,
methods already wrapped by another extension, and an unexpected projection
version are rejected.
There is no version-check bypass setting; revalidate and update the compatibility
guard before supporting another dsh version.

All existing session targets are checked before any patch is installed; newly
created/resumed sessions are checked before they are patched. Later wrappers
remain intact when this plugin unloads. Cordis-bound service methods are handled
through their stored descriptors for reliable cleanup.

Every rejection is decided before the first mutation. A refused version or
projection check therefore leaves the meter unpatched and the ownership marker
released, and the plugin can be mounted again after the cause is fixed. The
occupancy-ring registration wait is bounded; timing out degrades to host ring
numbers with a warning instead of refusing to start, because the meter is still
corrected. A session that cannot be wrapped is logged and left on host
behaviour rather than failing the `agent/created` that produced it.

A projection key cannot be overridden by registering it again — the registry
only increments the existing reference count and the first definition keeps
driving. The occupancy override therefore replaces the live definition in place
and drops its memoized cells, so every session refolds with the corrected fold.
It bumps the unit's `stateVersion`, so previously persisted rows are discarded
instead of seeding the corrected fold. Unloading restores the host definition
and again drops the cells.

## Install locally

```sh
npm install
npm run build
dsh plugin --profile demo add ./plugins/dsh-ephemeral-reasoning
```

Start a **new session** for reasoning-free saved history. The plugin also attaches
to already-live and resumed main agents, but cleans only new settlements: it
does not migrate reasoning saved before it was enabled.

## Configuration

| Setting | Default | Meaning |
| --- | --- | --- |
| `enabled` | `true` | Clean subsequent assistant settlements, then anchor the context total on the provider's billed count and correct it for dropped reasoning. |

Turning the plugin off restores normal saves: if it is unloaded mid-generation,
that pending response may save reasoning. Previously cleaned records stay clean
when the plugin is removed. Later wrappers are not overwritten during cleanup.

## Display and accounting

Original live chunks reach dsh's normal assistant-stream channel. Its existing
Web client keeps transient reasoning during generation, then retires it as the
cleaned response lands. Reconnecting during generation can reconstruct live
reasoning; reopening a completed cleaned response cannot. No reasoning sidecar
is stored.

Provider usage includes tokens spent generating discarded reasoning, and that
record stays untouched. Billing and usage counters remain unchanged; only the
derived context-pressure total is re-anchored.

### The measurement

```
total = billed total + appended content − dropped reasoning
```

| term | kind | what it is |
| --- | --- | --- |
| **billed total** | **exact** | The provider's count of the newest settled call. |
| **appended content** | **estimate** | Surface nodes added after that call. |
| **dropped reasoning** | **estimate** | Reasoning the provider stops resending once the turn closes. Zero while it is open. |

**Billed total — exact.** The provider's own count of the call it just billed:
`inputTokens + cacheReadTokens + cacheWriteTokens + outputTokens`. It comes from
the adapter's `usage` record stored on `assistant/message`, so every token the
provider accepted up to that call is exact — including reasoning this plugin
removed from storage. Nothing is recomputed. Source: the session log.

**Appended content — estimate.** Tool results, user text, and system or
developer updates added after that call. No server has seen these tokens, so
they are priced with dsh's fixed heuristic — `ceil(chars / 4) + 4` per block,
`4` per message — taken from the host meter's own route-priced nodes, so image
and file prices are whatever the route declares. Source: the host measurement's
nodes.

**Dropped reasoning — estimate, zero unless the turn closed.** While a turn is
open the provider resends its reasoning, so the billed total stays
authoritative. A `user/message` closes the turn; from then on the provider stops
resending that turn's reasoning and the measurement subtracts it. Per settled
message the split of the provider's single output count comes from two prices
taken under the same heuristic:

```
reasoning = outputTokens × removed / (removed + price(visible content))
```

`removed` is the price of the reasoning blocks sanitation deleted, recorded on
the settlement as `ephemeralReasoning.removedTokens`. Because both prices carry
the same per-character density, that density cancels in the ratio, so the split
does not depend on the script being space-delimited — pricing the visible text
alone would read a Chinese or Thai answer as a quarter of its real size and
report the rest as reasoning that never happened. `usage.reasoningTokens` is
used directly when an adapter reports it. A settlement written before this
evidence existed falls back to `max(0, outputTokens − price(visible))`.

When sanitation leaves an assistant message empty there is no visible price, so
the whole output counts as dropped — the same rule that removes it from the next
request. Every settled step of the closed turn contributes once, and a loop that
admits several `user/message` events in a row closes the turn once rather than
resetting the tally.

### When the anchor is dropped

A new anchor arrives with every settled call. Three events discard it and fall
back to a current-surface estimate (`tools + surface`):

- **A changed request envelope** — different route, tools, or envelope content
  bills different history.
- **A replacing surface op** — compaction and prune rewrite the billed history.
  Re-logging an identical envelope (a resumed series) keeps the anchor, since
  the same messages are still billed.
- **No settled call yet, or no logged `request/header`** — the ring anchors only
  where the meter anchors, so an unheadered session cannot report two numbers.
- **An explicit `requestHeader` that differs from the logged one** — the caller
  is billing a different route or tool set than the anchor recorded.

A replacement ends only the anchor. The open turn's reasoning tally survives,
because a replaced tool result leaves the rest of the surface in place and the
provider keeps resending that turn's reasoning; both surfaces therefore subtract
the same amount when the turn closes.

The fallback exists because the host meter's own anchor survives both cases: it
compares envelopes by content, so a compaction that re-logs an identical
envelope leaves it anchoring a pre-compaction total against a much smaller
surface.

### Trusting the provider

The host meter accepts a recorded usage anchor only when that total reaches its
own chars-per-token price of the same history. That `CHARS_PER_TOKEN = 4`
heuristic runs high on prose: on `deepseek-flash` it exceeded the billed prompt
by 3–12%, so the recorded total was rejected on all seven gated requests of an
unpatched run and the whole history was repriced.

### Known gaps

- **Appended content has no tokenizer.** dsh ships none, and there is no
  provider count endpoint, so it stays a chars/4 estimate. Its error is a few
  percent of the tail, not of the context.
- **Dropped reasoning assumes the provider discards prior-turn reasoning.**
  That is vendor behaviour, not a dsh API. Where it does not hold the
  measurement over-counts by that turn's reasoning until the next settlement.
- **The ring's surface fallback is heuristic-priced.** The occupancy projection
  has no access to the route's image and file pricing, so a session that sends
  attachments can show a slightly different fallback number than the meter while
  no anchor is in force.
- **Settlements written before this version carry no reasoning evidence.** They
  are measured with the older visible-text subtraction, which over-reports on
  dense scripts. Nothing migrates them.
- **A provider that misreports its own usage is trusted.** The anchor is the
  provider's number by design.
- **No retroactive migration.** Sessions whose reasoning was saved before this
  plugin was enabled still carry it.

The composer occupancy ring and the meter report the same total for every
session: the corrected occupancy unit is derived by the same rule that measures
the request, including the same anchoring conditions and the same no-anchor
fallback, so one session cannot show two different context numbers. Each rule is
pinned by a regression test that drives the real `TokenMeter` and the real
projection registry together. That
holds for subagent sessions too, which are sanitized and measured like any
other session. The context-composition panel is a separate
breakdown of stored content and does not sum to that total by construction.

The UI context-composition projection reads the clean saved log without
another patch. Its actual Host/client handoff and reconnect state machines are
tested; visual browser rendering has not been tested.


## Verification and scope

```sh
npm test -w dsh-ephemeral-reasoning
npm run typecheck
node --test --test-isolation=none investigations/ephemeral-reasoning/storage.test.mjs
node --test --test-isolation=none investigations/ephemeral-reasoning/display.test.mjs
node investigations/ephemeral-reasoning/full-loop.mjs --usage-probe
```

The display and full-loop probes use an existing installed dsh. See the
investigation README for installation-path selection and opt-in real-account
verification. Tests cover save/replay agreement, normal/interrupted/failed
streams, reconnect and display retirement, billing preservation, metering,
compaction, native tools, retries, cancellation, lifecycle, and disk resume.
Regression tests drive the real meter and the real projection registry together
over the cases that previously diverged: repeated user messages in one loop, a
prune in the middle of a turn, an unheadered session, a rewritten surface, and a
refused override.
The real DeepSeek run also checks increasing cache reuse and clean wire history.

Only structured reasoning and designated replay state are stripped; ordinary
answer text and unrelated provider diagnostics are not heuristically redacted.
Other plugins may independently record live chunks. Provider-specific reasoning
signature or replay requirements remain a compatibility consideration; no
provider incompatibility is currently demonstrated by these tests. Verify each
dsh upgrade and additional provider before broad deployment. No release is published.

MIT
