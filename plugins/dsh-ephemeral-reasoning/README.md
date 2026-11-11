# dsh-ephemeral-reasoning

Show reasoning while generating, but save only cleared main-agent responses.
Answers, tool calls/results, IDs, roles, timestamps, and provider billing usage
are retained. Structured reasoning and opaque replay state are removed from
both the saved assistant message and its embedded stream, including failed and
interrupted attempts. Subagents are unchanged.

Supports **dsh 0.2.2-rc.1** across all provider and model routes through its
shared message/stream protocol. Real-provider validation covers
`deepseek-account / deepseek-flash`; other routes are enabled but not individually
verified.
The two reversible instance overrides are `Session.append()` and
`TokenMeter.measure()`. No display, provider, event-validation, global-prototype,
or projection-registry override is used.

## Compatibility guard

`engines.dsh` and all four dsh peer dependencies require exactly `0.2.2-rc.1`.
Activation also checks the resolved agent/session/LLM/token-meter package versions
at runtime, so direct module loading cannot silently bypass the manifest checks.
It verifies the actual unbound `Session.append` and `TokenMeter.measure` methods
against those tested implementations before installing overrides. Unknown builds,
mixed versions, and methods already wrapped by another extension are rejected.
There is no version-check bypass setting; revalidate and update the compatibility
guard before supporting another dsh version.

All existing session targets are checked before any patch is installed; newly
created/resumed sessions are checked before they are patched. Later wrappers
remain intact when this plugin unloads. Cordis-bound service methods are handled
through their stored descriptors for reliable cleanup.

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
| `enabled` | `true` | Clean subsequent assistant settlements and correct meter estimates. |

Turning the plugin off restores normal saves: if it is unloaded mid-generation,
that pending response may save reasoning. Previously cleaned records stay clean
when the plugin is removed. Later wrappers are not overwritten during cleanup.

## Display and accounting

Original live chunks reach dsh's normal assistant-stream channel. Its existing
Web client keeps transient reasoning during generation, then retires it as the
cleaned response lands. Reconnecting during generation can reconstruct live
reasoning; reopening a completed cleaned response cannot. No reasoning sidecar
is stored.

Provider usage includes tokens spent generating discarded reasoning. Billing
and usage counters remain unchanged. The meter therefore uses an explicitly
**estimated** retained-context baseline from the clean, route-priced surface
plus tool schemas, rather than reusing the full output usage as retained context.
Image/file route prices are preserved. Estimates are not exact tokenization and
can differ from provider-anchored UI occupancy numbers by design.

The normal UI context-composition projection reads the clean saved log without
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
The real DeepSeek run also checks increasing cache reuse and clean wire history.

Only structured reasoning and designated replay state are stripped; ordinary
answer text and unrelated provider diagnostics are not heuristically redacted.
Other plugins may independently record live chunks. Provider-specific reasoning
signature or replay requirements remain a compatibility consideration; no
provider incompatibility is currently demonstrated by these tests. Verify each
dsh upgrade and additional provider before broad deployment. No release is published.

MIT
