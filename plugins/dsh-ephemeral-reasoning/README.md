# dsh-ephemeral-reasoning

Removes structured reasoning from the conversation dsh persists and replays.
Live reasoning still reaches the client during generation; the stored assistant
message and its embedded stream keep only visible content. Answers, tool
calls/results, IDs, timestamps, and provider usage are retained. Interrupted and
failed attempts are cleaned the same way, and every agent session is cleaned,
main and subagent alike. Replaying an earlier turn's reasoning can poison the
next turn's reasoning, and reasoning kept in the log is billed again on later
steps. A subagent keeps its log across turns, so it pays that repeatedly.

Removing reasoning changes the context the host counts, so the plugin re-anchors
context accounting on the provider's billed total and subtracts reasoning that
is no longer resent. Three private surfaces are patched:

- `Session.append` sanitizes each settlement before admission, publication, and
  persistence.
- `TokenMeter.measure` corrects the host-side context measurement.
- the live `contextPressure` projection definition corrects the client-side
  composer ring and stats, including its `stateVersion`, `apply`, `view`, and
  memoized cells.

## Install

Requires **dsh 0.2.2-rc.5** exactly. `engines.dsh`, the peer dependencies, and a
runtime package-version check all enforce it, and there is no bypass. Revalidate
the patched methods before supporting another dsh version.

```sh
npm install
npm run build
dsh plugin --profile <profile> add ./plugins/dsh-ephemeral-reasoning
```

Config: `enabled` (default `true`). Setting it to `false` disables sanitation and
metering.

Start a new session for clean history. Sessions that already contain reasoning
are cleaned only for new settlements.

## Known issues

- **Attachment tail: the ring trails the meter, as in dsh.** dsh's
  `contextPressure` projection is an O(1) scalar fold and prices appended
  content with the fixed heuristic; only `measure()` resolves the route's
  image/file pricing. The plugin keeps that projection model, so the gap is
  inherited rather than introduced. A prompt or tool result carrying an
  attachment appended after the newest settlement reports fewer tokens in the
  ring than in the meter until the next settlement. Fixing it means duplicating
  route pricing and per-node state inside a pure projection, which is why
  upstream keeps the scalar design instead.
- **The appended tail has no tokenizer, as in dsh.** dsh ships none and there is
  no provider count endpoint. Both the projection and `measure()` use the fixed
  `ceil(chars / 4) + 4` per-block estimate for text; only declared image/file
  pricing replaces it, and the plugin adds no second estimator. The error is a
  few percent of the tail.
- **Dropped reasoning trusts vendor behaviour.** The subtraction is this
  plugin's own estimate; dsh has no such notion. It assumes the provider stops
  resending prior-turn reasoning after `user/message` or `turn/end`. If it does
  not, the total is short by that turn's reasoning until the next settlement.
  There is no API guarantee to check.
- **Provider replay requirements are not modelled.** A provider that requires
  reasoning echoed back would need its own compatibility path. No such
  incompatibility has been demonstrated.
- **Settlements without reasoning evidence.** Splitting `outputTokens` into
  reasoning and visible content needs the removed reasoning's price, recorded as
  `ephemeralReasoning.removedTokens`. A settlement without it (an earlier build,
  a disabled plugin, or an unwrapped loop) falls back to
  `max(0, outputTokens - visible price)`, which under-estimates dropped reasoning
  on dense scripts and leaves the context total high until a later turn settles
  with evidence and becomes the new anchor.
- **No retroactive migration.** Rewriting history was out of scope: sessions
  loaded before the plugin keep their reasoning, and old settlements keep the
  fallback above.
- **Provider usage is trusted.** The anchor is the provider's own number,
  including when it misreports it.
- **Private APIs and exact-version pinning.** The three patches are not public
  seams. A dsh upgrade requires revalidation, and an unrecognized build is
  refused before any mutation.
- **rc.5 has no live-provider run.** The upgrade is covered by unit tests
  against the real host objects; the last live run was rc.1 on
  `deepseek-account / deepseek-flash`. Other routes are enabled but not
  individually verified.
- **An unwrappable session is left unsanitized.** Failing `agent/created` for a
  session the plugin cannot patch would be worse, so it logs a warning and
  falls back to host behaviour for that session.
- **Reasoning is not recoverable afterwards.** No sidecar is stored, so
  reopening a cleaned response cannot show its reasoning. Reconnecting during
  generation can still reconstruct live chunks held by other listeners.

## Appendix

### Metering

Two surfaces, both patched, use the same formula:

```
total = billed total + appended content - dropped reasoning
```

**Billed total: exact.** `inputTokens + cacheReadTokens + cacheWriteTokens +
outputTokens` from the newest usable `assistant/message` usage record. It covers
everything the provider accepted up to that call, including reasoning the log no
longer holds.

**Appended content: estimate.** Surface nodes added after the anchor. The meter
prices them with the route-aware node prices from `measure()`; the ring can only
add the host's heuristic surface total (see the attachment issue above).

**Dropped reasoning: estimate, zero while the turn is open.** The provider still
resends the open turn's reasoning, so the billed total stays authoritative.
`user/message` and `turn/end` close the turn; from then on the turn's reasoning
is subtracted. Each settled message contributes once:

```
reasoning = outputTokens × removed / (removed + price(visible))
```

`removed` is the price of the deleted reasoning blocks, recorded on the
settlement as `ephemeralReasoning.removedTokens`. Both prices use the same
per-character density, so the ratio is independent of script. When the adapter
reports `usage.reasoningTokens`, that exact count is used instead.

The anchor is dropped on a changed request envelope or a replacing surface op
(compaction, prune). The meter also drops it when an explicit `requestHeader`
differs from the logged one. Both surfaces then fall back to `tools + surface`:
route-priced in `measure()`, heuristic in the projection. The projection bumps
the unit's `stateVersion` to 8, so rows written by the old fold refold instead
of seeding.
