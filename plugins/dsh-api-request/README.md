# dsh-api-request

Shared outbound request-shaping and token-metering API service (`apiRequest`) for DeepSeek Harness (`dsh`) plugins.

Instead of letting multiple plugins independently monkey-patch host internals and fight over token accounting, `dsh-api-request` owns the three private host surfaces once and exposes a deterministic, ordered transform pipeline to consumer plugins:

- **`LlmRuntime.forAdapter`**: runs registered `RequestTransform`s in `(order, id)` order right before adapter dispatch, leaving persisted session logs untouched.
- **`TokenMeter.measure`**: diffs transformed vs. original durable messages (`nodes[]` and `surfaceTokens`) and reconciles ephemeral additions (`attachedTokens`) across provider `usage` anchor settlements.
- **`contextPressure` (`sessionProjections`)**: updates the client-side occupancy ring (`stateVersion = 10`) using the registered transforms' pure closed-turn deductions.

## Public API

Consumer plugins declare `inject = ['apiRequest']` and depend on `dsh-api-request` in `peerDependencies`—no direct DSH private API imports or version pinning required.

```ts
import type { Context } from '@deepseek-ai/cordis'
import { API_REQUEST_PROTOCOL } from 'dsh-api-request'

export const inject = ['apiRequest']

export function apply(ctx: Context) {
  if (ctx.apiRequest.protocolVersion !== API_REQUEST_PROTOCOL) {
    throw new Error(`Requires apiRequest protocol v${API_REQUEST_PROTOCOL}`)
  }

  ctx.effect(() =>
    ctx.apiRequest.register({
      id: 'my-plugin/transform',
      order: 200, // ascending; ties broken deterministically by id
      scope: 'conversation', // default: skips auxiliary purpose calls
      apply({ messages, appliedIds, lastUserIndex }) {
        if (appliedIds.includes('higher-priority/override')) return undefined
        // Return a new RequestMessage[] or undefined if unchanged
        return undefined
      },
    }),
  )
}
```

### Transform Ordering & Conflict Handling

- **Deterministic order**: transforms execute in ascending `order`, with ties broken lexicographically by `id` (never by plugin mount order).
- **Conflict awareness**: `TransformContext.appliedIds` lists the `id`s of preceding transforms that actually modified `messages`, allowing downstream transforms to skip, yield, or adapt.
- **Scope**: `scope` defaults to `'conversation'` (`purpose === undefined`), so auxiliary calls (such as compaction or title generation) are not accidentally modified unless `'all'` or specific `LlmRequestPurpose` values are requested.

## Message Diff Metering

After running the pipeline, `dsh-api-request` diffs the output against the input using message identity (`id`):

- **Unchanged durable messages** (reference equality): zero overhead.
- **Modified durable messages** (e.g. stripping closed-turn `reasoning`, regex cleaning, or pre-computed summaries): adjusts `nodes[i].tokens` / `heuristicTokens` and `surfaceTokens` by `estimateMessage(after) - estimateMessage(before)`, and splits `removedReasoningTokens` vs. `removedOtherTokens` for settled `usage` anchors.
- **Removed durable messages** (`id` absent in output): sets the corresponding node contribution to `0`.
- **Ephemeral messages** (no `id`, e.g. `RequestUserInput` reminders): priced into `attachedTokens`, recorded on dispatch (`forAdapter`), captured onto the `assistant/message` anchor when `usage` settles, and automatically deducted once the ephemeral attachment is no longer present:

```text
total = billed
      + appended_after_anchor
      - dropped_from_anchor
      - anchor_attached
      + current_attached
```

When no valid usage anchor exists (or the request envelope / surface is replaced), it falls back to `surfaceTokens + estimateToolsTokens(header) + current_attached`.

## Install & Host Compatibility

Supports **dsh `0.2.2-rc.1`** and **`0.2.2-rc.5`** via the internal host driver registry (`src/drivers/`). On startup, `dsh-api-request` verifies package versions and probes `LlmRuntime.forAdapter` and `TokenMeter.measure` before installing any patch—failing closed without mutating host services if an unrecognized or already-wrapped implementation is detected.

```sh
npm install
npm run build
dsh plugin --profile <profile> add ./plugins/dsh-api-request
```

Because Cordis `inject` only orders activation and does not auto-install missing plugins, `dsh-api-request` must be added to the profile alongside any consumer plugin (or included together in a bundle's `cordis.patch.yml`).

## Known Issues

- **Attachment tail: the ring trails the meter, as in dsh.** `contextPressure` is a pure O(1) scalar fold over `SessionEvent`s; route-specific image/file pricing and ephemeral `forAdapter`-only attachments are reflected in `TokenMeter.measure()`, while the ring stays a pure fold (deducting closed-turn omissions only).
- **No tokenizer in the appended tail.** Like dsh, text deltas use `ceil(chars / 4) + 4` per block + `4` per message.
- **Provider usage is trusted.** Anchors rely on the provider's reported `usage` counts.
- **Private APIs and fail-closed driver pinning.** Upgrading dsh beyond tested versions requires adding/validating a driver in `src/drivers/`.
