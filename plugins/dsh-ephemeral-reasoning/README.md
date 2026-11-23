# dsh-ephemeral-reasoning

Retains structured reasoning in persisted dsh session logs while stripping closed-turn reasoning before outbound model dispatch (for both main and subagent sessions).

Replaying an earlier turn's reasoning can poison subsequent turns and repeatedly bills those tokens on every follow-up turn. Stripping reasoning only on outbound dispatch keeps the persisted session log complete for UI reopen, offline replay, and context-degradation analysis, while preserving within-turn reasoning keeps multi-step tool-call chains coherent until the turn closes (`user/message` or `turn/end`).

All host patching, token-meter reconciliation, and `contextPressure` projection updates are delegated to [`dsh-api-request`](../dsh-api-request/README.md) via `ctx.apiRequest.register(...)`.

## Install

Requires [`dsh-api-request`](../dsh-api-request/README.md) (`apiRequest` protocol v1) to be installed and enabled in the same profile:

```sh
npm install
npm run build
dsh plugin --profile <profile> add ./plugins/dsh-api-request
dsh plugin --profile <profile> add ./plugins/dsh-ephemeral-reasoning
```

Config: `enabled` (default `true`). Setting it to `false` skips registering the transform.

Start a new session for clean history. Sessions that already contain reasoning are cleaned on outbound dispatch for all closed turns.

## Known Issues

- **Dropped reasoning trusts vendor behaviour.** Accounting assumes the provider stops resending prior-turn reasoning once `user/message` or `turn/end` closes the turn.
- **Provider replay requirements are not modelled.** A provider that strictly requires closed-turn `reasoning` or `replayState` to be echoed back would need its own compatibility path.
- **Metering and host-patch caveats** (heuristic tail estimation, `contextPressure` scalar fold limits, and dsh version driver pinning) are documented in [`dsh-api-request`](../dsh-api-request/README.md).
