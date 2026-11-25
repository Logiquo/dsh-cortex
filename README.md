# dsh-cortex

A collection of DeepSeek Harness (`dsh`) plugins designed to improve `dsh`'s performance on software engineering (SWE) tasks through context offloading, outbound request transformation, reasoning cleanup, and ephemeral prompt reminders.

## Installation

Requires Node.js **24.9+** and `dsh` (`>=0.2.2-rc.1 <0.3.0`).

Build the workspace and add any plugin to a `dsh` profile:

```sh
npm install
npm run build
dsh plugin --profile <profile> add ./plugins/<plugin-name>
```

> **Note:** `dsh-ephemeral-reasoning` and `dsh-ephemeral-reminder` depend on `dsh-api-request`; add `dsh-api-request` to the same profile first.

## Plugins

- [`dsh-api-request`](plugins/dsh-api-request/README.md): Shared `apiRequest` service providing an ordered outbound transform pipeline, message-diff token metering, and fail-closed multi-version host drivers.
- [`dsh-context-offload`](plugins/dsh-context-offload/README.md): Forward-looking execution briefs, session-scoped history archives, and built-in `history_search` / `history_read` tools built on `BasicCompactionEngine`.
- [`dsh-ephemeral-reasoning`](plugins/dsh-ephemeral-reasoning/README.md): Outbound closed-turn reasoning stripping and `replayState` cleanup built on `dsh-api-request`.
- [`dsh-ephemeral-reminder`](plugins/dsh-ephemeral-reminder/README.md): Outbound ephemeral `<system-reminder>` injection on the latest user message built on `dsh-api-request`.

## Development

```sh
npm install
npm run typecheck
npm run build
```

## License

MIT
