<p align="center">
  <strong>English</strong> ·
  <a href="docs/README.zh-CN.md">简体中文</a> ·
  <a href="docs/README.zh-TW.md">正體中文</a> ·
  <a href="docs/README.ja.md">日本語</a> ·
  <a href="docs/README.fr.md">Français</a> ·
  <a href="docs/README.es.md">Español</a>
</p>

# dsh-cortex

A collection of DeepSeek Harness (`dsh`) plugins designed to improve `dsh`'s performance on software engineering (SWE) tasks through context offloading, local 0.8B degradation monitoring and intervention, outbound request transformation, reasoning cleanup, and ephemeral prompt reminders.

## Installation

Requires Node.js **24.9+** and `dsh` (`>=0.2.2-rc.5 <0.3.0`).

Build the workspace and add any plugin to a `dsh` profile:

```sh
npm install
npm run build
dsh plugin --profile <profile> add ./plugins/<plugin-name>
```

> **Notes:**
> - `dsh-ephemeral-reasoning` and `dsh-ephemeral-reminder` depend on `dsh-api-request`; add `dsh-api-request` to the same profile first.
> - `dsh-context-guard` and `dsh-context-offload` work independently and automatically pair when installed together (sharing `~/.dsh/context-cortex/session-<hash>/` for hybrid lexical/semantic `history_search`, `0.8B` micro-card summaries in `history_read`, and custom execution-brief auto-compaction).

## Plugins

- [`dsh-api-request`](plugins/dsh-api-request/README.md): Shared `apiRequest` service providing an ordered outbound transform pipeline, message-diff token metering, and fail-closed multi-version host drivers.
- [`dsh-context-guard`](plugins/dsh-context-guard/README.md): Zero-config local `0.8B` context-degradation monitor, summarizer, and `embeddinggemma-2` embedder (auto-downloading `llama.cpp` and weights from [`Logic-z/dsh-cortex`](https://huggingface.co/Logic-z/dsh-cortex)). Appends zero-KV-cache-invalidation `<system-reminder>` interventions onto root `tool` results via `tools/post-execute`, escalates to full-prefix mid-turn compaction after 2 in-turn reminders, and provides hybrid retrieval to `dsh-context-offload`.
- [`dsh-context-offload`](plugins/dsh-context-offload/README.md): Forward-looking execution briefs, session-scoped `offload.sqlite` history archives, and turn-1 `history_search` / `history_read` tools built on `BasicCompactionEngine` (automatically upgraded with semantic + micro-card search when `dsh-context-guard` is present).
- [`dsh-ephemeral-reasoning`](plugins/dsh-ephemeral-reasoning/README.md): Outbound closed-turn reasoning stripping and `replayState` cleanup built on `dsh-api-request`.
- [`dsh-ephemeral-reminder`](plugins/dsh-ephemeral-reminder/README.md): Outbound ephemeral `<system-reminder>` injection on the latest user message built on `dsh-api-request`.

## Development

```sh
npm install
npm run typecheck
npm run build
npm test
```

## License

MIT
