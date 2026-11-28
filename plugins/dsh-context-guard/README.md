# dsh-context-guard

Local 0.8B context-degradation recorder, scorer, and semantic history search provider for DeepSeek Harness (`>=0.2.2-rc.5 <0.3.0`).

> **Features**: records V2.4 chronological cards, distilled `0.8B` micro-cards, `embeddinggemma-2` vectors, and `0.8B-Judge` `p_degraded` scores into `guard.sqlite` (+ optional `timeline.txt`), powers hybrid lexical/semantic retrieval in `dsh-context-offload`, appends zero-pronoun `<system-reminder>` interventions persistently onto the latest root `tool` result via `tools/post-execute` when sustained degradation is detected (2 hits `>= 0.50` within 16 actions, with an 8-action cooldown after start, `/compact`, real user messages, and prior interventions; preserves `lastUserIndex` for 0 KV-Cache invalidation with `dsh-ephemeral-reasoning` and `dsh-ephemeral-reminder`), and escalates to a full-prefix mid-turn `ctx.compaction.compactRegion` (using `dsh-context-offload`'s custom execution brief when installed) if degradation persists after 2 reminders within the same tool-call chain.

## Zero-Config Local Runtime

Requires Node.js 24.9+ (uses built-in `node:sqlite`). No manual `llama-server` or Python/PyTorch setup is required:
- On first startup, `dsh-context-guard` automatically checks both the pinned prebuilt `ggml-org/llama.cpp` release (`b12651`) and all model weights in `~/.dsh/context-cortex/`, and streams any missing files atomically in the background:
  - Runtime binary from `ggml-org/llama.cpp` -> `~/.dsh/context-cortex/runtime/<build>/`
  - Model weights from [`Logic-z/dsh-cortex`](https://huggingface.co/Logic-z/dsh-cortex) (configurable via `hfRepo` / `hfEndpoint` / `HF_ENDPOINT`) -> `~/.dsh/context-cortex/models/`:
    - `embeddinggemma-2-Q8_0.gguf` (`~296 MB`)
    - `sum08b-Q4_K_M.gguf` (`~510 MB`)
    - `judge08b-Q4_K_M.gguf` (`~510 MB`) + `judge08b-head.json` (`~18 KB`)
- Spawns isolated `llama-server` child processes (preferring Vulkan/Metal GPU with automatic CPU fallback) and cleans them up on exit.

## Co-located Storage (`~/.dsh/context-cortex/session-<hash>/`)

```text
~/.dsh/context-cortex/session-<hash>/
├── offload.sqlite    # owned by dsh-context-offload
├── guard.sqlite      # owned by dsh-context-guard (WAL mode)
└── timeline.txt      # human-readable micro-card & score log
```

## Commands

- `/context-guard`: show a concise summary of `guard.sqlite` coverage, card/scored counts, latest `p_degraded` scores, intervention/auto-compact counts, and local runtime status.

## Validation

```sh
npm run typecheck
npm test
```
