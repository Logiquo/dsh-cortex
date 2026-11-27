# dsh-context-guard

Local 0.8B context-degradation recorder, scorer, and semantic history search provider for DeepSeek Harness (`>=0.2.2-rc.1 <0.3.0`).

> **Current phase (passive monitoring only)**: records V2.4 chronological cards, distilled `0.8B` micro-cards, `embeddinggemma-2` vectors, and `0.8B-Judge` `p_degraded` scores into `guard.sqlite` (+ optional `timeline.txt`), and powers hybrid semantic retrieval in `dsh-context-offload`. Does **not** inject reminders or trigger compaction yet.

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

- `/context-guard status`: show `guard.sqlite` coverage, card/scored counts, latest `p_degraded` scores, runtime status, and last error.
- `/context-guard rebuild`: rebuild the current session's `guard.sqlite` and `timeline.txt` from canonical `dsh` history.

## Validation

```sh
npm run typecheck
npm test
```
