<p align="center">
  <a href="../README.md">English</a> ·
  <a href="README.zh-CN.md">简体中文</a> ·
  <strong>正體中文</strong> ·
  <a href="README.ja.md">日本語</a> ·
  <a href="README.fr.md">Français</a> ·
  <a href="README.es.md">Español</a>
</p>

# dsh-cortex

面向 DeepSeek Harness（`dsh`）的外掛集合，透過上下文卸載、本機 0.8B 退化監控與干預、出站請求轉換、推理內容清理以及暫時性提示注入，全面提升 `dsh` 在軟體工程（SWE）任務上的表現。

## 安裝

需要 Node.js **24.9+** 與 `dsh`（`>=0.2.2-rc.1 <0.3.0`）。

建置工作區並將任意外掛加入 `dsh` profile：

```sh
npm install
npm run build
dsh plugin --profile <profile> add ./plugins/<plugin-name>
```

> **說明：**
> - `dsh-ephemeral-reasoning` 與 `dsh-ephemeral-reminder` 相依於 `dsh-api-request`；請先將 `dsh-api-request` 加入同一個 profile。
> - `dsh-context-guard` 與 `dsh-context-offload` 既可獨立運作，同時安裝時亦會自動協同（共用 `~/.dsh/context-cortex/session-<hash>/`，自動啟用混合詞法／語意 `history_search`、`history_read` 的 `0.8B` 微卡片摘要以及客製執行簡報自動壓縮）。

## 外掛列表

- [`dsh-api-request`](../plugins/dsh-api-request/README.md)：共用 `apiRequest` 服務，提供有序出站轉換管線、訊息差異 token 計量以及 fail-closed 多版本宿主驅動程式。
- [`dsh-context-guard`](../plugins/dsh-context-guard/README.md)：零設定本機 `0.8B` 上下文退化監控、摘要與 `embeddinggemma-2` 向量化外掛（啟動時自動從 [`Logic-z/dsh-cortex`](https://huggingface.co/Logic-z/dsh-cortex) 下載 `llama.cpp` 與模型權重）。透過 `tools/post-execute` 將零 KV-Cache 失效的 `<system-reminder>` 干預附著於根 `tool` 回傳結果，同一 tool-call chain 內提醒 2 次後自動升級為全前綴回合中壓縮，並為 `dsh-context-offload` 提供混合檢索能力。
- [`dsh-context-offload`](../plugins/dsh-context-offload/README.md)：基於 `BasicCompactionEngine` 建構的前瞻式執行簡報、工作階段級 `offload.sqlite` 歷史封存，以及首回合即開放的 `history_search` / `history_read` 工具（安裝 `dsh-context-guard` 後自動升級為語意＋微卡片混合檢索）。
- [`dsh-ephemeral-reasoning`](../plugins/dsh-ephemeral-reasoning/README.md)：基於 `dsh-api-request` 的出站已結束回合推理剝離與 `replayState` 清理外掛。
- [`dsh-ephemeral-reminder`](../plugins/dsh-ephemeral-reminder/README.md)：基於 `dsh-api-request` 的出站暫時性 `<system-reminder>` 注入外掛（僅附著於最新使用者訊息）。

## 開發

```sh
npm install
npm run typecheck
npm run build
npm test
```

## 授權條款

MIT
