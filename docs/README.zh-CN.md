<p align="center">
  <a href="../README.md">English</a> ·
  <strong>简体中文</strong> ·
  <a href="README.zh-TW.md">正體中文</a> ·
  <a href="README.ja.md">日本語</a> ·
  <a href="README.fr.md">Français</a> ·
  <a href="README.es.md">Español</a>
</p>

# dsh-cortex

面向 DeepSeek Harness（`dsh`）的插件集合，通过上下文卸载、本地 0.8B 退化监控与干预、出站请求变换、思考过程清理以及临时提示词注入，全面提升 `dsh` 在软件工程（SWE）任务上的表现。

## 安装

需要 Node.js **24.9+** 与 `dsh`（`>=0.2.2-rc.1 <0.3.0`）。

构建工作区并将任意插件添加到 `dsh` profile：

```sh
npm install
npm run build
dsh plugin --profile <profile> add ./plugins/<plugin-name>
```

> **说明：**
> - `dsh-ephemeral-reasoning` 与 `dsh-ephemeral-reminder` 依赖 `dsh-api-request`；请先将 `dsh-api-request` 添加到同一 profile。
> - `dsh-context-guard` 与 `dsh-context-offload` 既可独立使用，同时安装时也会自动联动（共享 `~/.dsh/context-cortex/session-<hash>/`，自动启用混合词法/语义 `history_search`、`history_read` 的 `0.8B` 微卡片摘要以及定制执行简报自动压缩）。

## 插件列表

- [`dsh-api-request`](../plugins/dsh-api-request/README.md)：共享 `apiRequest` 服务，提供有序出站变换管线、消息增量 token 计量以及 fail-closed 多版本宿主驱动。
- [`dsh-context-guard`](../plugins/dsh-context-guard/README.md)：零配置本地 `0.8B` 上下文退化监控、摘要与 `embeddinggemma-2` 向量化插件（启动时自动从 [`Logic-z/dsh-cortex`](https://huggingface.co/Logic-z/dsh-cortex) 下载 `llama.cpp` 与模型权重）。通过 `tools/post-execute` 将零 KV-Cache 击穿的 `<system-reminder>` 干预附着到根 `tool` 返回结果，同一 tool-call chain 内提醒 2 次后自动升级为全前缀 turn 中压缩，并为 `dsh-context-offload` 提供混合检索能力。
- [`dsh-context-offload`](../plugins/dsh-context-offload/README.md)：基于 `BasicCompactionEngine` 构建的面向后续执行的压缩简报、会话级 `offload.sqlite` 历史归档，以及首轮即暴露的 `history_search` / `history_read` 工具（安装 `dsh-context-guard` 后自动升级为语义 + 微卡片混合检索）。
- [`dsh-ephemeral-reasoning`](../plugins/dsh-ephemeral-reasoning/README.md)：基于 `dsh-api-request` 的出站已结束轮次推理剥离与 `replayState` 清理插件。
- [`dsh-ephemeral-reminder`](../plugins/dsh-ephemeral-reminder/README.md)：基于 `dsh-api-request` 的出站临时 `<system-reminder>` 注入插件（仅附着于最新用户消息）。

## 开发

```sh
npm install
npm run typecheck
npm run build
npm test
```

## 许可证

MIT
