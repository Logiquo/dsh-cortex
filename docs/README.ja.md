<p align="center">
  <a href="../README.md">English</a> ·
  <a href="README.zh-CN.md">简体中文</a> ·
  <a href="README.zh-TW.md">正體中文</a> ·
  <strong>日本語</strong> ·
  <a href="README.fr.md">Français</a> ·
  <a href="README.es.md">Español</a>
</p>

# dsh-cortex

ソフトウェアエンジニアリング（SWE）タスクにおける DeepSeek Harness（`dsh`）の性能を高めるためのプラグイン集です。コンテキストのオフロード、ローカル 0.8B モデルによる劣化監視と介入、アウトバウンドリクエスト変換、推論トレースの整理、エフェメラルなリマインダー注入を提供します。

## インストール

Node.js **24.9+** および `dsh`（`>=0.2.2-rc.5 <0.3.0`）が必要です。

ワークスペースをビルドし、任意のプラグインを `dsh` プロファイルに追加してください：

```sh
npm install
npm run build
dsh plugin --profile <profile> add ./plugins/<plugin-name>
```

> **補足：**
> - `dsh-ephemeral-reasoning` と `dsh-ephemeral-reminder` は `dsh-api-request` に依存します。先に同じプロファイルへ `dsh-api-request` を追加してください。
> - `dsh-context-guard` と `dsh-context-offload` は単独でも動作しますが、両方インストールすると自動的に連携します（`~/.dsh/context-cortex/session-<hash>/` を共有し、ハイブリッド語彙／意味検索 `history_search`、`history_read` での `0.8B` マイクロカード要約、カスタム実行ブリーフによる自動コンパクションが有効化されます）。

## プラグイン一覧

- [`dsh-api-request`](../plugins/dsh-api-request/README.md)：順序付きアウトバウンド変換パイプライン、メッセージ差分トークン計測、フェイルクローズ設計のマルチバージョンホストドライバを提供する共有 `apiRequest` サービス。
- [`dsh-context-guard`](../plugins/dsh-context-guard/README.md)：ゼロ設定のローカル `0.8B` コンテキスト劣化監視・要約・`embeddinggemma-2` 埋め込みプラグイン（`llama.cpp` と重みを [`Logic-z/dsh-cortex`](https://huggingface.co/Logic-z/dsh-cortex) から自動ダウンロード）。`tools/post-execute` 経由で KV キャッシュを破壊しない `<system-reminder>` 介入をルート `tool` 結果に付与し、同一ターン内で 2 回リマインドしても劣化が続く場合はフルプレフィックスのターン中コンパクションへエスカレートします。また `dsh-context-offload` へハイブリッド検索機能を提供します。
- [`dsh-context-offload`](../plugins/dsh-context-offload/README.md)：`BasicCompactionEngine` 上に構築された前向き実行ブリーフ、セッション単位の `offload.sqlite` 履歴アーカイブ、および 1 ターン目から利用可能な `history_search` / `history_read` ツール（`dsh-context-guard` 共存時は意味検索＋マイクロカード検索へ自動アップグレード）。
- [`dsh-ephemeral-reasoning`](../plugins/dsh-ephemeral-reasoning/README.md)：`dsh-api-request` 上で完了済みターンの推論ブロック除去と `replayState` 整理を行うアウトバウンドプラグイン。
- [`dsh-ephemeral-reminder`](../plugins/dsh-ephemeral-reminder/README.md)：`dsh-api-request` 上で最新のユーザーメッセージにエフェメラルな `<system-reminder>` を注入するアウトバウンドプラグイン。

## 開発

```sh
npm install
npm run typecheck
npm run build
npm test
```

## ライセンス

MIT
