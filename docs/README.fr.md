<p align="center">
  <a href="../README.md">English</a> ·
  <a href="README.zh-CN.md">简体中文</a> ·
  <a href="README.zh-TW.md">正體中文</a> ·
  <a href="README.ja.md">日本語</a> ·
  <strong>Français</strong> ·
  <a href="README.es.md">Español</a>
</p>

# dsh-cortex

Une collection de plugins pour DeepSeek Harness (`dsh`) conçue pour améliorer les performances de `dsh` sur les tâches d'ingénierie logicielle (SWE) grâce au délestage de contexte, à la surveillance locale 0.8B de la dégradation avec intervention, à la transformation des requêtes sortantes, au nettoyage du raisonnement et aux rappels éphémères.

## Installation

Requiert Node.js **24.9+** et `dsh` (`>=0.2.2-rc.5 <0.3.0`).

Compilez l'espace de travail et ajoutez n'importe quel plugin à un profil `dsh` :

```sh
npm install
npm run build
dsh plugin --profile <profile> add ./plugins/<plugin-name>
```

> **Remarques :**
> - `dsh-ephemeral-reasoning` et `dsh-ephemeral-reminder` dépendent de `dsh-api-request` ; ajoutez d'abord `dsh-api-request` au même profil.
> - `dsh-context-guard` et `dsh-context-offload` fonctionnent indépendamment et s'associent automatiquement lorsqu'ils sont installés ensemble (partage de `~/.dsh/context-cortex/session-<hash>/` pour la recherche hybride lexicale/sémantique `history_search`, les résumés micro-cartes `0.8B` dans `history_read`, et la compaction automatique par brief d'exécution personnalisé).

## Plugins

- [`dsh-api-request`](../plugins/dsh-api-request/README.md) : Service `apiRequest` partagé fournissant un pipeline ordonné de transformation sortante, un comptage de tokens par différentiel de messages et des pilotes hôtes multi-versions à sécurité intégrée (*fail-closed*).
- [`dsh-context-guard`](../plugins/dsh-context-guard/README.md) : Moniteur local `0.8B` zéro-configuration de dégradation du contexte, résumeur et moteur d'embeddings `embeddinggemma-2` (téléchargement automatique de `llama.cpp` et des poids depuis [`Logic-z/dsh-cortex`](https://huggingface.co/Logic-z/dsh-cortex)). Ajoute des interventions `<system-reminder>` sans invalidation du KV-Cache aux résultats `tool` racines via `tools/post-execute`, escalade vers une compaction complète du préfixe en cours de tour après 2 rappels, et fournit la recherche hybride à `dsh-context-offload`.
- [`dsh-context-offload`](../plugins/dsh-context-offload/README.md) : Briefs d'exécution orientés suite des tâches, archives d'historique `offload.sqlite` par session et outils `history_search` / `history_read` disponibles dès le premier tour sur `BasicCompactionEngine` (enrichis automatiquement par la recherche sémantique et les micro-cartes lorsque `dsh-context-guard` est présent).
- [`dsh-ephemeral-reasoning`](../plugins/dsh-ephemeral-reasoning/README.md) : Suppression sortante du raisonnement des tours terminés et nettoyage de `replayState` basés sur `dsh-api-request`.
- [`dsh-ephemeral-reminder`](../plugins/dsh-ephemeral-reminder/README.md) : Injection sortante éphémère de `<system-reminder>` sur le dernier message utilisateur basée sur `dsh-api-request`.

## Développement

```sh
npm install
npm run typecheck
npm run build
npm test
```

## Licence

MIT
