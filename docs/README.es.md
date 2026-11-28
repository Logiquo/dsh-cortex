<p align="center">
  <a href="../README.md">English</a> ·
  <a href="README.zh-CN.md">简体中文</a> ·
  <a href="README.zh-TW.md">正體中文</a> ·
  <a href="README.ja.md">日本語</a> ·
  <a href="README.fr.md">Français</a> ·
  <strong>Español</strong>
</p>

# dsh-cortex

Una colección de plugins para DeepSeek Harness (`dsh`) diseñada para mejorar el rendimiento de `dsh` en tareas de ingeniería de software (SWE) mediante descarga de contexto, monitoreo local 0.8B de degradación con intervención, transformación de solicitudes salientes, limpieza de razonamiento y recordatorios efímeros.

## Instalación

Requiere Node.js **24.9+** y `dsh` (`>=0.2.2-rc.1 <0.3.0`).

Compila el espacio de trabajo y añade cualquier plugin a un perfil de `dsh`:

```sh
npm install
npm run build
dsh plugin --profile <profile> add ./plugins/<plugin-name>
```

> **Notas:**
> - `dsh-ephemeral-reasoning` y `dsh-ephemeral-reminder` dependen de `dsh-api-request`; añade primero `dsh-api-request` al mismo perfil.
> - `dsh-context-guard` y `dsh-context-offload` funcionan de forma independiente y se emparejan automáticamente cuando se instalan juntos (compartiendo `~/.dsh/context-cortex/session-<hash>/` para búsqueda híbrida léxica/semántica en `history_search`, resúmenes en micro-tarjetas `0.8B` en `history_read` y compactación automática con briefs de ejecución personalizados).

## Plugins

- [`dsh-api-request`](../plugins/dsh-api-request/README.md): Servicio `apiRequest` compartido que ofrece un pipeline ordenado de transformación saliente, medición de tokens por diferencia de mensajes y controladores de host multiversión con fallo seguro (*fail-closed*).
- [`dsh-context-guard`](../plugins/dsh-context-guard/README.md): Monitor local `0.8B` sin configuración para degradación de contexto, resumidor y generador de embeddings `embeddinggemma-2` (descarga automática de `llama.cpp` y pesos desde [`Logic-z/dsh-cortex`](https://huggingface.co/Logic-z/dsh-cortex)). Adjunta intervenciones `<system-reminder>` sin invalidar la KV-Cache a los resultados `tool` raíz mediante `tools/post-execute`, escala a una compactación completa del prefijo a mitad de turno tras 2 recordatorios y proporciona recuperación híbrida a `dsh-context-offload`.
- [`dsh-context-offload`](../plugins/dsh-context-offload/README.md): Briefs de ejecución orientados a los siguientes pasos, archivos históricos `offload.sqlite` por sesión y herramientas `history_search` / `history_read` disponibles desde el primer turno sobre `BasicCompactionEngine` (mejoradas automáticamente con búsqueda semántica y micro-tarjetas cuando `dsh-context-guard` está presente).
- [`dsh-ephemeral-reasoning`](../plugins/dsh-ephemeral-reasoning/README.md): Eliminación saliente del razonamiento de turnos cerrados y limpieza de `replayState` sobre `dsh-api-request`.
- [`dsh-ephemeral-reminder`](../plugins/dsh-ephemeral-reminder/README.md): Inyección saliente efímera de `<system-reminder>` en el último mensaje de usuario sobre `dsh-api-request`.

## Desarrollo

```sh
npm install
npm run typecheck
npm run build
npm test
```

## Licencia

MIT
