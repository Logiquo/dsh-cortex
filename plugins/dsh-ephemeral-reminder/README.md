# dsh-ephemeral-reminder

Append an ephemeral `<system-reminder>` text block to the latest `user` message on outbound LLM requests.

## Behavior

- **Tail-only injection**: Modifies only the last `user` message (`lastUserIndex`) per outbound request. Earlier `user` messages in history remain untouched, so the reminder is never persisted to session history and is not repeated across past turns.
- **`append-block` strategy**: Appends `{ type: 'text', text: reminderText }` to the tail `user` message's `content` array rather than inserting consecutive `role: 'user'` messages, maximizing compatibility across LLM adapters.
- **Conversation-scoped**: Registered on `dsh-api-request` with default `scope: 'conversation'` (`purpose === undefined`), so auxiliary calls such as `compaction` are left untouched while both main-agent and subagent conversation turns receive the reminder.
- **Token metering**: Handled automatically by `dsh-api-request` via message diffing (`durableDiffs`) and reconciled with `TokenMeter` and `contextPressure`.

## Configuration

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `enabled` | `boolean` | `true` | Enable or disable the outbound transform. |
| `reminderText` | `string` | `<system-reminder>...</system-reminder>` | Text block appended to the latest `user` message. |
| `order` | `number` | `300` | Execution order in `dsh-api-request` (`dsh-ephemeral-reasoning/strip` runs at `200`). |

## Installation

Install alongside `dsh-api-request`:

```bash
npm install dsh-api-request dsh-ephemeral-reminder
```
