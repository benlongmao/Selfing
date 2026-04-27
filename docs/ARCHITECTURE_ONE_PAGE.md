# One-page architecture (main path + ownership)

> Skim this when you do not want to read the whole repo. Details live in code; this file may stay stale until a big architectural change.

## 1. After the user sends one message (main path)

```
Frontend/API → backend/routers/chat.py
            → ChatService.chat() (backend/chat_service.py)
            → Build messages: prompt_builder + chat_message_builder
            → Call the model (tool loop: chat_tool_runner)
            → Persist: event_logger → chat_turns
            → Optional: WebSocket push
```

**One-liner:** almost all **conversation** goes through `chat_service.py`; results land in **`chat_turns`**.

---

## 2. What the background is doing (do not confuse with the main path)

| Thread / service | Entry file | Role |
|------------------|------------|------|
| Resting pulse | `app.py` → `_resting_pulse_loop` | About every minute: tiny **z_self** perturbation, goal / schedule / calendar **checks**; when due, **enqueue** only—**does not call the LLM** |
| Unified scheduler | `unified_scheduler.py` | **Serial** execution: heartbeat, idle pulse, tasks enqueued from schedule/calendar, plan tasks, etc. → internally calls `ChatService.chat()` |
| Heartbeat | `heartbeat_service.py` | Read `workspace/sandbox/HEARTBEAT.md` → callback **enqueues** into the unified scheduler |

**One-liner:** work that **needs thinking** usually **enters the queue first**, then the unified scheduler runs one `chat()`.

---

## 3. Memory: who owns what (when it gets confusing)

| Need | Primary owner |
|------|-----------------|
| Multi-turn transcript, scheduler-triggered inputs | SQLite **`chat_turns`** |
| Persona row semantic search | **`persona_store.py`** + FAISS (`persona_items`) |
| Multi-source memory blocks in the prompt | **`unified_memory.py`** (`UnifiedMemoryBus.retrieve_for_prompt`), wired via `prompt_builder_blocks_memory` |
| Agent-initiated “recall” tool | **`memory_search_tool.py`** |
| Long-lived agreements, progress (your ground truth) | **Markdown under `workspace/sandbox/`** (e.g. `HEARTBEAT.md`, your own `AGENT_FOCUS.md`, …) |

**One-liner:** **searchable DB** + **files on disk** together form full memory; chat alone is not enough.

---

## 4. Where to plug new features (without making a mess)

- **New periodic reminders:** edit `HEARTBEAT.md` or extend resting-pulse logic; **do not** spawn another background thread that calls `cs.chat()` directly.
- **New “remember this”:** prefer existing tables or the unified_memory pipeline; **or** write workspace files and tell HEARTBEAT / prompts **what to read first**.
- **New LLM background jobs:** only **`UnifiedScheduler.enqueue`**.

---

## 5. Minimum maintainer cost

- You do not need a formal diagram: **bookmark this file path** in README or notes.
- After a big architecture change: **spend ~5 minutes updating sections 1–3** in this page.
