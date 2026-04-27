# Two “heartbeat” mechanisms compared

## Overview

| Aspect | Mechanism A: HeartbeatService | Mechanism B: `scripts/heartbeat.py` |
|--------|------------------------------|--------------------------------------|
| **Location** | `backend/heartbeat_service.py`, daemon thread inside the backend | `scripts/heartbeat.py`, separate process |
| **Config** | `system.heartbeat_enabled`, `system.heartbeat_interval` | Script constants `TICK_INTERVAL=1800`, `API_BASE` env |
| **What fires** | Read `workspace/sandbox/HEARTBEAT.md` → when actionable, call **`ChatService.chat()`** | Periodic `POST /self/tick` → **`SelfTick.trigger()`** |
| **Main role** | Periodic **task execution**: the agent follows the task list in HEARTBEAT.md | Periodic **state refresh**: aggregate evidence, update `z_self`, may trigger mind-wandering / metacognition—“still here” |
| **Calls LLM?** | Yes (one full chat turn) | No inside tick alone (may indirectly trigger Mind Wandering → LLM) |

---

## Mechanism A: HeartbeatService (HEARTBEAT.md task heartbeat)

### Code and startup

- **Implementation:** `backend/heartbeat_service.py`
- **Startup:** `backend/app.py` `_start_background_threads_at_module_load()` when `config.get("system.heartbeat_enabled", False)` is true: construct `HeartbeatService(on_heartbeat=_on_heartbeat)` and `.start()`.
- **Callback:** `_on_heartbeat(prompt)` calls `ChatService.chat(user_input=prompt, session_id="<canonical>", ...)`—a full chat with the built heartbeat prompt.

### Behavior

1. Every `heartbeat_interval` seconds (default 3600) the `_heartbeat_loop` thread runs.
2. Each tick reads `workspace/sandbox/HEARTBEAT.md`.
3. **Skip if empty:** `_is_empty()` strips HTML comments and Markdown headings, then checks for unfinished items (e.g. `- [ ]`); if nothing actionable, no callback.
4. If actionable, `_build_heartbeat_prompt(content)` builds a timestamped prompt (e.g. `[HEARTBEAT - …]\nPeriodic heartbeat…`), then `on_heartbeat(prompt)` → **one full `chat()`** so the agent can read/write files, search, execute, etc.
5. HEARTBEAT.md can be edited by the agent via tools—shared human/agent task list.

### Config (`config/settings.yaml`)

```yaml
system:
  heartbeat_enabled: true
  heartbeat_interval: 3600
```

---

## Mechanism B: `scripts/heartbeat.py` (tick / external heartbeat)

### Code and startup

- **Implementation:** `scripts/heartbeat.py`
- **Startup:** Often `scripts/manage_services.sh` `start_heartbeat()` with `nohup`; independent of backend `heartbeat_enabled`.

### Behavior

1. Every `TICK_INTERVAL` seconds (script default 1800) calls `trigger_tick()`.
2. `trigger_tick()` `POST`s to `API_BASE` (default `http://localhost:8080`) **`/self/tick`** with body `{"sessionId": "<canonical>"}`.
3. **`/self/tick`** (`backend/routers/self.py`) calls `chat_service.self_tick.trigger(..., trigger_reason="manual_heartbeat")`.
4. **`SelfTick.trigger()`** (`backend/self_tick.py`): aggregate evidence, update `z_self`; may trigger mind wandering; **does not** start a user-visible chat unless inner logic (e.g. mind wandering) calls the LLM.

So the script is a **pure state / presence tick**: even without user messages, self-state updates on a cadence and may indirectly spawn background thought.

### Configuration

- Hard-coded in script: `TICK_INTERVAL`, `SESSION_ID`, `API_BASE=os.environ.get("API_BASE", "http://localhost:8080")`.
- To unify with `settings.yaml`, extend the script or env docs (not yet wired to YAML).

---

## Versus “resting pulse”

- **Resting pulse** (`_resting_pulse_loop`, `backend/app.py`): about once per minute—**tiny `z_self` noise**, small energy top-up, deadline checks, work log; **no LLM**, does **not** read HEARTBEAT.md, does **not** run SelfTick. Think “low-level physiology / housekeeping”—different from both heartbeats above.

- **HeartbeatService:** HEARTBEAT.md → **Chat** → **task heartbeat**.
- **`scripts/heartbeat.py`:** **`/self/tick`** → **SelfTick** → **state refresh + optional wandering**.

---

## Summary

| Goal | Use |
|------|-----|
| Agent follows HEARTBEAT.md on a schedule (checks, reminders, diary tasks) | Enable **HeartbeatService** (`system.heartbeat_enabled: true`) and keep actionable content in HEARTBEAT.md. |
| Refresh `z_self` and optional mind-wandering without user chat | Run **`scripts/heartbeat.py`** (or via `manage_services.sh`), backend `/self/tick` must be up. |
| Both | Enable HeartbeatService **and** run the script—they are **not** substitutes. |

**Overlap note:** if both run with short intervals, you stack token usage (scheduled chats + ticks). Increase `heartbeat_interval` and/or `TICK_INTERVAL` to balance.
