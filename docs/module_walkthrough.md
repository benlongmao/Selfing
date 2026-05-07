# Module-by-module code notes (S / Self-becoming)

> Snapshot date: 2026-03-10 (paths and line counts drift—verify in tree).

---

## Part 1 — Layout and entry

### Repo layout (high level)

```
repo/
├── start_server.py          # Uvicorn → FastAPI (:8080)
├── config/settings.yaml
├── data.db                  # SQLite (sessions, persona, z_self, chat_turns, …)
├── backend/
│   ├── app.py               # FastAPI app, routers, background threads
│   ├── chat_service.py      # Chat orchestration
│   ├── prompt_builder.py    # System prompt assembly
│   ├── routers/
│   ├── tools/
│   └── …
├── workspace/sandbox/       # HEARTBEAT.md, diaries, agent files
└── docs/
```

### `start_server.py`

- Sets `PYTHONPATH` / `DB_PATH`, runs `uvicorn backend.app:app`.

### `backend/app.py`

- Mounts routers (`system`, `persona`, `self`, `dimension`, `chat`, `meta`, `world`, `backup`, …).
- WebSocket `/ws/{session_id}`.
- Schema migrations on boot.
- Background: `_resting_pulse_loop` (micro `z_self` jitter, energy nudges, deadline checks—**no LLM**), heartbeat service hooks, optional wandering / autonomy threads (see current `app.py`).

### `config/settings.yaml` highlights

| Key | Typical | Meaning |
|-----|-----------|---------|
| `model_provider` | `deepseek_api` | Provider |
| `model_id` | `deepseek-chat` | Main chat model |
| `latent_dim` | `128` | `z_self` width |
| `self_tick_interval` | `4` | Chat turns between Self Tick triggers |
| `heartbeat_interval` | `3600` | HEARTBEAT service period |
| `dreaming_enabled` | `true` | Mind-wandering hooks |

---

## Part 2 — `chat_service.py` (core)

- **Size:** very large; main conversational pipeline.
- **Role:** restore history, daily narrative hooks, existential / will / pain gates, policy governor, sampling, prompt build, `run_tool_loop` or `_call_vllm`, introspection parse, `z_self` + stores update, Self Tick scheduling, persist `chat_turns`.

### Simplified `chat()` ordering

1. Normalize session (`get_effective_session`).
2. Restore prior turns from DB.
3. Notifications / daily narrative.
4. Existential modes (may short-circuit without LLM).
5. `will_conflict`, `real_consequences`, pleasure hooks, event-triggered reflection.
6. Policy + existential meaning probes, `will_tension`, `SelfBoundary`.
7. `compute_sampling_and_mode` + physiological `max_tokens` caps.
8. `prepare_prompt_and_introspection`, `build_messages`.
9. Tool loop or direct completion.
10. Parse introspection, update vectors, reflection, write `chat_turns`.

### Notable subsystems

- **Existential short-circuit** — solitude / rest modes may return guidance without calling the base model.
- **`real_consequences`** — hard block when energy/pain beyond thresholds.
- **`will_tension`** — persona vs instruction tension → energy / viscosity costs.
- **`_call_vllm`** — provider-specific streaming / reasoning fields.

---

## Part 3 — `prompt_builder.py`

- **`PromptBuilder.build_with_introspection_prompt`** composes identity, memory, merged affective/cognitive/relation blocks, tools, modes, etc.
- **`L0_ESSENTIAL_IDS`** — on non-first turns, inject a small L0 subset to save tokens while keeping safety anchors.

---

## Part 4 — `emotion_store.py` & `self_model.py`

### Emotion store

- 16-d affect vector in four subspaces (pleasure / arousal / control / social).
- `get_emotion_state` / `update_emotion` — state is **store-driven**, then injected into prompts (the “non-emergent affect” design note in older plans).

### Self model

- Owns **128-d `z_self`**, energy hooks, persistence.
- Methods: `get_z_self`, `save_z_self`, `initialize`, `update`, energy helpers.
- See **`docs/z_self_data_flow.md`** for the authoritative 2026-02-22 slice layout (this walkthrough’s older 64–88 table may be superseded).

---

## Part 5 — `persona_store.py` & `self_narrative.py`

### Persona

- SQLite + FAISS semantic retrieval; decay / archive; L0 locked, L1 core, L2 learned.
- `search_top_k`, `add_or_update`, `get_core_items`, `decay_memory`.
- Emotional bias reranking under high/low arousal (implementation detail in store).

### Narrative memory

- `self_biography` table with embeddings.
- `add_event` extracts “essence” with heuristics; retrieval merges vector similarity + recency.

---

## Part 6 — `tool_router.py` & tools

- **`get_tool_definitions` / `route` / `parse_request`** — large surface: files, workspace, browser, stocks, chemistry, research, tasks, PDF, self-heal, code proposals, … gated by feature flags.
- **`write_file`** — optional motivation logging, `z_self` fuse checks, post-write verification.
- Some introspection tools may be disabled in hardened configs.

---

## Progress tracker (historical)

| Area | Reviewed |
|------|----------|
| Layout / entry | Yes |
| `chat_service` | Yes |
| `prompt_builder` | Yes |
| `emotion_store` / `self_model` | Yes |
| `persona_store` / `self_narrative` | Yes |
| `tool_router` | Yes |

---

## Closing summary

End-to-end: **user text → `ChatService.chat` → gates & policy → prompt assembly → LLM (+ tools) → introspection → vector + persona updates → Self Tick → persistence**.

For embedding model upgrades or dimension changes, follow **`docs/localization_roadmap.md`** and **`LOCALE_EN.md`**.
