# `z_self` upstream / downstream

## Definition and storage

| Item | Detail |
|------|--------|
| **Definition** | 128-dim `numpy` vector, latent “self state” |
| **Storage** | `self_state.z_self` column (JSON) |
| **Core module** | `backend/self_model.py` (`SelfModel`) |
| **Layout constants** | `RULES_SUBSPACE_DIMS` and related in `self_model.py` |

### 128-d layout (trimmed 2026-02-22)

| Range | Name | Dims | Subspaces |
|-------|------|------|-------------|
| 0–31 | RULES | 32 | safety(0–8), epistemic(8–16), style(16–24), strategy(24–32) |
| 32–47 | EMOTION | 16 | pleasure, arousal, control, social |
| 48–63 | MOTIVATION | 16 | achievement, relationship, exploration, safety |
| 64–87 | WORLDVIEW_CACHE | 24 | Aggregated from `WorldStore.aggregate_worldview_for_z_self()`—**not PCA** |
| 88–103 | SOMATIC | 16 | energy, viscosity, pain, vitality |
| 104–127 | NEEDS | 24 | connection, clarity, safety |

---

## Upstream: who writes `z_self`

### First initialization

```
SelfModel.initialize(session_id)
  ├── ref_vector (from PersonaStore core rules) or zeros
  ├── EmotionStore.get_emotion_state → z_self[32:48]
  ├── MotivationStore.get_motivation_state → z_self[48:64]
  ├── SomaticStore.get_somatic_state(...) → z_self[88:104]
  └── _save_z_self()
```

- **When:** `get_z_self()` is `None` or first use of a session.
- **ref_vector:** `_init_ref_vector()` aggregates `persona_store.get_core_items()` with subspace weights.

### Self Tick (main write path)

After chat, background work may call `self_tick.add_evidence` → every `SELF_TICK_INTERVAL` turns, `self_tick.trigger()` → `self_model.update(...)` with evidence embedding, EMA-style rule slice updates, optional `DimensionInteraction`, sync from somatic/emotion/motivation stores.

### ChatService sync (stores → `z_self`)

| Call site | Writes | When |
|-----------|--------|------|
| `_sync_emotion_to_z_self` | `z_self[32:48]` | After emotion store updates |
| `_sync_motivation_to_z_self` | `z_self[48:64]` | After motivation updates |
| `_apply_rules_delta_to_z_self` | `z_self[:32] += delta` | After reflection / promotion |
| `will_tension` | viscosity slice boost | After will tension |

### Other writers

| Module | Effect |
|--------|--------|
| `WorldStore.sync_worldview_to_z_self` | `z_self[64:88]` aggregate | After belief MMR / evolve / init |
| `_resting_pulse_loop` | micro perturbation + small energy | ~every minute |
| `self_healing` | bump achievement slice on success | Healing loop |
| `core/homeostasis` | needs / energy | Homeostasis pass |
| **Self Tick recovery** | hard reset on severe drift | Drift over threshold |

**Note:** in typical `SelfModel.update`, slice **`[64:88]` is copied forward**, not EMA-updated from chat evidence directly.

---

## Downstream: who reads `z_self`

### Sampling (`temperature`, `top_p`)

`chat_sampling.compute_sampling_and_mode` → `self_model_sync.compute_generation_params`: rules / emotion / motivation / somatic slices adjust `temperature`, `top_p`, `internal_state_prompt`, then **`z_self_influence`** may further nudge sampling.

### Prompt assembly

`prompt_builder.build_with_introspection_prompt`: `get_summary`, `get_structured_summary`, generation params text, and **PersonalityMatcher** uses `z_self` to pick L1 rules.

### Behavior and policy

| Module | Uses `z_self` for |
|--------|-------------------|
| `will_conflict` | Autonomy / conflict hints |
| `real_consequences` | energy / pain gates |
| `z_self_influence` | emotion narrative, style directives, constraints |
| `Reflection` | somatic reflection prompts |
| `tool_router` | e.g. `max_tokens` caps from viscosity; self fact dimensions |
| `autonomous_action_engine` | energy / pain / novelty heuristics |

### Resting pulse & presence

`_resting_pulse_loop` reads/writes small `z_self` jitter; presence-style pulses ask “what do you want to do?” with state-aware behavior.

---

## Key files

| File | Responsibility |
|------|----------------|
| `self_model.py` | define, init, update, save, migrate layout |
| `self_model_sync.py` | `compute_generation_params` |
| `self_model_summary.py` | summaries, internal state prompt glue |
| `self_tick.py` | evidence, trigger, call `SelfModel.update` |
| `chat_service.py` | get/save `z_self`, sync stores, trigger tick |
| `chat_sampling.py` | merge sampling with `z_self` |
| `z_self_influence.py` | narrative + style + constraints from slices |
| `app.py` | resting pulse jitter |
