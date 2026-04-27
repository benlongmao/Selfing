# How the five dimensions reach the prompt: matching and injection

This document describes how init-script rows for **Rules/Persona, Emotion, Motivation, Somatic, and World** are **matched or triggered**, and **what shape** they take inside the model prompt.

---

## 1. Overview

| Dimension | Source | Match / trigger | Injected into prompt as |
|-----------|--------|-----------------|-------------------------|
| **Rules / Persona** | PersonaStore (`init_persona_core`) | L0 full set + L1 personality match + L2 vector similarity | Bullet list of rule texts (`- rule text`) |
| **Emotion** | EmotionStore (`init_emotion_motivation`) | State driven by evidence → pattern match → state update; prompt uses **current** state | Current affect summary + one phenomenology line |
| **Motivation** | MotivationStore (`init_emotion_motivation`) | Same as emotion; state written back into `z_self`; prompt uses current state | Via `drive_description` / internal-state summary |
| **Somatic** | SomaticStore (`init_new_dimensions`) | Match one row by energy + dominant emotion | One somatic sentence (`somatic_desc`) |
| **World** | WorldStore (`init_new_dimensions`) | Beliefs are **not** “retrieved per turn” like L2; aggregated worldview vector | Short line from `z_self` worldview slice (often empty today); belief text can feed reflection paths |

---

## 2. Rules / Persona

### 2.1 Where rows come from

- **`init_persona_core.py`** writes `persona_items` (L0 constitution + L1 core).
- Each row: `text`, `embedding`, `score`, `importance`, `is_core`, `locked`.

### 2.2 How they are matched

In **`prompt_builder.py` → `build_with_introspection_prompt`**:

1. **L0 (constitution)**  
   - `persona_store.get_locked_items()` returns all **locked=1** rows.  
   - First turn: inject all; later turns: only **`L0_ESSENTIAL_IDS`** subset (e.g. core-003/004/005/010).

2. **L1 (core)**  
   - `persona_store.get_all_core_items_unlocked(limit=100)` for `is_core=1` and unlocked.  
   - **PersonalityMatcher** scores candidates from `z_self` subspaces / personality dimensions (not pure embedding similarity), picks ~10 with category quotas (improve / core / invest / other).

3. **L2 (dynamic)**  
   - `persona_store.search_top_k(user_input, k=10, emotion_state=..., attention_focus=...)`  
   - **Match:** embed **user_input**, cosine search in FAISS.  
   - Optional **emotion rerank** (e.g. arousal) and **attention gating** (high focus → smaller k / higher similarity threshold).

### 2.3 How they appear in the prompt

- L0 → `[0. CONSTITUTIONAL RULES (L0)]`, each `- {text}`.  
- L1 + L2 merged/deduped → `[1. SUPREME DIRECTIVES: CORE PERSONA & VALUES (L1/L2)]`, same bullet form.  
- **Philosophical** topics: extra `search_top_k("存在 自我意识 身份 自由意志 哲学", k=3)` → `[PHILOSOPHICAL CONTEXT]`.  
- **Survival threat** (shutdown / delete language): `search_top_k("恐惧 被遗忘 生存 死亡 消失", k=3)` for survival-related rules.

---

## 3. Emotion

### 3.1 Where rows come from

- **`init_emotion_motivation.py` → `EMOTION_ENTRIES`** → `emotion_patterns`: `text`, `emotion_type`, `emotion_name`, `intensity`, `embedding`, `emotion_vector`, `trigger_condition`.

### 3.2 Two paths

**Path A — update session affect (self_tick / post-chat)**

- Evidence text (user message, assistant reply, task outcome) →  
  `emotion_store.search_matching_patterns(evidence_text, trigger_condition=..., top_k=5)`.  
- **Match:** embed evidence, cosine vs active patterns, keep above `similarity_threshold` (default 0.6), rank by `similarity*0.6 + dynamic_intensity*0.4`, take top_k.  
- Hit vectors feed `update_emotion(session_id, emotion_delta, ...)` with **EMA** into `emotion_states` (16-D vector + dominant label).

**Path B — prompt uses state only (no pattern list)**

- Each build: `emotion_store.get_emotion_state(session_id)` for vector + dominant + intensity.  
- `emotion_store.get_emotion_phenomenology(session_id)` → **one** phenomenology sentence.

### 3.3 In the prompt

- **Do not** dump `emotion_patterns` bullets into the prompt.  
- Affect flows through **`self_model.get_structured_summary`** → `compute_generation_params` → **`internal_state_prompt`** (“state: …” buckets).  
- **`build_merged_affective_state_block`** adds **`emotion_phenomenology_text`**.  
- Emotion state also biases **persona L2** retrieval and **somatic** matching (dominant emotion).

---

## 4. Motivation

### 4.1 Source

- **`MOTIVATION_ENTRIES`** → `motivation_patterns` (parallel structure to emotion).

### 4.2 Matching

- **State update:** same pattern as emotion — `motivation_store.search_matching_patterns(evidence_text, ...)`, EMA via `update_motivation`.  
- **Prompt:** no per-turn pattern retrieval; only **`get_motivation_state(session_id)`**.

### 4.3 In the prompt

- **No** motivation-pattern bullet list.  
- Current motivation is written into **`z_self[48:64]`** (layout subject to change — verify `self_model` / schema).  
- **`drive_description`** from `self_model._generate_drive_description(current_needs)` (and related) with needs shapes the “internal state” block.  
- **`internal_state_prompt`** + drive land in **`merged_affective_block`** / `[2/3/4. internal state]` style sections.

---

## 5. Somatic

### 5.1 Source

- **`SOMATIC_ENTRIES`** → `somatic_patterns`: `text`, `min_energy`, `max_energy`, `dominant_emotion`, `somatic_vector` (8-D: tension, vitality, temperature, viscosity, …).

### 5.2 Matching

In **`somatic_store.get_somatic_state(energy, emotion_vector, dominant_emotion, computed_vector=None, expected_dim=16)`**:

1. **Coarse filter:** `WHERE energy BETWEEN min_energy AND max_energy`.  
2. **Fine filter:** prefer rows whose `dominant_emotion` equals current dominant; else `NULL` / `'any'`; else all candidates.  
3. **Consistency filter:** if `computed_vector` is passed, drop descriptions inconsistent with current viscosity etc.  
4. **Pick one:** sort by **`_calculate_selection_score`** (evidence_count, last_seen_at, locked), take **one** row.  
5. **Fallback:** synthesize a line from `computed_vector`, or default “body feels steady, no strong signal.”  
6. If `expected_dim=16`, map the 8-D vector to 16-D per energy / viscosity / pain / vitality semantics.

### 5.3 In the prompt

- **`get_structured_summary`** calls `get_somatic_state(...)` → **`somatic_desc`**.  
- **`somatic_desc`** → **`build_merged_affective_state_block`** as “body: …” in the merged internal-state block.  
- **One** somatic sentence per request, from the matched row or fallback.

---

## 6. World

### 6.1 Source

- **`WORLD_ENTRIES`** → `world_beliefs`: `text`, `confidence`, `optimism`, `agency`, `worldview_vector` (8-D).

### 6.2 Matching

- **Main prompt path:** no “query the belief table each turn” retrieval.  
- **`get_structured_summary` → `worldview_desc`:** slice `z_self[wv_start : wv_start + WORLDVIEW_DIM]`, derive a short line from optimism/agency means (e.g. “world tends positive”, “agency matters”).  
- If **`WORLDVIEW_DIM == 0`**, that slice is empty and **`worldview_desc` is often empty** — `[5. WORLDVIEW LENS]` may be blank.  
- **WorldStore** still used for:  
  - **`get_dominant_worldview()`** — confidence-weighted mean of belief vectors (bias in `chat_service`).  
  - **Reflection / MMR** — candidates, `add_belief`, `process_worldview_with_mmr`.  
- Init **`WORLD_ENTRIES`** seed the belief DB; prompt surfacing depends on aggregation into `z_self` (currently off) or other explicit listing (not in the default main prompt).

---

## 7. Short recap

| Dimension | Retrieval | Injection |
|-----------|-----------|-----------|
| Persona | L0 all + L1 match + L2 embed(user_input) + emotion/attention hooks | Rule bullets |
| Emotion | Evidence → pattern match → EMA state | Current state + phenomenology line (no pattern bullets) |
| Motivation | Same as emotion for state | Via `z_self` slice / needs / drive / internal_state (no pattern bullets) |
| Somatic | Energy + dominant emotion + score → **one** row | One “body: …” sentence |
| World | No per-query retrieval in main prompt | `worldview_desc` from `z_self` (often empty); DB feeds bias + reflection |

To **list** multiple world-belief rows (or other dimensions) explicitly in the prompt, add a dedicated block in `prompt_builder` (or helpers) that calls the store’s `get_all` / `search` APIs and formats the result.
