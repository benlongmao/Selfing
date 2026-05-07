# English-first localization roadmap

Shared by **product / maintainers** and **implementers (including AI assistants)**. Align on goals first, then ship in phases so “embeddings + copy + persona” do not move all at once.

---

## Tiered goals (pick a bar)

| Tier | Scope | Typical output | Risk |
|------|-------|----------------|------|
| **A** | External face | English README, LICENSE, CONTRIBUTING, repo description, main docs summaries | Low |
| **B** | Runtime UI & system strings | Frontend/errors, `settings.yaml` display fields | Medium |
| **C** | Persona & seed content | `scripts/init_*.py` L0/L1, emotion/motivation/somatic defaults in English | High (volume) |
| **D** | Embeddings & stored vectors | Switch to **English / multilingual** embedder; migrate or rebuild SQLite `embedding` BLOBs; re-tune similarity thresholds | High (data + tuning) |

**Key point:** once default **persona / L0–L2 / evidence / biography / KB text** is English, many paths share one **embedder** for write-time vectors and query-time cosine (persona prototypes, unified memory, tool semantic groups, store evidence, …). Keeping a **Chinese-tuned** embedder (e.g. `bge-small-zh`) while the corpus is English makes retrieval **engineering-unreliable**, not merely “worse.” **Shipping a fully English instance should pair tier C with tier D.**

Suggested order: **A+B** can ship before **C**; **English rows in DB** and **embedder switch + re-embed** should be close in time—avoid a long window of “English text + Chinese embedding space.”

---

## Why “embeddings” is its own milestone

- Loader lives in `backend/embedder.py` (`EMBEDDER_MODEL`, `EMBEDDER_MODEL_SCOPE`, `MODELSCOPE_CACHE`); `models/` is cache only.
- Default in this fork: **`BAAI/bge-small-en-v1.5`** (384-d)—see `LOCALE_EN.md` §Embedding.
- Changing model: dimension or distribution may shift; check `embedder_fallback.py` assumptions.
- Many tables store **embedding BLOBs**. Old vectors are **not comparable** to new queries—plan **batch re-embed**, **new DB**, or **lazy recompute** (needs code + ops agreement).

---

## Phased checklist

### Phase 0 — Inventory (~0.5–1 day)

- [ ] List user-visible CJK: `frontend/`, `backend/app.py`, API messages, `prompt_builder`, …
- [ ] List `scripts/init_*.py` strings written to DB.
- [ ] List `embedder.encode` call sites and **hard-coded semantic anchors** (e.g. persona prototypes).
- [ ] List explanatory fields in `config/settings.yaml` and `.env.example`.

### Phase 1 — External (low risk)

- [x] English README (optional `README.zh.md` mirror).
- [x] CONTRIBUTING, optional Code of Conduct.
- [x] `.env.example` embedder variables documented.

### Phase 2 — Default English UX (UI + config)

- [~] Frontend strings + `apply_locale_en_index.py` reruns after big merges.
- [x] Branding / agent display defaults in `settings.yaml`.
- [ ] **Intermediate messaging:** if shipping before **C+D**, state clearly in README whether only UI/docs are English or persona/embeddings are still mixed.

### Phase 3 — Persona & init (English copy)

- [ ] Script-by-script: `init_persona_core.py`, `init_emotion_motivation.py`, `init_new_dimensions.py` with a shared glossary first.
- [ ] Smoke: empty DB → init → short chat.

### Phase 4 — Embeddings & data (bind with phase 3 for “full English”)

1. Model choice, license, CPU/GPU budget.
2. Contract: fallback / FAISS dims match chosen model (384 for default BGE small en).
3. Migration: re-embed or new instance; **backup `data.db`**.
4. Re-test thresholds (unified memory, tool semantic gate, …).

### Phase 5 — Acceptance

- [ ] Subjective eval: memory retrieval, persona recall, tool selection under English chat.
- [ ] `python3 -m unittest` + embedder smoke after dim changes.

---

## Decisions to write down before large refactors

1. Embedder + vector migration strategy (required for “full English” product).
2. Docs: **English-only** vs **bilingual** maintenance.
3. Existing instances: migrate `data.db` vs demo fresh DB.

---

## Index

| Area | Path |
|------|------|
| Embedder | `backend/embedder.py` |
| Fallback | `backend/embedder_fallback.py` |
| Philosophy | `docs/design_philosophy.md` |
| PR norms | `CONTRIBUTING.md` |

*Keep this file dated when goals change.*
