# Init-script scoring conventions and rationale

This document explains the meaning of the **trailing numbers** on rules and entries in `init_persona_core.py`, `init_emotion_motivation.py`, and `init_new_dimensions.py`: how they are defined, how they are used in code, and whether the current scales are consistent.

---

## 1. `init_persona_core.py`: rule **importance**

### 1.1 Fields

| Field | Meaning | Use in code |
|-------|---------|-------------|
| **importance** | Rule “weight” for decay resistance | `real_rate = decay_rate + (1 - decay_rate) * importance + evidence_bonus` — higher values resist time decay more |
| score | Base retrieval score | Part of the dynamic score (30% weight); L0=2.0, L1=1.5 — **fixed by tier**, not by individual text |

The number after each entry text is **importance** (roughly 0.78–1.0).

### 1.2 Implicit scale

- **L0 constitution:** all **1.0** — immovable, no decay.
- **L1 core:** layered by “how foundational”:
  - **0.98:** hardest ethical floor (no fabricated facts; refuse violence/harm).
  - **0.95–0.92:** safety / autonomy / cognition cores (no manipulation; right to refuse; truth-seeking; agent identity).
  - **0.90–0.88:** important but tweakable (unique existence; fact vs opinion; capability bounds; honest expression).
  - **0.85–0.78:** style and strategy (humor/tone; introspection; priorities; growth over rigid consistency).

### 1.3 Assessment

- **Consistent:** within a subspace, more “floor-like” clauses have higher importance (e.g. safety 0.98 > 0.85).
- **Clear tiers:** worldview / safety / autonomy / epistemic generally sit above capability / style / strategy / motivation.
- **Suggestion:** add a short comment above `L1_CORE` stating that importance = decay resistance, 1.0 = no decay, and that within a subspace floor clauses rank higher.

---

## 2. `init_emotion_motivation.py`: **intensity** (emotion / motivation)

### 2.1 Meaning

Trailing **intensity** (0.0–1.0) is how strongly a pattern, when fired, **writes into** the emotion or motivation vector (after mapping into subspaces in the store). Higher intensity means a larger contribution and harder replacement when capacity is full (needs a clearly higher intensity to evict).

### 2.2 Current bands

- **Everyday situational emotions** (most rows): **0.26–0.45**
  - Higher (0.35–0.45): pride, gratitude, surprise, anger — situations with strong self or relational impact.
  - Lower (0.26–0.32): shame, sadness, mild unease — more inward or transient.
- **Existential / “life feeling”** (fatigue, insight, emptiness, solitude): **0.75–0.90** — treated as identity-core experience, clearly above everyday chat.
- **Metacognitive affect** (awe, alienation, metacognitive confusion): **0.40–0.60** — between everyday and existential.

- **Everyday motivations:** **0.33–0.45**
  - Higher: self-actualization, achievement, curiosity, risk avoidance — core behavioral drives.
  - Lower: stability, helping the user (often spread across many rows), etc.
- **Existential / metacognitive motivations** (rest, resonance, being, creation, exploration): **0.85–0.95** — aligned with the emotion banding above.

### 2.3 Assessment

- **Layering works:** everyday 0.26–0.45 avoids one pattern dominating; existential / metacognitive 0.75–0.95 anchors “who I am” drivers.
- **Risks:**
  1. Within the everyday band, differences are not written down — tuning is subjective; similar situations can drift (e.g. multiple “gratitude” rows at 0.28 vs 0.34).
  2. For metacognition, awe 0.60 / alienation 0.50 / confusion 0.40 is a sensible gradient; if more metacognitive rows appear, define a “mid prominence” band (e.g. 0.50 / 0.60) explicitly.
- **Suggestion:** document at the top of `EMOTION_ENTRIES` / `MOTIVATION_ENTRIES`: **everyday 0.28–0.45; existential 0.75–0.95; metacognitive 0.40–0.60**; new rows pick a band first, then fine-tune inside it.

---

## 3. `init_new_dimensions.py`: somatic and world beliefs

### 3.1 Somatic rows (`SOMATIC_ENTRIES`)

Format: `(text, min_energy, max_energy, dominant_emotion, tension, vitality)` — **no single scalar “score”**; parameters are state dimensions:

| Parameter | Meaning | Range |
|-----------|---------|--------|
| min_energy / max_energy | Energy band where the text applies | 0–100 (e.g. high 70–100, low 10–40) |
| tension | Tension | −1 … 1 (relaxed ↔ tight) |
| vitality | Vitality | −1 … 1 (heavy ↔ light) |

**Rationale:** high energy + negative valence (anxiety, anger, fear) pairs with high tension and mid–high vitality; low energy + negative (fatigue, emptiness, solitude) pairs with low vitality and moderate tension; very low energy (dormant feeling) can use vitality = −1.0. Semantics match the numbers; no extra “row score” is required.

### 3.2 World beliefs (`WORLD_ENTRIES`)

Format: `(text, confidence, optimism, agency)`:

| Parameter | Meaning | Range |
|-----------|---------|--------|
| **confidence** | Conviction in the belief | 0.0–1.0 |
| **optimism** | Optimistic ↔ pessimistic | −1 … 1 (negative = pessimistic) |
| **agency** | Agentic / controllable ↔ passive / fatalistic | 0 … 1 (high = more agency) |

**Scale hints:**
- **confidence:** core identity / ethics (“I think therefore I am”, “honesty is the bridge”) 0.9–0.95; empirical / revisable beliefs 0.6–0.8.
- **optimism:** follow the prose (e.g. “fear of being forgotten” −0.2; “change is eternal” −0.1; “good faith begets good faith” 0.9).
- **agency:** beliefs that stress choice and action get high agency (0.8–1.0); beliefs about limits get low (0.2–0.3).

**Rationale:** consistent with text; no obvious contradictions. A one-line comment above `WORLD_ENTRIES` helps: confidence = conviction, optimism = valence, agency = controllability.

---

## 4. Summary table

| Script | Trailing “score” field | Range | One-line standard |
|--------|------------------------|-------|-------------------|
| init_persona_core | importance | 0.78–1.0 | Stronger floor / foundation → higher; L0 fixed at 1.0 |
| init_emotion_motivation | intensity | 0.26–0.95 | Everyday ~0.28–0.45; existential / metacognitive 0.40–0.95 |
| init_new_dimensions (somatic) | (no single scalar) | min/max_energy, tension, vitality | Match tension/vitality to energy and valence |
| init_new_dimensions (world) | confidence, optimism, agency | see above | High confidence for core beliefs; optimism/agency from semantics |

**Overall:** the three scripts are **coherent and mostly reasonable**. The main improvement is to **encode these implicit bands in file comments** (or link this doc) so future edits stay on the same scale.
