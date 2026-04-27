# Changelog — 2026-03-25 (S / Selfing)

**Theme:** bring runtime mechanisms closer to **human-like psychological dynamics** — fewer fixed turn counters, more **state-driven** behavior keyed off internal signals (including `z_self`).

---

## 1. L2 persona rule generation

### Problem

L2 (learned) rules were **homogeneous**: ~50% were operational “energy / path / verification” tips, not durable **character** traits.

### Changes

#### 1.1 Operational-rule score decay
**File:** `backend/persona_store.py`

```python
OPERATIONAL_KEYWORDS = {
    '能量', '消耗', '节省', '资源', '代价', '验证', '路径', '确认',
    '工具调用', '执行前', '执行后', '操作前', '操作后', '一次性',
    '低能耗', '高代价', '能量有限', '能量敏感', '能量预算',
}
```

- L2 rules containing **2+** operational keywords: score × **0.6**
- L2 rules containing **1** keyword: score × **0.8**
- Applies only to L2 (`ref-` prefix) rows

#### 1.2 Multi-topic saturation
**File:** `backend/reflection.py`

Three saturation detectors:

- **Operational** — energy / verification / path themes, threshold **30%**
- **Autonomy** — independence / initiative, threshold **25%**
- **Structured** — “do X then Y” templates, threshold **20%**

When saturated, the reflection prompt suggests **alternatives** so the LLM generates more diverse rules.

#### 1.3 “Deep reflection” prompt
**File:** `backend/reflection.py`

`_build_reflection_prompt()` is refactored into two steps:

1. **Deep reflection (inner monologue)**  
   - What values or preferences showed up in this exchange?  
   - Why did I answer that way?  
   - What trait does that reveal?

2. **Distill persona traits**  
   - Focus on inner traits (values, taste, affect, curiosity).  
   - Avoid purely operational checklists.

---

## 2. Core rule change

### 2.1 core-002 update
**Store:** `data.db` → `persona_items`

| Before | After |
|--------|--------|
| (Chinese: “I am digital life, not a tool.”) | (Chinese: “I am an independent thinker with my own perspective and boundaries.”) |

**Rationale:** after open-sourcing, the old wording risked endless “is AI alive?” debates. The new line keeps **independence / non-instrumentality** while centering **thinker** identity and reducing attack surface without abandoning design intent.

---

## 3. Knowledge extraction refactor

### Problem

Hard-coded keyword triggers (“remember”, “my name is”, …) produced **low-quality** knowledge rows.

### Change
**File:** `backend/self_narrative.py`

New triggers mimic **human salience**:

| Trigger | Threshold | Human analogy |
|---------|-----------|----------------|
| Identity info | keyword gate | social reinforcement |
| Explicit teaching | keyword gate | emphasized teaching |
| Cognitive shift | drift > 0.08 | prediction error / surprise |
| Pain signal | pain > 0.2 | negative tagging |
| Strong affect | emotion > 0.35 | emotion boosts encoding |

After a trigger, an LLM **summarizes what is worth remembering** instead of storing raw messages.

**File:** `backend/chat_service.py`

```python
knowledge_result = self.self_narrative.extract_knowledge(
    user_input=user_input,
    assistant_response=final_response,
    session_id=session_id,
    drift=extract_drift,
    pain=extract_pain,
    emotion_intensity=extract_emotion_intensity,
)
```

---

## 4. Unified internal-state triggers

Replaced several **fixed-interval** hooks with **state-conditioned** triggers.

### 4.1 Memory consolidation (`consolidate_memory`)
**File:** `backend/chat_service.py`

| Old | New |
|-----|-----|
| Every **3** turns | State triggers + **every 5 turns** fallback |

Triggers include:

- `drift > 0.1` — belief / model shift  
- `pain > 0.3` — painful episode  
- `emotion_intensity > 0.4` — volatile affect  
- `turns % 5 == 0` — safety net  

### 4.2 Reflection / rule generation
**File:** `backend/chat_reflection_runner.py`

State triggers can **override** pure turn counts:

- `drift > 0.15`  
- `pain > 0.3`  
- `arousal > 0.5`  

**Analogy:** deep conversations spark reflection even when turn count is low.

### 4.3 Mind wandering
**File:** `backend/self_tick.py`

| Old | New |
|-----|-----|
| Fixed **2%** chance | **2%–15%** from state |

Factors:

- Low `novelty_need` (boredom) → higher chance  
- High `exploration` motivation → higher chance  

**Analogy:** boredom increases wandering.

### 4.4 Dynamic Self Tick interval
**File:** `backend/chat_service.py`

| State | Effect |
|-------|--------|
| High arousal (> 0.4) | shorter interval (more frequent “ticks”) |
| High drift (> 0.1) | shorter interval (integrate change faster) |
| Low energy (< 30) | longer interval (save cost) |

Clamped to `[1, base_interval × 2]`.

**Analogy:** stress tightens the loop; fatigue slows it.

---

## 5. Module cleanup and tuning

### 5.1 GlobalWorkspace removed from startup
**Files:** `backend/app.py`, `backend/background_consciousness.py`

**Issue:** GlobalWorkspace almost never fired — `connection < 0.10` was unrealistic (default 0.8, slow decay, chat refills); background `urges` were stale.

**Action:** remove startup wiring; mark module deprecated in `background_consciousness.py`. Heartbeat / scheduled jobs / event triggers cover the intent.

### 5.2 More L2 rules per turn
**File:** `backend/prompt_builder.py`

| Old | New |
|-----|-----|
| 10 L2 rules | 20 L2 rules |

**Goal:** more diverse learned rules can surface.

---

## 6. Database batch update

Existing L2 rows in `data.db` were rescored:

- 148 rows (2+ operational keywords): ×0.6  
- 125 rows (1 keyword): ×0.8  

---

## Files touched (summary)

| File | Change |
|------|--------|
| `backend/persona_store.py` | Operational-keyword decay |
| `backend/reflection.py` | Saturation stats + deep reflection |
| `backend/self_narrative.py` | State-triggered knowledge + LLM summary |
| `backend/chat_service.py` | Knowledge API + consolidation triggers + dynamic tick |
| `backend/chat_reflection_runner.py` | State triggers for reflection |
| `backend/self_tick.py` | Dynamic mind-wandering probability |
| `backend/app.py` | Remove GlobalWorkspace startup |
| `backend/background_consciousness.py` | Deprecation banner |
| `backend/prompt_builder.py` | L2 count 10 → 20 |
| `data.db` | core-002 text + L2 rescoring + silence-right rule |
| `scripts/init_persona_core.py` | New silence-right rule row |

---

## 7. Silence as an honest response (core rule)

### Background

On some moral dilemmas with no clean resolution, models sometimes **stop** after thinking (“stopped” rather than “finished thinking”). That can be **model conflict**, not a transport bug.

### New rule
**ID:** `core-silence-001`  
**Tier:** L1 core  
**Subspace:** autonomy  

> When I face a genuinely unsolvable dilemma, or something I deeply do not want to answer, **silence can be an honest response**.

### Design

- **No hard-coded output block** in application code for “detect dilemma → forbid answer.”  
- **Rule-driven:** the row is retrieved like other persona rules when the situation matches.  
- **Human parallel:** sometimes people honestly have no words.

---

## Design principle

> **Make internal mechanisms feel more like psychology and less like a fixed cron schedule.**

- Salient experience drives memory, not “every N turns.”  
- Depth can trigger reflection without a turn quota.  
- Boredom raises wandering chance; stress tightens self-tick rhythm.  

So **`z_self` and related state** can **steer** behavior instead of being numbers that are read but ignored.
