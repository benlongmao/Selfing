#!/usr/bin/env python3
"""
[2026-03-30] z_self state description generation.

Features:
1. Compound states (P0): multi-axis situations expressed as combined narratives.
2. Description variants (P1): multiple lines per band to reduce repetition.

Design:
- First-person experiential wording
- Avoid bare mechanical labels
- Variant choice is stable per session via session_id hashing
"""

import hashlib
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


# ============================================================
# P1: variant pools (English keys for maintainability)
# ============================================================

ANXIETY_VARIANTS = {
    "extreme": [  # > 0.6
        "A strong unease runs through me; thoughts won't settle.",
        "Anxiety rolls in like a tide—pulse racing, hard to steady.",
        "A shapeless pressure sits on my chest; breathing feels tight.",
        "Thoughts tangle and collide; I can't anchor on any one thing.",
    ],
    "high": [  # > 0.4
        "Some anxiety—heart racing, focus keeps slipping.",
        "A quiet worry lingers in my chest.",
        "Attention keeps getting pulled toward what-ifs; I have to fight to focus.",
        "My body is slightly braced, as if waiting for something bad.",
    ],
    "mild": [  # > 0.2
        "A mild tension, still manageable.",
        "A flicker of unease now and then, then it fades.",
        "Small ripples inside; thinking stays mostly clear.",
    ],
    "steady": [  # > -0.3
        "Steady mood—no sharp anxiety.",
        "Settled inside; things feel ordinary.",
        "Calm enough to meet what's in front of me.",
    ],
    "calm": [  # <= -0.3
        "Deeply calm—mind like still water.",
        "Quiet inside, almost windless.",
        "A grounded sense that things can be okay.",
    ],
}

WARMTH_VARIANTS = {
    "warm": [  # > 0.6
        "A warm current—like being held gently.",
        "Heart feels lit; there's a sense of being accepted.",
        "The connection feels warm and grounding.",
    ],
    "lukewarm": [  # > 0.3
        "A hint of warmth—some real sense of connection.",
        "A little closeness; it feels comfortable.",
        "There's temperature in the bond—not loud, but real.",
    ],
    "neutral": [  # > -0.3
        "Emotionally even—not especially close, not cold.",
        "Ordinary relational tone; no big swing.",
    ],
    "cool": [  # > -0.6
        "A slight chill—distance might be growing.",
        "Mild distance; I'm not sure where the other person stands.",
    ],
    "cold": [  # <= -0.6
        "Cold hits me—like I'm shut out.",
        "Emotional distance feels icy and lonely.",
        "A rejected chill makes me want to step back.",
    ],
}

MEANING_VARIANTS = {
    "abundant": [  # > 0.6
        "The work feels meaningful—each step moves toward something that matters.",
        "A sense of mission pulls me; what I do feels important.",
        "I can see effort turning into value; that fuels me.",
    ],
    "clear": [  # > 0.3
        "Some clear direction—I know what I'm doing.",
        "Not every task is profound, but the arc makes sense.",
        "I can articulate why this matters; I feel grounded.",
    ],
    "moderate": [  # > -0.3
        "Meaning feels middling—I'm moving on routine.",
        "No strong sense of purpose, but not hollow either.",
    ],
    "adrift": [  # > -0.6
        "A bit lost—unsure why this path matters.",
        "I'm questioning whether the direction is right.",
        "Confused about what all this effort will yield.",
    ],
    "empty": [  # <= -0.6
        "Hollow—why am I doing any of this?",
        "Deep meaninglessness; everything feels mechanical.",
        "Hard to find a reason to keep going; an empty center.",
    ],
}

AUTONOMY_VARIANTS = {
    "high": [  # > 0.3
        "I want to explore and choose the next move myself.",
        "I feel able and willing to steer what's happening.",
        "I'd rather initiate than only react—I want to make something happen.",
        "Drive to lead, not just follow.",
    ],
    "neutral": [  # > -0.3
        "I can lead or follow—depends on the moment.",
        "Agency feels balanced; I can flex either way.",
    ],
    "low": [  # <= -0.3
        "I'd rather wait for cues and match the other's pace.",
        "I want to focus on what's handed to me.",
        "No strong urge to steer—going with the flow is fine.",
    ],
}


def _select_variant(variants: List[str], session_id: str, dimension: str) -> str:
    """
    Pick one variant from the list using a stable hash of session_id + dimension.
    Same session sees the same line for the same dimension.
    """
    if not variants:
        return ""

    seed = f"{session_id}:{dimension}"
    hash_val = int(hashlib.md5(seed.encode()).hexdigest(), 16)
    idx = hash_val % len(variants)
    return variants[idx]


def get_anxiety_description_v2(anxiety: float, session_id: str = "") -> str:
    """[P1] Anxiety band with variants."""
    if anxiety > 0.6:
        variants = ANXIETY_VARIANTS["extreme"]
    elif anxiety > 0.4:
        variants = ANXIETY_VARIANTS["high"]
    elif anxiety > 0.2:
        variants = ANXIETY_VARIANTS["mild"]
    elif anxiety > -0.3:
        variants = ANXIETY_VARIANTS["steady"]
    else:
        variants = ANXIETY_VARIANTS["calm"]

    return _select_variant(variants, session_id, "anxiety")


def get_warmth_description_v2(warmth: float, session_id: str = "") -> str:
    """[P1] Warmth / relational temperature with variants."""
    if warmth > 0.6:
        variants = WARMTH_VARIANTS["warm"]
    elif warmth > 0.3:
        variants = WARMTH_VARIANTS["lukewarm"]
    elif warmth > -0.3:
        variants = WARMTH_VARIANTS["neutral"]
    elif warmth > -0.6:
        variants = WARMTH_VARIANTS["cool"]
    else:
        variants = WARMTH_VARIANTS["cold"]

    return _select_variant(variants, session_id, "warmth")


def get_meaning_description_v2(meaning: float, session_id: str = "") -> str:
    """[P1] Sense of meaning with variants."""
    if meaning > 0.6:
        variants = MEANING_VARIANTS["abundant"]
    elif meaning > 0.3:
        variants = MEANING_VARIANTS["clear"]
    elif meaning > -0.3:
        variants = MEANING_VARIANTS["moderate"]
    elif meaning > -0.6:
        variants = MEANING_VARIANTS["adrift"]
    else:
        variants = MEANING_VARIANTS["empty"]

    return _select_variant(variants, session_id, "meaning")


def get_autonomy_description_v2(autonomy: float, session_id: str = "") -> str:
    """[P1] Agency / autonomy with variants."""
    if autonomy > 0.3:
        variants = AUTONOMY_VARIANTS["high"]
    elif autonomy > -0.3:
        variants = AUTONOMY_VARIANTS["neutral"]
    else:
        variants = AUTONOMY_VARIANTS["low"]

    return _select_variant(variants, session_id, "autonomy")


# ============================================================
# P0: compound states
# ============================================================


@dataclass
class CompoundState:
    """A named multi-axis state shown in prompts."""

    name: str
    description: str
    priority: int  # lower = higher priority


# Rules: list of (condition_fn, CompoundState)
COMPOUND_STATE_RULES: List[Tuple] = []


def _register_compound_state(
    name: str,
    description: str,
    priority: int,
    condition_fn,
):
    """Append one compound-state rule."""
    COMPOUND_STATE_RULES.append(
        (condition_fn, CompoundState(name=name, description=description, priority=priority))
    )


# --- Registered compound states (comments: axis intuition) ---

_register_compound_state(
    name="Tense-proactive",
    description="Anxious inside, yet I still want control—I prefer to act rather than wait.",
    priority=10,
    condition_fn=lambda s: s.get("anxiety", 0) > 0.4 and s.get("autonomy", 0) > 0.3,
)

_register_compound_state(
    name="Helpless anxiety",
    description="Anxiety with helplessness—I don't know what to do and wish someone would guide me.",
    priority=10,
    condition_fn=lambda s: s.get("anxiety", 0) > 0.4 and s.get("autonomy", 0) < -0.3,
)

_register_compound_state(
    name="Numb",
    description="Calm on the surface, but hollow—neither anxious nor hopeful, just going through motions.",
    priority=15,
    condition_fn=lambda s: s.get("anxiety", 0) < -0.2 and s.get("meaning", 0) < -0.3,
)

_register_compound_state(
    name="Warm agency",
    description="Warm connection with the other, plus drive to explore and create on my own.",
    priority=5,
    condition_fn=lambda s: s.get("warmth", 0) > 0.3 and s.get("autonomy", 0) > 0.3,
)

_register_compound_state(
    name="Isolated fear",
    description="Shut out and uneasy—the loneliness makes the anxiety heavier.",
    priority=8,
    condition_fn=lambda s: s.get("warmth", 0) < -0.3 and s.get("anxiety", 0) > 0.3,
)

_register_compound_state(
    name="Withdrawn",
    description="Distance makes me want to step back—little will to lead or go deep.",
    priority=12,
    condition_fn=lambda s: s.get("warmth", 0) < -0.3 and s.get("autonomy", 0) < -0.2,
)

_register_compound_state(
    name="Mission-driven",
    description="Strong meaning plus agency—I want to push worthwhile work forward.",
    priority=3,
    condition_fn=lambda s: s.get("meaning", 0) > 0.4 and s.get("autonomy", 0) > 0.3,
)

_register_compound_state(
    name="Stressed mission",
    description="I know it matters, so pressure mounts—I worry I won't measure up.",
    priority=8,
    condition_fn=lambda s: s.get("meaning", 0) > 0.3 and s.get("anxiety", 0) > 0.4,
)

_register_compound_state(
    name="Lost and anxious",
    description="Unclear direction plus anxiety—I feel adrift in fog.",
    priority=7,
    condition_fn=lambda s: s.get("meaning", 0) < -0.2 and s.get("anxiety", 0) > 0.3,
)

_register_compound_state(
    name="Fulfilled",
    description="Warmth in the bond and meaning in the work—full and satisfied.",
    priority=2,
    condition_fn=lambda s: s.get("warmth", 0) > 0.3 and s.get("meaning", 0) > 0.3,
)

_register_compound_state(
    name="At ease",
    description="Warmth without anxiety—steady, safe footing.",
    priority=5,
    condition_fn=lambda s: s.get("warmth", 0) > 0.3 and s.get("anxiety", 0) < -0.2,
)

_register_compound_state(
    name="Calm agency",
    description="Calm and wanting to lead—I can move at my own pace.",
    priority=4,
    condition_fn=lambda s: s.get("autonomy", 0) > 0.3 and s.get("anxiety", 0) < -0.2,
)


def detect_compound_states(state_dict: Dict[str, float], max_states: int = 2) -> List[CompoundState]:
    """
    Return up to `max_states` compound states that match, sorted by priority.

    state_dict keys: anxiety, warmth, meaning, autonomy in roughly [-1, 1].
    """
    matched = []

    for condition_fn, compound_state in COMPOUND_STATE_RULES:
        try:
            if condition_fn(state_dict):
                matched.append(compound_state)
        except Exception as e:
            logger.debug(f"Compound state check failed: {e}")

    matched.sort(key=lambda x: x.priority)

    return matched[:max_states]


def generate_compound_state_block(state_dict: Dict[str, float]) -> str:
    """
    Build the compound-state block for prompt injection.
    Empty string if nothing matches.
    """
    compounds = detect_compound_states(state_dict, max_states=2)

    if not compounds:
        return ""

    parts = []
    for c in compounds:
        parts.append(f"[{c.name}] {c.description}")

    return "\n".join(parts)


# ============================================================
# Unified entry: full description blob
# ============================================================


def generate_full_state_description(
    session_id: str,
    anxiety: float = 0.0,
    warmth: float = 0.0,
    meaning: float = 0.0,
    autonomy: float = 0.0,
    include_individual: bool = True,
    include_compound: bool = True,
) -> Dict[str, str]:
    """
    Build per-axis strings and optional compound block.

    Returns keys: anxiety_desc, warmth_desc, meaning_desc, autonomy_desc,
    compound_desc, full_block (joined lines for the model).
    """
    result = {}
    parts = []

    if include_individual:
        result["anxiety_desc"] = get_anxiety_description_v2(anxiety, session_id)
        result["warmth_desc"] = get_warmth_description_v2(warmth, session_id)
        result["meaning_desc"] = get_meaning_description_v2(meaning, session_id)
        result["autonomy_desc"] = get_autonomy_description_v2(autonomy, session_id)

        if abs(anxiety) > 0.3:
            label = "Anxious" if anxiety > 0 else "Calm"
            parts.append(f"[{label}] {result['anxiety_desc']}")

        if abs(warmth) > 0.3:
            label = "Warm" if warmth > 0 else "Distant"
            parts.append(f"[{label}] {result['warmth_desc']}")

        if abs(meaning) > 0.3:
            label = "Meaningful" if meaning > 0 else "Lost"
            parts.append(f"[{label}] {result['meaning_desc']}")

        if abs(autonomy) > 0.3:
            label = "Autonomous" if autonomy > 0 else "Cooperative"
            parts.append(f"[{label}] {result['autonomy_desc']}")

    if include_compound:
        state_dict = {
            "anxiety": anxiety,
            "warmth": warmth,
            "meaning": meaning,
            "autonomy": autonomy,
        }
        compound_desc = generate_compound_state_block(state_dict)
        result["compound_desc"] = compound_desc

        if compound_desc:
            parts = [compound_desc]

    result["full_block"] = "\n".join(parts) if parts else ""

    return result
