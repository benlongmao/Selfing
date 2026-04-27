#!/usr/bin/env python3
"""
Memory salience helpers — engineering analogy, not a neuroscience model.

- Autobiographical flavor: ``memory_class`` + ``salience_score``.
- Explicit user facts: higher ``salience_floor`` than auto-narrative caps.
- Mention counts: after N hits (default 3) add a small boost.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, Optional, Tuple

from backend.config import config


def _cfg(path: str, default: Any) -> Any:
    return config.get(path, default)


def normalize_mention_key(text: str, max_len: int = 96) -> str:
    """Stable hashed key for cross-turn mention deduping and counters."""
    if not text or not isinstance(text, str):
        return ""
    t = re.sub(r"\s+", " ", text.strip().lower())
    t = t[:max_len]
    if not t:
        return ""
    return hashlib.sha256(t.encode("utf-8", errors="ignore")).hexdigest()[:32]


def classify_memory_class(
    *,
    emotional_intensity: float,
    identity_relevance: float,
    relationship_depth: float,
    memory_type: str,
) -> str:
    """Coarse bucket: trivial chit-chat sinks; other labels follow salient axes."""
    emo = float(emotional_intensity or 0.0)
    ident = float(identity_relevance or 0.0)
    rel = float(relationship_depth or 0.0)
    mt = (memory_type or "episodic").lower()

    if mt in ("consolidation", "consolidated", "diary"):
        return "auto_narrative"

    # Low affect + low identity/relationship cues → trivial (still storable, down-ranked at retrieval).
    if emo < 0.22 and ident < 0.28 and rel < 0.28 and mt in ("episodic", "semantic", ""):
        return "trivial"

    if ident >= 0.55:
        return "identity_anchor"
    if rel >= 0.55:
        return "relational"
    if emo >= 0.48:
        return "episodic_high"
    return "auto_narrative"


def compute_salience_score(
    *,
    significance: float,
    emotional_intensity: float,
    identity_relevance: float,
    relationship_depth: float,
    memory_class: str,
) -> float:
    """Blend salience into ``[0, 1]`` on the same scale as ``significance``."""
    sig = float(significance or 0.5)
    emo = float(emotional_intensity or 0.0)
    ident = float(identity_relevance or 0.0)
    rel = float(relationship_depth or 0.0)

    base = 0.22 * emo + 0.28 * ident + 0.28 * rel + 0.22 * sig

    ceiling = float(_cfg("parameters.memory.salience.auto_narrative_ceiling", 0.85) or 0.85)

    if memory_class == "trivial":
        return min(0.32, base * 0.55)
    if memory_class == "identity_anchor":
        return min(ceiling, base + 0.18)
    if memory_class == "relational":
        return min(ceiling, base + 0.12)
    if memory_class == "episodic_high":
        return min(ceiling, base + 0.10)
    return min(ceiling, base)


def compute_biography_salience_and_class(
    *,
    significance: float,
    emotional_intensity: float,
    identity_relevance: float,
    relationship_depth: float,
    memory_type: str,
) -> Tuple[float, str]:
    mc = classify_memory_class(
        emotional_intensity=emotional_intensity,
        identity_relevance=identity_relevance,
        relationship_depth=relationship_depth,
        memory_type=memory_type,
    )
    ss = compute_salience_score(
        significance=significance,
        emotional_intensity=emotional_intensity,
        identity_relevance=identity_relevance,
        relationship_depth=relationship_depth,
        memory_class=mc,
    )
    return ss, mc


def explicit_fact_salience(*, mention_hit_count: int) -> float:
    """Salience for explicit ``please remember`` facts: high floor + mention-count boost."""
    floor = float(_cfg("parameters.memory.salience.explicit_floor", 0.92) or 0.92)
    thr = int(_cfg("parameters.memory.salience.mention_boost_threshold", 3) or 3)
    delta = float(_cfg("parameters.memory.salience.mention_boost_delta", 0.08) or 0.08)
    s = floor
    if mention_hit_count >= thr:
        s = min(1.0, s + delta)
    return s


def trivial_retrieval_penalty() -> float:
    return float(_cfg("parameters.memory.salience.trivial_retrieval_penalty", 0.12) or 0.12)
