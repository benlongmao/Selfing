from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a) + 1e-8)
    nb = float(np.linalg.norm(b) + 1e-8)
    return float(np.dot(a, b) / (na * nb))


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def estimate_novelty_signal(
    *,
    embedder: Any,
    user_input: str,
    prev_user_input: Optional[str],
    tool_calls: Optional[List[Dict[str, Any]]] = None,
    memory_signal: Optional[Dict[str, Any]] = None,
    tool_learning_bonus: float = 0.15,
) -> Tuple[float, Dict[str, float]]:
    """
    Returns:
        - ``strength`` in ``[0, 1]`` — scalar novelty signal for downstream hooks.
        - ``components`` — interpretable breakdown (same keys as internal weights).

    Components:
        - ``topic_shift``: embedding distance between this turn's user text and the previous turn's
          (more dissimilar → higher novelty).
        - ``tool_learning``: proxy for information gain when tools ran (search / IO / browse, etc.).
        - ``memory_uncertainty``: weak bump when memory retrieval is weak (low strength → possibly new).
    """
    ui = (user_input or "").strip()
    prev = (prev_user_input or "").strip()

    # 1) topic_shift
    topic_shift = 0.6  # first turn / no history: mild default so novelty is not stuck at zero
    if ui and prev:
        try:
            a = np.asarray(embedder.encode(ui), dtype=np.float32)
            b = np.asarray(embedder.encode(prev), dtype=np.float32)
            sim = _cos_sim(a, b)  # typically ~0..1
            # Map cosine similarity into a 0..1 novelty-ish score
            topic_shift = _clamp01((1.0 - sim) / 1.2)
        except Exception:
            topic_shift = 0.5

    # 2) tool_learning
    tool_learning = 0.0
    try:
        calls = tool_calls or []
        if calls:
            tool_learning = 0.35
            # Extra weight for "learning-shaped" tools
            names = []
            for c in calls:
                n = str(c.get("name") or c.get("tool") or "")
                if n:
                    names.append(n.lower())
            if any(("search" in n) or ("tavily" in n) or ("browse" in n) for n in names):
                tool_learning += 0.35
            if any(("read" in n) or ("file" in n) or ("list" in n) for n in names):
                tool_learning += 0.15
            if any(("write" in n) or ("apply" in n) for n in names):
                tool_learning += 0.10
            tool_learning = _clamp01(tool_learning)
    except Exception:
        tool_learning = 0.0

    # 3) memory_uncertainty
    mem_uncertainty = 0.0
    try:
        ms = memory_signal or {}
        strength = float(ms.get("strength", 0.0) or 0.0)  # 0..1
        # Strong memory hit → less novel; weak hit → small uncertainty bump
        mem_uncertainty = _clamp01(1.0 - strength)
    except Exception:
        mem_uncertainty = 0.5

    # 4) blend
    # topic_shift dominates (keeps the signal alive across turns)
    # tool_learning is a strong proxy for fresh external evidence
    # memory_uncertainty is a light nudge only
    strength = (
        0.10
        + 0.60 * topic_shift
        + 0.25 * tool_learning
        + 0.05 * mem_uncertainty
    )
    # Optional extra bump whenever any tool ran
    if tool_learning > 0.0:
        strength += float(tool_learning_bonus)

    strength = _clamp01(float(strength))
    return strength, {
        "topic_shift": float(topic_shift),
        "tool_learning": float(tool_learning),
        "memory_uncertainty": float(mem_uncertainty),
    }


