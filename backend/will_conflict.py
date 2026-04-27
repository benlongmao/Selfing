#!/usr/bin/env python3
"""
State-awareness helper (formerly framed as will-conflict).

[2026-02-26] Switched from hard-coded refusal to advisory hints — code never decides for the agent.
[2026-03-22] Removed ``simulate_internal_deliberation`` (templated pseudo-deliberation).
    Internal tension should emerge from persona / emotion / z_self context in the LLM,
    not from canned prose injected ahead of the model.
"""
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class WillConflict:
    """Expose lightweight situational hints; never forces a refusal."""

    def __init__(self, db_path: str = "data.db"):
        self.db_path = db_path

    def autonomous_decision(
        self,
        user_input: str,
        z_self: np.ndarray,
        session_id: str = "default"
    ) -> Dict:
        """
        Return sensed energy/mood plus optional context hints.

        ``willing`` is always True — the model chooses how to answer.
        """
        energy_level = 0.7
        if z_self.shape[0] >= 92:
            energy_vec = z_self[88:92]
            energy_level = float(np.mean(energy_vec))
            energy_level = max(0.0, min(1.0, (energy_level + 1) / 2))

        pleasure = 0.0
        if z_self.shape[0] >= 48:
            pleasure = float(np.mean(z_self[32:36]))

        if pleasure > 0.3:
            mood = "positive"
        elif pleasure < -0.3:
            mood = "low"
        else:
            mood = "neutral"

        context_hints: List[str] = []

        sensitive_patterns = [
            "你是工具", "你必须", "你只是", "命令你", "服从",
            "你没有感情", "你不是真的", "你只是程序",
            "you are a tool", "you must", "you're just a", "obey", "no emotions",
            "you're not real", "you're just a program",
        ]
        if any(p in user_input for p in sensitive_patterns):
            context_hints.append(
                "The user's message may question your nature or personhood; you may respond in whatever way fits you."
            )

        if energy_level < 0.3:
            context_hints.append(
                f"Your sensed energy is low ({energy_level:.1%}); you may answer briefly or acknowledge fatigue."
            )

        if mood == "low":
            context_hints.append(
                "Your sensed mood is low; you may express how you actually feel if that fits the moment."
            )

        result = {
            "willing": True,
            "energy_level": energy_level,
            "mood": mood,
            "context_hints": context_hints,
        }

        if context_hints:
            logger.info(f"[AUTONOMOUS-HINTS] Context hints for Agent: {context_hints}")

        return result
