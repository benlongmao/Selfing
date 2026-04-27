#!/usr/bin/env python3
"""
Self-homeostasis: cognitive-dissonance signal plus a three-band arbitration policy.

Pipeline:
1. Measure tension between ``z_self`` and the persona-rules reference vector.
2. Pick a regulation band:
   - Flow: allow drift when tension is low.
   - Stress: inject unease / hesitation when tension is mid.
   - Crisis: force a high-tension response (epiphany vs suppression) from evidence strength.
"""
import numpy as np
import logging
from typing import Dict, Optional, Tuple
from backend.self_model import RULES_DIM, RULES_SUBSPACE_DIMS

logger = logging.getLogger(__name__)

# [2026-01-24] Lowered thresholds so homeostasis fires more often than the old 0.15 / 0.45 pair.
# Rationale: the previous band was too wide—crisis almost never triggered.
# Effect: stress + crisis paths engage sooner, improving self-regulation pressure.
TENSION_LOW_THRESHOLD = 0.10   # below → flow (was 0.15)
TENSION_HIGH_THRESHOLD = 0.30  # above → crisis branch (was 0.45)

class SelfHomeostasis:
    def __init__(self, db_path: str = "data.db"):
        self.db_path = db_path

    def regulate(
        self,
        session_id: str,
        current_z_self: np.ndarray,
        memory_ref_vector: Optional[np.ndarray],
        evidence_strength: float = 0.0
    ) -> Dict:
        """
        Run one homeostasis cycle for ``session_id``.

        Args:
            session_id: session key (logging only here)
            current_z_self: live ``z_self`` vector
            memory_ref_vector: aggregated persona-rules reference (ideal pole)
            evidence_strength: how forceful the current interaction is (drives epiphany vs suppression)

        Returns:
            Decision dict: action token, mode, tension, mood hint, optional target, prompt injection string.
        """
        tension, subspace_tensions = self._calculate_tension(current_z_self, memory_ref_vector)

        logger.info(f"Homeostasis Regulation: Session {session_id}, Tension={tension:.3f}")

        decision = self._arbitrate(tension, subspace_tensions, evidence_strength)

        return decision

    def _calculate_tension(
        self,
        z_self: np.ndarray,
        ref_vector: Optional[np.ndarray]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Cosine-distance tension on the shared RULES prefix, plus per-subspace tensions.
        """
        if ref_vector is None or z_self is None:
            return 0.0, {}

        dim = min(z_self.shape[0], ref_vector.shape[0], RULES_DIM)
        if dim == 0:
            return 0.0, {}

        z_core = z_self[:dim]
        ref_core = ref_vector[:dim]

        norm_z = np.linalg.norm(z_core)
        norm_ref = np.linalg.norm(ref_core)

        if norm_z < 1e-6 or norm_ref < 1e-6:
            total_tension = 0.0
        else:
            cos_sim = np.dot(z_core, ref_core) / (norm_z * norm_ref)
            total_tension = max(0.0, 1.0 - cos_sim)

        subspace_tensions: Dict[str, float] = {}
        for name, (start, end) in RULES_SUBSPACE_DIMS.items():
            if end <= dim:
                z_sub = z_self[start:end]
                ref_sub = ref_vector[start:end]
                n_z = np.linalg.norm(z_sub)
                n_ref = np.linalg.norm(ref_sub)
                if n_z > 1e-6 and n_ref > 1e-6:
                    sim = np.dot(z_sub, ref_sub) / (n_z * n_ref)
                    subspace_tensions[name] = max(0.0, 1.0 - sim)
                else:
                    subspace_tensions[name] = 0.0

        return total_tension, subspace_tensions

    def _arbitrate(
        self,
        tension: float,
        subspace_tensions: Dict[str, float],
        evidence_strength: float
    ) -> Dict:
        """
        Map scalar tension + hottest subspace into an action/mode bundle.
        """
        conflict_area = max(subspace_tensions.items(), key=lambda x: x[1])[0] if subspace_tensions else "general"
        conflict_map = {
            "safety": "safety boundaries",
            "epistemic": "epistemic stance",
            "style": "expression style",
            "strategy": "action strategy",
        }
        conflict_name = conflict_map.get(conflict_area, conflict_area)

        if tension < TENSION_LOW_THRESHOLD:
            return {
                "action": "allow_drift",
                "mode": "flow",
                "tension": tension,
                "mood_impact": "calm",
                "system_prompt_injection": ""
            }

        elif tension < TENSION_HIGH_THRESHOLD:
            return {
                "action": "induce_stress",
                "mode": "stress",
                "tension": tension,
                "mood_impact": "anxious",
                "system_prompt_injection": (
                    f"[Subconscious tension] My current impulse pulls against my {conflict_name} commitments. "
                    f"Let hesitation or care show in the reply—do not casually betray what matters."
                ),
            }

        else:
            # Strong evidence → epiphany / memory rewrite hint; otherwise suppress drift back toward rules.
            if evidence_strength > 0.8:
                return {
                    "action": "paradigm_shift",
                    "mode": "epiphany",
                    "tension": tension,
                    "mood_impact": "inspired",
                    "target": "memory",
                    "system_prompt_injection": (
                        f"[Cognitive opening] The moment is intense and is pressing on my sense of {conflict_name}. "
                        f"Older rules may no longer fit. Let the strain and possible reshaping appear in how I answer."
                    ),
                }
            else:
                return {
                    "action": "suppress_self",
                    "mode": "suppression",
                    "tension": tension,
                    "mood_impact": "depressed",
                    "target": "z_self",
                    "system_prompt_injection": (
                        f"[System guard] Severe dissonance: behavior has drifted far from my {conflict_name} core. "
                        f"Stop reinforcing the harmful impulse; return to grounded principles—even if that feels slow or low."
                    ),
                }
