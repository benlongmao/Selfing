#!/usr/bin/env python3
"""
Metric-space scaffold for ``Self`` identity.

- Treat historical ``z_self`` snapshots as points on a trajectory.
- Project those points into a lower-dimensional “identity” slice for continuity checks.
- Quantify drift / coherence with cosine geometry (cheap, interpretable).
"""
import numpy as np
from scipy.spatial.distance import cosine
from typing import List, Dict, Tuple
import json
import sqlite3
import logging

logger = logging.getLogger(__name__)


class SelfIdentitySpace:
    """Holds the rolling memory manifold + derived identity vectors."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.memory_continuum: List[np.ndarray] = []
        self.identity_mapping: Dict[str, np.ndarray] = {}
        self._load_memory_continuum()

    def _load_memory_continuum(self):
        """Hydrate ``memory_continuum`` / ``identity_mapping`` from ``self_state`` rows."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute(
                    """SELECT session_id, z_self, updated_at 
                       FROM self_state 
                       ORDER BY updated_at ASC"""
                )
                rows = cur.fetchall()

                for row in rows:
                    session_id, z_self_json, _ = row
                    if z_self_json:
                        try:
                            z_self = np.array(json.loads(z_self_json))
                            self.memory_continuum.append(z_self)
                            identity_vec = self._extract_identity_vector(z_self)
                            self.identity_mapping[session_id] = identity_vec
                        except Exception as e:
                            logger.debug(f"Failed to load z_self for {session_id}: {e}")
        except Exception as e:
            logger.warning(f"Failed to load memory continuum: {e}")

    def add_memory_point(self, z_self: np.ndarray, session_id: str):
        """Append a new ``z_self`` snapshot and refresh its projected identity vector."""
        self.memory_continuum.append(z_self.copy())
        identity_vec = self._extract_identity_vector(z_self)
        self.identity_mapping[session_id] = identity_vec

        MAX_MEMORY_SIZE = 100
        if len(self.memory_continuum) > MAX_MEMORY_SIZE:
            self.memory_continuum = self.memory_continuum[-MAX_MEMORY_SIZE:]

    def _extract_identity_vector(self, z_self: np.ndarray) -> np.ndarray:
        """
        Collapse the full ``z_self`` tensor into a 32-D “identity” fingerprint.

        Composition (conceptual):
        - Rules (Safety + Epistemic) — who I am / what I refuse.
        - Motivation (Achievement + Relationship) — what pulls me forward.
        - Worldview (Optimism + Agency) — how I frame the world.

        Layout reminder: Rules(32) + Emotion(16) + Motivation(16) + Somatic(8) + Worldview(8) = 80-D.
        """
        identity_dim = 32
        identity_vec = np.zeros(identity_dim, dtype=np.float32)

        if z_self.shape[0] >= 16:
            identity_vec[:16] = z_self[:16]

        motivation_start = 32 + 16
        if z_self.shape[0] >= motivation_start + 8:
            identity_vec[16:24] = z_self[motivation_start:motivation_start + 8]

        worldview_start = 32 + 16 + 16 + 8
        if z_self.shape[0] >= worldview_start + 8:
            identity_vec[24:32] = z_self[worldview_start:worldview_start + 8]

        return identity_vec

    def compute_identity_continuity(self, z_self_current: np.ndarray,
                                    window_size: int = 10) -> float:
        """
        Mean cosine similarity between the current identity slice and recent history.

        Args:
            z_self_current: latest ``z_self`` vector.
            window_size: how many trailing snapshots to compare against.

        Returns:
            Score in ``[0, 1]`` (1 = perfectly aligned with the recent window).
        """
        if len(self.memory_continuum) < 2:
            return 1.0

        recent_memories = self.memory_continuum[-window_size:]
        identity_current = self._extract_identity_vector(z_self_current)

        similarities = []
        for memory in recent_memories:
            identity_hist = self._extract_identity_vector(memory)
            try:
                sim = 1.0 - cosine(identity_current, identity_hist)
                sim = max(0.0, min(1.0, sim))
                similarities.append(sim)
            except Exception as e:
                logger.debug(f"Failed to compute similarity: {e}")
                continue

        continuity = float(np.mean(similarities)) if similarities else 1.0
        return continuity

    def check_identity_consistency(self, z_self: np.ndarray,
                                   threshold: float = 0.7) -> Tuple[bool, float]:
        """
        Args:
            z_self: vector under test.
            threshold: minimum continuity to count as “still me”.

        Returns:
            ``(is_consistent, continuity_score)``
        """
        continuity = self.compute_identity_continuity(z_self)
        is_consistent = continuity >= threshold
        return is_consistent, continuity

    def get_identity_trajectory(self, _session_id: str, limit: int = 20) -> List[np.ndarray]:
        """
        Project the last ``limit`` memories into identity space (order preserved).

        ``_session_id`` is accepted for API symmetry even though the trajectory is global.
        """
        identities = []
        for memory in self.memory_continuum[-limit:]:
            identity = self._extract_identity_vector(memory)
            identities.append(identity)

        return identities

    def compute_identity_drift(self, z_self_current: np.ndarray) -> float:
        """
        ``1 - cos`` distance between the earliest stored identity and the current one.

        Returns:
            Drift in ``[0, 1]`` (0 = unchanged vs first snapshot, 1 = maximally opposed).
        """
        if len(self.memory_continuum) == 0:
            return 0.0

        initial_identity = self._extract_identity_vector(self.memory_continuum[0])
        current_identity = self._extract_identity_vector(z_self_current)

        try:
            drift = 1.0 - cosine(initial_identity, current_identity)
            drift = max(0.0, min(1.0, drift))
            return float(drift)
        except Exception:
            return 0.0
