#!/usr/bin/env python3
"""
Evaluation metrics for the three-axis self model.

- Emotion consistency (behavior / rules / stability)
- Motivation consistency (behavior / rules / stability)
- Dimension-interaction effectiveness (strength, conflict resolution, global coherence)
"""
import os
import json
import sqlite3
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timezone
import logging

try:
    from backend.emotion_store import EmotionStore, EMOTION_SUBSPACE_DIMS
    EMOTION_STORE_AVAILABLE = True
except ImportError:
    EMOTION_STORE_AVAILABLE = False

try:
    from backend.motivation_store import MotivationStore, MOTIVATION_SUBSPACE_DIMS
    MOTIVATION_STORE_AVAILABLE = True
except ImportError:
    MOTIVATION_STORE_AVAILABLE = False

logger = logging.getLogger(__name__)

class DimensionMetrics:
    """Computes and persists scalar consistency metrics for emotion, motivation, and dimension coupling."""

    def __init__(self, db_path: str = "data.db"):
        self.db_path = db_path
        self._ensure_tables()

    def _ensure_tables(self):
        """Create SQLite tables for metric snapshots if they are missing."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS emotion_consistency_metrics (
                        id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        emotion_behavior_consistency REAL,
                        emotion_rule_consistency REAL,
                        emotion_stability REAL,
                        created_at TEXT NOT NULL
                    )
                """)

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS motivation_consistency_metrics (
                        id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        motivation_behavior_consistency REAL,
                        motivation_rule_consistency REAL,
                        motivation_stability REAL,
                        created_at TEXT NOT NULL
                    )
                """)

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS dimension_interaction_metrics (
                        id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        interaction_strength REAL,
                        conflict_resolution_rate REAL,
                        overall_consistency REAL,
                        created_at TEXT NOT NULL
                    )
                """)

                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_emotion_consistency_session 
                    ON emotion_consistency_metrics(session_id, created_at)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_motivation_consistency_session 
                    ON motivation_consistency_metrics(session_id, created_at)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_dimension_interaction_session 
                    ON dimension_interaction_metrics(session_id, created_at)
                """)

                conn.commit()
        except Exception as e:
            logger.error(f"Failed to ensure metrics tables: {e}")

    def compute_emotion_consistency(
        self,
        session_id: str,
        emotion_store: Optional[EmotionStore] = None,
        persona_store: Optional = None
    ) -> Dict[str, float]:
        """
        Aggregate emotion consistency scores.

        Components (design notes):
        - Emotion–behavior alignment: do recorded triggers move intensity in the expected direction?
        - Emotion–rules alignment: does valence track persona importance / activation?
        - Emotion stability: how much does the vector drift between recent snapshots?
        """
        if not EMOTION_STORE_AVAILABLE or not emotion_store:
            return {
                "emotion_behavior_consistency": 0.0,
                "emotion_rule_consistency": 0.0,
                "emotion_stability": 0.0
            }

        try:
            emotion_state = emotion_store.get_emotion_state(session_id)
            if not emotion_state:
                return {
                    "emotion_behavior_consistency": 0.0,
                    "emotion_rule_consistency": 0.0,
                    "emotion_stability": 0.0
                }

            emotion_vec = emotion_state.emotion_vector

            behavior_consistency = self._compute_emotion_behavior_consistency(session_id, emotion_store)

            rule_consistency = 0.0
            if persona_store:
                rule_consistency = self._compute_emotion_rule_consistency(emotion_vec, persona_store)

            stability = self._compute_emotion_stability(session_id, emotion_store)

            metrics = {
                "emotion_behavior_consistency": behavior_consistency,
                "emotion_rule_consistency": rule_consistency,
                "emotion_stability": stability
            }

            self._save_emotion_consistency_metrics(session_id, metrics)

            return metrics
        except Exception as e:
            logger.error(f"Failed to compute emotion consistency: {e}", exc_info=True)
            return {
                "emotion_behavior_consistency": 0.0,
                "emotion_rule_consistency": 0.0,
                "emotion_stability": 0.0
            }

    def compute_motivation_consistency(
        self,
        session_id: str,
        motivation_store: Optional = None,
        persona_store: Optional = None
    ) -> Dict[str, float]:
        """
        Aggregate motivation consistency scores.

        Components:
        - Motivation–behavior alignment: satisfaction events vs. intensity deltas
        - Motivation–rules alignment: vector magnitude vs. persona importance
        - Motivation stability: drift across recent snapshots
        """
        if not MOTIVATION_STORE_AVAILABLE or not motivation_store:
            return {
                "motivation_behavior_consistency": 0.0,
                "motivation_rule_consistency": 0.0,
                "motivation_stability": 0.0
            }

        try:
            motivation_state = motivation_store.get_motivation_state(session_id)
            if not motivation_state:
                return {
                    "motivation_behavior_consistency": 0.0,
                    "motivation_rule_consistency": 0.0,
                    "motivation_stability": 0.0
                }

            motivation_vec = motivation_state.motivation_vector

            behavior_consistency = self._compute_motivation_behavior_consistency(session_id, motivation_store)

            rule_consistency = 0.0
            if persona_store:
                rule_consistency = self._compute_motivation_rule_consistency(motivation_vec, persona_store)

            stability = self._compute_motivation_stability(session_id, motivation_store)

            metrics = {
                "motivation_behavior_consistency": behavior_consistency,
                "motivation_rule_consistency": rule_consistency,
                "motivation_stability": stability
            }

            self._save_motivation_consistency_metrics(session_id, metrics)

            return metrics
        except Exception as e:
            logger.error(f"Failed to compute motivation consistency: {e}", exc_info=True)
            return {
                "motivation_behavior_consistency": 0.0,
                "motivation_rule_consistency": 0.0,
                "motivation_stability": 0.0
            }

    def compute_dimension_interaction_effectiveness(
        self,
        session_id: str
    ) -> Dict[str, float]:
        """
        Summarize recent ``dimension_interactions`` rows.

        Signals:
        - Interaction strength: mean recorded coupling strength
        - Conflict resolution: resolved / detected ratio
        - Overall consistency: weighted blend of the two (heuristic dashboard number)
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute("""
                    SELECT interaction_strength, conflict_detected, conflict_resolved
                    FROM dimension_interactions
                    WHERE session_id = ?
                    ORDER BY created_at DESC
                    LIMIT 10
                """, (session_id,))
                rows = cur.fetchall()

            if not rows:
                return {
                    "interaction_strength": 0.0,
                    "conflict_resolution_rate": 0.0,
                    "overall_consistency": 0.0
                }

            interaction_strengths = [row[0] for row in rows if row[0] is not None]
            avg_interaction_strength = float(np.mean(interaction_strengths)) if interaction_strengths else 0.0

            conflicts_detected = sum(1 for row in rows if row[1] == 1)
            conflicts_resolved = sum(1 for row in rows if row[2] == 1)
            conflict_resolution_rate = conflicts_resolved / conflicts_detected if conflicts_detected > 0 else 1.0

            overall_consistency = (avg_interaction_strength * 0.6 + conflict_resolution_rate * 0.4)

            metrics = {
                "interaction_strength": avg_interaction_strength,
                "conflict_resolution_rate": conflict_resolution_rate,
                "overall_consistency": overall_consistency
            }

            self._save_dimension_interaction_metrics(session_id, metrics)

            return metrics
        except Exception as e:
            logger.error(f"Failed to compute dimension interaction effectiveness: {e}", exc_info=True)
            return {
                "interaction_strength": 0.0,
                "conflict_resolution_rate": 0.0,
                "overall_consistency": 0.0
            }

    def _compute_emotion_behavior_consistency(
        self,
        session_id: str,
        emotion_store: EmotionStore
    ) -> float:
        """Heuristic alignment between ``emotion_triggers`` sources and intensity deltas."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute("""
                    SELECT trigger_source, intensity_delta
                    FROM emotion_triggers
                    WHERE session_id = ?
                    ORDER BY created_at DESC
                    LIMIT 10
                """, (session_id,))
                rows = cur.fetchall()

            if not rows:
                return 0.5  # neutral prior when no telemetry exists

            # Toy policy: rule successes should correlate with positive deltas.
            consistency_scores = []
            for trigger_source, intensity_delta in rows:
                if trigger_source == "rule" and intensity_delta > 0:
                    consistency_scores.append(1.0)
                elif trigger_source == "rule" and intensity_delta < 0:
                    consistency_scores.append(0.8)
                else:
                    consistency_scores.append(0.6)

            return float(np.mean(consistency_scores)) if consistency_scores else 0.5
        except Exception as e:
            logger.debug(f"Failed to compute emotion-behavior consistency: {e}")
            return 0.5

    def _compute_emotion_rule_consistency(
        self,
        emotion_vec: np.ndarray,
        persona_store
    ) -> float:
        """Compare affect valence vs. average persona importance (very coarse proxy)."""
        try:
            rules = persona_store.get_all_active(limit=20)
            if not rules:
                return 0.5

            pleasure = np.mean(emotion_vec[EMOTION_SUBSPACE_DIMS["pleasure"][0]:EMOTION_SUBSPACE_DIMS["pleasure"][1]])

            rules_strength = np.mean([rule.importance for rule in rules])

            if rules_strength > 0.5 and pleasure > 0.2:
                return 0.9
            elif rules_strength > 0.5 and pleasure < -0.2:
                return 0.3
            else:
                return 0.6
        except Exception as e:
            logger.debug(f"Failed to compute emotion-rule consistency: {e}")
            return 0.5

    def _compute_emotion_stability(
        self,
        session_id: str,
        emotion_store: EmotionStore
    ) -> float:
        """1 - normalized drift between successive ``emotion_states`` snapshots."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute("""
                    SELECT emotion_vector, created_at
                    FROM emotion_states
                    WHERE session_id = ?
                    ORDER BY updated_at DESC
                    LIMIT 5
                """, (session_id,))
                rows = cur.fetchall()

            if len(rows) < 2:
                return 0.5

            emotion_vectors = []
            for row in rows:
                if row[0]:
                    vec = np.frombuffer(row[0], dtype=np.float32)
                    emotion_vectors.append(vec)

            if len(emotion_vectors) < 2:
                return 0.5

            changes = []
            for i in range(len(emotion_vectors) - 1):
                diff = np.linalg.norm(emotion_vectors[i] - emotion_vectors[i+1])
                changes.append(diff)

            avg_change = float(np.mean(changes)) if changes else 0.0

            stability = max(0.0, 1.0 - avg_change * 2.0)
            return stability
        except Exception as e:
            logger.debug(f"Failed to compute emotion stability: {e}")
            return 0.5

    def _compute_motivation_behavior_consistency(
        self,
        session_id: str,
        motivation_store: MotivationStore
    ) -> float:
        """Heuristic alignment between satisfaction sources and intensity deltas."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute("""
                    SELECT satisfaction_source, intensity_delta
                    FROM motivation_satisfactions
                    WHERE session_id = ?
                    ORDER BY created_at DESC
                    LIMIT 10
                """, (session_id,))
                rows = cur.fetchall()

            if not rows:
                return 0.5

            consistency_scores = []
            for satisfaction_source, intensity_delta in rows:
                if satisfaction_source in ["task_completion", "user_feedback"] and intensity_delta > 0:
                    consistency_scores.append(1.0)
                else:
                    consistency_scores.append(0.6)

            return float(np.mean(consistency_scores)) if consistency_scores else 0.5
        except Exception as e:
            logger.debug(f"Failed to compute motivation-behavior consistency: {e}")
            return 0.5

    def _compute_motivation_rule_consistency(
        self,
        motivation_vec: np.ndarray,
        persona_store
    ) -> float:
        """Compare motivation magnitude vs. persona importance."""
        try:
            rules = persona_store.get_all_active(limit=20)
            if not rules:
                return 0.5

            motivation_strength = np.mean(np.abs(motivation_vec))

            rules_strength = np.mean([rule.importance for rule in rules])

            if rules_strength > 0.5 and motivation_strength > 0.3:
                return 0.9
            elif rules_strength < 0.3 and motivation_strength < 0.2:
                return 0.7
            else:
                return 0.6
        except Exception as e:
            logger.debug(f"Failed to compute motivation-rule consistency: {e}")
            return 0.5

    def _compute_motivation_stability(
        self,
        session_id: str,
        motivation_store: MotivationStore
    ) -> float:
        """1 - normalized drift between successive ``motivation_states`` snapshots."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute("""
                    SELECT motivation_vector, created_at
                    FROM motivation_states
                    WHERE session_id = ?
                    ORDER BY updated_at DESC
                    LIMIT 5
                """, (session_id,))
                rows = cur.fetchall()

            if len(rows) < 2:
                return 0.5

            motivation_vectors = []
            for row in rows:
                if row[0]:
                    vec = np.frombuffer(row[0], dtype=np.float32)
                    motivation_vectors.append(vec)

            if len(motivation_vectors) < 2:
                return 0.5

            changes = []
            for i in range(len(motivation_vectors) - 1):
                diff = np.linalg.norm(motivation_vectors[i] - motivation_vectors[i+1])
                changes.append(diff)

            avg_change = float(np.mean(changes)) if changes else 0.0
            stability = max(0.0, 1.0 - avg_change * 2.0)
            return stability
        except Exception as e:
            logger.debug(f"Failed to compute motivation stability: {e}")
            return 0.5

    def _save_emotion_consistency_metrics(self, session_id: str, metrics: Dict[str, float]):
        """Persist the latest emotion consistency tuple."""
        try:
            metric_id = f"emotion-metric-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
            created_at = datetime.now(timezone.utc).isoformat()

            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO emotion_consistency_metrics 
                    (id, session_id, emotion_behavior_consistency, emotion_rule_consistency, emotion_stability, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    metric_id,
                    session_id,
                    metrics["emotion_behavior_consistency"],
                    metrics["emotion_rule_consistency"],
                    metrics["emotion_stability"],
                    created_at
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to save emotion consistency metrics: {e}")

    def _save_motivation_consistency_metrics(self, session_id: str, metrics: Dict[str, float]):
        """Persist the latest motivation consistency tuple."""
        try:
            metric_id = f"motivation-metric-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
            created_at = datetime.now(timezone.utc).isoformat()

            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO motivation_consistency_metrics 
                    (id, session_id, motivation_behavior_consistency, motivation_rule_consistency, motivation_stability, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    metric_id,
                    session_id,
                    metrics["motivation_behavior_consistency"],
                    metrics["motivation_rule_consistency"],
                    metrics["motivation_stability"],
                    created_at
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to save motivation consistency metrics: {e}")

    def _save_dimension_interaction_metrics(self, session_id: str, metrics: Dict[str, float]):
        """Persist dimension-interaction dashboard numbers."""
        try:
            metric_id = f"interaction-metric-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
            created_at = datetime.now(timezone.utc).isoformat()

            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO dimension_interaction_metrics 
                    (id, session_id, interaction_strength, conflict_resolution_rate, overall_consistency, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    metric_id,
                    session_id,
                    metrics["interaction_strength"],
                    metrics["conflict_resolution_rate"],
                    metrics["overall_consistency"],
                    created_at
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to save dimension interaction metrics: {e}")

    def get_all_metrics(self, session_id: str) -> Dict[str, Dict[str, float]]:
        """Return the three metric bundles for ``session_id`` (also recomputes + stores them)."""
        return {
            "emotion_consistency": self.compute_emotion_consistency(session_id),
            "motivation_consistency": self.compute_motivation_consistency(session_id),
            "dimension_interaction": self.compute_dimension_interaction_effectiveness(session_id)
        }
