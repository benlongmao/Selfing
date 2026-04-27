#!/usr/bin/env python3
"""
Hypothesis generation for the agent (“if … then …” style probes).

Goals:
1. Learn coarse causal patterns from historical telemetry.
2. Emit hypotheses that can later be scored against outcomes.
3. Track verification accuracy over time.
4. Feed lightweight priors into planning / reflection.
"""

import sqlite3
import json
import logging
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class Hypothesis:
    """In-memory representation mirrored by the ``hypotheses`` table."""
    id: str
    session_id: str
    hypothesis_type: str  # "causal", "predictive", "explanatory"
    condition: str  # antecedent / context string
    prediction: str  # consequent string
    confidence: float  # 0-1
    evidence_count: int  # supporting rows / votes
    created_at: str

    status: str = "pending"  # "pending", "confirmed", "refuted", "partial"
    accuracy: Optional[float] = None  # post-verification score
    verified_at: Optional[str] = None

    source_data: Optional[Dict] = None  # serialized provenance blob
    tags: Optional[List[str]] = None  # optional classifier tags


class HypothesisGenerator:
    """SQLite-backed hypothesis lifecycle (generate → persist → verify)."""

    def __init__(self, db_path: str = "data.db"):
        self.db_path = db_path
        self._ensure_tables()

    def _ensure_tables(self):
        """Create hypothesis-related tables if missing."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS hypotheses (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    hypothesis_type TEXT NOT NULL,
                    condition TEXT NOT NULL,
                    prediction TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    evidence_count INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'pending',
                    accuracy REAL,
                    created_at TEXT NOT NULL,
                    verified_at TEXT,
                    source_data TEXT,
                    tags TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS hypothesis_verifications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    hypothesis_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    actual_outcome TEXT NOT NULL,
                    expected_outcome TEXT NOT NULL,
                    match_score REAL NOT NULL,
                    verified_at TEXT NOT NULL,
                    metadata TEXT,
                    FOREIGN KEY (hypothesis_id) REFERENCES hypotheses(id)
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS causal_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    pattern_type TEXT NOT NULL,
                    condition_key TEXT NOT NULL,
                    outcome_key TEXT NOT NULL,
                    correlation_strength REAL NOT NULL,
                    observation_count INTEGER DEFAULT 1,
                    last_observed TEXT NOT NULL,
                    metadata TEXT
                )
            """)

            conn.commit()

    # --- Generation ---

    def generate_causal_hypothesis(
        self,
        session_id: str,
        action: str,
        current_state: Dict
    ) -> Optional[Hypothesis]:
        """
        Causal template: “If I take action X, energy moves in direction Y.”

        Uses ``autonomous_actions_log`` rows for the same ``action_type``.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute("""
                    SELECT 
                        energy_before, 
                        energy_after,
                        result,
                        metadata
                    FROM autonomous_actions_log
                    WHERE session_id = ? AND action_type = ? AND status = 'completed'
                    ORDER BY execution_completed DESC
                    LIMIT 10
                """, (session_id, action))

                records = cur.fetchall()

            if len(records) < 2:
                return self._generate_exploratory_hypothesis(
                    session_id, action, current_state
                )

            energy_changes = []

            for record in records:
                energy_before = record[0]
                energy_after = record[1]
                if energy_before and energy_after:
                    energy_changes.append(energy_after - energy_before)

            avg_energy_change = np.mean(energy_changes) if energy_changes else 0
            std_energy_change = np.std(energy_changes) if energy_changes else 0

            confidence = self._calculate_hypothesis_confidence(
                len(records), std_energy_change
            )

            condition = f"Take action: {action}"

            if avg_energy_change > 5:
                prediction = f"Energy should rise materially (avg +{avg_energy_change:.1f})"
            elif avg_energy_change < -5:
                prediction = f"Energy should fall materially (avg {avg_energy_change:.1f})"
            else:
                prediction = f"Energy delta stays small (avg {avg_energy_change:.1f})"

            hypothesis = Hypothesis(
                id=f"hyp_{session_id}_{datetime.now().timestamp()}",
                session_id=session_id,
                hypothesis_type="causal",
                condition=condition,
                prediction=prediction,
                confidence=confidence,
                evidence_count=len(records),
                created_at=datetime.now(timezone.utc).isoformat(),
                source_data={
                    "action": action,
                    "sample_size": len(records),
                    "avg_energy_change": float(avg_energy_change),
                    "std_energy_change": float(std_energy_change)
                },
                tags=["action_outcome", "energy"]
            )

            self._save_hypothesis(hypothesis)

            logger.info(
                f"[HypothesisGen] Generated causal hypothesis: {condition} -> {prediction} "
                f"(confidence={confidence:.2f})"
            )

            return hypothesis

        except Exception as e:
            logger.error(f"Failed to generate causal hypothesis: {e}")
            return None

    def generate_predictive_hypothesis(
        self,
        session_id: str,
        current_state: Dict
    ) -> Optional[Hypothesis]:
        """
        Predictive template: “Given today’s state, the next likely action is …”.

        Uses a coarse histogram over historical ``action_type`` near the current energy band.
        """
        try:
            energy = current_state.get("energy", 50)
            pain = current_state.get("pain", 0)

            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute("""
                    SELECT action_type, COUNT(*) as freq
                    FROM autonomous_actions_log
                    WHERE session_id = ? 
                      AND status = 'completed'
                      AND energy_before BETWEEN ? AND ?
                    GROUP BY action_type
                    ORDER BY freq DESC
                    LIMIT 3
                """, (session_id, energy - 10, energy + 10))

                likely_actions = cur.fetchall()

            if not likely_actions:
                return None

            most_likely_action = likely_actions[0][0]
            frequency = likely_actions[0][1]
            total_records = sum(row[1] for row in likely_actions)

            confidence = frequency / max(total_records, 1)

            condition = f"Current energy={energy:.1f}, pain={pain:.2f}"
            prediction = f"Likely next action: {most_likely_action}"

            hypothesis = Hypothesis(
                id=f"hyp_{session_id}_{datetime.now().timestamp()}",
                session_id=session_id,
                hypothesis_type="predictive",
                condition=condition,
                prediction=prediction,
                confidence=confidence,
                evidence_count=frequency,
                created_at=datetime.now(timezone.utc).isoformat(),
                source_data={
                    "energy": energy,
                    "pain": pain,
                    "predicted_action": most_likely_action,
                    "likely_actions": [
                        {"action": row[0], "freq": row[1]}
                        for row in likely_actions
                    ]
                },
                tags=["self_prediction", "action"]
            )

            self._save_hypothesis(hypothesis)

            logger.info(
                f"[HypothesisGen] Generated predictive hypothesis: {prediction} "
                f"(confidence={confidence:.2f})"
            )

            return hypothesis

        except Exception as e:
            logger.error(f"Failed to generate predictive hypothesis: {e}")
            return None

    def generate_explanatory_hypothesis(
        self,
        session_id: str,
        observation: str,
        context: Dict
    ) -> Optional[Hypothesis]:
        """
        Explanatory template: "Observation X is probably because Y."

        Small pain-focused heuristic when lexical cues appear (EN + ZH).
        """
        try:
            obs_l = observation.lower()
            pain_en = ("pain", "hurt", "hurting", "ache", "aching", "suffer", "distress")
            pain_zh = ("痛苦", "疼", "难受")
            if any(t in obs_l for t in pain_en) or any(t in observation for t in pain_zh):
                with sqlite3.connect(self.db_path) as conn:
                    cur = conn.execute("""
                        SELECT z_self, tick
                        FROM self_state
                        WHERE session_id = ?
                    """, (session_id,))
                    row = cur.fetchone()

                if row:
                    z_self = np.array(json.loads(row[0]))
                    energy = float(z_self[66]) if len(z_self) > 66 else 50

                    if energy < 30:
                        explanation = "Low energy likely drives the pain signal"
                        confidence = 0.8
                    else:
                        explanation = "Likely cognitive dissonance or internal conflict"
                        confidence = 0.6
                else:
                    explanation = "Cause unclear (insufficient telemetry)"
                    confidence = 0.3

                hypothesis = Hypothesis(
                    id=f"hyp_{session_id}_{datetime.now().timestamp()}",
                    session_id=session_id,
                    hypothesis_type="explanatory",
                    condition=observation,
                    prediction=explanation,
                    confidence=confidence,
                    evidence_count=1,
                    created_at=datetime.now(timezone.utc).isoformat(),
                    source_data=context,
                    tags=["pain", "explanation"]
                )

                self._save_hypothesis(hypothesis)

                return hypothesis

            return None

        except Exception as e:
            logger.error(f"Failed to generate explanatory hypothesis: {e}")
            return None

    def _generate_exploratory_hypothesis(
        self,
        session_id: str,
        action: str,
        current_state: Dict
    ) -> Hypothesis:
        """Low-data fallback: mark the causal question as exploratory."""
        hypothesis = Hypothesis(
            id=f"hyp_{session_id}_{datetime.now().timestamp()}",
            session_id=session_id,
            hypothesis_type="causal",
            condition=f"Take action: {action}",
            prediction="Outcome unknown — needs deliberate exploration",
            confidence=0.1,
            evidence_count=0,
            created_at=datetime.now(timezone.utc).isoformat(),
            source_data={"action": action, "note": "exploratory"},
            tags=["exploratory", "unknown"]
        )

        self._save_hypothesis(hypothesis)
        return hypothesis

    def _calculate_hypothesis_confidence(
        self,
        sample_size: int,
        variance: float
    ) -> float:
        """
        Heuristic confidence from sample size and energy-variance stability.

        Larger ``sample_size`` and lower ``variance`` increase confidence (capped).
        """
        size_factor = min(0.8, 0.2 + sample_size * 0.06)

        if variance < 5:
            variance_factor = 1.0
        elif variance < 10:
            variance_factor = 0.8
        elif variance < 20:
            variance_factor = 0.6
        else:
            variance_factor = 0.4

        confidence = size_factor * variance_factor
        return min(0.95, max(0.1, confidence))

    # --- Verification ---

    def verify_hypothesis(
        self,
        hypothesis_id: str,
        actual_outcome: Dict
    ) -> float:
        """
        Score a stored hypothesis against ``actual_outcome``.

        Returns:
            Accuracy scalar in ``[0, 1]``.
        """
        try:
            hypothesis = self.get_hypothesis(hypothesis_id)
            if not hypothesis:
                return 0.0

            if hypothesis.hypothesis_type == "causal":
                accuracy = self._verify_causal(hypothesis, actual_outcome)
            elif hypothesis.hypothesis_type == "predictive":
                accuracy = self._verify_predictive(hypothesis, actual_outcome)
            else:
                accuracy = 0.5  # explanatory rows are hard to auto-grade

            if accuracy > 0.7:
                status = "confirmed"
            elif accuracy < 0.3:
                status = "refuted"
            else:
                status = "partial"

            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE hypotheses
                    SET status = ?, accuracy = ?, verified_at = ?
                    WHERE id = ?
                """, (
                    status,
                    accuracy,
                    datetime.now(timezone.utc).isoformat(),
                    hypothesis_id
                ))

                conn.execute("""
                    INSERT INTO hypothesis_verifications
                    (hypothesis_id, session_id, actual_outcome, expected_outcome, 
                     match_score, verified_at, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    hypothesis_id,
                    hypothesis.session_id,
                    json.dumps(actual_outcome),
                    hypothesis.prediction,
                    accuracy,
                    datetime.now(timezone.utc).isoformat(),
                    json.dumps({"hypothesis_type": hypothesis.hypothesis_type})
                ))

                conn.commit()

            logger.info(
                f"[HypothesisGen] Verified hypothesis {hypothesis_id}: "
                f"accuracy={accuracy:.2f}, status={status}"
            )

            return accuracy

        except Exception as e:
            logger.error(f"Failed to verify hypothesis: {e}")
            return 0.0

    def _verify_causal(self, hypothesis: Hypothesis, actual: Dict) -> float:
        """Compare recorded mean energy delta vs ``actual['energy_change']``."""
        source = hypothesis.source_data
        if not source:
            return 0.5

        expected_change = source.get("avg_energy_change", 0)
        actual_change = actual.get("energy_change", 0)

        if expected_change == 0:
            error = abs(actual_change)
        else:
            error = abs(actual_change - expected_change) / abs(expected_change)

        accuracy = max(0.0, 1.0 - error)
        return accuracy

    def _verify_predictive(self, hypothesis: Hypothesis, actual: Dict) -> float:
        """Naive equality check on predicted vs actual action tokens."""
        predicted_action = hypothesis.source_data.get("predicted_action") or hypothesis.source_data.get("action")
        actual_action = actual.get("action")

        if predicted_action == actual_action:
            return 1.0
        else:
            return 0.0

    # --- Persistence helpers ---

    def _save_hypothesis(self, hypothesis: Hypothesis):
        """Insert or replace hypothesis row (insert-only here)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO hypotheses
                (id, session_id, hypothesis_type, condition, prediction, 
                 confidence, evidence_count, status, created_at, source_data, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                hypothesis.id,
                hypothesis.session_id,
                hypothesis.hypothesis_type,
                hypothesis.condition,
                hypothesis.prediction,
                hypothesis.confidence,
                hypothesis.evidence_count,
                hypothesis.status,
                hypothesis.created_at,
                json.dumps(hypothesis.source_data) if hypothesis.source_data else None,
                json.dumps(hypothesis.tags) if hypothesis.tags else None
            ))
            conn.commit()

    def get_hypothesis(self, hypothesis_id: str) -> Optional[Hypothesis]:
        """Fetch a single hypothesis by primary key."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("""
                SELECT id, session_id, hypothesis_type, condition, prediction,
                       confidence, evidence_count, status, accuracy, created_at,
                       verified_at, source_data, tags
                FROM hypotheses
                WHERE id = ?
            """, (hypothesis_id,))
            row = cur.fetchone()

        if not row:
            return None

        return Hypothesis(
            id=row[0],
            session_id=row[1],
            hypothesis_type=row[2],
            condition=row[3],
            prediction=row[4],
            confidence=row[5],
            evidence_count=row[6],
            status=row[7],
            accuracy=row[8],
            created_at=row[9],
            verified_at=row[10],
            source_data=json.loads(row[11]) if row[11] else None,
            tags=json.loads(row[12]) if row[12] else None
        )

    def get_recent_hypotheses(
        self,
        session_id: str,
        limit: int = 10
    ) -> List[Hypothesis]:
        """Return newest hypotheses for a session."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("""
                SELECT id, session_id, hypothesis_type, condition, prediction,
                       confidence, evidence_count, status, accuracy, created_at,
                       verified_at, source_data, tags
                FROM hypotheses
                WHERE session_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (session_id, limit))
            rows = cur.fetchall()

        hypotheses = []
        for row in rows:
            hypotheses.append(Hypothesis(
                id=row[0],
                session_id=row[1],
                hypothesis_type=row[2],
                condition=row[3],
                prediction=row[4],
                confidence=row[5],
                evidence_count=row[6],
                status=row[7],
                accuracy=row[8],
                created_at=row[9],
                verified_at=row[10],
                source_data=json.loads(row[11]) if row[11] else None,
                tags=json.loads(row[12]) if row[12] else None
            ))

        return hypotheses

    def get_statistics(self, session_id: str) -> Dict:
        """Aggregate counts / mean accuracy for dashboards."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("""
                SELECT COUNT(*) FROM hypotheses WHERE session_id = ?
            """, (session_id,))
            total = cur.fetchone()[0]

            cur = conn.execute("""
                SELECT status, COUNT(*) 
                FROM hypotheses 
                WHERE session_id = ?
                GROUP BY status
            """, (session_id,))
            status_counts = dict(cur.fetchall())

            cur = conn.execute("""
                SELECT AVG(accuracy)
                FROM hypotheses
                WHERE session_id = ? AND accuracy IS NOT NULL
            """, (session_id,))
            avg_accuracy = cur.fetchone()[0] or 0

        return {
            "total": total,
            "status_counts": status_counts,
            "average_accuracy": avg_accuracy,
            "confirmed_rate": status_counts.get("confirmed", 0) / max(total, 1)
        }


_global_generator: Optional[HypothesisGenerator] = None


def get_hypothesis_generator(db_path: str = "data.db") -> HypothesisGenerator:
    """Process-wide lazy singleton."""
    global _global_generator
    if _global_generator is None or _global_generator.db_path != db_path:
        _global_generator = HypothesisGenerator(db_path)
        logger.info("[HypothesisGen] Initialized global generator")
    return _global_generator
