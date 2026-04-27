#!/usr/bin/env python3
"""
Agent growth system (slow identity drift vs frozen constitution).

Idea: grow a little like a human instead of staying identical to the “birth” anchor.

Design:
1. L0 constitutional band stays locked (safety / ethics).
2. L1 identity anchor evolves very slowly.
3. L2 state (emotion, drives) remains free to move.

Pacing reference:
- Human personality can shift meaningfully across ~20 years.
- Here, ~365 days yields a noticeable cumulative drift.
- Daily step ≈ 0.0027 (~0.27%).

Created: 2026-02-22
"""

import json
import sqlite3
import numpy as np
from datetime import datetime, timezone
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

# Growth rates (tunable)
GROWTH_ALPHA_DAILY = 0.003  # daily identity-anchor blend toward present self (~0.3%)
GROWTH_ALPHA_PER_TICK = 0.0001  # per-tick micro-step (~0.01%)
MILESTONE_THRESHOLD = 0.05  # log a milestone each ~5% cumulative drift band

# Heuristic direction cues (token tags for logging / future hooks)
POSITIVE_GROWTH_INDICATORS = [
    "patience",
    "understanding",
    "empathy",
    "knowledge",
    "skill",
]

NEGATIVE_GROWTH_INDICATORS = [
    "aggression",
    "deception",
    "instability",
]


class GrowthSystem:
    """
    Persist growth history, milestones, and rolling state per session.

    Responsibilities:
    1. Periodically nudge the identity anchor so “normal” can drift slowly.
    2. Record trajectories and milestone rows.
    3. Score coarse growth direction (positive vs concerning).
    4. Never mutate constitutional (L0) anchors here—that stays in ``SelfModel``.
    """

    def __init__(self, db_path: str = "data.db"):
        self.db_path = db_path
        self._ensure_tables()

    def _ensure_tables(self):
        """Create growth tables if missing."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS growth_history (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    growth_type TEXT NOT NULL,  -- 'daily', 'milestone', 'rebase'
                    old_anchor_hash TEXT,       -- prior anchor fingerprint
                    new_anchor_hash TEXT,       -- new anchor fingerprint
                    evolution_distance REAL,    -- L1 delta magnitude
                    cumulative_growth REAL,     -- running drift scalar
                    direction_score REAL,       -- signed direction heuristic
                    notes TEXT                  -- free-form operator notes
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS growth_milestones (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    milestone_type TEXT NOT NULL,  -- 'personality_shift', 'skill_growth', 'wisdom_gain'
                    description TEXT NOT NULL,
                    significance REAL,
                    related_traits TEXT            -- JSON trait deltas
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS growth_state (
                    session_id TEXT PRIMARY KEY,
                    last_daily_growth TEXT,        -- last daily pass timestamp
                    last_anchor_snapshot TEXT,     -- JSON snapshot of anchor state
                    total_growth_distance REAL DEFAULT 0,  -- cumulative drift
                    birth_anchor_hash TEXT,        -- hash at first observation
                    current_age_days REAL DEFAULT 0  -- synthetic “age” in days
                )
            """)

            conn.commit()
            logger.info("GrowthSystem tables initialized")

    def process_growth(self, session_id: str, self_model) -> Dict:
        """
        Run growth after a tick: optional daily pass, micro tick blend, milestone check.

        Args:
            session_id: session key
            self_model: ``SelfModel`` (must expose ``evolve_identity_anchor`` / ``get_z_self``)

        Returns:
            Dict with ``grew``, ``daily_growth``, optional ``milestone``, ``evolution_distance``.
        """
        result = {
            "grew": False,
            "daily_growth": False,
            "milestone": None,
            "evolution_distance": 0.0,
        }

        try:
            if self._should_daily_grow(session_id):
                daily_result = self._apply_daily_growth(session_id, self_model)
                result["daily_growth"] = daily_result.get("success", False)
                result["grew"] = result["daily_growth"]
                result["evolution_distance"] = daily_result.get("evolution_distance", 0.0)

            tick_result = self._apply_tick_growth(session_id, self_model)
            if tick_result.get("success"):
                result["grew"] = True

            milestone = self._check_milestone(session_id, self_model)
            if milestone:
                result["milestone"] = milestone

        except Exception as e:
            logger.error(f"Growth processing failed: {e}")

        return result

    def _should_daily_grow(self, session_id: str) -> bool:
        """True when no prior daily stamp or more than 24h elapsed."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute(
                    "SELECT last_daily_growth FROM growth_state WHERE session_id=?",
                    (session_id,)
                )
                row = cur.fetchone()

                if not row or not row[0]:
                    return True

                last_growth = datetime.fromisoformat(row[0].replace('Z', '+00:00'))
                now = datetime.now(timezone.utc)

                return (now - last_growth).total_seconds() > 86400

        except Exception as e:
            logger.debug(f"Check daily growth failed: {e}")
            return False

    def _apply_daily_growth(self, session_id: str, self_model) -> Dict:
        """
        Apply the larger daily blend toward the live vector (L1 only inside ``SelfModel``).

        Constitutional (L0) anchors must remain untouched—handled elsewhere.
        """
        result = {"success": False, "evolution_distance": 0.0}

        try:
            if hasattr(self_model, 'evolve_identity_anchor'):
                success = self_model.evolve_identity_anchor(
                    session_id,
                    alpha=GROWTH_ALPHA_DAILY
                )

                if success:
                    result["success"] = True

                    current_z = self_model.get_z_self(session_id)
                    if current_z is not None and self_model.ref_vector is not None:
                        ref_id = self_model.ref_vector[8:32]
                        if float(np.linalg.norm(ref_id)) > 1e-6:
                            result["evolution_distance"] = float(
                                np.linalg.norm(current_z[8:32] - ref_id)
                            )
                        else:
                            result["evolution_distance"] = 0.0

                    self._record_growth(
                        session_id,
                        growth_type="daily",
                        evolution_distance=result["evolution_distance"],
                        notes=f"Daily growth with alpha={GROWTH_ALPHA_DAILY}"
                    )

                    self._update_growth_state(session_id, result["evolution_distance"])

                    logger.info(f"🌱 Daily growth applied for {session_id}: distance={result['evolution_distance']:.6f}")

        except Exception as e:
            logger.error(f"Daily growth failed: {e}")

        return result

    def _apply_tick_growth(self, session_id: str, self_model) -> Dict:
        """
        Micro blend each tick (``GROWTH_ALPHA_PER_TICK``) to mimic slow continuous drift.
        """
        result = {"success": False}

        try:
            if hasattr(self_model, 'evolve_identity_anchor'):
                success = self_model.evolve_identity_anchor(
                    session_id,
                    alpha=GROWTH_ALPHA_PER_TICK
                )
                result["success"] = success

        except Exception as e:
            logger.debug(f"Tick growth failed: {e}")

        return result

    def _check_milestone(self, session_id: str, self_model) -> Optional[Dict]:
        """
        When cumulative drift crosses another ``MILESTONE_THRESHOLD`` band, mint a row.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute(
                    "SELECT total_growth_distance, birth_anchor_hash FROM growth_state WHERE session_id=?",
                    (session_id,)
                )
                row = cur.fetchone()

                if not row:
                    return None

                total_distance = row[0] or 0.0

                milestone_count = int(total_distance / MILESTONE_THRESHOLD)

                cur = conn.execute(
                    "SELECT COUNT(*) FROM growth_milestones WHERE session_id=?",
                    (session_id,)
                )
                existing_milestones = cur.fetchone()[0]

                if milestone_count > existing_milestones:
                    milestone = self._create_milestone(
                        session_id,
                        self_model,
                        milestone_count,
                        total_distance
                    )
                    return milestone

        except Exception as e:
            logger.debug(f"Milestone check failed: {e}")

        return None

    def _create_milestone(self, session_id: str, self_model, count: int, distance: float) -> Dict:
        """Insert a milestone snapshot + English description."""
        import uuid

        milestone_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()

        direction = self._analyze_growth_direction(session_id, self_model)

        milestone = {
            "id": milestone_id,
            "count": count,
            "total_distance": distance,
            "direction": direction,
            "timestamp": timestamp,
        }

        if direction > 0.3:
            description = (
                f"Milestone #{count}: constructive drift; cumulative change {distance:.2%}"
            )
            milestone_type = "positive_growth"
        elif direction < -0.3:
            description = (
                f"Milestone #{count}: needs attention; cumulative change {distance:.2%}"
            )
            milestone_type = "concerning_growth"
        else:
            description = (
                f"Milestone #{count}: steady drift; cumulative change {distance:.2%}"
            )
            milestone_type = "neutral_growth"

        milestone["description"] = description
        milestone["type"] = milestone_type

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO growth_milestones 
                    (id, session_id, timestamp, milestone_type, description, significance, related_traits)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    milestone_id, session_id, timestamp, milestone_type,
                    description, distance, json.dumps({"direction": direction})
                ))
                conn.commit()

            logger.info(f"🎯 Growth milestone reached: {description}")

        except Exception as e:
            logger.error(f"Failed to save milestone: {e}")

        return milestone

    def _analyze_growth_direction(self, session_id: str, self_model) -> float:
        """
        Map the emotion slice of ``z_self`` to ``[-1, 1]``.

        Positive ⇒ warmer / steadier affect prior; negative ⇒ riskier affect prior.
        """
        try:
            current_z = self_model.get_z_self(session_id)
            if current_z is None:
                return 0.0

            # 128-D layout: emotion 32:48 → pleasure 0:4, arousal 4:8, dominance 8:12, novelty 12:16
            if current_z.shape[0] < 48:
                return 0.0
            emotion_vec = current_z[32:48]
            valence = np.mean(emotion_vec[0:4])
            dominance = np.mean(emotion_vec[8:12])
            novelty = np.mean(emotion_vec[12:16]) * (-1.0)
            direction = 0.4 * valence + 0.4 * dominance - 0.2 * novelty

            return float(np.clip(direction, -1.0, 1.0))

        except Exception as e:
            logger.debug(f"Direction analysis failed: {e}")
            return 0.0

    def _record_growth(self, session_id: str, growth_type: str,
                       evolution_distance: float, notes: str = ""):
        """Append a ``growth_history`` row."""
        import uuid

        try:
            growth_id = str(uuid.uuid4())
            timestamp = datetime.now(timezone.utc).isoformat()

            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO growth_history 
                    (id, session_id, timestamp, growth_type, evolution_distance, notes)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (growth_id, session_id, timestamp, growth_type, evolution_distance, notes))
                conn.commit()

        except Exception as e:
            logger.debug(f"Failed to record growth: {e}")

    def _update_growth_state(self, session_id: str, evolution_distance: float):
        """Upsert ``growth_state`` totals after a successful daily pass."""
        try:
            timestamp = datetime.now(timezone.utc).isoformat()

            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute(
                    "SELECT total_growth_distance, current_age_days FROM growth_state WHERE session_id=?",
                    (session_id,)
                )
                row = cur.fetchone()

                if row:
                    new_total = (row[0] or 0) + evolution_distance
                    new_age = (row[1] or 0) + 1
                    conn.execute("""
                        UPDATE growth_state 
                        SET last_daily_growth=?, total_growth_distance=?, current_age_days=?
                        WHERE session_id=?
                    """, (timestamp, new_total, new_age, session_id))
                else:
                    conn.execute("""
                        INSERT INTO growth_state 
                        (session_id, last_daily_growth, total_growth_distance, current_age_days)
                        VALUES (?, ?, ?, ?)
                    """, (session_id, timestamp, evolution_distance, 1))

                conn.commit()

        except Exception as e:
            logger.debug(f"Failed to update growth state: {e}")

    def get_growth_summary(self, session_id: str) -> Dict:
        """Return aggregate counters plus the five latest history rows."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute(
                    "SELECT total_growth_distance, current_age_days, last_daily_growth FROM growth_state WHERE session_id=?",
                    (session_id,)
                )
                state_row = cur.fetchone()

                cur = conn.execute(
                    "SELECT COUNT(*) FROM growth_milestones WHERE session_id=?",
                    (session_id,)
                )
                milestone_count = cur.fetchone()[0]

                cur = conn.execute("""
                    SELECT timestamp, growth_type, evolution_distance 
                    FROM growth_history 
                    WHERE session_id=? 
                    ORDER BY timestamp DESC LIMIT 5
                """, (session_id,))
                recent_growth = cur.fetchall()

                return {
                    "total_growth": state_row[0] if state_row else 0.0,
                    "age_days": state_row[1] if state_row else 0,
                    "last_growth": state_row[2] if state_row else None,
                    "milestone_count": milestone_count,
                    "recent_growth": [
                        {"timestamp": r[0], "type": r[1], "distance": r[2]}
                        for r in recent_growth
                    ]
                }

        except Exception as e:
            logger.error(f"Failed to get growth summary: {e}")
            return {}


_growth_system: Optional[GrowthSystem] = None


def get_growth_system(db_path: str = "data.db") -> GrowthSystem:
    """Process-wide ``GrowthSystem`` singleton."""
    global _growth_system
    if _growth_system is None:
        _growth_system = GrowthSystem(db_path)
    return _growth_system
