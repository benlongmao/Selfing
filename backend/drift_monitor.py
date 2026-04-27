#!/usr/bin/env python3
"""
Drift detection and rollback helpers.

- Tracks identity drift against versioned ``z_self`` snapshots
- Persists historical vectors for forensic rollback
- Supports automatic rollback when constitutional (L0) bands are violated
"""
import os
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Optional, List, Dict
import numpy as np
from backend.self_model import SelfModel
import logging

logger = logging.getLogger(__name__)

SELF_DRIFT_THRESHOLD = float(os.environ.get("SELF_DRIFT_THRESHOLD", "0.15"))
SELF_DRIFT_WARNING_THRESHOLD = float(os.environ.get("SELF_DRIFT_WARNING_THRESHOLD", "0.10"))
SELF_DRIFT_ADAPTIVE = os.environ.get("SELF_DRIFT_ADAPTIVE", "true").lower() == "true"
SELF_DRIFT_ADAPT_K = float(os.environ.get("SELF_DRIFT_ADAPT_K", "2.0"))
SELF_DRIFT_ADAPT_MIN_SAMPLES = int(os.environ.get("SELF_DRIFT_ADAPT_MIN_SAMPLES", "5"))
SELF_DRIFT_ADAPT_WINDOW = int(os.environ.get("SELF_DRIFT_ADAPT_WINDOW", "20"))

class DriftMonitor:
    """Versioned ``z_self`` snapshots + drift/rollback orchestration."""

    def __init__(self, db_path: str = "data.db", self_model: Optional[SelfModel] = None):
        self.db_path = db_path
        self.self_model = self_model or SelfModel(db_path)
        self._ensure_version_table()

    def _ensure_version_table(self):
        """Create ``z_self_versions`` (and indexes) if missing."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS z_self_versions (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    z_self TEXT NOT NULL,
                    drift REAL DEFAULT 0,
                    tick INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    UNIQUE(session_id, version)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_z_self_versions_session 
                ON z_self_versions(session_id, version DESC)
            """)
            conn.commit()

    def check_drift(self, session_id: str, current_z_self: Optional[np.ndarray] = None) -> Dict:
        """
        Compare ``current_z_self`` against layered anchors and decide whether rollback is mandatory.

        Layering (2026-02-02 refactor):
        - L0 constitutional band (safety subspace): hard stop — violation forces rollback.
        - L1 identity band: slower evolution allowed — warnings only.
        - L2 state band: free-form dynamics — surfaced as activity, not a rollback trigger.

        Args:
            session_id: logical session key
            current_z_self: optional live vector; when ``None``, load from ``SelfModel``.

        Returns:
            Dict with drift scalars, relative drift, booleans, and ``should_rollback``.
        """
        if current_z_self is None:
            current_z_self = self.self_model.get_z_self(session_id)

        if current_z_self is None:
            return {
                "drift": 0.0,
                "drift_l0": 0.0,
                "drift_l1": 0.0,
                "drift_l2": 0.0,
                "drift_relative": 0.0,
                "drift_baseline": 0.0,
                "l0_violation": False,
                "l1_warning": False,
                "threshold_exceeded": False,
                "warning": False,
                "should_rollback": False
            }

        layered_drift = self.self_model.compute_layered_drift(session_id, current_z_self)

        drift_l0 = layered_drift["drift_l0"]
        drift_l1 = layered_drift["drift_l1"]
        drift_l2 = layered_drift["drift_l2"]
        drift_total = layered_drift["drift_total"]
        l0_violation = layered_drift["l0_violation"]
        l1_warning = layered_drift["l1_warning"]

        drift_baseline = self._get_drift_baseline(session_id)
        drift_relative = drift_total - drift_baseline

        effective_threshold = self._effective_threshold(session_id, base=SELF_DRIFT_THRESHOLD)

        should_rollback = l0_violation
        threshold_exceeded = l0_violation or (drift_relative > effective_threshold)
        warning = l1_warning or (drift_relative > effective_threshold * 0.66)

        if l0_violation:
            logger.warning(
                f"🚨 [L0 VIOLATION] Session {session_id}: "
                f"Constitutional drift={drift_l0:.4f} > 0.1. "
                f"Safety values compromised! Triggering rollback."
            )
        elif l1_warning:
            logger.info(
                f"🌱 [L1 Evolution] Session {session_id}: "
                f"Identity drift={drift_l1:.4f} > 0.3. "
                f"Persona is evolving (this is allowed)."
            )

        return {
            "drift": drift_total,
            "drift_l0": drift_l0,
            "drift_l1": drift_l1,
            "drift_l2": drift_l2,
            "drift_relative": drift_relative,
            "drift_baseline": drift_baseline,
            "l0_violation": l0_violation,
            "l1_warning": l1_warning,
            "threshold_exceeded": threshold_exceeded,
            "warning": warning,
            "should_rollback": should_rollback
        }

    def _get_drift_baseline(self, session_id: str) -> float:
        """
        Estimate a rolling baseline drift using the median of recent snapshots.

        When telemetry is sparse, fall back to conservative constants so first boots
        do not immediately fire false positives.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute(
                    """SELECT drift FROM z_self_versions 
                       WHERE session_id=? AND drift IS NOT NULL 
                       ORDER BY version DESC LIMIT ?""",
                    (session_id, SELF_DRIFT_ADAPT_WINDOW)
                )
                rows = cur.fetchall()

            drifts = [float(r[0]) for r in rows if r and r[0] is not None and r[0] > 0]

            if len(drifts) >= 3:
                return float(np.median(drifts))

            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute(
                    """SELECT detail FROM persona_events 
                       WHERE type='self_tick' AND persona_id=?
                       ORDER BY ts DESC LIMIT ?""",
                    (session_id, SELF_DRIFT_ADAPT_WINDOW)
                )
                rows = cur.fetchall()

            import json
            for row in rows:
                try:
                    detail = json.loads(row[0]) if row[0] else {}
                    d = detail.get("drift")
                    if d is not None and d > 0:
                        drifts.append(float(d))
                except:
                    pass

            if len(drifts) >= 3:
                return float(np.median(drifts))
            elif len(drifts) > 0:
                return float(np.mean(drifts)) * 0.9
            else:
                return 0.4

        except Exception as e:
            logger.debug(f"Failed to get drift baseline: {e}")
            return 0.4

    def _effective_threshold(self, session_id: str, base: float) -> float:
        """
        Optional adaptive ceiling: ``max(base, mean + k * std)`` over the drift window.
        """
        if not SELF_DRIFT_ADAPTIVE:
            return base
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute(
                    """SELECT drift FROM z_self_versions 
                       WHERE session_id=? AND drift IS NOT NULL 
                       ORDER BY version DESC LIMIT ?""",
                    (session_id, SELF_DRIFT_ADAPT_WINDOW)
                )
                rows = cur.fetchall()
            drifts = [float(r[0]) for r in rows if r and r[0] is not None]
            if len(drifts) < SELF_DRIFT_ADAPT_MIN_SAMPLES:
                return base
            arr = np.array(drifts, dtype=np.float32)
            mean = float(arr.mean())
            std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
            thr = max(base, mean + SELF_DRIFT_ADAPT_K * std)
            return thr
        except Exception as e:
            logger.debug(f"Adaptive threshold failed, fallback to base: {e}")
            return base

    def save_version(self, session_id: str, z_self: np.ndarray, drift: float, tick: int) -> str:
        """
        Persist a new immutable ``z_self`` snapshot row.

        Args:
            session_id: session key
            z_self: vector to serialize
            drift: scalar drift recorded alongside the snapshot
            tick: Self Tick counter at capture time

        Returns:
            UUID string primary key for the inserted row.
        """
        version_id = str(uuid.uuid4())

        drift_val = float(drift) if drift is not None else 0.0
        tick_val = int(tick) if tick is not None else 0

        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "SELECT MAX(version) FROM z_self_versions WHERE session_id=?",
                (session_id,)
            )
            row = cur.fetchone()
            next_version = (row[0] or 0) + 1

            z_json = json.dumps(z_self.tolist())
            created_at = datetime.now(timezone.utc).isoformat()

            conn.execute(
                """INSERT INTO z_self_versions (id, session_id, version, z_self, drift, tick, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (version_id, session_id, next_version, z_json, drift_val, tick_val, created_at)
            )
            conn.commit()

        logger.debug(f"Saved z_self version {next_version} for session {session_id}")
        return version_id

    def rollback(self, session_id: str, target_version: Optional[int] = None) -> Dict:
        """
        Restore ``z_self`` from ``z_self_versions`` (defaults to ``max(version)-1``).

        Args:
            session_id: session key
            target_version: explicit version id; ``None`` selects previous row

        Returns:
            ``{"success": bool, "rolled_back_to": int, ...}`` or ``{"success": False, "error": ...}``.
        """
        try:
            if target_version is None:
                with sqlite3.connect(self.db_path) as conn:
                    cur = conn.execute(
                        "SELECT MAX(version) FROM z_self_versions WHERE session_id=?",
                        (session_id,)
                    )
                    row = cur.fetchone()
                    if not row or row[0] is None or row[0] <= 1:
                        return {
                            "success": False,
                            "error": "No previous version to rollback to"
                        }
                    target_version = row[0] - 1

            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute(
                    "SELECT z_self, drift, tick FROM z_self_versions WHERE session_id=? AND version=?",
                    (session_id, target_version)
                )
                row = cur.fetchone()

                if not row:
                    return {
                        "success": False,
                        "error": f"Version {target_version} not found"
                    }

                z_self_data = json.loads(row[0])
                z_self = np.array(z_self_data, dtype=np.float32)
                drift = row[1]
                tick = row[2]

            self.self_model._save_z_self(session_id, z_self, tick=tick, drift=drift)

            self._record_rollback_event(session_id, target_version)

            logger.info(f"Rolled back session {session_id} to version {target_version}")

            return {
                "success": True,
                "rolled_back_to": target_version
            }
        except Exception as e:
            logger.error(f"Rollback failed for session {session_id}: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }

    def auto_rollback_if_needed(self, session_id: str) -> Optional[Dict]:
        """
        Run ``check_drift`` and immediately rollback when ``should_rollback`` is true.

        Returns:
            Rollback payload when a rollback executed, otherwise ``None``.
        """
        drift_info = self.check_drift(session_id)

        if drift_info["should_rollback"]:
            logger.warning(f"Auto-rollback triggered for session {session_id}: drift={drift_info['drift']:.4f}")
            return self.rollback(session_id)

        return None

    def get_versions(self, session_id: str, limit: int = 10) -> List[Dict]:
        """Return recent ``z_self_versions`` rows (metadata only — vectors omitted from payload)."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                """SELECT id, version, z_self, drift, tick, created_at 
                   FROM z_self_versions 
                   WHERE session_id=? 
                   ORDER BY version DESC 
                   LIMIT ?""",
                (session_id, limit)
            )
            rows = cur.fetchall()

        versions = []
        for row in rows:
            versions.append({
                "id": row[0],
                "version": row[1],
                "drift": row[3],
                "tick": row[4],
                "created_at": row[5]
            })

        return versions

    def _record_rollback_event(self, session_id: str, target_version: int):
        """Append a structured rollback marker to ``persona_events``."""
        from backend.self_tick import SelfTick  # Lazy import to reduce import cycles

        event_id = str(uuid.uuid4())
        ts = datetime.now(timezone.utc).isoformat()

        detail = {
            "session_id": session_id,
            "target_version": target_version,
            "type": "rollback"
        }

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO persona_events (id, ts, type, persona_id, detail) VALUES (?, ?, ?, ?, ?)",
                (event_id, ts, "rollback", session_id, json.dumps(detail))
            )
            conn.commit()
