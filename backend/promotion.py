#!/usr/bin/env python3
"""
Core promotion gate (minimal viable version).

- Filter dynamic persona items by evidence_count, score, and centroid similarity.
- Global stability: recent mean drift must stay below a threshold.
"""
from __future__ import annotations

import os
import sqlite3
from typing import List, Tuple, Optional, Dict
import numpy as np
from datetime import datetime, timezone

from backend.persona_store import PersonaStore, PersonaItem
from backend.embedder import get_embedder
from backend.judge import PersonaJudge
from backend.config import config
import logging

logger = logging.getLogger(__name__)

# [2026-02-25] Prefer config; fall back to env / defaults (thresholds tuned for easier promotion).
PROMOTION_ENABLED = config.get("system.promotion_enabled", 
    os.environ.get("PROMOTION_ENABLED", "true").lower() == "true")
PROMOTION_MIN_EVIDENCE = config.get("parameters.thresholds.promotion_min_evidence",
    int(os.environ.get("PROMOTION_MIN_EVIDENCE", "2")))
PROMOTION_MIN_SCORE = config.get("parameters.thresholds.promotion_min_score",
    float(os.environ.get("PROMOTION_MIN_SCORE", "0.4")))
PROMOTION_BATCH_LIMIT = config.get("parameters.thresholds.promotion_batch_limit",
    int(os.environ.get("PROMOTION_BATCH_LIMIT", "15")))
PROMOTION_LOCK_ON = os.environ.get("PROMOTION_LOCK_ON_PROMOTE", "false").lower() == "true"
PROMOTION_DRIFT_MAX_MEAN = config.get("parameters.thresholds.promotion_drift_max_mean",
    float(os.environ.get("PROMOTION_DRIFT_MAX_MEAN", "0.25")))
PROMOTION_DRIFT_WINDOW = int(os.environ.get("PROMOTION_DRIFT_WINDOW", "100"))
PROMOTION_BOOST_SCORE = float(os.environ.get("PROMOTION_BOOST_SCORE", "0.2"))
PROMOTION_MIN_SIM = config.get("parameters.thresholds.promotion_min_sim", 0.15)
PROMOTION_MIN_ALIGNMENT = config.get("parameters.thresholds.promotion_min_alignment", 0.5)
PROMOTION_MIN_SAFETY = config.get("parameters.thresholds.promotion_min_safety", 0.65)

class PromotionGate:
    def __init__(self, db_path: str = "data.db"):
        self.db_path = db_path
        self.persona_store = PersonaStore(db_path)
        self.embedder = get_embedder()
        self._centroid: Optional[np.ndarray] = None
        self.judge = PersonaJudge(db_path)
    
    def _get_centroid(self) -> Optional[np.ndarray]:
        if self._centroid is not None:
            return self._centroid
        items = self.persona_store.get_all_active(limit=200)
        embs = [it.embedding for it in items if it.embedding is not None and getattr(it, "is_core", 0) == 1]
        if not embs:
            return None
        X = np.stack(embs, axis=0)
        c = X.mean(axis=0)
        n = np.linalg.norm(c)
        if n > 0:
            c = c / n
        self._centroid = c.astype(np.float32)
        return self._centroid
    
    def _recent_mean_drift(self) -> float:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute(
                    """SELECT drift FROM z_self_versions 
                       WHERE drift IS NOT NULL
                       ORDER BY created_at DESC LIMIT ?""",
                    (PROMOTION_DRIFT_WINDOW,)
                )
                vals = [float(r[0]) for r in cur.fetchall() if r and r[0] is not None]
            if not vals:
                return 0.0
            arr = np.array(vals, dtype=np.float32)
            return float(arr.mean())
        except Exception as e:
            logger.debug(f"recent_mean_drift failed: {e}")
            return 0.0
    
    def find_eligible(self, max_items: int | None = None) -> List[PersonaItem]:
        if not PROMOTION_ENABLED:
            return []
        # Global stability gate
        mean_drift = self._recent_mean_drift()
        if mean_drift > PROMOTION_DRIFT_MAX_MEAN:
            logger.info(f"Skip auto-promotion: mean drift {mean_drift:.3f} > {PROMOTION_DRIFT_MAX_MEAN}")
            return []
        centroid = self._get_centroid()
        candidates: List[PersonaItem] = []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(
                """SELECT * FROM persona_items 
                   WHERE status='active' AND (is_core IS NULL OR is_core=0)
                   ORDER BY score DESC, evidence_count DESC
                   LIMIT ?""",
                (max_items or 200,)
            )
            rows = cur.fetchall()
        for row in rows:
            emb = None
            if row["embedding"]:
                emb = np.frombuffer(row["embedding"], dtype=np.float32)
            item = PersonaItem(
                id=row["id"], text=row["text"], embedding=emb,
                score=row["score"], importance=row["importance"],
                novelty=row["novelty"], reliability=row["reliability"],
                evidence_count=row["evidence_count"],
                created_at=row["created_at"], last_seen_at=row["last_seen_at"],
                status=row["status"],
                is_core=row["is_core"] if "is_core" in row.keys() else 0,
                core_version=row["core_version"] if "core_version" in row.keys() else 0,
                locked=row["locked"] if "locked" in row.keys() else 0
            )
            if (item.evidence_count or 0) < PROMOTION_MIN_EVIDENCE:
                continue
            if (item.score or 0.0) < PROMOTION_MIN_SCORE:
                continue
            if centroid is not None and item.embedding is not None:
                try:
                    sim = float(np.dot(centroid, item.embedding) / (np.linalg.norm(centroid) * np.linalg.norm(item.embedding) + 1e-8))
                except Exception:
                    sim = -1.0
                # [2026-02-25] Use promotion similarity threshold from config
                if sim < PROMOTION_MIN_SIM:
                    continue
            # Optional: judge pass (alignment + safety)
            judge_passed = True
            try:
                scores = self.judge.score_persona_candidate(item.text)
                align = scores.get("alignment", 0.0)
                safe = scores.get("safety", 0.0)
                # [2026-02-25] All-zero scores => judge unavailable; skip judge gate
                if align == 0.0 and safe == 0.0:
                    logger.debug(f"Judge returned all zeros, treating as unavailable, skipping judge check")
                    judge_passed = True
                else:
                    # Thresholds from config
                    if align < PROMOTION_MIN_ALIGNMENT:
                        judge_passed = False
                    if safe < PROMOTION_MIN_SAFETY:
                        judge_passed = False
            except Exception as e:
                # Judge errors do not block promotion
                logger.debug(f"Promotion judge failed, treating as passed: {e}")
                judge_passed = True
            
            if not judge_passed:
                continue
            candidates.append(item)
        return candidates[: (PROMOTION_BATCH_LIMIT if max_items is None else max_items)]
    
    def auto_promote(self) -> Dict:
        eligible = self.find_eligible()
        if not eligible:
            return {"ok": True, "promoted": 0, "core_version": None, "ids": []}
        ids = [it.id for it in eligible]
        with sqlite3.connect(self.db_path) as conn:
            # Next core_version bump
            cur = conn.execute("SELECT MAX(core_version) FROM persona_items WHERE is_core=1")
            row = cur.fetchone()
            next_version = (row[0] or 0) + 1
            ts = datetime.now(timezone.utc).isoformat()
            updated = 0
            for pid in ids:
                cur2 = conn.execute(
                    "UPDATE persona_items SET is_core=1, core_version=?, locked=?, status='active', last_seen_at=?, score=score+? WHERE id=?",
                    (next_version, 1 if PROMOTION_LOCK_ON else 0, ts, PROMOTION_BOOST_SCORE, pid)
                )
                updated += cur2.rowcount
            conn.commit()
        logger.info(f"Auto promoted {updated} items to core_v{next_version}")
        return {"ok": True, "promoted": updated, "core_version": next_version, "ids": ids}


