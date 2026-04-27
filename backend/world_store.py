#!/usr/bin/env python3
"""
World-model store for durable beliefs.

- Persists text beliefs with embeddings and an 8-D worldview vector per row.
- Lightweight search, conflict arbitration, and cap enforcement.

z_self contract: this module is the source of truth for beliefs; ``aggregate_worldview_for_z_self``
writes a deterministic 24-D block into ``z_self[64:88]`` (not PCA). Drift on the rule side still uses
the first 32 dims elsewhere; pull full belief text via ``search_beliefs`` / prompt blocks when needed.
"""
import os
import sqlite3
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
import logging
from datetime import datetime, timezone
import math

try:
    from backend.embedder import get_embedder
except Exception:
    from backend.embedder_fallback import get_embedder_fallback as get_embedder

logger = logging.getLogger(__name__)

WORLDVIEW_DIM = 8
MAX_WORLDVIEW_BELIEFS = int(os.environ.get("MAX_WORLDVIEW_BELIEFS", "50"))
WORLDVIEW_SUBSPACE_DIMS = {
    "optimism": (0, 4),    # pessimism ← → optimism
    "agency": (4, 8),      # fatalism ← → agency
}

@dataclass
class BeliefItem:
    id: str
    text: str
    confidence: float  # 0.0 - 1.0
    embedding: Optional[np.ndarray]
    worldview_vector: np.ndarray  # 8-D packed vector
    created_at: str
    evidence_count: int = 0
    last_seen_at: str = ""
    locked: int = 0

class WorldStore:
    def __init__(self, db_path: str = "data.db"):
        self.db_path = db_path
        self.embedder = get_embedder()
        self.dim = WORLDVIEW_DIM
        self.DYNAMIC_CONFIDENCE_BASE_WEIGHT = 0.3
        self.DYNAMIC_CONFIDENCE_EVIDENCE_WEIGHT = 0.4
        self.DYNAMIC_CONFIDENCE_RECENCY_WEIGHT = 0.2
        self.DYNAMIC_CONFIDENCE_CONTEXT_WEIGHT = 0.1
        self.EVIDENCE_DECAY_HALF_LIFE_DAYS = 30.0
        self._ensure_tables()

    def neutral_worldview_vector(self) -> np.ndarray:
        """Neutral 8-D vector when empty or degenerate (avoids all-zero optimism/agency bias)."""
        return np.full(self.dim, 0.5, dtype=np.float32)

    def _pack_worldview_stats(self, n: int, mean_conf: float, locked_ratio: float) -> np.ndarray:
        """Bounded stats packed into ``z_self[80:88]``."""
        s = np.zeros(8, dtype=np.float32)
        s[0] = float(min(1.0, math.log1p(max(0, n)) / math.log1p(51))) if n > 0 else 0.0
        s[1] = float(max(0.0, min(1.0, mean_conf)))
        s[2] = float(max(0.0, min(1.0, locked_ratio)))
        cap = float(MAX_WORLDVIEW_BELIEFS) if MAX_WORLDVIEW_BELIEFS else 50.0
        s[3] = float(min(1.0, n / cap)) if cap > 0 else 0.0
        return s

    def aggregate_worldview_for_z_self(self) -> np.ndarray:
        """
        Deterministic 24-D aggregate for ``z_self[64:88]`` (no PCA).

        Layout:
        - ``[0:8]`` global confidence-weighted mean(worldview_vector)
        - ``[8:16]`` locked-row weighted mean, or copy of global if none locked
        - ``[16:24]`` summary stats from ``_pack_worldview_stats``
        """
        out = np.zeros(24, dtype=np.float32)
        neutral = self.neutral_worldview_vector()
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "SELECT worldview_vector, confidence, locked FROM world_beliefs"
            )
            rows = cur.fetchall()

        if not rows:
            out[0:8] = neutral
            out[8:16] = neutral
            out[16:24] = self._pack_worldview_stats(0, 0.0, 0.0)
            return out

        vecs: List[np.ndarray] = []
        weights: List[float] = []
        locked_vecs: List[np.ndarray] = []
        locked_weights: List[float] = []
        confidences: List[float] = []
        n_locked = 0

        for r in rows:
            v = np.frombuffer(r[0], dtype=np.float32)
            if v.shape[0] != self.dim:
                continue
            w = float(r[1]) if r[1] and float(r[1]) > 0 else 0.1
            vecs.append(v)
            weights.append(w)
            confidences.append(float(r[1]) if r[1] is not None else 0.0)
            if r[2]:
                n_locked += 1
                locked_vecs.append(v)
                locked_weights.append(w)

        if not vecs:
            out[0:8] = neutral
            out[8:16] = neutral
            out[16:24] = self._pack_worldview_stats(0, 0.0, 0.0)
            return out

        tw = sum(weights)
        gsum = np.zeros(self.dim, dtype=np.float32)
        for v, w in zip(vecs, weights):
            gsum += v * w
        global_vec = gsum / tw if tw > 0 else np.mean(np.stack(vecs, axis=0), axis=0)
        global_vec = np.clip(global_vec, -1.0, 1.0)
        out[0:8] = global_vec

        if locked_vecs:
            lw = sum(locked_weights)
            lsum = np.zeros(self.dim, dtype=np.float32)
            for v, w in zip(locked_vecs, locked_weights):
                lsum += v * w
            locked_avg = lsum / lw if lw > 0 else np.mean(np.stack(locked_vecs, axis=0), axis=0)
            out[8:16] = np.clip(locked_avg, -1.0, 1.0)
        else:
            out[8:16] = out[0:8]

        mean_conf = float(sum(confidences) / len(confidences)) if confidences else 0.0
        locked_ratio = n_locked / len(rows)
        out[16:24] = self._pack_worldview_stats(len(rows), mean_conf, locked_ratio)
        return out
        
    def _ensure_tables(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS world_beliefs (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    confidence REAL DEFAULT 0.5,
                    embedding BLOB,
                    worldview_vector BLOB,
                    created_at TEXT,
                    evidence_count INTEGER DEFAULT 0,
                    last_seen_at TEXT,
                    locked INTEGER DEFAULT 0
                )
            """)
            for column in ["evidence_count INTEGER DEFAULT 0", "last_seen_at TEXT", "locked INTEGER DEFAULT 0"]:
                try:
                    conn.execute(f"ALTER TABLE world_beliefs ADD COLUMN {column}")
                except sqlite3.OperationalError:
                    pass
            conn.commit()
            
    def add_belief(self, text: str, confidence: float, optimism: float, agency: float):
        """
        Insert or reconcile a belief (semantic near-duplicate detection + arbitration).
        """
        existing_beliefs = self.search_beliefs(text, top_k=1)
        if existing_beliefs:
            top_belief = existing_beliefs[0]
            new_emb = self.embedder.encode(text)
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute("SELECT embedding FROM world_beliefs WHERE id = ?", (top_belief.id,))
                row = cur.fetchone()
                if row and row[0]:
                    old_emb = np.frombuffer(row[0], dtype=np.float32)
                    sim = np.dot(new_emb, old_emb) / (
                        np.linalg.norm(new_emb) * np.linalg.norm(old_emb) + 1e-8
                    )

                    if sim > 0.85:
                        logger.info(
                            f"Updating existing belief '{top_belief.text[:20]}...' (sim={sim:.2f})"
                        )
                        self._update_existing_belief(top_belief.id, confidence)
                        return True

                    if sim > 0.6:
                        logger.info(
                            f"Conflict detected between '{text[:20]}' and '{top_belief.text[:20]}' "
                            f"(sim={sim:.2f}). Triggering arbitration."
                        )
                        return self._arbitrate_conflict(
                            text, confidence, optimism, agency, top_belief
                        )

        return self._force_add_belief(text, confidence, optimism, agency)

    def _update_existing_belief(self, belief_id: str, new_confidence: float):
        """Bump confidence / evidence on a near-duplicate hit."""
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE world_beliefs 
                SET confidence = (confidence + ?) / 2, 
                    last_seen_at = ?,
                    evidence_count = evidence_count + 1
                WHERE id = ?
            """, (new_confidence, now, belief_id))
            conn.commit()

    def _arbitrate_conflict(self, new_text: str, new_conf: float, opt: float, agn: float, old_belief: BeliefItem) -> bool:
        """
        Simple strength-based policy: overwrite, reject, or co-exist.
        """
        if old_belief.locked:
            logger.info("Arbitration: Old belief is locked. Rejecting new belief.")
            return False
            
        old_strength = old_belief.confidence * (1.0 + math.log(1 + old_belief.evidence_count))
        new_strength = new_conf

        if new_strength > old_strength * 1.2:
            logger.info("Arbitration: New belief is stronger. Overwriting.")
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM world_beliefs WHERE id = ?", (old_belief.id,))
                conn.commit()
            return self._force_add_belief(new_text, new_conf, opt, agn)
            
        elif new_strength < old_strength * 0.5:
            logger.info("Arbitration: New belief is too weak. Rejected.")
            return False

        else:
            logger.info("Arbitration: Strength similar. Adding as co-existing belief.")
            return self._force_add_belief(new_text, new_conf, opt, agn)

    def _force_add_belief(self, text: str, confidence: float, optimism: float, agency: float):
        """Insert after optional eviction when over cap."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("SELECT COUNT(*) FROM world_beliefs")
            current_count = cur.fetchone()[0]
            if current_count >= MAX_WORLDVIEW_BELIEFS:
                logger.info(f"Worldview belief limit reached. Removing oldest non-locked belief.")
                conn.execute("""
                    DELETE FROM world_beliefs 
                    WHERE id IN (
                        SELECT id FROM world_beliefs 
                        WHERE locked = 0 
                        ORDER BY created_at ASC 
                        LIMIT 1
                    )
                """)
        
        embedding = self.embedder.encode(text)
        
        vec = np.zeros(self.dim, dtype=np.float32)
        vec[0:4] = optimism
        vec[4:8] = agency
        
        belief_id = f"belief-{abs(hash(text))}"
        now = datetime.now(timezone.utc).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO world_beliefs 
                (id, text, confidence, embedding, worldview_vector, created_at, evidence_count, last_seen_at, locked)
                VALUES (?, ?, ?, ?, ?, ?, 1, ?, 0)
            """, (belief_id, text, confidence, embedding.astype(np.float32).tobytes(), 
                  vec.tobytes(), now, now))
            conn.commit()
        return True

            
    def _calculate_dynamic_confidence(
        self,
        base_confidence: float,
        evidence_count: int,
        last_seen_at: Optional[str],
        created_at: Optional[str],
        belief_id: str = ""
    ) -> float:
        """
        Evidence + recency weighted score in ``[0, 1]``.

        ``dynamic_confidence = base*0.3 + evidence*0.4 + recency*0.2 + context*0.1`` (see field weights on ``self``).
        Locked rows get a floor bump when ``belief_id`` resolves to ``locked=1``.
        """
        now = datetime.now(timezone.utc)
        
        if evidence_count > 0:
            max_evidence = 100.0
            evidence_strength = min(1.0, math.log(1 + evidence_count) / math.log(1 + max_evidence))
        else:
            evidence_strength = 0.0
        
        if last_seen_at:
            try:
                last_seen = datetime.fromisoformat(last_seen_at.replace('Z', '+00:00'))
                if last_seen.tzinfo is None:
                    last_seen = last_seen.replace(tzinfo=timezone.utc)
                days_since_last_use = (now - last_seen).total_seconds() / 86400.0
                recency_factor = math.exp(-days_since_last_use / self.EVIDENCE_DECAY_HALF_LIFE_DAYS)
            except (ValueError, AttributeError) as e:
                logger.debug(f"Failed to parse last_seen_at '{last_seen_at}': {e}")
                recency_factor = 0.5
            except Exception as e:
                logger.warning(f"Unexpected error parsing last_seen_at '{last_seen_at}': {e}")
                recency_factor = 0.5
        else:
            recency_factor = 0.3
        
        context_relevance = 0.5

        if belief_id:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cur = conn.execute(
                        "SELECT locked FROM world_beliefs WHERE id = ?",
                        (belief_id,)
                    )
                    row = cur.fetchone()
                    if row and row[0] == 1:
                        return min(1.0, base_confidence + 0.2)
            except sqlite3.Error as e:
                logger.debug(f"Database error checking locked belief: {e}")
            except Exception as e:
                logger.warning(f"Unexpected error checking locked belief: {e}")
        
        dynamic_confidence = (
            base_confidence * self.DYNAMIC_CONFIDENCE_BASE_WEIGHT +
            evidence_strength * self.DYNAMIC_CONFIDENCE_EVIDENCE_WEIGHT +
            recency_factor * self.DYNAMIC_CONFIDENCE_RECENCY_WEIGHT +
            context_relevance * self.DYNAMIC_CONFIDENCE_CONTEXT_WEIGHT
        )
        dynamic_confidence = max(0.0, min(1.0, dynamic_confidence))
        
        return dynamic_confidence
            
    def search_beliefs(self, query: str, top_k: int = 3) -> List[BeliefItem]:
        """Semantic top-k with similarity + dynamic confidence blend."""
        query_emb = self.embedder.encode(query)
        
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("SELECT id, text, confidence, embedding, worldview_vector, created_at, evidence_count, last_seen_at, locked FROM world_beliefs")
            rows = cur.fetchall()
            
        if not rows:
            return []
            
        results = []
        for row in rows:
            if not row[3]: continue
            emb = np.frombuffer(row[3], dtype=np.float32)
            sim = np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb) + 1e-8)

            dynamic_confidence = self._calculate_dynamic_confidence(
                base_confidence=row[2],
                evidence_count=row[6] or 0,
                last_seen_at=row[7],
                created_at=row[5],
                belief_id=row[0]
            )
            
            combined_score = sim * 0.6 + dynamic_confidence * 0.4
            results.append((combined_score, sim, row))
            
        results.sort(key=lambda x: x[0], reverse=True)
        
        final_items = []
        for _, _, row in results[:top_k]:
            final_items.append(BeliefItem(
                id=row[0], text=row[1], confidence=row[2],
                embedding=None,
                worldview_vector=np.frombuffer(row[4], dtype=np.float32),
                created_at=row[5],
                evidence_count=row[6] or 0,
                last_seen_at=row[7] or "",
                locked=row[8] or 0
            ))
        return final_items
    
    def get_dominant_worldview(self) -> np.ndarray:
        """Confidence-weighted mean worldview vector; neutral when empty."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("SELECT worldview_vector, confidence FROM world_beliefs")
            rows = cur.fetchall()

        if not rows:
            return self.neutral_worldview_vector()

        vecs = []
        weights = []
        for r in rows:
            vecs.append(np.frombuffer(r[0], dtype=np.float32))
            weights.append(r[1] if r[1] > 0 else 0.1)

        if not vecs:
            return self.neutral_worldview_vector()

        weighted_sum = np.zeros(self.dim, dtype=np.float32)
        total_weight = sum(weights)

        for v, w in zip(vecs, weights):
            weighted_sum += v * w

        result = weighted_sum / total_weight if total_weight > 0 else np.mean(vecs, axis=0)
        return np.asarray(result, dtype=np.float32)

    def get_all_beliefs(self, status: str = "active", limit: int = 100) -> List[BeliefItem]:
        """
        All rows sorted by dynamic confidence, then locked-first stable ordering.
        """
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("""
                SELECT id, text, confidence, embedding, worldview_vector, created_at, evidence_count, last_seen_at, locked 
                FROM world_beliefs 
                ORDER BY created_at ASC
            """)
            rows = cur.fetchall()
        
        beliefs_with_dynamic_confidence = []
        for r in rows:
            embedding = None
            if r[3]:
                embedding = np.frombuffer(r[3], dtype=np.float32)

            dynamic_confidence = self._calculate_dynamic_confidence(
                base_confidence=r[2],
                evidence_count=r[6] or 0,
                last_seen_at=r[7],
                created_at=r[5],
                belief_id=r[0]
            )
            
            belief = BeliefItem(
                id=r[0], text=r[1], confidence=r[2],
                embedding=embedding,
                worldview_vector=np.frombuffer(r[4], dtype=np.float32),
                created_at=r[5],
                evidence_count=r[6] or 0,
                last_seen_at=r[7] or "",
                locked=r[8] or 0
            )
            beliefs_with_dynamic_confidence.append((belief, dynamic_confidence))
        
        beliefs_with_dynamic_confidence.sort(key=lambda x: x[1], reverse=True)

        beliefs = [belief for belief, _ in beliefs_with_dynamic_confidence[:limit]]

        locked_beliefs = [b for b in beliefs if b.locked == 1]
        unlocked_beliefs = [b for b in beliefs if b.locked == 0]
        
        return locked_beliefs + unlocked_beliefs
    
    def evolve_beliefs(self) -> Dict:
        """Prune unlocked, low-confidence beliefs older than 30 days."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute("SELECT id, confidence, created_at, locked FROM world_beliefs")
                rows = cur.fetchall()
                
                removed_count = 0
                now = datetime.now(timezone.utc)
                
                for row in rows:
                    if row[3]:
                        continue
                    
                    confidence = row[1]
                    created_at_str = row[2]
                    
                    try:
                        created_at = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
                        days_since = (now - created_at).days
                        
                        if confidence < 0.3 and days_since > 30:
                            conn.execute("DELETE FROM world_beliefs WHERE id = ?", (row[0],))
                            removed_count += 1
                    except Exception:
                        pass
                        
                conn.commit()
                return {
                    "evolved": 0,
                    "removed": removed_count,
                    "summary": f"Removed {removed_count} stale low-confidence belief(s).",
                }
                
        except Exception as e:
            logger.error(f"Failed to evolve beliefs: {e}")
            return {"error": str(e)}
