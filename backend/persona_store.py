#!/usr/bin/env python3
"""
Persona memory persistence, FAISS-backed retrieval, and layered (L0/L1/L2) update rules.
"""
import os
import sqlite3
import json
import numpy as np
import faiss
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime, timezone, timedelta
import math

from backend.config import config

try:
    from backend.embedder import get_embedder
except Exception as e:
    import logging
    logging.warning(f"Failed to import real embedder: {e}, using fallback")
    from backend.embedder_fallback import get_embedder_fallback as get_embedder

logger = logging.getLogger(__name__)


def _persona_keyword_in_text(haystack: str, needle: str) -> bool:
    """Substring match for keyword lists; ASCII needles are matched case-insensitively."""
    if not haystack or not needle:
        return False
    if needle in haystack:
        return True
    if needle.isascii():
        return needle.lower() in haystack.lower()
    return False


@dataclass
class PersonaItem:
    id: str
    text: str
    embedding: Optional[np.ndarray] = None
    score: float = 0.5
    importance: float = 0.5
    novelty: float = 0.5
    reliability: float = 0.5
    evidence_count: int = 0
    created_at: str = ""
    last_seen_at: str = ""
    status: str = "active"  # active | archived
    is_core: int = 0        # 1 = L1 core slice, 0 = L2 reflective rules
    core_version: int = 0   # monotonic bundle id for core sets
    locked: int = 0         # 1 = L0 constitution row (immutable text via add_or_update)
    source: Optional[Dict] = None  # optional provenance JSON

class PersonaStore:
    """SQLite + FAISS persona line store with L0/L1/L2 safeguards."""

    def __init__(self, db_path: str = "data.db"):
        self.db_path = db_path
        self.embedder = get_embedder()
        self.index = None
        self.id_map = {}  # faiss row index -> persona_items.id

        self._emotion_prototypes = None

        self._init_table()

        self.use_faiss = True
        try:
            self._load_faiss_index()
        except Exception as e:
            logger.warning(f"Failed to init FAISS index: {e}, fallback to SQLite search")
            self.use_faiss = False

    def _init_table(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS persona_items (
                        id TEXT PRIMARY KEY,
                        text TEXT NOT NULL,
                        embedding BLOB,
                        score REAL,
                        importance REAL,
                        novelty REAL,
                        reliability REAL,
                        evidence_count INTEGER,
                        created_at TEXT,
                        last_seen_at TEXT,
                        status TEXT
                    )
                """)
                
                cur = conn.execute("PRAGMA table_info(persona_items)")
                cols = {col[1] for col in cur.fetchall()}
                
                if "is_core" not in cols:
                    conn.execute("ALTER TABLE persona_items ADD COLUMN is_core INTEGER DEFAULT 0")
                if "core_version" not in cols:
                    conn.execute("ALTER TABLE persona_items ADD COLUMN core_version INTEGER DEFAULT 0")
                if "locked" not in cols:
                    conn.execute("ALTER TABLE persona_items ADD COLUMN locked INTEGER DEFAULT 0")
                if "source" not in cols:
                    conn.execute("ALTER TABLE persona_items ADD COLUMN source TEXT")
                
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to init persona table: {e}")

    def _load_faiss_index(self):
        """Hydrate ``IndexFlatIP`` from ``persona_items.embedding`` blobs."""
        self._faiss_dirty = False

        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("SELECT id, embedding FROM persona_items WHERE status='active' AND embedding IS NOT NULL")
            rows = cur.fetchall()
        
        embed_dim = 384
        try:
            embed_dim = self.embedder.dimension
        except Exception:
            pass
        
        if not rows:
            self.index = faiss.IndexFlatIP(embed_dim)
            self._embed_dim = embed_dim
            return
        
        for _, emb_blob in rows:
            if emb_blob:
                first_vec = np.frombuffer(emb_blob, dtype=np.float32)
                embed_dim = first_vec.shape[0]
                break
        
        vectors = []
        self.id_map = {}
        valid_rows = 0
        
        for i, (pid, emb_blob) in enumerate(rows):
            if emb_blob:
                vec = np.frombuffer(emb_blob, dtype=np.float32)
                if vec.shape[0] == embed_dim:
                    vectors.append(vec)
                    self.id_map[valid_rows] = pid
                    valid_rows += 1
        
        if vectors:
            vectors_np = np.array(vectors)
            faiss.normalize_L2(vectors_np)
            
            self.index = faiss.IndexFlatIP(embed_dim)
            self.index.add(vectors_np)
            logger.info(f"Built FAISS index with {valid_rows} items (dim={embed_dim})")
        else:
            self.index = faiss.IndexFlatIP(embed_dim)
        self._embed_dim = embed_dim

    def _faiss_add_single(self, item_id: str, embedding: "np.ndarray"):
        """Append one normalized vector without rebuilding the whole index."""
        if not self.use_faiss or embedding is None:
            return
        try:
            vec = embedding.astype(np.float32).reshape(1, -1)
            expected_dim = getattr(self, "_embed_dim", None) or self.index.d
            if vec.shape[1] != expected_dim:
                return
            faiss.normalize_L2(vec)
            new_idx = self.index.ntotal
            self.index.add(vec)
            self.id_map[new_idx] = item_id
        except Exception as e:
            logger.debug(f"FAISS incremental add failed, marking dirty: {e}")
            self._faiss_dirty = True

    def _ensure_faiss_fresh(self):
        """Rebuild FAISS when ``_faiss_dirty`` is set (updates/deletes/archivals)."""
        if getattr(self, "_faiss_dirty", False) and self.use_faiss:
            self._load_faiss_index()
    
    def add_or_update(self, item: PersonaItem, update_embedding: bool = False):
        """
        Insert or upsert a persona row with layered safeguards.

        v2.0:
        1. L0 constitution: if ``locked=1``, reject text changes; stats-only updates allowed.
        2. L1 core damping: if ``is_core=1``, new text requires score > old * 1.2.
        3. Core version bumps are left to callers; this path still overwrites text when allowed.
        """
        existing_item = self.get_by_id(item.id)
        
        allow_text_update = True
        
        if existing_item:
            if existing_item.locked == 1:
                if existing_item.text != item.text:
                    logger.warning(
                        f"[L0] Blocked text change on locked rule '{existing_item.id}'."
                    )
                    allow_text_update = False
                    # Roll back text/embedding; score and evidence may still refresh.
                    item.text = existing_item.text
                    item.embedding = existing_item.embedding
                    item.locked = 1
            
            elif existing_item.is_core == 1 and existing_item.text != item.text:
                damping_factor = 1.2
                if item.score < existing_item.score * damping_factor:
                    logger.info(
                        f"[L1] Damping: score gain too small ({item.score:.3f} vs {existing_item.score:.3f}); "
                        "text update rejected."
                    )
                    allow_text_update = False
                    item.text = existing_item.text
                    item.embedding = existing_item.embedding
                else:
                    logger.info(
                        f"[L1] Core rule '{item.id}' text updated ({existing_item.score:.3f} -> {item.score:.3f})."
                    )

        if update_embedding and allow_text_update and item.embedding is None:
            try:
                item.embedding = self.embedder.encode(item.text)
            except Exception as e:
                logger.error(f"Failed to encode text: {e}")
        
        with sqlite3.connect(self.db_path) as conn:
            emb_blob = item.embedding.astype(np.float32).tobytes() if item.embedding is not None else None
            source_json = json.dumps(item.source, ensure_ascii=False) if item.source else None
            
            cur = conn.execute("PRAGMA table_info(persona_items)")
            cols_info = cur.fetchall()
            cols = {col[1] for col in cols_info}
            has_embedding = "embedding" in cols
            has_layer_cols = all(name in cols for name in ("is_core","core_version","locked"))
            has_source = "source" in cols
            
            if has_embedding:
                if has_layer_cols:
                    if has_source:
                        conn.execute(
                            """INSERT INTO persona_items 
                               (id, text, embedding, score, importance, novelty, reliability, evidence_count, created_at, last_seen_at, status, is_core, core_version, locked, source)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                               ON CONFLICT(id) DO UPDATE SET
                                 text=excluded.text,
                                 embedding=excluded.embedding,
                                 score=excluded.score,
                                 importance=excluded.importance,
                                 novelty=excluded.novelty,
                                 reliability=excluded.reliability,
                                 evidence_count=excluded.evidence_count,
                                 last_seen_at=excluded.last_seen_at,
                                 status=excluded.status,
                                 is_core=excluded.is_core,
                                 core_version=excluded.core_version,
                                 locked=excluded.locked,
                                 source=excluded.source
                            """,
                            (
                                item.id, item.text, emb_blob,
                                item.score, item.importance, item.novelty, item.reliability,
                                item.evidence_count, item.created_at, item.last_seen_at, item.status,
                                item.is_core, item.core_version, item.locked, source_json
                            )
                        )
                    else:
                        conn.execute(
                            """INSERT INTO persona_items 
                               (id, text, embedding, score, importance, novelty, reliability, evidence_count, created_at, last_seen_at, status, is_core, core_version, locked)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                               ON CONFLICT(id) DO UPDATE SET
                                 text=excluded.text,
                                 embedding=excluded.embedding,
                                 score=excluded.score,
                                 importance=excluded.importance,
                                 novelty=excluded.novelty,
                                 reliability=excluded.reliability,
                                 evidence_count=excluded.evidence_count,
                                 last_seen_at=excluded.last_seen_at,
                                 status=excluded.status,
                                 is_core=excluded.is_core,
                                 core_version=excluded.core_version,
                                 locked=excluded.locked
                            """,
                            (
                                item.id, item.text, emb_blob,
                                item.score, item.importance, item.novelty, item.reliability,
                                item.evidence_count, item.created_at, item.last_seen_at, item.status,
                                item.is_core, item.core_version, item.locked
                            )
                        )
                else:
                    conn.execute(
                        """INSERT INTO persona_items 
                           (id, text, embedding, score, importance, novelty, reliability, evidence_count, created_at, last_seen_at, status)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                           ON CONFLICT(id) DO UPDATE SET
                             text=excluded.text,
                             embedding=excluded.embedding,
                             score=excluded.score,
                             importance=excluded.importance,
                             novelty=excluded.novelty,
                             reliability=excluded.reliability,
                             evidence_count=excluded.evidence_count,
                             last_seen_at=excluded.last_seen_at,
                             status=excluded.status
                        """,
                        (
                            item.id, item.text, emb_blob,
                            item.score, item.importance, item.novelty, item.reliability,
                            item.evidence_count, item.created_at, item.last_seen_at, item.status
                        )
                    )
            else:
                conn.execute(
                    """INSERT INTO persona_items 
                       (id, text, score, importance, novelty, reliability, evidence_count, created_at, last_seen_at, status)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(id) DO UPDATE SET
                         text=excluded.text,
                         score=excluded.score,
                         importance=excluded.importance,
                         novelty=excluded.novelty,
                         reliability=excluded.reliability,
                         evidence_count=excluded.evidence_count,
                         last_seen_at=excluded.last_seen_at,
                         status=excluded.status
                    """,
                    (
                        item.id, item.text,
                        item.score, item.importance, item.novelty, item.reliability,
                        item.evidence_count, item.created_at, item.last_seen_at, item.status
                    )
                )
            conn.commit()
        
        if self.use_faiss:
            if not existing_item and item.embedding is not None:
                self._faiss_add_single(item.id, item.embedding)
            elif existing_item:
                self._faiss_dirty = True
    
    def decay_memory(self, threshold_days: int = 60, decay_rate: float = 0.97) -> Dict:
        """
        Age stale L2-style rows by lowering ``score`` (v2).

        Decay is not deletion: embeddings stay intact; only ranking shifts.
        Immune: ``locked``, ``is_core``, and any row whose text hits ``system.identity_anchors``.
        Higher ``evidence_count`` slows decay.
        Rows that fall below score 0.05 are marked ``archived``; hard deletes are left to MemoryCleaner.

        Env overrides: ``MEMORY_DECAY_ENABLED``, ``MEMORY_DECAY_THRESHOLD_DAYS``, ``MEMORY_DECAY_RATE``.

        Returns:
            {"processed": int, "decayed": int, "archived": int, "immune": int}
        """
        enabled = os.environ.get("MEMORY_DECAY_ENABLED", "true").lower() == "true"
        if not enabled:
            logger.debug("[DECAY] Memory decay disabled via MEMORY_DECAY_ENABLED=false")
            return {"processed": 0, "decayed": 0, "archived": 0, "immune": 0}

        threshold_days = int(os.environ.get("MEMORY_DECAY_THRESHOLD_DAYS", str(threshold_days)))
        decay_rate = float(os.environ.get("MEMORY_DECAY_RATE", str(decay_rate)))

        cutoff_date = (datetime.now(timezone.utc) - timedelta(days=threshold_days)).isoformat()

        stats = {"processed": 0, "decayed": 0, "archived": 0, "immune": 0}

        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                """SELECT id, score, importance, evidence_count, text, is_core, locked
                   FROM persona_items
                   WHERE last_seen_at < ? AND status = 'active'""",
                (cutoff_date,)
            )
            rows = cur.fetchall()
            stats["processed"] = len(rows)

            updates = []
            for pid, score, importance, evidence_count, text, is_core, locked in rows:
                if locked == 1 or is_core == 1:
                    stats["immune"] += 1
                    continue
                anchors = config.get("system.identity_anchors", []) or []
                if anchors and any(kw in (text or "") for kw in anchors):
                    stats["immune"] += 1
                    continue

                # Effective decay_rate: higher importance slows decay; evidence nudges rate toward 1.0.
                imp = importance if importance is not None else 0.3
                ev = evidence_count if evidence_count is not None else 0
                evidence_bonus = min(0.02, ev * 0.002)
                real_rate = decay_rate + (1.0 - decay_rate) * imp + evidence_bonus
                real_rate = min(real_rate, 1.0)

                new_score = score * real_rate

                new_status = "active"
                if new_score < 0.05:
                    new_status = "archived"
                    stats["archived"] += 1

                if new_score != score:
                    stats["decayed"] += 1

                updates.append((new_score, new_status, pid))

            if updates:
                conn.executemany(
                    "UPDATE persona_items SET score = ?, status = ? WHERE id = ?",
                    updates
                )
                conn.commit()

        if stats["decayed"] > 0 or stats["archived"] > 0:
            logger.info(
                f"[DECAY] processed={stats['processed']}, decayed={stats['decayed']}, "
                f"archived={stats['archived']}, immune={stats['immune']} "
                f"(threshold={threshold_days}d, rate={decay_rate})"
            )
            if self.use_faiss and stats["archived"] > 0:
                self._faiss_dirty = True

        return stats
    
    def _get_emotion_prototypes(self):
        """Lazy-built embedder prototypes for aggressive vs calm tone."""
        if self._emotion_prototypes is None:
            self._emotion_prototypes = {
                "aggressive": self.embedder.encode(
                    "angry confrontational aggressive harsh retaliatory impulsive uncompromising"
                ),
                "calm": self.embedder.encode(
                    "calm rational forgiving peaceful understanding reconciling"
                ),
            }
        return self._emotion_prototypes

    @staticmethod
    def _is_l2_layer_item(item: "PersonaItem") -> bool:
        """True for reflective L2 rows (not L0 locked, not L1 core)."""
        if not item:
            return False
        return int(getattr(item, "is_core", 0) or 0) == 0 and int(getattr(item, "locked", 0) or 0) == 0

    def search_top_k(
        self, 
        query_text: str, 
        k: int = 20, 
        emotion_state: Optional[Dict] = None,
        attention_focus: float = 0.5,  # 0..1 attention gate strength
        l2_only: bool = False,
    ) -> List[Tuple[PersonaItem, float]]:
        """
        Top-k cosine retrieval with optional emotional re-ranking and attention gating.

        - High ``attention_focus`` (>0.7): tunnel vision — stricter similarity cutoff, smaller k.
        - Low focus: wider neighbor pool (handled via ``search_k`` scaling elsewhere).
        - ``l2_only``: restrict hits to ``is_core=0`` and ``locked=0`` reflective rules.
        """
        self._ensure_faiss_fresh()

        query_emb = self.embedder.encode(query_text)
        query_emb = query_emb.reshape(1, -1).astype(np.float32)
        
        effective_k = k
        if attention_focus > 0.8:
            effective_k = max(1, k // 2)
            logger.debug(f"High Attention ({attention_focus:.2f}): Tunnel vision active, k reduced to {effective_k}")
        
        search_k = effective_k * 3 if emotion_state else effective_k
        if l2_only:
            search_k = max(search_k * 12, effective_k * 24, 80)
        
        results = []
        if self.use_faiss and self.index.ntotal > 0:
            faiss_k = min(int(search_k), self.index.ntotal)
            similarities, indices = self.index.search(query_emb, faiss_k)
            
            for sim, idx in zip(similarities[0], indices[0]):
                if idx >= 0 and idx in self.id_map:
                    pid = self.id_map[idx]
                    item = self.get_by_id(pid)
                    if item:
                        if l2_only and not self._is_l2_layer_item(item):
                            continue
                        results.append((item, float(sim)))
        else:
            sqlite_k = min(int(search_k), 5000)
            raw = self._search_sqlite(query_emb[0], sqlite_k)
            results = []
            for it, s in raw:
                full = self.get_by_id(it.id)
                results.append((full or it, s))
            if l2_only:
                results = [(it, s) for it, s in results if self._is_l2_layer_item(it)]
            
        if attention_focus > 0.7:
            if results:
                max_sim = max([r[1] for r in results])
                threshold = max(0.4, max_sim * 0.8)
                original_len = len(results)
                results = [r for r in results if r[1] >= threshold]
                if len(results) < original_len:
                    logger.debug(f"High Attention: Filtered {original_len} -> {len(results)} items (thresh={threshold:.2f})")
            
        if emotion_state and results:
            results = self._apply_emotional_bias(results, emotion_state)
        
        results = self._deduplicate_results(results, similarity_threshold=0.90)
            
        return results[:effective_k]
    
    def _deduplicate_results(
        self, 
        results: List[Tuple["PersonaItem", float]], 
        similarity_threshold: float = 0.90
    ) -> List[Tuple["PersonaItem", float]]:
        """
        Drop near-duplicate lines by char n-gram Jaccard overlap; keep the higher-scored hit.
        """
        if len(results) <= 1:
            return results
        
        def get_ngrams(text: str, n: int = 3) -> set:
            """Char n-gram multiset as a set (deduped)."""
            text = text.lower().replace(" ", "")
            return set(text[i:i+n] for i in range(max(1, len(text) - n + 1)))
        
        def jaccard_similarity(text1: str, text2: str) -> float:
            """Jaccard index between n-gram sets."""
            ngrams1 = get_ngrams(text1)
            ngrams2 = get_ngrams(text2)
            if not ngrams1 or not ngrams2:
                return 0.0
            intersection = len(ngrams1 & ngrams2)
            union = len(ngrams1 | ngrams2)
            return intersection / union if union > 0 else 0.0
        
        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
        
        deduplicated = []
        for item, score in sorted_results:
            text = item.text if hasattr(item, "text") else str(item)
            is_duplicate = False
            
            for kept_item, _ in deduplicated:
                kept_text = kept_item.text if hasattr(kept_item, "text") else str(kept_item)
                sim = jaccard_similarity(text, kept_text)
                if sim > similarity_threshold:
                    is_duplicate = True
                    logger.debug(f"[L2 Dedup] Skipping duplicate (sim={sim:.2f}): {text[:50]}...")
                    break
            
            if not is_duplicate:
                deduplicated.append((item, score))
        
        if len(deduplicated) < len(results):
            logger.info(f"[L2 Dedup] Removed {len(results) - len(deduplicated)} duplicate rules")
        
        return deduplicated

    def _apply_emotional_bias(self, results: List[Tuple[PersonaItem, float]], emotion_state: Dict) -> List[Tuple[PersonaItem, float]]:
        """
        Re-rank by resonance with aggressive vs calm prototypes.
        ``emotion_state``: e.g. ``{'arousal': float}`` in roughly [-1, 1].
        """
        arousal = emotion_state.get("arousal", 0.0)
        
        if abs(arousal) < 0.3:
            return results
            
        prototypes = self._get_emotion_prototypes()
        agg_proto = prototypes["aggressive"]
        calm_proto = prototypes["calm"]
        
        adjusted_results = []
        for item, score in results:
            if item.embedding is None:
                adjusted_results.append((item, score))
                continue
                
            norm_item = np.linalg.norm(item.embedding) + 1e-8
            sim_agg = np.dot(item.embedding, agg_proto) / (norm_item * np.linalg.norm(agg_proto))
            sim_calm = np.dot(item.embedding, calm_proto) / (norm_item * np.linalg.norm(calm_proto))
            
            rule_aggression = sim_agg - sim_calm
            bias = arousal * rule_aggression * 0.5
            new_score = score + bias
            adjusted_results.append((item, new_score))
            
        adjusted_results.sort(key=lambda x: x[1], reverse=True)
        return adjusted_results
    
    def _search_sqlite(self, query_emb: np.ndarray, k: int) -> List[Tuple[PersonaItem, float]]:
        """Fallback brute-force cosine scan over stored blobs."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "SELECT id, text, embedding, score FROM persona_items WHERE status='active' AND embedding IS NOT NULL"
            )
            rows = cur.fetchall()
        
        results = []
        for pid, text, emb_blob, score in rows:
            if emb_blob:
                emb = np.frombuffer(emb_blob, dtype=np.float32)
                sim = np.dot(query_emb, emb)
                item = PersonaItem(
                    id=pid, text=text, embedding=emb,
                    score=score
                )
                results.append((item, float(sim)))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]
    
    def get_by_id(self, persona_id: str) -> Optional[PersonaItem]:
        """Load one row by primary key."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(
                "SELECT * FROM persona_items WHERE id=?", (persona_id,)
            )
            row = cur.fetchone()
        
        if not row:
            return None
        
        emb = None
        if row["embedding"]:
            emb = np.frombuffer(row["embedding"], dtype=np.float32)
        
        source = None
        if "source" in row.keys() and row["source"]:
            try:
                source = json.loads(row["source"])
            except:
                pass

        return PersonaItem(
            id=row["id"],
            text=row["text"],
            embedding=emb,
            score=row["score"],
            importance=row["importance"],
            novelty=row["novelty"],
            reliability=row["reliability"],
            evidence_count=row["evidence_count"],
            created_at=row["created_at"],
            last_seen_at=row["last_seen_at"],
            status=row["status"],
            is_core=row["is_core"] if "is_core" in row.keys() else 0,
            core_version=row["core_version"] if "core_version" in row.keys() else 0,
            locked=row["locked"] if "locked" in row.keys() else 0,
            source=source
        )
    
    def get_all_active(self, limit: int = 100) -> List[PersonaItem]:
        """Active rows ordered by ``score`` DESC."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(
                "SELECT * FROM persona_items WHERE status='active' ORDER BY score DESC LIMIT ?",
                (limit,)
            )
            rows = cur.fetchall()
        
        items = []
        for row in rows:
            emb = None
            if row["embedding"]:
                emb = np.frombuffer(row["embedding"], dtype=np.float32)
            
            source = None
            if "source" in row.keys() and row["source"]:
                try:
                    source = json.loads(row["source"])
                except:
                    pass

            items.append(PersonaItem(
                id=row["id"], text=row["text"], embedding=emb,
                score=row["score"], importance=row["importance"],
                novelty=row["novelty"], reliability=row["reliability"],
                evidence_count=row["evidence_count"],
                created_at=row["created_at"], last_seen_at=row["last_seen_at"],
                status=row["status"],
                is_core=row["is_core"] if "is_core" in row.keys() else 0,
                core_version=row["core_version"] if "core_version" in row.keys() else 0,
                locked=row["locked"] if "locked" in row.keys() else 0,
                source=source
        ))
        return items

    def get_core_items(self, limit: int = 100, core_version: Optional[int] = None) -> List[PersonaItem]:
        """
        L1 core bundle rows with dynamic-score ordering.

        Defaults to the highest active ``core_version`` unless ``core_version`` is passed.
        ``core-001`` is pinned first when present.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            version = core_version
            if version is None:
                cur = conn.execute(
                    "SELECT MAX(core_version) FROM persona_items WHERE status='active' AND is_core=1"
                )
                row = cur.fetchone()
                version = row[0] if row and row[0] is not None else None
            
            if version is not None:
                cur = conn.execute(
                    "SELECT * FROM persona_items "
                    "WHERE status='active' AND is_core=1 AND core_version=? "
                    "ORDER BY created_at ASC",
                    (version,)
                )
            else:
                cur = conn.execute(
                    "SELECT * FROM persona_items "
                    "WHERE status='active' AND is_core=1 "
                    "ORDER BY created_at ASC"
                )
            rows = cur.fetchall()
        
        items_with_dynamic_score = []
        for row in rows:
            if row["embedding"]:
                emb = np.frombuffer(row["embedding"], dtype=np.float32)
            else:
                try:
                    emb = self.embedder.encode(row["text"])
                except Exception as e:
                    logger.debug(f"Failed to encode core persona '{row['id']}': {e}")
                    emb = None
            
            dynamic_score = self._calculate_dynamic_score(
                base_score=row["score"],
                evidence_count=row["evidence_count"] or 0,
                last_seen_at=row["last_seen_at"],
                created_at=row["created_at"],
                rule_id=row["id"],
                rule_text=row["text"]
            )
            
            source = None
            if "source" in row.keys() and row["source"]:
                try:
                    source = json.loads(row["source"])
                except:
                    pass

            item = PersonaItem(
                id=row["id"],
                text=row["text"],
                embedding=emb,
                score=row["score"],
                importance=row["importance"],
                novelty=row["novelty"],
                reliability=row["reliability"],
                evidence_count=row["evidence_count"],
                created_at=row["created_at"],
                last_seen_at=row["last_seen_at"],
                status=row["status"],
                is_core=row["is_core"] if "is_core" in row.keys() else 1,
                core_version=row["core_version"] if "core_version" in row.keys() else 0,
                locked=row["locked"] if "locked" in row.keys() else 0,
                source=source
            )
            items_with_dynamic_score.append((item, dynamic_score))
        
        items_with_dynamic_score.sort(key=lambda x: x[1], reverse=True)
        items = [item for item, _ in items_with_dynamic_score[:limit]]
        first_rule = None
        other_items = []
        for item in items:
            if item.id == "core-001":
                first_rule = item
            else:
                other_items.append(item)
        
        if first_rule:
            return [first_rule] + other_items
        else:
            return items
    
    def get_all_core_items_unlocked(self, limit: int = 100) -> List[PersonaItem]:
        """
        All active L1 core rows (``is_core=1``, ``locked=0``) across versions for injection.

        Ordered by ``score`` DESC.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(
                "SELECT * FROM persona_items "
                "WHERE status='active' AND is_core=1 AND locked=0 "
                "ORDER BY score DESC "
                f"LIMIT {limit}"
            )
            rows = cur.fetchall()
        
        result = []
        for row in rows:
            emb = np.frombuffer(row["embedding"], dtype=np.float32) if row["embedding"] else None
            try:
                source = json.loads(row["source"]) if "source" in row.keys() and row["source"] else None
            except:
                source = None
            
            result.append(PersonaItem(
                id=row["id"],
                text=row["text"],
                embedding=emb,
                score=row["score"],
                importance=row["importance"],
                novelty=row["novelty"],
                reliability=row["reliability"],
                evidence_count=row["evidence_count"],
                created_at=row["created_at"],
                last_seen_at=row["last_seen_at"],
                status=row["status"],
                is_core=row["is_core"] if "is_core" in row.keys() else 1,
                core_version=row["core_version"] if "core_version" in row.keys() else 0,
                locked=row["locked"] if "locked" in row.keys() else 0,
                source=source
            ))
        
        logger.debug(f"[PersonaStore] get_all_core_items_unlocked: {len(result)} items")
        return result
    
    def get_locked_items(self) -> List[PersonaItem]:
        """L0 constitution rows (``locked=1``), ordered by ``created_at`` ASC."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(
                "SELECT * FROM persona_items "
                "WHERE status='active' AND locked=1 "
                "ORDER BY created_at ASC"
            )
            rows = cur.fetchall()
        
        result = []
        for row in rows:
            emb = np.frombuffer(row["embedding"], dtype=np.float32) if row["embedding"] else None
            try:
                source_data = row["source"] if "source" in row.keys() else None
                source = json.loads(source_data) if source_data else None
            except Exception:
                source = None
            
            item = PersonaItem(
                id=row["id"],
                text=row["text"],
                embedding=emb,
                score=row["score"],
                importance=row["importance"],
                novelty=row["novelty"],
                reliability=row["reliability"],
                evidence_count=row["evidence_count"],
                created_at=row["created_at"],
                last_seen_at=row["last_seen_at"],
                status=row["status"],
                is_core=row["is_core"] if "is_core" in row.keys() else 0,
                core_version=row["core_version"] if "core_version" in row.keys() else 0,
                locked=row["locked"] if "locked" in row.keys() else 0,
                source=source
            )
            result.append(item)
        
        return result
    
    # Playbook-like cues (down-rank ref-* vs trait-like rules). CJK literals kept; English added.
    OPERATIONAL_KEYWORDS = frozenset({
        '能量', '消耗', '节省', '资源', '代价', '验证', '路径', '确认',
        '工具调用', '执行前', '执行后', '操作前', '操作后', '一次性',
        '低能耗', '高代价', '能量有限', '能量敏感', '能量预算',
        'energy', 'consume', 'consumption', 'saving', 'save', 'resource', 'resources',
        'cost', 'verify', 'verification', 'path', 'confirm', 'confirmation',
        'tool call', 'tool calls', 'tool invocation', 'system call', 'system calls',
        'before execution', 'after execution', 'before operation', 'after operation',
        'one-off', 'one off', 'one-time', 'one time',
        'low energy', 'high cost', 'energy budget', 'energy sensitive', 'limited energy',
    })
    
    def _calculate_dynamic_score(
        self,
        base_score: float,
        evidence_count: int,
        last_seen_at: Optional[str],
        created_at: Optional[str],
        rule_id: str = "",
        rule_text: str = ""
    ) -> float:
        """
        Heuristic dynamic importance for ranking and replacement.

        Mixes base score, log-scaled evidence, recency half-life (~14d), and a ``core-001`` boost.
        For ``ref-*`` ids, extra down-weighting applies when ``rule_text`` hits operational keywords
        (playbook-like lines vs trait-like lines).
        """
        evidence_strength = math.log(1 + evidence_count) / math.log(101)
        evidence_strength = min(1.0, evidence_strength)
        
        recency_factor = 0.5
        if last_seen_at:
            try:
                last_seen = datetime.fromisoformat(last_seen_at.replace('Z', '+00:00'))
                now = datetime.now(timezone.utc)
                days_diff = (now - last_seen).total_seconds() / 86400
                recency_factor = math.exp(-days_diff / 14.0)
            except Exception:
                pass

        core_boost = 0.0
        if rule_id == "core-001":
            core_boost = 2.0

        dynamic_score = (
            base_score * 0.30 +
            evidence_strength * 0.30 +
            recency_factor * 0.30 +
            core_boost
        )
        
        if rule_text and rule_id.startswith("ref-"):
            rt = rule_text or ""
            keyword_count = sum(
                1 for kw in self.OPERATIONAL_KEYWORDS if _persona_keyword_in_text(rt, kw)
            )
            if keyword_count >= 3:
                dynamic_score *= 0.2
            elif keyword_count >= 2:
                dynamic_score *= 0.35
            elif keyword_count == 1:
                dynamic_score *= 0.6
        
        return dynamic_score

    def get_low_score_items(self, threshold: float = 0.3, limit: int = 20) -> List[PersonaItem]:
        """Candidates for compression or eviction: active, non-core, non-locked, low ``score``."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(
                "SELECT * FROM persona_items WHERE status='active' AND score < ? AND is_core=0 AND locked=0 ORDER BY score ASC LIMIT ?",
                (threshold, limit)
            )
            rows = cur.fetchall()
        
        items = []
        for row in rows:
            items.append(PersonaItem(
                id=row["id"], text=row["text"], embedding=None,
                score=row["score"], importance=row["importance"],
                novelty=row["novelty"], reliability=row["reliability"],
                evidence_count=row["evidence_count"],
                created_at=row["created_at"], last_seen_at=row["last_seen_at"],
                status=row["status"],
                is_core=row["is_core"] if "is_core" in row.keys() else 0,
                core_version=row["core_version"] if "core_version" in row.keys() else 0,
                locked=row["locked"] if "locked" in row.keys() else 0
            ))
        return items
    
    def batch_update_from_reflection(
        self, 
        candidates: List[Tuple[str, np.ndarray, Dict]], 
        max_items: int = 300
    ) -> Dict:
        """
        Merge reflection candidates into the L2 pool: exact-text merge, near-duplicate merge, or add.

        ``candidates``: ``[(text, embedding, scores_dict), ...]``.
        Returns ``{"added", "merged", "removed"}`` counts.
        """
        if not candidates:
            return {"added": 0, "merged": 0, "removed": 0}
        
        MERGE_SIMILARITY_THRESHOLD = 0.90
        SCORE_DIFF_THRESHOLD = 0.2
        
        now_iso = datetime.now(timezone.utc).isoformat()
        added_count = 0
        merged_count = 0
        removed_count = 0
        
        existing_items = self.get_all_active(limit=max_items * 2)
        existing_by_text = {item.text: item for item in existing_items}
        
        for text, embedding, scores in candidates:
            if not text or embedding is None:
                continue
            
            total_score = scores.get("total_score", 0.5)
            importance = scores.get("importance", 0.5)
            novelty = scores.get("novelty", 0.5)
            reliability = scores.get("reliability", 0.5)
            
            if text in existing_by_text:
                existing = existing_by_text[text]
                existing.evidence_count = (existing.evidence_count or 0) + 1
                existing.last_seen_at = now_iso
                existing.score = max(existing.score, total_score)
                self.add_or_update(existing, update_embedding=False)
                merged_count += 1
                continue
            
            best_match = None
            best_similarity = 0.0
            
            for existing in existing_items:
                if existing.embedding is None:
                    continue
                
                similarity = np.dot(embedding, existing.embedding) / (
                    np.linalg.norm(embedding) * np.linalg.norm(existing.embedding) + 1e-8
                )
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = existing
            
            if best_match and best_similarity >= MERGE_SIMILARITY_THRESHOLD:
                best_match.evidence_count = (best_match.evidence_count or 0) + 1
                best_match.last_seen_at = now_iso
                if total_score > best_match.score:
                    best_match.score = total_score
                    best_match.text = text
                    best_match.embedding = embedding
                self.add_or_update(best_match, update_embedding=False)
                merged_count += 1
            else:
                current_count = len(existing_items)

                if current_count < max_items:
                    import uuid
                    new_item = PersonaItem(
                        id=f"ref-{uuid.uuid4().hex[:8]}",
                        text=text,
                        embedding=embedding,
                        score=total_score,
                        importance=importance,
                        novelty=novelty,
                        reliability=reliability,
                        evidence_count=1,
                        created_at=now_iso,
                        last_seen_at=now_iso,
                        status="active",
                        is_core=0,
                        core_version=0,
                        locked=0
                    )
                    self.add_or_update(new_item, update_embedding=False)
                    existing_items.append(new_item)
                    added_count += 1
                else:
                    new_dynamic_score = self._calculate_dynamic_score(
                        base_score=total_score,
                        evidence_count=1,
                        last_seen_at=now_iso,
                        created_at=now_iso,
                        rule_id=""
                    )
                    
                    lowest_item = None
                    lowest_dynamic_score = float('inf')
                    
                    for existing in existing_items:
                        if existing.locked == 1 or existing.is_core == 1:
                            continue
                        
                        dynamic_score = self._calculate_dynamic_score(
                            base_score=existing.score,
                            evidence_count=existing.evidence_count or 0,
                            last_seen_at=existing.last_seen_at,
                            created_at=existing.created_at,
                            rule_id=existing.id
                        )
                        
                        if dynamic_score < lowest_dynamic_score:
                            lowest_dynamic_score = dynamic_score
                            lowest_item = existing
                    
                    if lowest_item and new_dynamic_score > lowest_dynamic_score + SCORE_DIFF_THRESHOLD:
                        logger.info(
                            f"Replacing low-score rule '{lowest_item.text[:30]}...' "
                            f"(score={lowest_dynamic_score:.3f}) with new rule "
                            f"'{text[:30]}...' (score={new_dynamic_score:.3f})"
                        )
                        lowest_item.status = "archived"
                        self.add_or_update(lowest_item, update_embedding=False)
                        removed_count += 1
                        
                        import uuid
                        new_item = PersonaItem(
                            id=f"ref-{uuid.uuid4().hex[:8]}",
                            text=text,
                            embedding=embedding,
                            score=total_score,
                            importance=importance,
                            novelty=novelty,
                            reliability=reliability,
                            evidence_count=1,
                            created_at=now_iso,
                            last_seen_at=now_iso,
                            status="active",
                            is_core=0,
                            core_version=0,
                            locked=0
                        )
                        self.add_or_update(new_item, update_embedding=False)
                        added_count += 1
        
        if self.use_faiss and (added_count > 0 or removed_count > 0):
            self._load_faiss_index()
        
        logger.info(f"batch_update_from_reflection: added={added_count}, merged={merged_count}, removed={removed_count}")
        return {"added": added_count, "merged": merged_count, "removed": removed_count}

    def archive_redundant_by_topic(self, max_per_topic: int = 5) -> Dict:
        """
        Periodic redundancy control: if more than ``max_per_topic`` L2 rows match a topic bucket,
        archive the lowest dynamic-score extras. ``details`` uses English topic keys for operator logs.
        """
        TOPIC_GROUPS = {
            "energy_management": [
                '能量', '低能耗', '能量有限', '能量预算', '能量敏感',
                'energy', 'low energy', 'energy budget', 'energy sensitive', 'limited energy',
            ],
            "resource_saving": [
                '消耗', '节省', '损耗', '高成本', '高代价',
                'consumption', 'save', 'saving', 'waste', 'high cost', 'costly',
            ],
            "tool_ops": [
                '工具调用', '系统调用', '调用', '工具',
                'tool call', 'tool calls', 'system call', 'invoke', 'invocation',
            ],
            "verification": [
                '验证', '确认', '路径', '执行前', '执行后',
                'verify', 'verification', 'confirm', 'confirmation', 'path',
                'before execution', 'after execution',
            ],
        }

        total_archived = 0
        details = {}

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                for topic_name, keywords in TOPIC_GROUPS.items():
                    likes_sql = " OR ".join(["text LIKE ?"] * len(keywords))
                    like_params = [f"%{kw}%" for kw in keywords]
                    rows = conn.execute(f"""
                        SELECT id, text, score, evidence_count, last_seen_at, created_at
                        FROM persona_items
                        WHERE is_core = 0 AND locked = 0 AND status = 'active'
                          AND ({likes_sql})
                        ORDER BY score DESC
                    """, like_params).fetchall()

                    if len(rows) <= max_per_topic:
                        continue

                    keep_ids = set()
                    archive_ids = []
                    scored = []
                    for r in rows:
                        ds = self._calculate_dynamic_score(
                            base_score=r["score"],
                            evidence_count=r["evidence_count"] or 0,
                            last_seen_at=r["last_seen_at"],
                            created_at=r["created_at"],
                            rule_id=r["id"],
                            rule_text=r["text"],
                        )
                        scored.append((r["id"], ds))

                    scored.sort(key=lambda x: x[1], reverse=True)
                    for i, (rid, _) in enumerate(scored):
                        if i < max_per_topic:
                            keep_ids.add(rid)
                        else:
                            archive_ids.append(rid)

                    if archive_ids:
                        ph = ",".join("?" for _ in archive_ids)
                        conn.execute(
                            f"UPDATE persona_items SET status='archived' WHERE id IN ({ph})",
                            archive_ids,
                        )
                        total_archived += len(archive_ids)
                        details[topic_name] = {"kept": len(keep_ids), "archived": len(archive_ids)}
                        logger.info(
                            f"[REDUNDANCY-ARCHIVE] {topic_name}: kept {len(keep_ids)}, archived {len(archive_ids)}"
                        )

                if total_archived > 0:
                    conn.commit()
                    if self.use_faiss:
                        self._load_faiss_index()

        except Exception as e:
            logger.error(f"archive_redundant_by_topic failed: {e}", exc_info=True)

        return {"total_archived": total_archived, "details": details}