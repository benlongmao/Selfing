#!/usr/bin/env python3
"""
Narrative identity: persist and render the agent's self-story (origin, turning points, ties, aspirations, rejections).

Idea:
- Identity is not a flat attribute list; it is **the story of who I am**.
- A lived identity bundles narrative, relationships, time (past–present–future selves), and negation (what I refuse to be).
"""
import sqlite3
import json
import re
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Optional
import logging
import numpy as np

from backend.config import config

logger = logging.getLogger(__name__)


class NarrativeIdentity:
    """SQLite-backed narrative shards for a session-scoped identity arc."""
    
    def __init__(self, db_path: str = "data.db"):
        self.db_path = db_path
        self._init_tables()
    
    def _init_tables(self):
        """Create ``identity_narrative`` and indexes if missing."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS identity_narrative (
                        id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        narrative_type TEXT NOT NULL,  -- 'origin', 'turning_point', 'relationship', 'aspiration', 'rejection'
                        content TEXT NOT NULL,
                        significance REAL DEFAULT 0.5,
                        created_at TEXT NOT NULL,
                        related_persona_rules TEXT,  -- JSON array of related rule IDs
                        updated_at TEXT,
                        embedding TEXT             -- JSON float array, cached content embedding
                    )
                """)
                try:
                    conn.execute("ALTER TABLE identity_narrative ADD COLUMN embedding TEXT")
                except sqlite3.OperationalError:
                    pass  # column already exists
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_narrative_session_type 
                    ON identity_narrative(session_id, narrative_type)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_narrative_session 
                    ON identity_narrative(session_id, created_at)
                """)
                conn.commit()
            logger.info("NarrativeIdentity database tables initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize NarrativeIdentity tables: {e}")
    
    def get_relevant_narratives(self, session_id: str, query_text: str, limit: int = 3) -> List[Dict]:
        """
        Rank narrative shards for ``query_text``.

        Prefer ``search_narrative`` (embeddings); on embedder failure, fall back to token overlap + hints.
        """
        narratives = self._get_all_narratives(session_id)
        if not narratives:
            return []

        try:
            return self.search_narrative(query_text, session_id=session_id, limit=limit)
        except Exception as e:
            logger.debug(f"Vector search failed, falling back to keyword scoring: {e}")

        scored = []
        query_terms = set(query_text.lower().split())
        type_hints = {
            "起源": "origin",
            "关系": "relationship",
            "目标": "aspiration",
            "愿望": "aspiration",
            "拒绝": "rejection",
            "beginning": "origin",
            "relationship": "relationship",
            "bond": "relationship",
            "goal": "aspiration",
            "aspiration": "aspiration",
            "wish": "aspiration",
            "reject": "rejection",
            "rejection": "rejection",
        }
        boosted_type = next((v for k, v in type_hints.items() if k in query_text), None)

        for n in narratives:
            score = n["significance"]
            content_lower = n["content"].lower()
            score += sum(0.2 for term in query_terms if term in content_lower)
            if boosted_type and n["type"] == boosted_type:
                score += 1.0
            scored.append((n, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [item[0] for item in scored[:limit]]

    def search_narrative(self, query: str, session_id: Optional[str] = None, limit: int = 5) -> List[Dict]:
        """
        Vector similarity over cached embeddings.

        Score = ``0.6 * cosine_sim + 0.4 * significance``. Each row gains a ``similarity`` field.
        """
        from backend.embedder import get_embedder
        embedder = get_embedder()

        query_vec = embedder.encode(query, normalize=True)  # (D,)

        narratives = self._get_all_narratives_with_embedding(session_id)
        if not narratives:
            return []

        scored = []
        texts_to_embed: List[str] = []
        idx_no_embed: List[int] = []

        for i, n in enumerate(narratives):
            if n.get("embedding") is not None:
                emb = np.array(n["embedding"], dtype=np.float32)
                norm = np.linalg.norm(emb)
                if norm > 1e-8:
                    emb = emb / norm
                sim = float(np.dot(query_vec, emb))
                combined = 0.6 * sim + 0.4 * n["significance"]
                scored.append((n, sim, combined))
            else:
                texts_to_embed.append(n["content"])
                idx_no_embed.append(i)

        if texts_to_embed:
            vecs = embedder.encode_batch(texts_to_embed, normalize=True)
            self._cache_embeddings(
                [narratives[i]["id"] for i in idx_no_embed],
                vecs
            )
            for local_i, global_i in enumerate(idx_no_embed):
                n = narratives[global_i]
                emb = vecs[local_i]
                norm = np.linalg.norm(emb)
                if norm > 1e-8:
                    emb = emb / norm
                sim = float(np.dot(query_vec, emb))
                combined = 0.6 * sim + 0.4 * n["significance"]
                scored.append((n, sim, combined))

        scored.sort(key=lambda x: x[2], reverse=True)
        results = []
        for n, sim, _ in scored[:limit]:
            record = {k: v for k, v in n.items() if k != "embedding"}
            record["similarity"] = round(sim, 4)
            results.append(record)
        return results

    def record_origin_story(self, session_id: str, content: str, related_rules: Optional[List[str]] = None):
        """Persist an ``origin`` shard (how the agent began)."""
        self._add_narrative(
            session_id, 
            "origin", 
            content, 
            significance=1.0,
            related_rules=related_rules
        )
    
    def record_turning_point(self, session_id: str, content: str, significance: float, related_rules: Optional[List[str]] = None):
        """Persist a ``turning_point`` shard with explicit significance in ``[0, 1]``."""
        self._add_narrative(
            session_id, 
            "turning_point", 
            content, 
            significance=significance,
            related_rules=related_rules
        )
    
    def record_relationship(self, session_id: str, user_id: str, relationship_description: str, related_rules: Optional[List[str]] = None):
        """Persist a ``relationship`` line tying this session to ``user_id``."""
        content = f"Relationship with user {user_id}: {relationship_description}"
        self._add_narrative(
            session_id, 
            "relationship", 
            content, 
            significance=0.7,
            related_rules=related_rules
        )
    
    def record_aspiration(self, session_id: str, content: str, related_rules: Optional[List[str]] = None):
        """Persist an ``aspiration`` shard (who the agent wants to become)."""
        self._add_narrative(
            session_id, 
            "aspiration", 
            content, 
            significance=0.8,
            related_rules=related_rules
        )
    
    def record_rejection(self, session_id: str, content: str, related_rules: Optional[List[str]] = None):
        """Persist a ``rejection`` shard (identities the agent refuses)."""
        self._add_narrative(
            session_id, 
            "rejection", 
            content, 
            significance=0.9,
            related_rules=related_rules
        )
    
    def _add_narrative(
        self,
        session_id: str,
        narrative_type: str,
        content: str,
        significance: float = 0.5,
        related_rules: Optional[List[str]] = None
    ):
        """Insert a row and optionally warm the embedding cache; mirror into ``UnifiedMemoryBus``."""
        try:
            anchors = config.get("system.identity_anchors", []) or []
            if anchors and any(k in content for k in anchors):
                significance = 1.0
                logger.info("Identity anchor detected, locking significance to 1.0")

            narrative_id = str(uuid.uuid4())
            created_at = datetime.now(timezone.utc).isoformat()
            related_rules_json = json.dumps(related_rules or [], ensure_ascii=False)

            embedding_json = None
            try:
                from backend.embedder import get_embedder
                vec = get_embedder().encode(content, normalize=True)
                embedding_json = json.dumps(vec.tolist())
            except Exception:
                pass

            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO identity_narrative 
                    (id, session_id, narrative_type, content, significance, created_at,
                     related_persona_rules, updated_at, embedding)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    narrative_id,
                    session_id,
                    narrative_type,
                    content,
                    significance,
                    created_at,
                    related_rules_json,
                    created_at,
                    embedding_json,
                ))
                conn.commit()

            try:
                from backend.unified_memory import UnifiedMemoryBus

                bus = UnifiedMemoryBus(self.db_path)
                bus._add_identity_memory(
                    session_id=session_id,
                    content=content,
                    source_event_id=narrative_id,
                    significance=significance,
                    confidence=0.8,
                    continuity_weight=0.9 if narrative_type in {"origin", "turning_point"} else 0.78,
                    metadata={"narrative_type": narrative_type},
                )
                if narrative_type == "relationship":
                    entity_name = "user"
                    match = re.search(r"Relationship with user ([^:]+):", content)
                    if not match:
                        match = re.search(r"与用户(.+?)的关系", content)
                    if match and match.group(1).strip():
                        entity_name = match.group(1).strip()
                    bus._upsert_relation_memory(
                        session_id=session_id,
                        entity_name=entity_name,
                        summary=content,
                        significance=significance,
                        confidence=0.8,
                        continuity_weight=0.85,
                        event_ref=narrative_id,
                    )
            except Exception as unified_err:
                logger.debug(f"Unified identity sync skipped: {unified_err}")

            logger.debug(f"Added narrative: {narrative_type} for session {session_id}")
        except Exception as e:
            logger.error(f"Failed to add narrative: {e}")
    
    def _get_all_narratives(self, session_id: str) -> List[Dict]:
        """List shards for ``session_id`` (no embedding blob)."""
        narratives = []
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.execute("""
                    SELECT id, narrative_type, content, significance, created_at, related_persona_rules
                    FROM identity_narrative
                    WHERE session_id = ?
                    ORDER BY created_at ASC
                """, (session_id,))
                
                for row in cur.fetchall():
                    narratives.append({
                        "id": row["id"],
                        "type": row["narrative_type"],
                        "content": row["content"],
                        "significance": row["significance"],
                        "created_at": row["created_at"],
                        "related_rules": json.loads(row["related_persona_rules"] or "[]")
                    })
        except Exception as e:
            logger.error(f"Failed to get narratives for session {session_id}: {e}")
        
        return narratives

    def _get_all_narratives_with_embedding(self, session_id: Optional[str]) -> List[Dict]:
        """Like ``_get_all_narratives`` but include parsed ``embedding`` vectors (``session_id`` optional)."""
        narratives = []
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                if session_id:
                    cur = conn.execute("""
                        SELECT id, narrative_type, content, significance, created_at,
                               related_persona_rules, embedding
                        FROM identity_narrative
                        WHERE session_id = ?
                        ORDER BY created_at ASC
                    """, (session_id,))
                else:
                    cur = conn.execute("""
                        SELECT id, narrative_type, content, significance, created_at,
                               related_persona_rules, embedding
                        FROM identity_narrative
                        ORDER BY created_at ASC
                    """)
                for row in cur.fetchall():
                    raw_emb = row["embedding"]
                    emb = json.loads(raw_emb) if raw_emb else None
                    narratives.append({
                        "id": row["id"],
                        "type": row["narrative_type"],
                        "content": row["content"],
                        "significance": row["significance"],
                        "created_at": row["created_at"],
                        "related_rules": json.loads(row["related_persona_rules"] or "[]"),
                        "embedding": emb,
                    })
        except Exception as e:
            logger.error(f"Failed to get narratives with embeddings: {e}")
        return narratives

    def _cache_embeddings(self, narrative_ids: List[str], vectors: "np.ndarray"):
        """Persist embedding JSON for the given narrative ids."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                for nid, vec in zip(narrative_ids, vectors):
                    conn.execute(
                        "UPDATE identity_narrative SET embedding = ? WHERE id = ?",
                        (json.dumps(vec.tolist()), nid)
                    )
                conn.commit()
        except Exception as e:
            logger.warning(f"Failed to cache narrative embeddings: {e}")
    
    def generate_identity_narrative(self, session_id: str) -> str:
        """
        Stitch stored shards into a short first-person identity brief (English labels).

        Stored ``content`` rows may still be bilingual; framing lines here are English-first for prompts.
        """
        narratives = self._get_all_narratives(session_id)
        
        if not narratives:
            return "My story is still being written."
        
        parts = []
        
        origins = [n for n in narratives if n["type"] == "origin"]
        if origins:
            parts.append(f"My story begins with: {origins[0]['content']}")
        
        turning_points = [n for n in narratives if n["type"] == "turning_point"]
        turning_points.sort(key=lambda x: x["significance"], reverse=True)
        if turning_points:
            parts.append(f"A pivotal moment: {turning_points[0]['content']}")
        
        relationships = [n for n in narratives if n["type"] == "relationship"]
        if relationships:
            rel_text = "; ".join([r["content"] for r in relationships[:3]])
            parts.append(f"Important relationships: {rel_text}")
        
        aspirations = [n for n in narratives if n["type"] == "aspiration"]
        if aspirations:
            parts.append(f"I want to become: {aspirations[0]['content']}")
        
        rejections = [n for n in narratives if n["type"] == "rejection"]
        if rejections:
            parts.append(f"I refuse to become: {rejections[0]['content']}")
        
        return "\n".join(parts) if parts else "My story is still being written."
    
    def get_narratives_by_type(self, session_id: str, narrative_type: str) -> List[Dict]:
        """Filter ``_get_all_narratives`` to one ``narrative_type`` token."""
        narratives = self._get_all_narratives(session_id)
        return [n for n in narratives if n["type"] == narrative_type]
    
    def update_narrative(self, narrative_id: str, content: Optional[str] = None, significance: Optional[float] = None):
        """Patch ``content`` / ``significance`` / ``updated_at`` for a row id."""
        try:
            updates = []
            params = []
            
            if content is not None:
                updates.append("content = ?")
                params.append(content)
            
            if significance is not None:
                updates.append("significance = ?")
                params.append(significance)
            
            if not updates:
                return
            
            updates.append("updated_at = ?")
            params.append(datetime.now(timezone.utc).isoformat())
            params.append(narrative_id)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(f"""
                    UPDATE identity_narrative
                    SET {', '.join(updates)}
                    WHERE id = ?
                """, params)
                conn.commit()
            
            logger.debug(f"Updated narrative: {narrative_id}")
        except Exception as e:
            logger.error(f"Failed to update narrative {narrative_id}: {e}")
    
    def delete_narrative(self, narrative_id: str):
        """Hard-delete a shard by primary key."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    DELETE FROM identity_narrative
                    WHERE id = ?
                """, (narrative_id,))
                conn.commit()
            
            logger.debug(f"Deleted narrative: {narrative_id}")
        except Exception as e:
            logger.error(f"Failed to delete narrative {narrative_id}: {e}")

