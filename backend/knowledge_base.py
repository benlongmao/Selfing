"""
Knowledge base for the agent (Knowledge Base).

Lets the agent learn, persist, retrieve, and refresh facts with semantic search
and optional embeddings. Categories and validation literals remain **Chinese**
strings for SQLite / tool-schema compatibility (see ``VALID_CATEGORIES``).

Created: 2026-02-07. Version: v1.0.
"""

import os
import re
import sqlite3
import uuid
import json
import logging
import time
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple

logger = logging.getLogger(__name__)

# Optional text embedder (384-d English model when configured).
try:
    from backend.embedder import get_embedder
    EMBEDDER_AVAILABLE = True
except ImportError:
    EMBEDDER_AVAILABLE = False
    get_embedder = None


class KnowledgeBase:
    """
    Persistent knowledge store with optional vector search.

    Responsibilities:
    - Ingest facts from multiple ``source`` types.
    - Semantic (or keyword) retrieval scoped by ``session_id``.
    - Update, link, and delete rows while keeping embeddings in sync when possible.
    """

    # Allowed ``source`` values (stored as TEXT; keep literals stable).
    SOURCE_WEB_SEARCH = "web_search"
    SOURCE_USER_TEACH = "user_teach"
    SOURCE_SELF_DISCOVERY = "self_discovery"
    SOURCE_READING = "reading"
    SOURCE_EXPERIENCE = "experience"
    SOURCE_REFLECTION = "reflection"
    
    VALID_SOURCES = {
        "web_search", "user_teach", "self_discovery",
        "reading", "experience", "reflection", "user_identity",
    }
    
    # Category labels (Chinese strings are the canonical DB / contract values).
    CATEGORY_TECH = "技术"
    CATEGORY_SCIENCE = "科学"
    CATEGORY_COMMON = "常识"
    CATEGORY_PERSONAL = "个人经验"
    CATEGORY_PROJECT = "项目相关"
    CATEGORY_USER = "用户偏好"
    CATEGORY_WORLD = "世界知识"
    
    VALID_CATEGORIES = {
        "技术", "科学", "常识", "个人经验",
        "项目相关", "用户偏好", "世界知识", "用户身份", "哲学",
    }
    
    def __init__(self, db_path: str = "data.db"):
        self.db_path = db_path
        self.embedder = None
        self._last_wiki_metadata_sync = 0.0
        
        if EMBEDDER_AVAILABLE:
            try:
                self.embedder = get_embedder()
                logger.info("KnowledgeBase: Embedder initialized")
            except Exception as e:
                logger.warning(f"KnowledgeBase: Failed to initialize embedder: {e}")
        
        self._ensure_tables()
    
    def _ensure_tables(self):
        """Create SQLite tables if they are missing."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_items (
                    id TEXT PRIMARY KEY,
                    session_id TEXT DEFAULT 'selfing-session',
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source TEXT,
                    category TEXT,
                    tags TEXT,
                    confidence REAL DEFAULT 0.5,
                    embedding BLOB,
                    created_at TEXT NOT NULL,
                    updated_at TEXT,
                    last_accessed TEXT,
                    access_count INTEGER DEFAULT 0,
                    usefulness_score REAL DEFAULT 0.5,
                    related_items TEXT,
                    metadata TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS learning_goals (
                    id TEXT PRIMARY KEY,
                    session_id TEXT DEFAULT 'selfing-session',
                    topic TEXT NOT NULL,
                    reason TEXT,
                    status TEXT DEFAULT 'active',
                    progress REAL DEFAULT 0.0,
                    knowledge_gained INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT,
                    completed_at TEXT,
                    metadata TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS learning_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT DEFAULT 'selfing-session',
                    goal_id TEXT,
                    action TEXT,
                    content TEXT,
                    result TEXT,
                    tokens_used INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (goal_id) REFERENCES learning_goals(id)
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_links (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    relation_type TEXT,
                    strength REAL DEFAULT 0.5,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (source_id) REFERENCES knowledge_items(id),
                    FOREIGN KEY (target_id) REFERENCES knowledge_items(id)
                )
            """)
            
            conn.commit()
            logger.info("KnowledgeBase: Tables initialized")

    def _json_loads(self, value: Any, default: Any) -> Any:
        if value in (None, ""):
            return default
        if isinstance(value, (dict, list)):
            return value
        try:
            return json.loads(value)
        except Exception:
            return default

    def _governance_weight(self, metadata: Dict[str, Any]) -> float:
        """Weight DB retrieval by wiki governance metadata when available."""
        wiki = metadata.get("wiki") if isinstance(metadata, dict) else {}
        if not isinstance(wiki, dict):
            wiki = {}
        status = str(wiki.get("status", metadata.get("wiki_status", "active")))
        priority = str(wiki.get("priority", metadata.get("wiki_priority", "normal")))
        kind = str(wiki.get("kind", metadata.get("wiki_kind", "")))

        status_weight = {
            "active": 1.0,
            "historical": 0.72,
            "archive_candidate": 0.55,
            "archived": 0.35,
        }.get(status, 0.85)
        priority_weight = {"high": 1.2, "normal": 1.0, "low": 0.7}.get(priority, 1.0)
        kind_weight = {
            "operation_record": 0.72,
            "world_event": 0.8,
            "methodology": 1.08,
            "lesson": 1.08,
            "preference": 1.12,
            "project": 1.12,
        }.get(kind, 1.0)
        return status_weight * priority_weight * kind_weight

    def _query_terms(self, query: str) -> List[str]:
        terms = set()
        query_lower = (query or "").lower()
        terms.update(w for w in query_lower.split() if len(w) > 1)
        terms.update(re.findall(r"[a-zA-Z]{2,}", query_lower))
        for chunk in re.findall(r"[\u4e00-\u9fff]+", query or ""):
            if len(chunk) <= 4:
                terms.add(chunk)
            for size in (2, 3, 4):
                if len(chunk) >= size:
                    terms.update(chunk[i:i + size] for i in range(len(chunk) - size + 1))
        return [term for term in terms if term]

    def _keyword_score_for_row(self, query: str, row: sqlite3.Row) -> float:
        query_lower = (query or "").lower()
        title_lower = (row["title"] or "").lower()
        content_lower = (row["content"] or "").lower()
        tags_lower = (row["tags"] or "").lower() if "tags" in row.keys() else ""
        metadata = self._json_loads(row["metadata"], {}) if "metadata" in row.keys() else {}
        wiki = metadata.get("wiki", {}) if isinstance(metadata, dict) else {}
        wiki_text = " ".join(
            str(wiki.get(k, ""))
            for k in ("summary", "use_when", "kind", "status", "priority", "path")
        ).lower()

        score = 0.0
        if query_lower and query_lower in title_lower:
            score += 0.6
        if query_lower and query_lower in content_lower:
            score += 0.3
        if query_lower and query_lower in wiki_text:
            score += 0.25

        terms = self._query_terms(query)
        if terms:
            matched = 0
            for term in terms:
                if term in title_lower:
                    score += 0.12
                    matched += 1
                elif term in tags_lower or term in wiki_text:
                    score += 0.08
                    matched += 1
                elif term in content_lower:
                    score += 0.04
                    matched += 1
            score += 0.2 * (matched / max(len(terms), 1))
        return min(score, 1.0)

    def _merge_wiki_metadata(self, metadata: Dict[str, Any], entry: Dict[str, Any]) -> Dict[str, Any]:
        merged = dict(metadata or {})
        wiki_meta = {
            "path": entry.get("path", ""),
            "title": entry.get("title", ""),
            "category": entry.get("category", ""),
            "kind": entry.get("kind", ""),
            "status": entry.get("status", ""),
            "priority": entry.get("priority", ""),
            "summary": entry.get("summary", ""),
            "use_when": entry.get("use_when", ""),
            "tags": entry.get("tags", []) or [],
            "aliases": entry.get("aliases", []) or [],
            "supersedes": entry.get("supersedes", []) or [],
            "last_compiled": entry.get("last_compiled", ""),
            "synced_at": datetime.now(timezone.utc).isoformat(),
        }
        merged["wiki"] = wiki_meta
        # Flat keys keep older scripts/tooling useful without nested JSON parsing.
        merged["wiki_path"] = wiki_meta["path"]
        merged["wiki_kind"] = wiki_meta["kind"]
        merged["wiki_status"] = wiki_meta["status"]
        merged["wiki_priority"] = wiki_meta["priority"]
        merged["wiki_summary"] = wiki_meta["summary"]
        merged["wiki_use_when"] = wiki_meta["use_when"]
        return merged

    def _update_item_metadata_from_wiki_entry(
        self,
        knowledge_id: str,
        entry: Dict[str, Any],
        session_id: Optional[str] = None,
    ) -> bool:
        if not knowledge_id or not entry:
            return False
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if session_id is None:
                row = conn.execute(
                    "SELECT id, tags, metadata FROM knowledge_items WHERE id = ?",
                    (knowledge_id,),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT id, tags, metadata FROM knowledge_items WHERE id = ? AND session_id = ?",
                    (knowledge_id, session_id),
                ).fetchone()
            if not row:
                return False

            metadata = self._json_loads(row["metadata"], {})
            new_metadata = self._merge_wiki_metadata(metadata, entry)
            existing_tags = self._json_loads(row["tags"], [])
            merged_tags = sorted(set(existing_tags + (entry.get("tags") or [])))
            conn.execute(
                """
                UPDATE knowledge_items
                SET metadata = ?, tags = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    json.dumps(new_metadata, ensure_ascii=False),
                    json.dumps(merged_tags, ensure_ascii=False),
                    datetime.now(timezone.utc).isoformat(),
                    knowledge_id,
                ),
            )
            conn.commit()
            return True

    def sync_wiki_metadata_from_catalog(
        self,
        session_id: Optional[str] = None,
        force_rebuild: bool = False,
    ) -> Dict[str, Any]:
        """Sync wiki catalog governance metadata back into ``knowledge_items.metadata``."""
        try:
            from backend.knowledge_compiler import get_compiler

            compiler = get_compiler()
            entries = compiler.rebuild_catalog() if force_rebuild else compiler._load_catalog()
            if not entries:
                entries = compiler.rebuild_catalog()

            seen_ids = set()
            updated = 0
            for entry in entries:
                for knowledge_id in entry.get("knowledge_ids") or []:
                    if knowledge_id in seen_ids:
                        continue
                    seen_ids.add(knowledge_id)
                    if self._update_item_metadata_from_wiki_entry(knowledge_id, entry, session_id=session_id):
                        updated += 1
            return {
                "success": True,
                "entries": len(entries),
                "knowledge_ids": len(seen_ids),
                "updated": updated,
            }
        except Exception as e:
            logger.debug(f"[KNOWLEDGE] Wiki metadata sync skipped: {e}")
            return {"success": False, "error": str(e)}

    def _maybe_sync_wiki_metadata(self, session_id: Optional[str] = None):
        now = time.monotonic()
        if now - self._last_wiki_metadata_sync < 300:
            return
        result = self.sync_wiki_metadata_from_catalog(session_id=session_id)
        if result.get("success"):
            self._last_wiki_metadata_sync = now

    def _augment_results_with_wiki(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Add catalog-only wiki hits as virtual knowledge rows."""
        try:
            from backend.knowledge_compiler import get_compiler

            existing_paths = {
                ((item.get("wiki") or {}).get("path") or "")
                for item in results
                if isinstance(item.get("wiki"), dict)
            }
            wiki_hits = get_compiler().search_wiki(query, limit=max(top_k, 3))
            for hit in wiki_hits:
                path = hit.get("path", "")
                if not path or path in existing_paths:
                    continue
                summary = hit.get("summary") or hit.get("use_when") or ""
                content = summary
                if hit.get("use_when") and hit.get("use_when") not in content:
                    content = f"{content}\nUse when: {hit.get('use_when')}".strip()
                results.append({
                    "id": f"wiki:{path}",
                    "title": hit.get("title", ""),
                    "content": content[:500],
                    "source": "wiki",
                    "category": hit.get("category", ""),
                    "tags": hit.get("tags", []) or [],
                    "confidence": 0.65,
                    "similarity": round(float(hit.get("score", 0.0)), 3),
                    "raw_similarity": round(float(hit.get("score", 0.0)), 3),
                    "access_count": 0,
                    "wiki": hit,
                    "metadata": {"wiki": hit},
                })
                existing_paths.add(path)
            results.sort(key=lambda item: item.get("similarity", 0.0), reverse=True)
            return results[:top_k]
        except Exception as e:
            logger.debug(f"[KNOWLEDGE] Wiki result augmentation skipped: {e}")
            return results[:top_k]
    
    def add_knowledge(
        self,
        title: str,
        content: str,
        source: str,
        category: str,
        tags: List[str] = None,
        confidence: float = 0.5,
        session_id: str = "selfing-session",
        metadata: Dict = None
    ) -> Dict[str, Any]:
        """
        Insert a new knowledge row after validation and optional dedup.

        Args:
            title: Short headline for the fact.
            content: Body text (distilled, not raw dumps).
            source: One of ``VALID_SOURCES``.
            category: One of ``VALID_CATEGORIES`` (Chinese literals are canonical).
            tags: Optional tag list serialized to JSON.
            confidence: Score in ``[0, 1]``.
            session_id: Session scope for isolation.
            metadata: Arbitrary JSON metadata.

        Returns:
            ``success`` plus ``knowledge_id`` / ``message``, or ``error``.
        """
        try:
            if source not in self.VALID_SOURCES:
                logger.warning(f"[KNOWLEDGE] Invalid source '{source}', correcting to 'experience'")
                source = "experience"

            if category not in self.VALID_CATEGORIES:
                logger.warning(
                    f"[KNOWLEDGE] Invalid category '{category}', "
                    f"correcting to canonical default '{self.CATEGORY_COMMON}'"
                )
                category = self.CATEGORY_COMMON

            rejection_reason = self._check_quality(title, content)
            if rejection_reason:
                logger.info(f"[KNOWLEDGE] Rejected: {rejection_reason} | title={title[:50]}")
                return {"success": False, "error": rejection_reason}
            
            knowledge_id = f"K-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}"
            created_at = datetime.now(timezone.utc).isoformat()
            
            embedding_blob = None
            if self.embedder:
                try:
                    text_to_embed = f"{title}. {content}"
                    embedding = self.embedder.encode(text_to_embed)
                    embedding_blob = embedding.tobytes()
                except Exception as e:
                    logger.warning(f"Failed to generate embedding: {e}")
            
            similar = self.search_knowledge(title, top_k=3, session_id=session_id)
            if similar:
                for item in similar:
                    if item.get("similarity", 0) > 0.85:
                        return {
                            "success": False,
                            "error": "Very similar knowledge already exists",
                            "similar_id": item["id"],
                            "similar_title": item["title"],
                            "suggestion": "Update the existing entry instead of inserting a duplicate",
                        }
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO knowledge_items 
                    (id, session_id, title, content, source, category, tags, 
                     confidence, embedding, created_at, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    knowledge_id, session_id, title, content, source, category,
                    json.dumps(tags or [], ensure_ascii=False),
                    confidence, embedding_blob, created_at,
                    json.dumps(metadata or {}, ensure_ascii=False)
                ))
                conn.commit()
            
            logger.info(f"[KNOWLEDGE] Added: {knowledge_id} - {title}")
            
            # Level-1 wiki append (fast path, no LLM tokens).
            try:
                from backend.knowledge_compiler import get_compiler
                compiler = get_compiler()
                wiki_result = compiler.quick_append(
                    title=title, content=content,
                    category=category, source_id=knowledge_id,
                    confidence=confidence,
                )
                entry = wiki_result.get("entry") if isinstance(wiki_result, dict) else None
                if entry:
                    self._update_item_metadata_from_wiki_entry(knowledge_id, entry, session_id=session_id)
            except Exception as e:
                logger.debug(f"[KNOWLEDGE] Wiki compilation skipped: {e}")
            
            return {
                "success": True,
                "knowledge_id": knowledge_id,
                "title": title,
                "category": category,
                "message": "Knowledge saved",
            }
            
        except Exception as e:
            logger.error(f"[KNOWLEDGE] Failed to add knowledge: {e}")
            return {"success": False, "error": str(e)}
    
    def _check_quality(self, title: str, content: str) -> Optional[str]:
        """
        Lightweight quality gate before insert.

        Returns ``None`` when the row should be accepted, otherwise a human-readable
        rejection reason (English for operators / tools).
        """
        title_stripped = title.strip()
        content_stripped = content.strip()
        title_lower = title_stripped.lower()

        if len(content_stripped) < 10:
            return "Content too short to be useful knowledge"

        # Reject definition-style titles (Chinese marker kept for user/tool parity).
        if title_stripped.startswith("定义：") or title_lower.startswith("definition:"):
            return "Bad title: do not prefix with a definition marker; summarize the fact instead"

        if (content_stripped.endswith("？") or content_stripped.endswith("?")) and len(content_stripped) < 50:
            return "Content looks like a bare question, not a distilled fact"

        if len(content_stripped) > 800:
            return "Content too long (>800 characters); distill key points before storing"

        if len(title_stripped) > 15 and content_stripped.startswith(title_stripped[:15]):
            if title_stripped.endswith("..."):
                return "Title appears to truncate the body; choose a concise headline"

        return None
    
    def search_knowledge(
        self,
        query: str,
        top_k: int = 5,
        category: str = None,
        min_confidence: float = 0.0,
        session_id: str = "selfing-session"
    ) -> List[Dict[str, Any]]:
        """
        Retrieve knowledge by semantic similarity or keyword overlap.

        Args:
            query: Natural-language query.
            top_k: Max rows to return after ranking.
            category: Optional filter on ``category`` (Chinese literal).
            min_confidence: Minimum stored confidence.
            session_id: Session scope.

        Returns:
            Ranked list of dicts with truncated ``content`` for UI/prompt use.
        """
        try:
            self._maybe_sync_wiki_metadata(session_id=session_id)
            results = []
            
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                sql = """
                    SELECT id, title, content, source, category, tags, 
                           confidence, embedding, created_at, access_count,
                           usefulness_score, metadata
                    FROM knowledge_items 
                    WHERE session_id = ? AND confidence >= ?
                """
                params = [session_id, min_confidence]
                
                if category:
                    sql += " AND category = ?"
                    params.append(category)
                
                cursor = conn.execute(sql, params)
                rows = cursor.fetchall()
            
            if not rows:
                return self._augment_results_with_wiki(query, [], top_k)
            
            if self.embedder:
                try:
                    query_embedding = self.embedder.encode(query)
                    
                    scored_results = []
                    for row in rows:
                        metadata = self._json_loads(row["metadata"], {})
                        if row["embedding"]:
                            item_embedding = np.frombuffer(row["embedding"], dtype=np.float32)
                            raw_similarity = np.dot(query_embedding, item_embedding) / (
                                np.linalg.norm(query_embedding) * np.linalg.norm(item_embedding) + 1e-8
                            )
                        else:
                            raw_similarity = self._keyword_score_for_row(query, row)
                        similarity = float(raw_similarity) * self._governance_weight(metadata)
                        
                        scored_results.append((row, float(similarity), float(raw_similarity), metadata))
                    
                    scored_results.sort(key=lambda x: x[1], reverse=True)

                    for row, similarity, raw_similarity, metadata in scored_results[:top_k]:
                        if similarity > 0.1:
                            results.append({
                                "id": row["id"],
                                "title": row["title"],
                                "content": row["content"][:500] + "..." if len(row["content"]) > 500 else row["content"],
                                "source": row["source"],
                                "category": row["category"],
                                "tags": json.loads(row["tags"]) if row["tags"] else [],
                                "confidence": row["confidence"],
                                "similarity": round(similarity, 3),
                                "raw_similarity": round(raw_similarity, 3),
                                "access_count": row["access_count"],
                                "wiki": metadata.get("wiki", {}),
                                "metadata": metadata,
                            })
                    
                except Exception as e:
                    logger.warning(f"Semantic search failed, falling back to keyword: {e}")
                    results = self._keyword_search(query, rows, top_k)
            else:
                results = self._keyword_search(query, rows, top_k)

            if results:
                self._update_access(results[0]["id"])
            
            return self._augment_results_with_wiki(query, results, top_k)
            
        except Exception as e:
            logger.error(f"[KNOWLEDGE] Search failed: {e}")
            return []
    
    def _keyword_search(self, query: str, rows: List, top_k: int) -> List[Dict]:
        """Fallback ranking when embeddings are missing or encoding fails."""
        results = []
        query_lower = query.lower()
        
        for row in rows:
            metadata = self._json_loads(row["metadata"], {}) if "metadata" in row.keys() else {}
            raw_score = self._keyword_score_for_row(query, row)
            score = raw_score * self._governance_weight(metadata)
            
            if score > 0:
                results.append({
                    "id": row["id"],
                    "title": row["title"],
                    "content": row["content"][:500] + "..." if len(row["content"]) > 500 else row["content"],
                    "source": row["source"],
                    "category": row["category"],
                    "tags": json.loads(row["tags"]) if row["tags"] else [],
                    "confidence": row["confidence"],
                    "similarity": round(min(score, 1.0), 3),
                    "raw_similarity": round(raw_score, 3),
                    "access_count": row["access_count"],
                    "wiki": metadata.get("wiki", {}),
                    "metadata": metadata,
                })
        
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]
    
    def _update_access(self, knowledge_id: str):
        """Bump access counters for the top hit."""
        if knowledge_id.startswith("wiki:"):
            return
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE knowledge_items 
                    SET access_count = access_count + 1,
                        last_accessed = ?
                    WHERE id = ?
                """, (datetime.now(timezone.utc).isoformat(), knowledge_id))
                conn.commit()
        except Exception as e:
            logger.warning(f"Failed to update access: {e}")
    
    def update_knowledge(
        self,
        knowledge_id: str,
        new_content: str = None,
        new_confidence: float = None,
        add_tags: List[str] = None,
        new_title: str = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Patch an existing row (content, title, tags, confidence, embedding).

        Args:
            knowledge_id: Primary key of the row.
            new_content: Replacement body text.
            new_confidence: Replacement confidence score.
            add_tags: Tags merged into the JSON tag list.
            new_title: Optional replacement title.
            session_id: When set, updates are rejected if the row belongs to another session.

        Returns:
            ``success`` / ``message`` or ``error``.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row

                cursor = conn.execute(
                    "SELECT * FROM knowledge_items WHERE id = ?",
                    (knowledge_id,)
                )
                row = cursor.fetchone()
                
                if not row:
                    return {"success": False, "error": f"Knowledge not found: {knowledge_id}"}

                if session_id is not None and row["session_id"] != session_id:
                    return {"success": False, "error": "Knowledge belongs to a different session"}

                nt = str(new_title).strip() if new_title is not None and str(new_title).strip() else None
                has_change = bool(new_content) or new_confidence is not None or bool(add_tags) or nt is not None
                if not has_change:
                    return {
                        "success": False,
                        "error": "Provide at least one of new_title, new_content, new_confidence, or add_tags",
                    }

                title_final = nt if nt is not None else row["title"]
                content_final = new_content if new_content else row["content"]

                updates = []
                params = []

                if nt is not None:
                    updates.append("title = ?")
                    params.append(nt)

                if new_content:
                    updates.append("content = ?")
                    params.append(new_content)

                if new_confidence is not None:
                    updates.append("confidence = ?")
                    params.append(new_confidence)

                if add_tags:
                    existing_tags = json.loads(row["tags"]) if row["tags"] else []
                    all_tags = list(set(existing_tags + add_tags))
                    updates.append("tags = ?")
                    params.append(json.dumps(all_tags, ensure_ascii=False))

                if self.embedder and (new_content or nt is not None):
                    try:
                        text_to_embed = f"{title_final}. {content_final}"
                        embedding = self.embedder.encode(text_to_embed)
                        updates.append("embedding = ?")
                        params.append(embedding.tobytes())
                    except Exception:
                        pass

                updates.append("updated_at = ?")
                params.append(datetime.now(timezone.utc).isoformat())

                params.append(knowledge_id)

                conn.execute(f"""
                    UPDATE knowledge_items 
                    SET {", ".join(updates)}
                    WHERE id = ?
                """, params)
                conn.commit()
            
            logger.info(f"[KNOWLEDGE] Updated: {knowledge_id}")
            
            try:
                from backend.knowledge_compiler import get_compiler
                category = row["category"] if row else self.CATEGORY_COMMON
                wiki_result = get_compiler().quick_append(
                    title=title_final, content=content_final,
                    category=category, source_id=knowledge_id,
                )
                entry = wiki_result.get("entry") if isinstance(wiki_result, dict) else None
                if entry:
                    self._update_item_metadata_from_wiki_entry(
                        knowledge_id,
                        entry,
                        session_id=session_id or row["session_id"],
                    )
            except Exception as e2:
                logger.debug(f"[KNOWLEDGE] Wiki compilation on update skipped: {e2}")
            
            return {
                "success": True,
                "knowledge_id": knowledge_id,
                "message": "Knowledge updated",
            }
            
        except Exception as e:
            logger.error(f"[KNOWLEDGE] Update failed: {e}")
            return {"success": False, "error": str(e)}
    
    def link_knowledge(
        self,
        source_id: str,
        target_id: str,
        relation_type: str = "related",
        strength: float = 0.5
    ) -> Dict[str, Any]:
        """
        Create or refresh a directed edge in ``knowledge_links``.

        Args:
            source_id: Origin knowledge id.
            target_id: Destination knowledge id.
            relation_type: ``related``, ``prerequisite``, ``extends``, ``contradicts``, etc.
            strength: Edge weight in ``[0, 1]``.

        Returns:
            Success payload or ``error``.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT id FROM knowledge_items WHERE id IN (?, ?)",
                    (source_id, target_id)
                )
                if len(cursor.fetchall()) != 2:
                    return {"success": False, "error": "One or both knowledge rows are missing"}

                cursor = conn.execute("""
                    SELECT id FROM knowledge_links 
                    WHERE source_id = ? AND target_id = ?
                """, (source_id, target_id))
                
                if cursor.fetchone():
                    conn.execute("""
                        UPDATE knowledge_links 
                        SET relation_type = ?, strength = ?
                        WHERE source_id = ? AND target_id = ?
                    """, (relation_type, strength, source_id, target_id))
                else:
                    conn.execute("""
                        INSERT INTO knowledge_links 
                        (source_id, target_id, relation_type, strength, created_at)
                        VALUES (?, ?, ?, ?, ?)
                    """, (source_id, target_id, relation_type, strength,
                          datetime.now(timezone.utc).isoformat()))
                
                conn.commit()
            
            return {
                "success": True,
                "message": f"Linked as {relation_type}",
                "source_id": source_id,
                "target_id": target_id
            }
            
        except Exception as e:
            logger.error(f"[KNOWLEDGE] Link failed: {e}")
            return {"success": False, "error": str(e)}
    
    def get_related_knowledge(
        self,
        knowledge_id: str,
        max_depth: int = 1
    ) -> List[Dict[str, Any]]:
        """Return outbound neighbors for ``knowledge_id``."""
        try:
            related = []
            
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                cursor = conn.execute("""
                    SELECT k.id, k.title, k.category, l.relation_type, l.strength
                    FROM knowledge_links l
                    JOIN knowledge_items k ON l.target_id = k.id
                    WHERE l.source_id = ?
                    ORDER BY l.strength DESC
                """, (knowledge_id,))
                
                for row in cursor.fetchall():
                    related.append({
                        "id": row["id"],
                        "title": row["title"],
                        "category": row["category"],
                        "relation": row["relation_type"],
                        "strength": row["strength"]
                    })
            
            return related
            
        except Exception as e:
            logger.error(f"[KNOWLEDGE] Get related failed: {e}")
            return []
    
    def get_knowledge_stats(self, session_id: str = "selfing-session") -> Dict[str, Any]:
        """Aggregate counts and recent activity for a session."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                cursor = conn.execute(
                    "SELECT COUNT(*) as total FROM knowledge_items WHERE session_id = ?",
                    (session_id,)
                )
                total = cursor.fetchone()["total"]
                
                cursor = conn.execute("""
                    SELECT category, COUNT(*) as count 
                    FROM knowledge_items 
                    WHERE session_id = ?
                    GROUP BY category
                """, (session_id,))
                by_category = {row["category"]: row["count"] for row in cursor.fetchall()}
                
                cursor = conn.execute("""
                    SELECT source, COUNT(*) as count 
                    FROM knowledge_items 
                    WHERE session_id = ?
                    GROUP BY source
                """, (session_id,))
                by_source = {row["source"]: row["count"] for row in cursor.fetchall()}
                
                cursor = conn.execute("""
                    SELECT id, title, created_at 
                    FROM knowledge_items 
                    WHERE session_id = ?
                    ORDER BY created_at DESC
                    LIMIT 5
                """, (session_id,))
                recent = [{"id": row["id"], "title": row["title"], "created_at": row["created_at"]} 
                          for row in cursor.fetchall()]
                
                cursor = conn.execute("""
                    SELECT id, title, access_count 
                    FROM knowledge_items 
                    WHERE session_id = ?
                    ORDER BY access_count DESC
                    LIMIT 5
                """, (session_id,))
                most_accessed = [{"id": row["id"], "title": row["title"], "access_count": row["access_count"]} 
                                 for row in cursor.fetchall()]
            
            return {
                "success": True,
                "total_knowledge": total,
                "by_category": by_category,
                "by_source": by_source,
                "recent_additions": recent,
                "most_accessed": most_accessed
            }
            
        except Exception as e:
            logger.error(f"[KNOWLEDGE] Stats failed: {e}")
            return {"success": False, "error": str(e)}
    
    def delete_knowledge(self, knowledge_id: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Delete a row (and incident edges). When ``session_id`` is supplied, the row
        must belong to that session to avoid cross-tenant mistakes.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                if session_id is not None:
                    cur = conn.execute(
                        "SELECT session_id FROM knowledge_items WHERE id = ?",
                        (knowledge_id,),
                    )
                    r = cur.fetchone()
                    if not r:
                        return {"success": False, "error": "Knowledge not found"}
                    if r[0] != session_id:
                        return {"success": False, "error": "Knowledge belongs to a different session"}

                conn.execute("DELETE FROM knowledge_links WHERE source_id = ? OR target_id = ?",
                            (knowledge_id, knowledge_id))

                cursor = conn.execute("DELETE FROM knowledge_items WHERE id = ?", (knowledge_id,))

                if cursor.rowcount == 0:
                    return {"success": False, "error": "Knowledge not found"}
                
                conn.commit()
            
            return {"success": True, "message": "Knowledge deleted"}
            
        except Exception as e:
            logger.error(f"[KNOWLEDGE] Delete failed: {e}")
            return {"success": False, "error": str(e)}


class LearningGoalManager:
    """Lightweight CRUD helpers for ``learning_goals`` / ``learning_logs``."""
    
    def __init__(self, db_path: str = "data.db"):
        self.db_path = db_path
    
    def create_goal(
        self,
        topic: str,
        reason: str,
        session_id: str = "selfing-session"
    ) -> Dict[str, Any]:
        """Insert an ``active`` learning goal."""
        try:
            goal_id = f"LG-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:4]}"
            created_at = datetime.now(timezone.utc).isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO learning_goals 
                    (id, session_id, topic, reason, status, created_at)
                    VALUES (?, ?, ?, ?, 'active', ?)
                """, (goal_id, session_id, topic, reason, created_at))
                conn.commit()
            
            logger.info(f"[LEARNING] Goal created: {goal_id} - {topic}")
            
            return {
                "success": True,
                "goal_id": goal_id,
                "topic": topic,
                "message": "Learning goal created",
            }
            
        except Exception as e:
            logger.error(f"[LEARNING] Create goal failed: {e}")
            return {"success": False, "error": str(e)}
    
    def update_progress(
        self,
        goal_id: str,
        progress: float,
        knowledge_gained: int = 0
    ) -> Dict[str, Any]:
        """Persist progress counters and optionally mark the goal completed."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE learning_goals 
                    SET progress = ?, 
                        knowledge_gained = knowledge_gained + ?,
                        updated_at = ?
                    WHERE id = ?
                """, (progress, knowledge_gained, 
                      datetime.now(timezone.utc).isoformat(), goal_id))
                
                if progress >= 1.0:
                    conn.execute("""
                        UPDATE learning_goals 
                        SET status = 'completed', completed_at = ?
                        WHERE id = ?
                    """, (datetime.now(timezone.utc).isoformat(), goal_id))
                
                conn.commit()
            
            return {"success": True, "progress": progress}
            
        except Exception as e:
            logger.error(f"[LEARNING] Update progress failed: {e}")
            return {"success": False, "error": str(e)}
    
    def get_active_goals(self, session_id: str = "selfing-session") -> List[Dict]:
        """Return open goals for a session."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT * FROM learning_goals 
                    WHERE session_id = ? AND status = 'active'
                    ORDER BY created_at DESC
                """, (session_id,))
                
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"[LEARNING] Get active goals failed: {e}")
            return []
    
    def log_action(
        self,
        goal_id: str,
        action: str,
        content: str,
        result: str,
        tokens_used: int = 0,
        session_id: str = "selfing-session"
    ):
        """Append a row to ``learning_logs``."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO learning_logs 
                    (session_id, goal_id, action, content, result, tokens_used, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (session_id, goal_id, action, content, result, tokens_used,
                      datetime.now(timezone.utc).isoformat()))
                conn.commit()
        except Exception as e:
            logger.warning(f"[LEARNING] Log action failed: {e}")


_knowledge_base = None
_learning_manager = None


def get_knowledge_base(db_path: str = "data.db") -> KnowledgeBase:
    """Return the process-wide ``KnowledgeBase`` instance."""
    global _knowledge_base
    if _knowledge_base is None:
        _knowledge_base = KnowledgeBase(db_path)
    return _knowledge_base


def get_learning_manager(db_path: str = "data.db") -> LearningGoalManager:
    """Return the process-wide ``LearningGoalManager`` instance."""
    global _learning_manager
    if _learning_manager is None:
        _learning_manager = LearningGoalManager(db_path)
    return _learning_manager
