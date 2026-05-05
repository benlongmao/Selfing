#!/usr/bin/env python3
"""
Search long-term memory: conversations, diaries, knowledge, and persona rules.
"""

import logging
import os
import re
import sqlite3
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MemorySearchTool:
    """Cross-source memory search for the agent."""

    def __init__(self, db_path: str = "data.db", workspace_dir: str = "workspace/sandbox"):
        self.db_path = db_path
        self.workspace_dir = Path(workspace_dir)
        self.embedder = None
        self._persona_store = None
        self._self_narrative = None
        self._unified_memory = None

        try:
            from backend.embedder import get_embedder
            self.embedder = get_embedder()
        except Exception as e:
            logger.warning(f"[MEMORY] Embedder not available: {e}")

        try:
            from backend.persona_store import PersonaStore
            self._persona_store = PersonaStore(db_path)
        except Exception as e:
            logger.debug(f"[MEMORY] PersonaStore not available: {e}")

        try:
            from backend.self_narrative import SelfNarrative
            self._self_narrative = SelfNarrative(db_path)
        except Exception as e:
            logger.debug(f"[MEMORY] SelfNarrative not available: {e}")

        try:
            from backend.unified_memory import UnifiedMemoryBus
            self._unified_memory = UnifiedMemoryBus(db_path)
        except Exception as e:
            logger.debug(f"[MEMORY] UnifiedMemoryBus not available: {e}")

    def search_conversations(
        self,
        query: str,
        days: int = 90,
        limit: int = 10,
        include_context: bool = True
    ) -> Dict[str, Any]:
        """
        Search chat history: unified memory bus (when available) plus ``chat_turns`` keyword SQL.
        ``days`` defaults line up with db_cleanup.chat_turns retention.
        """
        results = []
        seen_texts: set = set()

        if self._unified_memory:
            try:
                unified_hits = self._unified_memory.retrieve_for_tool(
                    query=query,
                    session_id="selfing-session",
                    limit=limit,
                )
                for item in unified_hits:
                    sig = item.content[:80]
                    if sig in seen_texts:
                        continue
                    seen_texts.add(sig)
                    results.append({
                        "id": item.source_id,
                        "session_id": item.session_id,
                        "timestamp": item.created_at,
                        "user_input": "",
                        "assistant_output": item.content[:300],
                        "match_in": item.memory_type,
                        "matched_text": item.content[:300],
                        "method": "unified",
                        "source_table": item.source_table,
                        "score": round(float(item.score), 4),
                    })
            except Exception as e:
                logger.debug(f"[MEMORY] Unified conversation search failed: {e}")

        # Supplement: SQL keyword hits on chat_turns
        try:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.execute("""
                    SELECT id, session_id, user_input, assistant_output, created_at
                    FROM chat_turns
                    WHERE created_at >= ?
                      AND (user_input LIKE ? OR assistant_output LIKE ?)
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (cutoff, f"%{query}%", f"%{query}%", limit))
                for row in cur.fetchall():
                    sig = (row["user_input"] or "")[:80]
                    if sig in seen_texts:
                        continue
                    seen_texts.add(sig)
                    match = {
                        "id": row["id"],
                        "session_id": row["session_id"],
                        "timestamp": row["created_at"],
                        "user_input": row["user_input"][:200] if row["user_input"] else "",
                        "assistant_output": row["assistant_output"][:300] if row["assistant_output"] else "",
                        "match_in": "user" if query.lower() in (row["user_input"] or "").lower() else "assistant",
                        "method": "keyword",
                    }
                    if query.lower() in (row["user_input"] or "").lower():
                        match["matched_text"] = self._highlight(row["user_input"], query)[:200]
                    else:
                        match["matched_text"] = self._highlight(row["assistant_output"], query)[:200]
                    results.append(match)
        except Exception as e:
            logger.error(f"[MEMORY] Search conversations failed: {e}")

        return {
            "success": True,
            "query": query,
            "results": results[:limit],
            "count": min(len(results), limit),
            "searched_days": days,
        }

    def search_diaries(
        self,
        query: str,
        limit: int = 10
    ) -> Dict[str, Any]:
        """Search markdown diaries under ``diaries/`` and ``autonomous_diaries/``."""
        results: List[Dict] = []

        try:
            diary_dirs = [
                self.workspace_dir / "diaries",
                self.workspace_dir / "autonomous_diaries",
            ]

            for diary_dir in diary_dirs:
                if not diary_dir.exists():
                    continue

                for f in sorted(diary_dir.glob("*.md"), reverse=True):
                    try:
                        content = f.read_text(encoding="utf-8")
                        if query.lower() in content.lower():
                            matched_paragraph = self._extract_matching_paragraph(content, query)

                            results.append({
                                "filename": f.name,
                                "path": str(f.relative_to(self.workspace_dir)),
                                "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
                                "matched_text": matched_paragraph,
                                "size_kb": round(f.stat().st_size / 1024, 1)
                            })

                            if len(results) >= limit:
                                break
                    except Exception as e:
                        logger.debug(f"[MEMORY] Failed to read diary {f}: {e}")

                if len(results) >= limit:
                    break

            return {
                "success": True,
                "query": query,
                "results": results,
                "count": len(results)
            }

        except Exception as e:
            logger.error(f"[MEMORY] Search diaries failed: {e}")
            return {"success": False, "error": str(e), "results": []}

    def search_knowledge(
        self,
        query: str,
        limit: int = 10
    ) -> Dict[str, Any]:
        """Knowledge base: try vector search, then fall back to SQL ``LIKE``."""
        results = []

        try:
            from backend.knowledge_base import KnowledgeBase
            kb = KnowledgeBase(self.db_path)
            semantic_hits = kb.search_knowledge(query, top_k=limit)
            for item in semantic_hits:
                results.append({
                    "id": item.get("id", ""),
                    "title": item.get("title", ""),
                    "content": (item.get("content") or "")[:300],
                    "source": item.get("source", ""),
                    "category": item.get("category", ""),
                    "tags": item.get("tags", ""),
                    "confidence": item.get("confidence", 0.0),
                    "created_at": item.get("created_at", ""),
                    "similarity": round(float(item.get("similarity", 0.0)), 4),
                    "method": "vector",
                })
            if results:
                return {
                    "success": True,
                    "query": query,
                    "results": results,
                    "count": len(results),
                    "method": "vector",
                }
        except Exception as e:
            logger.debug(f"[MEMORY] Semantic knowledge search failed, falling back: {e}")

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.execute("""
                    SELECT id, title, content, source, category, tags, confidence, created_at
                    FROM knowledge_items
                    WHERE title LIKE ? OR content LIKE ? OR tags LIKE ?
                    ORDER BY usefulness_score DESC, created_at DESC
                    LIMIT ?
                """, (f"%{query}%", f"%{query}%", f"%{query}%", limit))
                for row in cur.fetchall():
                    results.append({
                        "id": row["id"],
                        "title": row["title"],
                        "content": row["content"][:300] if row["content"] else "",
                        "source": row["source"],
                        "category": row["category"],
                        "tags": row["tags"],
                        "confidence": row["confidence"],
                        "created_at": row["created_at"],
                        "method": "keyword",
                    })
            return {
                "success": True,
                "query": query,
                "results": results,
                "count": len(results),
                "method": "keyword",
            }
        except Exception as e:
            logger.error(f"[MEMORY] Search knowledge failed: {e}")
            return {"success": False, "error": str(e), "results": []}

    def search_rules(
        self,
        query: str,
        include_inactive: bool = False,
        limit: int = 20
    ) -> Dict[str, Any]:
        """Persona rules: FAISS when available, else SQL ``LIKE`` on ``persona_items``."""
        results = []

        if self._persona_store and not include_inactive:
            try:
                hits = self._persona_store.search_top_k(query, k=limit)
                for item, sim in hits:
                    results.append({
                        "id": item.id,
                        "text": item.text,
                        "category": getattr(item, "category", None),
                        "score": item.score,
                        "is_core": bool(getattr(item, "is_core", 0)),
                        "status": item.status,
                        "created_at": item.created_at,
                        "similarity": round(float(sim), 4),
                    })
                if results:
                    return {
                        "success": True,
                        "query": query,
                        "results": results,
                        "count": len(results),
                        "method": "vector",
                    }
            except Exception as e:
                logger.debug(f"[MEMORY] Vector rule search failed, falling back: {e}")

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                status_filter = "" if include_inactive else "AND status = 'active'"
                cur = conn.execute(f"""
                    SELECT id, text, category, score, is_core, status, created_at
                    FROM persona_items
                    WHERE text LIKE ? {status_filter}
                    ORDER BY score DESC, created_at DESC
                    LIMIT ?
                """, (f"%{query}%", limit))
                for row in cur.fetchall():
                    results.append({
                        "id": row["id"],
                        "text": row["text"],
                        "category": row["category"],
                        "score": row["score"],
                        "is_core": bool(row["is_core"]),
                        "status": row["status"],
                        "created_at": row["created_at"],
                    })
            return {
                "success": True,
                "query": query,
                "results": results,
                "count": len(results),
                "method": "keyword",
            }
        except Exception as e:
            logger.error(f"[MEMORY] Search rules failed: {e}")
            return {"success": False, "error": str(e), "results": []}

    def recall(
        self,
        query: str,
        memory_types: Optional[List[str]] = None,
        limit: int = 5
    ) -> Dict[str, Any]:
        """
        One-shot search over selected memory types and optional compiled wiki.
        *memory_types*: any of conversations, diaries, knowledge, rules.
        """
        if memory_types is None:
            memory_types = ["conversations", "diaries", "knowledge", "rules"]

        results: Dict = {
            "query": query,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "memories": {}
        }

        if "conversations" in memory_types:
            results["memories"]["conversations"] = self.search_conversations(query, limit=limit)

        if "diaries" in memory_types:
            results["memories"]["diaries"] = self.search_diaries(query, limit=limit)

        if "knowledge" in memory_types:
            results["memories"]["knowledge"] = self.search_knowledge(query, limit=limit)

        if "rules" in memory_types:
            results["memories"]["rules"] = self.search_rules(query, limit=limit)

        try:
            from backend.knowledge_compiler import get_compiler
            wiki_hits = get_compiler().search_wiki(query, limit=3)
            if wiki_hits:
                results["memories"]["wiki"] = {
                    "success": True,
                    "count": len(wiki_hits),
                    "results": wiki_hits,
                }
        except Exception as e:
            logger.debug(f"[MEMORY] Wiki search skipped: {e}")

        total_found = sum(
            m.get("count", 0)
            for m in results["memories"].values()
            if isinstance(m, dict)
        )
        results["total_found"] = total_found
        results["success"] = True

        return results

    def summarize_chat_turns_for_calendar_day(
        self,
        session_id: str,
        calendar_date: Optional[str] = None,
        relative: Optional[str] = None,
        max_snippets: int = 15,
    ) -> Dict[str, Any]:
        """
        Summarize ``chat_turns`` for one local calendar day (authoritative), without keyword LIKE.
        Fixes false empty results when ``recall_memory`` only substring-matches (e.g. “yesterday”).
        """
        from backend.config import config
        from backend.chat_turn_calendar import (
            effective_max_snippets,
            get_calendar_tzinfo,
            local_date_from_utc,
            try_resolve_calendar_date_string,
            utc_window_for_local_calendar_day,
        )

        rel = (relative or "").strip().lower()
        cal = (calendar_date or "").strip()
        if not cal and not rel:
            return {
                "success": False,
                "error": "Provide calendar_date (ISO or natural language) or relative (yesterday|today).",
            }
        if cal and rel:
            return {
                "success": False,
                "error": "Pass only one of calendar_date or relative, not both.",
            }

        tz_name_cfg = os.environ.get("S_CHAT_TURNS_CALENDAR_TZ") or config.get(
            "memory.chat_turns_calendar_timezone", "UTC"
        )
        tz, tz_effective = get_calendar_tzinfo(str(tz_name_cfg))
        ref_utc = datetime.now(timezone.utc)

        resolved_from: str
        local_day: date
        if rel:
            if rel not in ("yesterday", "today"):
                return {"success": False, "error": "relative only supports yesterday or today"}
            ref_local = local_date_from_utc(ref_utc, tz)
            if rel == "today":
                local_day = ref_local
            else:
                local_day = ref_local - timedelta(days=1)
            resolved_from = rel
        else:
            parsed = try_resolve_calendar_date_string(cal, ref_utc=ref_utc, tz=tz)
            if not parsed.get("success"):
                return {
                    "success": False,
                    "error": parsed.get("error", "Could not parse calendar_date"),
                    "original": parsed.get("original", cal),
                }
            local_day = parsed["local_date"]
            resolved_from = str(parsed.get("resolved_from", "parsed"))

        target_iso = local_day.isoformat()
        start_utc, end_utc = utc_window_for_local_calendar_day(local_day, tz)
        start_s = start_utc.isoformat()
        end_s = end_utc.isoformat()

        sid = (session_id or "").strip() or "selfing-session"
        auto_expand = bool(
            config.get("memory.chat_turns_summary_auto_expand_snippets", True)
        )
        req_ms = max(1, min(30, int(max_snippets or 15)))

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.execute(
                    """
                    SELECT COUNT(*) AS n,
                           MIN(created_at) AS first_at,
                           MAX(created_at) AS last_at
                    FROM chat_turns
                    WHERE session_id = ? AND created_at >= ? AND created_at < ?
                    """,
                    (sid, start_s, end_s),
                )
                row = cur.fetchone()
                n = int(row["n"] or 0)
                first_at = row["first_at"]
                last_at = row["last_at"]

                if auto_expand:
                    ms_eff = effective_max_snippets(req_ms, n)
                else:
                    ms_eff = min(req_ms, n) if n else req_ms

                snippets: List[Dict[str, Any]] = []
                nearby_days: List[Dict[str, Any]] = []
                if n > 0 and ms_eff > 0:
                    cur2 = conn.execute(
                        """
                        SELECT created_at, user_input, assistant_output
                        FROM chat_turns
                        WHERE session_id = ? AND created_at >= ? AND created_at < ?
                        ORDER BY created_at ASC
                        LIMIT ?
                        """,
                        (sid, start_s, end_s, ms_eff),
                    )
                    for r in cur2.fetchall():
                        u = (r["user_input"] or "").strip()
                        a = (r["assistant_output"] or "").strip()
                        snippets.append(
                            {
                                "created_at": r["created_at"],
                                "user_preview": (u[:160] + "…") if len(u) > 160 else u,
                                "assistant_preview": (a[:200] + "…") if len(a) > 200 else a,
                            }
                        )
                elif n == 0:
                    for delta in (-1, 1):
                        nd = local_day + timedelta(days=delta)
                        ns, ne = utc_window_for_local_calendar_day(nd, tz)
                        cur3 = conn.execute(
                            """
                            SELECT COUNT(*) AS cnt FROM chat_turns
                            WHERE session_id = ? AND created_at >= ? AND created_at < ?
                            """,
                            (sid, ns.isoformat(), ne.isoformat()),
                        )
                        cnt = int(cur3.fetchone()[0] or 0)
                        if cnt > 0:
                            nearby_days.append(
                                {"calendar_date": nd.isoformat(), "turn_count": cnt}
                            )

            tz_note = (
                f"Local calendar day uses timezone «{tz_effective}»; "
                f"created_at is UTC; query window [{start_s}, {end_s})."
            )
            return {
                "success": True,
                "session_id": sid,
                "calendar_date": target_iso,
                "calendar_timezone_effective": tz_effective,
                "resolved_from": resolved_from,
                "turn_count": n,
                "first_turn_at": first_at,
                "last_turn_at": last_at,
                "snippets": snippets,
                "snippets_requested": req_ms,
                "snippets_returned_cap": ms_eff if n else req_ms,
                "snippets_truncated": n > len(snippets),
                "nearby_days_with_activity": nearby_days,
                "note": tz_note
                + " For busy days, pass a larger max_snippets (max 30). "
                "By default snippet count auto-expands unless memory.chat_turns_summary_auto_expand_snippets is false.",
            }
        except Exception as e:
            logger.error("[MEMORY] summarize_chat_turns_for_calendar_day failed: %s", e)
            return {"success": False, "error": str(e)}

    def get_recent_memories(self, hours: int = 24, limit: int = 20) -> Dict[str, Any]:
        """Recent chat snippets, optional autonomous log rows, and new active rules."""
        results: Dict = {
            "conversations": [],
            "autonomous_actions": [],
            "new_rules": []
        }

        try:
            cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()

            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row

                cur = conn.execute("""
                    SELECT user_input, assistant_output, created_at
                    FROM chat_turns
                    WHERE created_at >= ?
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (cutoff, limit))

                for row in cur.fetchall():
                    results["conversations"].append({
                        "user": row["user_input"][:100] if row["user_input"] else "",
                        "assistant": row["assistant_output"][:150] if row["assistant_output"] else "",
                        "time": row["created_at"]
                    })

                try:
                    cur = conn.execute("""
                        SELECT action_type, action_name, created_at
                        FROM autonomous_actions_log
                        WHERE execution_started >= ?
                        ORDER BY execution_started DESC
                        LIMIT ?
                    """, (cutoff, 10))

                    for row in cur.fetchall():
                        results["autonomous_actions"].append({
                            "type": row["action_type"],
                            "name": row["action_name"],
                            "time": row["created_at"]
                        })
                except Exception:
                    pass

                cur = conn.execute("""
                    SELECT text, category, created_at
                    FROM persona_items
                    WHERE created_at >= ? AND status = 'active'
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (cutoff, 5))

                for row in cur.fetchall():
                    results["new_rules"].append({
                        "text": row["text"][:100],
                        "category": row["category"],
                        "time": row["created_at"]
                    })

            results["success"] = True
            results["hours"] = hours

        except Exception as e:
            logger.error(f"[MEMORY] Get recent memories failed: {e}")
            results["success"] = False
            results["error"] = str(e)

        return results

    def _highlight(self, text: str, query: str) -> str:
        """Wrap the first case-insensitive match in **bold**."""
        if not text or not query:
            return text or ""

        pattern = re.compile(re.escape(query), re.IGNORECASE)
        return pattern.sub(f"**{query}**", text)

    def _extract_matching_paragraph(self, content: str, query: str) -> str:
        """Return a short excerpt around the first query hit."""
        paragraphs = content.split("\n\n")

        for p in paragraphs:
            if query.lower() in p.lower():
                return self._highlight(p[:300], query)

        for line in content.split("\n"):
            if query.lower() in line.lower():
                return self._highlight(line[:200], query)

        return content[:200]

    def get_tool_definitions(self) -> List[Dict]:
        """OpenAI tool definitions for recall and recent context."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "recall_memory",
                    "description": (
                        "Primary long-term memory search. Covers unified bus + vector/keyword history, diaries, "
                        "knowledge, and persona rules. Narrow memory_types to save tokens. "
                        "Use when the user asks about prior sessions (e.g. “before / last time / do you remember”), "
                        "needs historical context, or you must verify past actions.\n\n"
                        "**Important:** For “did we talk yesterday / on May 1 / that day” or turn counts by calendar day, "
                        "call **get_chat_turns_day_summary** first. The conversations path here is substring-based and "
                        "can return zero rows for date-style questions."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Query string or theme (any language)."
                            },
                            "memory_types": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": ["conversations", "diaries", "knowledge", "rules"]
                                },
                                "description": "Which stores to search; omit to search all four."
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Per-type result cap (default 5, max 20)."
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_recent_context",
                    "description": (
                        "Short recent context: latest chats, autonomous actions, new rules, within *hours*."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "hours": {
                                "type": "integer",
                                "description": "How far back in hours (default 24)."
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_chat_turns_day_summary",
                    "description": (
                        "**Timeline (authoritative)** Summarize ``chat_turns`` for one local calendar day.\n\n"
                        "Prefer this when the user asks whether you spoke on a given day, how many turns, or what "
                        "you talked about *that calendar day*—do not rely on recall_memory alone for that.\n"
                        "Returns turn_count, first/last timestamps, and short previews. The server parses many "
                        "natural date strings; on zero turns, nearby_days_with_activity may list adjacent days.\n\n"
                        "Use recall_memory afterward for keyword/topic drill-down."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "calendar_date": {
                                "type": "string",
                                "description": (
                                    "A calendar day: YYYY-MM-DD, or natural text such as “May 1”, “May 1, 2026”, "
                                    "“5/1”, etc. (server-side parse). Mutually exclusive with relative_day."
                                ),
                            },
                            "relative_day": {
                                "type": "string",
                                "enum": ["yesterday", "today"],
                                "description": "Shorthand for yesterday/today in the configured calendar timezone.",
                            },
                            "max_snippets": {
                                "type": "integer",
                                "description": (
                                    "Max dialogue snippets to return (default 15, max 30). When many turns exist and "
                                    "this is ≥15, the server may auto-raise toward 30 unless auto-expand is disabled in config."
                                ),
                            },
                        },
                        "required": [],
                    },
                }
            },
        ]


_instance: Optional[MemorySearchTool] = None


def get_memory_search_tool(db_path: str = "data.db") -> MemorySearchTool:
    global _instance
    if _instance is None:
        _instance = MemorySearchTool(db_path)
    return _instance
