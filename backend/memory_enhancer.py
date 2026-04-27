#!/usr/bin/env python3
"""
Memory enhancer — bridge between ``MemoryImportanceEvaluator`` and spaced repetition.

Responsibilities:
1. Pull high-salience candidates from recent ``chat_turns``.
2. Persist items via ``SpacedRepetitionEngine`` (JSON under ``data/``).
3. Record review activity into the autonomy summary store when configured.
4. Expose a daily review hook for the scheduler / resting pulse.
"""

import logging
import sqlite3
from datetime import datetime, timezone, timedelta, date
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from backend.config import config
from backend.database import DB_PATH
from backend.memory_importance import MemoryImportanceEvaluator
from backend.spaced_repetition import SpacedRepetitionEngine, MemoryItem
from backend.autonomous_memory import save_autonomy_summary
from backend.s_identity import get_primary_session

logger = logging.getLogger(__name__)


def _spaced_repetition_data_dir(db_path: Optional[str]) -> str:
    """
    Directory for spaced-repetition JSON (sibling ``data/`` next to ``*.db``).

    For ``:memory:`` or empty paths, use ``<project>/data/spaced_repetition_default``.
    """
    if not db_path or (isinstance(db_path, str) and db_path.strip() in ("", ":memory:")):
        root = Path(__file__).resolve().parents[1]
        d = root / "data" / "spaced_repetition_default"
        d.mkdir(parents=True, exist_ok=True)
        return str(d)
    p = Path(db_path).expanduser().resolve()
    if p.suffix == ".db":
        base = p.parent / "data"
    else:
        base = p / "data"
    base.mkdir(parents=True, exist_ok=True)
    return str(base)


def _parse_created_at(raw: Any) -> Optional[datetime]:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s)
    except Exception:
        return None


@dataclass
class EnhancedMemory:
    """High-salience memory slice (not necessarily persisted to SR yet)."""
    memory_id: str
    content: str
    importance_score: float
    source_type: str  # 'conversation', 'diary', 'knowledge', 'rule'
    source_id: str
    created_at: datetime
    last_reviewed: Optional[datetime]
    review_count: int = 0


class MemoryEnhancer:
    """Coordinates importance scoring and spaced-repetition persistence."""
    
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or DB_PATH
        self.importance_evaluator = MemoryImportanceEvaluator(self.db_path)
        self.spaced_engine = SpacedRepetitionEngine(_spaced_repetition_data_dir(self.db_path))

    @property
    def engine(self) -> SpacedRepetitionEngine:
        """Alias for tests: ``engine`` == ``spaced_engine``."""
        return self.spaced_engine

    def _me_cfg(self, key: str, default: Any) -> Any:
        raw = config.get("parameters.memory_enhancer") or {}
        if isinstance(raw, dict) and key in raw:
            return raw.get(key)
        return default

    @staticmethod
    def _stable_spaced_id(source_type: str, source_id: str) -> str:
        safe_type = (source_type or "unknown").replace("/", "_")[:40]
        safe_sid = (source_id or "na").replace("/", "_")[:80]
        return f"sr-{safe_type}-{safe_sid}"[:200]

    def load_high_importance_memories(
        self,
        min_score: Optional[float] = None,
        limit: Optional[int] = None,
        max_age_days: Optional[int] = None,
    ) -> List[EnhancedMemory]:
        """
        Scan recent ``chat_turns`` for the primary session, score with ``MemoryImportanceEvaluator``,
        and return top candidates (does **not** auto-write to spaced repetition).
        """
        min_score = float(min_score if min_score is not None else self._me_cfg("load_min_score", 0.42))
        limit = int(limit if limit is not None else self._me_cfg("load_return_limit", 20))
        max_age_days = int(max_age_days if max_age_days is not None else self._me_cfg("max_age_days", 45))
        scan_limit = int(self._me_cfg("chat_scan_limit", 120))
        limit = max(1, min(200, limit))
        scan_limit = max(20, min(500, scan_limit))

        memories: List[EnhancedMemory] = []
        if not self.db_path or self.db_path == ":memory:":
            return memories

        session_id = get_primary_session()
        cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """
                    SELECT id, user_input, assistant_output, created_at
                    FROM chat_turns
                    WHERE session_id = ?
                      AND (IFNULL(user_input,'') != '' OR IFNULL(assistant_output,'') != '')
                    ORDER BY datetime(created_at) DESC
                    LIMIT ?
                    """,
                    (session_id, scan_limit),
                ).fetchall()
        except Exception as e:
            logger.warning("[MemoryEnhancer] load_high_importance query failed: %s", e)
            return memories

        for row in rows:
            uid = str(row["id"])
            u_in = (row["user_input"] or "").strip()
            a_out = (row["assistant_output"] or "").strip()
            blob = f"[User] {u_in}\n[Assistant] {a_out}".strip()
            if len(blob) < 24:
                continue
            if len(blob) > 4000:
                blob = blob[:3997] + "..."

            created = _parse_created_at(row["created_at"])
            if created is not None and created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            if created is not None and created < cutoff:
                continue

            try:
                score = self.importance_evaluator.evaluate(
                    content=blob,
                    user_input=u_in[:2000],
                    assistant_response=a_out[:2000],
                    created_at=created,
                )
                total = float(score.total_score)
            except Exception as ex:
                logger.debug("[MemoryEnhancer] evaluate skip %s: %s", uid, ex)
                continue

            if total < min_score:
                continue

            memories.append(
                EnhancedMemory(
                    memory_id=f"chat:{uid}",
                    content=blob,
                    importance_score=total,
                    source_type="conversation",
                    source_id=uid,
                    created_at=created or datetime.now(timezone.utc),
                    last_reviewed=None,
                    review_count=0,
                )
            )

        memories.sort(key=lambda m: m.importance_score, reverse=True)
        return memories[:limit]
    
    def add_memory_to_spaced_repetition(
        self,
        content: str,
        importance_score: float,
        source_type: str,
        source_id: str,
        created_at: Optional[datetime] = None,
        category: str = "episodic",
    ) -> str:
        """
        Persist one item into ``SpacedRepetitionEngine`` (``spaced_memory.json``) with a stable id for dedup.
        """
        try:
            stable_id = self._stable_spaced_id(source_type, source_id)
            if stable_id in self.spaced_engine.memory_items:
                return stable_id

            if importance_score > 0.7:
                initial_interval = 1
            elif importance_score > 0.4:
                initial_interval = 3
            else:
                initial_interval = 7

            next_d = date.today() + timedelta(days=initial_interval)
            st = (source_type or "unknown").replace("/", "_")[:40]
            sid = (source_id or "na").replace("/", "_")[:80]
            tags = [f"src:{st}:{sid}"]

            item = MemoryItem(
                id=stable_id,
                content=(content or "")[:8000],
                category=(category or "episodic")[:32],
                importance_score=float(importance_score),
                interval=initial_interval,
                repetitions=0,
                ease_factor=2.5,
                next_review_date=next_d.isoformat(),
                last_review_date=None,
                memory_strength=0.0,
                tags=tags,
            )
            self.spaced_engine.add_memory_item(item)
            logger.info(
                "[MemoryEnhancer] spaced item %s importance=%.2f next=%s",
                stable_id,
                importance_score,
                item.next_review_date,
            )
            return stable_id
        except Exception as e:
            logger.error(f"Failed to add memory to spaced repetition: {e}")
            return ""
    
    def review_due_memories(
        self,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Review up to ``limit`` due items; update SM-2 state and optional autonomy summaries."""
        if session_id is None:
            session_id = get_primary_session()
        try:
            due_items = self.spaced_engine.get_due_items(limit=10)

            if not due_items:
                return {
                    "reviewed": 0,
                    "total_due": 0,
                    "items": [],
                    "next_review_count": len(self.spaced_engine.get_due_items(limit=10_000)),
                }

            reviewed_count = 0
            review_details = []

            for item in due_items:
                quality = 3
                ok, _info = self.spaced_engine.review_memory(item.id, quality)
                if not ok:
                    continue

                if self.db_path and self.db_path != ":memory:":
                    try:
                        save_autonomy_summary(
                            db_path=self.db_path,
                            session_id=session_id,
                            action_type="memory_review",
                            action_name="Spaced repetition review",
                            summary=f"Reviewed memory: {item.content[:50]}...",
                            energy_before=50.0,
                            energy_after=45.0,
                            artifacts=[f"memory:{item.id}"],
                            importance_score=item.importance_score,
                        )
                    except Exception as ex:
                        logger.debug("[MemoryEnhancer] save_autonomy_summary skip: %s", ex)

                reviewed_count += 1
                review_details.append(
                    {
                        "memory_id": item.id,
                        "content_preview": item.content[:100],
                        "importance": item.importance_score,
                        "interval": item.interval,
                    }
                )

            stats = {
                "reviewed": reviewed_count,
                "total_due": len(due_items),
                "items": review_details,
                "next_review_count": len(self.spaced_engine.get_due_items(limit=10_000)),
            }

            logger.info(
                "Reviewed %s due memories, %s still due",
                reviewed_count,
                stats["next_review_count"],
            )
            return stats

        except Exception as e:
            logger.error(f"Failed to review due memories: {e}")
            return {"reviewed": 0, "total_due": 0, "error": str(e)}
    
    def daily_memory_review_task(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Daily job: review due cards, then ingest new high-score chat turns (capped, deduped)."""
        if session_id is None:
            session_id = get_primary_session()
        logger.info("Starting daily memory review task for session: %s", session_id)

        review_stats = self.review_due_memories(session_id)

        # Ingest high-scoring turns into spaced repetition (dedup + daily cap)
        ingest_min = float(self._me_cfg("ingest_min_score", 0.55))
        ingest_cap = int(self._me_cfg("ingest_max_per_day", 12))
        scan_pool = max(ingest_cap * 4, int(self._me_cfg("ingest_scan_pool", 48)))
        candidates = self.load_high_importance_memories(
            min_score=ingest_min,
            limit=min(200, scan_pool),
            max_age_days=int(self._me_cfg("max_age_days", 45)),
        )
        ingested_new = 0
        for m in candidates:
            if ingested_new >= max(0, ingest_cap):
                break
            key = self._stable_spaced_id(m.source_type, m.source_id)
            if key in self.spaced_engine.memory_items:
                continue
            mid = self.add_memory_to_spaced_repetition(
                content=m.content,
                importance_score=m.importance_score,
                source_type=m.source_type,
                source_id=m.source_id,
                created_at=m.created_at,
            )
            if mid:
                ingested_new += 1

        summary = (
            f"Memory review done: reviewed {review_stats.get('reviewed', 0)} item(s), "
            f"{review_stats.get('total_due', 0)} were due; "
            f"new spaced-repetition items: {ingested_new}"
        )

        # Optional autonomy summary row
        if self.db_path and self.db_path != ":memory:":
            try:
                save_autonomy_summary(
                    db_path=self.db_path,
                    session_id=session_id,
                    action_type="scheduled_task",
                    action_name="Daily memory review",
                    summary=summary,
                    energy_before=60.0,
                    energy_after=55.0,
                    artifacts=[f"sr_ingest:{ingested_new}"],
                    importance_score=0.3,
                )
            except Exception as ex:
                logger.warning("[MemoryEnhancer] daily summary save failed: %s", ex)

        reviewed = int(review_stats.get("reviewed", 0) or 0)
        failed = 1 if review_stats.get("error") else 0
        successful = max(0, reviewed - failed)

        return {
            "success": True,
            "review_stats": review_stats,
            "summary": summary,
            "reviewed_count": reviewed,
            "successful_reviews": successful,
            "failed_reviews": failed,
            "ingested_new": ingested_new,
        }

    def background_process(self) -> Dict[str, Any]:
        """Minute tick: count salient candidates, count due items, occasional JSON cleanup."""
        try:
            stats = {
                "memories_loaded": 0,
                "memories_cleaned": 0,
                "due_count": 0,
                "status": "completed",
            }
            
            # Ingestion runs in the daily task; here we only count salient candidates (no SR writes).
            try:
                pool = self.load_high_importance_memories(
                    min_score=float(self._me_cfg("background_min_score", 0.5)),
                    limit=int(self._me_cfg("background_scan_limit", 30)),
                )
                stats["memories_loaded"] = len(pool)
            except Exception as ex:
                logger.debug("[MemoryEnhancer] background load_high skip: %s", ex)

            due_items = self.spaced_engine.get_due_items(limit=10_000)
            stats["due_count"] = len(due_items) if due_items else 0

            current_time = datetime.now(timezone.utc)
            if not hasattr(self, "_last_cleanup"):
                self._last_cleanup = current_time

            if (current_time - self._last_cleanup).total_seconds() > 24 * 3600:
                cleaned = self.spaced_engine.cleanup_old_memories(max_age_days=365)
                stats["memories_cleaned"] = cleaned
                self._last_cleanup = current_time
            
            logger.debug(f"Memory enhancer background process: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Memory enhancer background process failed: {e}")
            return {"error": str(e)}
    
    def get_statistics(self) -> Dict[str, Any]:
        """Aggregate spaced-repetition stats for dashboards / work logs."""
        try:
            raw = self.spaced_engine.get_review_stats()
            items = list(self.spaced_engine.memory_items.values())
            avg_imp = (
                sum(i.importance_score for i in items) / len(items) if items else 0.0
            )
            now_iso = datetime.now(timezone.utc).isoformat()
            enhanced_stats = {
                "total_memories": raw.get("total_memories", 0),
                "due_memories": raw.get("due_for_review", 0),
                "reviewed_today": 0,
                "average_importance": round(avg_imp, 4),
                "avg_importance": round(avg_imp, 4),
                "avg_memory_strength": raw.get("avg_memory_strength", 0.0),
                "by_category": raw.get("by_category", {}),
                "next_review_distribution": raw.get("next_review_distribution", {}),
                "engine": "spaced_repetition_v1",
                "last_updated": now_iso,
                "last_review": now_iso,
            }

            return enhanced_stats

        except Exception as e:
            logger.error(f"Failed to get memory enhancer statistics: {e}")
            return {"error": str(e)}


_memory_enhancer_instance: Optional[MemoryEnhancer] = None


def get_memory_enhancer() -> Optional[MemoryEnhancer]:
    """Return the process-wide singleton if ``init_memory_enhancer`` ran."""
    return _memory_enhancer_instance


def init_memory_enhancer(db_path: Optional[str] = None) -> MemoryEnhancer:
    """Create (or replace) the global ``MemoryEnhancer`` singleton."""
    global _memory_enhancer_instance
    _memory_enhancer_instance = MemoryEnhancer(db_path)
    return _memory_enhancer_instance


def daily_memory_review_entrypoint(session_id: Optional[str] = None) -> Dict[str, Any]:
    """Thin entrypoint for cron / scheduler wiring."""
    enhancer = get_memory_enhancer()
    if not enhancer:
        enhancer = init_memory_enhancer()

    return enhancer.daily_memory_review_task(session_id)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    enhancer = MemoryEnhancer(":memory:")
    print("Memory enhancer initialized successfully")

    memory_id = enhancer.add_memory_to_spaced_repetition(
        content="Smoke test: self-evolution backlog item (S-44).",
        importance_score=0.8,
        source_type="test",
        source_id="test-001",
    )
    print(f"Added memory: {memory_id}")

    stats = enhancer.review_due_memories()
    print(f"Review stats: {stats}")