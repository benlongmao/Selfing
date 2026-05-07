#!/usr/bin/env python3
"""
[2026-01-31] Database cleanup module.

Prunes ``chat_turns`` and other large tables on a schedule so the SQLite file does not grow without bound.

Features:
1. Drop ``chat_turns`` rows older than N days (per policy / min-count guardrails).
2. Drop expired ``sensory_memory_temp`` rows.
3. Drop old ``prompt_logs`` rows.
4. Manual and scheduled entry points.

[P2 memory policy] Long-term ``self_biography`` writes and triggers:
- This module INSERTs into ``self_biography`` during ``cleanup_chat_turns``: a non-LLM summary is generated before old rows are deleted.
- ``retrieve_related_memory`` only reads ``self_biography`` for conversational recall, so ``db_cleanup`` must run regularly.
- Triggers: (1) on startup: after ``app.py`` loads, if ``db_cleanup.enabled`` and ``db_cleanup.run_on_startup``; (2) scheduled: ``scheduled_tasks.setup_db_cleanup_task`` can register a daily job (e.g. 03:00) calling ``run_scheduled_cleanup``.
- If cleanup rarely runs, ``self_biography`` stays sparse and long-horizon recall weakens; run at least daily in production.
- This module does not touch ``persona_items``; tiered rule cleanup lives in ``memory_cleaner`` (``backend/memory_cleaner.py``).
"""
import sqlite3
import logging
import os
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, List
from pathlib import Path

from backend.config import config

logger = logging.getLogger(__name__)


class DatabaseCleaner:
    """Prune large SQLite tables on a schedule (chat turns, temp sensory rows, logs, events)."""

    # Defaults when YAML keys are absent (overridden by ``db_cleanup.*`` in config).
    DEFAULT_CONFIG = {
        "chat_turns": {
            "enabled": True,
            "keep_days": 30,           # keep rows from the last N UTC days
            "keep_min_count": 100,   # per session_id, retain at least this many newest rows
        },
        "sensory_memory_temp": {
            "enabled": True,
            "keep_hours": 48,          # rolling window for ephemeral sensory buffer rows
        },
        "prompt_logs": {
            "enabled": True,
            "keep_days": 7,
        },
        "events": {
            "enabled": True,
            "keep_days": 14,
        },
    }
    
    def __init__(self, db_path: str = "data.db"):
        self.db_path = db_path

        # Load retention knobs from ``config`` (see ``db_cleanup`` in settings.yaml).
        self.config = {
            "chat_turns": {
                "enabled": config.get("db_cleanup.chat_turns.enabled", True),
                "keep_days": int(config.get("db_cleanup.chat_turns.keep_days", 30) or 30),
                "keep_min_count": int(config.get("db_cleanup.chat_turns.keep_min_count", 100) or 100),
            },
            "sensory_memory_temp": {
                "enabled": config.get("db_cleanup.sensory_memory_temp.enabled", True),
                "keep_hours": int(config.get("db_cleanup.sensory_memory_temp.keep_hours", 48) or 48),
            },
            "prompt_logs": {
                "enabled": config.get("db_cleanup.prompt_logs.enabled", True),
                "keep_days": int(config.get("db_cleanup.prompt_logs.keep_days", 7) or 7),
            },
            "events": {
                "enabled": config.get("db_cleanup.events.enabled", True),
                "keep_days": int(config.get("db_cleanup.events.keep_days", 14) or 14),
            },
        }
        
        logger.info(f"DatabaseCleaner initialized: {db_path}")
    
    def _generate_conversation_summary(self, messages: List[Dict]) -> str:
        """
        Build a compact, non-LLM summary for rows about to be deleted from ``chat_turns``.

        English-first text is stored into ``self_biography`` for later semantic recall.
        Token extraction keeps Chinese morphemes plus Latin words; stoplists cover both.
        """
        if not messages:
            return ""

        import re
        from collections import Counter

        user_count = sum(1 for m in messages if m.get("role") == "user")

        all_text = " ".join(
            (m.get("user_input", "") or "") + " " + (m.get("assistant_output", "") or "")
            for m in messages
        )

        # Chinese function words + common English glue words (substring tokens for recall, not NLP perfection).
        stopwords = {
            "的", "了", "是", "在", "我", "你", "有", "这", "个", "和", "就", "不", "也", "都",
            "要", "会", "可以", "能", "吗", "呢", "吧", "啊", "哦", "嗯", "好", "对", "那", "什么",
            "怎么", "请", "帮",
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "to", "of", "and",
            "or", "for", "with", "on", "in", "at", "by", "from", "as", "it", "this", "that", "these",
            "those", "you", "we", "they", "he", "she", "i", "me", "my", "your", "our", "their", "not",
            "no", "yes", "do", "does", "did", "so", "if", "but", "than", "then", "there", "here",
        }
        words = re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z]+", all_text.lower())
        filtered = [w for w in words if w not in stopwords and len(w) > 1]
        word_counts = Counter(filtered)
        top_words = [word for word, _ in word_counts.most_common(5)]
        topics_str = ", ".join(top_words) if top_words else "general conversation"

        timestamps = [m.get("created_at", "") for m in messages if m.get("created_at")]
        time_range = ""
        if len(timestamps) >= 2:
            time_range = f" ({timestamps[0][:10]} ~ {timestamps[-1][:10]})"
        elif timestamps:
            time_range = f" ({timestamps[0][:10]})"

        return (
            f"[Conversation summary{time_range}] spans {user_count} user message(s); "
            f"themes / salient tokens: {topics_str}"
        )
    
    def _save_summary_to_biography(self, conn: sqlite3.Connection, session_id: str, summary: str, messages: List[Dict]) -> bool:
        """
        Persist the archive summary into ``self_biography`` (first-class long-horizon recall).

        ``retrieve_related_memory`` reads this table for conversational recall, so summaries
        must land here—not only in raw ``chat_turns``—before old turns are deleted.
        """
        import uuid
        import numpy as np

        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS self_biography (
                    id TEXT PRIMARY KEY,
                    session_id TEXT,
                    content TEXT NOT NULL,
                    emotion_tag TEXT,
                    significance REAL,
                    created_at TEXT NOT NULL,
                    embedding BLOB
                )
            """)

            memory_id = f"bio-archive-{uuid.uuid4().hex[:8]}"

            # Anchor the biography row to the earliest pruned turn (keeps chronological flavor).
            created_at = messages[0].get("created_at") if messages else datetime.now(timezone.utc).isoformat()

            embedding_blob = None
            try:
                from backend.embedder import get_embedder
                embedder = get_embedder()
                vec = embedder.encode(summary)
                if vec is not None:
                    embedding_blob = vec.astype(np.float32).tobytes()
            except Exception as emb_err:
                logger.warning(f"[DB-CLEANUP] Failed to embed summary: {emb_err}")

            conn.execute("""
                INSERT INTO self_biography (id, session_id, content, emotion_tag, significance, created_at, embedding)
                VALUES (?, ?, ?, 'archived', 0.6, ?, ?)
            """, (memory_id, session_id, summary, created_at, embedding_blob))
            
            logger.info(f"[DB-CLEANUP] ✅ Saved conversation summary to self_biography: {memory_id}")
            return True
            
        except Exception as e:
            logger.warning(f"[DB-CLEANUP] Failed to save summary to biography: {e}")
            return False
    
    def cleanup_chat_turns(self, dry_run: bool = False, preserve_summaries: bool = True) -> Dict:
        """
        Prune ``chat_turns`` while optionally archiving a non-LLM summary per ``session_id``.

        Flow:
        1. Select rows older than ``keep_days`` that exceed ``keep_min_count`` per session.
        2. Group by ``session_id`` and synthesize a compact summary.
        3. Insert the summary into ``self_biography`` (when ``preserve_summaries``).
        4. Delete the pruned ``chat_turns`` rows.

        Args:
            dry_run: when ``True``, only report counts—no deletes.
            preserve_summaries: when ``True`` (default), write biography rows before DELETE.

        Returns:
            Status dict (``deleted``, ``summaries_saved``, errors, etc.).
        """
        cfg = self.config["chat_turns"]
        if not cfg["enabled"]:
            return {"status": "disabled", "deleted": 0}
        
        keep_days = cfg["keep_days"]
        keep_min_count = cfg["keep_min_count"]
        cutoff_date = (datetime.now(timezone.utc) - timedelta(days=keep_days)).isoformat()
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row

                cur = conn.execute("""
                    SELECT t.* FROM chat_turns t
                    WHERE t.created_at < ?
                    AND (
                        SELECT COUNT(*) FROM chat_turns t2 
                        WHERE t2.session_id = t.session_id
                    ) > ?
                    AND t.id NOT IN (
                        SELECT id FROM (
                            SELECT id FROM chat_turns t3
                            WHERE t3.session_id = t.session_id
                            ORDER BY t3.created_at DESC
                            LIMIT ?
                        )
                    )
                    ORDER BY t.session_id, t.created_at
                """, (cutoff_date, keep_min_count, keep_min_count))
                
                rows = cur.fetchall()

                if not rows:
                    return {"status": "success", "deleted": 0, "summaries_saved": 0, "message": "No old records to clean"}

                from collections import defaultdict
                sessions_to_clean = defaultdict(list)
                for row in rows:
                    sessions_to_clean[row["session_id"]].append(dict(row))
                
                if dry_run:
                    return {
                        "status": "dry_run",
                        "would_delete": len(rows),
                        "sessions_affected": len(sessions_to_clean),
                        "cutoff_date": cutoff_date,
                    }
                
                summaries_saved = 0
                if preserve_summaries:
                    for session_id, messages in sessions_to_clean.items():
                        if len(messages) >= 2:  # skip trivial single-row groups
                            summary = self._generate_conversation_summary(messages)
                            if summary and self._save_summary_to_biography(conn, session_id, summary, messages):
                                summaries_saved += 1

                ids_to_delete = [row["id"] for row in rows]
                placeholders = ",".join(["?" for _ in ids_to_delete])
                conn.execute(f"DELETE FROM chat_turns WHERE id IN ({placeholders})", ids_to_delete)
                deleted = conn.total_changes
                conn.commit()
                
                logger.info(f"[DB-CLEANUP] Cleaned {deleted} old chat_turns records, saved {summaries_saved} summaries")
                
                return {
                    "status": "success",
                    "deleted": deleted,
                    "summaries_saved": summaries_saved,
                    "sessions_affected": len(sessions_to_clean),
                    "cutoff_date": cutoff_date,
                    "keep_days": keep_days,
                }
                
        except Exception as e:
            logger.error(f"[DB-CLEANUP] Failed to clean chat_turns: {e}")
            return {"status": "error", "error": str(e)}
    
    def cleanup_sensory_memory(self, dry_run: bool = False) -> Dict:
        """Delete expired rows from ``sensory_memory_temp``."""
        cfg = self.config["sensory_memory_temp"]
        if not cfg["enabled"]:
            return {"status": "disabled", "deleted": 0}
        
        keep_hours = cfg["keep_hours"]
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=keep_hours)).isoformat()
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                if dry_run:
                    cur = conn.execute(
                        "SELECT COUNT(*) FROM sensory_memory_temp WHERE timestamp < ?",
                        (cutoff,)
                    )
                    would_delete = cur.fetchone()[0]
                    return {"status": "dry_run", "would_delete": would_delete}
                
                conn.execute("DELETE FROM sensory_memory_temp WHERE timestamp < ?", (cutoff,))
                deleted = conn.total_changes
                conn.commit()
                
                logger.info(f"[DB-CLEANUP] Cleaned {deleted} sensory_memory_temp records")
                return {"status": "success", "deleted": deleted}
                
        except sqlite3.OperationalError as e:
            if "no such table" in str(e):
                return {"status": "skipped", "reason": "table not exists"}
            raise
        except Exception as e:
            logger.error(f"[DB-CLEANUP] Failed to clean sensory_memory_temp: {e}")
            return {"status": "error", "error": str(e)}
    
    def cleanup_prompt_logs(self, dry_run: bool = False) -> Dict:
        """Delete aged rows from ``prompt_logs``."""
        cfg = self.config["prompt_logs"]
        if not cfg["enabled"]:
            return {"status": "disabled", "deleted": 0}
        
        keep_days = cfg["keep_days"]
        cutoff = (datetime.now(timezone.utc) - timedelta(days=keep_days)).isoformat()
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                if dry_run:
                    cur = conn.execute(
                        "SELECT COUNT(*) FROM prompt_logs WHERE created_at < ?",
                        (cutoff,)
                    )
                    would_delete = cur.fetchone()[0]
                    return {"status": "dry_run", "would_delete": would_delete}
                
                conn.execute("DELETE FROM prompt_logs WHERE created_at < ?", (cutoff,))
                deleted = conn.total_changes
                conn.commit()
                
                logger.info(f"[DB-CLEANUP] Cleaned {deleted} prompt_logs records")
                return {"status": "success", "deleted": deleted}
                
        except sqlite3.OperationalError as e:
            if "no such table" in str(e):
                return {"status": "skipped", "reason": "table not exists"}
            raise
        except Exception as e:
            logger.error(f"[DB-CLEANUP] Failed to clean prompt_logs: {e}")
            return {"status": "error", "error": str(e)}
    
    def cleanup_events(self, dry_run: bool = False) -> Dict:
        """Delete aged rows from ``events``."""
        cfg = self.config["events"]
        if not cfg["enabled"]:
            return {"status": "disabled", "deleted": 0}
        
        keep_days = cfg["keep_days"]
        cutoff = (datetime.now(timezone.utc) - timedelta(days=keep_days)).isoformat()
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                if dry_run:
                    cur = conn.execute(
                        "SELECT COUNT(*) FROM events WHERE timestamp < ?",
                        (cutoff,)
                    )
                    would_delete = cur.fetchone()[0]
                    return {"status": "dry_run", "would_delete": would_delete}
                
                conn.execute("DELETE FROM events WHERE timestamp < ?", (cutoff,))
                deleted = conn.total_changes
                conn.commit()
                
                logger.info(f"[DB-CLEANUP] Cleaned {deleted} events records")
                return {"status": "success", "deleted": deleted}
                
        except sqlite3.OperationalError as e:
            if "no such table" in str(e):
                return {"status": "skipped", "reason": "table not exists"}
            raise
        except Exception as e:
            logger.error(f"[DB-CLEANUP] Failed to clean events: {e}")
            return {"status": "error", "error": str(e)}
    
    def cleanup_all(self, dry_run: bool = False) -> Dict:
        """Run every enabled cleaner (chat turns, sensory temp, prompt logs, events)."""
        results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dry_run": dry_run,
            "tables": {}
        }
        
        results["tables"]["chat_turns"] = self.cleanup_chat_turns(dry_run)
        results["tables"]["sensory_memory_temp"] = self.cleanup_sensory_memory(dry_run)
        results["tables"]["prompt_logs"] = self.cleanup_prompt_logs(dry_run)
        results["tables"]["events"] = self.cleanup_events(dry_run)

        total_deleted = sum(
            r.get("deleted", 0) for r in results["tables"].values()
        )
        results["total_deleted"] = total_deleted
        
        if not dry_run:
            logger.info(f"[DB-CLEANUP] Cleanup completed: {total_deleted} records deleted")
        
        return results
    
    def get_table_stats(self) -> Dict:
        """Return row counts (and DB file size) for the main operational tables."""
        stats = {}
        
        tables = [
            "chat_turns",
            "sensory_memory_temp", 
            "prompt_logs",
            "events",
            "memories",
            "persona_items",
            "self_state",
        ]
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                for table in tables:
                    try:
                        cur = conn.execute(f"SELECT COUNT(*) FROM {table}")
                        count = cur.fetchone()[0]
                        stats[table] = {"count": count}
                    except sqlite3.OperationalError:
                        stats[table] = {"count": 0, "note": "table not exists"}
                
                if os.path.exists(self.db_path):
                    stats["_db_size_mb"] = os.path.getsize(self.db_path) / 1024 / 1024
                
        except Exception as e:
            logger.error(f"[DB-CLEANUP] Failed to get stats: {e}")
            stats["_error"] = str(e)
        
        return stats


# --- module-level singleton -------------------------------------------------
_cleaner_instance: Optional[DatabaseCleaner] = None


def get_db_cleaner(db_path: str = "data.db") -> DatabaseCleaner:
    """Return the process-wide ``DatabaseCleaner`` (lazy init)."""
    global _cleaner_instance
    if _cleaner_instance is None:
        _cleaner_instance = DatabaseCleaner(db_path)
    return _cleaner_instance


def run_scheduled_cleanup(db_path: str = "data.db") -> Dict:
    """Entry point for cron / ``scheduled_tasks`` — runs ``cleanup_all`` for real."""
    cleaner = get_db_cleaner(db_path)
    return cleaner.cleanup_all(dry_run=False)


# --- CLI --------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="S / Self-becoming SQLite maintenance utility")
    parser.add_argument("--db", default="data.db", help="Path to SQLite database file")
    parser.add_argument("--dry-run", action="store_true", help="Print counts only; do not DELETE")
    parser.add_argument("--stats", action="store_true", help="Print per-table row counts")
    parser.add_argument("--table", choices=["chat_turns", "sensory", "prompt_logs", "events", "all"],
                        default="all", help="Which cleaner to run")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    cleaner = DatabaseCleaner(args.db)
    
    if args.stats:
        print("\nDatabase statistics")
        print("-" * 40)
        stats = cleaner.get_table_stats()
        for table, info in stats.items():
            if table.startswith("_"):
                continue
            count = info.get("count", "N/A")
            note = info.get("note", "")
            suffix = f" ({note})" if note else ""
            print(f"  {table}: {count} rows{suffix}")
        if "_db_size_mb" in stats:
            print(f"\n  Database file: {stats['_db_size_mb']:.2f} MB")
        print()
    else:
        mode = "dry-run" if args.dry_run else "execute"
        print(f"\nDatabase cleanup ({mode})")
        print("-" * 40)

        if args.table == "all":
            results = cleaner.cleanup_all(dry_run=args.dry_run)
            for table, result in results["tables"].items():
                status = result.get("status", "unknown")
                deleted = result.get("deleted", result.get("would_delete", 0))
                verb = "would delete" if args.dry_run else "deleted"
                print(f"  {table}: {status}, {verb} {deleted} row(s)")
            print(f"\n  Total rows affected (deleted counter): {results['total_deleted']}")
        else:
            table_map = {
                "chat_turns": cleaner.cleanup_chat_turns,
                "sensory": cleaner.cleanup_sensory_memory,
                "prompt_logs": cleaner.cleanup_prompt_logs,
                "events": cleaner.cleanup_events,
            }
            result = table_map[args.table](dry_run=args.dry_run)
            print(f"  Result: {result}")

        print()
