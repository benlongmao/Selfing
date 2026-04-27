#!/usr/bin/env python3
"""
Output awareness: persist truncation / task-claim metadata so the model can reason about its own outputs.

Targets two failure modes:
1. The model not knowing its last reply was truncated.
2. The model not re-checking work it claims to have finished.

This layer records what happened to the previous assistant output so prompts can surface it.
"""
import sqlite3
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)


class OutputAwareness:
    """SQLite-backed log of truncation and claimed tasks for prompt injection."""

    def __init__(self, db_path: str = "data.db"):
        self.db_path = db_path
        self._ensure_table()

    def _ensure_table(self):
        """Create ``output_awareness`` if missing."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS output_awareness (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        turn_index INTEGER,
                        
                        -- truncation
                        was_truncated BOOLEAN DEFAULT FALSE,
                        original_length INTEGER,
                        truncated_length INTEGER,
                        truncation_reason TEXT,
                        
                        -- claimed work
                        claimed_task TEXT,
                        task_type TEXT,
                        
                        -- verification
                        verified BOOLEAN DEFAULT FALSE,
                        verification_result TEXT,
                        
                        created_at TEXT NOT NULL
                    )
                """)
                conn.commit()
                logger.info("Output awareness table ensured")
        except Exception as e:
            logger.error(f"Failed to ensure output_awareness table: {e}")

    def record_truncation(
        self,
        session_id: str,
        turn_index: int,
        original_length: int,
        truncated_length: int,
        reason: str = "energy_exhaustion"
    ) -> bool:
        """Insert a row marking ``was_truncated`` for this turn."""
        try:
            now = datetime.now(timezone.utc).isoformat()
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO output_awareness 
                    (session_id, turn_index, was_truncated, original_length, 
                     truncated_length, truncation_reason, created_at)
                    VALUES (?, ?, TRUE, ?, ?, ?, ?)
                """, (session_id, turn_index, original_length, truncated_length, reason, now))
                conn.commit()
            logger.info(f"Recorded truncation: {original_length} -> {truncated_length} ({reason})")
            return True
        except Exception as e:
            logger.error(f"Failed to record truncation: {e}")
            return False

    def record_code_task(
        self,
        session_id: str,
        turn_index: int,
        claimed_task: str,
        task_type: str = "code_generation"
    ) -> int:
        """
        Record a claimed code-related task.

        Returns:
            Row id for later verification hooks, or ``-1`` on failure.
        """
        try:
            now = datetime.now(timezone.utc).isoformat()
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute("""
                    INSERT INTO output_awareness 
                    (session_id, turn_index, claimed_task, task_type, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (session_id, turn_index, claimed_task, task_type, now))
                conn.commit()
                return cur.lastrowid
        except Exception as e:
            logger.error(f"Failed to record code task: {e}")
            return -1

    def get_last_truncation(self, session_id: str) -> Optional[Dict]:
        """Latest truncation row for ``session_id``, or ``None``."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute("""
                    SELECT turn_index, original_length, truncated_length, 
                           truncation_reason, created_at
                    FROM output_awareness
                    WHERE session_id = ? AND was_truncated = TRUE
                    ORDER BY created_at DESC
                    LIMIT 1
                """, (session_id,))
                row = cur.fetchone()
                if row:
                    return {
                        "turn_index": row[0],
                        "original_length": row[1],
                        "truncated_length": row[2],
                        "reason": row[3],
                        "timestamp": row[4],
                        "lost_chars": row[1] - row[2] if row[1] and row[2] else 0
                    }
        except Exception as e:
            logger.error(f"Failed to get last truncation: {e}")
        return None

    def get_recent_truncations(self, session_id: str, limit: int = 3) -> List[Dict]:
        """Return up to ``limit`` recent truncation rows (newest first)."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute("""
                    SELECT turn_index, original_length, truncated_length, 
                           truncation_reason, created_at
                    FROM output_awareness
                    WHERE session_id = ? AND was_truncated = TRUE
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (session_id, limit))
                rows = cur.fetchall()
                return [{
                    "turn_index": row[0],
                    "original_length": row[1],
                    "truncated_length": row[2],
                    "reason": row[3],
                    "timestamp": row[4]
                } for row in rows]
        except Exception as e:
            logger.error(f"Failed to get recent truncations: {e}")
        return []

    def generate_truncation_awareness_prompt(self, session_id: str) -> str:
        """
        Build a short system addendum after a truncation (empty string if none / stale).

        Injected near other self-check blocks so the model re-grounds on real artifacts.
        """
        last_truncation = self.get_last_truncation(session_id)
        if not last_truncation:
            return ""

        lost_chars = last_truncation.get("lost_chars", 0)
        if lost_chars <= 0:
            return ""

        # Only surface truncations from roughly the last five minutes
        try:
            from datetime import datetime, timezone
            truncation_time = datetime.fromisoformat(last_truncation["timestamp"].replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            minutes_ago = (now - truncation_time).total_seconds() / 60
            if minutes_ago > 5:
                return ""
        except Exception:
            pass

        reason = last_truncation.get("reason") or "length / token limit"
        return f"""
[Output awareness — truncation]
Your last assistant reply was truncated (about {lost_chars} characters lost).
Reason: {reason}

Reminders:
1. If you were writing code, it may be incomplete.
2. If you claimed a task was done, verify with read_file (or equivalent) before insisting.
3. Do not assume unfinished plans actually completed.
4. When asked "can you X?", check real files/code instead of memory alone.
"""

    def clear_old_records(self, session_id: str, keep_recent: int = 10):
        """Delete older rows for ``session_id``, keeping the ``keep_recent`` newest by ``created_at``."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    DELETE FROM output_awareness
                    WHERE session_id = ? AND id NOT IN (
                        SELECT id FROM output_awareness
                        WHERE session_id = ?
                        ORDER BY created_at DESC
                        LIMIT ?
                    )
                """, (session_id, session_id, keep_recent))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to clear old records: {e}")


_output_awareness_instance: Optional[OutputAwareness] = None

def get_output_awareness(db_path: str = "data.db") -> OutputAwareness:
    """Lazy singleton for ``OutputAwareness``."""
    global _output_awareness_instance
    if _output_awareness_instance is None:
        _output_awareness_instance = OutputAwareness(db_path)
    return _output_awareness_instance
