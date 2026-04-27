#!/usr/bin/env python3
"""
Sensory buffer — ultra-short-lived raw percept trace (iconic-memory analogue).

[v2.0] Snap-persistence via SQLite so a restart does not wipe the last few turns.
"""
import logging
import sqlite3
import json
from typing import Dict, List, Optional
from datetime import datetime, timezone, timedelta
from collections import deque

logger = logging.getLogger(__name__)


class SensoryBuffer:
    """
    Ring buffer of recent multimodal snippets (conversation, heartbeat, wandering, …).

    - Bounded by ``max_turns`` in RAM; DB row cap is slightly looser to cut DELETE churn.
    - Sleep / metabolism readers pull ``get_all_for_metabolism`` to promote salient lines.
    - [v2.0] SQLite mirror survives process restarts.
    """

    def __init__(self, db_path: str = "data.db", max_turns: int = 20):
        """
        Args:
            db_path: SQLite path backing the temp table.
            max_turns: deque maxlen; older conversation rows are trimmed from DB opportunistically.
        """
        self.db_path = db_path
        self.max_turns = max_turns

        self._init_table()

        # {session_id: deque([{id, content, type, turn_index, timestamp}, ...])}
        self.buffers: Dict[str, deque] = {}

        self._restore_from_db()

    def _init_table(self):
        """Create ``sensory_memory_temp`` if missing."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS sensory_memory_temp (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        content TEXT NOT NULL,
                        type TEXT DEFAULT 'conversation',
                        turn_index INTEGER,
                        timestamp TEXT NOT NULL,
                        metadata TEXT
                    )
                """)
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_sensory_session ON sensory_memory_temp(session_id, timestamp)"
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to init sensory_memory_temp table: {e}")

    def _restore_from_db(self):
        """Hydrate in-memory deques from SQLite (oldest → newest)."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.execute("SELECT * FROM sensory_memory_temp ORDER BY timestamp ASC")
                rows = cur.fetchall()

                count = 0
                for row in rows:
                    session_id = row["session_id"]
                    if session_id not in self.buffers:
                        self.buffers[session_id] = deque(maxlen=self.max_turns)

                    entry = {
                        "id": row["id"],
                        "content": row["content"],
                        "type": row["type"],
                        "turn_index": row["turn_index"],
                        "timestamp": row["timestamp"]
                    }
                    self.buffers[session_id].append(entry)
                    count += 1

                if count > 0:
                    logger.info(f"Restored {count} sensory inputs from persistence.")
        except Exception as e:
            logger.error(f"Failed to restore sensory buffer from DB: {e}")

    def add_sensory_input(
        self,
        session_id: str,
        content: str,
        input_type: str = "conversation",
        turn_index: Optional[int] = None
    ):
        """
        Dual-write a row to SQLite then append to the in-RAM deque.

        Args:
            session_id: logical session key.
            content: raw text (e.g. ``User: …`` / heartbeat payload).
            input_type: ``conversation`` | ``heartbeat`` | ``mind_wandering`` | ``shadow_theater`` | …
            turn_index: optional monotonic turn id for debugging.
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        db_id = None
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    """INSERT INTO sensory_memory_temp 
                       (session_id, content, type, turn_index, timestamp) 
                       VALUES (?, ?, ?, ?, ?)""",
                    (session_id, content, input_type, turn_index, timestamp)
                )
                db_id = cursor.lastrowid
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to persist sensory input: {e}")

        if session_id not in self.buffers:
            self.buffers[session_id] = deque(maxlen=self.max_turns)

        entry = {
            "id": db_id,
            "content": content,
            "type": input_type,
            "turn_index": turn_index,
            "timestamp": timestamp
        }

        self.buffers[session_id].append(entry)
        logger.debug(f"Added sensory input (session={session_id}, type={input_type})")

        if len(self.buffers[session_id]) >= self.max_turns:
            self._prune_db(session_id)

    def _prune_db(self, session_id: str):
        """
        Tiered retention inside SQLite:

        - Instrumentation types (heartbeat, reminders, …) keep only the newest five rows.
        - ``conversation`` rows keep ``max_turns + 10`` so user turns are not starved by chatter.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                system_types = ('heartbeat', 'system_reminder', 'internal_wakeup', 'mind_wandering')
                conn.execute(
                    f"""DELETE FROM sensory_memory_temp 
                        WHERE session_id = ? 
                        AND type IN {system_types}
                        AND id NOT IN (
                            SELECT id FROM sensory_memory_temp 
                            WHERE session_id = ? AND type IN {system_types}
                            ORDER BY timestamp DESC 
                            LIMIT 5
                        )""",
                    (session_id, session_id)
                )

                conn.execute(
                    f"""DELETE FROM sensory_memory_temp 
                            WHERE session_id = ? 
                        AND type = 'conversation'
                        AND id NOT IN (
                            SELECT id FROM sensory_memory_temp 
                            WHERE session_id = ? AND type = 'conversation'
                            ORDER BY timestamp DESC 
                            LIMIT ?
                        )""",
                    (session_id, session_id, self.max_turns + 10)
                )

                conn.commit()
                logger.debug(f"Pruned sensory db for {session_id} (prioritized user conversations)")
        except Exception as e:
            logger.warning(f"Failed to prune sensory db: {e}")

    def get_recent_inputs(
        self,
        session_id: str,
        limit: Optional[int] = None,
        input_type: Optional[str] = None
    ) -> List[Dict]:
        """Return recent deque entries (RAM-first)."""
        if session_id not in self.buffers:
            return []

        buffer = self.buffers[session_id]
        results = list(buffer)

        if input_type:
            results = [r for r in results if r["type"] == input_type]

        if limit:
            results = results[-limit:]

        return results

    def get_conversation_history(self, session_id: str, limit: int = 20) -> List[Dict[str, str]]:
        """
        OpenAI-style ``[{role, content}, …]`` for prompt warm-start.

        [P3] Only lines whose ``content`` begins with ``User: `` / ``AI: `` are treated as chat turns;
        other ``type`` values are ignored. Writers should call ``add_sensory_input`` with those prefixes.
        """
        inputs = self.get_recent_inputs(session_id, limit=limit, input_type="conversation")
        history = []
        for entry in inputs:
            content = entry["content"]
            if content.startswith("User: "):
                history.append({"role": "user", "content": content[6:]})
            elif content.startswith("AI: "):
                history.append({"role": "assistant", "content": content[4:]})
        return history

    def get_all_for_metabolism(self, session_id: str) -> str:
        """Flatten every buffered row for sleep / consolidation passes."""
        inputs = self.get_recent_inputs(session_id)
        if not inputs:
            return ""

        lines = []
        for entry in inputs:
            type_label = {
                "conversation": "💬",
                "heartbeat": "💓",
                "mind_wandering": "🧠",
                "shadow_theater": "🎭"
            }.get(entry["type"], "📝")

            lines.append(f"{type_label} [{entry['timestamp']}] {entry['content']}")

        return "\n".join(lines)

    def clear_old_inputs(self, session_id: str, keep_last_n: int = 10):
        """Trim RAM + DB down to ``keep_last_n`` newest rows (post-metabolism hygiene)."""
        if session_id in self.buffers:
            buffer = self.buffers[session_id]
            if len(buffer) > keep_last_n:
                new_buffer = deque(list(buffer)[-keep_last_n:], maxlen=self.max_turns)
                self.buffers[session_id] = new_buffer

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    f"""DELETE FROM sensory_memory_temp 
                        WHERE session_id = ? AND id NOT IN (
                            SELECT id FROM sensory_memory_temp 
                            WHERE session_id = ? 
                            ORDER BY timestamp DESC 
                            LIMIT ?
                        )""",
                    (session_id, session_id, keep_last_n)
                )
                conn.commit()
            logger.debug(f"Cleared old inputs for {session_id}, kept last {keep_last_n}")
        except Exception as e:
            logger.error(f"Failed to clear old inputs from DB: {e}")

    def clear_all(self, session_id: str):
        """Drop every row for ``session_id`` from RAM and SQLite."""
        if session_id in self.buffers:
            self.buffers[session_id].clear()

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM sensory_memory_temp WHERE session_id = ?", (session_id,))
                conn.commit()
            logger.debug(f"Cleared all buffer for session {session_id}")
        except Exception as e:
            logger.error(f"Failed to clear all DB buffer: {e}")
