#!/usr/bin/env python3
"""
World / task state (P1.1).

Lightweight session snapshot: task stage, environment summary, last autonomous action.
Provides situational context (stage, environment, last action) for the self model.
"""
import os
import json
import sqlite3
from datetime import datetime, timezone
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)
from backend.s_identity import get_effective_session

class WorldState:
    """Persisted per-session task + environment snapshot."""

    def __init__(self, db_path: str = "data.db"):
        self.db_path = db_path
        self._ensure_table()

    def _ensure_table(self):
        """Create ``world_state`` if missing (legacy DB safe)."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS world_state (
                        session_id TEXT PRIMARY KEY,
                        task_stage TEXT DEFAULT 'plan',
                        env_summary TEXT,
                        last_action TEXT,
                        updated_at TEXT NOT NULL
                    )
                """)
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to ensure world_state table: {e}")

    def get_state(self, session_id: str) -> Optional[Dict]:
        """
        Returns ``{task_stage, env_summary, last_action, updated_at}`` or ``None``.
        """
        session_id = get_effective_session(session_id)
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute(
                    "SELECT task_stage, env_summary, last_action, updated_at FROM world_state WHERE session_id=?",
                    (session_id,)
                )
                row = cur.fetchone()
                if row:
                    return {
                        "task_stage": row[0] or "idle",
                        "env_summary": json.loads(row[1]) if row[1] else {},
                        "last_action": row[2] or "",
                        "updated_at": row[3]
                    }
        except Exception as e:
            logger.error(f"Failed to get world_state for session {session_id}: {e}")
        return None

    def update_state(
        self,
        session_id: str,
        task_stage: Optional[str] = None,
        env_summary: Optional[Dict] = None,
        last_action: Optional[str] = None
    ):
        """
        Upsert; unspecified fields keep their previous values (or defaults on first insert).

        ``task_stage`` is typically ``plan|execute|review|idle``.
        """
        session_id = get_effective_session(session_id)
        try:
            updated_at = datetime.now(timezone.utc).isoformat()

            current = self.get_state(session_id)

            new_task_stage = task_stage if task_stage else (current["task_stage"] if current else "idle")
            new_env_summary = env_summary if env_summary else (current["env_summary"] if current else {})
            new_last_action = last_action if last_action else (current["last_action"] if current else "")

            env_summary_json = json.dumps(new_env_summary, ensure_ascii=False) if new_env_summary else None

            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """INSERT INTO world_state (session_id, task_stage, env_summary, last_action, updated_at)
                       VALUES (?, ?, ?, ?, ?)
                       ON CONFLICT(session_id) DO UPDATE SET
                         task_stage=excluded.task_stage,
                         env_summary=excluded.env_summary,
                         last_action=excluded.last_action,
                         updated_at=excluded.updated_at
                    """,
                    (session_id, new_task_stage, env_summary_json, new_last_action, updated_at)
                )
                conn.commit()
                logger.debug(f"Updated world_state for session {session_id}: stage={new_task_stage}")
        except Exception as e:
            logger.error(f"Failed to update world_state for session {session_id}: {e}", exc_info=True)

    def get_state_text(self, session_id: str) -> str:
        """
        Compact English blurb for prompt injection (empty when unset).
        """
        session_id = get_effective_session(session_id)
        state = self.get_state(session_id)
        if not state:
            return ""

        parts = []
        parts.append(f"Task stage: {state['task_stage']}")

        if state.get('last_action'):
            parts.append(f"Last action: {state['last_action']}")

        if state.get('env_summary') and isinstance(state['env_summary'], dict):
            env = state['env_summary']
            if env:
                env_parts = []
                for key, value in env.items():
                    env_parts.append(f"{key}: {value}")
                if env_parts:
                    parts.append(f"Environment: {', '.join(env_parts)}")

        return "; ".join(parts) if parts else ""
