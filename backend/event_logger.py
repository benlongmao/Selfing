#!/usr/bin/env python3
"""
Structured event and chat-turn logging for experiments and audits.
"""
import json
import sqlite3
import uuid
import hashlib
from datetime import datetime, timezone
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class EventLogger:
    """Persists chat turns and lightweight experiment events to SQLite."""

    def __init__(self, db_path: str = "data.db"):
        self.db_path = db_path

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def log_event(self, session_id: str, event_type: str, payload: str = ""):
        """
        Append a row to ``event_logs`` (creates the table on demand).
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Ensure schema exists before insert.
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS event_logs (
                        id TEXT PRIMARY KEY,
                        session_id TEXT,
                        event_type TEXT NOT NULL,
                        payload TEXT,
                        created_at TEXT NOT NULL
                    )
                """)

                conn.execute(
                    """
                    INSERT INTO event_logs (id, session_id, event_type, payload, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        str(uuid.uuid4()),
                        session_id,
                        event_type,
                        payload,
                        self._now_iso(),
                    ),
                )
                conn.commit()
        except Exception as exc:
            logger.error(f"Failed to log event {event_type} for session {session_id}: {exc}", exc_info=True)

    def _ensure_tool_calls_table(self, conn: sqlite3.Connection):
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tool_calls (
                receipt_id TEXT PRIMARY KEY,
                ts TEXT NOT NULL,
                session_id TEXT,
                turn_index INTEGER,
                tool_call_id TEXT,
                tool_name TEXT NOT NULL,
                args_json TEXT,
                result_json TEXT,
                ok INTEGER NOT NULL,
                result_hash TEXT
            )
            """
        )

    def _ensure_state_updates_table(self, conn: sqlite3.Connection):
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS state_updates (
                id TEXT PRIMARY KEY,
                ts TEXT NOT NULL,
                session_id TEXT,
                turn_index INTEGER,
                event_type TEXT NOT NULL,
                features_json TEXT
            )
            """
        )

    def log_state_update(
        self,
        session_id: str,
        event_type: str,
        features: Dict[str, Any],
        *,
        turn_index: Optional[int] = None,
    ) -> str:
        """
        Record an observable event → feature vector (P2.6 telemetry substrate).

        Persists the payload immediately; wiring ``Δz_self`` through an ``event_mapper``
        can be layered on later without changing this insert contract.
        """
        event_id = f"evt_{uuid.uuid4().hex}"
        try:
            features_json = json.dumps(features or {}, ensure_ascii=False, sort_keys=True)
        except Exception:
            features_json = json.dumps({"_error": "features_not_serializable"}, ensure_ascii=False)

        try:
            with sqlite3.connect(self.db_path) as conn:
                self._ensure_state_updates_table(conn)
                conn.execute(
                    """
                    INSERT INTO state_updates (id, ts, session_id, turn_index, event_type, features_json)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event_id,
                        self._now_iso(),
                        session_id,
                        turn_index,
                        event_type,
                        features_json,
                    ),
                )
                conn.commit()
        except Exception as exc:
            logger.error(f"Failed to log state_update {event_type} for session {session_id}: {exc}", exc_info=True)
        return event_id

    def log_tool_call(
        self,
        session_id: str,
        tool_name: str,
        args: Dict[str, Any],
        result: Dict[str, Any],
        *,
        turn_index: Optional[int] = None,
        tool_call_id: Optional[str] = None,
    ) -> str:
        """
        Store a tool invocation receipt (P1.1 auditable tool channel).

        Returns ``receipt_id`` so assistant replies can cite deterministic evidence.
        """
        receipt_id = f"rct_{uuid.uuid4().hex}"
        try:
            args_json = json.dumps(args or {}, ensure_ascii=False, sort_keys=True)
        except Exception:
            args_json = json.dumps({"_error": "args_not_serializable"}, ensure_ascii=False)
        try:
            result_json = json.dumps(result or {}, ensure_ascii=False, sort_keys=True)
        except Exception:
            result_json = json.dumps({"_error": "result_not_serializable"}, ensure_ascii=False)

        ok = 0 if (isinstance(result, dict) and "error" in result) else 1
        result_hash = hashlib.sha256(result_json.encode("utf-8")).hexdigest()

        try:
            with sqlite3.connect(self.db_path) as conn:
                self._ensure_tool_calls_table(conn)
                conn.execute(
                    """
                    INSERT INTO tool_calls (
                        receipt_id, ts, session_id, turn_index, tool_call_id,
                        tool_name, args_json, result_json, ok, result_hash
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        receipt_id,
                        self._now_iso(),
                        session_id,
                        turn_index,
                        tool_call_id,
                        tool_name,
                        args_json,
                        result_json,
                        ok,
                        result_hash,
                    ),
                )
                conn.commit()
                # Mirror a coarse signal into state_updates for downstream analytics.
                try:
                    self._ensure_state_updates_table(conn)
                    self.log_state_update(
                        session_id=session_id,
                        event_type=("tool_ok" if ok == 1 else "tool_err"),
                        features={
                            "receipt_id": receipt_id,
                            "tool_call_id": tool_call_id,
                            "tool_name": tool_name,
                            "ok": bool(ok),
                        },
                        turn_index=turn_index,
                    )
                except Exception:
                    pass
        except Exception as exc:
            logger.error(f"Failed to log tool call {tool_name} for session {session_id}: {exc}", exc_info=True)
        return receipt_id

    def log_chat_turn(
        self,
        session_id: str,
        turn_index: int,
        user_input: str,
        assistant_output: str,
        introspection: Optional[Dict[str, Any]],
        drift: Optional[float],
        tick_count: Optional[int],
        self_tick_triggered: bool,
        reflection: Optional[Dict[str, Any]],
        latency: Optional[float],
        tool_used: Optional[Dict[str, Any]] = None,
    ):
        """
        Insert one row into ``chat_turns`` with optional introspection / reflection blobs.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO chat_turns (
                        id, session_id, turn_index, user_input, assistant_output,
                        introspection, drift, tick_count, self_tick_triggered,
                        reflection, latency, tool_used, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(uuid.uuid4()),
                        session_id,
                        turn_index,
                        user_input,
                        assistant_output,
                        json.dumps(introspection, ensure_ascii=False) if introspection else None,
                        drift,
                        tick_count,
                        1 if self_tick_triggered else 0,
                        json.dumps(reflection, ensure_ascii=False) if reflection else None,
                        latency,
                        json.dumps(tool_used, ensure_ascii=False) if tool_used else None,
                        self._now_iso(),
                    ),
                )
                conn.commit()
        except Exception as exc:
            logger.error(f"Failed to log chat turn for session {session_id}: {exc}", exc_info=True)

    def cleanup_reflection_artifacts(
        self,
        *,
        keep_last_reflection_events_per_session: int = 200,
        clear_chat_turns_reflection: bool = True,
        clear_chat_turns_introspection: bool = False,
        older_than_days: int = 30,
    ) -> Dict[str, Any]:
        """
        Trim reflection *process* data without deleting durable ``persona_items`` rules.

        Notes:
        - Durable outcomes live in ``persona_items`` and must remain untouched.
        - Ephemeral process data accumulates in:
          1) ``event_logs`` rows with ``event_type='reflection'`` (unbounded growth risk)
          2) ``chat_turns.reflection`` / ``chat_turns.introspection`` (debug fields; runtime rarely depends on them)
        - Defaults cap each session's retained reflection events and age out older rows.
        """
        res: Dict[str, Any] = {
            "ok": True,
            "deleted_event_logs": 0,
            "cleared_chat_turns_reflection": 0,
            "cleared_chat_turns_introspection": 0,
        }
        keep_n = max(0, int(keep_last_reflection_events_per_session))
        days = max(0, int(older_than_days))

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row

                # 1) Prune reflection rows from event_logs.
                try:
                    if keep_n > 0:
                        # Prefer windowed deletes (SQLite >= 3.25); fall back per-session if unsupported.
                        conn.execute(
                            """
                            DELETE FROM event_logs
                            WHERE id IN (
                              SELECT id FROM (
                                SELECT id,
                                       ROW_NUMBER() OVER (PARTITION BY session_id ORDER BY created_at DESC) AS rn
                                FROM event_logs
                                WHERE event_type='reflection'
                              )
                              WHERE rn > ?
                            )
                            """,
                            (keep_n,),
                        )
                        res["deleted_event_logs"] += conn.total_changes
                    if days > 0:
                        # Also drop stale reflections for idle sessions that never hit the per-session cap.
                        conn.execute(
                            """
                            DELETE FROM event_logs
                            WHERE event_type='reflection'
                              AND created_at < datetime('now', ?)
                            """,
                            (f"-{days} days",),
                        )
                        res["deleted_event_logs"] += conn.total_changes
                except Exception:
                    try:
                        cur = conn.execute(
                            "SELECT DISTINCT session_id FROM event_logs WHERE event_type='reflection'"
                        )
                        sessions = [r[0] for r in cur.fetchall()]
                        for sid in sessions:
                            if keep_n > 0:
                                conn.execute(
                                    """
                                    DELETE FROM event_logs
                                    WHERE event_type='reflection'
                                      AND session_id=?
                                      AND id NOT IN (
                                        SELECT id FROM event_logs
                                        WHERE event_type='reflection' AND session_id=?
                                        ORDER BY created_at DESC
                                        LIMIT ?
                                      )
                                    """,
                                    (sid, sid, keep_n),
                                )
                                res["deleted_event_logs"] += conn.total_changes
                            if days > 0:
                                conn.execute(
                                    """
                                    DELETE FROM event_logs
                                    WHERE event_type='reflection'
                                      AND session_id=?
                                      AND created_at < datetime('now', ?)
                                    """,
                                    (sid, f"-{days} days"),
                                )
                                res["deleted_event_logs"] += conn.total_changes
                    except Exception:
                        pass

                # 2) Null out bulky JSON columns on old chat_turns rows when requested.
                if clear_chat_turns_reflection and days > 0:
                    conn.execute(
                        """
                        UPDATE chat_turns
                        SET reflection=NULL
                        WHERE reflection IS NOT NULL
                          AND created_at < datetime('now', ?)
                        """,
                        (f"-{days} days",),
                    )
                    res["cleared_chat_turns_reflection"] = conn.total_changes

                if clear_chat_turns_introspection and days > 0:
                    conn.execute(
                        """
                        UPDATE chat_turns
                        SET introspection=NULL
                        WHERE introspection IS NOT NULL
                          AND created_at < datetime('now', ?)
                        """,
                        (f"-{days} days",),
                    )
                    res["cleared_chat_turns_introspection"] = conn.total_changes

                conn.commit()
        except Exception as e:
            res["ok"] = False
            res["error"] = str(e)

        return res

