#!/usr/bin/env python3
"""
Calendar tool — schedule one-off events and read timelines.

Events live in ``calendar_events`` inside ``data.db``.
"""
import logging
import sqlite3
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


def parse_event_start_local(start_time_raw: Optional[str]) -> Optional[datetime]:
    """
    Parse stored ``start_time`` into a naive local wall-clock ``datetime`` comparable to ``datetime.now()``.

    ISO strings with ``Z`` or explicit offsets are converted to local time then tz-stripped (no hard-coded +8).
    """
    if start_time_raw is None:
        return None
    s = str(start_time_raw).strip()
    if not s:
        return None
    ts = s.replace(" ", "T", 1)
    try:
        if ts.endswith("Z"):
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            if dt.tzinfo:
                return dt.astimezone().replace(tzinfo=None)
            return dt
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is not None:
            return dt.astimezone().replace(tzinfo=None)
        return dt
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _parse_iso_utc(s: Optional[str]) -> Optional[datetime]:
    """Parse UTC-ish ISO timestamps such as ``reminder_sent_at`` / ``archived_at``."""
    if not s or not str(s).strip():
        return None
    try:
        t = str(s).strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(t)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _active_status_sql() -> str:
    return "(status IS NULL OR TRIM(COALESCE(status,'')) = '' OR status = 'active')"


class CalendarTool:
    def __init__(self, db_path: str = "data.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Ensure the ``calendar_events`` table exists."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS calendar_events (
                        id TEXT PRIMARY KEY,
                        session_id TEXT,
                        title TEXT NOT NULL,
                        description TEXT,
                        start_time TEXT NOT NULL, -- ISO8601
                        end_time TEXT,            -- ISO8601
                        created_at TEXT,
                        status TEXT DEFAULT 'active', -- active, completed, cancelled
                        reminder_sent_at TEXT
                    )
                """)
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to init calendar db: {e}")
        # Compatible with old tables: add reminder_sent_at column if not present
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("ALTER TABLE calendar_events ADD COLUMN reminder_sent_at TEXT")
                conn.commit()
        except sqlite3.OperationalError:
            pass
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("ALTER TABLE calendar_events ADD COLUMN archived_at TEXT")
                conn.commit()
        except sqlite3.OperationalError:
            pass

    def run_maintenance_gc(self) -> Dict[str, Any]:
        """
        System maintenance: archive finished one-shot events and optionally purge stale rows.

        There is no built-in “same row repeats daily”; for recurring work use ``schedule_task`` or insert new rows.
        """
        from backend.config import config

        out: Dict[str, Any] = {"archived": 0, "purged": 0}

        archive_after_h = float(config.get("parameters.calendar.archive_after_reminder_hours", 48) or 48)
        stale_unrem_h = float(config.get("parameters.calendar.stale_unreminded_archive_hours", 336) or 0)
        purge_days = float(config.get("parameters.calendar.purge_expired_after_days", 90) or 0)

        now_utc = datetime.now(timezone.utc)
        now_local = datetime.now()
        now_iso = now_utc.isoformat()

        to_archive: List[str] = []
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.execute(
                    f"SELECT id, start_time, reminder_sent_at, status FROM calendar_events WHERE {_active_status_sql()}"
                )
                rows = cur.fetchall()
                for row in rows:
                    eid = row["id"]
                    rem = row["reminder_sent_at"]
                    if rem:
                        rem_dt = _parse_iso_utc(rem)
                        if rem_dt is None:
                            continue
                        age_h = (now_utc - rem_dt).total_seconds() / 3600.0
                        if archive_after_h > 0 and age_h >= archive_after_h:
                            to_archive.append(eid)
                        continue
                    if stale_unrem_h <= 0:
                        continue
                    st_local = parse_event_start_local(row["start_time"])
                    if st_local is None:
                        continue
                    if st_local > now_local:
                        continue
                    age_unrem_h = (now_local - st_local).total_seconds() / 3600.0
                    if age_unrem_h >= stale_unrem_h:
                        to_archive.append(eid)

                for eid in to_archive:
                    conn.execute(
                        "UPDATE calendar_events SET status = 'expired', archived_at = ? WHERE id = ?",
                        (now_iso, eid),
                    )
                conn.commit()
                out["archived"] = len(to_archive)

                if purge_days > 0:
                    cur = conn.execute(
                        "SELECT id, archived_at FROM calendar_events WHERE status = 'expired' AND archived_at IS NOT NULL"
                    )
                    for r in cur.fetchall():
                        adt = _parse_iso_utc(r["archived_at"])
                        if adt is None:
                            continue
                        if (now_utc - adt).total_seconds() >= purge_days * 86400.0:
                            conn.execute("DELETE FROM calendar_events WHERE id = ?", (r["id"],))
                            out["purged"] += 1
                    conn.commit()
        except Exception as e:
            logger.error(f"run_maintenance_gc failed: {e}")
        return out

    def get_due_events_for_reminder(self, session_id: str = "default", window_minutes: int = 2) -> List[Dict]:
        """
        Return calendar rows that are due for reminder delivery (``start_time`` in the past, no reminder yet).

        ``window_minutes`` is kept only for API compatibility; actual selection is “due and unreminded”, with
        ``max_overdue_hours`` to avoid scanning very stale rows forever.

        Rows past ``max_overdue_hours`` without a reminder are marked ``expired`` so background loops do not churn.
        """
        _ = window_minutes
        try:
            from backend.config import config

            max_overdue_h = float(config.get("parameters.calendar.max_overdue_hours", 168) or 168)
            max_overdue_h = max(0.0, min(max_overdue_h, 8760.0))

            now = datetime.now()
            now_utc_iso = datetime.now(timezone.utc).isoformat()
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    f"SELECT * FROM calendar_events WHERE session_id = ? AND {_active_status_sql()} "
                    "AND reminder_sent_at IS NULL ORDER BY start_time ASC",
                    (session_id,),
                )
                rows = cursor.fetchall()
                result: List[Dict] = []
                for row in rows:
                    event_time = parse_event_start_local(row["start_time"])
                    if event_time is None:
                        continue
                    if event_time > now:
                        continue
                    if max_overdue_h > 0:
                        age_sec = (now - event_time).total_seconds()
                        if age_sec > max_overdue_h * 3600:
                            eid = row["id"]
                            conn.execute(
                                "UPDATE calendar_events SET status = 'expired', archived_at = ? WHERE id = ?",
                                (now_utc_iso, eid),
                            )
                            logger.info(
                                "calendar: expired unreminded past max_overdue session=%s id=%s",
                                session_id,
                                eid,
                            )
                            continue
                    result.append(dict(row))
                conn.commit()
            return result
        except Exception as e:
            logger.error(f"get_due_events_for_reminder failed: {e}")
            return []

    def mark_reminder_sent(self, event_id: str) -> None:
        """Mark ``reminder_sent_at`` so duplicate pushes are not sent."""
        try:
            now = datetime.now(timezone.utc).isoformat()
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "UPDATE calendar_events SET reminder_sent_at = ? WHERE id = ?",
                    (now, event_id)
                )
                conn.commit()
        except Exception as e:
            logger.error(f"mark_reminder_sent failed: {e}")

    def add_event(self, title: str, start_time: str, description: str = "", end_time: str = None, session_id: str = "default") -> str:
        """
        Insert a single calendar event.

        Prefer ISO8601 or ``YYYY-MM-DD HH:MM`` strings; values are stored mostly verbatim for flexibility.
        """
        try:
            event_id = f"evt_{uuid.uuid4().hex[:8]}"
            now = datetime.now(timezone.utc).isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO calendar_events (id, session_id, title, description, start_time, end_time, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (event_id, session_id, title, description, start_time, end_time, now)
                )
                conn.commit()
            return f"✅ Added event [{title}] at {start_time}"
        except Exception as e:
            logger.error(f"Add event error: {e}")
            return f"❌ Failed to add event: {e}"

    def list_events(self, date_str: str = None, limit: int = 10, session_id: str = "default") -> str:
        """
        List events for ``session_id``.

        If ``date_str`` (YYYY-MM-DD) is set, filter to that day; otherwise return the next ``limit`` rows.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                act = _active_status_sql()
                if date_str:
                    cursor.execute(
                        f"SELECT * FROM calendar_events WHERE session_id = ? AND {act} "
                        "AND start_time LIKE ? ORDER BY start_time ASC LIMIT ?",
                        (session_id, f"{date_str}%", limit),
                    )
                else:
                    cursor.execute(
                        f"SELECT * FROM calendar_events WHERE session_id = ? AND {act} "
                        "ORDER BY start_time ASC LIMIT ?",
                        (session_id, limit),
                    )
                
                rows = cursor.fetchall()
                if not rows:
                    return "📅 No calendar entries yet."
                
                result = []
                for row in rows:
                    end_str = f" - {row['end_time']}" if row['end_time'] else ""
                    desc_str = f" ({row['description']})" if row['description'] else ""
                    result.append(f"- [{row['start_time']}{end_str}] {row['title']}{desc_str} (ID: {row['id']})")
                
                return "\n".join(result)
        except Exception as e:
            return f"❌ Failed to list events: {e}"

    def delete_event(self, event_id: str) -> str:
        """Delete a row by id."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("DELETE FROM calendar_events WHERE id = ?", (event_id,))
                conn.commit()
                if cursor.rowcount > 0:
                    return f"✅ Deleted event: {event_id}"
                else:
                    return f"❌ Event not found: {event_id}"
        except Exception as e:
            return f"❌ Delete failed: {e}"

    def get_event(self, event_id: str) -> Optional[Dict]:
        """Fetch one event as a dict."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("SELECT * FROM calendar_events WHERE id = ?", (event_id,))
                row = cursor.fetchone()
                return dict(row) if row else None
        except Exception as e:
            logger.error(f"Get event error: {e}")
            return None

    def calculate_time_delta(self, event_id_a: str, event_id_b: Optional[str] = None) -> str:
        """
        Compute the delta between two events, or between ``event_id_a`` and “now” when ``event_id_b`` is omitted.
        """
        try:
            event_a = self.get_event(event_id_a)
            if not event_a:
                return f"❌ Event A not found: {event_id_a}"
            
            time_a = datetime.fromisoformat(event_a['start_time'].replace(' ', 'T'))
            
            if event_id_b:
                event_b = self.get_event(event_id_b)
                if not event_b:
                    return f"❌ Event B not found: {event_id_b}"
                time_b = datetime.fromisoformat(event_b['start_time'].replace(' ', 'T'))
                label = f"Between [{event_a['title']}] and [{event_b['title']}]"
            else:
                time_b = datetime.now()
                label = f"Between [{event_a['title']}] and now"
            
            delta = time_b - time_a
            days = delta.days
            seconds = delta.seconds
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            
            return f"⏳ {label}: Δ = {days}d {hours}h {minutes}m."
        except Exception as e:
            return f"❌ Failed to compute delta: {e}"

    def get_timeline_narrative(self, limit: int = 20, session_id: str = "default") -> str:
        """
        Narrative timeline of stored events (handles naive local vs UTC ``Z`` strings for display).
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT * FROM calendar_events WHERE session_id = ? ORDER BY start_time ASC LIMIT ?",
                    (session_id, limit),
                )
                rows = cursor.fetchall()

                if not rows:
                    return "📖 Timeline is empty — add an event to start the story."

                now = datetime.now()
                narrative = [f"📜 Timeline (local now {now.strftime('%H:%M')}):\n"]

                for row in rows:
                    try:
                        # Try to parse the stored time string
                        time_str = row['start_time'].replace(' ', 'T')
                        # If it is ISO format with Z, it is treated as UTC
                        if 'T' in time_str and (time_str.endswith('Z') or '+00:00' in time_str):
                            event_time_utc = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
                            # Legacy rows stored as UTC Z — shift +8h for display (Asia/Shanghai wall clock).
                            event_time = event_time_utc + timedelta(hours=8)
                        else:
                            # Otherwise, it will be regarded as local time and parsed directly.
                            event_time = datetime.fromisoformat(time_str)
                        
                        delta = now - event_time
                        
                        if delta.days == 0 and delta.total_seconds() > 0:
                            time_desc = "today"
                        elif delta.days == 1:
                            time_desc = "yesterday"
                        elif delta.total_seconds() < 0:
                            time_desc = "upcoming"  # future-dated entry
                        else:
                            time_desc = f"{delta.days} day(s) ago"
                        
                        display_time = event_time.strftime("%Y-%m-%d %H:%M:%S")
                        st = (row["status"] or "active").strip() or "active"
                        tag = " [archived]" if st == "expired" else ""
                        narrative.append(f"[{display_time}] ({time_desc}) - {row['title']}{tag}")
                        if row['description']:
                            narrative.append(f"   └─ {row['description']}")
                    except Exception as e:
                        logger.warning(f"Failed to parse timeline row: {e}")
                        continue
                
                return "\n".join(narrative)
        except Exception as e:
            return f"❌ Failed to build timeline: {e}"

    def get_tool_definitions(self) -> List[Dict]:
        """OpenAI-style tool definitions."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "add_calendar_event",
                    "description": (
                        "Add a one-shot calendar event: each ``start_time`` triggers at most one reminder cycle; "
                        "there is no weekly/daily recurrence here. For repeating jobs use ``schedule_task`` with "
                        "``frequency``. Prefer ISO8601 or ``YYYY-MM-DD HH:MM:SS``."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string", "description": "Title of the event"},
                            "start_time": {"type": "string", "description": "Start time in 'YYYY-MM-DD HH:MM:SS' format"},
                            "description": {"type": "string", "description": "Optional details about the event"},
                            "end_time": {"type": "string", "description": "Optional end time"}
                        },
                        "required": ["title", "start_time"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_calendar_events",
                    "description": "List stored calendar rows (single timestamps only). For recurring automation see scheduled jobs APIs.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "date_str": {"type": "string", "description": "Optional date filter (YYYY-MM-DD)"},
                            "limit": {"type": "integer", "description": "Max number of events to return", "default": 10}
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "delete_calendar_event",
                    "description": "Delete an event by its ID.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "event_id": {"type": "string", "description": "The ID of the event to delete"}
                        },
                        "required": ["event_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "calculate_time_delta",
                    "description": "Calculate the time difference between two events, or between an event and now.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "event_id_a": {"type": "string", "description": "ID of the first event"},
                            "event_id_b": {"type": "string", "description": "ID of the second event (optional, defaults to now)"}
                        },
                        "required": ["event_id_a"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_timeline_narrative",
                    "description": "Get a narrative summary of your personal timeline and history.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "limit": {"type": "integer", "description": "Max events to show", "default": 20}
                        },
                        "required": []
                    }
                }
            }
        ]
