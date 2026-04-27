#!/usr/bin/env python3
"""
Capture explicit user-stated facts into `user_profiles` (SQLite), bypassing L0/L1/L2 rules.

Constraints (avoid writing someone else's self-intro into this session):
- Only lines with explicit markers (Chinese or English; see regexes below).
- No generic "I am …" extraction (disabled by design).
"""

from __future__ import annotations

import logging
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from backend.config import config
from backend.memory_salience import explicit_fact_salience, normalize_mention_key

logger = logging.getLogger(__name__)


def _iso_date_ymd(iso_ts: Optional[str]) -> str:
    if not iso_ts or not isinstance(iso_ts, str):
        return ""
    s = iso_ts.strip()
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        return s[:10]
    return ""


_REMEMBER_LINE = re.compile(
    r"^\s*(?:"
    r"请记住|请记下|【\s*用户事实\s*】|\[\s*用户事实\s*\]|"
    r"Please remember|Please note|\[\s*USER\s*FACT\s*\]|\[\s*user\s*fact\s*\]"
    r")\s*[:：]\s*(.+?)\s*$",
    re.MULTILINE | re.IGNORECASE,
)
_CALL_ME = re.compile(
    r"(?:"
    r"请叫我|称呼我(?:为)?|叫我|昵称\s*[:：]|"
    r"(?i)(?:\bcall me\b|\bi go by\b|\bi'?m known as\b|nickname\s*[:：])"
    r")\s*"
    r"([^\n。！？!?]{1,40})",
)


def _normalize_fact_line(s: str) -> str:
    t = (s or "").strip()
    t = re.sub(r"\s+", " ", t)
    return t


def _extract_remember_facts(text: str) -> List[str]:
    out: List[str] = []
    for m in _REMEMBER_LINE.finditer(text or ""):
        line = _normalize_fact_line(m.group(1))
        if line and len(line) <= 800:
            out.append(line)
    return out


def _extract_call_me_name(text: str) -> Optional[str]:
    m = _CALL_ME.search(text or "")
    if not m:
        return None
    raw = _normalize_fact_line(m.group(1))
    if not raw:
        return None
    # Strip common trailing punctuation
    raw = raw.strip("「」\"'（）()[]【】")
    if len(raw) > 40 or len(raw) < 1:
        return None
    # Reject path/code-like values
    if any(x in raw for x in ("/", "\\", "import ", "def ", "```", "http://", "https://")):
        return None
    return raw


def parse_user_fact_message(text: str) -> Tuple[Optional[str], List[str]]:
    """
    Parse a single user message for a display name and fact lines to persist (either may be empty).
    """
    if not (text or "").strip():
        return None, []

    name = _extract_call_me_name(text)
    facts = _extract_remember_facts(text)
    return name, facts


def _merge_facts(existing: str, new_lines: List[str], max_chars: int) -> str:
    prev_lines = []
    if existing:
        for ln in str(existing).split("\n"):
            t = _normalize_fact_line(ln)
            if t:
                prev_lines.append(t)
    seen = set(prev_lines)
    for nl in new_lines:
        if nl not in seen:
            prev_lines.append(nl)
            seen.add(nl)
    merged = "\n".join(prev_lines)
    if len(merged) > max_chars:
        merged = merged[-max_chars:]
        # Prefer dropping a short leading fragment up to the first newline
        idx = merged.find("\n")
        if idx > 0 and idx < len(merged) // 4:
            merged = merged[idx + 1 :]
    return merged


def _existing_fact_line_set(old_facts: str) -> Set[str]:
    out: Set[str] = set()
    for ln in str(old_facts or "").split("\n"):
        t = _normalize_fact_line(ln)
        if t:
            out.add(t)
    return out


def _ensure_user_stated_facts_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS user_stated_facts (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            content TEXT NOT NULL,
            turn_index INTEGER,
            created_at TEXT NOT NULL
        )
        """
    )
    cur = conn.execute("PRAGMA table_info(user_stated_facts)")
    cols = {row[1] for row in cur.fetchall()}
    if "salience_score" not in cols:
        conn.execute(
            "ALTER TABLE user_stated_facts ADD COLUMN salience_score REAL DEFAULT 0.92"
        )
    if "mention_key" not in cols:
        conn.execute("ALTER TABLE user_stated_facts ADD COLUMN mention_key TEXT")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_user_stated_facts_session "
        "ON user_stated_facts(session_id, created_at DESC)"
    )


def _ensure_memory_mention_track_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memory_mention_track (
            session_id TEXT NOT NULL,
            mention_key TEXT NOT NULL,
            hit_count INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (session_id, mention_key)
        )
        """
    )


def _bump_mention_count(
    conn: sqlite3.Connection,
    session_id: str,
    mention_key: str,
    now: str,
) -> int:
    """Per-session mention count for a normalized key (boost salience after N hits)."""
    if not mention_key:
        return 0
    row = conn.execute(
        "SELECT hit_count FROM memory_mention_track WHERE session_id = ? AND mention_key = ?",
        (session_id, mention_key),
    ).fetchone()
    if row:
        c = int(row[0] or 0) + 1
        conn.execute(
            "UPDATE memory_mention_track SET hit_count = ?, updated_at = ? "
            "WHERE session_id = ? AND mention_key = ?",
            (c, now, session_id, mention_key),
        )
        return c
    conn.execute(
        "INSERT INTO memory_mention_track (session_id, mention_key, hit_count, updated_at) "
        "VALUES (?, ?, 1, ?)",
        (session_id, mention_key, now),
    )
    return 1


def _update_stated_fact_salience(
    conn: sqlite3.Connection,
    session_id: str,
    mention_key: str,
    content_line: str,
    salience: float,
) -> None:
    """Raise salience on repeat mentions (match by mention_key or legacy empty key + content)."""
    conn.execute(
        """
        UPDATE user_stated_facts
        SET salience_score = MAX(COALESCE(salience_score, 0.0), ?),
            mention_key = COALESCE(mention_key, ?)
        WHERE session_id = ? AND kind = 'fact_line'
          AND (mention_key = ? OR (mention_key IS NULL AND content = ?))
        """,
        (salience, mention_key, session_id, mention_key, content_line),
    )


def _append_user_stated_facts(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    now: str,
    turn_index: Optional[int],
    fact_rows: List[Tuple[str, float, str]],
    new_display_name: Optional[str],
) -> int:
    """Append stated-fact rows (parallel to user_profiles text for audit after cleanup)."""
    n = 0
    for t, sal, mkey in fact_rows:
        if not t:
            continue
        conn.execute(
            """
            INSERT INTO user_stated_facts
            (id, session_id, kind, content, turn_index, created_at, salience_score, mention_key)
            VALUES (?, ?, 'fact_line', ?, ?, ?, ?, ?)
            """,
            (str(uuid.uuid4()), session_id, t, turn_index, now, sal, mkey or None),
        )
        n += 1
    if new_display_name:
        dn = _normalize_fact_line(new_display_name)
        if dn:
            dn_key = normalize_mention_key("display_name:" + dn)
            conn.execute(
                """
                INSERT INTO user_stated_facts
                (id, session_id, kind, content, turn_index, created_at, salience_score, mention_key)
                VALUES (?, ?, 'display_name', ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    session_id,
                    dn,
                    turn_index,
                    now,
                    explicit_fact_salience(mention_hit_count=1),
                    dn_key or None,
                ),
            )
            n += 1
    return n


def fetch_recent_stated_facts_for_prompt(
    db_path: str,
    session_id: str,
    *,
    limit: int = 14,
) -> List[str]:
    """Recent stated-fact lines with turn/date for the fact-trail block in prompts."""
    if not db_path or not session_id:
        return []
    lim = max(1, min(40, int(limit)))
    out: List[str] = []
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(
                """
                SELECT content, kind, turn_index, created_at
                FROM user_stated_facts
                WHERE session_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (session_id, lim),
            )
            for row in cur:
                d = _iso_date_ymd(row["created_at"])
                tid = row["turn_index"]
                ti = f"turn {int(tid)}" if tid is not None else "turn unknown"
                if (row["kind"] or "") == "display_name":
                    out.append(f"• [{ti} · {d}] display name: {row['content']}")
                else:
                    out.append(f"• [{ti} · {d}] {row['content']}")
    except Exception as e:
        logger.debug("fetch_recent_stated_facts_for_prompt: %s", e)
    return list(reversed(out))


def apply_user_fact_capture(
    db_path: str,
    session_id: str,
    user_text: str,
    *,
    turn_index: Optional[int] = None,
) -> Dict[str, Any]:
    """
    When explicit patterns match, upsert ``user_profiles`` and append ``user_stated_facts`` with turn provenance.
    """
    enabled = bool(config.get("system.user_fact_capture_enabled", True))
    if not enabled or not db_path or not session_id:
        return {"updated": False}

    max_chars = int(config.get("system.user_facts_max_chars", 4000) or 4000)
    max_chars = max(500, min(20000, max_chars))

    name, fact_lines = parse_user_fact_message(user_text)
    if not name and not fact_lines:
        return {"updated": False}

    now = datetime.now(timezone.utc).isoformat()
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_profiles (
                    session_id TEXT PRIMARY KEY,
                    name TEXT,
                    facts TEXT,
                    last_seen TEXT
                )
                """
            )
            _ensure_user_stated_facts_table(conn)
            _ensure_memory_mention_track_table(conn)

            cur = conn.execute(
                "SELECT name, facts FROM user_profiles WHERE session_id = ?",
                (session_id,),
            )
            row = cur.fetchone()
            old_name = (row[0] or "").strip() if row else ""
            old_facts = row[1] if row and len(row) > 1 else ""

            final_name = (name.strip() if name else old_name) or ""
            existing_lines = _existing_fact_line_set(old_facts or "")
            new_fact_only = [_normalize_fact_line(x) for x in fact_lines]
            new_fact_only = [x for x in new_fact_only if x and x not in existing_lines]

            if fact_lines:
                final_facts = _merge_facts(old_facts or "", fact_lines, max_chars)
            else:
                final_facts = (old_facts or "").strip()

            name_changed = bool(name) and name.strip() != old_name

            conn.execute(
                """
                INSERT INTO user_profiles (session_id, name, facts, last_seen)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    name = COALESCE(excluded.name, user_profiles.name),
                    facts = COALESCE(excluded.facts, user_profiles.facts),
                    last_seen = excluded.last_seen
                """,
                (
                    session_id,
                    final_name or None,
                    final_facts or None,
                    now,
                ),
            )

            fact_rows: List[Tuple[str, float, str]] = []
            seen_turn: Set[str] = set()
            for raw in fact_lines:
                t = _normalize_fact_line(raw)
                if not t or t in seen_turn:
                    continue
                seen_turn.add(t)
                mkey = normalize_mention_key(t)
                hit = _bump_mention_count(conn, session_id, mkey, now)
                sal = explicit_fact_salience(mention_hit_count=hit)
                if t in new_fact_only:
                    fact_rows.append((t, sal, mkey))
                else:
                    _update_stated_fact_salience(conn, session_id, mkey, t, sal)

            stated_n = _append_user_stated_facts(
                conn,
                session_id=session_id,
                now=now,
                turn_index=turn_index,
                fact_rows=fact_rows,
                new_display_name=name.strip() if name_changed else None,
            )
            conn.commit()

        logger.info(
            "[USER-FACT] session=%s set_name=%s new_fact_lines=%d stated_rows=%d turn=%s",
            session_id,
            bool(name),
            len(new_fact_only),
            stated_n,
            turn_index,
        )
        return {
            "updated": True,
            "set_name": bool(name),
            "appended_facts": len(fact_lines),
            "new_fact_lines": len(new_fact_only),
            "stated_rows": stated_n,
        }
    except Exception as e:
        logger.warning("[USER-FACT] apply failed: %s", e)
        return {"updated": False, "error": str(e)}


def format_user_profile_block_for_prompt(
    db_path: str,
    session_id: str,
    *,
    max_facts_chars: Optional[int] = None,
) -> str:
    """
    Prompt block `[USER profile]` (same shape as chat_service / unified-memory fallback).
    """
    if not db_path or not session_id:
        return ""
    mc = max_facts_chars
    if mc is None:
        mc = int(config.get("system.user_facts_max_chars", 4000) or 4000)
        mc = min(20000, max(500, mc))
    try:
        with sqlite3.connect(db_path) as conn:
            cur = conn.execute(
                "SELECT name, facts FROM user_profiles WHERE session_id = ?",
                (session_id,),
            )
            row = cur.fetchone()
            name = ""
            facts = ""
            if row:
                name = (row[0] or "").strip()
                facts = (row[1] or "").strip()
            trail = fetch_recent_stated_facts_for_prompt(db_path, session_id, limit=14)
            if not name and not facts and not trail:
                return ""
            if len(facts) > mc:
                facts = facts[-mc:]
            lines = ["[USER profile]"]
            if name:
                lines.append(f"Display name: {name}")
            if facts:
                lines.append(f"Confirmed facts: {facts}")
            if trail:
                lines.append("")
                lines.append("[Fact trail · by turn and date]")
                lines.extend(trail)
            return "\n".join(lines) + "\n\n"
    except Exception as e:
        logger.debug("format_user_profile_block_for_prompt: %s", e)
        return ""
