"""
Session history helpers.

- Append turn history
- Memory-reminder detection
- Long-message truncation
- [2026-04-07] When the user asks about the prior reply, inject a tiny tail from chat_turns for token savings and verification

Note: rolling history compression lives in chat_message_builder.py.
"""
from __future__ import annotations

import logging
import re
import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from backend.config import config

logger = logging.getLogger(__name__)

# Mandarin cues that the user is referring to an earlier turn (substring match on raw text).
_PAST_REFERENCE_MARKERS_CN = ["刚才", "之前", "上次", "你刚刚", "前面", "前文", "刚刚的", "回顾", "复述", "你说过"]
_PAST_REFERENCE_MARKERS_EN = [
    "earlier",
    "previous",
    "above",
    "last time",
    "you said",
    "recap",
    "quote",
    "just now",
    "a moment ago",
    "previously",
    "you just said",
    "prior message",
    "earlier message",
    "what you wrote",
    "paraphrase",
    "repeat that",
    "summarize what",
]


def _should_add_memory_reminder(user_text: str) -> bool:
    """True if the user message looks like a reference to earlier context."""
    if not user_text:
        return False
    lower = user_text.lower()
    markers_cn = config.get("parameters.chat.memory_reminder.markers_cn", _PAST_REFERENCE_MARKERS_CN) or _PAST_REFERENCE_MARKERS_CN
    markers_en = config.get("parameters.chat.memory_reminder.markers_en", _PAST_REFERENCE_MARKERS_EN) or _PAST_REFERENCE_MARKERS_EN
    if any(m in user_text for m in markers_cn):
        return True
    return any(m in lower for m in markers_en)


def _extract_assistant_snippet_for_prompt(raw: str, max_chars: int) -> str:
    """
    Take a very short slice from assistant output: prefer a tail containing \\boxed, else plain tail.
    Collapse whitespace to single spaces to save tokens.
    """
    if not raw or not str(raw).strip():
        return ""
    t = str(raw).strip()
    max_chars = max(32, min(int(max_chars), 500))
    key = "\\boxed"
    idx = t.rfind(key)
    if idx >= 0:
        frag = t[idx : idx + max_chars + 64]
    else:
        frag = t[-max_chars:]
    frag = re.sub(r"\s+", " ", frag).strip()
    if len(frag) > max_chars:
        frag = frag[: max_chars - 1] + "…"
    return frag


def fetch_last_assistant_snippet_from_db(
    db_path: str,
    session_id: str,
    max_chars: int = 120,
) -> Optional[str]:
    """Load latest non-empty assistant_output for the session; return a truncated snippet."""
    if not db_path or not session_id:
        return None
    try:
        with sqlite3.connect(db_path) as conn:
            cur = conn.execute(
                """
                SELECT assistant_output FROM chat_turns
                WHERE session_id = ? AND assistant_output IS NOT NULL
                  AND length(trim(assistant_output)) > 0
                ORDER BY turn_index DESC, created_at DESC
                LIMIT 1
                """,
                (session_id,),
            )
            row = cur.fetchone()
            if not row or not row[0]:
                return None
            snip = _extract_assistant_snippet_for_prompt(row[0], max_chars)
            return snip or None
    except Exception as e:
        logger.debug("[MEMORY-REMINDER] fetch_last_assistant_snippet_from_db failed: %s", e)
        return None


def augment_user_with_memory_reminder(
    db_path: str,
    session_id: str,
    user_input: str,
    *,
    has_session_history: bool,
) -> str:
    """
    If past-reference markers hit, prepend a short system-facing hint; optionally attach last assistant tail (strict char cap).
    Otherwise return user_input unchanged.
    """
    if not has_session_history or not (user_input or "").strip():
        return user_input
    if not config.get("parameters.chat.memory_reminder.enabled", True):
        return user_input
    if not _should_add_memory_reminder(user_input):
        return user_input

    reminder = (
        "(System note: the user is referring to earlier content. Cross-check against the conversation "
        "history above before answering; do not quote the full transcript—distill key points.)\n\n"
    )
    extra = ""
    if config.get("parameters.chat.memory_reminder.inject_last_assistant_tail", True):
        mx = int(config.get("parameters.chat.memory_reminder.last_assistant_snippet_max_chars", 120) or 120)
        mx = max(40, min(mx, 400))
        snip = fetch_last_assistant_snippet_from_db(db_path, session_id, mx)
        if snip:
            extra = f"[Last assistant tail · for verification]{snip}\n\n"
    return reminder + extra + user_input


def _truncate_middle(text: str, head: int = 320, tail: int = 160) -> str:
    """Truncate long text, keep head and tail so the next round is not flooded with verbatim repeats."""
    if not text:
        return text
    if head <= 0 or tail <= 0:
        return text[: max(head + tail, 0)]
    if len(text) <= head + tail + 40:
        return text
    return f"{text[:head]}\n...\n{text[-tail:]}"


def append_history(
    session_history: Dict[str, List[Dict]],
    session_id: str,
    user_input: str,
    final_response: str,
    logger: logging.Logger,
    enable_print: bool = True,
    max_len: int = 20,
    is_system_reminder: bool = False,
    # [2026-03-18] Scheduled/calendar reminders: UI shows as system, not as a normal user turn
) -> None:
    """Append to in-memory session history and trim to max_len."""
    if session_id not in session_history:
        session_history[session_id] = []
        logger.debug(f"[MEMORY-DEBUG] Created new session_history for {session_id}")
    
    if enable_print:
        print(f"[MEMORY-DEBUG] Before append: session_history[{session_id}] length = {len(session_history[session_id])}")
    logger.info(f"[MEMORY-DEBUG] Before append: session_history[{session_id}] length = {len(session_history[session_id])}")
    
    # [2026-01-17] Timestamp on each turn so the agent can sense elapsed wall time
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    user_msg = {"role": "user", "content": user_input, "timestamp": timestamp}
    if is_system_reminder:
        user_msg["is_system_reminder"] = True
    session_history[session_id].append(user_msg)
    session_history[session_id].append({"role": "assistant", "content": final_response, "timestamp": timestamp})
    
    if enable_print:
        print(f"[MEMORY-DEBUG] After append: session_history[{session_id}] length = {len(session_history[session_id])}, last message: {session_history[session_id][-1]['role']}")
    logger.info(f"[MEMORY-DEBUG] After append: session_history[{session_id}] length = {len(session_history[session_id])}, last message: {session_history[session_id][-1]['role']}")
    
    if len(session_history[session_id]) > max_len:
        session_history[session_id] = session_history[session_id][-max_len:]
        logger.debug(f"[MEMORY-DEBUG] Trimmed session_history to {max_len} messages")
