"""
Autonomy gate — pause **system-driven** background work without blocking user-led /api/chat turns.

**In scope:** UnifiedScheduler jobs, resting-pulse scheduled/calendar enqueue, heartbeat enqueue,
and similar paths that fire without a user chat (see each call site).

**Out of scope:** Normal multi-turn chats and tool loops that are driven by the user message.

**State file:** ``run/autonomy_gate.json`` by default (runtime state, ignored for open-source exports).

**User-visible tokens (ASCII + legacy Chinese phrases):**
- Pause: ``[S44_AUTONOMY_PAUSE]`` or the Chinese phrases matched by ``_USER_PAUSE_PHRASE``.
- Resume: ``[S44_AUTONOMY_RESUME]``, full-width brackets, bare ``S44_AUTONOMY_RESUME``, or the
  Chinese phrases in ``_USER_RESUME_PHRASE`` (same effect as “start autonomy”).
- Negation / meta-mention / quoted text is filtered (see ``_should_ignore_user_command_match``).

**Assistant side:** whole-line tokens such as ``[S44_PAUSE]`` / ``[S44_TIRED]``; bracketed
``[S44_AUTONOMY_*]`` may appear mid-sentence — see ``apply_assistant_autonomy_markers``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from backend.config import config
from backend.utils.path_utils import get_project_root, get_workspace_root

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_FILENAME = "autonomy_gate.json"
_LEGACY_FILENAME = ".autonomy_gate.json"
_DEFAULT_STATE: Dict[str, Any] = {
    "paused": False,
    "updated_at": 0.0,
    "reason": "",
    "updated_by": "",
}
# Bare token must not match the same substring inside bracketed forms (avoid false resume on “explain [S44_AUTONOMY_RESUME]”).
_BARE_AUTONOMY_RESUME_RE = re.compile(
    r"(?<!\[)(?<!【)(?<![A-Za-z0-9_])S44_AUTONOMY_RESUME(?!\])(?!】)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)
_QUOTE_WRAP_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("`", "`"),
    ('"', '"'),
    ("'", "'"),
    ("“", "”"),
    ("‘", "’"),
    ("「", "」"),
    ("『", "』"),
)
_META_PREFIX_RE = re.compile(
    r"(解释|说明|介绍|讨论|分析|提到|提及|引用|看到|显示|输出|打印|举例|示例|例如|比如|说的是|写的是)\s*$",
    re.IGNORECASE,
)
_META_SUFFIX_RE = re.compile(
    r"^\s*(的作用|是什么意思|啥意思|这几个字|这个词|这个口令|这个标记|这个字符串|这句话|这个短语)",
    re.IGNORECASE,
)
_NON_EXECUTION_SUFFIX_RE = re.compile(
    r"(不是现在执行|不是要你执行|不用现在执行|无需现在执行|别现在执行|先别执行|先不要执行)",
    re.IGNORECASE,
)


def _state_path() -> str:
    configured = str(
        config.get("system.autonomy_gate_state_path", "run/autonomy_gate.json") or ""
    ).strip()
    if configured:
        if os.path.isabs(configured):
            return os.path.abspath(configured)
        return os.path.abspath(os.path.join(get_project_root(), configured))
    return os.path.abspath(os.path.join(get_project_root(), "run", _FILENAME))


def _legacy_state_path() -> str:
    return os.path.join(get_workspace_root(), _LEGACY_FILENAME)


def gate_enabled() -> bool:
    return bool(config.get("system.autonomy_gate_enabled", True))


def _coerce_bool(v: Any, default: bool = False) -> bool:
    """
    Normalize JSON ``paused`` flags: hand-edited files often store ``"false"`` strings which
    would otherwise be truthy in Python.
    """
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("false", "0", "no", "off", "", "none", "null"):
            return False
        if s in ("true", "1", "yes", "on"):
            return True
        return default
    if isinstance(v, dict):
        for key in ("paused", "flag", "enabled", "active"):
            if key in v:
                return _coerce_bool(v.get(key), default)
        return default
    return default


def load_state() -> Dict[str, Any]:
    path = _state_path()
    if not os.path.isfile(path):
        legacy_path = _legacy_state_path()
        if os.path.isfile(legacy_path):
            path = legacy_path
    if not os.path.isfile(path):
        return dict(_DEFAULT_STATE)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return dict(_DEFAULT_STATE)
        data.setdefault("paused", _DEFAULT_STATE["paused"])
        data.setdefault("updated_at", _DEFAULT_STATE["updated_at"])
        data.setdefault("reason", _DEFAULT_STATE["reason"])
        data.setdefault("updated_by", _DEFAULT_STATE["updated_by"])
        data["paused"] = _coerce_bool(data.get("paused"), False)
        raw_sessions = data.get("sessions")
        if isinstance(raw_sessions, dict):
            cleaned: Dict[str, Any] = {}
            for sid, val in raw_sessions.items():
                if _coerce_bool(val, False):
                    cleaned[str(sid)] = True
            data["sessions"] = cleaned
        return data
    except Exception as e:
        logger.warning("[AUTONOMY-GATE] Failed to read state: %s", e)
        return dict(_DEFAULT_STATE)


def _write_state(data: Dict[str, Any]) -> None:
    path = _state_path()
    root = os.path.dirname(path)
    os.makedirs(root, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _set_autonomous_pause_by_actor(
    paused: bool,
    reason: str = "",
    *,
    actor: str = "system",
) -> bool:
    if not gate_enabled():
        return False
    actor = (actor or "system").strip().lower()
    with _LOCK:
        data = load_state()
        data["paused"] = bool(paused)
        data["reason"] = (reason or "")[:500]
        data["updated_at"] = time.time()
        data["updated_by"] = actor
        _write_state(data)
    logger.info(
        "[AUTONOMY-GATE] paused=%s actor=%s reason=%s",
        paused,
        actor,
        reason or "-",
    )
    return True


def set_autonomous_pause(paused: bool, reason: str = "") -> None:
    _set_autonomous_pause_by_actor(paused, reason, actor="system")


def set_autonomous_pause_from_user(paused: bool, session_id: str) -> bool:
    reason = f"user:{session_id}" if paused else ""
    return _set_autonomous_pause_by_actor(paused, reason, actor="user")


def set_autonomous_pause_from_assistant(paused: bool, session_id: str) -> bool:
    reason = f"assistant:{session_id}" if paused else ""
    return _set_autonomous_pause_by_actor(paused, reason, actor="assistant")


def set_autonomous_pause_from_cli(paused: bool, reason: str = "") -> bool:
    default_reason = "cli: scripts/autonomy_gate.sh pause" if paused else ""
    return _set_autonomous_pause_by_actor(
        paused,
        reason or default_reason,
        actor="cli",
    )


def is_autonomous_execution_paused(session_id: str = "selfing-session") -> bool:
    if not gate_enabled():
        return False
    data = load_state()
    if data.get("paused"):
        return True
    per = data.get("sessions") or {}
    if isinstance(per, dict) and per.get(session_id):
        return True
    return False


# Assistant lines treated like user tokens (whole-line, case-insensitive).
_AGENT_PAUSE_LINES: Tuple[str, ...] = (
    "[S44_PAUSE]",
    "[S44_AUTONOMY_PAUSE]",
    "[S44_TIRED]",
)
_AGENT_RESUME_LINES: Tuple[str, ...] = (
    "[S44_AUTONOMY_RESUME]",
)

# Bilingual surface forms (Chinese retained for legacy chats; English added for EN-first installs).
_USER_PAUSE_PHRASE = re.compile(
    r"停止自主行动|停止自主执行|pause\s+autonomous\s+(action|execution)|stop\s+autonomous\s+(action|execution)",
    re.IGNORECASE,
)
_USER_RESUME_PHRASE = re.compile(
    r"恢复自主行动|恢复自主执行|开始自主行动|开始自主执行|"
    r"resume\s+autonomous\s+(action|execution)|start\s+autonomous\s+(action|execution)",
    re.IGNORECASE,
)


def _negation_glued_before_pause_phrase(t: str, phrase_start: int) -> bool:
    """Detect negation glued immediately before the pause phrase (colloquial Chinese)."""
    sl = t[max(0, phrase_start - 14) : phrase_start + 1]
    return bool(
        re.search(
            r"(不要|别|请勿|勿|无需|不必|没有|不想|不打算|不能|不该|别提|避免|拒绝)\s*停\s*$",
            sl,
            re.IGNORECASE,
        )
    )


def _negation_glued_before_resume_phrase(t: str, phrase_start: int) -> bool:
    """Negation glued before resume / start-autonomy colloquial phrases (Chinese)."""
    sl = t[max(0, phrase_start - 16) : phrase_start + 1]
    if phrase_start < len(t) and t[phrase_start] == "开":
        return bool(
            re.search(
                r"(不要|别|请勿|勿|无需|不必|没有|不想|不打算|不能|不该|别提|避免|拒绝)\s*开\s*$",
                sl,
                re.IGNORECASE,
            )
        )
    return bool(
        re.search(
            r"(不要|别|请勿|勿|无需|不必|没有|不想|不打算|不能|不该|别提|避免|拒绝)\s*恢\s*$",
            sl,
            re.IGNORECASE,
        )
    )


def _is_wrapped_in_quotes(t: str, start: int, end: int) -> bool:
    left = start - 1
    while left >= 0 and t[left].isspace():
        left -= 1
    right = end
    while right < len(t) and t[right].isspace():
        right += 1
    if left < 0 or right >= len(t):
        return False
    for open_q, close_q in _QUOTE_WRAP_PAIRS:
        if t[left] == open_q and t[right] == close_q:
            return True
    return False


def _is_meta_mention_context(t: str, start: int, end: int) -> bool:
    prefix = t[max(0, start - 24) : start]
    suffix = t[end : min(len(t), end + 32)]
    if _META_PREFIX_RE.search(prefix):
        return True
    if _META_SUFFIX_RE.search(suffix):
        return True
    if _NON_EXECUTION_SUFFIX_RE.search(suffix):
        return True
    return False


def _should_ignore_user_command_match(t: str, start: int, end: int) -> bool:
    return _is_wrapped_in_quotes(t, start, end) or _is_meta_mention_context(t, start, end)


def _user_pause_positions(t: str) -> List[int]:
    pos: List[int] = []
    for m in re.finditer(r"\[S44_AUTONOMY_PAUSE\]", t, re.IGNORECASE):
        if _should_ignore_user_command_match(t, m.start(), m.end()):
            continue
        pos.append(m.start())
    for m in _USER_PAUSE_PHRASE.finditer(t):
        if _negation_glued_before_pause_phrase(t, m.start()):
            continue
        if _should_ignore_user_command_match(t, m.start(), m.end()):
            continue
        pos.append(m.start())
    return pos


def _user_resume_positions(t: str) -> List[int]:
    pos: List[int] = []
    for m in re.finditer(r"\[S44_AUTONOMY_RESUME\]", t, re.IGNORECASE):
        if _should_ignore_user_command_match(t, m.start(), m.end()):
            continue
        pos.append(m.start())
    # Full-width brackets + bare token (legacy typing habits).
    for m in re.finditer(r"【\s*S44_AUTONOMY_RESUME\s*】", t, re.IGNORECASE):
        if _should_ignore_user_command_match(t, m.start(), m.end()):
            continue
        pos.append(m.start())
    for m in _BARE_AUTONOMY_RESUME_RE.finditer(t):
        if _should_ignore_user_command_match(t, m.start(), m.end()):
            continue
        pos.append(m.start())
    for m in _USER_RESUME_PHRASE.finditer(t):
        if _negation_glued_before_resume_phrase(t, m.start()):
            continue
        if _should_ignore_user_command_match(t, m.start(), m.end()):
            continue
        pos.append(m.start())
    return pos


def _line_matches_token(line: str, tokens: Tuple[str, ...]) -> bool:
    s = (line or "").strip()
    if not s:
        return False
    sup = s.upper()
    for t in tokens:
        if sup == t.upper():
            return True
    return False


def apply_assistant_autonomy_markers(text: str, session_id: str) -> Optional[str]:
    """
    Parse assistant-visible text (answer + optional chain-of-thought) and flip the gate.

    1) Whole-line exact tokens (``[S44_PAUSE]``, ``[S44_TIRED]``, …) avoid mid-sentence false positives.
    2) Substring fallback for ``[S44_AUTONOMY_PAUSE]`` / ``[S44_AUTONOMY_RESUME]`` (case-insensitive),
       including mid-line or inside backticks, so models are not forced to emit a dedicated line.
    If both pause and resume markers appear, **pause wins**.
    """
    if not text or not gate_enabled():
        return None
    if not bool(config.get("system.autonomy_gate_agent_markers", True)):
        return None
    has_pause = False
    has_resume = False
    for ln in text.splitlines():
        if _line_matches_token(ln, _AGENT_PAUSE_LINES):
            has_pause = True
        if _line_matches_token(ln, _AGENT_RESUME_LINES):
            has_resume = True
    if has_pause:
        set_autonomous_pause_from_assistant(True, session_id)
        return "paused"
    if has_resume:
        set_autonomous_pause_from_assistant(False, session_id)
        return "resumed"
    # Bracketed autonomy tokens: substring match (aligned with user-side “contains” semantics).
    if re.search(r"\[S44_AUTONOMY_PAUSE\]", text, re.IGNORECASE):
        set_autonomous_pause_from_assistant(True, session_id)
        return "paused"
    if re.search(r"\[S44_AUTONOMY_RESUME\]", text, re.IGNORECASE):
        set_autonomous_pause_from_assistant(False, session_id)
        return "resumed"
    if re.search(r"【\s*S44_AUTONOMY_RESUME\s*】", text, re.IGNORECASE):
        set_autonomous_pause_from_assistant(False, session_id)
        return "resumed"
    if _BARE_AUTONOMY_RESUME_RE.search(text):
        set_autonomous_pause_from_assistant(False, session_id)
        return "resumed"
    return None


def apply_user_autonomy_command_from_text(session_id: str, text: str) -> Optional[str]:
    """
    Parse pause/resume intent from a **user** turn (not system reminders).

    Returns ``"paused"``, ``"resumed"``, or ``None``.
    """
    if not text or not gate_enabled():
        return None
    if not config.get("system.autonomy_gate_user_text_commands", True):
        return None
    t = text.strip()
    ppos = _user_pause_positions(t)
    rpos = _user_resume_positions(t)
    want_pause = bool(ppos)
    want_resume = bool(rpos)
    if want_pause and want_resume:
        # If both appear, the later command wins.
        last_pause = max(ppos)
        last_resume = max(rpos)
        if last_resume > last_pause:
            set_autonomous_pause_from_user(False, session_id)
            return "resumed"
        set_autonomous_pause_from_user(True, session_id)
        return "paused"
    if want_pause:
        set_autonomous_pause_from_user(True, session_id)
        return "paused"
    if want_resume:
        set_autonomous_pause_from_user(False, session_id)
        return "resumed"
    return None
