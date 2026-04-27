from fastapi import APIRouter
from fastapi.responses import Response
from datetime import datetime, timezone
from typing import Optional, Dict, Any
import json

from backend.database import DB_PATH
from backend.event_logger import EventLogger
from backend.config import config

router = APIRouter(tags=["System"])

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _frontend_fetch_timeout_ms() -> int:
    _default = 3600000  # 1h; matches default parameters.chat.frontend_fetch_timeout_ms
    raw = config.get("parameters.chat.frontend_fetch_timeout_ms", _default)
    try:
        ms = int(raw) if raw is not None else _default
    except (TypeError, ValueError):
        ms = _default
    return max(ms, 60000)


@router.get("/api/config.js")
def get_config_js():
    """Inline JS config for the browser (agent name, fetch timeout)."""
    agent_name = config.get("system.agent_name", "Agent")
    ft_ms = _frontend_fetch_timeout_ms()
    payload = json.dumps(
        {"agent_name": agent_name, "frontend_fetch_timeout_ms": ft_ms},
        ensure_ascii=False,
    )
    content = f"window.__AGENT_CONFIG__ = {payload};"
    return Response(content=content, media_type="application/javascript; charset=utf-8")


@router.get("/health")
def health():
    return {"ok": True, "ts": now_iso()}


@router.get("/api/system/config")
def get_system_config():
    """System fields for the UI (e.g. agent_name for chat history)."""
    return {
        "agent_name": config.get("system.agent_name", "Agent"),
        "frontend_fetch_timeout_ms": _frontend_fetch_timeout_ms(),
    }


@router.post("/maintenance/cleanup_reflection_artifacts")
def cleanup_reflection_artifacts(body: Optional[Dict[str, Any]] = None):
    """
    Clean reflection pipeline artifacts (does not delete persona_items rules).
    Defaults: keep last 200 reflection event_logs per session; trim chat_turns.reflection older than 30 days.
    """
    body = body or {}
    keep_last = int(body.get("keep_last_reflection_events_per_session", 200) or 200)
    older_days = int(body.get("older_than_days", 30) or 30)
    clear_reflection = bool(body.get("clear_chat_turns_reflection", True))
    clear_introspection = bool(body.get("clear_chat_turns_introspection", False))

    logger = EventLogger(DB_PATH)
    return logger.cleanup_reflection_artifacts(
        keep_last_reflection_events_per_session=keep_last,
        clear_chat_turns_reflection=clear_reflection,
        clear_chat_turns_introspection=clear_introspection,
        older_than_days=older_days,
    )

