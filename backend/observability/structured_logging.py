#!/usr/bin/env python3
"""
Structured logging helpers: emit one JSON object per line for easy parsing in core flows.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict

DEFAULT_LOGGER_NAME = "observability"


def _default_serializer(value: Any):
    """Best-effort coercion of odd values into JSON-serializable primitives."""
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:  # pragma: no cover - last-resort path
            pass
    # NumPy scalars / Decimal → float when possible
    try:
        return float(value)  # type: ignore[arg-type]
    except Exception:
        pass
    return str(value)


def emit_structured_log(
    logger: logging.Logger,
    event: str,
    level: str = "info",
    **fields: Dict[str, Any],
) -> None:
    """
    Merge ``event``, ``ts``, and ``fields`` into one JSON string and log at ``level``.
    """
    payload: Dict[str, Any] = {
        "event": event,
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    payload.update(fields)
    
    message = json.dumps(payload, ensure_ascii=False, default=_default_serializer)
    level = level.lower()
    log_fn = getattr(logger, level, logger.info)
    log_fn(message)


def get_structured_logger(name: str = DEFAULT_LOGGER_NAME) -> logging.Logger:
    """Return a namespaced logger (default ``observability``) for log aggregation."""
    return logging.getLogger(name)


