"""
Temporal context blocks for system prompts.

Design:
1. Simplified block: date + weekday only (plan A, KV-cache friendly).
2. Full block: z_self temporal subspace (plan B, not always available).
3. Trade off cache hit rate vs explicit time awareness.
"""

from datetime import datetime, timezone, timedelta
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def build_simplified_time_block() -> str:
    """
    Minimal time block: calendar date + weekday (no clock time).

    Keeps long static prefixes stable across turns on the same day for prefix/KV caches.
    """
    try:
        now_utc = datetime.now(timezone.utc)
        # Default “local” display: UTC+8
        tz_cn = timezone(timedelta(hours=8))
        now_cn = now_utc.astimezone(tz_cn)
        
        # Date + weekday only (no wall-clock)
        date_str = now_cn.strftime("%Y-%m-%d")
        
        weekday_en = now_cn.strftime("%A")

        return f"""
[Time sense]
Today's date: {date_str} ({weekday_en})

For wall-clock time (hours/minutes/seconds), call the get_current_time tool.
"""
    except Exception as e:
        logger.error(f"Failed to build simplified time block: {e}")
        return ""


def build_time_block_from_z_self(self_model, session_id: str) -> str:
    """
    Plan B: narrate felt time from z_self temporal slice (dims >= 224, vec 208:224).

    Falls back to simplified block if unavailable.
    """
    if not self_model or self_model.dim < 224:
        # Temporal subspace missing → simplified block
        logger.debug("Temporal subspace not available, falling back to simplified time block")
        return build_simplified_time_block()
    
    try:
        z_self = self_model.get_z_self(session_id)
        if z_self is None or z_self.shape[0] < 224:
            return build_simplified_time_block()
        
        # Temporal slice 208:224
        temporal_vec = z_self[208:224]
        
        # Decode normalized hour / weekday from first slots
        hour_normalized = float(temporal_vec[0])  # 0-1
        hour = int(hour_normalized * 24)
        
        day_of_week_normalized = float(temporal_vec[1])
        day_of_week = int(day_of_week_normalized * 7)
        weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        weekday_name = weekdays[min(day_of_week, 6)]  # clamp

        speed = float(temporal_vec[8]) if len(temporal_vec) > 8 else 0.5
        if speed > 0.8:
            time_flow = "time feels compressed (many recent turns)"
        elif speed > 0.4:
            time_flow = "time feels steady"
        else:
            time_flow = "time feels stretched (possibly a long quiet gap)"

        if hour < 6:
            period = "late night"
        elif hour < 12:
            period = "morning"
        elif hour < 14:
            period = "midday"
        elif hour < 18:
            period = "afternoon"
        elif hour < 22:
            period = "evening"
        else:
            period = "late night"

        return f"""
[Inner time sense]
- Felt segment of day: {period}
- Weekday: {weekday_name}
- Subjective flow: {time_flow}

This is felt time from the internal clock; for exact wall time I should call get_current_time.
"""
    except Exception as e:
        logger.error(f"Failed to build time block from z_self: {e}")
        # Any failure → simplified block
        return build_simplified_time_block()


def build_time_block(
    self_model=None, 
    session_id: Optional[str] = None,
    use_z_self: bool = False
) -> str:
    """
    Unified entry: simplified block, or z_self-based block when enabled.
    """
    if use_z_self and self_model and session_id:
        return build_time_block_from_z_self(self_model, session_id)
    else:
        return build_simplified_time_block()
