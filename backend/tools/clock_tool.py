#!/usr/bin/env python3
"""
Clock tool: expose current UTC time plus a fixed-offset \"display\" timezone for prompts.
"""
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

# Default display offset (UTC+8). Production deployments may override via config later.
_DISPLAY_OFFSET = timedelta(hours=8)


class ClockTool:
    def __init__(self):
        pass

    def get_current_time(self) -> Dict[str, Any]:
        """Return machine time in ISO UTC plus a human-readable offset line."""
        now_utc = datetime.now(timezone.utc)
        tz_display = timezone(_DISPLAY_OFFSET)
        now_display = now_utc.astimezone(tz_display)

        return {
            "success": True,
            "iso_utc": now_utc.isoformat(),
            "local_readable": now_display.strftime("%Y-%m-%d %H:%M:%S %A (UTC+8)"),
            "timestamp": now_utc.timestamp(),
        }

    def get_tool_definitions(self) -> List[Dict]:
        """OpenAI-style tool schema."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_current_time",
                    "description": "Get the current real-world time and date. Use this to timestamp your memories, logs, or understand temporal context.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                },
            }
        ]
