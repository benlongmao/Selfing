#!/usr/bin/env python3
"""
S44 evolution helpers.

Incremental improvements around SelfTick-style flows without rewriting the core loop.
Prefer small, observable changes over large risky edits.
"""

import os
import json
import logging
import re
from datetime import datetime
from typing import Dict, Any

from backend.utils.path_utils import get_workspace_root

logger = logging.getLogger(__name__)

class S44Evolution:
    """Optional diagnostics / heuristics for idle detection and autonomous continuation."""

    def __init__(self, db_path: str = "data.db"):
        self.db_path = db_path
        logger.info("S44Evolution module initialized")

    def enhance_idle_detection(self, session_id: str, trigger_reason: str) -> Dict[str, Any]:
        """
        Enrich idle detection with structured counters for logging.

        Args:
            session_id: Active session id.
            trigger_reason: Why the check ran (e.g. ``scheduled``, ``manual``, ``idle``).

        Returns:
            Dict with pending task counts, scheduled backlog, and recent chat volume.
        """
        result = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "trigger_reason": trigger_reason,
            "enhanced_logging": True,
            "pending_tasks_count": 0,
            "scheduled_tasks_count": 0,
            "recent_conversations_count": 0
        }

        # 1) HEARTBEAT.md numbered tasks (path anchored to workspace root)
        try:
            heartbeat_path = os.path.join(get_workspace_root(), "HEARTBEAT.md")
            if os.path.exists(heartbeat_path):
                with open(heartbeat_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    pending_items = re.findall(r'^\d+\.\s+.*', content, re.MULTILINE)
                    result["pending_tasks_count"] = len(pending_items)

                    cn_hp = re.findall(r'优先级\s*[:：].*[89]', content)
                    en_hp = re.findall(
                        r'Priority\s*[:：].*(?:[89]|10)\b', content, re.IGNORECASE
                    )
                    result["high_priority_tasks"] = len(cn_hp) + len(en_hp)
            else:
                logger.debug("HEARTBEAT.md not found")
        except Exception as e:
            logger.debug(f"Pending-task scan failed: {e}")

        # 2) Due scheduled tasks
        try:
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM scheduled_tasks WHERE executed = 0 AND scheduled_time <= datetime('now')"
            )
            result["scheduled_tasks_count"] = cursor.fetchone()[0]
            conn.close()
        except Exception as e:
            logger.debug(f"Scheduled-task probe failed: {e}")

        # 3) Recent conversation volume
        try:
            from backend.memory import ConversationMemory
            mem = ConversationMemory(self.db_path)
            recent_convs = mem.get_recent_conversations(session_id, limit=5)
            result["recent_conversations_count"] = len(recent_convs)
        except Exception as e:
            logger.debug(f"Recent conversation fetch failed: {e}")

        logger.info(f"[S44] Idle probe at {result['timestamp']}")
        logger.info(f"[S44] session={session_id}, trigger={trigger_reason}")
        logger.info(
            f"[S44] pending_tasks={result['pending_tasks_count']}, "
            f"high_priority={result.get('high_priority_tasks', 0)}"
        )
        logger.info(f"[S44] due_scheduled_tasks={result['scheduled_tasks_count']}")
        logger.info(f"[S44] recent_conversations={result['recent_conversations_count']}")

        return result

    def should_trigger_autonomous_continuation(self, session_id: str, trigger_reason: str) -> bool:
        """
        Lightweight policy hint for autonomous continuation.

        Heuristics:
        1. High-priority pending tasks in HEARTBEAT.md
        2. Due scheduled tasks exist
        3. Idle trigger with zero very-recent user turns
        """
        detection_info = self.enhance_idle_detection(session_id, trigger_reason)

        if detection_info.get("high_priority_tasks", 0) > 0:
            logger.info("[S44] High-priority backlog detected — continuation recommended")
            return True

        if detection_info.get("scheduled_tasks_count", 0) > 0:
            logger.info("[S44] Due scheduled tasks detected — continuation recommended")
            return True

        if trigger_reason == "idle" and detection_info.get("recent_conversations_count", 0) == 0:
            logger.info("[S44] Idle with no recent user traffic — continuation recommended")
            return True

        return False

    def create_evolution_marker(self):
        """Write a JSON marker describing this evolution slice (debug/audit)."""
        marker = {
            "evolution_id": "s44_evolution_v1",
            "timestamp": datetime.now().isoformat(),
            "description": "S44 evolution helper: richer idle detection + continuation hints",
            "changes": [
                "Introduced S44Evolution helper class",
                "Expanded idle logging around HEARTBEAT + scheduler + memory",
                "Added continuation heuristics",
            ],
            "principles": [
                "Small, reversible steps",
                "Avoid rewriting core tick loops blindly",
                "Prefer observable telemetry",
            ]
        }

        marker_path = os.path.join(
            get_workspace_root(), "evolution_markers", "s44_evolution_v1.json"
        )
        os.makedirs(os.path.dirname(marker_path), exist_ok=True)
        with open(marker_path, 'w', encoding='utf-8') as f:
            json.dump(marker, f, indent=2, ensure_ascii=False)

        logger.info(f"[S44] Evolution marker written: {marker_path}")
        return marker
