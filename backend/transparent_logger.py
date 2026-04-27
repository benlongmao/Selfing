#!/usr/bin/env python3
"""
Transparent operation log — records every file operation the agent performs.

[2026-02-05] Design intent:
- A bridge of trust, not a control shackle
- A learning aid, not surveillance theater
- A mirror for self-improvement

Agent-authored, transparency-first safety posture.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class TransparentLogger:
    """Append-only logger for transparent file operations."""

    def __init__(self, log_dir: str = "workspace/sandbox/logs/transparent_operations"):
        """
        Args:
            log_dir: Directory relative to the project root where JSON + markdown logs live.

        [2026-02-05 fix] Resolve paths from this file so cwd does not matter.
        """
        # Project root = parent of backend/
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.log_dir = os.path.join(project_root, log_dir)
        self._ensure_log_dir()

    def _ensure_log_dir(self):
        """Create the log directory if missing."""
        try:
            os.makedirs(self.log_dir, exist_ok=True)
        except Exception as e:
            logger.error(f"[TransparentLogger] Failed to create log directory: {e}")

    def log_operation(
        self,
        operation_type: str,
        file_path: str,
        motivation: Optional[str] = None,
        z_self_state: Optional[Dict] = None,
        result: Optional[Dict] = None,
        fuse_triggered: bool = False,
        fuse_level: int = 0,
        user_override_reason: Optional[str] = None,
    ) -> str:
        """
        Persist one file operation.

        Args:
            operation_type: e.g. write / read / delete
            file_path: Target path
            motivation: Optional human-readable reason
            z_self_state: Optional z_self snapshot
            result: Optional structured outcome
            fuse_triggered: Whether a fuse rule fired
            fuse_level: Fuse severity (0–3)
            user_override_reason: Optional user override explanation

        Returns:
            Path to the JSON log file, or empty string on failure.
        """
        timestamp = datetime.now()
        log_entry = {
            "timestamp": timestamp.isoformat(),
            "operation_type": operation_type,
            "file_path": file_path,
            "motivation": motivation,
            "z_self_state": z_self_state,
            "fuse_triggered": fuse_triggered,
            "fuse_level": fuse_level,
            "user_override_reason": user_override_reason,
            "result": result,
        }

        log_filename = f"{timestamp.strftime('%Y%m%d_%H%M%S')}_{operation_type}_{Path(file_path).name}.json"
        log_path = os.path.join(self.log_dir, log_filename)

        try:
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(log_entry, f, indent=2, ensure_ascii=False)

            self._append_to_main_log(log_entry)

            logger.info(f"[TransparentLogger] Logged operation: {log_path}")
            return log_path

        except Exception as e:
            logger.error(f"[TransparentLogger] Failed to write log: {e}")
            return ""

    def _append_to_main_log(self, log_entry: Dict):
        """Append a human-readable Markdown row to ``operations_log.md``."""
        main_log_path = os.path.join(self.log_dir, "operations_log.md")

        try:
            fuse_yes = "Yes" if log_entry.get("fuse_triggered") else "No"
            markdown_entry = f"""
---

## {log_entry['timestamp']} - {log_entry['operation_type']}

**File path**: `{log_entry['file_path']}`

**Motivation**: {log_entry.get('motivation', 'N/A')}

**State snapshot**:
```json
{json.dumps(log_entry.get('z_self_state', {}), indent=2, ensure_ascii=False)}
```

**Fuse**:
- Triggered: {fuse_yes}
- Level: Level {log_entry.get('fuse_level', 0)}
- Override reason: {log_entry.get('user_override_reason', 'N/A')}

**Result**:
```json
{json.dumps(log_entry.get('result', {}), indent=2, ensure_ascii=False)}
```

"""

            if not os.path.exists(main_log_path):
                with open(main_log_path, 'w', encoding='utf-8') as f:
                    f.write("# Agent transparent operation log\n\n")
                    f.write("**Design**: trust bridge, learning aid, self-improvement mirror\n\n")
                    f.write(f"**Created**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            with open(main_log_path, 'a', encoding='utf-8') as f:
                f.write(markdown_entry)

        except Exception as e:
            logger.error(f"[TransparentLogger] Failed to append to main log: {e}")

    def get_recent_logs(self, limit: int = 10) -> List[Dict]:
        """
        Load the newest JSON log entries (newest first, capped by ``limit``).
        """
        try:
            log_files = sorted(
                [f for f in os.listdir(self.log_dir) if f.endswith('.json')],
                reverse=True
            )

            logs = []
            for log_file in log_files[:limit]:
                log_path = os.path.join(self.log_dir, log_file)
                with open(log_path, 'r', encoding='utf-8') as f:
                    logs.append(json.load(f))

            return logs

        except Exception as e:
            logger.error(f"[TransparentLogger] Failed to get recent logs: {e}")
            return []

    def analyze_patterns(self, days: int = 7) -> Dict:
        """
        Placeholder for future pattern mining over recent operations.

        Args:
            days: Look-back window (not yet wired).

        Returns:
            Stub payload until Phase 3 analytics land.
        """
        # TODO: fuse correlation by z_self bands, post-warning outcomes, risky op clusters
        return {
            "status": "not_implemented",
            "message": "Pattern analysis is planned for Phase 3"
        }


_transparent_logger: Optional[TransparentLogger] = None


def get_transparent_logger() -> TransparentLogger:
    """Lazy singleton accessor."""
    global _transparent_logger
    if _transparent_logger is None:
        _transparent_logger = TransparentLogger()
    return _transparent_logger
