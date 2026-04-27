#!/usr/bin/env python3
"""
S work logger — structured self-telemetry under ``workspace/sandbox/s_logs``.

Cadence:
1. High frequency (~15 min): z_self-ish summary, energy, active task.
2. Mid frequency (~2 h): goals / todos / resource snapshot.
3. Daily (~24 h): narrative summary + reflection hooks.

Intended for the instance’s own continuity signals, not end-user product UI.
"""

import os
import json
import sqlite3
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Optional, Any, List

logger = logging.getLogger(__name__)


class SWorkLogger:
    """Persist timed telemetry rows (JSON / JSONL) plus SQLite tracker."""
    
    def __init__(self, db_path: str = "data.db", workspace_dir: str = "workspace/sandbox"):
        self.db_path = db_path
        self.workspace_dir = Path(workspace_dir)
        self.log_dir = self.workspace_dir / "s_logs"
        self._ensure_dirs()
        self._ensure_tables()
        
    def _ensure_dirs(self):
        """Create ``s_logs`` under the sandbox."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
    def _ensure_tables(self):
        """Create ``s_work_log_tracker`` if missing."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS s_work_log_tracker (
                        session_id TEXT PRIMARY KEY,
                        last_high_freq TEXT,
                        last_mid_freq TEXT,
                        last_daily TEXT,
                        high_freq_count INTEGER DEFAULT 0,
                        mid_freq_count INTEGER DEFAULT 0,
                        daily_count INTEGER DEFAULT 0
                    )
                """)
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to ensure s_work_log tables: {e}")
    
    def _get_tracker(self, session_id: str) -> Dict:
        """Load tracker row (bootstrap when absent)."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.execute(
                    "SELECT * FROM s_work_log_tracker WHERE session_id = ?",
                    (session_id,)
                )
                row = cur.fetchone()
                if row:
                    return dict(row)
                else:
                    # Bootstrap tracker
                    now = datetime.now(timezone.utc).isoformat()
                    conn.execute("""
                        INSERT INTO s_work_log_tracker 
                        (session_id, last_high_freq, last_mid_freq, last_daily,
                         high_freq_count, mid_freq_count, daily_count)
                        VALUES (?, ?, ?, ?, 0, 0, 0)
                    """, (session_id, now, now, now))
                    conn.commit()
                    return {
                        "session_id": session_id,
                        "last_high_freq": now,
                        "last_mid_freq": now,
                        "last_daily": now,
                        "high_freq_count": 0,
                        "mid_freq_count": 0,
                        "daily_count": 0
                    }
        except Exception as e:
            logger.error(f"Failed to get tracker: {e}")
            return {}
    
    def _update_tracker(self, session_id: str, field: str, count_field: str):
        """Bump last-run timestamp + counter."""
        try:
            now = datetime.now(timezone.utc).isoformat()
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(f"""
                    UPDATE s_work_log_tracker
                    SET {field} = ?, {count_field} = {count_field} + 1
                    WHERE session_id = ?
                """, (now, session_id))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to update tracker: {e}")
    
    # ==================== cadence gates ====================
    
    def should_log_high_freq(self, session_id: str) -> bool:
        """True when ≥15 minutes since the last high-frequency snapshot."""
        tracker = self._get_tracker(session_id)
        if not tracker:
            return True
        
        last = tracker.get("last_high_freq")
        if not last:
            return True
        
        try:
            last_time = datetime.fromisoformat(last.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            return (now - last_time).total_seconds() >= 900  # 15 minutes
        except Exception:
            return True
    
    def should_log_mid_freq(self, session_id: str) -> bool:
        """True when ≥2 hours since the last mid-frequency snapshot."""
        tracker = self._get_tracker(session_id)
        if not tracker:
            return True
        
        last = tracker.get("last_mid_freq")
        if not last:
            return True
        
        try:
            last_time = datetime.fromisoformat(last.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            return (now - last_time).total_seconds() >= 7200  # 2 hours
        except Exception:
            return True
    
    def should_log_daily(self, session_id: str) -> bool:
        """True when ≥24 hours since the last daily rollup."""
        tracker = self._get_tracker(session_id)
        if not tracker:
            return True
        
        last = tracker.get("last_daily")
        if not last:
            return True
        
        try:
            last_time = datetime.fromisoformat(last.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            return (now - last_time).total_seconds() >= 86400  # 24 hours
        except Exception:
            return True
    
    # ==================== writers ====================
    
    def log_high_freq(self, session_id: str, z_self_summary: Dict, energy: float, 
                      current_task: str = "", notes: str = ""):
        """Append one JSONL high-frequency row."""
        if not self.should_log_high_freq(session_id):
            return False
        
        now = datetime.now(timezone.utc)
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        
        # One JSONL file per UTC day
        log_file = self.log_dir / f"high_freq_{date_str}.jsonl"
        
        entry = {
            "timestamp": now.isoformat(),
            "time": time_str,
            "session_id": session_id,
            "type": "high_freq",
            "energy": round(energy, 2),
            "z_self": {
                "openness": round(z_self_summary.get("openness", 0), 3),
                "stability": round(z_self_summary.get("stability", 0), 3),
                "confidence": round(z_self_summary.get("confidence", 0), 3),
                "curiosity": round(z_self_summary.get("curiosity", 0), 3),
            },
            "current_task": current_task[:100] if current_task else "",
            "notes": notes[:200] if notes else ""
        }
        
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
            self._update_tracker(session_id, "last_high_freq", "high_freq_count")
            logger.debug(f"High freq log written for {session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to write high freq log: {e}")
            return False
    
    def log_mid_freq(self, session_id: str, project_status: Dict, goals: list,
                     resource_usage: Dict = None, todos: list = None):
        """Append one JSONL mid-frequency row."""
        if not self.should_log_mid_freq(session_id):
            return False
        
        now = datetime.now(timezone.utc)
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        
        log_file = self.log_dir / f"mid_freq_{date_str}.jsonl"
        
        entry = {
            "timestamp": now.isoformat(),
            "time": time_str,
            "session_id": session_id,
            "type": "mid_freq",
            "project_status": project_status,
            "active_goals": [
                {"title": g.get("title", ""), "progress": g.get("progress", 0)}
                for g in (goals or [])[:5]
            ],
            "resource_usage": resource_usage or {},
            "todos": todos or []
        }
        
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
            self._update_tracker(session_id, "last_mid_freq", "mid_freq_count")
            logger.debug(f"Mid freq log written for {session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to write mid freq log: {e}")
            return False
    
    def log_daily(self, session_id: str, summary: str, achievements: list = None,
                  challenges: list = None, next_day_plan: str = "",
                  self_reflection: str = ""):
        """Overwrite the daily JSON snapshot for the current UTC date."""
        if not self.should_log_daily(session_id):
            return False
        
        now = datetime.now(timezone.utc)
        date_str = now.strftime("%Y-%m-%d")
        
        log_file = self.log_dir / f"daily_{date_str}.json"
        
        entry = {
            "timestamp": now.isoformat(),
            "date": date_str,
            "session_id": session_id,
            "type": "daily",
            "summary": summary,
            "achievements": achievements or [],
            "challenges": challenges or [],
            "next_day_plan": next_day_plan,
            "self_reflection": self_reflection,
            "stats": self._get_day_stats(session_id, date_str)
        }
        
        try:
            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(entry, f, ensure_ascii=False, indent=2)
            
            self._update_tracker(session_id, "last_daily", "daily_count")
            logger.info(f"Daily log written for {session_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to write daily log: {e}")
            return False
    
    def _get_day_stats(self, session_id: str, date_str: str) -> Dict:
        """Aggregate same-day high-frequency energy samples."""
        stats = {
            "high_freq_logs": 0,
            "mid_freq_logs": 0,
            "avg_energy": 0,
            "energy_range": [0, 0]
        }
        
        # Scan today's high_freq JSONL for this session
        high_freq_file = self.log_dir / f"high_freq_{date_str}.jsonl"
        if high_freq_file.exists():
            try:
                energies = []
                count = 0
                with open(high_freq_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            entry = json.loads(line)
                            if entry.get("session_id") == session_id:
                                count += 1
                                energies.append(entry.get("energy", 0))
                        except Exception:
                            pass
                stats["high_freq_logs"] = count
                if energies:
                    stats["avg_energy"] = round(sum(energies) / len(energies), 2)
                    stats["energy_range"] = [min(energies), max(energies)]
            except Exception as e:
                logger.error(f"Failed to get day stats: {e}")
        
        return stats
    
    # ==================== readers ====================
    
    def get_recent_logs(self, session_id: str, log_type: str = "high_freq", 
                        limit: int = 20) -> list:
        """Return recent log entries for ``session_id`` (newest first)."""
        logs = []
        
        pattern = f"{log_type}_*.jsonl" if log_type != "daily" else f"{log_type}_*.json"
        files = sorted(self.log_dir.glob(pattern), reverse=True)[:3]
        
        for log_file in files:
            try:
                if log_type == "daily":
                    with open(log_file, "r", encoding="utf-8") as f:
                        entry = json.load(f)
                        if entry.get("session_id") == session_id:
                            logs.append(entry)
                else:
                    with open(log_file, "r", encoding="utf-8") as f:
                        for line in f:
                            try:
                                entry = json.loads(line)
                                if entry.get("session_id") == session_id:
                                    logs.append(entry)
                            except Exception:
                                pass
            except Exception as e:
                logger.error(f"Failed to read log file {log_file}: {e}")
        
        # Newest first, trim to limit
        logs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return logs[:limit]
    
    def get_log_summary(self, session_id: str) -> Dict:
        """Compact counters + last snapshot for prompt injection."""
        tracker = self._get_tracker(session_id)
        
        # Latest high-frequency row (if any)
        recent_high = self.get_recent_logs(session_id, "high_freq", limit=1)
        
        return {
            "total_high_freq": tracker.get("high_freq_count", 0),
            "total_mid_freq": tracker.get("mid_freq_count", 0),
            "total_daily": tracker.get("daily_count", 0),
            "last_high_freq": tracker.get("last_high_freq", ""),
            "last_mid_freq": tracker.get("last_mid_freq", ""),
            "last_daily": tracker.get("last_daily", ""),
            "recent_entry": recent_high[0] if recent_high else None
        }
    
    # ==================== research notes ====================
    
    def log_research_note(
        self, 
        session_id: str, 
        goal_id: str,
        goal_title: str,
        research_note: str,
        energy: float = 0,
        thinking_summary: str = ""
    ) -> bool:
        """Append a research episode note (JSONL)."""
        now = datetime.now(timezone.utc)
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")
        
        log_file = self.log_dir / f"research_{date_str}.jsonl"
        
        entry = {
            "timestamp": now.isoformat(),
            "time": time_str,
            "session_id": session_id,
            "type": "research",
            "goal_id": goal_id,
            "goal_title": goal_title,
            "research_note": research_note,
            "thinking_summary": thinking_summary,
            "energy": round(energy, 2),
        }
        
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
            logger.info(f"Research note logged for {session_id}: {goal_title[:30]}...")
            return True
        except Exception as e:
            logger.error(f"Failed to log research note: {e}")
            return False

    # ==================== legacy shim ====================

    def log_work(self, session_id: str, *args, **kwargs) -> bool:
        """
        Legacy entrypoint mapping ``log_work(...)`` → ``log_high_freq``.

        Keeps older background threads from crashing on missing methods.
        """
        payload: Dict[str, Any] = {}
        if args and isinstance(args[0], dict):
            payload.update(args[0])
        if kwargs:
            payload.update(kwargs)

        z_self_summary = payload.get("z_self_summary") or payload.get("z_self") or {}
        if not isinstance(z_self_summary, dict):
            z_self_summary = {}
        try:
            energy = float(payload.get("energy", 0.0) or 0.0)
        except Exception:
            energy = 0.0
        current_task = str(
            payload.get("current_task")
            or payload.get("task")
            or payload.get("task_name")
            or ""
        )
        notes = str(payload.get("notes") or payload.get("summary") or "")

        return self.log_high_freq(
            session_id=session_id,
            z_self_summary=z_self_summary,
            energy=energy,
            current_task=current_task,
            notes=notes,
        )
    
    def get_recent_research_notes(self, session_id: str, limit: int = 5) -> List[Dict]:
        """Return recent research JSONL rows for ``session_id``."""
        notes = []
        
        # Newest research files first
        files = sorted(self.log_dir.glob("research_*.jsonl"), reverse=True)[:3]
        
        for log_file in files:
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            entry = json.loads(line)
                            if entry.get("session_id") == session_id:
                                notes.append(entry)
                        except Exception:
                            pass
            except Exception as e:
                logger.error(f"Failed to read research log {log_file}: {e}")
        
        # Newest first
        notes.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        return notes[:limit]


# Process-wide singleton
_logger_instance = None

def get_work_logger(db_path: str = "data.db") -> SWorkLogger:
    """Shared ``SWorkLogger`` instance."""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = SWorkLogger(db_path)
    return _logger_instance

