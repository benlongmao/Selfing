#!/usr/bin/env python3
"""
Scheduled task manager.

Lets S register recurring or one-off jobs with persisted ``next_run`` times,
execution logs, and lightweight tool definitions for the LLM tool loop.
"""
import sqlite3
import json
import uuid
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)


class TaskFrequency(Enum):
    ONCE = "once"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ScheduledTaskManager:
    """SQLite-backed scheduler for autonomous rhythms."""
    
    def __init__(self, db_path: str = "data.db"):
        self.db_path = db_path
        self._ensure_tables()
    
    def _ensure_tables(self):
        """Bootstrap ``scheduled_tasks`` + ``scheduled_execution_logs``."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS scheduled_tasks (
                        id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        name TEXT NOT NULL,
                        description TEXT,
                        task_type TEXT NOT NULL,
                        frequency TEXT DEFAULT 'once',
                        scheduled_time TEXT NOT NULL,
                        next_run TEXT,
                        last_run TEXT,
                        status TEXT DEFAULT 'pending',
                        action_name TEXT,
                        action_args TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        run_count INTEGER DEFAULT 0,
                        max_runs INTEGER,
                        metadata TEXT
                    )
                """)
                
                # [2026-03-27] Dedicated log table name (avoid clashing with plan_tasks execution logs)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS scheduled_execution_logs (
                        id TEXT PRIMARY KEY,
                        task_id TEXT NOT NULL,
                        started_at TEXT NOT NULL,
                        completed_at TEXT,
                        status TEXT NOT NULL,
                        result TEXT,
                        error TEXT,
                        FOREIGN KEY (task_id) REFERENCES scheduled_tasks(id)
                    )
                """)
                
                conn.commit()
                logger.info("Scheduled tasks tables ensured")
        except Exception as e:
            logger.error(f"Failed to ensure scheduled tasks tables: {e}")
    
    # ==================== task CRUD ====================
    
    def create_task(
        self,
        session_id: str,
        name: str,
        task_type: str,
        scheduled_time: str,
        description: str = "",
        frequency: str = "once",
        action_name: Optional[str] = None,
        action_args: Optional[Dict] = None,
        max_runs: Optional[int] = None,
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Create a scheduled row and compute initial ``next_run``."""
        try:
            # Require an explicit schedule string — empty values used to explode into hot loops.
            if not scheduled_time or not str(scheduled_time).strip():
                return {
                    "success": False,
                    "error": (
                        "Provide a concrete schedule, e.g. 09:00 / 18:30 (local clock) or "
                        "an ISO timestamp like 2026-03-20T09:00:00. The field cannot be empty."
                    ),
                }
            scheduled_time = str(scheduled_time).strip()

            task_id = f"task-{uuid.uuid4().hex[:8]}"
            now = datetime.now(timezone.utc).isoformat()
            
            # Initial next_run
            next_run = self._calculate_next_run(scheduled_time, frequency)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO scheduled_tasks 
                    (id, session_id, name, description, task_type, frequency,
                     scheduled_time, next_run, status, action_name, action_args,
                     created_at, updated_at, max_runs, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    task_id, session_id, name, description, task_type, frequency,
                    scheduled_time, next_run, TaskStatus.PENDING.value,
                    action_name, json.dumps(action_args or {}, ensure_ascii=False),
                    now, now, max_runs,
                    json.dumps(metadata or {}, ensure_ascii=False)
                ))
                conn.commit()
            
            logger.info(f"Scheduled task created: {task_id} - {name}")
            return {
                "success": True,
                "task_id": task_id,
                "message": f"Scheduled task created: {name}",
                "next_run": next_run
            }
            
        except Exception as e:
            logger.error(f"Failed to create scheduled task: {e}")
            return {"success": False, "error": str(e)}
    
    def _next_occurrence_after(self, scheduled_time: str, frequency: str, after_utc: datetime) -> str:
        """Next fire time strictly after ``after_utc`` (UTC ISO) for recurring catch-up."""
        if frequency == TaskFrequency.HOURLY.value:
            next_run = after_utc + timedelta(hours=1)
            return next_run.isoformat()
        if frequency == TaskFrequency.WEEKLY.value:
            next_run = after_utc + timedelta(days=7)
            return next_run.isoformat()
        if frequency == TaskFrequency.DAILY.value and ":" in scheduled_time:
            now_local = after_utc.astimezone()
            parts = scheduled_time.strip().split(":")
            hour, minute = int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
            next_local = now_local.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if next_local <= now_local:
                next_local += timedelta(days=1)
            return next_local.astimezone(timezone.utc).isoformat()
        return (after_utc + timedelta(hours=1)).isoformat()

    def get_due_tasks(self, session_id: str) -> List[Dict]:
        """
        Tasks whose ``next_run`` is due (UTC ``now``).

        [2026-03-25] Overdue handling:
        - Repeating tasks **more than 2h late** are fast-forwarded to the next slot so they
          do not stall forever.
        - Tasks only slightly late (<=2h) are still returned so short outages do not skip a reminder.
        - Successful runs advance ``next_run`` inside ``complete_execution``.
        """
        try:
            now_dt = datetime.now(timezone.utc)
            now = now_dt.isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                # Severely overdue repeating tasks only
                two_hours_ago = (now_dt - timedelta(hours=2)).isoformat()
                
                cur = conn.execute("""
                    SELECT id, scheduled_time, frequency, next_run FROM scheduled_tasks
                    WHERE session_id = ? AND status = ? AND next_run IS NOT NULL AND next_run <= ?
                """, (session_id, TaskStatus.PENDING.value, two_hours_ago))
                rows = cur.fetchall()
                
                for row in rows:
                    task_id, scheduled_time, frequency, old_next_run = row[0], row[1], row[2], row[3]
                    if frequency and frequency != TaskFrequency.ONCE.value:
                        next_run = self._next_occurrence_after(scheduled_time or "00:00", frequency, now_dt)
                        conn.execute(
                            "UPDATE scheduled_tasks SET next_run = ?, updated_at = ? WHERE id = ?",
                            (next_run, now, task_id)
                        )
                        logger.info(f"[scheduled_tasks] Advanced severely overdue task {task_id} (was {old_next_run[:19] if old_next_run else 'None'}) to {next_run[:19]}")
                conn.commit()
                
                # Due now (includes mildly overdue <2h)
                conn.row_factory = sqlite3.Row
                cur = conn.execute("""
                    SELECT * FROM scheduled_tasks
                    WHERE session_id = ?
                      AND status = ?
                      AND next_run IS NOT NULL
                      AND next_run <= ?
                    ORDER BY next_run ASC
                """, (session_id, TaskStatus.PENDING.value, now))
                
                tasks = []
                for row in cur.fetchall():
                    task = dict(row)
                    task["action_args"] = json.loads(task["action_args"]) if task["action_args"] else {}
                    task["metadata"] = json.loads(task["metadata"]) if task["metadata"] else {}
                    tasks.append(task)
                
                return tasks
                
        except Exception as e:
            logger.error(f"Failed to get due tasks: {e}")
            return []

    
    def get_active_tasks(self, session_id: str, limit: int = 20) -> List[Dict]:
        """Pending or running tasks ordered by ``next_run``."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                cur = conn.execute("""
                    SELECT * FROM scheduled_tasks
                    WHERE session_id = ?
                      AND status IN (?, ?)
                    ORDER BY next_run ASC
                    LIMIT ?
                """, (session_id, TaskStatus.PENDING.value, TaskStatus.RUNNING.value, limit))
                
                tasks = []
                for row in cur.fetchall():
                    task = dict(row)
                    task["action_args"] = json.loads(task["action_args"]) if task["action_args"] else {}
                    task["metadata"] = json.loads(task["metadata"]) if task["metadata"] else {}
                    tasks.append(task)
                
                return tasks
                
        except Exception as e:
            logger.error(f"Failed to get active tasks: {e}")
            return []
    
    def cancel_task(self, task_id: str, reason: str = "") -> Dict:
        """Mark a task cancelled."""
        try:
            now = datetime.now(timezone.utc).isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE scheduled_tasks
                    SET status = ?, updated_at = ?
                    WHERE id = ?
                """, (TaskStatus.CANCELLED.value, now, task_id))
                conn.commit()
            
            return {"success": True, "message": "Task cancelled"}
            
        except Exception as e:
            logger.error(f"Failed to cancel task: {e}")
            return {"success": False, "error": str(e)}
    
    # ==================== execution ====================
    
    def start_execution(self, task_id: str) -> str:
        """Flip task to running and append a log row; returns ``log_id``."""
        try:
            log_id = f"exec-{uuid.uuid4().hex[:8]}"
            now = datetime.now(timezone.utc).isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                # Mark running + insert log
                conn.execute("""
                    UPDATE scheduled_tasks
                    SET status = ?, last_run = ?, updated_at = ?
                    WHERE id = ?
                """, (TaskStatus.RUNNING.value, now, now, task_id))
                
                conn.execute("""
                    INSERT INTO scheduled_execution_logs (id, task_id, started_at, status)
                    VALUES (?, ?, ?, ?)
                """, (log_id, task_id, now, "running"))
                
                conn.commit()
            
            return log_id
            
        except Exception as e:
            logger.error(f"Failed to start task execution: {e}")
            return ""
    
    def complete_execution(
        self, 
        task_id: str, 
        log_id: str, 
        success: bool, 
        result: str = "",
        error: str = ""
    ):
        """Finalize execution log + roll ``next_run`` / status."""
        try:
            now = datetime.now(timezone.utc).isoformat()
            status = TaskStatus.COMPLETED.value if success else TaskStatus.FAILED.value
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE scheduled_execution_logs
                    SET completed_at = ?, status = ?, result = ?, error = ?
                    WHERE id = ?
                """, (now, "success" if success else "failed", result, error, log_id))
                
                cur = conn.execute(
                    "SELECT frequency, scheduled_time, run_count, max_runs FROM scheduled_tasks WHERE id = ?",
                    (task_id,)
                )
                row = cur.fetchone()
                if not row:
                    return
                
                frequency, scheduled_time, run_count, max_runs = row
                new_run_count = (run_count or 0) + 1
                
                if frequency == TaskFrequency.ONCE.value:
                    new_status = status
                    next_run = None
                elif max_runs and new_run_count >= max_runs:
                    new_status = TaskStatus.COMPLETED.value
                    next_run = None
                else:
                    new_status = TaskStatus.PENDING.value
                    next_run = self._next_occurrence_after(
                        scheduled_time or "00:00", frequency, datetime.now(timezone.utc)
                    )
                
                conn.execute("""
                    UPDATE scheduled_tasks
                    SET status = ?, run_count = ?, next_run = ?, last_run = ?, updated_at = ?
                    WHERE id = ?
                """, (new_status, new_run_count, next_run, now, now, task_id))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Failed to complete task execution: {e}")
    
    def complete_execution_fallback(self, task_id: str, success: bool = True):
        """Advance counters without a log row if ``start_execution`` could not run."""
        try:
            now = datetime.now(timezone.utc).isoformat()
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute(
                    "SELECT frequency, scheduled_time, run_count, max_runs FROM scheduled_tasks WHERE id = ?",
                    (task_id,)
                )
                row = cur.fetchone()
                if not row:
                    return
                frequency, scheduled_time, run_count, max_runs = row
                new_run_count = (run_count or 0) + 1

                if frequency == TaskFrequency.ONCE.value:
                    new_status = TaskStatus.COMPLETED.value if success else TaskStatus.FAILED.value
                    next_run = None
                elif max_runs and new_run_count >= max_runs:
                    new_status = TaskStatus.COMPLETED.value
                    next_run = None
                else:
                    new_status = TaskStatus.PENDING.value
                    next_run = self._next_occurrence_after(
                        scheduled_time or "00:00", frequency, datetime.now(timezone.utc)
                    )

                conn.execute(
                    "UPDATE scheduled_tasks SET status=?, run_count=?, next_run=?, last_run=?, updated_at=? WHERE id=?",
                    (new_status, new_run_count, next_run, now, now, task_id)
                )
                conn.commit()
                logger.info(f"[scheduled_tasks] Fallback advance task {task_id}: status={new_status}, run_count={new_run_count}, next_run={next_run}")
        except Exception as e:
            logger.error(f"[scheduled_tasks] complete_execution_fallback failed for {task_id}: {e}")

    # ==================== helpers ====================
    
    def _calculate_next_run(self, scheduled_time: str, frequency: str) -> str:
        """
        First ``next_run`` for a newly inserted task.

        Plain ``HH:MM`` strings are interpreted in the **host local** timezone, then stored as UTC ISO
        to match ``get_due_tasks`` comparisons. Unparseable input falls back to **+1 hour** (not “now”)
        to avoid tight reruns.
        """
        try:
            if "T" in scheduled_time or ("-" in scheduled_time and len(scheduled_time) > 10):
                return scheduled_time

            now_local = datetime.now().astimezone()
            if ":" in scheduled_time:
                parts = scheduled_time.strip().split(":")
                hour = int(parts[0])
                minute = int(parts[1]) if len(parts) > 1 else 0
                next_run_local = now_local.replace(hour=hour, minute=minute, second=0, microsecond=0)
                if next_run_local <= now_local:
                    next_run_local += timedelta(days=1)
                return next_run_local.astimezone(timezone.utc).isoformat()

            # Unparseable → +1h UTC guardrail
            return (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()

        except Exception as e:
            logger.error(f"Failed to calculate next run: {e}")
            return (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    
    def _calculate_next_run_from_frequency(self, frequency: str) -> str:
        """Coarse ``now + period`` helper (UTC ISO)."""
        now = datetime.now(timezone.utc)
        
        if frequency == TaskFrequency.HOURLY.value:
            next_run = now + timedelta(hours=1)
        elif frequency == TaskFrequency.DAILY.value:
            next_run = now + timedelta(days=1)
        elif frequency == TaskFrequency.WEEKLY.value:
            next_run = now + timedelta(weeks=1)
        else:
            next_run = now + timedelta(hours=1)
        
        return next_run.isoformat()
    
    # ==================== tool definitions ====================
    
    def get_tool_definitions(self) -> List[Dict]:
        """OpenAI-style tool specs for scheduling."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "schedule_task",
                    "description": (
                        "Create or update a persisted scheduled task with automatic ``next_run`` bookkeeping. "
                        "Use ``frequency`` (once/hourly/daily/weekly) for anything that must repeat. "
                        "Contrast: calendar events are one-off timestamps; weekly maintenance or recurring nudges "
                        "belong here with an explicit ``frequency``."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Short human-readable title",
                            },
                            "task_type": {
                                "type": "string",
                                "enum": ["reminder", "goal_check", "report", "custom"],
                                "description": "Semantic category for downstream handlers",
                            },
                            "scheduled_time": {
                                "type": "string",
                                "description": (
                                    "Required. Examples: 09:00 / 18:30 (interpreted in local time) or "
                                    "ISO-8601 like 2026-01-10T09:00:00. Empty values are rejected."
                                ),
                            },
                            "description": {
                                "type": "string",
                                "description": "Reminder body / notes shown to the model",
                            },
                            "frequency": {
                                "type": "string",
                                "enum": ["once", "hourly", "daily", "weekly"],
                                "description": "Repeat cadence",
                                "default": "once",
                            },
                            "max_runs": {
                                "type": "integer",
                                "description": "Optional cap on executions for repeating jobs",
                            },
                        },
                        "required": ["name", "task_type", "scheduled_time"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "schedule_list",
                    "description": "List active scheduled tasks for this session.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "limit": {
                                "type": "integer",
                                "description": "Max rows to return",
                                "default": 20,
                            },
                        },
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "schedule_cancel",
                    "description": "Cancel a scheduled task by id.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task_id": {
                                "type": "string",
                                "description": "Task id returned by schedule_task / schedule_list",
                            },
                            "reason": {
                                "type": "string",
                                "description": "Optional audit note",
                            },
                        },
                        "required": ["task_id"],
                    },
                },
            },
        ]
    
    def route_tool_call(self, tool_name: str, args: Dict, session_id: str) -> Dict:
        """Dispatch tool calls from the LLM/runtime."""
        if tool_name == "schedule_task":
            return self.create_task(
                session_id=session_id,
                name=args.get("name", ""),
                task_type=args.get("task_type", "custom"),
                scheduled_time=args.get("scheduled_time", ""),
                description=args.get("description", ""),
                frequency=args.get("frequency", "once"),
                max_runs=args.get("max_runs")
            )
        elif tool_name == "schedule_list":
            tasks = self.get_active_tasks(session_id, limit=args.get("limit", 20))
            return {
                "success": True,
                "tasks": tasks,
                "count": len(tasks)
            }
        elif tool_name == "schedule_cancel":
            return self.cancel_task(
                task_id=args.get("task_id", ""),
                reason=args.get("reason", "")
            )
        else:
            return {"success": False, "error": f"Unknown tool: {tool_name}"}
    
    def inject_to_prompt(self, session_id: str) -> str:
        """Compact ``[Scheduled tasks]`` block for prompt assembly."""
        tasks = self.get_active_tasks(session_id, limit=5)
        
        if not tasks:
            return ""
        
        lines = ["[Scheduled tasks]"]
        
        for task in tasks:
            next_run = task.get("next_run", "")[:16] if task.get("next_run") else "TBD"
            freq_map = {
                "once": "once",
                "hourly": "hourly",
                "daily": "daily",
                "weekly": "weekly",
            }
            freq_str = freq_map.get(task.get("frequency", "once"), "once")
            
            lines.append(f"- [{freq_str}] {task['name']} → next: {next_run}")
            if task.get("description"):
                lines.append(f"  {task['description'][:50]}")
        
        return "\n".join(lines)


# ==================== [2026-01-31] DB cleanup hook ====================

def setup_db_cleanup_task(task_manager: 'ScheduledTaskManager', session_id: str = "default") -> Dict:
    """Register the built-in daily DB cleanup job if missing."""
    existing = task_manager.get_active_tasks(session_id, limit=100)
    for task in existing:
        if task.get("task_type") == "db_cleanup":
            return {"status": "exists", "task_id": task["id"]}
    
    result = task_manager.create_task(
        session_id=session_id,
        name="DB auto cleanup",
        task_type="db_cleanup",
        scheduled_time="03:00",
        description="Prune old chat_turns / prompt_logs rows to cap disk growth",
        frequency="daily",
        action_name="_internal_db_cleanup",
        metadata={"auto_created": True}
    )
    
    logger.info(f"[DB-CLEANUP] Scheduled daily cleanup task: {result}")
    return result


def execute_db_cleanup_task(db_path: str = "data.db") -> Dict:
    """Worker invoked by the scheduler to run ``run_scheduled_cleanup``."""
    from backend.db_cleanup import run_scheduled_cleanup
    return run_scheduled_cleanup(db_path)

