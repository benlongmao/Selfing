"""
Task executor — multi-step plans with dependencies, retries, and SQLite persistence.

Created: 2026-02-07 · v1.0
"""

import sqlite3
import uuid
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Callable
from enum import Enum

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"


class TaskType(Enum):
    THINK = "think"
    SEARCH = "search"
    CODE = "code"
    WRITE = "write"
    READ = "read"
    TOOL = "tool"
    VERIFY = "verify"
    COMMUNICATE = "communicate"


class TaskExecutor:
    """SQLite-backed planner for ordered ``plan_tasks`` under ``execution_plans``."""
    
    def __init__(self, db_path: str = "data.db"):
        self.db_path = db_path
        self._ensure_tables()
    
    def _ensure_tables(self):
        """Create execution plan tables if missing."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS execution_plans (
                    id TEXT PRIMARY KEY,
                    session_id TEXT DEFAULT 'selfing-session',
                    goal_id TEXT,
                    title TEXT NOT NULL,
                    description TEXT,
                    status TEXT DEFAULT 'pending',
                    total_tasks INTEGER DEFAULT 0,
                    completed_tasks INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    metadata TEXT
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS plan_tasks (
                    id TEXT PRIMARY KEY,
                    plan_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT,
                    task_type TEXT NOT NULL,
                    order_index INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'pending',
                    depends_on TEXT,
                    input_data TEXT,
                    output_data TEXT,
                    error_message TEXT,
                    retry_count INTEGER DEFAULT 0,
                    max_retries INTEGER DEFAULT 3,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    FOREIGN KEY (plan_id) REFERENCES execution_plans(id)
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS task_execution_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    details TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (task_id) REFERENCES plan_tasks(id)
                )
            """)
            
            conn.commit()
            logger.info("TaskExecutor: Tables initialized")
    
    def create_plan(
        self,
        title: str,
        description: str = "",
        goal_id: str = None,
        session_id: str = "selfing-session"
    ) -> Dict[str, Any]:
        """Insert a new ``execution_plans`` row in ``pending`` state."""
        try:
            plan_id = f"PLAN-{datetime.now().strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:4]}"
            created_at = datetime.now(timezone.utc).isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO execution_plans 
                    (id, session_id, goal_id, title, description, status, created_at)
                    VALUES (?, ?, ?, ?, ?, 'pending', ?)
                """, (plan_id, session_id, goal_id, title, description, created_at))
                conn.commit()
            
            logger.info(f"[TASK] Plan created: {plan_id} - {title}")
            
            return {
                "success": True,
                "plan_id": plan_id,
                "title": title,
                "message": "Execution plan created"
            }
            
        except Exception as e:
            logger.error(f"[TASK] Failed to create plan: {e}")
            return {"success": False, "error": str(e)}
    
    def add_task(
        self,
        plan_id: str,
        title: str,
        task_type: str,
        description: str = "",
        depends_on: List[str] = None,
        input_data: Dict = None,
        order_index: int = None
    ) -> Dict[str, Any]:
        """Append a ``plan_tasks`` row and bump ``execution_plans.total_tasks``."""
        try:
            task_id = f"TASK-{uuid.uuid4().hex[:8]}"
            created_at = datetime.now(timezone.utc).isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                if order_index is None:
                    cursor = conn.execute(
                        "SELECT MAX(order_index) FROM plan_tasks WHERE plan_id = ?",
                        (plan_id,)
                    )
                    max_idx = cursor.fetchone()[0]
                    order_index = (max_idx or 0) + 1
                
                conn.execute("""
                    INSERT INTO plan_tasks 
                    (id, plan_id, title, description, task_type, order_index, 
                     status, depends_on, input_data, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?)
                """, (
                    task_id, plan_id, title, description, task_type, order_index,
                    json.dumps(depends_on or [], ensure_ascii=False),
                    json.dumps(input_data or {}, ensure_ascii=False),
                    created_at
                ))
                
                conn.execute("""
                    UPDATE execution_plans 
                    SET total_tasks = total_tasks + 1
                    WHERE id = ?
                """, (plan_id,))
                
                conn.commit()
            
            logger.info(f"[TASK] Task added: {task_id} - {title}")
            
            return {
                "success": True,
                "task_id": task_id,
                "title": title,
                "order": order_index,
                "message": "Task added to plan"
            }
            
        except Exception as e:
            logger.error(f"[TASK] Failed to add task: {e}")
            return {"success": False, "error": str(e)}
    
    def delete_plan_task(self, task_id: str) -> Dict[str, Any]:
        """
        Delete one ``plan_tasks`` row, scrub ``depends_on`` references, and reconcile counters.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute(
                    "SELECT id, plan_id, status FROM plan_tasks WHERE id = ?",
                    (task_id,),
                )
                row = cur.fetchone()
                if not row:
                    return {"success": False, "error": f"Task not found: {task_id}"}
                _, plan_id, task_status = row

                cur = conn.execute(
                    "SELECT id, depends_on FROM plan_tasks WHERE plan_id = ? AND id != ?",
                    (plan_id, task_id),
                )
                for sid, deps_raw in cur.fetchall():
                    try:
                        deps = json.loads(deps_raw or "[]")
                        if not isinstance(deps, list):
                            deps = []
                    except json.JSONDecodeError:
                        deps = []
                    if task_id in deps:
                        deps = [d for d in deps if d != task_id]
                        conn.execute(
                            "UPDATE plan_tasks SET depends_on = ? WHERE id = ?",
                            (json.dumps(deps, ensure_ascii=False), sid),
                        )

                conn.execute(
                    "DELETE FROM task_execution_logs WHERE task_id = ?", (task_id,)
                )
                conn.execute("DELETE FROM plan_tasks WHERE id = ?", (task_id,))

                prow = conn.execute(
                    "SELECT total_tasks, completed_tasks, status, completed_at "
                    "FROM execution_plans WHERE id = ?",
                    (plan_id,),
                ).fetchone()
                if not prow:
                    conn.commit()
                    return {
                        "success": True,
                        "task_id": task_id,
                        "plan_id": plan_id,
                        "message": "Task deleted (plan row missing; counters unchanged)",
                    }

                tot, comp, pstat, pcat = (
                    int(prow[0] or 0),
                    int(prow[1] or 0),
                    prow[2] or "pending",
                    prow[3],
                )
                new_tot = max(0, tot - 1)
                if task_status == TaskStatus.COMPLETED.value:
                    new_comp = max(0, comp - 1)
                else:
                    new_comp = comp
                new_comp = min(new_comp, new_tot) if new_tot else 0

                new_stat = pstat
                new_cat = pcat
                if pstat == "completed" and new_tot > 0 and new_comp < new_tot:
                    new_stat = "in_progress"
                    new_cat = None
                if new_tot == 0:
                    new_stat = "pending"
                    new_comp = 0
                    new_cat = None

                conn.execute(
                    """
                    UPDATE execution_plans
                    SET total_tasks = ?, completed_tasks = ?, status = ?, completed_at = ?
                    WHERE id = ?
                    """,
                    (new_tot, new_comp, new_stat, new_cat, plan_id),
                )

                conn.commit()

            logger.info(f"[TASK] Plan task deleted: {task_id} (plan={plan_id})")
            return {
                "success": True,
                "task_id": task_id,
                "plan_id": plan_id,
                "plan_total_tasks": new_tot,
                "plan_completed_tasks": new_comp,
                "plan_status": new_stat,
                "message": "Task removed from plan",
            }
        except Exception as e:
            logger.error(f"[TASK] Failed to delete plan task: {e}")
            return {"success": False, "error": str(e)}
    
    def decompose_goal(
        self,
        goal_title: str,
        goal_description: str,
        session_id: str = "selfing-session"
    ) -> Dict[str, Any]:
        """Seed a plan plus a suggested ordered task-type checklist (template only)."""
        plan_result = self.create_plan(
            title=f"Execution plan: {goal_title}",
            description=goal_description,
            session_id=session_id
        )
        
        if not plan_result.get("success"):
            return plan_result
        
        plan_id = plan_result["plan_id"]
        
        return {
            "success": True,
            "plan_id": plan_id,
            "goal_title": goal_title,
            "suggested_task_types": [
                {"type": "think", "description": "Analyze and understand the goal"},
                {"type": "search", "description": "Gather supporting information"},
                {"type": "code", "description": "Implement or change code"},
                {"type": "write", "description": "Create or update files"},
                {"type": "verify", "description": "Validate the outcome"},
                {"type": "communicate", "description": "Report back to the user"}
            ],
            "message": "Plan created — add concrete tasks next",
            "next_step": "Call add_task(...) for each concrete step"
        }
    
    def get_plan(self, plan_id: str) -> Dict[str, Any]:
        """Return one plan row plus ordered tasks."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                cursor = conn.execute(
                    "SELECT * FROM execution_plans WHERE id = ?",
                    (plan_id,)
                )
                plan_row = cursor.fetchone()
                
                if not plan_row:
                    return {"success": False, "error": f"Plan not found: {plan_id}"}
                
                plan = dict(plan_row)
                
                cursor = conn.execute("""
                    SELECT * FROM plan_tasks 
                    WHERE plan_id = ?
                    ORDER BY order_index
                """, (plan_id,))
                
                tasks = []
                for row in cursor.fetchall():
                    task = dict(row)
                    task["depends_on"] = json.loads(task["depends_on"]) if task["depends_on"] else []
                    task["input_data"] = json.loads(task["input_data"]) if task["input_data"] else {}
                    task["output_data"] = json.loads(task["output_data"]) if task["output_data"] else {}
                    tasks.append(task)
                
                plan["tasks"] = tasks
                
                if plan["total_tasks"] > 0:
                    plan["progress"] = round(plan["completed_tasks"] / plan["total_tasks"] * 100, 1)
                else:
                    plan["progress"] = 0
                
                return {
                    "success": True,
                    "plan": plan
                }
                
        except Exception as e:
            logger.error(f"[TASK] Failed to get plan: {e}")
            return {"success": False, "error": str(e)}
    
    def get_next_task(self, plan_id: str) -> Dict[str, Any]:
        """Return the first pending task whose dependencies are satisfied."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                cursor = conn.execute("""
                    SELECT * FROM plan_tasks 
                    WHERE plan_id = ?
                    ORDER BY order_index
                """, (plan_id,))
                
                tasks = []
                completed_ids = set()
                
                for row in cursor.fetchall():
                    task = dict(row)
                    task["depends_on"] = json.loads(task["depends_on"]) if task["depends_on"] else []
                    tasks.append(task)
                    
                    if task["status"] == TaskStatus.COMPLETED.value:
                        completed_ids.add(task["id"])
                
                for task in tasks:
                    if task["status"] == TaskStatus.PENDING.value:
                        deps_satisfied = all(dep in completed_ids for dep in task["depends_on"])
                        if deps_satisfied:
                            return {
                                "success": True,
                                "has_next": True,
                                "task": task
                            }
                
                in_progress = [t for t in tasks if t["status"] == TaskStatus.IN_PROGRESS.value]
                if in_progress:
                    task = in_progress[0]
                    # [2026-03-30] Stale guard: default 30 minutes
                    started_at = task.get("started_at")
                    is_stale = False
                    stale_minutes = 0
                    if started_at:
                        try:
                            start_time = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
                            now = datetime.now(timezone.utc)
                            elapsed = (now - start_time).total_seconds()
                            stale_minutes = int(elapsed / 60)
                            is_stale = elapsed > 1800  # 30 minutes
                        except Exception:
                            pass
                    
                    return {
                        "success": True,
                        "has_next": False,
                        "message": "A task is already in progress",
                        "in_progress_task": task,
                        "is_stale": is_stale,
                        "stale_minutes": stale_minutes
                    }
                
                all_done = all(t["status"] in [TaskStatus.COMPLETED.value, TaskStatus.SKIPPED.value] 
                              for t in tasks)
                if all_done:
                    return {
                        "success": True,
                        "has_next": False,
                        "message": "All tasks finished",
                        "plan_completed": True
                    }
                
                blocked = [t for t in tasks if t["status"] == TaskStatus.BLOCKED.value]
                if blocked:
                    return {
                        "success": True,
                        "has_next": False,
                        "message": "Tasks are blocked",
                        "blocked_tasks": blocked
                    }
                
                return {
                    "success": True,
                    "has_next": False,
                    "message": "No runnable task"
                }
                
        except Exception as e:
            logger.error(f"[TASK] Failed to get next task: {e}")
            return {"success": False, "error": str(e)}
    
    def start_task(self, task_id: str) -> Dict[str, Any]:
        """Mark a task ``in_progress`` and append a start log row."""
        try:
            now = datetime.now(timezone.utc).isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE plan_tasks 
                    SET status = ?, started_at = ?
                    WHERE id = ?
                """, (TaskStatus.IN_PROGRESS.value, now, task_id))
                
                conn.execute("""
                    INSERT INTO task_execution_logs (task_id, action, details, created_at)
                    VALUES (?, 'start', 'Task started', ?)
                """, (task_id, now))
                
                conn.commit()
            
            logger.info(f"[TASK] Task started: {task_id}")
            
            return {
                "success": True,
                "task_id": task_id,
                "status": "in_progress",
                "message": "Task started"
            }
            
        except Exception as e:
            logger.error(f"[TASK] Failed to start task: {e}")
            return {"success": False, "error": str(e)}
    
    def complete_task(
        self,
        task_id: str,
        output_data: Dict = None,
        notes: str = ""
    ) -> Dict[str, Any]:
        """Mark a task completed, bump plan counters, maybe close the plan."""
        try:
            now = datetime.now(timezone.utc).isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT plan_id FROM plan_tasks WHERE id = ?",
                    (task_id,)
                )
                row = cursor.fetchone()
                if not row:
                    return {"success": False, "error": f"Task not found: {task_id}"}
                
                plan_id = row[0]
                
                conn.execute("""
                    UPDATE plan_tasks 
                    SET status = ?, completed_at = ?, output_data = ?
                    WHERE id = ?
                """, (
                    TaskStatus.COMPLETED.value, now,
                    json.dumps(output_data or {}, ensure_ascii=False),
                    task_id
                ))
                
                conn.execute("""
                    UPDATE execution_plans 
                    SET completed_tasks = completed_tasks + 1
                    WHERE id = ?
                """, (plan_id,))
                
                cursor = conn.execute("""
                    SELECT total_tasks, completed_tasks 
                    FROM execution_plans 
                    WHERE id = ?
                """, (plan_id,))
                plan_row = cursor.fetchone()
                
                plan_completed = False
                if plan_row and plan_row[0] == plan_row[1]:
                    conn.execute("""
                        UPDATE execution_plans 
                        SET status = 'completed', completed_at = ?
                        WHERE id = ?
                    """, (now, plan_id))
                    plan_completed = True
                
                conn.execute("""
                    INSERT INTO task_execution_logs (task_id, action, details, created_at)
                    VALUES (?, 'complete', ?, ?)
                """, (task_id, notes or "Task completed", now))
                
                conn.commit()
            
            logger.info(f"[TASK] Task completed: {task_id}")
            
            return {
                "success": True,
                "task_id": task_id,
                "status": "completed",
                "plan_completed": plan_completed,
                "message": "Task completed" + (" — entire plan finished." if plan_completed else "")
            }
            
        except Exception as e:
            logger.error(f"[TASK] Failed to complete task: {e}")
            return {"success": False, "error": str(e)}
    
    def fail_task(
        self,
        task_id: str,
        error_message: str,
        retry: bool = True
    ) -> Dict[str, Any]:
        """Record failure, optionally roll back to ``pending`` for retry."""
        try:
            now = datetime.now(timezone.utc).isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT retry_count, max_retries FROM plan_tasks WHERE id = ?",
                    (task_id,)
                )
                row = cursor.fetchone()
                if not row:
                    return {"success": False, "error": f"Task not found: {task_id}"}
                
                retry_count, max_retries = row
                
                if retry and retry_count < max_retries:
                    new_status = TaskStatus.PENDING.value
                    new_retry_count = retry_count + 1
                    message = f"Task failed; retry scheduled ({new_retry_count}/{max_retries})"
                else:
                    new_status = TaskStatus.FAILED.value
                    new_retry_count = retry_count
                    message = "Task failed; max retries exceeded"
                
                conn.execute("""
                    UPDATE plan_tasks 
                    SET status = ?, error_message = ?, retry_count = ?
                    WHERE id = ?
                """, (new_status, error_message, new_retry_count, task_id))
                
                conn.execute("""
                    INSERT INTO task_execution_logs (task_id, action, details, created_at)
                    VALUES (?, 'fail', ?, ?)
                """, (task_id, f"Error: {error_message}", now))
                
                conn.commit()
            
            logger.warning(f"[TASK] Task failed: {task_id} - {error_message}")
            
            return {
                "success": True,
                "task_id": task_id,
                "status": new_status,
                "retry_count": new_retry_count,
                "can_retry": new_status == TaskStatus.PENDING.value,
                "message": message
            }
            
        except Exception as e:
            logger.error(f"[TASK] Failed to mark task as failed: {e}")
            return {"success": False, "error": str(e)}
    
    def get_active_plans(self, session_id: str = "selfing-session") -> List[Dict]:
        """List pending / in-progress plans for a session."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                cursor = conn.execute("""
                    SELECT * FROM execution_plans 
                    WHERE session_id = ? AND status IN ('pending', 'in_progress')
                    ORDER BY created_at DESC
                """, (session_id,))
                
                plans = []
                for row in cursor.fetchall():
                    plan = dict(row)
                    if plan["total_tasks"] > 0:
                        plan["progress"] = round(plan["completed_tasks"] / plan["total_tasks"] * 100, 1)
                    else:
                        plan["progress"] = 0
                    plans.append(plan)
                
                return plans
                
        except Exception as e:
            logger.error(f"[TASK] Failed to get active plans: {e}")
            return []
    
    def cancel_plan(self, plan_id: str) -> Dict[str, Any]:
        """Cancel plan + skip outstanding tasks."""
        try:
            now = datetime.now(timezone.utc).isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE execution_plans 
                    SET status = 'cancelled', completed_at = ?
                    WHERE id = ?
                """, (now, plan_id))
                
                conn.execute("""
                    UPDATE plan_tasks 
                    SET status = 'skipped'
                    WHERE plan_id = ? AND status IN ('pending', 'in_progress')
                """, (plan_id,))
                
                conn.commit()
            
            return {
                "success": True,
                "plan_id": plan_id,
                "message": "Plan cancelled"
            }
            
        except Exception as e:
            logger.error(f"[TASK] Failed to cancel plan: {e}")
            return {"success": False, "error": str(e)}


_task_executor = None


def get_task_executor(db_path: str = "data.db") -> TaskExecutor:
    """Process-wide ``TaskExecutor`` singleton."""
    global _task_executor
    if _task_executor is None:
        _task_executor = TaskExecutor(db_path)
    return _task_executor
