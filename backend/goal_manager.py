#!/usr/bin/env python3
"""
Goal manager: long-horizon goals, milestones, progress logs, and prompt injection.

Features:
1. Create / update / delete goals scoped to a session
2. Milestone tracking
3. Progress logging and due-date helpers
4. Compact goal summaries for system prompts
"""
import sqlite3
import json
import uuid
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)


class GoalStatus(Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


class MilestoneStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"


class GoalManager:
    """SQLite-backed goals, milestones, and progress for a session."""
    
    def __init__(self, db_path: str = "data.db"):
        self.db_path = db_path
        self._ensure_tables()
        
    def _ensure_tables(self):
        """Create goals / milestones / progress tables when missing."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # goals
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS goals (
                        id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        title TEXT NOT NULL,
                        description TEXT,
                        priority INTEGER DEFAULT 5,
                        status TEXT DEFAULT 'active',
                        deadline TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        completed_at TEXT,
                        metadata TEXT
                    )
                """)
                
                # goal_milestones
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS goal_milestones (
                        id TEXT PRIMARY KEY,
                        goal_id TEXT NOT NULL,
                        title TEXT NOT NULL,
                        description TEXT,
                        order_index INTEGER DEFAULT 0,
                        status TEXT DEFAULT 'pending',
                        due_date TEXT,
                        completed_at TEXT,
                        notes TEXT,
                        FOREIGN KEY (goal_id) REFERENCES goals(id)
                    )
                """)
                
                # goal_progress_logs
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS goal_progress_logs (
                        id TEXT PRIMARY KEY,
                        goal_id TEXT NOT NULL,
                        milestone_id TEXT,
                        action TEXT NOT NULL,
                        notes TEXT,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (goal_id) REFERENCES goals(id)
                    )
                """)
                
                conn.commit()
                logger.info("Goal manager tables ensured")
        except Exception as e:
            logger.error(f"Failed to ensure goal tables: {e}")
    
    # --- Goal CRUD ---
    
    def add_goal(
        self,
        session_id: str,
        title: str,
        description: str = "",
        priority: int = 5,
        deadline: Optional[str] = None,
        milestones: Optional[List[str]] = None,
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Create a goal plus optional milestone titles.

        Args:
            session_id: owning session
            title: short title
            description: free-text description
            priority: 1-10 (10 = highest)
            deadline: optional ISO8601 deadline
            milestones: optional ordered milestone titles
            metadata: arbitrary JSON-able metadata

        Returns:
            ``{"success": True, "goal_id": ..., "message": ...}`` or ``{"success": False, "error": ...}``.
        """
        try:
            goal_id = f"goal-{uuid.uuid4().hex[:8]}"
            now = datetime.now(timezone.utc).isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO goals (id, session_id, title, description, priority, 
                                      status, deadline, created_at, updated_at, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    goal_id, session_id, title, description, priority,
                    GoalStatus.ACTIVE.value, deadline, now, now,
                    json.dumps(metadata or {}, ensure_ascii=False)
                ))
                
                # optional milestone rows
                if milestones:
                    for idx, milestone_title in enumerate(milestones):
                        milestone_id = f"ms-{uuid.uuid4().hex[:8]}"
                        conn.execute("""
                            INSERT INTO goal_milestones (id, goal_id, title, order_index, status)
                            VALUES (?, ?, ?, ?, ?)
                        """, (milestone_id, goal_id, milestone_title, idx, MilestoneStatus.PENDING.value))
                
                conn.commit()
            
            logger.info(f"Goal created: {goal_id} - {title}")
            return {
                "success": True,
                "goal_id": goal_id,
                "message": f"Goal created: {title}",
                "milestones_count": len(milestones) if milestones else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to add goal: {e}")
            return {"success": False, "error": str(e)}
    
    def get_goal(self, goal_id: str) -> Optional[Dict]:
        """Return one goal dict with milestones embedded, or ``None``."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                cur = conn.execute("SELECT * FROM goals WHERE id = ?", (goal_id,))
                row = cur.fetchone()
                if not row:
                    return None
                
                goal = dict(row)
                goal["metadata"] = json.loads(goal["metadata"]) if goal["metadata"] else {}
                
                cur = conn.execute(
                    "SELECT * FROM goal_milestones WHERE goal_id = ? ORDER BY order_index",
                    (goal_id,)
                )
                goal["milestones"] = [dict(r) for r in cur.fetchall()]
                
                return goal
                
        except Exception as e:
            logger.error(f"Failed to get goal: {e}")
            return None
    
    def get_active_goals(self, session_id: str, limit: int = 10) -> List[Dict]:
        """List ``active`` goals only."""
        return self._get_goals_by_status(session_id, [GoalStatus.ACTIVE.value], limit)
    
    def get_all_goals(self, session_id: str, limit: int = 10, include_completed: bool = False) -> List[Dict]:
        """List ``active`` + ``paused`` goals, optionally including ``completed``."""
        statuses = [GoalStatus.ACTIVE.value, GoalStatus.PAUSED.value]
        if include_completed:
            statuses.append(GoalStatus.COMPLETED.value)
        return self._get_goals_by_status(session_id, statuses, limit)
    
    def _get_goals_by_status(self, session_id: str, statuses: List[str], limit: int = 10) -> List[Dict]:
        """Fetch goals for ``session_id`` whose ``status`` is in ``statuses``."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                placeholders = ','.join(['?' for _ in statuses])
                cur = conn.execute(f"""
                    SELECT * FROM goals 
                    WHERE session_id = ? AND status IN ({placeholders})
                    ORDER BY priority DESC, created_at DESC
                    LIMIT ?
                """, (session_id, *statuses, limit))
                
                goals = []
                for row in cur.fetchall():
                    goal = dict(row)
                    goal["metadata"] = json.loads(goal["metadata"]) if goal["metadata"] else {}
                    
                    ms_cur = conn.execute("""
                        SELECT status, COUNT(*) as count 
                        FROM goal_milestones 
                        WHERE goal_id = ?
                        GROUP BY status
                    """, (goal["id"],))
                    
                    ms_stats = {r["status"]: r["count"] for r in ms_cur.fetchall()}
                    goal["milestone_stats"] = ms_stats
                    goal["progress"] = self._calculate_progress(ms_stats)
                    
                    goals.append(goal)
                
                return goals
                
        except Exception as e:
            logger.error(f"Failed to get goals by status: {e}")
            return []
    
    def update_goal_status(self, goal_id: str, status: str, notes: str = "") -> Dict:
        """Update ``goals.status`` and append a progress log row."""
        try:
            now = datetime.now(timezone.utc).isoformat()
            completed_at = now if status == GoalStatus.COMPLETED.value else None
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE goals 
                    SET status = ?, updated_at = ?, completed_at = ?
                    WHERE id = ?
                """, (status, now, completed_at, goal_id))
                
                self._log_progress(conn, goal_id, None, f"status_changed_to_{status}", notes)
                conn.commit()
            
            return {"success": True, "message": f"Goal status updated to: {status}"}
            
        except Exception as e:
            logger.error(f"Failed to update goal status: {e}")
            return {"success": False, "error": str(e)}
    
    def delete_goal(self, goal_id: str, session_id: str) -> Dict[str, Any]:
        """Hard-delete a goal and its milestones/logs when it belongs to ``session_id``."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute(
                    "SELECT id, session_id, title FROM goals WHERE id = ?",
                    (goal_id,),
                )
                row = cur.fetchone()
                if not row:
                    return {"success": False, "error": f"Goal not found: {goal_id}"}
                _gid, sid, title = row
                if sid != session_id:
                    return {
                        "success": False,
                        "error": "Permission denied: goal belongs to a different session",
                    }
                conn.execute(
                    "DELETE FROM goal_progress_logs WHERE goal_id = ?",
                    (goal_id,),
                )
                conn.execute(
                    "DELETE FROM goal_milestones WHERE goal_id = ?",
                    (goal_id,),
                )
                conn.execute("DELETE FROM goals WHERE id = ?", (goal_id,))
                conn.commit()

            logger.info(f"Goal deleted: {goal_id} - {title}")
            return {
                "success": True,
                "goal_id": goal_id,
                "title": title,
                "message": f"Goal permanently deleted: {title}",
            }
        except Exception as e:
            logger.error(f"Failed to delete goal: {e}")
            return {"success": False, "error": str(e)}
    
    # --- Milestones ---
    
    def update_milestone(
        self,
        milestone_id: str,
        status: Optional[str] = None,
        notes: Optional[str] = None
    ) -> Dict:
        """Update milestone ``status`` / ``notes`` and bump parent goal ``updated_at``."""
        try:
            now = datetime.now(timezone.utc).isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute(
                    "SELECT goal_id, title FROM goal_milestones WHERE id = ?",
                    (milestone_id,)
                )
                row = cur.fetchone()
                if not row:
                    return {"success": False, "error": "Milestone not found"}
                
                goal_id, title = row
                
                updates = ["notes = COALESCE(?, notes)"]
                params = [notes]
                
                if status:
                    updates.append("status = ?")
                    params.append(status)
                    if status == MilestoneStatus.COMPLETED.value:
                        updates.append("completed_at = ?")
                        params.append(now)
                
                params.append(milestone_id)
                
                conn.execute(f"""
                    UPDATE goal_milestones 
                    SET {', '.join(updates)}
                    WHERE id = ?
                """, params)
                
                action = f"milestone_updated_{status}" if status else "milestone_notes_added"
                self._log_progress(conn, goal_id, milestone_id, action, notes or "")
                
                conn.execute(
                    "UPDATE goals SET updated_at = ? WHERE id = ?",
                    (now, goal_id)
                )
                
                conn.commit()
            
            return {
                "success": True,
                "message": f"Milestone '{title}' updated",
                "new_status": status
            }
            
        except Exception as e:
            logger.error(f"Failed to update milestone: {e}")
            return {"success": False, "error": str(e)}
    
    def add_milestone(self, goal_id: str, title: str, description: str = "") -> Dict:
        """Append a milestone at the end of ``order_index``."""
        try:
            milestone_id = f"ms-{uuid.uuid4().hex[:8]}"
            
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute(
                    "SELECT MAX(order_index) FROM goal_milestones WHERE goal_id = ?",
                    (goal_id,)
                )
                max_idx = cur.fetchone()[0] or -1
                
                conn.execute("""
                    INSERT INTO goal_milestones (id, goal_id, title, description, order_index, status)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (milestone_id, goal_id, title, description, max_idx + 1, MilestoneStatus.PENDING.value))
                
                self._log_progress(conn, goal_id, milestone_id, "milestone_added", title)
                conn.commit()
            
            return {
                "success": True,
                "milestone_id": milestone_id,
                "message": f"Milestone added: {title}"
            }
            
        except Exception as e:
            logger.error(f"Failed to add milestone: {e}")
            return {"success": False, "error": str(e)}
    
    # --- Progress ---
    
    def log_progress(self, goal_id: str, notes: str, milestone_id: Optional[str] = None) -> Dict:
        """Append a free-form progress note."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                self._log_progress(conn, goal_id, milestone_id, "progress_update", notes)
                conn.commit()
            return {"success": True, "message": "Progress logged"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_progress_history(self, goal_id: str, limit: int = 20) -> List[Dict]:
        """Return recent ``goal_progress_logs`` rows."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.execute("""
                    SELECT * FROM goal_progress_logs
                    WHERE goal_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (goal_id, limit))
                return [dict(r) for r in cur.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get progress history: {e}")
            return []
    
    def _log_progress(self, conn, goal_id: str, milestone_id: Optional[str], 
                      action: str, notes: str):
        """Insert a single progress log row (expects an open connection)."""
        log_id = f"log-{uuid.uuid4().hex[:8]}"
        now = datetime.now(timezone.utc).isoformat()
        conn.execute("""
            INSERT INTO goal_progress_logs (id, goal_id, milestone_id, action, notes, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (log_id, goal_id, milestone_id, action, notes, now))
    
    def _calculate_progress(self, ms_stats: Dict) -> float:
        """Aggregate milestone counts into ``[0,1]`` completion (in-progress counts half)."""
        total = sum(ms_stats.values())
        if total == 0:
            return 0.0
        completed = ms_stats.get(MilestoneStatus.COMPLETED.value, 0)
        in_progress = ms_stats.get(MilestoneStatus.IN_PROGRESS.value, 0)
        return (completed + in_progress * 0.5) / total
    
    # --- Prompt injection ---
    
    def inject_to_prompt(self, session_id: str) -> str:
        """Compact English block summarizing active/paused goals for system prompts."""
        goals = self.get_all_goals(session_id, limit=5, include_completed=False)
        
        if not goals:
            return ""
        
        lines = ["[Current goals]"]
        
        for i, goal in enumerate(goals, 1):
            progress = goal.get("progress", 0)
            progress_bar = self._progress_bar(progress)
            status_indicator = ""
            if goal["status"] == "paused":
                status_indicator = " ⏸️ [paused]"
            
            deadline_str = ""
            if goal.get("deadline"):
                deadline_str = f" | due: {goal['deadline'][:10]}"
            
            lines.append(f"\n{i}. [{goal['title']}] priority {goal['priority']}/10{deadline_str}{status_indicator}")
            lines.append(f"   progress: {progress_bar} {progress*100:.0f}%")
            
            if goal.get("description"):
                lines.append(f"   description: {goal['description'][:100]}...")
            
            ms_stats = goal.get("milestone_stats", {})
            if ms_stats:
                completed = ms_stats.get("completed", 0)
                total = sum(ms_stats.values())
                lines.append(f"   milestones: {completed}/{total} completed")
        
        lines.append("\n(Manage goals with goal_* tools.)")
        
        return "\n".join(lines)
    
    def _progress_bar(self, progress: float, width: int = 10) -> str:
        """ASCII-ish bar using block characters."""
        filled = int(progress * width)
        empty = width - filled
        return f"[{'█' * filled}{'░' * empty}]"
    
    # --- OpenAI-style tool definitions ---
    
    def get_tool_definitions(self) -> List[Dict]:
        """Return ``chat.completions`` function tool specs for goal CRUD."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "goal_add",
                    "description": "Create a long-horizon goal that may span days and multiple steps.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "title": {
                                "type": "string",
                                "description": "Short, explicit title"
                            },
                            "description": {
                                "type": "string",
                                "description": "Detailed description of the goal"
                            },
                            "priority": {
                                "type": "integer",
                                "description": "Priority 1-10 (10 is highest)",
                                "default": 5
                            },
                            "deadline": {
                                "type": "string",
                                "description": "Optional ISO8601 deadline (e.g. 2026-01-15)"
                            },
                            "milestones": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Ordered milestone titles that decompose the goal"
                            }
                        },
                        "required": ["title"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "goal_list",
                    "description": "List goals with progress. Defaults to active + paused rows.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "limit": {
                                "type": "integer",
                                "description": "Maximum rows to return",
                                "default": 10
                            },
                            "include_paused": {
                                "type": "boolean",
                                "description": "Include paused goals (default true)",
                                "default": True
                            },
                            "include_completed": {
                                "type": "boolean",
                                "description": "Include completed goals (default false)",
                                "default": False
                            }
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "goal_update_status",
                    "description": "Update goal lifecycle status (active, completed, paused, cancelled).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "goal_id": {
                                "type": "string",
                                "description": "Goal id (e.g. goal-xxxxxxxx)"
                            },
                            "status": {
                                "type": "string",
                                "enum": ["active", "completed", "paused", "cancelled"],
                                "description": "New status token"
                            },
                            "notes": {
                                "type": "string",
                                "description": "Optional note explaining the transition"
                            }
                        },
                        "required": ["goal_id", "status"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "goal_update_milestone",
                    "description": "Update milestone status or append milestone notes.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "milestone_id": {
                                "type": "string",
                                "description": "Milestone id (ms-xxxxxxxx)"
                            },
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed", "blocked"],
                                "description": "New milestone status"
                            },
                            "notes": {
                                "type": "string",
                                "description": "Optional progress note"
                            }
                        },
                        "required": ["milestone_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "goal_log_progress",
                    "description": "Append a free-form progress update for a goal.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "goal_id": {
                                "type": "string",
                                "description": "Goal id"
                            },
                            "notes": {
                                "type": "string",
                                "description": "Progress narrative"
                            }
                        },
                        "required": ["goal_id", "notes"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "goal_detail",
                    "description": "Fetch a goal with milestones and recent progress history.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "goal_id": {
                                "type": "string",
                                "description": "Goal id"
                            }
                        },
                        "required": ["goal_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "goal_delete",
                    "description": "Permanently delete a goal plus milestones/logs for the current session. "
                    "Irreversible. Prefer goal_update_status(status='cancelled') when history must remain.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "goal_id": {
                                "type": "string",
                                "description": "Goal id to delete (e.g. goal-xxxxxxxx)"
                            }
                        },
                        "required": ["goal_id"]
                    }
                }
            }
        ]
    
    def route_tool_call(self, tool_name: str, args: Dict, session_id: str) -> Dict:
        """Dispatch ``goal_*`` tool names to ``GoalManager`` methods."""
        if tool_name == "goal_add":
            return self.add_goal(
                session_id=session_id,
                title=args.get("title", ""),
                description=args.get("description", ""),
                priority=args.get("priority", 5),
                deadline=args.get("deadline"),
                milestones=args.get("milestones")
            )
        elif tool_name == "goal_list":
            include_paused = args.get("include_paused", True)
            include_completed = args.get("include_completed", False)
            
            if include_paused or include_completed:
                goals = self.get_all_goals(
                    session_id, 
                    limit=args.get("limit", 10),
                    include_completed=include_completed
                )
            else:
                goals = self.get_active_goals(session_id, limit=args.get("limit", 10))
            
            return {
                "success": True,
                "goals": goals,
                "count": len(goals),
                "filters": {
                    "include_paused": include_paused,
                    "include_completed": include_completed
                }
            }
        elif tool_name == "goal_update_status":
            return self.update_goal_status(
                goal_id=args.get("goal_id", ""),
                status=args.get("status", ""),
                notes=args.get("notes", "")
            )
        elif tool_name == "goal_update_milestone":
            return self.update_milestone(
                milestone_id=args.get("milestone_id", ""),
                status=args.get("status"),
                notes=args.get("notes")
            )
        elif tool_name == "goal_log_progress":
            return self.log_progress(
                goal_id=args.get("goal_id", ""),
                notes=args.get("notes", "")
            )
        elif tool_name == "goal_detail":
            goal = self.get_goal(args.get("goal_id", ""))
            if goal:
                goal["progress_history"] = self.get_progress_history(goal["id"])
                return {"success": True, "goal": goal}
            return {"success": False, "error": "Goal not found"}
        elif tool_name == "goal_delete":
            return self.delete_goal(
                goal_id=args.get("goal_id", ""),
                session_id=session_id,
            )
        else:
            return {"success": False, "error": f"Unknown tool: {tool_name}"}
    
    # --- Due dates ---
    
    def check_due_soon(self, session_id: str, hours: int = 24) -> List[Dict]:
        """Return active goals whose deadline falls within the next ``hours`` hours."""
        try:
            now = datetime.now(timezone.utc)
            deadline_threshold = (now + timedelta(hours=hours)).isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                cur = conn.execute("""
                    SELECT * FROM goals 
                    WHERE session_id = ? 
                      AND status = ?
                      AND deadline IS NOT NULL 
                      AND deadline <= ?
                      AND deadline >= ?
                    ORDER BY deadline ASC
                """, (session_id, GoalStatus.ACTIVE.value, deadline_threshold, now.isoformat()))
                
                due_goals = []
                for row in cur.fetchall():
                    goal = dict(row)
                    deadline_dt = datetime.fromisoformat(goal["deadline"].replace('Z', '+00:00'))
                    remaining = deadline_dt - now
                    goal["hours_remaining"] = max(0, remaining.total_seconds() / 3600)
                    goal["is_overdue"] = remaining.total_seconds() < 0
                    due_goals.append(goal)
                
                return due_goals
                
        except Exception as e:
            logger.error(f"Failed to check due goals: {e}")
            return []
    
    def get_overdue_goals(self, session_id: str) -> List[Dict]:
        """Active goals whose deadline is already in the past."""
        try:
            now = datetime.now(timezone.utc).isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                cur = conn.execute("""
                    SELECT * FROM goals 
                    WHERE session_id = ? 
                      AND status = ?
                      AND deadline IS NOT NULL 
                      AND deadline < ?
                    ORDER BY deadline ASC
                """, (session_id, GoalStatus.ACTIVE.value, now))
                
                return [dict(row) for row in cur.fetchall()]
                
        except Exception as e:
            logger.error(f"Failed to get overdue goals: {e}")
            return []


class NotificationQueue:
    """Lightweight SQLite queue for operator/user notifications."""
    
    def __init__(self, db_path: str = "data.db"):
        self.db_path = db_path
        self._ensure_table()
    
    def _ensure_table(self):
        """Create ``notification_queue`` when missing."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS notification_queue (
                        id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        notification_type TEXT NOT NULL,
                        title TEXT NOT NULL,
                        content TEXT,
                        priority INTEGER DEFAULT 5,
                        created_at TEXT NOT NULL,
                        delivered_at TEXT,
                        dismissed_at TEXT,
                        metadata TEXT
                    )
                """)
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to ensure notification table: {e}")
    
    def add(
        self, 
        session_id: str, 
        notification_type: str, 
        title: str, 
        content: str = "",
        priority: int = 5,
        metadata: Optional[Dict] = None
    ) -> str:
        """Enqueue a notification row; returns the generated id or empty string on failure.

        Args:
            session_id: recipient session
            notification_type: e.g. ``goal_due``, ``goal_overdue``, ``reminder``, ``system``
            title: short headline
            content: optional body
            priority: 1-10
            metadata: JSON-able extras
        """
        try:
            notification_id = f"notif-{uuid.uuid4().hex[:8]}"
            now = datetime.now(timezone.utc).isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO notification_queue 
                    (id, session_id, notification_type, title, content, priority, created_at, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    notification_id, session_id, notification_type, 
                    title, content, priority, now,
                    json.dumps(metadata or {}, ensure_ascii=False)
                ))
                conn.commit()
            
            logger.info(f"Notification added: {notification_type} - {title}")
            return notification_id
            
        except Exception as e:
            logger.error(f"Failed to add notification: {e}")
            return ""
    
    def get_pending(self, session_id: str, limit: int = 10) -> List[Dict]:
        """Notifications that are neither delivered nor dismissed."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.execute("""
                    SELECT * FROM notification_queue
                    WHERE session_id = ? 
                      AND delivered_at IS NULL 
                      AND dismissed_at IS NULL
                    ORDER BY priority DESC, created_at ASC
                    LIMIT ?
                """, (session_id, limit))
                
                return [dict(row) for row in cur.fetchall()]
                
        except Exception as e:
            logger.error(f"Failed to get pending notifications: {e}")
            return []
    
    def mark_delivered(self, notification_ids: List[str]):
        """Stamp ``delivered_at`` for the given ids."""
        if not notification_ids:
            return
        try:
            now = datetime.now(timezone.utc).isoformat()
            with sqlite3.connect(self.db_path) as conn:
                placeholders = ','.join(['?' for _ in notification_ids])
                conn.execute(f"""
                    UPDATE notification_queue 
                    SET delivered_at = ?
                    WHERE id IN ({placeholders})
                """, [now] + notification_ids)
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to mark notifications delivered: {e}")
    
    def dismiss(self, notification_id: str):
        """Soft-close a notification via ``dismissed_at``."""
        try:
            now = datetime.now(timezone.utc).isoformat()
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE notification_queue 
                    SET dismissed_at = ?
                    WHERE id = ?
                """, (now, notification_id))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to dismiss notification: {e}")
    
    def format_for_prompt(self, notifications: List[Dict]) -> str:
        """Render pending notifications as a short English prompt block."""
        if not notifications:
            return ""
        
        lines = ["[📬 Pending notifications]"]
        
        type_icons = {
            "goal_due": "⏰",
            "goal_overdue": "🚨",
            "reminder": "📌",
            "system": "ℹ️",
            "scheduled_task": "🔄"
        }
        
        for n in notifications:
            icon = type_icons.get(n["notification_type"], "📬")
            lines.append(f"{icon} {n['title']}")
            if n.get("content"):
                lines.append(f"   {n['content'][:100]}")
        
        return "\n".join(lines)

