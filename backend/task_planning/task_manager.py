#!/usr/bin/env python3
"""
Task manager: CRUD and SQLite persistence for ``tasks``.

[2026-01-29] Created.
"""

import sqlite3
import logging
from typing import List, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Stored ``status`` column values."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class TaskPriority(Enum):
    """Numeric ``priority`` column (higher = more urgent)."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4


@dataclass
class Task:
    """In-memory representation of one row in ``tasks``."""
    id: Optional[int] = None
    title: str = ""
    description: str = ""
    status: str = TaskStatus.PENDING.value
    priority: int = TaskPriority.MEDIUM.value

    created_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    deadline: Optional[str] = None
    estimated_minutes: int = 60
    actual_minutes: int = 0

    depends_on: Optional[str] = None  # JSON list of task ids
    parent_task_id: Optional[int] = None

    tags: Optional[str] = None  # JSON list
    notes: Optional[str] = None
    result: Optional[str] = None

    energy_cost: float = 5.0
    motivation_gain: float = 0.1


class TaskManager:
    """SQLite-backed task store."""

    def __init__(self, db_path: str = None):
        """
        Args:
            db_path: SQLite path; defaults to ``DATA_DB_PATH`` from project config.
        """
        if db_path is None:
            from backend.project_paths import DATA_DB_PATH

            db_path = DATA_DB_PATH

        self.db_path = db_path
        self._init_database()
        logger.info(f"TaskManager initialized with db: {db_path}")

    def _init_database(self):
        """Create ``tasks`` table when missing."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS tasks (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        title TEXT NOT NULL,
                        description TEXT,
                        status TEXT DEFAULT 'pending',
                        priority INTEGER DEFAULT 2,

                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        started_at TIMESTAMP,
                        completed_at TIMESTAMP,
                        deadline TIMESTAMP,
                        estimated_minutes INTEGER DEFAULT 60,
                        actual_minutes INTEGER DEFAULT 0,

                        depends_on TEXT,
                        parent_task_id INTEGER,

                        tags TEXT,
                        notes TEXT,
                        result TEXT,

                        energy_cost REAL DEFAULT 5.0,
                        motivation_gain REAL DEFAULT 0.1,

                        FOREIGN KEY (parent_task_id) REFERENCES tasks(id)
                    )
                """
                )

                conn.commit()
                logger.info("Tasks table initialized")

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}", exc_info=True)

    def create_task(self, task: Task) -> Optional[int]:
        """Insert ``task``; returns new row id or ``None``."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                if not task.created_at:
                    task.created_at = datetime.now().isoformat()

                cursor.execute(
                    """
                    INSERT INTO tasks (
                        title, description, status, priority,
                        created_at, deadline, estimated_minutes,
                        depends_on, parent_task_id,
                        tags, notes,
                        energy_cost, motivation_gain
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        task.title,
                        task.description,
                        task.status,
                        task.priority,
                        task.created_at,
                        task.deadline,
                        task.estimated_minutes,
                        task.depends_on,
                        task.parent_task_id,
                        task.tags,
                        task.notes,
                        task.energy_cost,
                        task.motivation_gain,
                    ),
                )

                task_id = cursor.lastrowid
                conn.commit()

                logger.info(f"Task created: {task_id} - {task.title}")
                return task_id

        except Exception as e:
            logger.error(f"Failed to create task: {e}", exc_info=True)
            return None

    def get_task(self, task_id: int) -> Optional[Task]:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                cursor.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
                row = cursor.fetchone()

                if row:
                    return Task(**dict(row))
                return None

        except Exception as e:
            logger.error(f"Failed to get task {task_id}: {e}")
            return None

    def update_task(self, task: Task) -> bool:
        if not task.id:
            logger.error("Task ID is required for update")
            return False

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute(
                    """
                    UPDATE tasks SET
                        title = ?,
                        description = ?,
                        status = ?,
                        priority = ?,
                        started_at = ?,
                        completed_at = ?,
                        deadline = ?,
                        estimated_minutes = ?,
                        actual_minutes = ?,
                        depends_on = ?,
                        parent_task_id = ?,
                        tags = ?,
                        notes = ?,
                        result = ?,
                        energy_cost = ?,
                        motivation_gain = ?
                    WHERE id = ?
                """,
                    (
                        task.title,
                        task.description,
                        task.status,
                        task.priority,
                        task.started_at,
                        task.completed_at,
                        task.deadline,
                        task.estimated_minutes,
                        task.actual_minutes,
                        task.depends_on,
                        task.parent_task_id,
                        task.tags,
                        task.notes,
                        task.result,
                        task.energy_cost,
                        task.motivation_gain,
                        task.id,
                    ),
                )

                conn.commit()
                logger.info(f"Task updated: {task.id}")
                return True

        except Exception as e:
            logger.error(f"Failed to update task: {e}", exc_info=True)
            return False

    def delete_task(self, task_id: int) -> bool:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
                conn.commit()
                logger.info(f"Task deleted: {task_id}")
                return True
        except Exception as e:
            logger.error(f"Failed to delete task: {e}")
            return False

    def list_tasks(
        self,
        status: Optional[str] = None,
        priority: Optional[int] = None,
        limit: int = 100,
    ) -> List[Task]:
        """Return tasks ordered by priority desc, then created_at desc."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()

                query = "SELECT * FROM tasks WHERE 1=1"
                params = []

                if status:
                    query += " AND status = ?"
                    params.append(status)

                if priority is not None:
                    query += " AND priority = ?"
                    params.append(priority)

                query += " ORDER BY priority DESC, created_at DESC LIMIT ?"
                params.append(limit)

                cursor.execute(query, params)
                rows = cursor.fetchall()

                return [Task(**dict(row)) for row in rows]

        except Exception as e:
            logger.error(f"Failed to list tasks: {e}")
            return []

    def get_pending_tasks(self) -> List[Task]:
        return self.list_tasks(status=TaskStatus.PENDING.value)

    def get_in_progress_tasks(self) -> List[Task]:
        return self.list_tasks(status=TaskStatus.IN_PROGRESS.value)

    def start_task(self, task_id: int) -> bool:
        task = self.get_task(task_id)
        if not task:
            return False

        task.status = TaskStatus.IN_PROGRESS.value
        task.started_at = datetime.now().isoformat()
        return self.update_task(task)

    def complete_task(
        self,
        task_id: int,
        result: Optional[str] = None,
        actual_minutes: Optional[int] = None,
    ) -> bool:
        task = self.get_task(task_id)
        if not task:
            return False

        task.status = TaskStatus.COMPLETED.value
        task.completed_at = datetime.now().isoformat()
        if result:
            task.result = result
        if actual_minutes is not None:
            task.actual_minutes = actual_minutes

        return self.update_task(task)

    def cancel_task(self, task_id: int, reason: Optional[str] = None) -> bool:
        task = self.get_task(task_id)
        if not task:
            return False

        task.status = TaskStatus.CANCELLED.value
        if reason:
            task.notes = f"{task.notes or ''}\nCancellation reason: {reason}"

        return self.update_task(task)
