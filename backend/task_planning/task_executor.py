#!/usr/bin/env python3
"""
Task executor: lightweight in-process timer + status transitions.

[2026-01-29] Created.
"""

import time
import logging
from typing import Optional, Dict
from datetime import datetime

from .task_manager import TaskManager, Task, TaskStatus

logger = logging.getLogger(__name__)


class TaskExecutor:
    """Tracks at most one ``IN_PROGRESS`` task for simple agent loops."""

    def __init__(self, task_manager: Optional[TaskManager] = None):
        """
        Args:
            task_manager: Shared ``TaskManager`` instance (created if omitted).
        """
        self.task_manager = task_manager or TaskManager()
        self.current_task: Optional[Task] = None
        self.start_time: Optional[float] = None

        logger.info("TaskExecutor initialized")

    def start_task(self, task_id: int) -> bool:
        if self.current_task:
            logger.warning(f"Task {self.current_task.id} is already in progress")
            return False

        task = self.task_manager.get_task(task_id)
        if not task:
            logger.error(f"Task {task_id} not found")
            return False

        if self.task_manager.start_task(task_id):
            self.current_task = self.task_manager.get_task(task_id)
            self.start_time = time.time()
            logger.info(f"Task started: {task_id} - {task.title}")
            return True

        return False

    def complete_task(self, result: Optional[str] = None) -> bool:
        if not self.current_task:
            logger.warning("No task in progress")
            return False

        if self.start_time:
            elapsed_seconds = time.time() - self.start_time
            actual_minutes = int(elapsed_seconds / 60)
        else:
            actual_minutes = 0

        task_id = self.current_task.id
        success = self.task_manager.complete_task(
            task_id,
            result=result,
            actual_minutes=actual_minutes,
        )

        if success:
            logger.info(f"Task completed: {task_id} ({actual_minutes} minutes)")

            estimated = self.current_task.estimated_minutes
            if estimated > 0:
                accuracy = actual_minutes / estimated
                logger.info(f"  Time estimation accuracy: {accuracy:.2f}x")

                if accuracy > 1.5:
                    logger.warning("  Task took much longer than estimated")
                elif accuracy < 0.5:
                    logger.warning("  Task finished much faster than estimated")

            self.current_task = None
            self.start_time = None
            return True

        return False

    def pause_task(self, reason: Optional[str] = None) -> bool:
        if not self.current_task:
            logger.warning("No task in progress")
            return False

        task = self.current_task
        task.status = TaskStatus.PAUSED.value

        if reason:
            task.notes = f"{task.notes or ''}\nPause reason: {reason}"

        success = self.task_manager.update_task(task)

        if success:
            logger.info(f"Task paused: {task.id}")
            self.current_task = None
            self.start_time = None

        return success

    def fail_task(self, error: str) -> bool:
        if not self.current_task:
            logger.warning("No task in progress")
            return False

        task = self.current_task
        task.status = TaskStatus.FAILED.value
        task.result = f"Failed: {error}"

        success = self.task_manager.update_task(task)

        if success:
            logger.info(f"Task failed: {task.id}")
            self.current_task = None
            self.start_time = None

        return success

    def get_progress(self) -> Dict:
        if not self.current_task:
            return {"in_progress": False, "task": None}

        elapsed_seconds = time.time() - self.start_time if self.start_time else 0
        elapsed_minutes = int(elapsed_seconds / 60)

        estimated = self.current_task.estimated_minutes
        progress_percent = (elapsed_minutes / estimated * 100) if estimated > 0 else 0

        return {
            "in_progress": True,
            "task_id": self.current_task.id,
            "task_title": self.current_task.title,
            "elapsed_minutes": elapsed_minutes,
            "estimated_minutes": estimated,
            "progress_percent": min(100, progress_percent),
            "overdue": elapsed_minutes > estimated,
        }
