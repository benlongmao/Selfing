#!/usr/bin/env python3
"""
Task reviewer: post-completion retrospectives and daily digest prompts.

[2026-01-29] Created.
"""

import logging
from typing import Optional, Dict, List
from datetime import datetime

from .task_manager import TaskManager, Task, TaskStatus

logger = logging.getLogger(__name__)


class TaskReviewer:
    """LLM-assisted qualitative review on top of SQLite task facts."""

    def __init__(self, task_manager: Optional[TaskManager] = None, llm_client=None):
        """
        Args:
            task_manager: Shared store.
            llm_client: Optional client; otherwise ``SimpleLLMClient``.
        """
        self.task_manager = task_manager or TaskManager()

        if llm_client is None:
            from backend.simple_llm_client import SimpleLLMClient

            self.llm = SimpleLLMClient()
        else:
            self.llm = llm_client

        logger.info("TaskReviewer initialized")

    def review_task(self, task_id: int) -> Dict:
        logger.info(f"Reviewing task: {task_id}")

        task = self.task_manager.get_task(task_id)
        if not task:
            return {"success": False, "error": "Task not found"}

        if task.status != TaskStatus.COMPLETED.value:
            return {"success": False, "error": "Task is not completed"}

        try:
            prompt = self._build_review_prompt(task)

            response = self.llm.call(
                prompt=prompt,
                temperature=0.3,
                max_tokens=1000,
            )

            if not response["success"]:
                return {"success": False, "error": "LLM call failed"}

            review = self._parse_review(response["content"])

            review["task_id"] = task_id
            review["task_title"] = task.title
            review["estimated_minutes"] = task.estimated_minutes
            review["actual_minutes"] = task.actual_minutes
            review["accuracy"] = task.actual_minutes / max(task.estimated_minutes, 1)

            logger.info("Task review completed")
            return {"success": True, "review": review}

        except Exception as e:
            logger.error(f"Task review error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    def _build_review_prompt(self, task: Task) -> str:
        accuracy = task.actual_minutes / max(task.estimated_minutes, 1)
        accuracy_note = ""
        if accuracy > 1.5:
            accuracy_note = "(actual time much higher than estimate)"
        elif accuracy < 0.5:
            accuracy_note = "(actual time much lower than estimate)"

        return f"""You are reviewing a finished task for an autonomous agent diary.

[Task]
Title: {task.title}
Description: {task.description or '(none)'}
Estimated minutes: {task.estimated_minutes}
Actual minutes: {task.actual_minutes} {accuracy_note}
Outcome notes: {task.result or '(none)'}

[Deliverables]
1. One-paragraph summary (1–2 sentences).
2. What went well (1–3 bullets).
3. What should improve next time (1–3 bullets).
4. Estimation analysis (why the delta happened).
5. Concrete follow-ups for the next similar task (1–2 bullets).

Write in clear English suitable for an operator-facing diary entry.
"""

    def _parse_review(self, response_text: str) -> Dict:
        return {"summary": response_text, "timestamp": datetime.now().isoformat()}

    def daily_summary(self, date: Optional[str] = None) -> Dict:
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")

        logger.info(f"Creating daily summary for {date}")

        try:
            completed_tasks = self._get_tasks_by_date(date, TaskStatus.COMPLETED.value)

            if not completed_tasks:
                return {
                    "success": True,
                    "date": date,
                    "message": "No completed tasks on that date",
                }

            total_tasks = len(completed_tasks)
            total_estimated = sum(t.estimated_minutes for t in completed_tasks)
            total_actual = sum(t.actual_minutes for t in completed_tasks)
            avg_accuracy = total_actual / max(total_estimated, 1)

            prompt = self._build_daily_summary_prompt(
                completed_tasks,
                {
                    "total_tasks": total_tasks,
                    "total_estimated": total_estimated,
                    "total_actual": total_actual,
                    "avg_accuracy": avg_accuracy,
                },
            )

            response = self.llm.call(
                prompt=prompt,
                temperature=0.3,
                max_tokens=1500,
            )

            if not response["success"]:
                return {"success": False, "error": "LLM call failed"}

            return {
                "success": True,
                "date": date,
                "stats": {
                    "total_tasks": total_tasks,
                    "total_estimated_minutes": total_estimated,
                    "total_actual_minutes": total_actual,
                    "accuracy": avg_accuracy,
                },
                "summary": response["content"],
            }

        except Exception as e:
            logger.error(f"Daily summary error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    def _get_tasks_by_date(self, date: str, status: str) -> List[Task]:
        try:
            tasks = self.task_manager.list_tasks(status=status, limit=1000)

            filtered = []
            for task in tasks:
                if task.completed_at and task.completed_at.startswith(date):
                    filtered.append(task)

            return filtered

        except Exception as e:
            logger.error(f"Failed to get tasks by date: {e}")
            return []

    def _build_daily_summary_prompt(self, tasks: List[Task], stats: Dict) -> str:
        task_list = []
        for i, task in enumerate(tasks, 1):
            accuracy = task.actual_minutes / max(task.estimated_minutes, 1)
            task_list.append(
                f"{i}. {task.title} ({task.actual_minutes}/{task.estimated_minutes} min, {accuracy:.1f}x)"
            )

        return f"""You are writing an end-of-day retrospective for an autonomous agent.

[Completed tasks]
{chr(10).join(task_list)}

[Stats]
- tasks_completed: {stats['total_tasks']}
- total_estimated_minutes: {stats['total_estimated']}
- total_actual_minutes: {stats['total_actual']}
- avg_actual_vs_estimate: {stats['avg_accuracy']:.2f}x

[Deliverables]
1. Overall assessment of the day.
2. Highlights.
3. Problems or friction.
4. Time-management insights (were estimates trustworthy? why?).
5. Focus suggestions for tomorrow.

Write in clear English suitable for a diary file.
"""
