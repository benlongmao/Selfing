#!/usr/bin/env python3
"""
Task planner: LLM-backed decomposition, time estimates, and daily scheduling hints.

[2026-01-29] Created.
"""

import json
import logging
from typing import List, Dict, Optional
from datetime import datetime

from .task_manager import Task, TaskManager, TaskStatus, TaskPriority

logger = logging.getLogger(__name__)


class TaskPlanner:
    """Uses the configured ``SimpleLLMClient`` (or injected client) for planning prompts."""

    def __init__(self, task_manager: Optional[TaskManager] = None, llm_client=None):
        """
        Args:
            task_manager: Shared store.
            llm_client: Optional LLM client; otherwise ``SimpleLLMClient``.
        """
        self.task_manager = task_manager or TaskManager()

        if llm_client is None:
            from backend.simple_llm_client import SimpleLLMClient

            self.llm = SimpleLLMClient()
        else:
            self.llm = llm_client

        logger.info("TaskPlanner initialized")

    def decompose_task(self, task_description: str) -> List[Task]:
        """Return child ``Task`` rows from a natural-language brief."""
        logger.info(f"Decomposing task: {task_description[:50]}...")

        try:
            prompt = self._build_decompose_prompt(task_description)

            response = self.llm.call(
                prompt=prompt,
                temperature=0.3,
                max_tokens=2000,
            )

            if not response["success"]:
                logger.error("LLM call failed during decomposition")
                return self._create_fallback_tasks(task_description)

            subtasks = self._parse_subtasks(response["content"])

            logger.info(f"Decomposed into {len(subtasks)} subtasks")
            return subtasks

        except Exception as e:
            logger.error(f"Task decomposition error: {e}", exc_info=True)
            return self._create_fallback_tasks(task_description)

    def _build_decompose_prompt(self, task_description: str) -> str:
        return f"""You are a senior project manager who breaks work into executable chunks.

[Goal]
{task_description}

[Instructions]
Split the goal into ordered subtasks that are:
- explicit and testable
- completable within ~60 minutes each
- each with a crisp definition of done

Also:
- estimate ``estimated_minutes`` per subtask
- list zero-based ``depends_on_index`` references to prior subtasks when ordering matters

Return **JSON only** (no markdown fences), exactly this shape:
[
    {{
        "title": "Subtask one",
        "description": "What to do",
        "estimated_minutes": 30,
        "priority": 2,
        "depends_on_index": []
    }},
    {{
        "title": "Subtask two",
        "description": "What to do",
        "estimated_minutes": 45,
        "priority": 2,
        "depends_on_index": [0]
    }}
]

``depends_on_index`` entries are 0-based indices into this same array.
"""

    def _parse_subtasks(self, response_text: str) -> List[Task]:
        import re

        try:
            json_match = re.search(r"\[.*\]", response_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
            else:
                logger.warning("No JSON array found in model response")
                return []

            subtasks = []
            for item in data:
                task = Task(
                    title=item.get("title", "Subtask"),
                    description=item.get("description", ""),
                    estimated_minutes=item.get("estimated_minutes", 60),
                    priority=item.get("priority", TaskPriority.MEDIUM.value),
                    status=TaskStatus.PENDING.value,
                )
                subtasks.append(task)

            return subtasks

        except Exception as e:
            logger.error(f"Failed to parse subtasks: {e}")
            return []

    def _create_fallback_tasks(self, task_description: str) -> List[Task]:
        return [
            Task(
                title=task_description,
                description="Automatic decomposition failed; split manually.",
                estimated_minutes=60,
                priority=TaskPriority.MEDIUM.value,
            )
        ]

    def estimate_time(self, task: Task, context: Optional[Dict] = None) -> int:
        """Return revised minutes using history + optional runtime context."""
        logger.info(f"Estimating time for task: {task.title}")

        try:
            similar_tasks = self._find_similar_tasks(task.title)
            prompt = self._build_estimate_prompt(task, similar_tasks, context or {})

            response = self.llm.call(
                prompt=prompt,
                temperature=0.2,
                max_tokens=200,
            )

            if not response["success"]:
                logger.error("LLM call failed during time estimate")
                return task.estimated_minutes or 60

            estimated_minutes = self._parse_time_estimate(response["content"])

            logger.info(f"Estimated time: {estimated_minutes} minutes")
            return estimated_minutes

        except Exception as e:
            logger.error(f"Time estimation error: {e}")
            return task.estimated_minutes or 60

    def _find_similar_tasks(self, title: str, limit: int = 5) -> List[Dict]:
        try:
            completed_tasks = self.task_manager.list_tasks(
                status=TaskStatus.COMPLETED.value,
                limit=50,
            )

            similar = []
            keywords = set(title.lower().split())

            for task in completed_tasks:
                task_keywords = set(task.title.lower().split())
                overlap = len(keywords & task_keywords)

                if overlap > 0:
                    similar.append(
                        {
                            "title": task.title,
                            "estimated_minutes": task.estimated_minutes,
                            "actual_minutes": task.actual_minutes,
                            "accuracy": task.actual_minutes
                            / max(task.estimated_minutes, 1),
                        }
                    )

            return similar[:limit]

        except Exception as e:
            logger.error(f"Failed to find similar tasks: {e}")
            return []

    def _build_estimate_prompt(
        self,
        task: Task,
        similar_tasks: List[Dict],
        context: Dict,
    ) -> str:
        similar_section = ""
        if similar_tasks:
            similar_section = f"""
[Historical reference]
{json.dumps(similar_tasks, ensure_ascii=False, indent=2)}
"""

        context_section = f"""
[Runtime context]
- energy: {context.get('energy', 100)}/100
- completed_tasks_today: {context.get('completed_tasks_today', 0)}
- local_time: {context.get('current_time', datetime.now().strftime('%H:%M'))}
"""

        return f"""You estimate realistic wall-clock effort for a single task.

[Task]
Title: {task.title}
Description: {task.description or '(none)'}
Initial estimate: {task.estimated_minutes} minutes
{similar_section}
{context_section}

[Instructions]
Return **only one integer**: predicted actual minutes.
Consider interruptions (+10–20%), fatigue when energy is low (+20–30%), complexity,
and historical accuracy ratios.

Example output: 45
"""

    def _parse_time_estimate(self, response_text: str) -> int:
        import re

        numbers = re.findall(r"\d+", response_text)
        if numbers:
            estimated = int(numbers[0])
            return max(1, min(480, estimated))

        return 60

    def create_daily_plan(
        self,
        goals: Optional[List[str]] = None,
        available_hours: int = 8,
    ) -> Dict:
        """Produce a JSON plan object referencing pending task indices."""
        logger.info(f"Creating daily plan ({available_hours}h available)")

        try:
            pending_tasks = self.task_manager.get_pending_tasks()

            if not pending_tasks and not goals:
                return {
                    "success": True,
                    "message": "No pending tasks",
                    "tasks": [],
                    "schedule": [],
                }

            prompt = self._build_daily_plan_prompt(
                goals or [],
                pending_tasks,
                available_hours,
            )

            response = self.llm.call(
                prompt=prompt,
                temperature=0.3,
                max_tokens=2000,
            )

            if not response["success"]:
                logger.error("LLM call failed while building daily plan")
                return {"success": False, "error": "LLM call failed"}

            plan = self._parse_daily_plan(response["content"], pending_tasks)

            logger.info(f"Daily plan created with {len(plan.get('tasks', []))} tasks")
            return plan

        except Exception as e:
            logger.error(f"Daily plan creation error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    def _build_daily_plan_prompt(
        self,
        goals: List[str],
        pending_tasks: List[Task],
        available_hours: int,
    ) -> str:
        goals_section = ""
        if goals:
            goals_section = f"""
[Long-term goals]
{chr(10).join(f'{i+1}. {g}' for i, g in enumerate(goals))}
"""

        tasks_section = ""
        if pending_tasks:
            task_list = []
            for i, task in enumerate(pending_tasks[:20]):
                task_list.append(
                    f"{i+1}. {task.title} ({task.estimated_minutes} min, priority {task.priority})"
                )
            tasks_section = f"""
[Pending tasks]
{chr(10).join(task_list)}
"""

        return f"""You build a feasible schedule for today.
{goals_section}
{tasks_section}

[Available focus time]
{available_hours} hours

[Instructions]
1. Prefer highest priority work first.
2. Total planned minutes must fit inside available time (include ~20% buffer).
3. Respect dependencies between pending items.
4. Include short breaks where helpful.

Return **JSON only** (no markdown fences):
{{
    "selected_task_indices": [0, 2, 5],
    "schedule": [
        {{"time": "09:00-09:45", "task_index": 0, "task_title": "Example task"}},
        {{"time": "09:45-10:00", "task_index": null, "task_title": "Break"}},
        {{"time": "10:00-11:30", "task_index": 2, "task_title": "Example task"}}
    ],
    "total_minutes": 240,
    "buffer_minutes": 60
}}

``task_index`` refers to the 0-based index in the pending task list shown above.
"""

    def _parse_daily_plan(self, response_text: str, pending_tasks: List[Task]) -> Dict:
        import re

        try:
            json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
            else:
                logger.warning("No JSON object found in model response")
                return {"success": False, "error": "Failed to parse model response"}

            selected_indices = data.get("selected_task_indices", [])
            selected_tasks = []
            for idx in selected_indices:
                if 0 <= idx < len(pending_tasks):
                    selected_tasks.append(pending_tasks[idx])

            return {
                "success": True,
                "tasks": [{"id": t.id, "title": t.title} for t in selected_tasks],
                "schedule": data.get("schedule", []),
                "total_minutes": data.get("total_minutes", 0),
                "buffer_minutes": data.get("buffer_minutes", 0),
            }

        except Exception as e:
            logger.error(f"Failed to parse daily plan: {e}")
            return {"success": False, "error": "Failed to parse model response"}
