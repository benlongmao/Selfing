"""
Agent task planning tool (Task Planning Tool)

Lets the agent create and run multi-step plans: decompose work, track progress, and handle failures.

Created: 2026-02-07, v1.0
"""

import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

try:
    from backend.task_executor import get_task_executor, TaskExecutor
    TASK_EXECUTOR_AVAILABLE = True
except ImportError:
    TASK_EXECUTOR_AVAILABLE = False

# [2026-03-30] Unified memory bus for task–memory links
try:
    from backend.unified_memory import UnifiedMemoryBus
    UNIFIED_MEMORY_AVAILABLE = True
except ImportError:
    UNIFIED_MEMORY_AVAILABLE = False


class TaskPlanningTool:
    """
    Task planning for the agent.

    Capabilities:
    1. Create execution plans for complex goals
    2. Add tasks as concrete steps
    3. Start and complete tasks in order
    4. Inspect progress
    5. Record failures and retries
    """

    def __init__(self, db_path: str = "data.db"):
        self.db_path = db_path
        self.task_executor = None

        if TASK_EXECUTOR_AVAILABLE:
            self.task_executor = get_task_executor(db_path)
            logger.info("TaskPlanningTool: Task executor connected")

    def get_tool_definitions(self) -> List[Dict]:
        """Tool definitions for tool_router registration."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "create_execution_plan",
                    "description": "Create an execution plan for a complex goal. Use when work has multiple steps.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "title": {
                                "type": "string",
                                "description": "Plan title"
                            },
                            "description": {
                                "type": "string",
                                "description": "What the plan should achieve"
                            },
                            "goal_id": {
                                "type": "string",
                                "description": "Optional linked goal id"
                            }
                        },
                        "required": ["title"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "add_task_to_plan",
                    "description": "Add a task to a plan. Each task is one step.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "plan_id": {
                                "type": "string",
                                "description": "Plan id"
                            },
                            "title": {
                                "type": "string",
                                "description": "Task title"
                            },
                            "task_type": {
                                "type": "string",
                                "enum": ["think", "search", "code", "write", "read", "tool", "verify", "communicate"],
                                "description": "Task type"
                            },
                            "description": {
                                "type": "string",
                                "description": "Task description"
                            },
                            "depends_on": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Task ids this task depends on"
                            }
                        },
                        "required": ["plan_id", "title", "task_type"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "delete_plan_task",
                    "description": "Remove one subtask (plan_tasks) from a plan. Updates counts and strips this id from other tasks' depends_on. Use when a step is redundant or impossible.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task_id": {
                                "type": "string",
                                "description": "Subtask id (e.g. TASK-xxxx) from get_plan_details / get_next_task"
                            }
                        },
                        "required": ["task_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_plan_details",
                    "description": "Return plan metadata and the full task list.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "plan_id": {
                                "type": "string",
                                "description": "Plan id"
                            }
                        },
                        "required": ["plan_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_next_task",
                    "description": "Get the next runnable task for a plan.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "plan_id": {
                                "type": "string",
                                "description": "Plan id"
                            }
                        },
                        "required": ["plan_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "start_task",
                    "description": "Mark a task as started.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task_id": {
                                "type": "string",
                                "description": "Task id"
                            }
                        },
                        "required": ["task_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "complete_task",
                    "description": "Mark a task as completed.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task_id": {
                                "type": "string",
                                "description": "Task id"
                            },
                            "result": {
                                "type": "string",
                                "description": "Result summary"
                            },
                            "output_data": {
                                "type": "object",
                                "description": "Structured output"
                            }
                        },
                        "required": ["task_id"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "fail_task",
                    "description": "Mark a task failed; optionally request retry.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task_id": {
                                "type": "string",
                                "description": "Task id"
                            },
                            "error_message": {
                                "type": "string",
                                "description": "Error text"
                            },
                            "retry": {
                                "type": "boolean",
                                "description": "Whether to retry"
                            }
                        },
                        "required": ["task_id", "error_message"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_my_active_plans",
                    "description": "List this session's active execution plans.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "cancel_plan",
                    "description": "Cancel a plan.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "plan_id": {
                                "type": "string",
                                "description": "Plan id"
                            }
                        },
                        "required": ["plan_id"]
                    }
                }
            }
        ]

    def _record_task_memory(
        self,
        session_id: str,
        event_type: str,
        title: str,
        details: str = "",
    ) -> None:
        """[2026-03-30] Persist a task event to unified memory (bilingual tags for search)."""
        if not UNIFIED_MEMORY_AVAILABLE:
            return
        try:
            bus = UnifiedMemoryBus(self.db_path)
            user_input = f"[Task system] {event_type}: {title}"
            assistant_response = details or f"{event_type} (recorded): {title}"
            bus.record_interaction_event(
                session_id=session_id,
                user_input=user_input,
                assistant_response=assistant_response,
                introspection=f"Task event: {event_type}",
            )
            logger.debug(f"[TaskPlanningTool] Memory recorded: {event_type} - {title}")
        except Exception as e:
            logger.warning(f"[TaskPlanningTool] Failed to record memory: {e}")

    def _trigger_continuation_prompt(
        self,
        session_id: str,
        task_id: str,
        plan_completed: bool,
    ) -> None:
        """[2026-03-30] After a task, nudge the agent to continue or rest (bilingual)."""
        try:
            from backend.autonomy_gate import is_autonomous_execution_paused
            if is_autonomous_execution_paused(session_id):
                logger.debug("[TaskPlanning] continuation prompt skipped (autonomy pause)")
                return
        except Exception:
            pass
        try:
            from backend.unified_scheduler import get_scheduler, PRIORITY_PLAN_TASK
            import time

            scheduler = get_scheduler()
            if not scheduler:
                return

            if plan_completed:
                prompt = (
                    "[Task system] You finished the entire execution plan.\n"
                    "Next: (1) get_my_active_plans, (2) review what you did, (3) decide what to do next.\n"
                )
            else:
                prompt = (
                    f"[Task system] Task {task_id} is complete.\n"
                    "Next: (1) get_next_task, or (2) pause until the next tick.\n"
                )

            scheduler.enqueue(
                priority=PRIORITY_PLAN_TASK + 5,
                task_type="task_continuation",
                task_id=f"cont-{task_id}-{int(time.time())}",
                prompt=prompt,
                session_id=session_id,
                is_system_reminder=False,
                temperature=0.6,
            )
            logger.info(f"[TaskPlanningTool] Continuation prompt enqueued after task {task_id}")
        except Exception as e:
            logger.debug(f"[TaskPlanningTool] Failed to trigger continuation: {e}")

    def route_tool_call(self, func_name: str, args: Dict, session_id: str = "selfing-session") -> Dict:
        """Dispatch tool calls to the task executor."""
        if not self.task_executor:
            return {"success": False, "error": "Task executor not available"}

        if func_name == "create_execution_plan":
            result = self.task_executor.create_plan(
                title=args.get("title"),
                description=args.get("description", ""),
                goal_id=args.get("goal_id"),
                session_id=session_id
            )
            if result.get("success"):
                self._record_task_memory(
                    session_id=session_id,
                    event_type="plan_created",
                    title=args.get("title", ""),
                    details=f"plan_id: {result.get('plan_id')}, description: {args.get('description', '')}",
                )
            return result
        if func_name == "add_task_to_plan":
            return self.task_executor.add_task(
                plan_id=args.get("plan_id"),
                title=args.get("title"),
                task_type=args.get("task_type"),
                description=args.get("description", ""),
                depends_on=args.get("depends_on"),
                input_data=args.get("input_data")
            )
        if func_name == "delete_plan_task":
            return self.task_executor.delete_plan_task(args.get("task_id"))
        if func_name == "get_plan_details":
            return self.task_executor.get_plan(args.get("plan_id"))
        if func_name == "get_next_task":
            return self.task_executor.get_next_task(args.get("plan_id"))
        if func_name == "start_task":
            return self.task_executor.start_task(args.get("task_id"))
        if func_name == "complete_task":
            result = self.task_executor.complete_task(
                task_id=args.get("task_id"),
                output_data=args.get("output_data"),
                notes=args.get("result", "")
            )
            if result.get("success"):
                plan_note = "(all tasks done)" if result.get("plan_completed") else ""
                self._record_task_memory(
                    session_id=session_id,
                    event_type="task_completed",
                    title=args.get("task_id", ""),
                    details=f"{args.get('result', '')} {plan_note}".strip(),
                )
                self._trigger_continuation_prompt(session_id, args.get("task_id", ""), result.get("plan_completed", False))
            return result
        if func_name == "fail_task":
            result = self.task_executor.fail_task(
                task_id=args.get("task_id"),
                error_message=args.get("error_message"),
                retry=args.get("retry", True)
            )
            if result.get("success"):
                self._record_task_memory(
                    session_id=session_id,
                    event_type="task_failed",
                    title=args.get("task_id", ""),
                    details=f"error: {args.get('error_message', '')}, can_retry: {result.get('can_retry')}",
                )
            return result
        if func_name == "get_my_active_plans":
            plans = self.task_executor.get_active_plans(session_id)
            return {
                "success": True,
                "plans": plans,
                "count": len(plans)
            }
        if func_name == "cancel_plan":
            return self.task_executor.cancel_plan(args.get("plan_id"))
        return {"error": f"Unknown function: {func_name}"}


_task_planning_tool = None


def get_task_planning_tool(db_path: str = "data.db") -> TaskPlanningTool:
    """Singleton accessor."""
    global _task_planning_tool
    if _task_planning_tool is None:
        _task_planning_tool = TaskPlanningTool(db_path)
    return _task_planning_tool
