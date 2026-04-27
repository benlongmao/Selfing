#!/usr/bin/env python3
"""
Continuous task execution (background driver).

[2026-03-12] Drive the agent through an active execution plan at a higher cadence.

Design:
- Independent of Presence Pulse, heartbeat, GlobalWorkspace, etc.
- Runs only when a plan has pending work; otherwise polls at a slower idle interval.
- Each iteration: fetch next task → build an English task prompt → ``chat()`` → tools execute.
"""

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

SESSION_ID = "selfing-session"


def _get_next_task_prompt(session_id: str, db_path: str) -> Optional[str]:
    """
    If an active plan has a next task, return the user-turn prompt that should be sent to ``chat``.

    Otherwise return ``None``.
    """
    try:
        from backend.tools.task_planning_tool import get_task_planning_tool
        tool = get_task_planning_tool(db_path)
        if not tool or not tool.task_executor:
            return None

        plans = tool.task_executor.get_active_plans(session_id)
        if not plans:
            return None

        first_plan = plans[0]
        plan_id = first_plan.get("id") or first_plan.get("plan_id")
        plan_title = first_plan.get("title", "Untitled plan")

        next_result = tool.task_executor.get_next_task(plan_id)
        if not next_result.get("success"):
            return None
        if not next_result.get("has_next"):
            # In progress or plan already finished
            return None
        if not next_result.get("task"):
            return None

        task = next_result["task"]
        task_id = task.get("id")
        task_title = task.get("title", "Untitled task")
        task_desc = (task.get("description") or "")[:300]

        prompt = (
            f"[Continuous execution] You have an active plan: \"{plan_title}\".\n"
            f"Next task: {task_title}\n"
            f"Description: {task_desc}\n\n"
            f"Call start_task(task_id=\"{task_id}\") to begin, then complete_task when finished. "
            f"If you cannot proceed or the plan is finished, reply with DONE."
        )
        logger.info(f"[ContinuousExecution] Driving task: {plan_title} -> {task_title} ({task_id})")
        return prompt

    except Exception as e:
        logger.debug(f"[ContinuousExecution] Next task check skipped: {e}")
        return None


def run_continuous_execution_loop(db_path: str):
    """
    Main polling loop for continuous execution.

    - Shorter sleep when a plan is active
    - Longer sleep when idle to avoid hot-spinning the DB
    """
    from backend.config import config
    enabled = config.get("system.continuous_execution_enabled", True)
    interval_with_plan = config.get("system.continuous_execution_interval", 300)  # seconds
    interval_idle = config.get("system.continuous_execution_idle_interval", 120)  # idle poll

    if not enabled:
        logger.info("[ContinuousExecution] Disabled by config")
        return

    print("=" * 60)
    print("Continuous Task Execution started")
    print("When a plan is active, the agent is prompted to advance the next task.")
    print(f"Active-plan interval: {interval_with_plan}s | Idle poll: {interval_idle}s")
    print("=" * 60)
    logger.info("[ContinuousExecution] Loop started")

    from backend.routers.chat import get_chat_service
    chat_service = get_chat_service()

    while True:
        try:
            prompt = _get_next_task_prompt(SESSION_ID, db_path)
            if prompt:
                logger.info(f"[ContinuousExecution] Sending: {prompt[:100]}...")
                print("[ContinuousExecution] Prompting agent for the next task...")
                try:
                    resp = chat_service.chat(
                        user_input=prompt,
                        session_id=SESSION_ID,
                        temperature=0.5,
                        disable_intent_detection=True  # deterministic execution path
                    )
                    # [2026-03-18] Surface results to the UI (continuous work was invisible otherwise)
                    if resp:
                        answer = (resp.get("content") or resp.get("response") or "").strip()
                        if answer:
                            try:
                                from backend.websocket_manager import (
                                    get_websocket_manager,
                                    create_ws_message,
                                    WSMessageType,
                                )
                                ws_manager = get_websocket_manager()
                                ws_msg = create_ws_message(
                                    WSMessageType.PRESENCE_PULSE,
                                    content=f"[Continuous execution]\n{answer}",
                                    session_id=SESSION_ID,
                                    trigger="continuous_execution",
                                    prompt=prompt[:200] + ("..." if len(prompt) > 200 else ""),
                                )
                                ws_manager.queue_message(SESSION_ID, ws_msg)
                                logger.info("[ContinuousExecution] Result pushed to user via WebSocket")
                            except Exception as ws_err:
                                logger.debug(f"[ContinuousExecution] WebSocket push skipped: {ws_err}")
                except Exception as chat_err:
                    logger.warning(f"[ContinuousExecution] Chat error: {chat_err}")
                sleep_seconds = interval_with_plan
            else:
                sleep_seconds = interval_idle

        except Exception as e:
            logger.warning(f"[ContinuousExecution] Loop error: {e}")
            sleep_seconds = interval_idle

        time.sleep(sleep_seconds)
