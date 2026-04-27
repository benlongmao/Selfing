"""
统一任务调度器 (Unified Task Scheduler)

[2026-03-27] 替代多个互相竞争的后台循环，统一调度所有需要 LLM 执行的任务。

之前的问题：
- _spontaneous_action_loop (存在脉冲)、HeartbeatService、continuous_execution、
  静息脉冲中的定时/日历任务执行，全部各自调用 cs.chat()，多线程并发竞争
- 没有优先级：定时提醒和自由思考同等对待
- 执行结果推送不可靠

新设计：
- 所有需要 LLM 执行的后台任务通过此调度器的优先级队列串行处理
- 静息脉冲仍每 60s 运行，但只做轻量检测 + enqueue，不直接调用 cs.chat()
- 心跳检测到任务后 enqueue，不直接执行
- 队列空闲超过阈值时自动注入"自主思考"任务
"""

import heapq
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from backend.config import config

logger = logging.getLogger(__name__)

PRIORITY_SCHEDULED = 10
PRIORITY_PLAN_TASK = 20
PRIORITY_HEARTBEAT = 30
PRIORITY_IDLE_PULSE = 40

_scheduler_instance: Optional["UnifiedScheduler"] = None


def get_scheduler() -> Optional["UnifiedScheduler"]:
    return _scheduler_instance


def init_scheduler(db_path: str) -> "UnifiedScheduler":
    global _scheduler_instance
    _scheduler_instance = UnifiedScheduler(db_path)
    return _scheduler_instance


def enqueue_autonomy_resume_check(session_id: str = "demo-session") -> bool:
    """
    自主闸门从 paused -> resumed 后，立即补一条高优先级检查任务，
    避免必须等待下一次周期脉冲才开始执行。
    """
    sched = get_scheduler()
    if not sched:
        return False
    task_id = f"autonomy-resume-{session_id}-{int(time.time())}"
    prompt = (
        "[系统] 自主行动已恢复，请立即执行一次待办检查。\n"
        "先用一两句话说明：是否处理本轮待办；再展开。\n"
        "若有 HEARTBEAT 待办，请先 read_file('HEARTBEAT.md') 再推进。"
    )
    return sched.enqueue(
        priority=PRIORITY_HEARTBEAT,
        task_type="autonomy_resume_check",
        task_id=task_id,
        prompt=prompt,
        session_id=session_id,
        is_system_reminder=True,
        temperature=0.5,
    )


@dataclass(order=True)
class WorkItem:
    priority: int
    created_at: float = field(compare=True)
    task_type: str = field(compare=False)
    task_id: str = field(compare=False)
    prompt: str = field(compare=False)
    session_id: str = field(compare=False)
    metadata: Dict[str, Any] = field(default_factory=dict, compare=False)
    is_system_reminder: bool = field(default=True, compare=False)
    temperature: float = field(default=0.5, compare=False)


class UnifiedScheduler:

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._queue: List[WorkItem] = []
        self._lock = threading.Lock()
        self._queued_ids: set = set()
        self._last_execution_time = time.time()
        self._running = False

        self._presence_interval = int(config.get("system.presence_pulse_interval", 1800))
        self._last_presence_inject = time.time()
        self._cooldown = 5
        self._plan_check_interval = int(config.get("system.continuous_execution_interval", 300))
        self._last_plan_check = 0.0

    # -------------------- public API --------------------

    def enqueue(
        self,
        priority: int,
        task_type: str,
        task_id: str,
        prompt: str,
        session_id: str = "demo-session",
        metadata: Optional[Dict] = None,
        is_system_reminder: bool = True,
        temperature: float = 0.5,
    ) -> bool:
        with self._lock:
            if task_id in self._queued_ids:
                logger.debug(f"[SCHEDULER] Skipped duplicate: {task_type}/{task_id}")
                return False
            item = WorkItem(
                priority=priority,
                created_at=time.time(),
                task_type=task_type,
                task_id=task_id,
                prompt=prompt,
                session_id=session_id,
                metadata=metadata or {},
                is_system_reminder=is_system_reminder,
                temperature=temperature,
            )
            heapq.heappush(self._queue, item)
            self._queued_ids.add(task_id)
            logger.info(
                f"[SCHEDULER] Enqueued: P{priority} {task_type} [{task_id}] "
                f"(queue={len(self._queue)})"
            )
            return True

    def is_queued(self, task_id: str) -> bool:
        with self._lock:
            return task_id in self._queued_ids

    @property
    def queue_size(self) -> int:
        with self._lock:
            return len(self._queue)

    def start(self):
        self._running = True
        t = threading.Thread(target=self._executor_loop, daemon=True, name="UnifiedScheduler")
        t.start()
        logger.info("[SCHEDULER] Executor thread started")

    def stop(self):
        self._running = False

    # -------------------- internal --------------------

    def _dequeue(self) -> Optional[WorkItem]:
        with self._lock:
            if not self._queue:
                return None
            item = heapq.heappop(self._queue)
            self._queued_ids.discard(item.task_id)
            return item

    def _executor_loop(self):
        time.sleep(15)
        from backend.routers.chat import get_chat_service
        cs = get_chat_service()
        logger.info("[SCHEDULER] ChatService ready, executor running")

        while self._running:
            try:
                now = time.time()

                # Continuous planning: poll for next plan steps
                if now - self._last_plan_check >= self._plan_check_interval:
                    self._check_plan_tasks()
                    self._last_plan_check = now

                # Presence pulse: periodic low-priority inject (P40)
                if now - self._last_presence_inject >= self._presence_interval:
                    self._inject_idle_pulse()
                    self._last_presence_inject = now
                    # Yield so heartbeat / other threads can enqueue higher priority first
                    time.sleep(1)

                item = self._dequeue()

                if item is None:
                    time.sleep(30)
                    continue

                logger.info(
                    f"[SCHEDULER] >>> Executing: P{item.priority} {item.task_type} "
                    f"[{item.task_id}]"
                )
                self._execute(cs, item)
                self._last_execution_time = now
                time.sleep(self._cooldown)

            except Exception as e:
                logger.error(f"[SCHEDULER] Executor error: {e}", exc_info=True)
                time.sleep(30)

    def _execute(self, cs, item: WorkItem):
        answer = ""
        success = False

        # Autonomy pause: skip LLM for queued background items (calendar/scheduled too) so pause feels real
        try:
            from backend.autonomy_gate import is_autonomous_execution_paused
            _skip_types = {
                "plan_task",
                "stale_task_resume",
                "system_continue",
                "task_continuation",
                "heartbeat",
                "idle_pulse",
                "scheduled_task",
                "calendar_event",
                "self_evolution",
                "self_improvement",
                "enhance_memory",
                "memory_enhancement",
                "agent_evolution",
                "autonomous_evolution",
            }
            if item.task_type in _skip_types and is_autonomous_execution_paused(item.session_id):
                logger.info(
                    "[SCHEDULER] Skipped chat (autonomy pause): %s [%s]",
                    item.task_type,
                    item.task_id,
                )
                self._post_execute(item, False, "")
                return
        except Exception:
            pass

        try:
            resp = cs.chat(
                user_input=item.prompt,
                session_id=item.session_id,
                temperature=item.temperature,
                disable_intent_detection=False,
                is_system_reminder=item.is_system_reminder,
            )
            answer = (resp or {}).get("content") or (resp or {}).get("response") or ""
            success = True
            logger.info(
                f"[SCHEDULER] <<< Done: {item.task_type} [{item.task_id}] "
                f"({len(answer)} chars)"
            )
        except Exception as e:
            logger.warning(
                f"[SCHEDULER] Chat failed for {item.task_type}/{item.task_id}: {e}"
            )

        if answer:
            self._ws_push(answer, item)

        # If system job answer still has [S44_CONTINUE], enqueue one follow-up (separate from task_planning poll)
        if success and answer:
            self._maybe_enqueue_system_continue(item, answer)

        self._post_execute(item, success, answer)

    def _maybe_enqueue_system_continue(self, item: WorkItem, answer: str) -> None:
        """
        [S44_CONTINUE] 只保证「同一次 chat() 内」意图多轮；调度器单次执行结束后若仍带续轮标记，
        原先没有任何机制会在数分钟后自动再叫模型——与 continuous_execution（仅 task_planning 计划）无关。
        此处对系统来源任务补**一跳**续队列，且禁止链式无限续（metadata from_auto_continue）。
        """
        if item.metadata.get("from_auto_continue"):
            return
        try:
            from backend.autonomy_gate import is_autonomous_execution_paused
            if is_autonomous_execution_paused(item.session_id):
                logger.debug("[SCHEDULER] system_continue skipped (autonomy pause)")
                return
        except Exception:
            pass
        try:
            from backend.intent_markers import has_scheduler_continue_marker
        except Exception:
            return
        if not has_scheduler_continue_marker(answer):
            return
        allowed_parent = {
            "heartbeat",
            "idle_pulse",
            "calendar_event",
            "scheduled_task",
            "plan_task",
            "stale_task_resume",
        }
        if item.task_type not in allowed_parent:
            return
        prompt = (
            "[续轮·系统调度] 你上一条回复仍含 [S44_CONTINUE]。请在**本轮**继续完成未尽部分；"
            "若已全部完成请输出 [S44_COMPLETE]，勿再标 CONTINUE。"
        )
        # Fresh metadata: avoid _post_execute double-finishing calendar/scheduled parents
        meta = {
            "from_auto_continue": True,
            "parent_task_type": item.task_type,
        }
        if self.enqueue(
            priority=item.priority,
            task_type="system_continue",
            task_id=f"syscont-{int(time.time() * 1000)}",
            prompt=prompt,
            session_id=item.session_id,
            metadata=meta,
            is_system_reminder=True,
            temperature=item.temperature,
        ):
            logger.info(
                "[SCHEDULER] One-shot system_continue enqueued after [S44_CONTINUE] in %s",
                item.task_type,
            )

    def _ws_push(self, answer: str, item: WorkItem):
        try:
            from backend.websocket_manager import (
                get_websocket_manager, create_ws_message, WSMessageType,
            )
            ws_manager = get_websocket_manager()
            ws_msg = create_ws_message(
                WSMessageType.PRESENCE_PULSE,
                content=answer,
                session_id=item.session_id,
                trigger=item.task_type,
                prompt=item.prompt[:200],
            )
            ws_manager.queue_message(item.session_id, ws_msg)
        except Exception as e:
            logger.warning(f"[SCHEDULER] WS push failed: {e}")

    # -------------------- post-execution handlers --------------------

    def _post_execute(self, item: WorkItem, success: bool, answer: str):
        post_type = item.metadata.get("post_type", "")
        if post_type == "scheduled_task":
            self._post_scheduled_task(item.metadata, success)
        elif post_type == "calendar_event":
            self._post_calendar_event(item.session_id, item.metadata, success, answer)

    def _post_scheduled_task(self, meta: Dict, success: bool):
        try:
            from backend.scheduled_tasks import ScheduledTaskManager
            sm = ScheduledTaskManager(self.db_path)
            task_id = meta.get("task_id", "")
            log_id = meta.get("log_id", "")
            if log_id:
                sm.complete_execution(task_id, log_id, success=success, result="task executed")
            else:
                sm.complete_execution_fallback(task_id, success=success)
            logger.info(f"[SCHEDULER] Scheduled task advanced: {task_id}")
        except Exception as e:
            logger.error(f"[SCHEDULER] Scheduled task post-exec failed: {e}")

    def _post_calendar_event(self, session_id: str, meta: Dict, success: bool, answer: str):
        if not success:
            return
        try:
            from backend.tools.calendar_tool import CalendarTool
            from backend.goal_manager import NotificationQueue

            nq = NotificationQueue(self.db_path)
            nq.add(
                session_id=session_id,
                notification_type="reminder",
                title=meta.get("notif_title", "日历提醒"),
                content=meta.get("notif_content", ""),
                priority=8,
                metadata=meta.get("notif_meta", {}),
            )
            cal = CalendarTool(self.db_path)
            for eid in meta.get("evt_ids", []):
                cal.mark_reminder_sent(eid)
            logger.info(f"[SCHEDULER] Calendar events marked: {len(meta.get('evt_ids', []))}")
        except Exception as e:
            logger.error(f"[SCHEDULER] Calendar post-exec failed: {e}")

    # -------------------- task sources --------------------

    def _inject_idle_pulse(self):
        try:
            from backend.autonomy_gate import is_autonomous_execution_paused
            if is_autonomous_execution_paused("demo-session"):
                logger.debug("[SCHEDULER] idle_pulse skipped (autonomy pause)")
                return
        except Exception:
            pass
        import datetime as dt
        now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
        prompt = (
            f"[系统提醒 {now}] 现在你有权力决定要去做什么，想好了就去做！"
            f"若工作区里有你的进度/约定文件（如 HEARTBEAT.md、自建的笔记），拿不准时先用 read_file 核对再行动。"
        )
        self.enqueue(
            priority=PRIORITY_IDLE_PULSE,
            task_type="idle_pulse",
            task_id=f"idle-{int(time.time())}",
            prompt=prompt,
            session_id="demo-session",
            # True: same as heartbeat/calendar; ensures chat_service still reaches LLM in SOLITARY/RESTING
            is_system_reminder=True,
            temperature=0.7,
        )

    def _check_plan_tasks(self):
        if not bool(config.get("system.continuous_execution_enabled", True)):
            return
        try:
            from backend.autonomy_gate import is_autonomous_execution_paused
            if is_autonomous_execution_paused("demo-session"):
                logger.debug("[SCHEDULER] plan/stale check skipped (autonomy pause)")
                return
        except Exception:
            pass
        try:
            from backend.tools.task_planning_tool import get_task_planning_tool
            tool = get_task_planning_tool(self.db_path)
            if not tool or not tool.task_executor:
                return

            plans = tool.task_executor.get_active_plans("demo-session")
            if not plans:
                return

            # Skip empty shells (no child tasks) until a plan has a runnable step
            chosen = None
            next_result = None
            for plan in plans:
                pid = plan.get("id") or plan.get("plan_id")
                if not pid:
                    continue
                nr = tool.task_executor.get_next_task(pid)
                if (
                    nr.get("success")
                    and nr.get("has_next")
                    and nr.get("task")
                ):
                    chosen = plan
                    next_result = nr
                    break

            # [2026-03-30] Two cases: (1) next task ready (2) stale in_progress
            if chosen and next_result and next_result.get("has_next") and next_result.get("task"):
                # Case 1: enqueue next step
                plan_id = chosen.get("id") or chosen.get("plan_id")
                plan_title = chosen.get("title", "未命名计划")
                task = next_result["task"]
                task_id = task.get("id")
                task_title = task.get("title", "未命名任务")
                task_desc = (task.get("description") or "")[:300]

                prompt = (
                    f"[持续任务执行] 你当前有活跃计划「{plan_title}」。\n"
                    f"下一个任务：{task_title}\n"
                    f"任务描述：{task_desc}\n\n"
                    f"请使用 start_task(task_id=\"{task_id}\") 开始此任务，"
                    f"执行完成后使用 complete_task 标记完成。"
                    f"若无法执行或计划已全部完成，可输出 DONE。"
                )
                self.enqueue(
                    priority=PRIORITY_PLAN_TASK,
                    task_type="plan_task",
                    task_id=f"plan-{task_id}",
                    prompt=prompt,
                    session_id="demo-session",
                )
                return
            
            # Case 2: stale in_progress
            for plan in plans:
                pid = plan.get("id") or plan.get("plan_id")
                if not pid:
                    continue
                nr = tool.task_executor.get_next_task(pid)
                if not nr.get("success"):
                    continue
                
                in_progress_task = nr.get("in_progress_task")
                is_stale = nr.get("is_stale", False)
                stale_minutes = nr.get("stale_minutes", 0)
                
                if in_progress_task and is_stale:
                    # Stale task → nudge prompt
                    plan_title = plan.get("title", "未命名计划")
                    task_id = in_progress_task.get("id")
                    task_title = in_progress_task.get("title", "未命名任务")
                    
                    # [2026-03-30] Direct tool calls, fewer exploratory list_my_tools hops
                    prompt = (
                        f"[任务续跑提醒] 计划「{plan_title}」中的任务「{task_title}」"
                        f"已开始执行 {stale_minutes} 分钟，但尚未标记完成。\n\n"
                        f"**直接操作（无需先加载工具组，task_planning 已自动加载）**：\n"
                        f"- 如果已完成，调用 complete_task(task_id=\"{task_id}\")\n"
                        f"- 如果遇到问题，调用 fail_task(task_id=\"{task_id}\", error_message=\"...\")\n"
                        f"- 如果需要继续执行，继续工作并在完成后标记\n\n"
                        f"注意：请直接调用工具，不要先调用 list_my_tools 或 request_tool_group，工具已可用。"
                    )
                    self.enqueue(
                        priority=PRIORITY_PLAN_TASK,
                        task_type="stale_task_resume",
                        task_id=f"stale-{task_id}-{int(time.time())}",
                        prompt=prompt,
                        session_id="demo-session",
                        # [2026-03-30] Preload task_planning group
                        metadata={"preload_tool_groups": ["task_planning"]},
                    )
                    return
        except Exception as e:
            logger.warning(f"[SCHEDULER] Plan task check skipped: {e}")
