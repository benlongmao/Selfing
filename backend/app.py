import os

# Apply network prefs before the first requests/urllib3 connection (optional DEEPSEEK_FORCE_IPV4=1 on WSL).
try:
    from backend.net_prefs import apply_net_prefs

    apply_net_prefs()
except Exception:
    pass

import sqlite3
import threading
import time
import logging
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import Response

from backend.database import DB_PATH
from backend.websocket_manager import (
    get_websocket_manager, 
    create_ws_message, 
    WSMessageType
)
from backend.routers import system, persona, self, dimension, chat, meta, world, backup
from backend.promotion import PromotionGate
from backend.config import config
from backend.config_validator import validate_config_on_startup

logger = logging.getLogger(__name__)

# Calendar maintenance GC: throttled by parameters.calendar.gc_interval_seconds (one global pass).
_calendar_gc_last_ts = 0.0

# [Phase 3.1] Validate config at import/startup (non-fatal).
if not validate_config_on_startup(config):
    logger.error("Configuration validation failed! Please check settings.yaml")

app = FastAPI(
    title="Self-becoming",
    description="Experimental runtime for a single long-lived agent instance (S): persona, state, memory, and rhythm APIs.",
)

# CORS (P0.4: from config)
cors_origins = config.get("system.cors_origins", ["*"]) or ["*"]
allow_credentials = True
if cors_origins == ["*"]:
    # Browsers disallow allow_origins="*" together with allow_credentials=True
    allow_credentials = False
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static Files
if os.path.isdir("reports"):
    app.mount("/reports", StaticFiles(directory="reports", html=True), name="reports")

if os.path.isdir("frontend"):
    app.mount("/ui", StaticFiles(directory="frontend", html=True), name="ui")
    # [v2.1] Also serve /frontend for legacy bookmarks
    app.mount("/frontend", StaticFiles(directory="frontend", html=True), name="frontend")

@app.api_route("/favicon.ico", methods=["GET", "HEAD"], include_in_schema=False)
def favicon():
    return Response(status_code=204)

# Include Routers
app.include_router(system.router)
app.include_router(persona.router)
app.include_router(self.router)
app.include_router(dimension.router)
app.include_router(chat.router)
app.include_router(meta.router)
app.include_router(world.router)
app.include_router(backup.router)

logger = logging.getLogger(__name__)


# ----- WebSocket -----
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time pushes to the browser.

    Message kinds (payload `type`):
    - new_message: agent-initiated chat
    - state_update: energy / mood / state
    - notification: queued notifications
    - heartbeat: keepalive / ack
    """
    manager = get_websocket_manager()
    await manager.connect(websocket, session_id)
    
    try:
        while True:
            data = await websocket.receive_text()
            
            if data == "ping":
                await websocket.send_text("pong")
            elif data.startswith("{"):
                import json
                try:
                    msg = json.loads(data)
                    if msg.get("type") == "subscribe":
                        logger.debug(f"[WS] Client subscribed: {session_id}")
                except json.JSONDecodeError:
                    pass
                    
    except WebSocketDisconnect:
        manager.disconnect(websocket, session_id)
        logger.info(f"[WS] Client disconnected: {session_id}")
    except Exception as e:
        logger.warning(f"[WS] Connection error: {e}")
        manager.disconnect(websocket, session_id)


# P0.2: centralized idempotent schema migration (avoid ad-hoc ALTER at runtime)
def _ensure_column(conn: sqlite3.Connection, table: str, column: str, ddl_fragment: str):
    cols = []
    try:
        cur = conn.execute(f"PRAGMA table_info({table})")
        cols = [r[1] for r in cur.fetchall()]
    except Exception:
        cols = []
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl_fragment}")


def run_schema_migrations(db_path: str):
    """
    Idempotent schema bootstrap/migration.

    On a fresh repo the DB file may be empty until first connect; this creates core tables
    ``self_state`` / ``chat_turns`` when missing and adds columns as needed.
    """
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS self_state (
                    session_id TEXT PRIMARY KEY,
                    z_self TEXT,
                    confidence REAL,
                    tick INTEGER DEFAULT 0,
                    drift REAL DEFAULT 0,
                    updated_at TEXT,
                    last_summary TEXT,
                    self_summary TEXT,
                    needs TEXT,
                    energy REAL,
                    will_tension REAL,
                    limits TEXT,
                    calibration_ece REAL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_turns (
                    id TEXT PRIMARY KEY,
                    session_id TEXT,
                    turn_index INTEGER,
                    user_input TEXT,
                    assistant_output TEXT,
                    introspection TEXT,
                    drift REAL,
                    tick_count INTEGER,
                    self_tick_triggered INTEGER,
                    reflection TEXT,
                    latency REAL,
                    tool_used TEXT,
                    created_at TEXT,
                    metabolized INTEGER DEFAULT 0
                )
                """
            )
            # Ensure self_state has last_summary (used by SelfModel._save_z_self)
            _ensure_column(conn, "self_state", "last_summary", "TEXT DEFAULT ''")
            # Extended z_self metadata (JSON): anchors, confidence spread, alignment — additive only
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS self_state_meta (
                  session_id TEXT PRIMARY KEY,
                  meta_json TEXT NOT NULL,
                  updated_at TEXT NOT NULL
                )
                """
            )
            # episodic memory metabolization flag (rule_compressor)
            _ensure_column(conn, "chat_turns", "metabolized", "INTEGER DEFAULT 0")
            # Spaced-repetition memory table (reserved for future SQLite-backed SR; JSON engine is primary)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS spaced_repetition_memories (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    memory_content TEXT NOT NULL,
                    importance REAL DEFAULT 0.5,
                    due_date TEXT NOT NULL,
                    interval_days REAL DEFAULT 1.0,
                    ease_factor REAL DEFAULT 2.5,
                    repetition_count INTEGER DEFAULT 0,
                    last_reviewed TEXT,
                    created_at TEXT NOT NULL,
                    metadata TEXT
                )
                """
            )
            conn.commit()
    except Exception as e:
        logger.debug(f"Schema migration skipped/failed: {e}")


def create_daily_memory_review_task(scheduled_tasks, session_id: str) -> None:
    """
    Register a daily memory / spaced-repetition review task (idempotent).

    Uses English task name/description for operators and DB inspection.
    """
    task_name = "Daily memory review"
    try:
        with sqlite3.connect(scheduled_tasks.db_path) as conn:
            cur = conn.execute(
                "SELECT id FROM scheduled_tasks WHERE name = ? AND frequency = 'daily'",
                (task_name,),
            )
            if cur.fetchone():
                logger.info("Daily memory review task already exists; skip create")
                return

        result = scheduled_tasks.create_task(
            session_id=session_id,
            name=task_name,
            task_type="custom",
            scheduled_time="09:00",
            description=(
                "Spaced-repetition refresh: review due items and ingest high-salience chat turns."
            ),
            frequency="daily",
            action_name="memory_review",
            action_args={"task_type": "memory_review"},
            max_runs=None,
        )
        if result.get("success"):
            logger.info("Created daily memory review task; next_run=%s", result.get("next_run"))
        else:
            logger.error(
                "Failed to create daily memory review task: %s",
                result.get("error", "unknown error"),
            )
    except Exception as e:
        logger.error("create_daily_memory_review_task error: %s", e)


# Background Tasks
def _resting_pulse_loop(db_path: str):
    """
    Resting pulse (“digital breathing”): low-frequency background nudge on ``z_self``.

    Once per minute, no LLM — tiny noise on tail dimensions plus light energy refill.

    v2.0: goal due-soon / overdue checks and notification queue.
    v2.1: optional automatic work logs (mid-frequency / daily; high-frequency path disabled).
    """
    logger.info("Resting pulse loop started (Digital Breathing + Goal Monitoring + Work Logging)")
    global _calendar_gc_last_ts
    from backend.self_model import SelfModel
    from backend.persona_store import PersonaStore
    from backend.goal_manager import GoalManager, NotificationQueue
    from backend.scheduled_tasks import ScheduledTaskManager
    from backend.s_work_logger import get_work_logger
    import numpy as np
    import random
    
    persona_store = PersonaStore(db_path)
    self_model = SelfModel(db_path, persona_store)
    goal_manager = GoalManager(db_path)
    notification_queue = NotificationQueue(db_path)
    scheduled_tasks = ScheduledTaskManager(db_path)
    work_logger = get_work_logger(db_path)

    memory_enhancer = None
    memory_enhancer_ok = False
    try:
        from backend.memory_enhancer import init_memory_enhancer

        memory_enhancer = init_memory_enhancer(db_path)
        memory_enhancer_ok = True
        logger.info("MemoryEnhancer initialized")
    except ImportError as e:
        logger.warning("MemoryEnhancer unavailable (import failed): %s", e)
    except Exception as e:
        logger.error("MemoryEnhancer init failed: %s", e)

    default_session_id = config.get("system.default_session_id", "demo-session")
    try:
        create_daily_memory_review_task(scheduled_tasks, default_session_id)
    except Exception as e:
        logger.error("Failed to register daily memory review task: %s", e)

    last_goal_check: dict = {}  # per-session last goal-notification scan (epoch seconds)
    
    while True:
        try:
            with sqlite3.connect(db_path) as conn:
                cur = conn.execute("SELECT session_id FROM self_state")
                sessions = [row[0] for row in cur.fetchall()]
            
            for sid in sessions:
                # z_self micro-perturbation (persona head untouched)
                z_self = self_model.get_z_self(sid)
                if z_self is not None:
                    noise = np.zeros_like(z_self)
                    noise[32:] = (np.random.rand(z_self.shape[0] - 32) - 0.5) * 0.0001
                    z_self = np.clip(z_self + noise, -1.0, 1.0)
                    self_model.save_z_self(sid, z_self)
                    
                    self_model.update_energy(sid, 0.05)

                import time as time_module
                current_time = time_module.time()
                last_check = last_goal_check.get(sid, 0)
                
                if current_time - last_check >= 600:  # 10 minutes
                    last_goal_check[sid] = current_time

                    due_goals = goal_manager.check_due_soon(sid, hours=24)
                    for goal in due_goals:
                        hours_left = goal.get("hours_remaining", 24)
                        
                        if hours_left <= 2:
                            priority = 10
                            urgency = "Urgent"
                        elif hours_left <= 6:
                            priority = 8
                            urgency = "High"
                        elif hours_left <= 12:
                            priority = 6
                            urgency = "Notice"
                        else:
                            priority = 4
                            urgency = "Reminder"
                        
                        existing = notification_queue.get_pending(sid, limit=50)
                        already_notified = any(
                            n.get("metadata") and 
                            goal["id"] in str(n.get("metadata", ""))
                            for n in existing
                        )
                        
                        if not already_notified:
                            notification_queue.add(
                                session_id=sid,
                                notification_type="goal_due",
                                title=f"[{urgency}] Goal due soon: {goal['title']}",
                                content=f"Time left: {hours_left:.1f} h",
                                priority=priority,
                                metadata={"goal_id": goal["id"], "hours_remaining": hours_left}
                            )
                            logger.info(f"Goal due notification added for {sid}: {goal['title']}")
                    
                    overdue_goals = goal_manager.get_overdue_goals(sid)
                    for goal in overdue_goals:
                        existing = notification_queue.get_pending(sid, limit=50)
                        already_notified = any(
                            n.get("notification_type") == "goal_overdue" and
                            n.get("metadata") and 
                            goal["id"] in str(n.get("metadata", ""))
                            for n in existing
                        )
                        
                        if not already_notified:
                            notification_queue.add(
                                session_id=sid,
                                notification_type="goal_overdue",
                                title=f"[Overdue] Goal not completed: {goal['title']}",
                                content=f"Deadline: {goal['deadline'][:10]}",
                                priority=9,
                                metadata={"goal_id": goal["id"], "overdue": True}
                            )
                            logger.info(f"Overdue goal notification added for {sid}: {goal['title']}")
                
                # Due scheduled tasks → enqueue only; UnifiedScheduler runs the LLM turn
                try:
                    from datetime import datetime, timedelta, timezone
                    from backend.unified_scheduler import get_scheduler, PRIORITY_SCHEDULED
                    _sched = get_scheduler()
                    due_tasks = scheduled_tasks.get_due_tasks(sid)
                    for task in due_tasks[:1]:
                        task_id = task.get("id", "")
                        task_name = task.get("name", "Reminder")
                        task_desc = task.get("description", "") or "Time is up"

                        if _sched and _sched.is_queued(f"sched-{task_id}"):
                            continue

                        _five_min_ago = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
                        _already_fired = False
                        try:
                            with sqlite3.connect(db_path) as _dup_conn:
                                _dup_cur = _dup_conn.execute(
                                    "SELECT COUNT(*) FROM notification_queue "
                                    "WHERE session_id = ? AND created_at >= ? AND metadata LIKE ?",
                                    (sid, _five_min_ago, f'%{task_id}%')
                                )
                                _already_fired = (_dup_cur.fetchone()[0] or 0) > 0
                        except Exception:
                            pass
                        if _already_fired:
                            continue

                        try:
                            from backend.autonomy_gate import is_autonomous_execution_paused
                            if is_autonomous_execution_paused(sid):
                                logger.debug(
                                    "[SCHEDULED] Skipped due task (autonomy pause): %s",
                                    task_name,
                                )
                                continue
                        except Exception:
                            pass

                        log_id = scheduled_tasks.start_execution(task_id)

                        notification_queue.add(
                            session_id=sid,
                            notification_type="reminder",
                            title=f"Scheduled reminder: {task_name}",
                            content=task_desc,
                            priority=8,
                            metadata={"task_id": task_id, "scheduled_task": True}
                        )

                        prompt = (
                            f"[Scheduled task due] Name: {task_name}. "
                            f"Details: {task_desc}. "
                            f"Carry out this task using your tools as needed "
                            f"(e.g. write_file for notes, tavily_search, "
                            f"request_mind_wandering, etc.)."
                        )
                        if _sched:
                            _sched.enqueue(
                                priority=PRIORITY_SCHEDULED,
                                task_type="scheduled_task",
                                task_id=f"sched-{task_id}",
                                prompt=prompt,
                                session_id=sid,
                                metadata={
                                    "post_type": "scheduled_task",
                                    "task_id": task_id,
                                    "log_id": log_id,
                                },
                            )
                            logger.info(f"[SCHEDULED] Enqueued for {sid}: {task_name}")
                        else:
                            logger.warning(f"[SCHEDULED] No scheduler, task {task_id} skipped")

                except Exception as st_err:
                    logger.warning(f"[SCHEDULED] Task check failed for {sid}: {st_err}")
                
                # Calendar due reminders (events from calendar_tool) → same session as UI
                try:
                    from backend.tools.calendar_tool import CalendarTool
                    from backend.config import config as _app_config

                    cal = CalendarTool(db_path)
                    batch_max = int(
                        _app_config.get("parameters.calendar.reminder_batch_max", 15) or 15
                    )
                    batch_max = max(1, min(batch_max, 50))
                    due_events = cal.get_due_events_for_reminder(sid)[:batch_max]
                    if not due_events:
                        pass
                    else:
                        evt_ids = [str(e.get("id") or "").strip() for e in due_events]
                        evt_ids = [x for x in evt_ids if x]
                        if not evt_ids:
                            pass
                        else:
                            batch_key = ",".join(sorted(evt_ids))
                            existing = notification_queue.get_pending(sid, limit=50)

                            def _calendar_batch_already_pending() -> bool:
                                for n in existing:
                                    m = n.get("metadata") or {}
                                    if not m.get("calendar_event"):
                                        continue
                                    if m.get("calendar_batch") and str(
                                        m.get("event_ids", "")
                                    ) == batch_key:
                                        return True
                                    if not m.get("calendar_batch") and len(evt_ids) == 1:
                                        if m.get("event_id") == evt_ids[0]:
                                            return True
                                return False

                            if _calendar_batch_already_pending():
                                pass
                            else:
                                agent_name = (
                                    str(_app_config.get("system.agent_name") or "s-44").strip()
                                    or "s-44"
                                )
                                if len(due_events) == 1:
                                    evt = due_events[0]
                                    evt_id = evt_ids[0]
                                    evt_title = evt.get("title", "Reminder")
                                    evt_desc = evt.get("description", "") or evt_title
                                    prompt = (
                                        f"[Calendar due · system → assistant {agent_name}] "
                                        f"Event: {evt_title}. Notes: {evt_desc}. "
                                        f"It is time; reply to the user briefly and naturally "
                                        f"(what you will do / did); use tools if needed."
                                    )
                                    notif_title = f"Calendar reminder: {evt_title}"
                                    notif_content = evt_desc
                                    ws_prompt = f"[Calendar] {evt_title}"
                                    meta = {
                                        "event_id": evt_id,
                                        "calendar_event": True,
                                    }
                                else:
                                    lines = []
                                    for i, evt in enumerate(due_events, start=1):
                                        t = evt.get("title", "Reminder")
                                        d = (evt.get("description") or "").strip() or t
                                        lines.append(f"{i}. [{t}] — {d}")
                                    bul = "\n".join(lines)
                                    prompt = (
                                        f"[Calendar due · system → assistant {agent_name}] "
                                        f"{len(due_events)} calendar items are due (batched). "
                                        f"Reply concisely, optionally item by item; use tools if needed.\n{bul}"
                                    )
                                    notif_title = f"{len(due_events)} calendar item(s) due"
                                    notif_content = bul
                                    ws_prompt = f"[Calendar ×{len(due_events)}] " + ", ".join(
                                        (e.get("title") or "Reminder") for e in due_events[:5]
                                    )
                                    if len(due_events) > 5:
                                        ws_prompt += "…"
                                    meta = {
                                        "calendar_event": True,
                                        "calendar_batch": True,
                                        "event_ids": batch_key,
                                    }
                                from backend.unified_scheduler import get_scheduler, PRIORITY_SCHEDULED
                                _cal_sched = get_scheduler()
                                _cal_skip = False
                                try:
                                    from backend.autonomy_gate import is_autonomous_execution_paused
                                    if is_autonomous_execution_paused(sid):
                                        _cal_skip = True
                                        logger.debug(
                                            "[CALENDAR] Skipped LLM enqueue (autonomy pause), sid=%s",
                                            sid,
                                        )
                                except Exception:
                                    pass
                                if (
                                    _cal_sched
                                    and not _cal_skip
                                    and not _cal_sched.is_queued(f"cal-{batch_key}")
                                ):
                                    _cal_sched.enqueue(
                                        priority=PRIORITY_SCHEDULED,
                                        task_type="calendar_event",
                                        task_id=f"cal-{batch_key}",
                                        prompt=prompt,
                                        session_id=sid,
                                        metadata={
                                            "post_type": "calendar_event",
                                            "evt_ids": evt_ids,
                                            "notif_title": notif_title,
                                            "notif_content": notif_content,
                                            "notif_meta": meta,
                                        },
                                    )
                                    logger.info(
                                        "Calendar event enqueued for %s: %s event(s)",
                                        sid,
                                        len(evt_ids),
                                    )
                except Exception as cal_err:
                    logger.debug(f"Calendar reminder check failed for {sid}: {cal_err}")
                
                try:
                    # High-frequency work log (15 min) — disabled to reduce churn
                    # if work_logger.should_log_high_freq(sid):
                    #     summary = self_model.get_structured_summary(sid)
                    #     z_self_data = {
                    #         "openness": summary.get("openness", 0),
                    #         "stability": summary.get("stability", 0),
                    #         "confidence": summary.get("confidence", 0),
                    #         "curiosity": summary.get("curiosity", 0),
                    #     }
                    #     energy = summary.get("energy", 100.0)
                    #     
                    #     work_logger.log_high_freq(
                    #         session_id=sid,
                    #         z_self_summary=z_self_data,
                    #         energy=energy,
                    #         current_task="",  # could be wired to goal system later
                    #         notes=""
                    #     )
                    #     logger.debug(f"High freq log recorded for {sid}")
                    
                    if work_logger.should_log_mid_freq(sid):
                        active_goals = goal_manager.get_active_goals(sid)
                        
                        work_logger.log_mid_freq(
                            session_id=sid,
                            project_status={"active": True},
                            goals=active_goals,
                            resource_usage={},
                            todos=[]
                        )
                        logger.debug(f"Mid freq log recorded for {sid}")
                    
                    if work_logger.should_log_daily(sid):
                        summary = self_model.get_structured_summary(sid)
                        
                        work_logger.log_daily(
                            session_id=sid,
                            summary=f"Energy: {summary.get('energy', 0):.1f}%",
                            achievements=[],
                            challenges=[],
                            next_day_plan="",
                            self_reflection=""
                        )
                        logger.info(f"Daily log recorded for {sid}")
                        
                except Exception as log_error:
                    logger.error(f"Work logging error for {sid}: {log_error}")

            # Calendar DB GC: archive ended events / optional purge (global, throttled)
            try:
                gc_iv = float(config.get("parameters.calendar.gc_interval_seconds", 21600) or 21600)
                if gc_iv > 0:
                    now_ts = time.time()
                    if now_ts - _calendar_gc_last_ts >= gc_iv:
                        _calendar_gc_last_ts = now_ts
                        from backend.tools.calendar_tool import CalendarTool

                        stats = CalendarTool(db_path).run_maintenance_gc()
                        if stats.get("archived") or stats.get("purged"):
                            logger.info("[CALENDAR-GC] %s", stats)
            except Exception as cgc_err:
                logger.debug(f"Calendar GC failed: {cgc_err}")

        except Exception as e:
            logger.error(f"Resting pulse error: {e}")

        if memory_enhancer_ok and memory_enhancer:
            try:
                memory_enhancer.background_process()
            except Exception as me_err:
                logger.debug("Memory enhancer background_process skipped: %s", me_err)

        time.sleep(60)  # one-minute tick

def _dreaming_loop(db_path: str):
    logger.info("Dreaming loop started (Conditional Mode)")
    from backend.self_tick import SelfTick
    from backend.self_model import SelfModel
    from backend.persona_store import PersonaStore
    import random
    
    persona_store = PersonaStore(db_path)
    self_model = SelfModel(db_path, persona_store)
    ticker = SelfTick(db_path, self_model, persona_store)
    
    check_interval = config.get("system.dreaming_check_interval", 3600)
    only_when_idle = config.get("system.dreaming_only_when_idle", True)

    # Defer first tick after process start to avoid burst on restart
    logger.info(f"[DREAMING] Waiting {check_interval}s before first check...")
    time.sleep(check_interval)
    
    while True:
        try:
            with sqlite3.connect(db_path) as conn:
                cur = conn.execute("SELECT session_id FROM self_state")
                sessions = [row[0] for row in cur.fetchall()]

                for sid in sessions:
                    summary = self_model.get_structured_summary(sid)
                    energy = summary.get("energy", 100.0)

                    # Dreaming only when energy is low (matches process_dreaming gate)
                    if energy > 50.0:
                        continue

                    if only_when_idle:
                        try:
                            cur_research = conn.execute(
                                "SELECT status FROM research_state WHERE session_id=?", (sid,)
                            )
                            row = cur_research.fetchone()
                            if row and row[0] == 'researching':
                                logger.debug(f"Skip dreaming for {sid}: research in progress")
                                continue

                            from datetime import datetime, timedelta, timezone
                            soon = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()
                            cur_tasks = conn.execute(
                                "SELECT COUNT(*) FROM scheduled_tasks WHERE session_id=? AND next_run<=? AND enabled=1",
                                (sid, soon)
                            )
                            task_count = cur_tasks.fetchone()[0]
                            if task_count > 0:
                                logger.debug(f"Skip dreaming for {sid}: {task_count} tasks due soon")
                                continue
                        except Exception as idle_check_err:
                            logger.debug(f"Idle check failed: {idle_check_err}")

                    if random.random() > 0.25:
                        continue

                    logger.info(f"Feels like dreaming for session {sid} (Energy: {energy:.1f})")
                    ticker.process_dreaming(sid)

        except Exception as e:
            logger.error(f"Dreaming loop error: {e}")

        jitter = random.randint(-300, 300)
        time.sleep(check_interval + jitter)

def _promotion_cron_loop(db_path: str, interval_minutes: int):
    gate = PromotionGate(db_path)
    iv = max(interval_minutes, 1)
    logger.info(f"Promotion cron started (every {iv} minutes)")
    while True:
        try:
            res = gate.auto_promote()
            if res.get("promoted"):
                logger.info(f"Promotion cron promoted: {res}")
        except Exception as e:
            logger.debug(f"Promotion cron failed: {e}")
        time.sleep(iv * 60)

# Phase 3: presence pulse (optional; scheduler may supersede in some deployments)
def _spontaneous_action_loop(db_path: str):
    """
    Presence pulse (v5.3+): periodic nudge so the long-lived agent re-orients.

    Philosophy: ask clearly what to do next without forcing a menu; the agent may
    use tools, reply, rest, ``request_mind_wandering``, or ``set_my_rhythm``.

    [2026-02-05] Default interval widened (15m -> 30m) to save tokens.
    """
    base_interval = config.get("system.presence_pulse_interval", 1800)
    # ``PRESENCE_PULSE_INTERVAL`` (seconds) overrides YAML when set in the environment
    INTERVAL_SECONDS = int(os.environ.get("PRESENCE_PULSE_INTERVAL", str(base_interval)))
    INTERVAL_MINUTES = INTERVAL_SECONDS // 60
    
    print("=" * 60)
    print("Presence Pulse started (v5.4 - Token Optimized)")
    print("Philosophy: Ask clearly, but don't limit options")
    print("System asks: 'What do you want to do?'")
    print("Agent decides: Freely, based on z_self")
    print(f"Interval: {INTERVAL_MINUTES} minutes ({INTERVAL_SECONDS} seconds)")
    print("[2026-02-05] Token tuning: 30 min interval, last 3 turns of history")
    print("=" * 60)
    logger.info("=" * 60)
    logger.info("Presence Pulse started (v5.4 - Token Optimized)")
    logger.info(f"Interval: {INTERVAL_MINUTES} minutes ({INTERVAL_SECONDS} seconds)")
    logger.info("=" * 60)
    
    from backend.routers.chat import get_chat_service
    
    logger.info(f"Presence Pulse interval: {INTERVAL_MINUTES} minutes ({INTERVAL_SECONDS} seconds)")
    
    print("[PRESENCE] Getting ChatService singleton...")
    chat_service = get_chat_service()
    print("[PRESENCE] ChatService ready (singleton)")
    
    SESSION_ID = "selfing-session"

    # Initial delay avoids immediate LLM spend on rapid restarts
    print(f"[PRESENCE] Waiting {INTERVAL_MINUTES} minutes before first pulse...")
    logger.info(f"[PRESENCE] Waiting {INTERVAL_MINUTES} minutes before first pulse...")
    time.sleep(INTERVAL_SECONDS)
    
    while True:
        try:
            import datetime
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            simple_message = (
                f"[system reminder {now}] You may choose what to do next; "
                f"when ready, act (tools, reply, or rest)."
            )
            
            logger.info(f"[PRESENCE] Asking Agent: {simple_message}")
            print(f"[PRESENCE] Asking Agent: {simple_message}")
            
            try:
                response = chat_service.chat(
                    user_input=simple_message,
                    session_id=SESSION_ID,
                    temperature=0.7,
                    disable_intent_detection=False,
                )
                
                answer = None
                if response and response.get("content"):
                    answer = response["content"]
                elif response and response.get("response"):
                    answer = response["response"]
                
                if answer:
                    short_answer = answer[:100].replace('\n', ' ') + "..." if len(answer) > 100 else answer.replace('\n', ' ')
                    logger.info(f"[PRESENCE] Agent responds: {short_answer}")
                    print(f"[PRESENCE] Agent responds: {short_answer}")
                    
                    try:
                        ws_manager = get_websocket_manager()
                        ws_message = create_ws_message(
                            msg_type=WSMessageType.PRESENCE_PULSE,
                            content=answer,
                            session_id=SESSION_ID,
                            trigger="presence_pulse",
                            prompt=simple_message
                        )
                        ws_manager.queue_message(SESSION_ID, ws_message)
                        logger.info(f"[PRESENCE] WebSocket message queued for {SESSION_ID}")
                    except Exception as ws_err:
                        logger.debug(f"[PRESENCE] WebSocket push skipped: {ws_err}")
                else:
                    logger.warning(f"[PRESENCE] No response from Agent, got: {type(response)}")
                    print(f"[PRESENCE] No response from Agent, got: {response}")
                    
            except Exception as chat_error:
                logger.error(f"[PRESENCE] Chat error: {chat_error}")
                print(f"[PRESENCE] Chat error: {chat_error}")
                import traceback
                traceback.print_exc()
                    
        except Exception as e:
            logger.error(f"[PRESENCE] Error in presence pulse: {e}")
            print(f"[PRESENCE] Error: {e}")
        
        logger.info(f"[PRESENCE] Next check in {INTERVAL_MINUTES} minutes...")
        time.sleep(INTERVAL_SECONDS)

@app.on_event("startup")
async def on_startup():
    """
    FastAPI startup hook.

    Background threads are started at module import; this only wires the WebSocket manager
    to the running asyncio loop (avoids duplicate pulse threads).
    """
    startup_log = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "startup_simple.log")
    
    try:
        manager = get_websocket_manager()
        manager.set_event_loop(asyncio.get_event_loop())
        print("[STARTUP] WebSocket manager event loop configured")
        logger.info("[STARTUP] WebSocket manager event loop configured")
        
        with open(startup_log, "w") as f:
            f.write("✅ STARTUP EVENT TRIGGERED\n")
            f.write("ℹ️ Background threads already started at module load\n")
            f.write("✅ WebSocket manager configured\n")
            f.write(f"✅ Total threads: {threading.active_count()}\n")
            f.flush()
        
        print("[STARTUP] FastAPI startup event - threads already running from module load")
        print(f"[STARTUP] Active threads: {threading.active_count()}")
        
    except Exception as e:
        print(f"[STARTUP] Error: {e}")
        logger.error(f"[STARTUP] Error: {e}")


# Background threads start at module import because ``startup`` is not reliable in all ASGI hosts.

def _start_background_threads_at_module_load():
    """Start daemon threads at import time (does not rely on FastAPI ``startup``)."""
    try:
        print("=" * 60)
        print("[MODULE LOAD] Starting background threads...")
        print("=" * 60)
        logger.info("=" * 60)
        logger.info("[MODULE LOAD] Starting background threads...")
        logger.info("=" * 60)
        
        run_schema_migrations(DB_PATH)
        print("[MODULE] Schema migrations completed")
        logger.info("[MODULE] Schema migrations completed")
        
        if config.get("db_cleanup.enabled", True) and config.get("db_cleanup.run_on_startup", True):
            try:
                from backend.db_cleanup import get_db_cleaner
                cleaner = get_db_cleaner(DB_PATH)
                cleanup_result = cleaner.cleanup_all(dry_run=False)
                total_cleaned = cleanup_result.get("total_deleted", 0)
                print(f"[MODULE] ✅ Database cleanup completed: {total_cleaned} old records removed")
                logger.info(f"[MODULE] Database cleanup completed: {cleanup_result}")
            except Exception as cleanup_err:
                print(f"[MODULE] ⚠️ Database cleanup failed: {cleanup_err}")
                logger.warning(f"[MODULE] Database cleanup failed: {cleanup_err}")
        
        from backend.routers.backup import init_backup_routes
        init_backup_routes(DB_PATH)
        print("[MODULE] Backup system initialized")
        logger.info("[MODULE] Backup system initialized")
        
        t_pulse = threading.Thread(target=_resting_pulse_loop, args=(DB_PATH,), daemon=True)
        t_pulse.start()
        print("[MODULE] ✅ Resting Pulse activated")
        logger.info("[MODULE] ✅ Resting Pulse activated")
        
        if config.get("system.dreaming_enabled", False):
            t_dream = threading.Thread(target=_dreaming_loop, args=(DB_PATH,), daemon=True)
            t_dream.start()
            print("[MODULE] ✅ Dreaming loop activated")
            logger.info("[MODULE] ✅ Dreaming loop activated")
        else:
            print("[MODULE] ⚠️ Dreaming loop disabled by config")
            logger.info("[MODULE] ⚠️ Dreaming loop disabled by config")
        
        from backend.unified_scheduler import init_scheduler
        _unified_scheduler = init_scheduler(DB_PATH)
        _unified_scheduler.start()
        print("[MODULE] ✅ Unified Scheduler started (replaces spontaneous + continuous loops)")
        logger.info("[MODULE] ✅ Unified Scheduler started")
        
        if config.get("system.heartbeat_enabled", False):
            try:
                from backend.heartbeat_service import HeartbeatService
                from backend.unified_scheduler import get_scheduler, PRIORITY_HEARTBEAT
                def _on_heartbeat(prompt: str):
                    sched = get_scheduler()
                    if sched:
                        try:
                            from backend.autonomy_gate import is_autonomous_execution_paused
                            if is_autonomous_execution_paused("selfing-session"):
                                logger.debug("[HEARTBEAT] Skipped enqueue (autonomy pause)")
                                return
                        except Exception:
                            pass
                        sched.enqueue(
                            priority=PRIORITY_HEARTBEAT,
                            task_type="heartbeat",
                            task_id=f"heartbeat-{int(time.time())}",
                            prompt=prompt,
                            session_id="selfing-session",
                        )
                _heartbeat_svc = HeartbeatService(on_heartbeat=_on_heartbeat)
                _heartbeat_svc.start()
                print("[MODULE] ✅ HeartbeatService started (→ UnifiedScheduler)")
                logger.info("[MODULE] ✅ HeartbeatService started (→ UnifiedScheduler)")
            except Exception as e:
                logger.warning(f"[MODULE] HeartbeatService failed to start: {e}")
        else:
            print("[MODULE] ⚠️ HeartbeatService disabled by config")
            logger.info("[MODULE] ⚠️ HeartbeatService disabled by config")
        
        print("=" * 60)
        print("[MODULE LOAD] All background threads started successfully!")
        print("=" * 60)
        logger.info("=" * 60)
        logger.info("[MODULE LOAD] All background threads started successfully!")
        logger.info("=" * 60)
        
    except Exception as e:
        print(f"[MODULE] ❌ Failed to start background threads: {e}")
        import traceback
        traceback.print_exc()
        logger.error(f"[MODULE] Failed to start background threads: {e}", exc_info=True)


_start_background_threads_at_module_load()
