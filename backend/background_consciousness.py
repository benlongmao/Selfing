"""
[DEPRECATED — 2026-03-25]

This module is **obsolete** and is no longer started from ``app.py``.

Why it was retired:
1. Trigger design issues:
   - ``connection < 0.10`` was almost never hit (default ~0.8, slow decay, fast refill from chat)
   - Urges were not refreshed in the background thread, so ``get_urges()`` stayed stale
2. Superseded by:
   - Resting pulse / heartbeat tasks
   - Scheduled self tick
   - ``EventTriggeredSelfTickManager`` and related paths

Kept only as historical reference.
"""

import threading
import time
import logging
import sqlite3
import os
from typing import Dict, List, Optional
from backend.core.homeostasis import HomeostasisSystem
from backend.core.endogenous_system import EndogenousSystem
from backend.config import config

logger = logging.getLogger(__name__)


class GlobalWorkspace:
    """
    [DEPRECATED] Global workspace / continuity stream.

    Previously ran a long-lived background loop for proactive attention and spontaneous chat.
    """

    def __init__(self, db_path: str, chat_service):
        self.db_path = db_path
        self.chat_service = chat_service
        self.homeostasis = HomeostasisSystem(db_path)
        self.endogenous = EndogenousSystem(data_dir=os.path.dirname(db_path))
        self._stop_event = threading.Event()
        self._thread = None

    def start(self, interval: int = 1800):
        """
        Start the background loop.

        [2026-02-05] Token budget: default cadence moved from 15 minutes to 30 minutes (1800s).
        """
        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, args=(interval,), daemon=True)
        self._thread.start()
        logger.info(f"GlobalWorkspace (Continuity Stream) started with interval {interval}s")

    def stop(self):
        """Stop the worker thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join()

    def _loop(self, interval: int):
        # [2026-02-07] Sleep once before first tick so restarts do not fire immediately
        logger.info(f"[GlobalWorkspace] Waiting {interval}s before first check...")
        time.sleep(interval)

        while not self._stop_event.is_set():
            try:
                self._process_sessions()
            except Exception as e:
                logger.error(f"Error in GlobalWorkspace loop: {e}")

            time.sleep(interval)

    def _process_sessions(self):
        """Iterate sessions and maybe enqueue proactive chat."""
        # [2026-02-03] Workspace cleanup hook (internally rate-limited)
        self._maybe_cleanup_workspace()

        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("SELECT session_id FROM self_state")
            sessions = [row[0] for row in cur.fetchall()]

        for session_id in sessions:
            needs = self.homeostasis.load_needs(session_id)
            energy = self.homeostasis.get_energy(session_id)

            if energy < 20.0:
                continue  # too depleted for proactive outreach

            urges = self.endogenous.get_urges(session_id)

            should_initiate = False
            trigger_reason = ""

            # [2026-01-24] Loosened loneliness trigger (0.15 → 0.40)
            # [2026-02-07] Tightened again (0.40 → 0.10) to cut token spend; 0.10 ~= extreme isolation
            if needs.get("connection", 1.0) < 0.10:
                should_initiate = True
                trigger_reason = "[Extreme loneliness] The system spontaneously seeks connection."
            elif any("[Strong" in u for u in urges):
                should_initiate = True
                trigger_reason = f"[Strong urge] {next(u for u in urges if '[Strong' in u)}"

            if should_initiate:
                logger.info(f"🚀 GlobalWorkspace: Triggering proactive action for {session_id} due to {trigger_reason}")
                # [2026-01-27] Disable intent detection in this loop to avoid runaway multi-turn cost
                self.chat_service.chat(
                    user_input=f"[INTERNAL_WAKEUP] {trigger_reason}",
                    session_id=session_id,
                    disable_intent_detection=True,
                )

    def _maybe_cleanup_workspace(self):
        """Run workspace temp cleanup at most once per hour."""
        try:
            last_cleanup_file = os.path.join(os.path.dirname(self.db_path), ".last_workspace_cleanup")
            now = time.time()

            if os.path.exists(last_cleanup_file):
                with open(last_cleanup_file, "r") as f:
                    last_cleanup = float(f.read().strip())
                if now - last_cleanup < 3600:
                    return

            from backend.workspace_manager import get_workspace_manager

            ws_manager = get_workspace_manager(db_path=self.db_path)

            cleanup_result = ws_manager.cleanup_temp(days_threshold=7)
            if cleanup_result.get("deleted"):
                logger.info(f"🧹 WorkspaceManager: Cleaned {len(cleanup_result['deleted'])} temp files")

            with open(last_cleanup_file, "w") as f:
                f.write(str(now))

        except Exception as e:
            logger.debug(f"Workspace cleanup skipped: {e}")
