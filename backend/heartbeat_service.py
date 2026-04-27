#!/usr/bin/env python3
"""
Periodic heartbeat: wake the agent to process tasks listed in HEARTBEAT.md.
Inspired by nanobot's HeartbeatService pattern.

S already has a resting pulse loop that only perturbs z_self and checks goals.
This service adds an editable HEARTBEAT.md task list so both users and the agent
can define recurring work.

Flow:
1. Every N minutes read workspace/sandbox/HEARTBEAT.md
2. If there is actionable content, invoke the registered callback (typically ChatService)
3. HEARTBEAT.md can be edited by the agent via tools

[2026-02-07] Phase 4 — heartbeat service
"""

import logging
import os
import re
import time
import threading
from typing import Optional, Callable, Tuple, List

from backend.config import config

logger = logging.getLogger(__name__)

# Default interval between ticks (seconds)
DEFAULT_INTERVAL = 1800  # 30 minutes

# Default path to HEARTBEAT.md (inside sandbox; agent can read/write)
DEFAULT_HEARTBEAT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "workspace",
    "sandbox",
    "HEARTBEAT.md"
)


class HeartbeatService:
    """
    Heartbeat worker.

    Periodically reads HEARTBEAT.md and fires the callback when there is work.

    Suggested HEARTBEAT.md shape:
    # Heartbeat Tasks

    ## Active Tasks
    - [ ] Check system status every 30 minutes
    - [ ] If the user has been idle >2h, send a brief check-in

    ## Completed
    - [x] Initialize workspace (done)

    <!-- If the file is only headings and comments, the heartbeat tick is skipped -->
    """

    def __init__(
        self,
        heartbeat_path: str = DEFAULT_HEARTBEAT_PATH,
        interval_s: Optional[int] = None,
        on_heartbeat: Optional[Callable[[str], None]] = None,
    ):
        self.heartbeat_path = heartbeat_path
        self.interval_s = interval_s or config.get("system.heartbeat_interval", DEFAULT_INTERVAL)
        self.on_heartbeat = on_heartbeat

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._tick_count = 0  # [2026-04-07] throttle wiki lint / deep compile

        self._ensure_heartbeat_file()

        logger.info(f"HeartbeatService initialized: interval={self.interval_s}s, path={self.heartbeat_path}")

    def _ensure_heartbeat_file(self):
        """Create HEARTBEAT.md with an English template if missing."""
        if not os.path.exists(self.heartbeat_path):
            os.makedirs(os.path.dirname(self.heartbeat_path), exist_ok=True)
            with open(self.heartbeat_path, "w", encoding="utf-8") as f:
                f.write("""# Heartbeat Tasks

This file is scanned on each heartbeat. Agents and users can add recurring tasks here.

## Active Tasks
<!-- Add actionable checklist items below -->
<!-- Optional: create workspace/sandbox/AGENT_FOCUS.md with current priorities / must-read paths -->

## Completed
<!-- Move finished items here -->

## Agent external memory snapshot (optional; maintained by agent_memory_sync)

<!-- AGENT_MEMORY:BEGIN -->

_(Not synced yet; call agent_memory_sync with inject_markdown_path = workspace/sandbox/HEARTBEAT.md or HEARTBEAT.md.)_

<!-- AGENT_MEMORY:END -->

<!-- Heartbeat skips when the file has no real tasks (only headings/comments) -->
""")
            logger.info(f"Created default HEARTBEAT.md at {self.heartbeat_path}")

    def start(self):
        """Start the background heartbeat thread."""
        if self._running:
            logger.warning("HeartbeatService already running")
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
            name="HeartbeatService"
        )
        self._thread.start()
        logger.info(f"HeartbeatService started (interval: {self.interval_s}s)")

    def stop(self):
        """Stop the heartbeat thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("HeartbeatService stopped")

    def _heartbeat_loop(self):
        """Main sleep / tick loop."""
        while self._running:
            try:
                self._tick()
            except Exception as e:
                logger.error(f"Heartbeat tick error: {e}")

            # Wait until the next tick window
            for _ in range(int(self.interval_s)):
                if not self._running:
                    break
                time.sleep(1)

    def _tick(self):
        """One heartbeat: read file, trim completed history, optional wiki pass, callback."""
        if not os.path.exists(self.heartbeat_path):
            return

        try:
            with open(self.heartbeat_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            logger.warning(f"Failed to read HEARTBEAT.md: {e}")
            return

        if self._is_empty(content):
            logger.debug("Heartbeat: no actionable content, skipping")
            return

        # Parse, trim ## Completed, rewrite atomically
        header_and_active, completed_blocks = self._parse_heartbeat_content(content)
        keep = int(config.get("system.heartbeat_completed_blocks_to_keep", 5) or 5)
        trimmed = self._trim_completed_blocks(completed_blocks, keep)
        if len(trimmed) < len(completed_blocks):
            new_content = self._rebuild_heartbeat_file(header_and_active, trimmed)
            self._atomic_write(new_content)
            logger.info(f"Heartbeat trimmed Completed: {len(completed_blocks)} -> {len(trimmed)} blocks")

        # [2026-04-07] wiki maintenance: lint every 3 ticks, deep compile every 10
        self._tick_count += 1
        if self._tick_count % 3 == 0:
            try:
                from backend.knowledge_compiler import get_compiler
                compiler = get_compiler()
                issues = compiler.lint()
                if issues:
                    logger.info(f"[WIKI-LINT] {len(issues)} issues found, logged to _lint_log.md")
                if self._tick_count % 10 == 0:
                    result = compiler.deep_compile(max_pages=3)
                    if result.get("compiled", 0):
                        logger.info(f"[WIKI] Deep compiled {result['compiled']} pages")
            except Exception as e:
                logger.debug(f"Wiki maintenance skipped: {e}")

        # Callback prompt uses Active section only to save tokens
        if self.on_heartbeat:
            prompt = self._build_heartbeat_prompt(header_and_active)
            logger.info(f"Heartbeat triggered with {len(header_and_active)} chars (Active only)")
            try:
                self.on_heartbeat(prompt)
            except Exception as e:
                logger.error(f"Heartbeat callback failed: {e}")
        else:
            logger.debug("Heartbeat: no callback registered, skipping")

    def _parse_heartbeat_content(self, content: str) -> Tuple[str, List[str]]:
        """
        Split HEARTBEAT.md into (header_and_active, completed_blocks).
        completed_blocks are ordered newest-first after split.
        """
        parts = content.split("## Completed", 1)
        header_and_active = parts[0].rstrip()
        if len(parts) < 2:
            return header_and_active, []
        completed_raw = parts[1].lstrip()
        if not completed_raw:
            return header_and_active, []
        # Split on ### headings; first chunk may lack a leading ###
        raw_blocks = [b.strip() for b in completed_raw.split("\n### ") if b.strip()]
        return header_and_active, raw_blocks

    def _trim_completed_blocks(self, blocks: List[str], keep: int) -> List[str]:
        """Keep only the newest `keep` completed blocks."""
        if keep <= 0 or not blocks:
            return blocks
        return blocks[:keep]

    def _rebuild_heartbeat_file(self, header_and_active: str, completed_blocks: List[str]) -> str:
        """Rebuild full markdown; normalize ### prefixes when joining."""
        if not completed_blocks:
            return header_and_active + "\n\n## Completed\n\n"
        normalized = []
        for b in completed_blocks:
            b = b.strip()
            if b and not b.startswith("###"):
                b = "### " + b
            normalized.append(b)
        blocks_text = "\n\n".join(normalized)
        return header_and_active + "\n\n## Completed\n\n" + blocks_text + "\n"

    def _atomic_write(self, content: str) -> None:
        """Write via temp file + replace to avoid torn writes."""
        tmp = self.heartbeat_path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(content)
            os.replace(tmp, self.heartbeat_path)
        except Exception as e:
            logger.warning(f"Heartbeat atomic write failed: {e}")
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except Exception:
                    pass

    def _is_empty(self, content: str) -> bool:
        """True when there are no unchecked tasks and no substantive body text."""
        text = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)
        text = re.sub(r'^#+\s+.*$', '', text, flags=re.MULTILINE)
        text = text.strip()

        if not text:
            return True

        unchecked = re.findall(r'- \[ \]', content)
        if unchecked:
            return False

        lines = [l.strip() for l in text.split('\n') if l.strip()]
        return len(lines) == 0

    def _build_heartbeat_prompt(self, active_content: str) -> str:
        """Build the user/system message for the heartbeat callback (preview + instructions)."""
        from datetime import datetime
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        preview_max = int(
            config.get("system.heartbeat_preview_max_chars", 320) or 320
        )
        lines = [
            line for line in active_content.split('\n')
            if line.strip() and not line.strip().startswith('<!--')
        ]
        joined = '\n'.join(lines)
        preview = joined[:preview_max]
        if len(joined) > preview_max:
            preview += "..."

        if preview.strip():
            return (
                f"[Periodic check — please respond] In one or two sentences, say whether you will "
                f"handle this round's todos; then continue.\n"
                f"The file on disk is the source of truth; any analysis inside it is your prior self — "
                f"re-verify from raw data if unsure.\n\n"
                f"[Heartbeat {now}] Pending tasks (preview):\n{preview}\n\n"
                f"Use read_file('HEARTBEAT.md') for the full file, then update it when done."
            )
        else:
            return f"[Heartbeat {now}] HEARTBEAT.md has no open todos; use the time as you see fit."

    def trigger_now(self):
        """Run one heartbeat tick immediately (manual / tests)."""
        self._tick()
