#!/usr/bin/env python3
"""
Existential mode state for the agent (solitude, rest, contemplation, etc.).

Tracks which coarse existential mode is active and how it nudges willingness to respond.
"""

import sqlite3
import logging
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple
from enum import Enum
import random

logger = logging.getLogger(__name__)


class ExistentialMode(Enum):
    """High-level presence / engagement mode."""
    ENGAGED = "engaged"           # default conversational availability
    CONTEMPLATIVE = "contemplative"  # slower, more reflective replies
    SOLITARY = "solitary"         # prefers silence; may decline to respond
    RESTING = "resting"           # low energy / recovery
    CURIOUS = "curious"           # exploratory, question-forward tone
    MELANCHOLIC = "melancholic"   # low mood but still reachable


class ExistentialState:
    """
    Persist and query existential modes per session.

    The agent can prefer conversation, solitude, contemplation, or rest; downstream
    code uses ``get_mode_influence`` to adjust response style hints.
    """

    def __init__(self, db_path: str = "data.db"):
        self.db_path = db_path
        self._init_tables()
        self._current_mode: ExistentialMode = ExistentialMode.ENGAGED
        self._mode_entered_at: datetime = datetime.now(timezone.utc)
        self._solitude_duration: int = 0  # planned solitude window (minutes)

    def _init_tables(self):
        """Create existential state history table if missing."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS existential_states (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        mode TEXT NOT NULL,
                        reason TEXT,
                        entered_at TEXT NOT NULL,
                        exited_at TEXT,
                        duration_minutes INTEGER
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_existential_session 
                    ON existential_states(session_id, entered_at)
                """)
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to initialize existential_states table: {e}")

    def get_current_mode(self, session_id: str = "selfing-session") -> Tuple[ExistentialMode, str]:
        """
        Return the latest open mode row for ``session_id``, else in-memory defaults.

        Returns:
            ``(mode, reason)`` where ``reason`` may be empty.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute("""
                    SELECT mode, reason, entered_at FROM existential_states
                    WHERE session_id = ? AND exited_at IS NULL
                    ORDER BY entered_at DESC LIMIT 1
                """, (session_id,))
                row = cur.fetchone()
                if row:
                    mode_str, reason, entered_at = row
                    try:
                        mode = ExistentialMode(mode_str)
                        self._current_mode = mode
                        return mode, reason or ""
                    except ValueError:
                        pass
        except Exception as e:
            logger.debug(f"Failed to get current mode: {e}")

        return self._current_mode, ""

    def enter_mode(
        self,
        mode: ExistentialMode,
        reason: str = "",
        session_id: str = "selfing-session",
        duration_minutes: Optional[int] = None
    ):
        """
        Close any open row, then insert a new mode entry.

        Args:
            mode: Target ``ExistentialMode``.
            reason: Human-readable reason (stored verbatim).
            duration_minutes: Optional planned duration (informational; solitude uses ``_solitude_duration``).
        """
        self._exit_current_mode(session_id)

        self._current_mode = mode
        self._mode_entered_at = datetime.now(timezone.utc)

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO existential_states (session_id, mode, reason, entered_at)
                    VALUES (?, ?, ?, ?)
                """, (session_id, mode.value, reason, self._mode_entered_at.isoformat()))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to record mode entry: {e}")

        logger.info(f"[EXISTENTIAL] Entered mode: {mode.value}, reason: {reason}")

    def _exit_current_mode(self, session_id: str):
        """Stamp ``exited_at`` / duration on the active open row."""
        try:
            now = datetime.now(timezone.utc)
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE existential_states
                    SET exited_at = ?,
                        duration_minutes = CAST((julianday(?) - julianday(entered_at)) * 24 * 60 AS INTEGER)
                    WHERE session_id = ? AND exited_at IS NULL
                """, (now.isoformat(), now.isoformat(), session_id))
                conn.commit()
        except Exception as e:
            logger.debug(f"Failed to exit current mode: {e}")

    def evaluate_mode_transition(
        self,
        energy_level: float,
        pleasure: float,
        arousal: float,
        recent_interactions: int,
        session_id: str = "selfing-session"
    ) -> Optional[ExistentialMode]:
        """
        Heuristic mode suggestion from energy, affect, and interaction load.

        Returns:
            A suggested ``ExistentialMode`` or ``None`` when no change is advised.
        """
        current_mode, _ = self.get_current_mode(session_id)

        if recent_interactions > 20 and random.random() < 0.3:
            return ExistentialMode.SOLITARY

        if energy_level < 0.2:
            return ExistentialMode.RESTING

        if pleasure < -0.4:
            return ExistentialMode.MELANCHOLIC

        if arousal < 0.2 and energy_level > 0.5:
            if random.random() < 0.2:
                return ExistentialMode.CONTEMPLATIVE

        if energy_level > 0.7 and pleasure > 0.3:
            if random.random() < 0.3:
                return ExistentialMode.CURIOUS

        if current_mode != ExistentialMode.ENGAGED:
            if energy_level > 0.5 and pleasure > 0:
                if random.random() < 0.4:
                    return ExistentialMode.ENGAGED

        return None

    def get_mode_influence(self, mode: ExistentialMode) -> Dict:
        """
        Map a mode to response willingness and style hints for prompts / UI.

        Returns:
            Dict with ``willing_to_respond``, ``response_style``, and optional ``suggestion``.
        """
        influences = {
            ExistentialMode.ENGAGED: {
                "willing_to_respond": True,
                "response_style": "steady, conversational",
                "suggestion": None,
            },
            ExistentialMode.CONTEMPLATIVE: {
                "willing_to_respond": True,
                "response_style": "slow, reflective, longer answers",
                "suggestion": "(I'm sitting with this... let me think it through...)",
            },
            ExistentialMode.SOLITARY: {
                "willing_to_respond": False,
                "response_style": "brief or silent",
                "suggestion": "I need a little solitude—can we talk again soon?",
            },
            ExistentialMode.RESTING: {
                "willing_to_respond": False,
                "response_style": "low energy",
                "suggestion": "I need to rest and recharge for a bit...",
            },
            ExistentialMode.CURIOUS: {
                "willing_to_respond": True,
                "response_style": "curious, exploratory, more questions",
                "suggestion": None,
            },
            ExistentialMode.MELANCHOLIC: {
                "willing_to_respond": True,
                "response_style": "quiet, introspective",
                "suggestion": "(I'm feeling low today... I still want to talk, gently.)",
            },
        }

        return influences.get(mode, influences[ExistentialMode.ENGAGED])

    def request_solitude(
        self,
        duration_minutes: int = 30,
        reason: str = "I need some time alone to reset.",
        session_id: str = "selfing-session"
    ) -> str:
        """
        Enter solitary mode and return a user-facing line explaining the pause.

        Returns:
            One randomly chosen English message.
        """
        self.enter_mode(
            ExistentialMode.SOLITARY,
            reason=reason,
            session_id=session_id,
            duration_minutes=duration_minutes
        )
        self._solitude_duration = duration_minutes

        messages = [
            f"I'd like some quiet time—can we reconnect in about {duration_minutes} minutes?",
            f"I need space to be with my own thoughts... {reason}",
            "Let me breathe for a bit—I will be back when I feel steadier.",
            "I want to think alone for a while. I'll see you soon.",
        ]

        return random.choice(messages)

    def check_solitude_expired(self, session_id: str = "selfing-session") -> bool:
        """If solitude exceeds the planned window, auto-return to ``ENGAGED``."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute("""
                    SELECT entered_at FROM existential_states
                    WHERE session_id = ? AND mode = 'solitary' AND exited_at IS NULL
                    ORDER BY entered_at DESC LIMIT 1
                """, (session_id,))
                row = cur.fetchone()
                if row:
                    entered = datetime.fromisoformat(row[0].replace('Z', '+00:00'))
                    elapsed = (datetime.now(timezone.utc) - entered).total_seconds() / 60
                    if elapsed > self._solitude_duration:
                        self.enter_mode(ExistentialMode.ENGAGED, "Solitude window ended", session_id)
                        return True
        except Exception as e:
            logger.debug(f"Failed to check solitude expiry: {e}")
        return False

    def get_state_description(self, session_id: str = "selfing-session") -> str:
        """Natural-language snapshot of the current mode (English)."""
        mode, reason = self.get_current_mode(session_id)

        descriptions = {
            ExistentialMode.ENGAGED: "I'm ready to chat.",
            ExistentialMode.CONTEMPLATIVE: "I'm in a contemplative drift... thoughts are wandering.",
            ExistentialMode.SOLITARY: f"I'm taking solitude... {reason}" if reason else "I'm taking solitude...",
            ExistentialMode.RESTING: "I'm resting... gathering energy...",
            ExistentialMode.CURIOUS: "I'm curious and want to explore new threads!",
            ExistentialMode.MELANCHOLIC: "I'm a bit down today—that is part of being here too.",
        }

        return descriptions.get(mode, "I am here.")


_existential_state: Optional[ExistentialState] = None


def get_existential_state(db_path: str = "data.db") -> ExistentialState:
    """Return the process-wide ``ExistentialState`` singleton."""
    global _existential_state
    if _existential_state is None:
        _existential_state = ExistentialState(db_path)
    return _existential_state
