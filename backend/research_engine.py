#!/usr/bin/env python3
"""
Research engine for S.

Responsibilities:
1. Research lifecycle (start / pause / resume).
2. Guardrails (energy, cooldown, viscosity—not a hard daily cap).
3. Background research bookkeeping.
4. Persist research notes / outcomes.

Design:
- Research is not infinite; rest is required.
- The user can pause/resume via dialogue.
- Energy- and viscosity-driven, aligned with the somatic/homeostasis layer.
"""

import sqlite3
import json
import logging
import time
import random
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

logger = logging.getLogger(__name__)


class ResearchStatus(Enum):
    IDLE = "idle"
    RESEARCHING = "researching"
    PAUSED = "paused"  # user paused
    RESTING = "resting"  # forced rest (low energy)
    COOLDOWN = "cooldown"


class ResearchEngine:
    """Coordinates background research pacing and persistence."""

    # Defaults — no hard daily quota; energy + viscosity gate continuation.
    DEFAULT_CONFIG = {
        "enabled": True,
        "min_energy": 50,           # minimum energy to start research
        "energy_cost": 5,           # energy spent per research episode
        "cooldown_minutes": 10,
        "force_rest_energy": 30,  # below this, force rest
        "viscosity_increase": 0.02,  # viscosity bump per episode
        "research_probability": 0.4,
        "max_viscosity": 0.7,       # too "sticky" => pause research
    }
    
    def __init__(self, db_path: str = "data.db", config: Optional[Dict] = None):
        self.db_path = db_path
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self._ensure_tables()
        
    def _ensure_tables(self):
        """Create SQLite tables when missing."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS research_state (
                        session_id TEXT PRIMARY KEY,
                        status TEXT DEFAULT 'idle',
                        paused_by_user INTEGER DEFAULT 0,
                        research_count_today INTEGER DEFAULT 0,
                        last_research_time TEXT,
                        last_reset_date TEXT,
                        total_research_count INTEGER DEFAULT 0,
                        current_goal_id TEXT,
                        notes TEXT,
                        updated_at TEXT
                    )
                """)
                
                # Migrations: autonomous rhythm columns
                try:
                    conn.execute("ALTER TABLE research_state ADD COLUMN next_wake_time TEXT")
                except Exception:
                    pass  # column exists
                try:
                    conn.execute(
                        "ALTER TABLE research_state ADD COLUMN s_rest_request INTEGER DEFAULT 0"
                    )  # requested rest (seconds)
                except Exception:
                    pass
                try:
                    conn.execute(
                        "ALTER TABLE research_state ADD COLUMN last_activity_time TEXT"
                    )  # last activity timestamp
                except Exception:
                    pass

                # Research log table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS research_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        goal_id TEXT,
                        goal_title TEXT,
                        research_note TEXT,
                        energy_before REAL,
                        energy_after REAL,
                        duration_seconds INTEGER,
                        created_at TEXT NOT NULL
                    )
                """)
                
                conn.commit()
                logger.info("Research engine tables ensured")
        except Exception as e:
            logger.error(f"Failed to ensure research tables: {e}")
    
    def _get_state(self, session_id: str) -> Dict:
        """Load or initialize research_state row."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.execute(
                    "SELECT * FROM research_state WHERE session_id = ?",
                    (session_id,)
                )
                row = cur.fetchone()
                
                if row:
                    state = dict(row)
                    # Reset per-day counter when the calendar day changes
                    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                    if state.get("last_reset_date") != today:
                        conn.execute("""
                            UPDATE research_state 
                            SET research_count_today = 0, last_reset_date = ?
                            WHERE session_id = ?
                        """, (today, session_id))
                        conn.commit()
                        state["research_count_today"] = 0
                        state["last_reset_date"] = today
                    return state
                else:
                    # Bootstrap row
                    now = datetime.now(timezone.utc).isoformat()
                    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                    conn.execute("""
                        INSERT INTO research_state 
                        (session_id, status, paused_by_user, research_count_today,
                         last_reset_date, total_research_count, updated_at)
                        VALUES (?, ?, 0, 0, ?, 0, ?)
                    """, (session_id, ResearchStatus.IDLE.value, today, now))
                    conn.commit()
                    return {
                        "session_id": session_id,
                        "status": ResearchStatus.IDLE.value,
                        "paused_by_user": 0,
                        "research_count_today": 0,
                        "last_research_time": None,
                        "last_reset_date": today,
                        "total_research_count": 0,
                        "current_goal_id": None,
                        "notes": None,
                    }
        except Exception as e:
            logger.error(f"Failed to get research state: {e}")
            return {}
    
    def _update_state(self, session_id: str, **kwargs):
        """Patch fields on research_state."""
        try:
            now = datetime.now(timezone.utc).isoformat()

            # Build dynamic UPDATE
            fields = list(kwargs.keys()) + ["updated_at"]
            values = list(kwargs.values()) + [now]
            
            set_clause = ", ".join([f"{f} = ?" for f in fields])
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(f"""
                    UPDATE research_state 
                    SET {set_clause}
                    WHERE session_id = ?
                """, values + [session_id])
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to update research state: {e}")
    
    # ==================== can_research ====================

    def can_research(self, session_id: str, energy: float, viscosity: float = 0.0) -> Tuple[bool, str]:
        """
        Whether another research episode may start.

        Gates:
        1. User pause (highest priority)
        2. Forced-rest energy floor
        3. Minimum energy for research
        4. Viscosity ceiling ("cognitive drag")
        5. Cooldown since last episode

        There is **no** hard daily attempt cap—pacing is physiological.

        Returns:
            (can_research, reason)
        """
        if not self.config.get("enabled", True):
            return False, "Research is disabled"

        state = self._get_state(session_id)

        # 1) User pause
        if state.get("paused_by_user"):
            return False, "User asked to pause research"

        # 2) Forced rest band
        force_rest = self.config.get("force_rest_energy", 30)
        if energy < force_rest:
            return False, f"Energy critically low ({energy:.1f}%); rest required"

        # 3) Minimum energy gate
        min_energy = self.config.get("min_energy", 50)
        if energy < min_energy:
            return False, f"Insufficient energy ({energy:.1f}% < {min_energy}%); rest first"

        # 4) Viscosity gate
        max_viscosity = self.config.get("max_viscosity", 0.7)
        if viscosity > max_viscosity:
            return False, f"Cognitive drag too high ({viscosity:.2f} > {max_viscosity}); ease off"

        # 5) Cooldown
        last_time = state.get("last_research_time")
        if last_time:
            try:
                last_dt = datetime.fromisoformat(last_time.replace("Z", "+00:00"))
                now = datetime.now(timezone.utc)
                cooldown_minutes = self.config.get("cooldown_minutes", 10)
                if (now - last_dt).total_seconds() < cooldown_minutes * 60:
                    remaining = cooldown_minutes - (now - last_dt).total_seconds() / 60
                    return False, f"Cooldown active (~{remaining:.1f} minutes left)"
            except Exception:
                pass

        return True, "Research allowed"

    # ==================== user controls ====================

    def pause_research(self, session_id: str, reason: str = "") -> Dict:
        """Pause research (user-initiated)."""
        self._update_state(
            session_id,
            status=ResearchStatus.PAUSED.value,
            paused_by_user=1,
            notes=f"User pause: {reason}" if reason else "User pause",
        )
        
        state = self._get_state(session_id)
        logger.info(f"Research paused for {session_id}")
        
        return {
            "success": True,
            "message": "Research paused",
            "status": self.get_status_report(session_id)
        }
    
    def resume_research(self, session_id: str) -> Dict:
        """Resume research (user-initiated)."""
        self._update_state(
            session_id,
            status=ResearchStatus.IDLE.value,
            paused_by_user=0,
            notes="User resumed research",
        )
        
        logger.info(f"Research resumed for {session_id}")
        
        return {
            "success": True,
            "message": "Research resumed",
            "status": self.get_status_report(session_id)
        }
    
    def get_status_report(self, session_id: str, energy: float = None, viscosity: float = 0.0) -> Dict:
        """Human-readable status blob for tools / UI."""
        state = self._get_state(session_id)

        # Next time research is allowed (cooldown)
        next_available = "now"
        last_time = state.get("last_research_time")
        if last_time:
            try:
                last_dt = datetime.fromisoformat(last_time.replace("Z", "+00:00"))
                now = datetime.now(timezone.utc)
                cooldown_minutes = self.config.get("cooldown_minutes", 10)
                cooldown_end = last_dt + timedelta(minutes=cooldown_minutes)
                if now < cooldown_end:
                    remaining = (cooldown_end - now).total_seconds() / 60
                    next_available = f"in ~{remaining:.1f} minutes"
            except Exception:
                pass

        # Status line (energy + viscosity aware)
        status = state.get("status", "idle")
        if state.get("paused_by_user"):
            status_desc = "⏸️ Paused (awaiting user)"
        elif energy is not None and energy < 30:
            status_desc = "😴 Resting (recovering energy)"
        elif viscosity > 0.7:
            status_desc = "🧠 Cognitive fatigue (needs easing)"
        elif status == "cooldown":
            status_desc = "⏳ Cooldown"
        elif status == "researching":
            status_desc = "🔬 Researching"
        elif energy is not None and energy < 50:
            status_desc = "😌 Idle (below research energy threshold)"
        else:
            status_desc = "💤 Idle (research available)"

        can_research = True
        cannot_reason = ""
        if state.get("paused_by_user"):
            can_research = False
            cannot_reason = "User paused"
        elif energy is not None and energy < 50:
            can_research = False
            cannot_reason = "Insufficient energy"
        elif viscosity > 0.7:
            can_research = False
            cannot_reason = "Cognitive fatigue"
        
        return {
            "status": status,
            "status_desc": status_desc,
            "paused_by_user": bool(state.get("paused_by_user")),
            "research_count_today": state.get("research_count_today", 0),
            "total_research_count": state.get("total_research_count", 0),
            "next_available": next_available,
            "current_goal_id": state.get("current_goal_id"),
            "energy": energy,
            "viscosity": viscosity,
            "can_research": can_research,
            "cannot_reason": cannot_reason,
            "note": "No hard daily cap—pace research from your own energy and drag signals.",
        }
    
    # ==================== research execution ====================
    
    def start_research(self, session_id: str, goal_id: str = None) -> bool:
        """Mark a research episode as active."""
        self._update_state(
            session_id,
            status=ResearchStatus.RESEARCHING.value,
            current_goal_id=goal_id
        )
        return True
    
    def complete_research(
        self, 
        session_id: str, 
        goal_id: str,
        goal_title: str,
        research_note: str,
        energy_before: float,
        energy_after: float,
        duration_seconds: int = 0
    ) -> Dict:
        """Finalize a research episode and append a log row."""
        try:
            now = datetime.now(timezone.utc).isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                # Append log
                conn.execute("""
                    INSERT INTO research_logs 
                    (session_id, goal_id, goal_title, research_note,
                     energy_before, energy_after, duration_seconds, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    session_id, goal_id, goal_title, research_note,
                    energy_before, energy_after, duration_seconds, now
                ))
                
                # Reset session row
                conn.execute("""
                    UPDATE research_state 
                    SET status = ?,
                        research_count_today = research_count_today + 1,
                        total_research_count = total_research_count + 1,
                        last_research_time = ?,
                        current_goal_id = NULL,
                        updated_at = ?
                    WHERE session_id = ?
                """, (ResearchStatus.IDLE.value, now, now, session_id))
                
                conn.commit()
            
            logger.info(f"Research completed for {session_id}: {goal_title}")
            
            return {
                "success": True,
                "message": f"Research complete: {goal_title}",
                "note_length": len(research_note),
                "energy_cost": energy_before - energy_after
            }
            
        except Exception as e:
            logger.error(f"Failed to complete research: {e}")
            return {"success": False, "error": str(e)}
    
    def get_recent_research_notes(self, session_id: str, limit: int = 5) -> List[Dict]:
        """Recent ``research_logs`` rows."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.execute("""
                    SELECT * FROM research_logs
                    WHERE session_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (session_id, limit))
                
                return [dict(row) for row in cur.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get research notes: {e}")
            return []
    
    # ==================== tool definitions ====================
    
    def get_tool_definitions(self) -> List[Dict]:
        """OpenAI-style tool definitions for research controls."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "research_pause",
                    "description": (
                        "Pause background research. Call when the user says things like "
                        "'stop researching', 'take a break', or 'no more research for now' "
                        "(Chinese equivalents also apply)."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reason": {
                                "type": "string",
                                "description": "Why research is pausing",
                            }
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "research_resume",
                    "description": (
                        "Resume background research. Call when the user says "
                        "'resume research', 'continue researching', or 'start researching again'."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "research_status",
                    "description": (
                        "Show research status. Call when the user asks about research progress, "
                        "cooldowns, or whether research is paused."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "set_my_rhythm",
                    "description": (
                        "Set my preferred work/rest cadence for research. Call when the agent "
                        "wants to request a bounded rest window before the next episode."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "rest_minutes": {
                                "type": "integer",
                                "description": "Requested rest duration in minutes (1-30)",
                            },
                            "reason": {
                                "type": "string",
                                "description": "Why this cadence is requested",
                            }
                        },
                        "required": ["rest_minutes"]
                    }
                }
            }
        ]
    
    def route_tool_call(self, tool_name: str, args: Dict, session_id: str, energy: float = 100, viscosity: float = 0.0) -> Dict:
        """Dispatch tool calls from the LLM/runtime."""
        if tool_name == "research_pause":
            return self.pause_research(session_id, args.get("reason", ""))
        elif tool_name == "research_resume":
            return self.resume_research(session_id)
        elif tool_name == "research_status":
            return {
                "success": True,
                "status": self.get_status_report(session_id, energy, viscosity)
            }
        elif tool_name == "set_my_rhythm":
            rest_minutes = args.get("rest_minutes", 5)
            rest_minutes = max(1, min(30, rest_minutes))  # clamp 1–30 minutes
            reason = args.get("reason", "")
            return self.set_rest_request(session_id, rest_minutes * 60, reason)
        else:
            return {"success": False, "error": f"Unknown tool: {tool_name}"}
    
    # ==================== autonomous rhythm ====================
    
    def set_rest_request(self, session_id: str, rest_seconds: int, reason: str = "") -> Dict:
        """
        Persist S's voluntary rest request.

        After research, the agent may say it wants to rest for N minutes; honor that window here.
        """
        try:
            now = datetime.now(timezone.utc)
            next_wake = now + timedelta(seconds=rest_seconds)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE research_state 
                    SET s_rest_request = ?,
                        next_wake_time = ?,
                        last_activity_time = ?,
                        notes = ?,
                        updated_at = ?
                    WHERE session_id = ?
                """, (
                    rest_seconds,
                    next_wake.isoformat(),
                    now.isoformat(),
                    reason or f"S requested rest for {rest_seconds}s",
                    now.isoformat(),
                    session_id
                ))
                conn.commit()
            
            logger.info(f"S requested rest: {rest_seconds}s ({rest_seconds/60:.1f}min) for {session_id}")
            
            return {
                "success": True,
                "rest_seconds": rest_seconds,
                "next_wake_time": next_wake.isoformat(),
                "message": f"Acknowledged—resting for ~{rest_seconds/60:.1f} minutes",
            }
        except Exception as e:
            logger.error(f"Failed to set rest request: {e}")
            return {"success": False, "error": str(e)}
    
    def get_time_perception(self, session_id: str) -> Dict:
        """
        Lightweight time-since-activity + rest window bookkeeping.

        Returns seconds since last activity, pending rest request, and whether the wake window elapsed.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.execute("""
                    SELECT last_activity_time, s_rest_request, next_wake_time, last_research_time
                    FROM research_state WHERE session_id = ?
                """, (session_id,))
                row = cur.fetchone()
                
                if not row:
                    return {
                        "time_since_last_activity": 0,
                        "rest_completed": True,
                        "should_wake": True
                    }
                
                now = datetime.now(timezone.utc)
                
                # Seconds since last activity
                last_activity = row["last_activity_time"] or row["last_research_time"]
                if last_activity:
                    try:
                        last_dt = datetime.fromisoformat(last_activity.replace("Z", "+00:00"))
                        time_since = (now - last_dt).total_seconds()
                    except Exception:
                        time_since = 0
                else:
                    time_since = 0
                
                # Rest window still active?
                rest_completed = True
                should_wake = True
                next_wake = row["next_wake_time"]
                if next_wake:
                    try:
                        wake_dt = datetime.fromisoformat(next_wake.replace("Z", "+00:00"))
                        if now < wake_dt:
                            rest_completed = False
                            should_wake = False
                    except Exception:
                        pass
                
                return {
                    "time_since_last_activity": time_since,
                    "time_since_last_activity_minutes": time_since / 60,
                    "s_rest_request": row["s_rest_request"] or 0,
                    "rest_completed": rest_completed,
                    "should_wake": should_wake,
                    "next_wake_time": next_wake
                }
                
        except Exception as e:
            logger.error(f"Failed to get time perception: {e}")
            return {"time_since_last_activity": 0, "rest_completed": True, "should_wake": True}
    
    def record_activity(self, session_id: str):
        """Touch ``last_activity_time`` for rhythm tracking."""
        try:
            now = datetime.now(timezone.utc).isoformat()
            with sqlite3.connect(self.db_path) as conn:
                # Ensure row exists
                conn.execute("""
                    INSERT OR IGNORE INTO research_state (session_id, updated_at, last_reset_date)
                    VALUES (?, ?, ?)
                """, (session_id, now, datetime.now(timezone.utc).strftime("%Y-%m-%d")))
                
                # Bump activity timestamp
                conn.execute("""
                    UPDATE research_state 
                    SET last_activity_time = ?, updated_at = ?
                    WHERE session_id = ?
                """, (now, now, session_id))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to record activity: {e}")


# Process-wide singleton
_engine_instance = None

def get_research_engine(db_path: str = "data.db", config: Optional[Dict] = None) -> ResearchEngine:
    """Return the shared ``ResearchEngine`` instance."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = ResearchEngine(db_path, config)
    return _engine_instance

