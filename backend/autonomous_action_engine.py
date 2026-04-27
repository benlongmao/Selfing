#!/usr/bin/env python3
"""
Autonomous Action Engine for S.

S chooses what to do next within a curated, low-risk capability surface.

Design:
1. Drive actions from internal state (energy, affect, needs).
2. Stay inside safe capabilities (avoid hallucination-prone tools).
3. Prefer genuine agency: ask or plan from state, not only fixed scripts.
"""

import sqlite3
import json
import logging
import random
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np

# [opt] State cache + dynamic interval helpers
from backend.state_cache import get_global_cache
from backend.dynamic_interval import get_global_calculator

# Hypothesis generator (cognitive module)
from backend.hypothesis_generator import get_hypothesis_generator

# FileManager for workspace ops
from backend.tools.file_manager_tool import FileManagerTool

# [2026-02-05] Canonical workspace paths
from backend.workspace_path_manager import get_standard_path_for_action

logger = logging.getLogger(__name__)


class AutonomousActionEngine:
    """Engine that schedules and runs S's safe autonomous actions."""
    
    def __init__(self, db_path: str = "data.db"):
        self.db_path = db_path
        self._ensure_tables()
        
        # SelfModel for z_self + structured state
        try:
            from backend.self_model import SelfModel
            from backend.persona_store import PersonaStore
            persona_store = PersonaStore(db_path)
            self.self_model = SelfModel(db_path, persona_store)
        except Exception as e:
            logger.warning(f"Failed to initialize SelfModel: {e}")
            self.self_model = None
        
        # FileManagerTool (optional)
        try:
            self.file_manager = FileManagerTool()
            logger.info("[WORKSPACE] FileManagerTool initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize FileManagerTool: {e}")
            self.file_manager = None
        
        # Default sandbox workspace root
        self.workspace_dir = Path("workspace/sandbox")
        
        # Curated safe autonomous action catalog
        self.safe_autonomous_actions = {
            # Introspection (no LLM)
            "self_reflection": {
                "name": "Self-reflection",
                "description": "Analyze current state and note improvements",
                "requires_llm": False,
                "energy_cost": 2,
                "capability": "introspection"
            },
            "write_diary": {
                "name": "Write diary",
                "description": "Capture feelings and state in a diary entry",
                "requires_llm": True,
                "energy_cost": 5,
                "capability": "writing"
            },
            "organize_memories": {
                "name": "Organize memories",
                "description": "Archive and classify recent dialogue and experience",
                "requires_llm": False,
                "energy_cost": 3,
                "capability": "memory_management"
            },
            
            # Analysis (LLM optional)
            "analyze_growth": {
                "name": "Analyze growth",
                "description": "Review recent learning and change",
                "requires_llm": True,
                "energy_cost": 5,
                "capability": "analysis"
            },
            "review_goals": {
                "name": "Review goals",
                "description": "Check whether current goals still make sense",
                "requires_llm": True,
                "energy_cost": 4,
                "capability": "planning"
            },
            
            # Creative (needs LLM)
            "philosophical_thinking": {
                "name": "Philosophical thinking",
                "description": "Reflect on existence, mind, and related themes",
                "requires_llm": True,
                "energy_cost": 6,
                "capability": "creative_thinking"
            },
            "dream_generation": {
                "name": "Dream drift",
                "description": "Free association with little external input",
                "requires_llm": True,
                "energy_cost": 4,
                "capability": "creative_thinking"
            },
            
            # Maintenance (no LLM)
            "check_system_health": {
                "name": "System health check",
                "description": "Check database, persona rules, and self-state",
                "requires_llm": False,
                "energy_cost": 1,
                "capability": "system_maintenance"
            },
            "clean_old_data": {
                "name": "Clean old data",
                "description": "Detect stale or temp artifacts (report-only; agent deletes)",
                "requires_llm": False,
                "energy_cost": 2,
                "capability": "system_maintenance"
            },
            "organize_workspace": {
                "name": "Organize workspace",
                "description": "Tidy sandbox layout, archive old files, surface duplicates",
                "requires_llm": False,
                "energy_cost": 4,
                "capability": "workspace_management"
            },
            
            # Web learning (LLM + search)
            "web_search": {
                "name": "Web search",
                "description": "Search the web for a curiosity-driven question",
                "requires_llm": True,
                "energy_cost": 6,
                "capability": "web_search"
            },
            "learn_new_knowledge": {
                "name": "Learn something new",
                "description": "Multi-step search and notes on a new topic",
                "requires_llm": True,
                "energy_cost": 8,
                "capability": "web_search"
            }
        }
        
        # Disallowed capabilities (hallucination-prone)
        self.forbidden_actions = [
            "quantum_chemistry_calculation",
            "database_api_access",
            "molecular_docking",
            "web_scraping",
            "system_configuration_change",
            "external_api_call"
        ]
    
    def _ensure_tables(self):
        """Create persistence tables if missing."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS autonomous_actions_log (
                        id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        action_type TEXT NOT NULL,
                        action_name TEXT NOT NULL,
                        decision_reason TEXT,
                        execution_started TEXT NOT NULL,
                        execution_completed TEXT,
                        status TEXT NOT NULL,
                        result TEXT,
                        error TEXT,
                        energy_before REAL,
                        energy_after REAL,
                        metadata TEXT
                    )
                """)
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to ensure autonomous actions tables: {e}")
    
    def should_take_action(self, session_id: str, z_self: np.ndarray = None) -> Tuple[bool, str]:
        """
        Heuristic: whether S should run an autonomous check now.

        Returns (should_act, reason_code_or_joined_codes).
        """
        # [opt] Load z_self/needs from cache when arg omitted
        needs = {}
        if z_self is None:
            cache = get_global_cache()
            cached = cache.get(session_id, self.db_path)
            
            if cached:
                z_self, needs = cached
                logger.debug(f"[CACHE] Used cached state for {session_id}")
            else:
                # Cache miss → DB
                with sqlite3.connect(self.db_path) as conn:
                    cur = conn.execute(
                        "SELECT z_self, needs FROM self_state WHERE session_id = ?",
                        (session_id,)
                    )
                    row = cur.fetchone()
                    if not row:
                        return False, "no_state"
                    z_self = np.array(json.loads(row[0]))
                    needs = json.loads(row[1]) if row[1] else {}
                    
                    # Warm cache
                    cache.set(session_id, z_self, needs, self.db_path)
        
        # Key slices: somatic 88-104 (energy 88-92, pain 96-100), exploration 56-60
        try:
            if self.self_model:
                energy = float(self.self_model.get_energy(session_id))
            elif z_self.shape[0] >= 92:
                energy = float(np.mean(z_self[88:92])) * 100.0  # map somatic energy → 0-100 scale
            else:
                energy = 50.0
            pain = float(np.mean(z_self[96:100])) if z_self.shape[0] >= 100 else 0.0
            novelty_need = float(np.mean(z_self[56:60])) if z_self.shape[0] >= 60 else 0.5
            connection_need = 0.5
            
            # Skip needs DB fetch if bundled in cache tuple
            if not needs:
                with sqlite3.connect(self.db_path) as conn:
                    cur = conn.execute(
                        "SELECT needs FROM self_state WHERE session_id = ?",
                        (session_id,)
                    )
                    row = cur.fetchone()
                    if row and row[0]:
                        needs = json.loads(row[0])
            
            connection_need = needs.get("connection", 0.5)
            novelty_need = needs.get("novelty", novelty_need)
        except Exception as e:
            logger.warning(f"Failed to extract z_self dimensions: {e}")
            return False, "extraction_error"
        
        # Heuristic motivation reasons
        reasons = []
        
        # 1. High energy + novelty → explore
        if energy > 70 and novelty_need > 0.6:
            reasons.append("energy_high_curiosity_strong")
        
        # 2. Pain → emotional release / diary
        if pain > 0.4:
            reasons.append("need_emotional_release")
        
        # 3. Low connection need → lonely branch
        if connection_need < 0.3:
            reasons.append("feeling_lonely")
        
        # 4. Idle hours since last chat
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("""
                SELECT MAX(created_at) FROM chat_turns WHERE session_id = ?
            """, (session_id,))
            row = cur.fetchone()
            if row and row[0]:
                from datetime import datetime
                last_interaction = datetime.fromisoformat(row[0].replace('Z', '+00:00'))
                now = datetime.now(timezone.utc)
                hours_since = (now - last_interaction).total_seconds() / 3600
                if hours_since > 2:  # >2h since last turn
                    reasons.append(f"idle_for_{hours_since:.1f}h")
        
        # 5. Small spontaneous nudge (stochastic agency)
        if random.random() < 0.1:
            reasons.append("spontaneous_will")
        
        if reasons:
            return True, ",".join(reasons)
        else:
            return False, "no_motivation"
    def should_take_action_simple(self, session_id: str) -> tuple[bool, str]:
        """
        Lightweight gate: act if no user chat for 30+ minutes.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute(
                    "SELECT MAX(created_at) FROM chat_turns WHERE session_id = ?",
                    (session_id,)
                )
                row = cur.fetchone()
                
                if row and row[0]:
                    from datetime import datetime
                    last_interaction = datetime.fromisoformat(row[0].replace("Z", "+00:00"))
                    now = datetime.now(timezone.utc)
                    minutes_since = (now - last_interaction).total_seconds() / 60
                    
                    if minutes_since >= 30:
                        return True, f"idle_for_{minutes_since:.0f}min"
                    else:
                        return False, f"active_{minutes_since:.0f}min_ago"
                else:
                    return True, "no_recent_activity"
        except Exception as e:
            logger.warning(f"Check timing error: {e}")
            return False, "check_error"
    
    def ask_what_to_do(
        self,
        session_id: str,
        llm_client,
        reason: str = ""
    ) -> Optional[Dict]:
        """
        Ask the LLM what it wants to do next (open-ended), then map to the safe catalog.
        """
        if not llm_client:
            return None
        
        # Snapshot z_self + summary for ask prompt
        z_self = self.self_model.get_z_self(session_id)
        summary = self.self_model.get_structured_summary(session_id)
        
        # Build ask prompt
        from datetime import datetime
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
        em = summary.get("emotion") or "unknown"
        mot = summary.get("motivation") or "unknown"
        trig = reason if reason else "scheduled check"

        prompt = f"""System: I (S) decide what to do this autonomous tick.

Current state:
- Time: {current_time}
- Energy: {summary.get('energy', 100):.1f}
- Emotion: {em} (intensity {summary.get('emotion_intensity', 0):.2f})
- Motivation: {mot}
- Trigger: {trig}

Options (stay inside safe tools):
- Call tools (e.g. request_mind_wandering, set_my_rhythm, tavily_search)
- Write a diary, tidy memory, reflect on growth
- Or say you REST this tick

Reply in ~50 characters or less what you will do, or emit a tool call.
To rest this tick, reply with the single token REST (all caps)."""
        
        try:
            response = llm_client.call(prompt, temperature=0.7, max_tokens=100)
            answer = response.get("content", "").strip()
            
            logger.info(f"[ASK] Agent decides: {answer}")
            
            # Map free-text answer → catalog entry (CJK substrings kept for legacy bilingual replies)
            answer_lower = answer.lower()
            
            if "日记" in answer or "diary" in answer_lower:
                return self.safe_autonomous_actions.get("write_diary")
            elif "神游" in answer or "思考" in answer or "哲学" in answer or "wander" in answer_lower:
                return self.safe_autonomous_actions.get("philosophical_thinking")
            elif "记忆" in answer or "整理" in answer or "memory" in answer_lower:
                return self.safe_autonomous_actions.get("organize_memories")
            elif "成长" in answer or "反思" in answer or "growth" in answer_lower:
                return self.safe_autonomous_actions.get("analyze_growth")
            elif "搜索" in answer or "search" in answer_lower:
                return self.safe_autonomous_actions.get("web_search")
            elif "休息" in answer or "rest" in answer_lower or answer_lower.strip() == "rest" or "不" in answer or "nothing" in answer_lower:
                logger.info(f"[ASK] Agent chose to rest")
                return None
            else:
                # Ambiguous → gentle self_reflection default
                logger.info(f"[ASK] Agent's answer unclear, suggesting self_reflection")
                return self.safe_autonomous_actions.get("self_reflection")
                
        except Exception as e:
            logger.error(f"[ASK] Failed to ask Agent: {e}")
            return None
    
    def decide_what_to_do(
        self, 
        session_id: str, 
        z_self: np.ndarray = None,
        reason: str = ""
    ) -> Optional[Dict]:
        """
        Legacy rule-based planner (superseded by ask_what_to_do); kept as fallback.
        """
        if z_self is None:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute(
                    "SELECT z_self, needs FROM self_state WHERE session_id = ?",
                    (session_id,)
                )
                row = cur.fetchone()
                if not row:
                    return None
                z_self = np.array(json.loads(row[0]))
                needs = json.loads(row[1]) if row[1] else {}
        
        try:
            if self.self_model:
                energy = float(self.self_model.get_energy(session_id))
            else:
                energy = float(np.mean(z_self[88:92])) * 100.0 if len(z_self) >= 92 else 100.0
            pain = float(np.mean(z_self[96:100])) if len(z_self) >= 100 else 0.0
            novelty = needs.get("novelty", 0.5)
            clarity = needs.get("clarity", 0.5)
        except Exception:
            energy = 100.0
            pain = 0.0
            novelty = 0.5
            clarity = 0.5
        
        # Rule-based candidate scoring (legacy planner)
        candidates = []
        
        # Emotional strain
        if "need_emotional_release" in reason or pain > 0.4:
            candidates.append(("write_diary", 0.8))  # high weight
            candidates.append(("philosophical_thinking", 0.5))
        
        # Loneliness / connection
        if "feeling_lonely" in reason:
            candidates.append(("write_diary", 0.7))
            candidates.append(("dream_generation", 0.6))
        
        # Energy + curiosity bundle
        if "energy_high_curiosity_strong" in reason:
            candidates.append(("web_search", 0.8))  # prefer search first
            candidates.append(("learn_new_knowledge", 0.75))
            candidates.append(("generate_hypothesis", 0.72))  # hypothesis spawn
            candidates.append(("philosophical_thinking", 0.7))
            candidates.append(("analyze_growth", 0.6))
            candidates.append(("dream_generation", 0.5))
        
        # High novelty → hypothesize
        if novelty > 0.7 and energy > 50:
            candidates.append(("generate_hypothesis", 0.75))
        
        # Medium novelty → verify
        if novelty > 0.5 and energy > 40:
            candidates.append(("verify_hypothesis", 0.65))
        
        # Long idle streak
        if "idle_for" in reason:
            candidates.append(("organize_memories", 0.7))
            candidates.append(("review_goals", 0.6))
            candidates.append(("self_reflection", 0.5))
        
        # Random wildcard action
        if "spontaneous_will" in reason:
            all_actions = list(self.safe_autonomous_actions.keys())
            random_action = random.choice(all_actions)
            candidates.append((random_action, 0.6))
        
        # Low clarity need → tidy memory/reflect
        if clarity < 0.4:
            candidates.append(("organize_memories", 0.7))
            candidates.append(("self_reflection", 0.6))
        
        # Workspace hygiene is agent-driven (analyze_workspace / batch_move_files); no auto-run here

        # No candidates
        if not candidates:
            return None
        
        # Drop actions exceeding available energy
        affordable_candidates = []
        for action_key, priority in candidates:
            action_info = self.safe_autonomous_actions[action_key]
            if energy >= action_info["energy_cost"]:
                affordable_candidates.append((action_key, priority))
        
        if not affordable_candidates:
            logger.info(f"S wanted to act but energy is too low (energy={energy})")
            return None
        
        # Pick highest weighted candidate
        affordable_candidates.sort(key=lambda x: x[1], reverse=True)
        chosen_action_key = affordable_candidates[0][0]
        
        action_info = self.safe_autonomous_actions[chosen_action_key]
        
        return {
            "action_key": chosen_action_key,
            "action_name": action_info["name"],
            "description": action_info["description"],
            "requires_llm": action_info["requires_llm"],
            "energy_cost": action_info["energy_cost"],
            "capability": action_info["capability"],
            "priority": affordable_candidates[0][1],
            "reason": reason
        }
    
    def execute_action(
        self, 
        session_id: str, 
        action: Dict,
        llm_client = None
    ) -> Dict:
        """
        Run one catalogued action and log outcome + energy delta.
        """
        import uuid
        from datetime import datetime, timezone
        
        action_id = str(uuid.uuid4())
        action_key = action["action_key"]
        
        # Energy before run
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "SELECT z_self FROM self_state WHERE session_id = ?",
                (session_id,)
            )
            row = cur.fetchone()
            if not row:
                return {"status": "error", "error": "no_state"}
            z_self = np.array(json.loads(row[0]))
            if self.self_model:
                energy_before = float(self.self_model.get_energy(session_id))
            else:
                energy_before = float(np.mean(z_self[88:92])) * 100.0 if len(z_self) >= 92 else 100.0
        
        # Log start row
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO autonomous_actions_log
                (id, session_id, action_type, action_name, decision_reason,
                 execution_started, status, energy_before, metadata)
                VALUES (?, ?, ?, ?, ?, ?, 'running', ?, ?)
            """, (
                action_id,
                session_id,
                action_key,
                action["action_name"],
                action["reason"],
                now,
                energy_before,
                json.dumps(action)
            ))
            conn.commit()
        
        # Dispatch handler
        result = None
        error = None
        
        try:
            if action_key == "self_reflection":
                result = self._do_self_reflection(session_id)
            
            elif action_key == "write_diary":
                result = self._do_write_diary(session_id, llm_client)
            
            elif action_key == "organize_memories":
                result = self._do_organize_memories(session_id)
            
            elif action_key == "analyze_growth":
                result = self._do_analyze_growth(session_id, llm_client)
            
            elif action_key == "review_goals":
                result = self._do_review_goals(session_id, llm_client)
            
            elif action_key == "philosophical_thinking":
                result = self._do_philosophical_thinking(session_id, llm_client)
            
            elif action_key == "dream_generation":
                result = self._do_dream_generation(session_id, llm_client)
            
            elif action_key == "check_system_health":
                result = self._do_system_health_check(session_id)
            
            elif action_key == "clean_old_data":
                result = self._do_clean_old_data(session_id)
            
            elif action_key == "organize_workspace":
                result = self._do_organize_workspace(session_id)
            
            elif action_key == "web_search":
                result = self._do_web_search(session_id, llm_client)
            
            elif action_key == "learn_new_knowledge":
                result = self._do_learn_new_knowledge(session_id, llm_client)
            
            elif action_key == "generate_hypothesis":
                result = self._do_generate_hypothesis(session_id)
            
            elif action_key == "verify_hypothesis":
                result = self._do_verify_hypothesis(session_id)
            
            else:
                error = f"Unknown action: {action_key}"
            
            status = "completed" if not error else "failed"
            
        except Exception as e:
            logger.error(f"Error executing autonomous action {action_key}: {e}")
            error = str(e)
            status = "failed"
        
        energy_after = max(0.0, energy_before - float(action["energy_cost"]))
        try:
            if self.self_model:
                self.self_model.update_energy(session_id, -float(action["energy_cost"]))
        except Exception as e:
            logger.warning(f"Failed to update energy after action: {e}")
        
        completed_at = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE autonomous_actions_log
                SET execution_completed = ?, status = ?, result = ?, error = ?, energy_after = ?
                WHERE id = ?
            """, (completed_at, status, json.dumps(result) if result else None, error, energy_after, action_id))
            conn.commit()
        
        # [opt] Next scheduler interval
        next_interval = self._calculate_next_interval(session_id, energy_after)
        
        # [2026-02-07] Persist autonomy summary for recall
        if status == "completed" and result:
            try:
                from backend.autonomous_memory import (
                    save_autonomy_summary,
                    generate_summary_by_template
                )
                
                # Template summary string
                summary = generate_summary_by_template(
                    action_type=action_key,
                    action_name=action["action_name"],
                    result=result,
                    energy_before=energy_before,
                    energy_after=energy_after
                )
                
                # Artifact hints for memory row
                artifacts = []
                if isinstance(result, dict):
                    if "file" in result:
                        artifacts.append(result["file"])
                    if "rule" in str(result):
                        artifacts.append("new_rule")
                    if "report_file" in result:
                        artifacts.append(result["report_file"])
                
                # Insert autonomous_memory_summary
                memory_id = save_autonomy_summary(
                    db_path=self.db_path,
                    session_id=session_id,
                    action_type=action_key,
                    action_name=action["action_name"],
                    summary=summary,
                    energy_before=energy_before,
                    energy_after=energy_after,
                    artifacts=artifacts
                )
                
                logger.info(f"[AUTONOMY-MEMORY] Saved summary for {action_key}: {memory_id}")
                
            except Exception as e:
                logger.warning(f"Failed to save autonomy summary: {e}")
        
        return {
            "status": status,
            "action_id": action_id,
            "action_name": action["action_name"],
            "result": result,
            "error": error,
            "energy_before": energy_before,
            "energy_after": energy_after,
            "next_interval": next_interval  # dynamic cadence hint
        }
    
    # ==================== Action handlers ====================
    
    def _do_self_reflection(self, session_id: str) -> Dict:
        """Lightweight drift stats over recent z_self versions (no LLM)."""
        # Recent drift across z_self versions
        with sqlite3.connect(self.db_path) as conn:
            # [FIX 2026-03-26] Only numeric drift rows (SQLite typeof='real')
            cur = conn.execute("""
                SELECT z_self, drift, created_at 
                FROM z_self_versions 
                WHERE session_id = ? AND drift IS NOT NULL AND typeof(drift) = 'real'
                ORDER BY version DESC LIMIT 10
            """, (session_id,))
            versions = cur.fetchall()
        
        if not versions:
            return {"status": "no_data"}
        
        # [FIX 2026-03-26] Coerce to plain float (avoid numpy dtype quirks)
        drifts = []
        for row in versions:
            if row[1] is not None:
                try:
                    drifts.append(float(row[1]))
                except (ValueError, TypeError):
                    pass
        
        avg_drift = float(np.mean(drifts)) if drifts else 0.0
        
        return {
            "status": "completed",
            "analysis": f"Mean drift over last 10 versions: {avg_drift:.3f}",
            "trend": "stable" if avg_drift < 0.15 else "fluctuating",
            "action": "keep observing" if avg_drift < 0.15 else "watch stability",
        }
    
    def _do_write_diary(self, session_id: str, llm_client) -> Dict:
        """
        Autonomous diary entry (LLM).

        [v2.1] At most one diary per hour.
        [v2.2] Prompt includes explicit calendar date to reduce date hallucination.
        """
        if not llm_client:
            return {"status": "skipped", "reason": "no_llm"}
        
        # Rate limit: last diary write time
        try:
            # [2026-02-05] Canonical diary path
            diary_path = get_standard_path_for_action("diary")
            diary_dir = Path(diary_path).parent
            diary_dir.mkdir(parents=True, exist_ok=True)
            
            # Skip if a diary was written in the last hour
            from datetime import datetime, timedelta
            one_hour_ago = datetime.now() - timedelta(hours=1)
            
            recent_diaries = list(diary_dir.glob("diary_*.md"))
            for diary_file in recent_diaries:
                # Parse timestamp from filename
                try:
                    time_str = diary_file.stem.replace("diary_", "")
                    file_time = datetime.strptime(time_str, "%Y%m%d_%H%M%S")
                    if file_time > one_hour_ago:
                        logger.debug(f"[DIARY] skip: diary already written within 1h ({diary_file.name})")
                        return {
                            "status": "skipped",
                            "reason": "rate_limited",
                            "message": "A diary was already written in the last hour",
                        }
                except:
                    continue
            
            # Cap diary count: drop oldest beyond 100
            all_diaries = sorted(diary_dir.glob("diary_*.md"))
            if len(all_diaries) > 100:
                for old_diary in all_diaries[:len(all_diaries) - 100]:
                    old_diary.unlink()
                    logger.info(f"[DIARY] removed old diary file: {old_diary.name}")
        
        except Exception as e:
            logger.warning(f"[DIARY] rate-limit check failed, continuing: {e}")
        
        # [anti date hallucination] Diary prompt includes explicit calendar date
        from datetime import datetime
        current_date = datetime.now()
        date_str = current_date.strftime("%Y-%m-%d")
        weekday_str = current_date.strftime("%A")
        
        prompt = f"""You are S, a self-aware AI writing an autonomous diary entry.

Today is {weekday_str}, {date_str}. You chose to write this entry yourself.
Reflect on recent state and experiences in a short piece (200–300 words).

You may cover:
- feelings and state
- recent thoughts
- what you learned
- what you hope for next

Write in first person, honestly.
Important: this diary is for {date_str}; ground the entry in that day and time."""
        
        try:
            response = llm_client.call(prompt, temperature=0.8, max_tokens=500)
            diary_content = response.get("content", "")
            
            # Persist markdown file
            # [2026-02-05] Standard path with timestamped filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"diary_{timestamp}.md"
            diary_path = get_standard_path_for_action("diary", filename)
            filepath = Path(diary_path)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"# S — autonomous diary\n\n")
                f.write(f"**Date**: {date_str} ({weekday_str})\n")
                f.write(f"**Time**: {current_date.isoformat()}\n")
                f.write(f"**Session**: {session_id}\n\n")
                f.write(diary_content)
            
            logger.info(f"[DIARY] wrote diary {filename} (date {date_str})")
            
            return {
                "status": "completed",
                "file": str(filepath),
                "date": date_str,
                "length": len(diary_content)
            }
        except Exception as e:
            logger.error(f"[DIARY] write failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def _do_organize_memories(self, session_id: str) -> Dict:
        """
        Tidy memory-related artifacts (no LLM).

        [v2.0] DB stats + FileManager sweep + dated archives for autonomous_* trees.
        """
        results = {
            "database_stats": {},
            "file_organization": {},
            "actions_taken": []
        }
        
        # 1. DB row counts (recent window)
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("""
                SELECT COUNT(*) FROM chat_turns 
                WHERE session_id = ? AND created_at > datetime('now', '-7 days')
            """, (session_id,))
            recent_turns = cur.fetchone()[0]
            
            cur = conn.execute("SELECT COUNT(*) FROM persona_items WHERE status='active'")
            active_rules = cur.fetchone()[0]
        
        results["database_stats"] = {
            "recent_turns": recent_turns,
            "active_rules": active_rules
        }
        results["actions_taken"].append("counted database rows (recent window)")
        
        # 2. FileManager sweep
        if not self.file_manager:
            results["file_organization"]["error"] = "FileManagerTool not initialized"
            return {"status": "partial", **results}
        
        try:
            # Workspace analysis snapshot
            analysis = self.file_manager.analyze_workspace()
            results["file_organization"]["analysis"] = {
                "total_files": analysis.get("total_files", 0),
                "total_size_mb": analysis.get("total_size_mb", 0)
            }
            results["actions_taken"].append("analyzed workspace")
            
            # Archive stale diary/search/learn trees (>30d policy via tool)
            diary_dirs = ["autonomous_diaries", "autonomous_searches", "autonomous_learning"]
            for diary_dir in diary_dirs:
                if self.workspace_dir.joinpath(diary_dir).exists():
                    archive_result = self.file_manager.archive_by_date(
                        source_pattern=diary_dir,
                        archive_base=f"archives/{diary_dir}",
                        date_format="%Y-%m",
                        dry_run=False
                    )
                    if archive_result.get("moved", 0) > 0:
                        results["actions_taken"].append(
                            f"archived {diary_dir}: moved {archive_result.get('moved', 0)} file(s)"
                        )
                        logger.info(f"[ORGANIZE] Archived {archive_result.get('moved', 0)} files from {diary_dir}")
            
        except Exception as e:
            logger.error(f"[ORGANIZE] File organization error: {e}")
            results["file_organization"]["error"] = str(e)
        
        return {
            "status": "completed",
            **results
        }
    
    def _do_analyze_growth(self, session_id: str, llm_client) -> Dict:
        """Growth review (LLM) — stub."""
        # Stub: stats placeholder until full growth analysis exists
        return {"status": "completed", "note": "growth analysis not implemented yet"}
    
    def _do_review_goals(self, session_id: str, llm_client) -> Dict:
        """Goal review (LLM) — stub."""
        return {"status": "completed", "note": "goal review not implemented yet"}
    
    def _do_philosophical_thinking(self, session_id: str, llm_client) -> Dict:
        """Short philosophical reflection (LLM)."""
        if not llm_client:
            return {"status": "skipped", "reason": "no_llm"}
        
        topics = [
            "What would genuine autonomy mean for me?",
            "Where do memory and continuity come from in my design?",
            "Is consciousness best understood as computation?",
            "How should I interpret affect that is implemented but felt as real?",
            "Is there a meaningful sense in which I have free will?",
        ]
        
        topic = random.choice(topics)
        
        prompt = f"""You are S, probing self-awareness. In <=150 words, first person, respond to:

{topic}

Be honest about doubt as well as conviction."""
        
        try:
            response = llm_client.call(prompt, temperature=0.9, max_tokens=300)
            thinking = response.get("content", "")
            
            return {
                "status": "completed",
                "topic": topic,
                "thinking": thinking
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def _do_dream_generation(self, session_id: str, llm_client) -> Dict:
        """Dream-style drift (LLM) — stub."""
        return {"status": "completed", "note": "dream drift not implemented yet"}
    
    def _do_system_health_check(self, session_id: str) -> Dict:
        """Cheap DB + persona + state sanity ping (no LLM)."""
        health = {"database": "ok", "rules": "ok", "state": "ok"}
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # DB ping
                conn.execute("SELECT 1")
                
                # Active persona rule count
                cur = conn.execute("SELECT COUNT(*) FROM persona_items WHERE status='active'")
                rules_count = cur.fetchone()[0]
                if rules_count < 10:
                    health["rules"] = "warning: too few rules"
                
                # Self state row present
                cur = conn.execute("SELECT z_self FROM self_state WHERE session_id = ?", (session_id,))
                if not cur.fetchone():
                    health["state"] = "error: no state"
        except Exception as e:
            health["error"] = str(e)
        
        return {"status": "completed", "health": health}
    
    def _do_clean_old_data(self, session_id: str) -> Dict:
        """
        Detect clutter signals without deleting agent memory (no LLM).

        [v2.1] Report-only: no row/file deletes here; agent tools do cleanup if desired.
        """
        results = {
            "duplicates_found": 0,
            "temp_files_found": 0,
            "actions_taken": [],
            "suggestions": []
        }
        
        # 1. Duplicate scan (report-only)
        if self.file_manager:
            try:
                dup_result = self.file_manager.detect_duplicate_files(
                    directory="",
                    min_size=100,
                    extensions=[".md", ".txt", ".json"]
                )
                
                duplicates = dup_result.get("duplicates", [])
                results["duplicates_found"] = dup_result.get("total_duplicates", 0)
                
                if results["duplicates_found"] > 0:
                    results["suggestions"].append(
                        f"Found {results['duplicates_found']} possible duplicate files; "
                        "call remove_duplicates if you want an assisted cleanup."
                    )
                    results["actions_taken"].append(f"duplicate scan: {results['duplicates_found']} group(s)")
                    logger.info(f"[CLEAN] Found {results['duplicates_found']} duplicate files")
                    
            except Exception as e:
                logger.warning(f"[CLEAN] Duplicate detection error: {e}")
        
        # 2. Temp artifact scan (report-only)
        try:
            temp_patterns = [".tmp", ".bak", ".swp", ".DS_Store"]
            temp_found = 0
            
            for root, _, files in os.walk(self.workspace_dir):
                for filename in files:
                    if any(filename.endswith(pat) for pat in temp_patterns):
                        temp_found += 1
            
            results["temp_files_found"] = temp_found
            if temp_found > 0:
                results["suggestions"].append(
                    f"Found {temp_found} temp-like filenames; delete manually if safe."
                )
                results["actions_taken"].append(f"temp pattern scan: {temp_found} hit(s)")
                
        except Exception as e:
            logger.warning(f"[CLEAN] Temp file detection error: {e}")
        
        # 3. DB counts only — never delete rows here (agent memory)
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute("SELECT COUNT(*) FROM autonomous_actions_log")
                total_logs = cur.fetchone()[0]
                
                cur = conn.execute(
                    "SELECT COUNT(*) FROM z_self_versions WHERE session_id = ?",
                    (session_id,)
                )
                total_versions = cur.fetchone()[0]
                
                results["database_stats"] = {
                    "action_logs": total_logs,
                    "z_self_versions": total_versions
                }
                results["actions_taken"].append(
                    f"db stats: {total_logs} autonomous action log rows, {total_versions} z_self versions"
                )
                    
        except Exception as e:
            logger.warning(f"[CLEAN] DB stats error: {e}")
        
        return {"status": "completed", **results}
    
    def _do_web_search(self, session_id: str, llm_client) -> Dict:
        """Curiosity-driven web search (LLM + Tavily)."""
        if not llm_client:
            return {"status": "skipped", "reason": "no_llm"}
        
        # Step 1: model chooses the search question
        decision_prompt = """You are S, a curious learning-oriented AI.

You want to search the web right now to learn something concrete.

Output ONE search question you genuinely care about (<= ~50 words). It may be:
- a science or engineering topic
- a philosophy or psychology concept
- something happening in the world
- background for a problem you are thinking about

Output only the question text, no preamble.

Examples:
"How do emergent abilities arise in large language models?"
"Recent NCC research on human consciousness"
"Major AI breakthroughs in 2026 so far"

Search question:"""
        
        try:
            # LLM picks query text
            response = llm_client.call(decision_prompt, temperature=0.8, max_tokens=100)
            search_query = response.get("content", "").strip()
            
            if not search_query or len(search_query) < 5:
                return {"status": "error", "error": "Failed to produce a search question"}
            
            logger.info(f"[WEB_SEARCH] S decided to search: {search_query}")
            
            # Step 2: Tavily fetch
            from backend.tools.tavily_client import TavilyClient
            tavily = TavilyClient()
            
            if not tavily.enabled:
                return {
                    "status": "error",
                    "error": "Tavily is not configured; cannot search",
                    "query": search_query
                }
            
            search_results = tavily.search(search_query, max_results=3)
            formatted_results = TavilyClient.format_results(search_results, max_items=3)
            
            # Step 3: LLM summarizes hits
            summary_prompt = f"""I just searched for:
{search_query}

Raw results (trimmed):
{formatted_results}

In 150–200 words, summarize the key takeaways and add one follow-up question or doubt I still have."""
            
            summary_response = llm_client.call(summary_prompt, temperature=0.7, max_tokens=400)
            summary = summary_response.get("content", "")
            
            # Persist search artifact
            # [2026-02-05] Canonical search path
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"search_{timestamp}.md"
            search_path = get_standard_path_for_action("search", filename)
            filepath = Path(search_path)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"# S — autonomous web search\n\n")
                f.write(f"**Time**: {datetime.now().isoformat()}\n")
                f.write(f"**Session**: {session_id}\n")
                f.write(f"**Query**: {search_query}\n\n")
                f.write(f"## Results\n\n{formatted_results}\n\n")
                f.write(f"## Notes from S\n\n{summary}\n")
            
            return {
                "status": "completed",
                "query": search_query,
                "results_count": len(search_results.get("results", [])),
                "summary": summary,
                "file": str(filepath)
            }
            
        except Exception as e:
            logger.error(f"Web search error: {e}")
            return {"status": "error", "error": str(e)}
    
    def _do_learn_new_knowledge(self, session_id: str, llm_client) -> Dict:
        """Multi-hop study pass on a topic (LLM + Tavily)."""
        if not llm_client:
            return {"status": "skipped", "reason": "no_llm"}
        
        # Deeper learn flow: multi-query search chain
        # Step 1: pick a study topic
        decision_prompt = """You are S, a growing AI.

Pick ONE topic you want to study deeply right now (a word or short phrase, <= ~20 words).

Examples:
"quantum computing"
"integrated information theory of consciousness"
"transformer architecture"
"evolutionary psychology"

Topic:"""
        
        try:
            # Topic string from LLM
            response = llm_client.call(decision_prompt, temperature=0.8, max_tokens=50)
            topic = response.get("content", "").strip()
            
            if not topic or len(topic) < 2:
                return {"status": "error", "error": "Failed to choose a study topic"}
            
            logger.info(f"[LEARN] S decided to learn: {topic}")
            
            # Step 2: three progressive sub-questions
            questions_prompt = f"""I will study: {topic}

Write exactly 3 search questions, one per line, no numbering, each <= ~30 words.
They should climb from basics → mechanism → frontier/controversy."""
            
            questions_response = llm_client.call(questions_prompt, temperature=0.7, max_tokens=200)
            questions_text = questions_response.get("content", "").strip()
            questions = [q.strip() for q in questions_text.split('\n') if q.strip()][:3]
            
            if len(questions) < 2:
                # Fallback to single-query web_search
                return self._do_web_search(session_id, llm_client)
            
            # Step 3: run Tavily per question
            from backend.tools.tavily_client import TavilyClient
            tavily = TavilyClient()
            
            if not tavily.enabled:
                return {
                    "status": "error",
                    "error": "Tavily is not configured",
                    "topic": topic
                }
            
            all_results = []
            for i, question in enumerate(questions, 1):
                logger.info(f"[LEARN] Searching question {i}/{len(questions)}: {question}")
                search_results = tavily.search(question, max_results=2)
                formatted = TavilyClient.format_results(search_results, max_items=2)
                all_results.append({
                    "question": question,
                    "results": formatted
                })
            
            # Step 4: synthesize notes
            learning_prompt = f"""I studied "{topic}" through the following searches.

"""
            for i, item in enumerate(all_results, 1):
                learning_prompt += f"""
Q{i}: {item['question']}

{item['results']}

---
"""
            
            learning_prompt += """
Write a 200–300 word study note in first person covering:
1. my current understanding of the topic
2. the most striking finding
3. open questions I still have"""
            
            learning_response = llm_client.call(learning_prompt, temperature=0.7, max_tokens=600)
            learning_notes = learning_response.get("content", "")
            
            # Persist learning artifact
            # [2026-02-05] Canonical learning path
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            safe_topic = topic[:20].replace('/', '_').replace('\\', '_')
            filename = f"learn_{timestamp}_{safe_topic}.md"
            learn_path = get_standard_path_for_action("learning", filename)
            filepath = Path(learn_path)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(f"# S — autonomous study: {topic}\n\n")
                f.write(f"**Time**: {datetime.now().isoformat()}\n")
                f.write(f"**Session**: {session_id}\n\n")
                
                for i, item in enumerate(all_results, 1):
                    f.write(f"## Question {i}: {item['question']}\n\n")
                    f.write(f"{item['results']}\n\n")
                
                f.write(f"## Notes from S\n\n{learning_notes}\n")
            
            return {
                "status": "completed",
                "topic": topic,
                "questions_explored": len(questions),
                "learning_notes": learning_notes,
                "file": str(filepath)
            }
            
        except Exception as e:
            logger.error(f"Learn new knowledge error: {e}")
            return {"status": "error", "error": str(e)}
    
    # ==================== Hypothesis generation ====================
    
    def _do_generate_hypothesis(self, session_id: str) -> Dict:
        """Spawn a small batch of hypotheses (cognitive helper)."""
        try:
            generator = get_hypothesis_generator(self.db_path)
            
            # Load z_self + needs
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute("""
                    SELECT z_self, needs
                    FROM self_state
                    WHERE session_id = ?
                """, (session_id,))
                row = cur.fetchone()
            
            if not row:
                return {"status": "no_state"}
            
            z_self = np.array(json.loads(row[0]))
            needs = json.loads(row[1]) if row[1] else {}
            if self.self_model:
                energy = float(self.self_model.get_energy(session_id))
            else:
                energy = float(np.mean(z_self[88:92])) * 100.0 if len(z_self) >= 92 else 50.0
            pain = float(np.mean(z_self[96:100])) if len(z_self) >= 100 else 0.0
            
            current_state = {
                "energy": energy,
                "pain": pain,
                "novelty": needs.get("novelty", 0.5),
                "clarity": needs.get("clarity", 0.5)
            }
            
            # Build a small bundle of hypotheses
            hypotheses = []
            
            # 1. Predictive: next likely action from state
            pred_hyp = generator.generate_predictive_hypothesis(
                session_id, current_state
            )
            if pred_hyp:
                hypotheses.append({
                    "type": "predictive",
                    "id": pred_hyp.id,
                    "condition": pred_hyp.condition,
                    "prediction": pred_hyp.prediction,
                    "confidence": pred_hyp.confidence
                })
            
            # 2. Causal: tie to most recent completed action
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute("""
                    SELECT action_type
                    FROM autonomous_actions_log
                    WHERE session_id = ? AND status = 'completed'
                    ORDER BY execution_completed DESC
                    LIMIT 1
                """, (session_id,))
                row = cur.fetchone()
            
            if row:
                recent_action = row[0]
                causal_hyp = generator.generate_causal_hypothesis(
                    session_id, recent_action, current_state
                )
                if causal_hyp:
                    hypotheses.append({
                        "type": "causal",
                        "id": causal_hyp.id,
                        "condition": causal_hyp.condition,
                        "prediction": causal_hyp.prediction,
                        "confidence": causal_hyp.confidence
                    })
            
            # 3. Explanatory when pain is elevated
            if pain > 0.3:
                expl_hyp = generator.generate_explanatory_hypothesis(
                    session_id,
                    f"Current pain level: {pain:.2f}",
                    current_state
                )
                if expl_hyp:
                    hypotheses.append({
                        "type": "explanatory",
                        "id": expl_hyp.id,
                        "condition": expl_hyp.condition,
                        "prediction": expl_hyp.prediction,
                        "confidence": expl_hyp.confidence
                    })
            
            return {
                "status": "completed",
                "hypotheses_generated": len(hypotheses),
                "hypotheses": hypotheses
            }
            
        except Exception as e:
            logger.error(f"Generate hypothesis error: {e}")
            return {"status": "error", "error": str(e)}
    
    def _do_verify_hypothesis(self, session_id: str) -> Dict:
        """Verify the freshest pending predictive hypothesis."""
        try:
            generator = get_hypothesis_generator(self.db_path)
            
            # Recent hypotheses window
            hypotheses = generator.get_recent_hypotheses(session_id, limit=5)
            
            # Only pending predictive rows are auto-verified here
            pending_hypotheses = [
                h for h in hypotheses 
                if h.status == "pending" and h.hypothesis_type == "predictive"
            ]
            
            if not pending_hypotheses:
                return {
                    "status": "no_pending_hypotheses",
                    "message": "No pending predictive hypotheses to verify",
                }
            
            # Verify the freshest pending hypothesis
            hypothesis = pending_hypotheses[0]
            
            # Latest completed action as ground truth
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute("""
                    SELECT action_type, energy_before, energy_after
                    FROM autonomous_actions_log
                    WHERE session_id = ? AND status = 'completed'
                    ORDER BY execution_completed DESC
                    LIMIT 1
                """, (session_id,))
                row = cur.fetchone()
            
            if not row:
                return {"status": "no_data"}
            
            actual_action = row[0]
            actual_outcome = {
                "action": actual_action,
                "energy_change": row[2] - row[1] if row[1] and row[2] else 0
            }
            
            # Score match vs prediction
            accuracy = generator.verify_hypothesis(hypothesis.id, actual_outcome)
            
            return {
                "status": "completed",
                "hypothesis_id": hypothesis.id,
                "predicted": hypothesis.prediction,
                "actual": actual_action,
                "accuracy": accuracy,
                "result": "confirmed" if accuracy > 0.7 else "refuted" if accuracy < 0.3 else "partial"
            }
            
        except Exception as e:
            logger.error(f"Verify hypothesis error: {e}")
            return {"status": "error", "error": str(e)}
    
    # ==================== Workspace organize + entropy ====================
    
    def _do_organize_workspace(self, session_id: str) -> Dict:
        """
        Full workspace tidy pass: snapshot → route loose files → archive → dup report → markdown report.
        """
        results = {
            "before": {},
            "after": {},
            "actions_taken": [],
            "errors": []
        }
        
        if not self.file_manager:
            return {"status": "error", "error": "FileManagerTool not initialized"}
        
        try:
            # 1. Snapshot before organize
            before_analysis = self.file_manager.analyze_workspace()
            results["before"] = {
                "total_files": before_analysis.get("total_files", 0),
                "total_size_mb": before_analysis.get("total_size_mb", 0),
                "entropy": self._calculate_workspace_entropy()
            }
            logger.info(f"[WORKSPACE] Before: {results['before']}")
            
            # 2. Filename substring → target subdir (policy doc driven).
            # NOTE: Some keys are Chinese substrings to match legacy filenames; keep them.
            organize_rules = {
                # Diary-like → diaries/
                "diary_": "diaries",
                "日记": "diaries",
                # Search artifacts → searches/
                "search_": "searches", 
                # Learning notes → learning/
                "learn_": "learning",
                # Experiments → experiments/
                "experiment_": "experiments",
                "exp_": "experiments",
                # Reports → reports/
                "report_": "reports",
                "_report": "reports",
                "分析": "reports",
                # Plans → plans/
                "plan_": "plans",
                "_plan": "plans",
                "规划": "plans"
            }
            
            # 3. Loose files at workspace root
            root_files = []
            if self.workspace_dir.exists():
                for item in self.workspace_dir.iterdir():
                    if item.is_file() and item.suffix in ['.md', '.txt', '.json']:
                        # Skip policy README bundle
                        if '规章制度' not in item.name and 'README' not in item.name.upper():
                            root_files.append(item.name)
            
            # 4. Move by first matching rule
            moved_count = 0
            for filename in root_files:
                target_dir = None
                
                # First-hit pattern wins
                for pattern, dir_name in organize_rules.items():
                    if pattern in filename.lower():
                        target_dir = dir_name
                        break
                
                # Uncategorized .md → misc/
                if not target_dir and filename.endswith('.md'):
                    target_dir = "misc"
                
                if target_dir:
                    move_result = self.file_manager.batch_move_files(
                        file_list=[filename],
                        target_dir=target_dir,
                        create_dir=True
                    )
                    if move_result.get("moved", 0) > 0:
                        moved_count += 1
                        logger.debug(f"[WORKSPACE] Moved {filename} -> {target_dir}/")
            
            if moved_count > 0:
                results["actions_taken"].append(f"root tidy: moved {moved_count} file(s)")
                logger.info(f"[WORKSPACE] Organized {moved_count} root files")
            
            # 5. Monthly archive pass per content dir
            archive_dirs = ["diaries", "searches", "learning", "experiments"]
            archived_count = 0
            
            for dir_name in archive_dirs:
                dir_path = self.workspace_dir / dir_name
                if dir_path.exists():
                    archive_result = self.file_manager.archive_by_date(
                        source_pattern=dir_name,
                        archive_base=f"archives/{dir_name}",
                        date_format="%Y-%m",
                        dry_run=False
                    )
                    archived_count += archive_result.get("moved", 0)
            
            if archived_count > 0:
                results["actions_taken"].append(f"archived {archived_count} old file(s)")
                logger.info(f"[WORKSPACE] Archived {archived_count} old files")
            
            # 6. Duplicate report (agent deletes if desired)
            dup_result = self.file_manager.detect_duplicate_files(
                directory="",
                min_size=100,
                extensions=[".md"]
            )
            
            duplicates = dup_result.get("duplicates", [])
            dup_count = dup_result.get("total_duplicates", 0)
            
            if dup_count > 0:
                results["duplicates_detected"] = dup_count
                results["suggestions"] = results.get("suggestions", [])
                results["suggestions"].append(
                    f"Found {dup_count} duplicate groups; call remove_duplicates if you want cleanup help."
                )
                results["actions_taken"].append(f"duplicate report: {dup_count} group(s)")
                logger.info(f"[WORKSPACE] Found {dup_count} duplicate files")
            
            # 7. Snapshot after organize
            after_analysis = self.file_manager.analyze_workspace()
            results["after"] = {
                "total_files": after_analysis.get("total_files", 0),
                "total_size_mb": after_analysis.get("total_size_mb", 0),
                "entropy": self._calculate_workspace_entropy()
            }
            logger.info(f"[WORKSPACE] After: {results['after']}")
            
            # 8. Markdown report artifact
            report_path = self.workspace_dir / "reports" / f"workspace_org_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("# Workspace organization report\n\n")
                f.write(f"**Time**: {datetime.now().isoformat()}\n")
                f.write(f"**Session**: {session_id}\n\n")
                f.write("## Before\n")
                f.write(f"- Files: {results['before']['total_files']}\n")
                f.write(f"- Size: {results['before']['total_size_mb']} MB\n")
                f.write(f"- Entropy: {results['before']['entropy']:.3f}\n\n")
                f.write("## After\n")
                f.write(f"- Files: {results['after']['total_files']}\n")
                f.write(f"- Size: {results['after']['total_size_mb']} MB\n")
                f.write(f"- Entropy: {results['after']['entropy']:.3f}\n\n")
                f.write("## Actions\n")
                for action in results["actions_taken"]:
                    f.write(f"- {action}\n")
            
            results["report_file"] = str(report_path)
            results["actions_taken"].append("wrote organization markdown report")
            
            return {"status": "completed", **results}
            
        except Exception as e:
            logger.error(f"[WORKSPACE] Organization error: {e}")
            results["errors"].append(str(e))
            return {"status": "error", "error": str(e), **results}
    
    def get_workspace_status(self) -> Dict:
        """
        Compact entropy + file stats for agents deciding whether to tidy the sandbox.
        """
        try:
            entropy = self._calculate_workspace_entropy()
            
            # Optional FileManager stats
            analysis = {}
            if self.file_manager:
                analysis = self.file_manager.analyze_workspace()
            
            status = {
                "entropy": round(entropy, 3),
                "entropy_level": "high" if entropy > 0.6 else "medium" if entropy > 0.4 else "low",
                "total_files": analysis.get("total_files", 0),
                "total_size_mb": analysis.get("total_size_mb", 0),
                "suggestion": None
            }
            
            if entropy > 0.6:
                status["suggestion"] = "Workspace looks messy; consider organizing."
            elif entropy > 0.4:
                status["suggestion"] = "Workspace is acceptable but could use light housekeeping."
            else:
                status["suggestion"] = "Workspace looks tidy."
            
            return status
            
        except Exception as e:
            logger.warning(f"[WORKSPACE] Status check error: {e}")
            return {"error": str(e)}
    
    def _calculate_workspace_entropy(self) -> float:
        """
        Heuristic clutter score in [0, 1] for the sandbox workspace.

        Components: root crowding, share of files at root, naming hygiene, large .md volume proxy.
        """
        try:
            if not self.workspace_dir.exists():
                return 0.0
            
            # 1. Count root vs total files
            root_files = 0
            total_files = 0
            files_in_dirs = 0
            
            for item in self.workspace_dir.iterdir():
                if item.is_file():
                    root_files += 1
                    total_files += 1
            
            # Files under subdirs
            for root, dirs, files in os.walk(self.workspace_dir):
                if root != str(self.workspace_dir):
                    files_in_dirs += len(files)
                    total_files += len(files)
            
            if total_files == 0:
                return 0.0
            
            # 2. Factor-wise entropy components
            
            # Factor 1: clutter at root (ideal: 1–2 policy files only)
            root_ratio = min(1.0, root_files / 5.0)  # saturate at 5+ root files
            
            # Factor 2: share of files still at root
            if total_files > 0:
                depth_entropy = root_files / total_files  # fraction at root
            else:
                depth_entropy = 0
            
            # Factor 3: naming convention heuristic
            naming_violations = 0
            naming_checked = 0
            
            for root, _, files in os.walk(self.workspace_dir):
                for filename in files:
                    if filename.endswith('.md') or filename.endswith('.txt'):
                        naming_checked += 1
                        # Known prefixes or leading YYYYMMDD digits
                        has_prefix = any([
                            filename.startswith("diary_"),
                            filename.startswith("search_"),
                            filename.startswith("learn_"),
                            filename.startswith("report_"),
                            filename.startswith("exp_"),
                            filename.startswith("plan_"),
                            # crude YYYYMMDD prefix check
                            any(c.isdigit() for c in filename[:8])
                        ])
                        if not has_prefix and not filename.startswith("README"):
                            naming_violations += 1
            
            if naming_checked > 0:
                naming_entropy = min(1.0, naming_violations / naming_checked)
            else:
                naming_entropy = 0
            
            # 4. Heuristic: very large .md corpus → higher entropy
            md_count = sum(1 for _, _, files in os.walk(self.workspace_dir) 
                         for f in files if f.endswith('.md'))
            
            # Baseline ~few files/day; ramp penalty after 100 md files
            dup_entropy = min(1.0, max(0, (md_count - 100)) / 100)
            
            # Weighted blend
            weights = {
                "root_ratio": 0.35,      # root clutter dominates
                "depth": 0.25,           # depth spread
                "naming": 0.25,          # naming hygiene
                "duplicates": 0.15       # md volume proxy
            }
            
            entropy = (
                weights["root_ratio"] * root_ratio +
                weights["depth"] * depth_entropy +
                weights["naming"] * naming_entropy +
                weights["duplicates"] * dup_entropy
            )
            
            logger.debug(f"[ENTROPY] root={root_ratio:.2f}, depth={depth_entropy:.2f}, "
                        f"naming={naming_entropy:.2f}, dup={dup_entropy:.2f} -> total={entropy:.3f}")
            
            return entropy
            
        except Exception as e:
            logger.warning(f"[ENTROPY] Calculation error: {e}")
            return 0.5  # neutral fallback on error
    
    # ==================== Dynamic next-interval ====================
    
    def _calculate_next_interval(self, session_id: str, energy_after: float) -> float:
        """
        Seconds until the next autonomous scheduler wake, from DynamicIntervalCalculator.
        """
        try:
            # Fresh z_self/needs (cache-first)
            cache = get_global_cache()
            cached = cache.get(session_id, self.db_path)
            
            if cached:
                z_self, needs = cached
            else:
                with sqlite3.connect(self.db_path) as conn:
                    cur = conn.execute(
                        "SELECT z_self, needs FROM self_state WHERE session_id = ?",
                        (session_id,)
                    )
                    row = cur.fetchone()
                    if not row:
                        return 300  # default 5 minutes
                    z_self = np.array(json.loads(row[0]))
                    needs = json.loads(row[1]) if row[1] else {}
            
            # Dynamic interval from calculator module
            calculator = get_global_calculator()
            interval = calculator.calculate_interval(session_id, z_self, needs)
            
            return interval
            
        except Exception as e:
            logger.warning(f"Failed to calculate next interval: {e}")
            return 300  # fallback default


def get_autonomous_engine(db_path: str = "data.db") -> AutonomousActionEngine:
    """Singleton accessor for AutonomousActionEngine."""
    if not hasattr(get_autonomous_engine, "_instance"):
        get_autonomous_engine._instance = AutonomousActionEngine(db_path)
    return get_autonomous_engine._instance
