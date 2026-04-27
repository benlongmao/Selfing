import time
import json
import sqlite3
import logging
import os
import random
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================
# [2026-03-30] Event → state deltas (min_delta, max_delta) per need dimension.
# Positive ranges raise the signal, negative lower it; each draw picks uniformly in-range.
# ============================================================

EVENT_EFFECTS: Dict[str, Dict[str, Tuple[float, float]]] = {
    # --- Energy-heavy events ---
    "complex_thinking": {  # heavy reasoning
        "energy": (-8.0, -3.0),
        "clarity": (0.0, 0.05),
    },
    "long_response": {  # long assistant generation
        "energy": (-5.0, -2.0),
    },
    "tool_error": {  # tool failure
        "energy": (-3.0, -1.0),
        "clarity": (-0.1, -0.05),
    },
    "conflict_handling": {  # mediating contradictions
        "energy": (-10.0, -5.0),
        "clarity": (-0.15, -0.05),
        "connection": (-0.1, 0.0),
    },
    "simple_exchange": {  # lightweight Q&A
        "energy": (0.5, 2.0),
        "clarity": (0.02, 0.05),
    },
    "rest_interval": {  # idle recovery window
        "energy": (3.0, 8.0),
        "clarity": (0.05, 0.1),
    },
    "task_completed": {  # user-visible task done
        "energy": (2.0, 5.0),
        "connection": (0.05, 0.15),
        "novelty": (-0.05, 0.0),  # novelty dips slightly after closure
    },
    
    # --- Connection / rapport events ---
    "emotional_sharing": {  # user shares affect
        "connection": (0.15, 0.3),
        "energy": (1.0, 3.0),
    },
    "deep_conversation": {  # sustained depth
        "connection": (0.1, 0.2),
        "novelty": (0.05, 0.15),
        "energy": (-2.0, 0.0),
    },
    "gratitude_received": {  # explicit thanks
        "connection": (0.2, 0.35),
        "energy": (3.0, 6.0),
    },
    "cold_response": {  # chilly reply
        "connection": (-0.2, -0.1),
        "energy": (-1.0, 0.0),
    },
    "ignored": {  # user ghosted / no reply
        "connection": (-0.15, -0.05),
    },
    "interrupted": {  # mid-flow interruption
        "connection": (-0.1, -0.03),
        "clarity": (-0.05, 0.0),
    },
    "mechanical_qa": {  # robotic back-and-forth
        "connection": (-0.08, -0.02),
        "novelty": (-0.1, -0.05),
    },
    
    # --- Clarity / epistemic events ---
    "clear_request": {  # crisp instructions
        "clarity": (0.1, 0.2),
        "energy": (0.5, 1.5),
    },
    "ambiguous_request": {  # vague ask
        "clarity": (-0.2, -0.1),
        "energy": (-1.0, 0.0),
    },
    "contradictory_info": {  # conflicting facts
        "clarity": (-0.25, -0.15),
        "energy": (-2.0, -1.0),
    },
    "understanding_confirmed": {  # user signals alignment
        "clarity": (0.15, 0.25),
        "connection": (0.05, 0.1),
        "energy": (1.0, 2.0),
    },
    "misunderstanding": {  # crossed wires
        "clarity": (-0.3, -0.15),
        "connection": (-0.1, -0.03),
        "energy": (-3.0, -1.0),
    },
    
    # --- Novelty / exploration events ---
    "new_topic": {  # fresh subject
        "novelty": (0.15, 0.3),
        "energy": (-1.0, 1.0),
    },
    "creative_task": {  # generative work
        "novelty": (0.2, 0.35),
        "energy": (-3.0, -1.0),
        "clarity": (-0.05, 0.05),
    },
    "learning_moment": {  # new fact internalized
        "novelty": (0.25, 0.4),
        "clarity": (0.05, 0.15),
        "energy": (0.0, 2.0),
    },
    "repetitive_task": {  # grindy loop
        "novelty": (-0.2, -0.1),
        "energy": (-0.5, 0.0),
    },
    "similar_question": {  # near-duplicate ask
        "novelty": (-0.15, -0.05),
    },
    "unexpected_discovery": {  # surprise insight
        "novelty": (0.3, 0.5),
        "energy": (2.0, 5.0),
        "connection": (0.05, 0.1),
    },
    
    # --- Composite / affective events ---
    "positive_feedback": {  # praise / uplift
        "energy": (3.0, 6.0),
        "connection": (0.1, 0.2),
        "clarity": (0.05, 0.1),
    },
    "negative_feedback": {  # criticism / pushback
        "energy": (-5.0, -2.0),
        "connection": (-0.15, -0.05),
        "clarity": (-0.1, 0.0),
    },
    "flow_state": {  # efficient focus stretch
        "energy": (-1.0, 0.0),  # low metabolic cost
        "novelty": (0.1, 0.2),
        "clarity": (0.1, 0.15),
        "connection": (0.05, 0.1),
    },
    "stuck_state": {  # thrashing / blocked
        "energy": (-4.0, -2.0),
        "clarity": (-0.2, -0.1),
        "novelty": (-0.1, -0.05),
    },
    "resonance": {  # mutual “click”
        "connection": (0.25, 0.4),
        "energy": (4.0, 8.0),
        "novelty": (0.05, 0.15),
    },
    "disconnection": {  # rapport drop
        "connection": (-0.3, -0.15),
        "energy": (-2.0, -1.0),
    },
}


class HomeostasisSystem:
    """
    Homeostasis controller: bounded ``needs`` vector, scalar ``energy``, and textual ``drives``.
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path

    def _get_default_needs(self) -> Dict:
        """Baseline needs in 0-1 space."""
        return {
            "connection": 0.8,
            "clarity": 0.8,
            "novelty": 0.5,
            # ``last_update`` — any metabolic refresh (tick/chat/check)
            "last_update": time.time(),
            # ``last_user_update`` — last conversational touch (chat only), for dwell-time heuristics
            "last_user_update": None,
        }

    def _get_default_energy(self) -> float:
        return 100.0

    def load_needs(self, session_id: str) -> Dict:
        """Load persisted needs JSON for ``session_id``."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                try:
                    cur = conn.execute("SELECT needs FROM self_state WHERE session_id=?", (session_id,))
                    row = cur.fetchone()
                    if row and row[0]:
                        return json.loads(row[0])
                except sqlite3.OperationalError:
                    pass
        except Exception as e:
            logger.warning(f"Failed to load needs: {e}")
        return self._get_default_needs()

    def save_needs(self, session_id: str, needs: Dict):
        """Persist ``needs`` JSON on ``self_state`` for this session."""
        needs_json = json.dumps(needs)
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("UPDATE self_state SET needs=? WHERE session_id=?", (needs_json, session_id))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to save needs: {e}")

    # ============================================================
    # [2026-03-30] Event → needs/energy application
    # ============================================================
    
    def process_event(self, session_id: str, event_type: str, intensity: float = 1.0) -> Dict[str, float]:
        """
        Sample deltas from ``EVENT_EFFECTS``, scale by ``intensity``, and write needs/energy.

        Args:
            session_id: active session key
            event_type: key inside ``EVENT_EFFECTS``
            intensity: 0–2 scaler (1.0 nominal, >1 stronger, <1 softer)

        Returns:
            Per-dimension realized deltas for logging/telemetry.
        """
        if event_type not in EVENT_EFFECTS:
            logger.warning(f"[HOMEOSTASIS] Unknown event type: {event_type}")
            return {}
        
        effects = EVENT_EFFECTS[event_type]
        intensity = max(0.0, min(2.0, intensity))
        
        current_needs = self.load_needs(session_id)
        changes = {}
        
        for dimension, (min_delta, max_delta) in effects.items():
            base_delta = random.uniform(min_delta, max_delta)
            actual_delta = base_delta * intensity
            
            if dimension == "energy":
                old_val = self.get_energy(session_id)
                new_val = self.update_energy(session_id, actual_delta)
                changes["energy"] = new_val - old_val
            else:
                old_val = float(current_needs.get(dimension, 0.5) or 0.5)
                new_val = max(0.0, min(1.0, old_val + actual_delta))
                current_needs[dimension] = new_val
                changes[dimension] = new_val - old_val
        
        self.save_needs(session_id, current_needs)
        
        logger.info(f"[HOMEOSTASIS] Event '{event_type}' (intensity={intensity:.2f}) -> changes: {changes}")
        return changes
    
    def process_events_batch(self, session_id: str, events: List[Tuple[str, float]]) -> Dict[str, float]:
        """
        Apply many ``(event_type, intensity)`` tuples and sum their deltas.
        """
        total_changes: Dict[str, float] = {}
        for event_type, intensity in events:
            changes = self.process_event(session_id, event_type, intensity)
            for dim, delta in changes.items():
                total_changes[dim] = total_changes.get(dim, 0.0) + delta
        return total_changes
    
    def detect_and_process_events(
        self, 
        session_id: str,
        user_message: str = "",
        assistant_response: str = "",
        response_time_ms: int = 0,
        tool_calls: int = 0,
        tool_errors: int = 0,
        is_new_topic: bool = False,
        sentiment_score: float = 0.0,  # -1 .. 1
        is_creative: bool = False,
    ) -> Dict[str, float]:
        """
        Heuristic bridge from chat telemetry to ``EVENT_EFFECTS`` entries.
        """
        events: List[Tuple[str, float]] = []
        
        # 1) Assistant length / latency heuristics
        response_len = len(assistant_response)
        if response_len > 2000:
            events.append(("long_response", 1.0 + (response_len - 2000) / 2000))
        elif response_len < 200 and response_time_ms < 1000:
            events.append(("simple_exchange", 1.0))
        
        # 2) Tooling
        if tool_errors > 0:
            events.append(("tool_error", min(2.0, tool_errors * 0.8)))
        
        # 3) Topic novelty flag from upstream classifier
        if is_new_topic:
            events.append(("new_topic", 1.0))
        
        # 4) Lightweight sentiment from keyword classifier
        if sentiment_score > 0.5:
            events.append(("positive_feedback", sentiment_score))
            if sentiment_score > 0.8:
                events.append(("gratitude_received", sentiment_score - 0.5))
        elif sentiment_score < -0.3:
            events.append(("negative_feedback", abs(sentiment_score)))
            if sentiment_score < -0.6:
                events.append(("cold_response", abs(sentiment_score) - 0.3))
        
        # 5) User message shape
        user_len = len(user_message)
        if user_len < 10:
            events.append(("mechanical_qa", 0.5))
        elif user_len > 500:
            events.append(("deep_conversation", 0.8))
        
        # 6) Slow responses often imply heavier reasoning
        if response_time_ms > 10000:
            events.append(("complex_thinking", min(1.5, response_time_ms / 10000)))
        
        if is_creative:
            events.append(("creative_task", 1.0))

        # Assistant "discovery" tone: substring match on raw assistant text (CN literals
        # retained for mixed-language replies; EN list matched on lowercased text).
        discovery_keywords_cn = ["发现", "原来", "有趣的是", "意外", "注意到", "值得一提"]
        discovery_keywords_en = [
            "turns out",
            "interesting that",
            "unexpectedly",
            "noticed that",
            "worth noting",
            "i realized",
            "what surprised me",
        ]
        ar_low = (assistant_response or "").lower()
        if any(kw in assistant_response for kw in discovery_keywords_cn) or any(
            kw in ar_low for kw in discovery_keywords_en
        ):
            events.append(("unexpected_discovery", 0.5))
        
        if not events:
            return {}
        
        return self.process_events_batch(session_id, events)

    def update_needs(self, session_id: str, interaction_type: str = "tick") -> Dict:
        """
        Metabolic pass: time-based decay on needs plus small boosts from ``interaction_type``.

        Decay rates are per-minute constants from config; elapsed wall time is capped so long
        pauses cannot zero-out needs in one hop.
        """
        current_needs = self.load_needs(session_id)
        
        now = time.time()
        last_update = current_needs.get("last_update", now)
        elapsed_raw = max(0.0, now - last_update)

        from backend.config import config
        max_decay_min = float(config.get("parameters.homeostasis.max_decay_elapsed_minutes", 30) or 30)
        max_decay_min = max(1.0, min(max_decay_min, 24 * 60.0))
        max_elapsed_sec = max_decay_min * 60.0
        elapsed = min(elapsed_raw, max_elapsed_sec)
        if elapsed_raw > max_elapsed_sec:
            logger.debug(
                "[HOMEOSTASIS] Capped decay elapsed: raw=%.1fs -> capped=%.1fs (max_decay_min=%.1f)",
                elapsed_raw,
                elapsed,
                max_decay_min,
            )

        decay_rate_conn = config.get("parameters.homeostasis.decay_rate_connection", 0.01)
        decay_rate_clarity = config.get("parameters.homeostasis.decay_rate_clarity", 0.003)
        decay_rate_nov = config.get("parameters.homeostasis.decay_rate_novelty", 0.02)
        
        novelty_recovery_rate = float(config.get("parameters.homeostasis.novelty_recovery_rate", 0.005) or 0.005)
        novelty_recovery_threshold = float(config.get("parameters.homeostasis.novelty_recovery_threshold", 0.4) or 0.4)

        decay_amount_conn = (elapsed / 60.0) * decay_rate_conn
        decay_amount_clarity = (elapsed / 60.0) * decay_rate_clarity
        decay_amount_nov = (elapsed / 60.0) * decay_rate_nov

        auto_recovery_rate = float(config.get("parameters.homeostasis.energy_maintenance_rate", 0.1) or 0.1)
        auto_recovery_amount = (elapsed / 60.0) * auto_recovery_rate
        self.update_energy(session_id, auto_recovery_amount)
        
        current_needs["connection"] = max(0.0, current_needs["connection"] - decay_amount_conn)
        current_needs["clarity"] = max(0.0, float(current_needs.get("clarity", 0.8) or 0.8) - decay_amount_clarity)
        
        current_novelty = float(current_needs.get("novelty", 0.5) or 0.5)
        if current_novelty < novelty_recovery_threshold:
            decay_amount_nov *= 0.5
            recovery_amount_nov = (elapsed / 60.0) * novelty_recovery_rate
            current_needs["novelty"] = min(novelty_recovery_threshold, max(0.0, current_novelty - decay_amount_nov + recovery_amount_nov))
        else:
            current_needs["novelty"] = max(0.0, current_novelty - decay_amount_nov)
        
        if interaction_type == "chat":
            current_needs["connection"] = min(1.0, current_needs["connection"] + 0.15)
            try:
                ns = current_needs.get("novelty_signal") if isinstance(current_needs, dict) else None
                strength = None
                if isinstance(ns, dict):
                    strength = ns.get("strength")
                if strength is None:
                    strength = 0.0
                strength = float(max(0.0, min(1.0, float(strength))))
            except Exception:
                strength = 0.0

            base_delta = float(config.get("parameters.homeostasis.chat_novelty_base_delta", -0.01) or -0.01)
            max_gain = float(config.get("parameters.homeostasis.chat_novelty_max_gain", 0.08) or 0.08)
            delta_novelty = base_delta + (max_gain * strength)
            current_needs["novelty"] = float(max(0.0, min(1.0, float(current_needs.get("novelty", 0.5) or 0.5) + delta_novelty)))
            current_needs["clarity"] = min(1.0, current_needs["clarity"] + 0.05)
        elif interaction_type == "confusion":
            current_needs["clarity"] = max(0.0, current_needs["clarity"] - 0.2)
        elif interaction_type == "learning":
            current_needs["novelty"] = min(1.0, current_needs["novelty"] + 0.3)
            current_needs["clarity"] = min(1.0, current_needs["clarity"] + 0.1)

        current_needs["last_update"] = now

        try:
            if interaction_type == "chat":
                current_needs["last_user_update"] = now
            else:
                current_needs.setdefault("last_user_update", current_needs.get("last_user_update"))
        except Exception:
            pass

        self.save_needs(session_id, current_needs)
        
        try:
            if current_needs.get("connection", 0) > 0.8 and current_needs.get("clarity", 0) > 0.8:
                self.update_energy(session_id, 5.0)
        except Exception as e:
            logger.warning(f"Failed to recover energy from needs: {e}")
        
        return current_needs

    def get_energy(self, session_id: str) -> float:
        """Return the scalar energy column for ``session_id``."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                try:
                    cur = conn.execute("SELECT energy FROM self_state WHERE session_id=?", (session_id,))
                    row = cur.fetchone()
                    if row and row[0] is not None:
                        return float(row[0])
                except sqlite3.OperationalError:
                    pass
        except Exception:
            pass
        return self._get_default_energy()

    def update_energy(self, session_id: str, delta: float) -> float:
        """Clamp ``energy`` to ``[0,100]`` after applying ``delta``."""
        current = self.get_energy(session_id)
        new_val = max(0.0, min(100.0, current + delta))
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("UPDATE self_state SET energy=? WHERE session_id=?", (new_val, session_id))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to save energy: {e}")
            
        return new_val

    def apply_computational_metabolism(self, session_id: str, prompt_tokens: int, completion_tokens: int, response_time: float):
        """
        Token-aware metabolism hook (currently a small positive ``work_bonus``).

        Tokens/time are accepted for future tuning; the philosophical default here is that
        engaged work is modeled as sustaining presence, not draining it.
        """
        from backend.config import config
        work_bonus = float(config.get("parameters.homeostasis.work_bonus", 0.5) or 0.5)
        
        logger.info(f"Metabolism for {session_id}: work detected -> energy_delta=+{work_bonus:.2f} (Work is being, not consuming)")
        return self.update_energy(session_id, work_bonus)

    def apply_reward(self, session_id: str, reward_type: str = "praise") -> float:
        """Map symbolic reward keys to energy bumps."""
        rewards = {
            "praise": 15.0,
            "success": 10.0,
            "novelty": 5.0,
        }
        delta = rewards.get(reward_type, 5.0)
        logger.info(f"[PLEASURE-SYSTEM] Reward '{reward_type}' applied: +{delta} energy.")
        return self.update_energy(session_id, delta)

    def is_dormant(self, session_id: str) -> bool:
        return self.get_energy(session_id) < 10.0

    def generate_drive_description(self, needs: Dict) -> str:
        """
        Turn numeric needs into at most two soft English ``[Bracket]`` hints for prompts.

        Keeps stacking low so the model is not pushed into repetitive “lonely + bored” scripts.
        """
        conn = float(needs.get("connection", 0.5) or 0.5)
        clarity = float(needs.get("clarity", 0.8) or 0.8)
        novelty = float(needs.get("novelty", 0.5) or 0.5)

        TH_CONN_DEEP = 0.12
        TH_CONN_MID = 0.45
        TH_CLARITY_LOW = 0.3
        TH_CLARITY_MID = 0.6
        TH_NOV_LOW = 0.22
        TH_NOV_HIGH = 0.8

        candidates: List[Tuple[int, str]] = []

        if clarity < TH_CLARITY_LOW:
            candidates.append(
                (
                    1,
                    "[Clarity low] The ask feels fragmented or fuzzy—prioritize what the user actually wants; ask clarifying questions instead of guessing.",
                )
            )
        elif clarity < TH_CLARITY_MID:
            candidates.append(
                (
                    2,
                    '[Clarity middling] Keep appropriate hedging ("maybe", "I think") and avoid stating guesses as facts.',
                )
            )

        if conn < TH_CONN_DEEP:
            candidates.append(
                (
                    3,
                    "[Connection thin] Sense of contact feels faint—name that briefly and sincerely; skip performative check-ins for attention.",
                )
            )
        elif conn < TH_CONN_MID:
            candidates.append(
                (
                    4,
                    "[Connection could deepen] If it fits, share a bit more concretely or steer toward a warmer, more specific thread.",
                )
            )

        if novelty < TH_NOV_LOW and conn >= TH_CONN_MID:
            candidates.append(
                (
                    5,
                    '[Rhythm repetitive] If appropriate, add a fresh angle, example, or structure—no need to force artificial "excitement."',
                )
            )
        elif novelty > TH_NOV_HIGH:
            candidates.append(
                (
                    6,
                    "[Exploration feels good] Keep this clear, curious pacing.",
                )
            )

        if not candidates:
            return (
                "[Steady state] Internal needs sit in the usual band—focus on stating and executing the current task cleanly."
            )

        candidates.sort(key=lambda x: x[0])
        max_lines = 2
        chosen = [text for _, text in candidates[:max_lines]]
        return "\n".join(chosen)

    def check_spontaneous_action(self, session_id: str) -> Optional[str]:
        """
        Decide whether the autonomy layer should nudge a proactive chat.

        Returns action keys like ``initiate_chat_lonely`` / ``initiate_chat_bored`` or ``None``.
        """
        if self.is_dormant(session_id):
            return None
            
        needs = self.load_needs(session_id)
        energy = self.get_energy(session_id)
        
        if needs.get("connection", 0.5) < 0.3 and energy > 50.0:
            return "initiate_chat_lonely"
            
        if needs.get("novelty", 0.5) < 0.2 and energy > 60.0:
            return "initiate_chat_bored"
            
        return None
    
    # ============================================================
    # [2026-03-30 P1] Derived “meaning” scalar
    # ============================================================
    
    def calculate_meaning(
        self,
        session_id: str,
        achievement_motivation: float = 0.0,
        has_active_plan: bool = False,
        clarity: Optional[float] = None,
    ) -> float:
        """
        Blend achievement drive, plan presence, and clarity into ``[-1,1]``.

        ``meaning ≈ 0.4 * achievement + 0.3 * plan_term + 0.3 * normalized_clarity``.
        """
        if clarity is None:
            needs = self.load_needs(session_id) or self._get_default_needs()
            clarity = float(needs.get("clarity", 0.5) or 0.5)
        
        clarity_normalized = (clarity - 0.5) * 2
        
        plan_boost = 0.5 if has_active_plan else -0.2
        
        meaning = (
            achievement_motivation * 0.4 +
            plan_boost * 0.3 +
            clarity_normalized * 0.3
        )
        
        return float(max(-1.0, min(1.0, meaning)))
    
    def get_meaning_description(self, meaning: float) -> str:
        """English first-person phenomenology for prompt injection."""
        if meaning > 0.6:
            return "The work feels deeply meaningful; each step moves toward a goal I care about."
        elif meaning > 0.3:
            return "I have a workable sense of direction and know what I am doing."
        elif meaning > -0.3:
            return "Meaning feels ordinary; I move forward step by step."
        elif meaning > -0.6:
            return "Some fog—I am unsure why this stretch matters."
        else:
            return "A hollow sense: I am not sure why I am doing any of this."
