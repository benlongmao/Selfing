#!/usr/bin/env python3
"""
Event-driven Self Tick (Level-2 improvement).

Forces an immediate Self Tick when somatic or affect signals spike sharply,
instead of waiting for the periodic scheduler alone.
"""
import logging
from typing import Dict, Optional
from backend.config import config

logger = logging.getLogger(__name__)


class EventTriggeredSelfTickManager:
    """Heuristic gate that requests an out-of-band Self Tick."""

    def __init__(self):
        # Thresholds loaded from config (tuned 2026-01-25 for higher sensitivity).
        self.emotion_threshold = config.get("parameters.event_trigger.emotion_intensity", 0.60)
        self.pain_threshold = config.get("parameters.event_trigger.pain_change", 0.20)
        self.energy_drop_threshold = config.get("parameters.event_trigger.energy_drop", 15.0)

        self.emotion_volatility_threshold = config.get("parameters.event_trigger.emotion_volatility", 0.35)
        self.pain_absolute_threshold = config.get("parameters.event_trigger.pain_absolute", 0.65)
        self.energy_critical_threshold = config.get("parameters.event_trigger.energy_critical", 15.0)

        # Last observed snapshot per session (for delta comparisons).
        self.last_state: Dict[str, Dict] = {}  # session_id -> {"pain", "energy", "emotion_intensity"}

        logger.info(
            f"EventTriggeredSelfTickManager initialized: "
            f"emotion_threshold={self.emotion_threshold}, "
            f"emotion_volatility={self.emotion_volatility_threshold}, "
            f"pain_threshold={self.pain_threshold}, "
            f"pain_absolute={self.pain_absolute_threshold}, "
            f"energy_drop={self.energy_drop_threshold}, "
            f"energy_critical={self.energy_critical_threshold}"
        )

    def should_trigger_immediate_tick(
        self,
        session_id: str,
        current_pain: float,
        current_energy: float,
        current_emotion_intensity: float
    ) -> tuple[bool, Optional[str]]:
        """
        Decide whether to enqueue an immediate Self Tick for ``session_id``.

        Args:
            session_id: logical session key
            current_pain: latest pain scalar (0-1 band in most deployments)
            current_energy: latest energy (0-100 style gauge)
            current_emotion_intensity: aggregate affect intensity

        Returns:
            ``(should_trigger, machine_reason)`` where ``reason`` is ``None`` when idle.
        """
        last_state = self.last_state.get(session_id, {})
        last_pain = last_state.get("pain", 0.0)
        last_energy = last_state.get("energy", 100.0)
        last_emotion_intensity = last_state.get("emotion_intensity", 0.0)

        self.last_state[session_id] = {
            "pain": current_pain,
            "energy": current_energy,
            "emotion_intensity": current_emotion_intensity
        }

        if current_emotion_intensity >= self.emotion_threshold:
            logger.warning(
                f"🎭 [EVENT TRIGGER] Emotion intensity spike detected: {current_emotion_intensity:.2f} >= {self.emotion_threshold}"
            )
            return True, f"emotion_spike:{current_emotion_intensity:.2f}"

        pain_change = current_pain - last_pain
        if pain_change >= self.pain_threshold:
            logger.warning(
                f"💢 [EVENT TRIGGER] Pain surge detected: +{pain_change:.2f} (from {last_pain:.2f} to {current_pain:.2f})"
            )
            return True, f"pain_surge:+{pain_change:.2f}"

        energy_drop = last_energy - current_energy
        if energy_drop >= self.energy_drop_threshold:
            logger.warning(
                f"⚡ [EVENT TRIGGER] Energy crash detected: -{energy_drop:.1f} (from {last_energy:.1f} to {current_energy:.1f})"
            )
            return True, f"energy_crash:-{energy_drop:.1f}"

        if current_pain > 0.5 and current_energy < 30.0:
            logger.warning(
                f"⚠️ [EVENT TRIGGER] Combined crisis: pain={current_pain:.2f} + low_energy={current_energy:.1f}"
            )
            return True, f"combined_crisis:pain{current_pain:.2f}_energy{current_energy:.1f}"

        emotion_change = abs(current_emotion_intensity - last_emotion_intensity)
        if emotion_change >= self.emotion_volatility_threshold:
            logger.warning(
                f"🌊 [EVENT TRIGGER] Emotional volatility: change={emotion_change:.2f} "
                f"(from {last_emotion_intensity:.2f} to {current_emotion_intensity:.2f})"
            )
            return True, f"emotion_volatility:{emotion_change:.2f}"

        if current_pain >= self.pain_absolute_threshold:
            logger.warning(
                f"😣 [EVENT TRIGGER] High pain detected: {current_pain:.2f} >= {self.pain_absolute_threshold}"
            )
            return True, f"high_pain:{current_pain:.2f}"

        if current_energy <= self.energy_critical_threshold:
            logger.warning(
                f"🔋 [EVENT TRIGGER] Critical energy: {current_energy:.1f} <= {self.energy_critical_threshold}"
            )
            return True, f"critical_energy:{current_energy:.1f}"

        return False, None

    def reset_state(self, session_id: str):
        """Clear cached deltas (e.g., after rollback or session reset)."""
        if session_id in self.last_state:
            del self.last_state[session_id]
            logger.info(f"Event trigger state reset for session: {session_id}")

    def get_statistics(self, session_id: str) -> Dict:
        """Expose the last stored snapshot for diagnostics."""
        state = self.last_state.get(session_id, {})
        return {
            "last_pain": state.get("pain", 0.0),
            "last_energy": state.get("energy", 100.0),
            "last_emotion_intensity": state.get("emotion_intensity", 0.0),
            "has_baseline": session_id in self.last_state
        }


_event_trigger_instance: Optional[EventTriggeredSelfTickManager] = None


def get_event_trigger_manager() -> EventTriggeredSelfTickManager:
    """Process-wide singleton for event-triggered ticks."""
    global _event_trigger_instance
    if _event_trigger_instance is None:
        _event_trigger_instance = EventTriggeredSelfTickManager()
    return _event_trigger_instance
