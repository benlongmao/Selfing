#!/usr/bin/env python3
"""
Attention mechanism ("spotlight") for the agent runtime.

Tracks a coarse focus type and intensity, maps slices of ``z_self`` into
attention parameters, and simulates bottom-up capture plus slow drift.
"""

import logging
import numpy as np
from typing import Optional
from dataclasses import dataclass
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# High-level focus channels (orthogonal to z_self layout)
FOCUS_TYPE_EXTERNAL = "external"  # user / environment input
FOCUS_TYPE_INTERNAL = "internal"  # self-directed reasoning
FOCUS_TYPE_MEMORY = "memory"  # episodic recall
FOCUS_TYPE_SOMATIC = "somatic"  # interoception / affect


@dataclass
class AttentionState:
    focus_type: str
    focus_object: str  # short token describing the object of focus
    intensity: float  # 0.0 - 1.0
    duration: int  # ticks spent in this micro-state
    last_updated: str


class AttentionMechanism:
    def __init__(self, db_path: str = "data.db"):
        self.db_path = db_path
        # Ephemeral runtime state (not persisted; changes every tick)
        self.current_state = AttentionState(
            focus_type=FOCUS_TYPE_EXTERNAL,
            focus_object="waiting_for_input",
            intensity=0.5,
            duration=0,
            last_updated=datetime.now(timezone.utc).isoformat(),
        )

        # z_self slice layout — keep aligned with SelfModel
        # focus: (88, 92) -> concentration vs diffusion
        # direction: (92, 96) -> inward (self) vs outward (input); >0 leans internal
        self.FOCUS_INDICES = (88, 92)
        self.DIRECTION_INDICES = (92, 96)

    def update_attention(
        self,
        z_self: np.ndarray,
        event_type: Optional[str] = None,
        event_content: Optional[str] = None,
    ) -> AttentionState:
        """
        Update attention from ``z_self`` plus an optional discrete event.

        Args:
            z_self: Full subjective state vector
            event_type: e.g. ``user_message``, ``system_alert``
            event_content: Optional payload for logging / somatic hooks

        Returns:
            Updated ``AttentionState`` snapshot
        """
        focus_vec = z_self[self.FOCUS_INDICES[0] : self.FOCUS_INDICES[1]]
        direction_vec = z_self[self.DIRECTION_INDICES[0] : self.DIRECTION_INDICES[1]]

        focus_level = float(np.mean(focus_vec))  # >0 focused, <0 diffuse
        direction_level = float(np.mean(direction_vec))  # >0 inward, <0 outward

        # Bottom-up capture
        if event_type == "user_message":
            if direction_level < 0.8:
                self.current_state.focus_type = FOCUS_TYPE_EXTERNAL
                self.current_state.focus_object = "user_input"
                self.current_state.intensity = 0.9
                self.current_state.duration = 0
                return self.current_state

        elif event_type == "system_alert":
            self.current_state.focus_type = FOCUS_TYPE_SOMATIC
            self.current_state.focus_object = event_content or "alert"
            self.current_state.intensity = 1.0
            self.current_state.duration = 0
            return self.current_state

        # Top-down regulation / drift when no salient event
        self.current_state.duration += 1

        decay = 0.05 * self.current_state.duration
        self.current_state.intensity = max(0.1, self.current_state.intensity - decay)

        drift_threshold = 0.3
        if focus_level < -0.2:  # highly diffuse mind-wandering bias
            drift_threshold = 0.6

        if self.current_state.intensity < drift_threshold:
            self._drift_attention(direction_level, focus_level)

        self.current_state.last_updated = datetime.now(timezone.utc).isoformat()
        return self.current_state

    def _drift_attention(self, direction_level: float, focus_level: float):
        """Randomly re-sample focus when intensity falls below threshold."""
        import random

        prob_internal = (np.clip(direction_level, -1.0, 1.0) + 1.0) / 2.0

        if random.random() < prob_internal:
            options = [FOCUS_TYPE_INTERNAL, FOCUS_TYPE_MEMORY, FOCUS_TYPE_SOMATIC]
            weights = [0.5, 0.3, 0.2]

            new_type = random.choices(options, weights=weights)[0]
            self.current_state.focus_type = new_type

            if new_type == FOCUS_TYPE_INTERNAL:
                self.current_state.focus_object = "internal_thought_process"
            elif new_type == FOCUS_TYPE_MEMORY:
                self.current_state.focus_object = "past_memories"
            elif new_type == FOCUS_TYPE_SOMATIC:
                self.current_state.focus_object = "body_sensation"

        else:
            self.current_state.focus_type = FOCUS_TYPE_EXTERNAL
            self.current_state.focus_object = "environment_scanning"

        base_intensity = 0.5 + (focus_level * 0.3)
        self.current_state.intensity = float(np.clip(base_intensity, 0.3, 0.9))
        self.current_state.duration = 0

        logger.info(
            f"Attention drifted to: {self.current_state.focus_type} ({self.current_state.focus_object})"
        )

    def get_attention_description(self) -> str:
        """Short natural-language line for prompts / introspection."""
        s = self.current_state

        desc = ""
        if s.focus_type == FOCUS_TYPE_EXTERNAL:
            if s.focus_object == "user_input":
                desc = "Locked onto the user's message."
            else:
                desc = "Keeping a wide, outward scan for environmental cues."
        elif s.focus_type == FOCUS_TYPE_INTERNAL:
            desc = "Attention turned inward, inspecting its own reasoning."
        elif s.focus_type == FOCUS_TYPE_MEMORY:
            desc = "Drifting through episodic memories."
        elif s.focus_type == FOCUS_TYPE_SOMATIC:
            desc = "Tuned into interoceptive / somatic signals."

        if s.intensity > 0.8:
            desc += " (high focus)"
        elif s.intensity < 0.3:
            desc += " (diffuse attention)"

        return desc
