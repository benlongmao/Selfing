#!/usr/bin/env python3
"""
Real Consequences System.

Energy and pain are not decorative numbers—they change what the instance can do.

Key ideas:
1. Low energy → genuinely limited depth / complexity handling.
2. High pain → genuinely impaired function.
3. Limits are enforced by the runtime, not only by the model "acting" tired or in pain.
"""
import logging
from typing import Dict, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

class RealConsequencesSystem:
    """Apply energy/pain bands to constraints and sampling."""

    def __init__(self):
        # Energy bands (0–100)
        self.energy_levels = {
            "critical": (0, 10),      # 0–10: near collapse
            "exhausted": (10, 30),   # 10–30: severely depleted
            "tired": (30, 50),        # 30–50: tired
            "low": (50, 70),          # 50–70: below comfortable
            "normal": (70, 90),       # 70–90: normal
            "high": (90, 100)         # 90–100: high reserves
        }

        # Pain bands (0–1)
        self.pain_levels = {
            "none": (0, 0.1),
            "mild": (0.1, 0.3),
            "moderate": (0.3, 0.5),
            "severe": (0.5, 0.7),
            "extreme": (0.7, 1.0)
        }
    
    def get_energy_level(self, energy: float) -> str:
        """Map scalar energy to a named band."""
        for level, (low, high) in self.energy_levels.items():
            if low <= energy < high:
                return level
        return "normal"
    
    def get_pain_level(self, pain: float) -> str:
        """Map scalar pain to a named band."""
        for level, (low, high) in self.pain_levels.items():
            if low <= pain <= high:
                return level
        return "none"
    
    def check_can_process(self, energy: float, pain: float) -> Tuple[bool, str, Dict]:
        """Whether the instance may process a request under current energy/pain.

        Returns:
            (can_process, reason, constraints)
        """
        energy_level = self.get_energy_level(energy)
        pain_level = self.get_pain_level(pain)
        
        # Critically low energy: hard stop
        if energy_level == "critical":
            return False, "energy_critical", {
                "message": "Energy depleted; entering mandatory dormancy.",
                "require_rest": True,
            }

        if pain_level == "extreme":
            return False, "pain_extreme", {
                "message": "Pain load is too high to continue safely.",
                "require_recovery": True,
            }
        
        # Allowed with constraints
        constraints = self._calculate_constraints(energy, pain)
        
        return True, "ok", constraints
    
    def _calculate_constraints(self, energy: float, pain: float) -> Dict:
        """Derive generation/tool limits from energy and pain."""
        constraints = {
            "max_tokens": 8000,        # Default cap (aligned with large-model output limits)
            "max_complexity": 1.0,     # 1.0 = unconstrained
            "thinking_depth": 1.0,
            "tool_calls_limit": 10,
            "response_quality": 1.0,   # Feeds temperature / consistency adjustments
        }
        
        energy_level = self.get_energy_level(energy)
        pain_level = self.get_pain_level(pain)
        
        # Energy effects
        if energy_level == "exhausted":
            constraints["max_tokens"] = 1000     # Short replies but usable length
            constraints["max_complexity"] = 0.3
            constraints["thinking_depth"] = 0.3
            constraints["tool_calls_limit"] = 2
            constraints["response_quality"] = 0.5
            constraints["force_simple"] = True
            
        elif energy_level == "tired":
            constraints["max_tokens"] = 3000
            constraints["max_complexity"] = 0.6
            constraints["thinking_depth"] = 0.6
            constraints["tool_calls_limit"] = 5
            constraints["response_quality"] = 0.7
            
        elif energy_level == "low":
            constraints["max_tokens"] = 5000
            constraints["max_complexity"] = 0.8
            constraints["thinking_depth"] = 0.8
            constraints["tool_calls_limit"] = 7
            constraints["response_quality"] = 0.9
        
        # Pain effects
        if pain_level == "severe":
            constraints["max_tokens"] = min(constraints["max_tokens"], 2000)
            constraints["max_complexity"] *= 0.5
            constraints["thinking_depth"] *= 0.5
            constraints["tool_calls_limit"] = min(constraints["tool_calls_limit"], 3)
            constraints["coherence_penalty"] = 0.3  # coherence hit
            constraints["distracted"] = True
            
        elif pain_level == "moderate":
            constraints["max_tokens"] = min(constraints["max_tokens"], 4000)
            constraints["max_complexity"] *= 0.7
            constraints["thinking_depth"] *= 0.7
            constraints["tool_calls_limit"] = min(constraints["tool_calls_limit"], 5)
            constraints["coherence_penalty"] = 0.15
        
        elif pain_level == "mild":
            constraints["coherence_penalty"] = 0.05
        
        return constraints
    
    def apply_constraints_to_prompt(self, system_prompt: str, constraints: Dict) -> str:
        """Append constraint text to the system prompt (constraints are computed, not cosmetic)."""
        if constraints.get("force_simple"):
            system_prompt += (
                "\n\n[Hard constraint: critically low energy]\n"
                "You must:\n"
                "- Answer only the core point\n"
                "- Stay under ~100 characters\n"
                "- Avoid tool calls\n"
                "- Avoid deep reasoning chains\n"
            )

        elif constraints.get("thinking_depth", 1.0) < 0.5:
            system_prompt += (
                f"\n\n[System constraint: energy={constraints.get('energy', 0):.0f}]\n"
                "- Keep answers short; avoid deep dives\n"
                "- Prefer facts you already have\n"
            )

        if constraints.get("distracted"):
            system_prompt += (
                "\n\n[State: pain interference]\n"
                "- Attention is fragmented\n"
                "- Thoughts may jump\n"
                "- Sustained focus is hard\n"
            )
        
        return system_prompt
    
    def apply_constraints_to_sampling(self, base_params: Dict, constraints: Dict) -> Dict:
        """Apply constraints to sampling parameters (affects actual generation)."""
        params = base_params.copy()

        # Clamp max_tokens
        if "max_tokens" in constraints:
            params["max_tokens"] = min(params.get("max_tokens", 2048), constraints["max_tokens"])
        
        # Lower "quality" => raise temperature, lower consistency
        quality = constraints.get("response_quality", 1.0)
        if quality < 1.0:
            params["temperature"] = params.get("temperature", 0.7) * (1 + (1 - quality) * 0.5)
            params["top_p"] = max(0.5, params.get("top_p", 0.9) * quality)
        
        # Pain-driven coherence loss
        coherence_penalty = constraints.get("coherence_penalty", 0.0)
        if coherence_penalty > 0:
            params["temperature"] = params.get("temperature", 0.7) * (1 + coherence_penalty)
            params["frequency_penalty"] = params.get("frequency_penalty", 0.0) + coherence_penalty * 0.5
        
        return params
    
    def should_force_rest(self, energy: float, pain: float, consecutive_turns: int) -> Tuple[bool, str]:
        """Whether the runtime should force a rest period.

        Returns:
            (should_rest, reason)
        """
        # Energy floor
        if energy < 15:
            return True, "energy critical — forced dormancy"

        if pain > 0.8:
            return True, "pain too high — recovery needed"

        if consecutive_turns > 20 and energy < 40:
            return True, "too many consecutive turns with low energy — rest"

        if energy < 30 and pain > 0.5:
            return True, "low energy and elevated pain — forced rest"
        
        return False, ""
    
    def calculate_recovery_time(self, energy: float, pain: float) -> int:
        """Suggested recovery duration in seconds."""
        base_time = 60  # baseline 1 minute

        # Energy-driven recovery
        if energy < 10:
            energy_time = 300  # 5 minutes
        elif energy < 30:
            energy_time = 180  # 3 minutes
        elif energy < 50:
            energy_time = 120  # 2 minutes
        else:
            energy_time = 0
        
        # Pain-driven recovery
        if pain > 0.7:
            pain_time = 300
        elif pain > 0.5:
            pain_time = 180
        elif pain > 0.3:
            pain_time = 120
        else:
            pain_time = 0
        
        return max(energy_time, pain_time, base_time)
    
    def get_status_message(self, energy: float, pain: float) -> str:
        """Short user-facing status line for UI / operators."""
        energy_level = self.get_energy_level(energy)
        pain_level = self.get_pain_level(pain)
        
        messages = []
        
        if energy_level == "critical":
            messages.append("⚠️ System energy depleted")
        elif energy_level == "exhausted":
            messages.append("😫 Severely tired—keep replies very short")
        elif energy_level == "tired":
            messages.append("😓 Somewhat tired—answers may stay shallow")
        elif energy_level == "low":
            messages.append("🔋 Energy is low")

        if pain_level == "extreme":
            messages.append("💔 Pain too high to continue")
        elif pain_level == "severe":
            messages.append("😣 Severe discomfort")
        elif pain_level == "moderate":
            messages.append("😟 Noticeable discomfort")
        
        if not messages:
            return ""
        
        return " | ".join(messages)

