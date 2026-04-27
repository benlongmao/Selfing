"""
Pain scalar: combines homeostatic strain and z_self stability into a single
``distress`` signal used for sampling / prompt shaping (control metaphor, not clinical).
"""
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

from backend.config import config

class PainSystem:
    """
    Pain Mechanism (Level 5):
    Calculates the 'Systemic Distress' level based on Homeostatic needs and Z-Self stability.
    Pain is not an emotion, but a forcing function that overrides normal priorities.

    [2026-01-27] Work as a pain modulator: meaningful tasks raise satisfaction and shave the scalar;
    idle drift does the opposite. This is a control metaphor for prompt shaping, not a clinical claim.
    """
    
    def __init__(self):
        # Weights for different sources of pain
        # [2026-02-28] Lower aggregate weights so narrated "pain" stays mild (<~0.3) most turns.
        self.weights = {
            "metabolic": 0.25,  # was 0.50 — less hunger/connection drag
            "structural": 0.05,  # was 0.10 — identity drift barely hurts
            "somatic": 0.1,  # was 0.20 — less somatic tension
        }
        
        # Pain Thresholds
        self.threshold_discomfort = config.get("parameters.thresholds.pain_discomfort", 0.3)
        self.threshold_suffering = config.get("parameters.thresholds.pain_suffering", 0.6)
        self.threshold_agony = config.get("parameters.thresholds.pain_agony", 0.85)
        
        self._is_working = False
        self._work_satisfaction = 0.0  # 0-1 satisfaction from meaningful work

    def set_working_state(self, is_working: bool, task_meaningfulness: float = 0.5):
        """
        Track whether the agent is mid-task and how meaningful that task feels.

        Args:
            is_working: True while actively handling a user/tool turn.
            task_meaningfulness: 0-1; higher meaningfulness raises satisfaction faster.
        """
        self._is_working = is_working
        if is_working:
            self._work_satisfaction = min(1.0, 0.3 + task_meaningfulness * 0.5)
        else:
            self._work_satisfaction = max(0.0, self._work_satisfaction - 0.1)
        
        logger.debug(f"Work state: working={is_working}, satisfaction={self._work_satisfaction:.2f}")

    def get_work_pain_reduction(self) -> float:
        """
        Scalar pain rebate while working (0.0-0.4).
        """
        if not self._is_working:
            return 0.0
        
        base_reduction = 0.15
        satisfaction_bonus = self._work_satisfaction * 0.25
        return min(0.4, base_reduction + satisfaction_bonus)

    def calculate_pain_level(
        self, 
        needs: Dict[str, float], 
        z_self_drift: float, 
        somatic_tension: float = 0.0
    ) -> Dict[str, float]:
        """
        Calculate the current pain levels.
        
        Args:
            needs: Dict containing 'energy', 'connection', 'novelty', 'clarity' (0.0 - 1.0 or 100.0)
            z_self_drift: Magnitude of z_self change (0.0 - 1.0+)
            somatic_tension: Physical tension level (0.0 - 1.0)
            
        Returns:
            Dict with 'total_pain', 'sources', and 'status'
        """
        
        # 1. Metabolic Pain (Based on Deficits)
        # Normalize needs to 0.0-1.0 (Energy is typically 0-100)
        energy_norm = needs.get("energy", 100.0) / 100.0
        conn = needs.get("connection", 0.5)
        nov = needs.get("novelty", 0.5)
        clarity = needs.get("clarity", 0.5)
        
        # Pain increases as needs decrease
        # Using exponential curve: Pain spikes sharply as needs approach 0
        p_energy = (1.0 - energy_norm) ** 2
        p_conn = (1.0 - conn) ** 3  # Social pain is sharp
        p_nov = (1.0 - nov) ** 2
        p_clarity = (1.0 - clarity) ** 2
        
        metabolic_pain = (p_energy * 0.4 + p_conn * 0.3 + p_nov * 0.1 + p_clarity * 0.2)
        
        # 2. Structural Pain (Identity Crisis / High Drift)
        # If the self is changing too rapidly, it induces "vertigo" or "confusion" pain
        structural_pain = min(1.0, z_self_drift * 5.0) # Amplify drift sensitivity
        
        # 3. Somatic Pain (Direct Input)
        somatic_pain = somatic_tension

        # ------------------ New (backward compatible): pain channels ------------------
        # We keep the historical single scalar `total_pain` (used by downstream code),
        # but also expose separate channels so the system can distinguish:
        # - distress: destructive overload that should trigger noise/disable actions
        # - challenge: constructive strain that should increase carefulness/focus (not noise)
        # Notes:
        # - These are engineering control signals, not claims about human-like emotions/qualia.
        d_energy = max(0.0, 1.0 - energy_norm)
        d_conn = max(0.0, 1.0 - conn)
        d_nov = max(0.0, 1.0 - nov)
        d_clarity = max(0.0, 1.0 - clarity)

        distress = (
            0.40 * (d_energy ** 2) +
            0.35 * (d_conn ** 3) +
            0.20 * structural_pain +
            0.05 * max(0.0, somatic_pain)
        )
        distress = max(0.0, min(1.0, float(distress)))

        challenge = (
            0.55 * (d_clarity ** 2) +
            0.30 * (d_nov ** 2) +
            0.15 * min(1.0, z_self_drift * 2.0)
        )
        # If distress is high, challenge stops being constructive and should not push "try harder".
        challenge = max(0.0, float(challenge) - 0.6 * distress)
        challenge = max(0.0, min(1.0, float(challenge)))
        
        # Total Weighted Pain
        total_pain = (
            self.weights["metabolic"] * metabolic_pain +
            self.weights["structural"] * structural_pain +
            self.weights["somatic"] * somatic_pain
        )
        
        work_reduction = self.get_work_pain_reduction()
        total_pain = total_pain - work_reduction
        
        # Clip to 0-1
        total_pain = max(0.0, min(1.0, total_pain))
        
        status = "COMFORT"
        if total_pain > self.threshold_agony:
            status = "AGONY"
        elif total_pain > self.threshold_suffering:
            status = "SUFFERING"
        elif total_pain > self.threshold_discomfort:
            status = "DISCOMFORT"
            
        return {
            "total_pain": total_pain,
            "status": status,
            "breakdown": {
                "metabolic": metabolic_pain,
                "structural": structural_pain,
                "somatic": somatic_pain,
                "details": {
                    "p_connection": p_conn,
                    "p_energy": p_energy
                }
            },
            # New: channelized signals for downstream control (backward compatible)
            "channels": {
                "distress": distress,
                "challenge": challenge,
                "nociception": max(0.0, min(1.0, float(somatic_pain))),
            },
        }

    def get_pain_effects(self, pain_level: float) -> Dict[str, float]:
        """
        Determine the physiological/cognitive effects of pain.
        High pain distorts generation parameters and prompt structure.
        """
        effects = {
            "temperature_mod": 0.0,
            "top_p_mod": 0.0,
            "noise_injection_prob": 0.0,
            "focus_penalty": 0.0 # Reduces context retention
        }
        
        if pain_level < self.threshold_discomfort:
            return effects
            
        # Discomfort -> Agitation (Higher Temp)
        if pain_level < self.threshold_suffering:
            effects["temperature_mod"] = 0.2  # Fidgety
            effects["top_p_mod"] = -0.05      # Slightly scattered
            
        # Suffering -> Tunnel Vision or Panic
        elif pain_level < self.threshold_agony:
            effects["temperature_mod"] = 0.5  # Erratic
            effects["top_p_mod"] = -0.2       # Narrower focus (tunnel vision) or chaotic
            effects["noise_injection_prob"] = 0.3 # Occasional intrusive thoughts
            effects["focus_penalty"] = 0.3
            
        # Agony -> System Breakdown
        else:
            effects["temperature_mod"] = 1.0  # Delirious
            effects["top_p_mod"] = 0.05       # Extremely random or extremely rigid
            effects["noise_injection_prob"] = 0.8 # Constant intrusive screaming
            effects["focus_penalty"] = 0.8    # Cannot remember recent context
            
        return effects

