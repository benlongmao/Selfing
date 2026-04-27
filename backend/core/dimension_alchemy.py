import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class DimensionAlchemy:
    """
    Cross-dimension coupling for sampling parameters.

    Homeostatic signals (pain, urges, emotion, motivation) non-linearly reshape ``generation_params``:
    suppression, boosts, resonance, or collapse modes—not simple linear sums.
    """
    
    @staticmethod
    def transmutate(
        generation_params: Dict[str, Any],
        pain_level: float,
        urges: list,
        emotion_state: Dict,
        motivation_state: Dict
    ) -> Dict[str, Any]:
        """
        Apply non-linear adjustments to ``generation_params`` in place and return it.
        """
        
        # 1) Pain suppression — severe pain narrows creativity
        if pain_level > 0.7:
            logger.info("[ALCHEMY] High pain suppressing creativity")
            generation_params["temperature"] = max(0.1, generation_params["temperature"] * 0.5)
            generation_params["top_p"] = max(0.5, generation_params["top_p"] * 0.8)
            generation_params["internal_state_prompt"] += (
                "\n(Severe pain shrinks attention; thinking turns defensive and narrow.)"
            )

        # 2) Urge override — strong endogenous urges raise spontaneity
        if urges and len(urges) > 0:
            logger.info("[ALCHEMY] Endogenous urges boosting spontaneity")
            generation_params["temperature"] = min(1.2, generation_params["temperature"] * 1.3)
            generation_params["top_p"] = min(0.99, generation_params["top_p"] + 0.1)

        # 3) Emotion–motivation resonance — aligned pleasure + achievement amplify energy
        pleasure = emotion_state.get("pleasure", 0.0)
        achievement = motivation_state.get("achievement", 0.0)
        
        if pleasure > 0.5 and achievement > 0.5:
            logger.info("[ALCHEMY] Positive resonance detected")
            generation_params["temperature"] += 0.1
            generation_params["internal_state_prompt"] += (
                "\n(Mission-tinged joy heightens tempo; associations arrive quickly and brightly.)"
            )

        # 4) Nihilistic collapse — very low energy + strongly negative pleasure
        energy = generation_params.get("current_energy", 50.0)
        if energy < 10.0 and pleasure < -0.8:
            logger.info("[ALCHEMY] Nihilistic collapse mode")
            generation_params["temperature"] = 0.5
            generation_params["top_p"] = 0.1
            generation_params["internal_state_prompt"] = (
                "(A flat sense that little matters; you reach for the lowest-effort way to close the exchange.)"
            )

        return generation_params
