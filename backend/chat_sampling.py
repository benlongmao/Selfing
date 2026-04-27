import math
import logging
from typing import Dict, Optional, Tuple

from backend.self_model import SelfModel
from backend.z_self_influence import get_z_self_influence

logger = logging.getLogger(__name__)


def compute_sampling_and_mode(
    self_model: Optional[SelfModel],
    session_id: str,
    base_temperature: float = 0.6,
    base_top_p: float = 0.95,
) -> Tuple[float, float, float, float, Optional[Dict], str, Dict]:
    """
    Derive decoding knobs plus ``interaction_mode`` / ``z_self`` summary blobs.

    Returns:
        ``(temperature, top_p, presence_penalty, frequency_penalty, interaction_mode,
        z_self_summary, generation_params)``

    [2026-02-02] Blends ``SelfModel.compute_generation_params`` with ``z_self_influence``:
    - legacy swing: about ±0.1 on temperature
    - influence swing: up to ~±0.3 from arousal / energy / uncertainty
    """
    if self_model is None:
        return base_temperature, base_top_p, 0.4, 0.2, None, "", {}

    z_self_summary = self_model.get_summary(session_id)
    generation_params = self_model.compute_generation_params(
        session_id,
        base_temperature=base_temperature,
        base_top_p=base_top_p,
    )

    def _to_float(val, default):
        try:
            return float(val)
        except Exception:
            return default

    temperature = _to_float(generation_params.get("temperature", base_temperature), base_temperature)
    top_p = _to_float(generation_params.get("top_p", base_top_p), base_top_p)
    presence_penalty = _to_float(generation_params.get("presence_penalty", 0.4), 0.4)
    frequency_penalty = _to_float(generation_params.get("frequency_penalty", 0.2), 0.2)

    # [2026-02-02] Blend physiology-driven influence with baseline params (50/50)
    try:
        z_self = self_model.get_z_self(session_id)
        if z_self is not None:
            influence = get_z_self_influence(z_self)
            influence_temp = influence.get("temperature", base_temperature)
            influence_top_p = influence.get("top_p", base_top_p)

            original_temp = temperature
            original_top_p = top_p

            temperature = 0.5 * original_temp + 0.5 * influence_temp
            top_p = 0.5 * original_top_p + 0.5 * influence_top_p

            logger.debug(
                f"[z_self Influence] Sampling adjusted: "
                f"temp {original_temp:.3f}→{temperature:.3f}, "
                f"top_p {original_top_p:.3f}→{top_p:.3f}"
            )
    except Exception as e:
        logger.debug(f"Failed to apply z_self influence to sampling: {e}")

    if math.isnan(temperature) or math.isinf(temperature):
        temperature = 0.7
    if math.isnan(top_p) or math.isinf(top_p):
        top_p = 0.95
    if math.isnan(presence_penalty) or math.isinf(presence_penalty):
        presence_penalty = 0.4
    if math.isnan(frequency_penalty) or math.isinf(frequency_penalty):
        frequency_penalty = 0.2

    temperature = max(0.3, min(1.2, temperature))
    top_p = max(0.7, min(1.0, top_p))

    try:
        interaction_mode = self_model.decide_interaction_mode(session_id)
    except Exception:
        interaction_mode = None

    return (
        temperature,
        top_p,
        presence_penalty,
        frequency_penalty,
        interaction_mode,
        z_self_summary,
        generation_params,
    )
