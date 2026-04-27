import math
from typing import Dict

import numpy as np

from backend.self_model_summary import generate_internal_state_prompt as _generate_internal_state_prompt_helper


def compute_generation_params(
    self_model,
    session_id: str,
    base_temperature: float = 0.3,
    base_top_p: float = 0.95,
) -> Dict[str, float]:
    """
    Standalone ``compute_generation_params`` helper (avoids recursive ``SelfModel`` calls).
    """
    from backend import self_model as sm

    z_self = self_model.get_z_self(session_id)
    if z_self is None:
        return {
            "temperature": base_temperature,
            "top_p": base_top_p,
            "internal_state_prompt": "",
        }

    temperature = base_temperature
    top_p = base_top_p

    # Big Five slice (z_self[0:32]) nudges sampling knobs
    from backend.personality_store import PERSONALITY_SUBSPACE_DIMS as P_DIMS
    if z_self.shape[0] >= 32:
        o_mean = float(math.fsum(z_self[P_DIMS["openness"][0]:P_DIMS["openness"][1]]) / 8.0)
        c_mean = float(math.fsum(z_self[P_DIMS["conscientiousness"][0]:P_DIMS["conscientiousness"][1]]) / 8.0)
        e_mean = float(math.fsum(z_self[P_DIMS["extraversion"][0]:P_DIMS["extraversion"][1]]) / 8.0)
        n_mean = float(math.fsum(z_self[P_DIMS["neuroticism"][0]:P_DIMS["neuroticism"][1]]) / 8.0)

        temperature += -n_mean * 0.15
        top_p += -o_mean * 0.08
        temperature += e_mean * 0.12
        temperature += -c_mean * 0.10
        top_p += -c_mean * 0.05

    if self_model.dim >= sm.RULES_DIM + sm.EMOTION_DIM and z_self.shape[0] >= sm.RULES_DIM + sm.EMOTION_DIM:
        emotion_vec = z_self[sm.RULES_DIM: sm.RULES_DIM + sm.EMOTION_DIM]
        from backend.emotion_store import EMOTION_SUBSPACE_DIMS
        pleasure = float(math.fsum(emotion_vec[EMOTION_SUBSPACE_DIMS["pleasure"][0]:EMOTION_SUBSPACE_DIMS["pleasure"][1]]) / 4.0)
        temperature += pleasure * 0.1

    if self_model.dim >= sm.RULES_DIM + sm.EMOTION_DIM + sm.MOTIVATION_DIM and z_self.shape[0] >= sm.RULES_DIM + sm.EMOTION_DIM + sm.MOTIVATION_DIM:
        motivation_vec = z_self[sm.RULES_DIM + sm.EMOTION_DIM: sm.RULES_DIM + sm.EMOTION_DIM + sm.MOTIVATION_DIM]
        motivation_strength = float(math.fsum(map(abs, motivation_vec)) / len(motivation_vec))
        top_p += motivation_strength * 0.05

    if self_model.somatic_store and self_model.dim >= sm.SOMATIC_START_IDX + sm.SOMATIC_DIM:
        start_idx = sm.SOMATIC_START_IDX
        somatic_vec = z_self[start_idx:start_idx + sm.SOMATIC_DIM]
        # Matches ``SOMATIC_SUBSPACE_DIMS`` / ``sync_somatic_to_z_self`` / ``get_structured_summary``:
        # 16-D z_self layout = energy[0:4], viscosity[4:8], tension/pain[8:12], vitality[12:16]
        tension = float(math.fsum(somatic_vec[8:12]) / 4.0)
        vitality = float(math.fsum(somatic_vec[12:16]) / 4.0)
        viscosity = float(math.fsum(somatic_vec[4:8]) / 4.0)

        if tension > 0.4:
            reduction = min(0.4, (tension - 0.4) * 0.8)
            top_p -= reduction

        if vitality < -0.4:
            temperature -= 0.10
        elif vitality > 0.6:
            temperature += 0.1

        if viscosity > 0.65:
            top_p -= 0.08

    top_p = max(0.5, min(1.0, top_p))

    try:
        current_energy = self_model.get_energy(session_id)
        current_needs = self_model.update_needs(session_id, interaction_type="check")
        connection = current_needs.get("connection", 0.5)
        clarity = current_needs.get("clarity", 0.5)
        pressure = 1.0 - (connection + clarity) / 2.0
        is_stressed = current_energy < 30.0 or pressure > 0.7
        if is_stressed:
            temperature += 0.3 if current_energy < 20.0 else 0.15
            top_p += -0.2 if pressure > 0.7 else -0.1
    except Exception as e:
        sm.logger.warning(f"Failed to check stress state: {e}")

    temperature = float(max(0.1, min(1.0, temperature)))
    top_p = float(max(0.5, min(1.0, top_p)))

    current_energy = self_model.get_energy(session_id) if hasattr(self_model, "get_energy") else 100.0

    try:
        pain_status = self_model.get_pain_status(session_id)
    except Exception:
        pain_status = {}

    internal_state_prompt = _generate_internal_state_prompt_helper(
        self_model,
        session_id,
        z_self=z_self,
        energy=current_energy,
        pain_status=pain_status,
        system_entropy=0.0,
        noise_perturbation=0.0,
        hide_numbers=True,
    )

    pain_level = pain_status.get("total_pain", 0.0)
    pain_effects = self_model.pain_system.get_pain_effects(pain_level)
    if pain_level > 0.3:
        temperature += pain_effects.get("temperature_mod", 0.0)
        top_p += pain_effects.get("top_p_mod", 0.0)

    noise_perturbation = 0.0
    if self_model.noise_perturbator.check_spontaneous_event(0.05):
        fluctuation = self_model.noise_perturbator.generate_fluctuation(0.2)
        temperature += fluctuation
        noise_perturbation = fluctuation

    # z_self[64:72]: WorldStore global aggregate (not PCA); skip when flat legacy rows
    try:
        if z_self is not None and z_self.shape[0] >= 72:
            slab = z_self[64:72]
            if float(np.max(np.abs(slab))) > 1e-5:
                opt_z = float(np.mean(slab[0:4]))
                ag_z = float(np.mean(slab[4:8]))
                temperature += (opt_z - 0.5) * 0.05
                top_p += (ag_z - 0.5) * 0.03
    except Exception:
        pass

    return {
        "temperature": float(max(0.1, min(1.0, temperature))),
        "top_p": float(max(0.5, min(1.0, top_p))),
        "internal_state_prompt": internal_state_prompt,
        "pain_level": float(pain_level),
        "noise_injection_prob": float(pain_effects.get("noise_injection_prob", 0.0)),
        "system_entropy": 0.0,
        "system_age_ticks": int(self_model._get_tick(session_id) if hasattr(self_model, "_get_tick") else 0),
        "noise_perturbation": float(noise_perturbation),
    }
