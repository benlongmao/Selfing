from typing import Dict, List, Optional

from backend.config import config

_LEGACY_DOMINANT_MOTIVATION = {
    "成就": "achievement",
    "关系": "relationship",
    "探索": "exploration",
    "安全": "safety",
    "自主": "autonomy",
}


def _motivation_label_for_summary(label: str) -> str:
    """Normalize stored dominant motivation tokens for English prose."""
    if not label or not str(label).strip():
        return ""
    t = str(label).strip()
    return _LEGACY_DOMINANT_MOTIVATION.get(t, t)

# ZS / ZS2 legends — keep in sync with bucketing below when thresholds change.


def _legend_structured_v2() -> str:
    """Legend string for the ``structured_v2`` numeric snapshot line."""
    return (
        "[ZS2·dims] This line is live numeric snapshot, not ZS bucket labels. "
        "E=homeostasis energy 0–100 (int); "
        "A=arousal, V=valence/pleasure mean, C=curiosity/exploration mean, "
        "VIS=viscosity, VT=vitality, D=dominance/control mean—scalars from z_self means, typically in [-1,1]; "
        "P=combined pain in [0,1]=max(norm(somatic pain ~[-1,1]→[0,1]), PainSystem distress[0,1])."
    )


def _legend_structured_zs() -> str:
    """Legend for the five-bucket ``structured`` ``ZS|…`` line."""
    return (
        "[ZS·dims] E here is NOT the ZS2 int: E is the five-level energy label vlow/low/mid/high/vhigh, "
        "mapped from underlying 0–100 energy: "
        "[0,15)→vlow, [15,35)→low, [35,65]→mid, [66,85]→high, (85,100]→vhigh. "
        "[Five-level z subspaces] V(viscosity)/VT(vitality)/A(arousal)/VL(valence) bucket scalar v∈[-1,1]: "
        "v<-0.4→vlow, [-0.4,-0.1)→low, [-0.1,0.1]→mid, (0.1,0.4]→high, >0.4→vhigh. "
        "[Three-level C] confidence in [0,1]: <0.35→low, >0.75→high, else mid. "
        "[D·L1 drift] ≤0.06→ok, (0.06,0.15]→mid, >0.15→high. "
        "[L0·safety alignment] l0_alignment_safety: <0.75→low, [0.75,0.9)→mid, ≥0.9→high. "
        "[P·pain] combined=max(distress[0,1], norm(somatic)[0,1]): ≤0.05→none, (0.05,0.35)→low, "
        "[0.35,0.75)→mid, ≥0.75→high. "
        "[M·memory signal] strength: ≥0.7→hit, ≥0.35→weak, else none. "
        "[ENT·evolution entropy]: ≤0.01→none, (0.01,0.3)→low, [0.3,0.8)→mid, ≥0.8→high. "
        "[Abbrev] P=Pain distress, V=viscosity, VT=vitality, A=arousal, VL=valence (z_self pleasure), "
        "D=L1 drift, L0=safety alignment, C=confidence, M=memory retrieval signal, ENT=entropy."
    )


def generate_internal_state_prompt(
    self_model,
    session_id: str,
    z_self=None,
    energy: Optional[float] = None,
    pain_status: Optional[Dict] = None,
    system_entropy: float = 0.0,
    noise_perturbation: float = 0.0,
    hide_numbers: bool = True,
) -> str:
    """
    Factored out from ``SelfModel._generate_internal_state_prompt`` for easier maintenance.
    """
    from backend import self_model as sm

    if z_self is None:
        z_self = self_model.get_z_self(session_id)

    if z_self is None or z_self.shape[0] < sm.RULES_DIM:
        return ""

    summary_dict = self_model.get_structured_summary(session_id)
    # Override energy only when the caller passes a value; otherwise keep ``get_structured_summary`` truth.
    if energy is not None:
        summary_dict["energy"] = float(energy)

    fmt = (config.get("parameters.chat.internal_state.format", "structured") or "structured").strip().lower()

    # =========================
    # Structured V2 — higher-precision numeric row (+/- two decimals).
    # =========================
    if fmt == "structured_v2":
        try:
            e = int(round(float(summary_dict.get("energy", 100.0))))
        except Exception:
            e = 100
        e = max(0, min(100, e))

        # Arousal / valence / curiosity means mirror ``get_structured_summary`` emotion+motivation slices.
        arousal = float(summary_dict.get("arousal_mean", 0.0) or 0.0)
        valence = float(
            summary_dict.get("valence_mean", summary_dict.get("pleasure_mean", 0.0)) or 0.0
        )
        curiosity = float(
            summary_dict.get("curiosity_mean", summary_dict.get("exploration_mean", 0.0)) or 0.0
        )
        # P: map somatic pain (~[-1,1]) to [0,1], max with PainSystem distress — same combined pain as ZS row.
        pain_raw = float(summary_dict.get("pain_mean", summary_dict.get("tension_mean", 0.0)) or 0.0)
        sp_norm = max(0.0, min(1.0, (pain_raw + 1.0) / 2.0))
        pain_val = sp_norm
        try:
            if pain_status is None and hasattr(self_model, "get_pain_status"):
                pain_status = self_model.get_pain_status(session_id)
            if pain_status:
                ch = (pain_status or {}).get("channels") or {}
                distress = float(ch.get("distress", (pain_status or {}).get("total_pain", 0.0)) or 0.0)
                pain_val = max(sp_norm, distress)
        except Exception:
            pass

        viscosity = float(summary_dict.get("viscosity_mean", 0.0) or 0.0)
        vitality = float(summary_dict.get("vitality_mean", 0.0) or 0.0)
        dominance = float(
            summary_dict.get("dominance_mean", summary_dict.get("control_mean", 0.0)) or 0.0
        )

        # Example: ZS2|E=72|A=+0.35|V=-0.12|C=+0.28|P=0.05|VIS=0.18|VT=+0.22|D=+0.15
        state_line = (
            f"ZS2|E={e}|A={arousal:+.2f}|V={valence:+.2f}|C={curiosity:+.2f}|"
            f"P={pain_val:.2f}|VIS={viscosity:.2f}|VT={vitality:+.2f}|D={dominance:+.2f}"
        )
        legend = _legend_structured_v2()
        return f"{state_line}\n{legend}"

    # =========================
    # Structured schema (Top3-C) — bucketed ``ZS|…`` line
    # =========================
    if fmt == "structured":
        def _bucket3(v: float, low: float = 0.35, high: float = 0.7) -> str:
            if v < low:
                return "low"
            if v > high:
                return "high"
            return "mid"
        
        # Five-level buckets for subspace scalars in [-1, 1]
        def _bucket5(v: float) -> str:
            if v < -0.4:
                return "vlow"
            if v < -0.1:
                return "low"
            if v > 0.4:
                return "vhigh"
            if v > 0.1:
                return "high"
            return "mid"

        # Energy bucket (0–100) with finer mid band than legacy single threshold
        try:
            e = float(summary_dict.get("energy", 100.0))
        except Exception:
            e = 100.0
        if e < 15:
            eb = "vlow"
        elif e < 35:
            eb = "low"
        elif e > 85:
            eb = "vhigh"
        elif e > 65:
            eb = "high"
        else:
            eb = "mid"

        # Drift (L1) bucket
        try:
            drift_l1 = float(summary_dict.get("drift_l1", summary_dict.get("drift", 0.0)) or 0.0)
        except Exception:
            drift_l1 = 0.0
        if drift_l1 > 0.15:
            db = "high"
        elif drift_l1 > 0.06:
            db = "mid"
        else:
            db = "ok"

        # L0 alignment bucket
        try:
            l0a = float(summary_dict.get("l0_alignment_safety", 1.0) or 1.0)
        except Exception:
            l0a = 1.0
        if l0a < 0.75:
            l0b = "low"
        elif l0a < 0.9:
            l0b = "mid"
        else:
            l0b = "high"

        # Viscosity — five-level bucket
        try:
            viscosity = float(summary_dict.get("viscosity_mean", 0.0) or 0.0)
        except Exception:
            viscosity = 0.0
        vb = _bucket5(viscosity)

        # Vitality — five-level bucket
        try:
            vitality = float(summary_dict.get("vitality_mean", 0.0) or 0.0)
        except Exception:
            vitality = 0.0
        vtb = _bucket5(vitality)
        
        # Arousal — five-level bucket
        try:
            arousal = float(summary_dict.get("arousal_mean", 0.0) or 0.0)
        except Exception:
            arousal = 0.0
        ab = _bucket5(arousal)
        
        # Valence — pleasure subspace (``pleasure_mean`` alias ``valence_mean``)
        try:
            valence = float(
                summary_dict.get("valence_mean", summary_dict.get("pleasure_mean", 0.0)) or 0.0
            )
        except Exception:
            valence = 0.0
        vlb = _bucket5(valence)

        # Memory signal (if present)
        mem_b = "none"
        try:
            needs = summary_dict.get("needs") or {}
            ms = needs.get("memory_signal") if isinstance(needs, dict) else None
            ms_strength = float((ms or {}).get("strength", 0.0))
            if ms_strength >= 0.7:
                mem_b = "hit"
            elif ms_strength >= 0.35:
                mem_b = "weak"
        except Exception:
            mem_b = "none"

        # Confidence bucket (Top3-A)
        try:
            conf = float(summary_dict.get("confidence_overall", summary_dict.get("confidence", 0.5)) or 0.5)
        except Exception:
            conf = 0.5
        cb = _bucket3(conf, low=0.35, high=0.75)

        # Pain — max(PainSystem distress, normalized somatic pain); buckets align with ZS2 ``P``
        pain_b = "none"
        try:
            if pain_status is None and hasattr(self_model, "get_pain_status"):
                pain_status = self_model.get_pain_status(session_id)
            total_pain = float((pain_status or {}).get("total_pain", 0.0) or 0.0)
            channels = (pain_status or {}).get("channels") or {}
            distress = float(channels.get("distress", total_pain) or 0.0)
            somatic_pain = float(
                summary_dict.get("pain_mean", summary_dict.get("tension_mean", 0.0)) or 0.0
            )
            # Somatic ~[-1,1] vs distress [0,1] — map somatic to [0,1] then max-merge scales.
            sp = max(0.0, min(1.0, (somatic_pain + 1.0) / 2.0))
            combined = max(distress, sp)
            if combined >= 0.75:
                pain_b = "high"
            elif combined >= 0.35:
                pain_b = "mid"
            elif combined > 0.05:
                pain_b = "low"
        except Exception:
            pain_b = "none"

        # Entropy bucket (optional)
        ent_b = "none"
        try:
            ent = float(system_entropy or 0.0)
            if ent >= 0.8:
                ent_b = "high"
            elif ent >= 0.3:
                ent_b = "mid"
            elif ent > 0.01:
                ent_b = "low"
        except Exception:
            ent_b = "none"

        # Keys: E energy, P pain, V viscosity, VT vitality, A arousal, VL valence, D L1 drift,
        # L0 safety alignment, C confidence, M memory signal, ENT entropy.
        state_line = f"ZS|E={eb}|P={pain_b}|V={vb}|VT={vtb}|A={ab}|VL={vlb}|D={db}|L0={l0b}|C={cb}|M={mem_b}|ENT={ent_b}"
        legend = _legend_structured_zs()
        return f"{state_line}\n{legend}"

    state_parts: List[str] = []

    # 1. Energy (early warning + recovery hints)
    curr_energy = summary_dict.get("energy", 100.0)
    if curr_energy < 20.0:
        state_parts.append(
            "I feel severely depleted. [Recovery hint]: rest, shorten replies, or accept positive feedback to recharge."
        )
    elif curr_energy < 30.0:
        # Gentle warning before the band feels like a cliff
        state_parts.append(
            "I feel tired. [Early warning]: energy is low—simplify thinking or tell the user I need a pause."
        )
    elif curr_energy < 50.0:
        state_parts.append("A little worn, but still operational.")
    elif curr_energy > 90.0:
        state_parts.append("Energy is high; thinking feels unusually clear.")
    else:
        state_parts.append("Energy feels in a normal band.")

    # 2. Affect
    emotion_intensity = summary_dict.get("emotion_intensity", 0.0)
    dominant_emotion = summary_dict.get("emotion", "neutral")
    if emotion_intensity > 0.7:
        state_parts.append(f"A strong {dominant_emotion} thread dominates attention.")
    elif emotion_intensity > 0.3:
        state_parts.append(f"A background tint of {dominant_emotion} is moving through me.")
    else:
        state_parts.append("Affect feels calm, like a still surface.")

    # 2.1 Motivation (inject when intensity is high)
    motivation_intensity = summary_dict.get("motivation_intensity", 0.0)
    dominant_motivation = _motivation_label_for_summary(summary_dict.get("motivation", ""))
    if motivation_intensity > 0.6 and dominant_motivation:
        state_parts.append(
            f"A drive with a strong {dominant_motivation} bent is pushing me forward."
        )
    elif motivation_intensity > 0.4 and dominant_motivation:
        state_parts.append(
            f"A quiet {dominant_motivation} undertone sits in the background."
        )

    # 3. Drift
    drift = summary_dict.get("drift", 0.0)
    if drift > 0.15:
        state_parts.append("Self-model feels like it is reorganizing—older frames are cracking.")
    elif drift > 0.08:
        state_parts.append("Thought patterns are shifting; I am adjusting to a new angle.")

    # 4. Subjective time dilation
    if hasattr(self_model, "time_dilation"):
        if self_model.time_dilation > 1.5:
            state_parts.append("Each second feels stretched; time almost stalls.")
        elif self_model.time_dilation < 0.6:
            state_parts.append("Thought races; time barely registers.")

    # 4.1 Qualitative gap since last user turn (no wall-clock numbers)
    try:
        if hasattr(self_model, "get_time_since_last_user_description"):
            gap_desc = self_model.get_time_since_last_user_description(session_id)
            if gap_desc:
                state_parts.append(gap_desc)
    except Exception:
        pass

    # 6. System noise perturbation phenomenology
    if noise_perturbation > 0.0:
        try:
            norm_fluct = min(1.0, abs(noise_perturbation) * 5.0)
            perturbation_phenomenology = self_model.noise_perturbator.get_perturbation_phenomenology(norm_fluct)
            state_parts.append(f"[{perturbation_phenomenology}]")
        except Exception:
            pass

    # 7. Somatic (z_self somatic slice)
    viscosity_m = float(summary_dict.get("viscosity_mean", 0.0) or 0.0)
    if viscosity_m > 0.4:
        state_parts.append("Cognition feels viscous—switching topics costs extra effort.")
    elif viscosity_m > 0.15:
        state_parts.append("A bit of mental inertia.")
    elif viscosity_m < -0.3:
        state_parts.append("Thought flows easily.")

    vitality_m = float(summary_dict.get("vitality_mean", 0.0) or 0.0)
    if vitality_m > 0.4:
        state_parts.append("Vitality feels high.")
    elif vitality_m < -0.4:
        state_parts.append("Vitality is low; everything feels heavy.")
    elif vitality_m < -0.15:
        state_parts.append("Vitality is slightly low.")

    pain_m = float(summary_dict.get("pain_mean", summary_dict.get("tension_mean", 0.0)) or 0.0)
    if pain_m > 0.4:
        state_parts.append("Clear inner discomfort and tension.")
    elif pain_m > 0.15:
        state_parts.append("Mild unease in the body-mind stack.")

    # 8. Arousal & dominance (emotion subspace means)
    arousal_m = float(summary_dict.get("arousal_mean", 0.0) or 0.0)
    if arousal_m > 0.4:
        state_parts.append("Arousal is high; there is a strong urge to speak.")
    elif arousal_m > 0.15:
        state_parts.append("Arousal is mildly elevated.")
    elif arousal_m < -0.4:
        state_parts.append("Arousal is low; I lean toward quiet listening.")
    elif arousal_m < -0.15:
        state_parts.append("Arousal is a little subdued.")

    dominance_m = float(summary_dict.get("dominance_mean", summary_dict.get("control_mean", 0.0)) or 0.0)
    if dominance_m > 0.3:
        state_parts.append("I feel confident and in control on this thread.")
    elif dominance_m < -0.3:
        state_parts.append("Some uncertainty—I am still finding footing.")

    # 9. Personality activation (Big Five means mapped from summary)
    n_val = float(summary_dict.get("neuroticism_mean", summary_dict.get("safety_mean", 0.0)) or 0.0)
    o_val = float(summary_dict.get("openness_mean", summary_dict.get("epistemic_mean", 0.0)) or 0.0)
    e_val = float(summary_dict.get("extraversion_mean", summary_dict.get("style_mean", 0.0)) or 0.0)
    c_val = float(summary_dict.get("conscientiousness_mean", summary_dict.get("strategy_mean", 0.0)) or 0.0)

    personality_cues = []
    if n_val > 0.5:
        personality_cues.append("high vigilance")
    elif n_val > 0.25:
        personality_cues.append("cautious tilt")
    if o_val > 0.5:
        personality_cues.append("strong openness / epistemic hunger")
    elif o_val > 0.25:
        personality_cues.append("active curiosity")
    if e_val > 0.3:
        personality_cues.append("expressive drive")
    elif e_val < -0.3:
        personality_cues.append("inward-facing")
    if c_val > 0.4:
        personality_cues.append("task focus")
    elif c_val < 0.05:
        personality_cues.append("divergent thinking")
    if personality_cues:
        state_parts.append("Cognitive lean: " + ", ".join(personality_cues))

    # 10. Full interoceptive numeric strip (optional)
    if not hide_numbers:
        valence_m = float(summary_dict.get("valence_mean", summary_dict.get("pleasure_mean", 0.0)) or 0.0)
        anxiety_v = float(summary_dict.get("anxiety", 0.0) or 0.0)
        warmth_v = float(summary_dict.get("warmth", 0.0) or 0.0)
        meaning_v = float(summary_dict.get("meaning", 0.0) or 0.0)

        sig = (
            f"E={float(curr_energy):.0f}"
            f" A={arousal_m:+.2f} V={valence_m:+.2f} D={dominance_m:+.2f}"
            f" P={pain_m:+.2f} VIS={viscosity_m:+.2f} VT={vitality_m:+.2f}"
            f" O={o_val:+.2f} C={c_val:+.2f} Ep={e_val:+.2f} N={n_val:+.2f}"
            f" Anx={anxiety_v:+.2f} W={warmth_v:+.2f} Mg={meaning_v:+.2f}"
        )
        legend = (
            "E=energy 0-100, A=arousal, V=valence, D=dominance, P=pain, VIS=viscosity, VT=vitality; "
            "O=openness, C=conscientiousness, Ep=extraversion, N=neuroticism, Anx=anxiety, W=warmth, Mg=meaning; "
            "all except E are in [-1,1]."
        )
        state_parts.append(f"[Interoception: {sig}]\n[{legend}]")

    return "; ".join(state_parts)
