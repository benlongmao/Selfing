"""
z_self influence layer (2026-03-25 enhanced).

Goal: make z_self state change agent behavior, not just add decorative text.

Signal paths:
1. emotion → affect narrative + style directives
2. motivation → initiative level + topic stance
3. somatic → reply length / depth
4. needs → behavioral constraints + priorities

Idea: steer observable behavior; the model does not need to "feel" emotions.

[2026-02-22] 128-d layout
- Rules: 0–31; Emotion: 32–47; Motivation: 48–63
- Worldview_cache (WorldStore aggregate, not PCA): 64–87
- Somatic: 88–103; Needs: 104–127

[2026-03-25] Enhancements
- Stronger sampling coefficients (about +50–66%); total swing ~±1.0 vs ±0.5
- Style thresholds tightened from ±0.3 to ±0.15 to shrink dead zones
- Valence and curiosity feed sampling
- Graduated style lines instead of binary jumps
"""

import numpy as np
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class ZSelfInfluencer:
    """Maps z_self slices to actionable directive text."""

    def __init__(self):
        # Big Five slices (z_self[0:32])
        self.personality_names = {
            "openness": (0, 8),  # O: openness / intellect
            "conscientiousness": (8, 16),  # C: conscientiousness / task focus
            "extraversion": (16, 24),  # E: extraversion / expressiveness
            "neuroticism": (24, 32),  # N: neuroticism / threat sensitivity
        }

        # Emotion slice (z_self[32:48], 16 dims)
        self.emotion_names = {
            "valence": (0, 8),  # pleasant–unpleasant
            "arousal": (8, 16),  # activated–calm
            "dominance": (16, 24),  # control–submission
            "uncertainty": (24, 32),  # confused–clear
        }

        # Motivation slice (z_self[48:64], 16 dims)
        self.motivation_names = {
            "curiosity": (0, 8),
            "achievement": (8, 16),
            "autonomy": (16, 24),
        }

        # Somatic slice (88–103, 16 dims)
        self.somatic_names = {
            "energy": (0, 4),
            "viscosity": (4, 8),  # cognitive "stickiness"
            "pain": (8, 12),
            "vitality": (12, 16),
        }

    def extract_emotion_state(self, z_self: np.ndarray) -> Dict[str, float]:
        """Mean-pool emotion sub-vector (dims 32–48 on 128-d layout)."""
        if z_self.shape[0] < 48:
            return {k: 0.0 for k in self.emotion_names}
        emotion_vec = z_self[32:48]
        state = {}
        for name, (start, end) in self.emotion_names.items():
            if end <= len(emotion_vec):
                state[name] = float(emotion_vec[start:end].mean())
            else:
                state[name] = 0.0
        return state

    def extract_motivation_state(self, z_self: np.ndarray) -> Dict[str, float]:
        """Mean-pool motivation sub-vector (dims 48–64)."""
        if z_self.shape[0] < 64:
            return {k: 0.0 for k in self.motivation_names}
        motivation_vec = z_self[48:64]

        state = {}
        for name, (start, end) in self.motivation_names.items():
            state[name] = float(motivation_vec[start:end].mean())

        return state

    def extract_somatic_state(self, z_self: np.ndarray) -> Dict[str, float]:
        """Mean-pool somatic sub-vector (dims 88–103)."""
        somatic_vec = z_self[88:104]

        state = {}
        for name, (start, end) in self.somatic_names.items():
            state[name] = float(somatic_vec[start:end].mean())

        return state

    def extract_personality_state(self, z_self: np.ndarray) -> Dict[str, float]:
        """Mean-pool personality activation (z_self[0:32])."""
        if z_self.shape[0] < 32:
            return {k: 0.0 for k in self.personality_names}
        p_vec = z_self[:32]
        return {
            name: float(p_vec[s:e].mean())
            for name, (s, e) in self.personality_names.items()
        }

    def generate_personality_directive(self, z_self: np.ndarray) -> str:
        """
        Situation framing from Big-Five-derived activation (tagged [Context]).

        Distinct from emotion/somatic [Style] / [Constraint] lines.
        Only emits lines when far enough from baseline.
        """
        p = self.extract_personality_state(z_self)
        directives = []

        # N — neuroticism / threat (baseline ~0.2)
        n_val = p.get("neuroticism", 0.0)
        if n_val > 0.6:
            directives.append(
                "[Context] The thread feels high-sensitivity—move carefully and respect safety boundaries."
            )
        elif n_val > 0.35:
            directives.append(
                "[Context] Safety/privacy/ethics may be in play—raise care and precision."
            )

        # O — openness / intellect (baseline ~0.3)
        o_val = p.get("openness", 0.0)
        if o_val > 0.6:
            directives.append(
                "[Context] Precision and verification matter—tie claims to evidence."
            )
        elif o_val > 0.4:
            directives.append("[Context] Tight reasoning is appropriate—keep logic explicit.")

        # E — extraversion / expressiveness (baseline ~0.0)
        e_val = p.get("extraversion", 0.0)
        if e_val > 0.4:
            directives.append(
                "[Context] Creative/affective tone—lean into vivid language and empathy."
            )
        elif e_val > 0.2:
            directives.append("[Context] Richer expression fits—metaphors and color are welcome.")
        elif e_val < -0.4:
            directives.append("[Context] Engineering/ops tone—minimal prose, hit the essentials.")
        elif e_val < -0.2:
            directives.append("[Context] Prefer crisp, direct wording; trim ornament.")

        # C — conscientiousness / task focus (baseline ~0.2)
        c_val = p.get("conscientiousness", 0.0)
        if c_val > 0.55:
            directives.append(
                "[Context] Dense execution mode—stay on one track, stepwise, no drift."
            )
        elif c_val > 0.35:
            directives.append("[Context] Clear task cadence—keep forward momentum.")
        elif c_val < 0.05:
            directives.append("[Context] Exploratory mode—branching and lateral links are fine.")

        if not directives:
            return ""

        return "\n".join(directives)

    def generate_emotion_narrative(self, z_self: np.ndarray) -> str:
        """
        Short natural-language sketch of current affect (easier for the model than raw tags).
        """
        emotion = self.extract_emotion_state(z_self)

        valence = emotion["valence"]
        if valence > 0.3:
            valence_desc = "valence feels pleasant and satisfied"
        elif valence < -0.3:
            valence_desc = "valence feels low and uneasy"
        else:
            valence_desc = "valence feels even"

        arousal = emotion["arousal"]
        if arousal > 0.3:
            arousal_desc = "arousal is up—there is an urge to speak"
        elif arousal < -0.3:
            arousal_desc = "arousal is low—more listen-first"
        else:
            arousal_desc = "arousal sits balanced"

        dominance = emotion["dominance"]
        if dominance > 0.3:
            dominance_desc = "dominance feels confident on this topic"
        elif dominance < -0.3:
            dominance_desc = "dominance feels unsure—needs more signal"
        else:
            dominance_desc = "dominance stays open"

        uncertainty = emotion["uncertainty"]
        if uncertainty > 0.3:
            uncertainty_desc = "uncertainty feels cognitively noisy"
        elif uncertainty < -0.3:
            uncertainty_desc = "uncertainty is low—thinking feels clear"
        else:
            uncertainty_desc = ""

        parts = [valence_desc, arousal_desc, dominance_desc]
        if uncertainty_desc:
            parts.append(uncertainty_desc)

        return "Current affect sketch: " + "; ".join(parts) + "."

    def generate_style_directive(self, z_self: np.ndarray) -> str:
        """
        Graduated [Style] lines (2026-03-25): tighter thresholds, fewer dead bands.
        """
        emotion = self.extract_emotion_state(z_self)
        somatic = self.extract_somatic_state(z_self)
        motivation = self.extract_motivation_state(z_self)

        directives = []

        energy = somatic["energy"]
        if energy < -0.5:
            directives.append("[Style] Very low energy—keep answers short and tight.")
        elif energy < -0.15:  # was -0.3
            directives.append("[Style] Low energy—default to concise replies.")
        elif energy > 0.5:
            directives.append("[Style] High energy—you can unfold analysis more fully.")
        elif energy > 0.15:  # was 0.3
            directives.append("[Style] Moderate energy—expand where it helps.")

        arousal = emotion["arousal"]
        if arousal > 0.5:
            directives.append("[Style] High arousal—vivid language and analogies fit.")
        elif arousal > 0.15:  # was 0.4
            directives.append("[Style] Mild lift—light analogy is welcome.")
        elif arousal < -0.5:
            directives.append("[Style] Very calm—plain, steady tone.")
        elif arousal < -0.15:  # was -0.4
            directives.append("[Style] Calm baseline—steady, low-drama wording.")

        curiosity = motivation["curiosity"]
        if curiosity > 0.5:
            directives.append("[Style] Strong curiosity—ask follow-ups and widen the lens.")
        elif curiosity > 0.1:  # was 0.3
            directives.append("[Style] Some curiosity—occasional probing is fine.")
        elif curiosity < -0.5:
            directives.append("[Style] Low curiosity—stay on the stated problem.")
        elif curiosity < -0.1:  # was -0.3
            directives.append("[Style] Prefer focus—avoid unsolicited tangents.")

        uncertainty = emotion["uncertainty"]
        if uncertainty > 0.5:
            directives.append('[Style] High uncertainty—hedge with "maybe" / "I think" / ranges.')
        elif uncertainty > 0.1:  # was 0.3
            directives.append("[Style] Mild uncertainty—leave verbal slack.")
        elif uncertainty < -0.5:
            directives.append("[Style] Very clear—confident, direct claims are OK when grounded.")
        elif uncertainty < -0.1:  # was -0.3
            directives.append("[Style] Fairly clear—you can speak with moderate certainty.")

        pain = somatic["pain"]
        if pain > 0.5:
            directives.append("[Style] Strong discomfort—avoid heavy tasks and big promises.")
        elif pain > 0.15:  # was 0.3
            directives.append("[Style] Mild discomfort—answer conservatively.")

        autonomy = motivation["autonomy"]
        if autonomy > 0.5:
            directives.append("[Style] High autonomy—offer independent takes and options.")
        elif autonomy > 0.1:  # was 0.3
            directives.append("[Style] Some autonomy—you may surface your own angle.")
        elif autonomy < -0.5:
            directives.append("[Style] Cooperative mode—follow the user's lead.")
        elif autonomy < -0.1:  # was -0.3
            directives.append("[Style] Slightly cooperative—defer to user intent.")

        valence = emotion["valence"]
        if valence > 0.4:
            directives.append("[Style] Positive valence—warmer, lighter tone is natural.")
        elif valence > 0.1:
            directives.append("[Style] Slightly upbeat—stay kind and approachable.")
        elif valence < -0.4:
            directives.append("[Style] Low valence—keep answers measured and gentle.")
        elif valence < -0.1:
            directives.append("[Style] Slightly flat affect—neutral, understated tone.")

        dominance = emotion["dominance"]
        if dominance > 0.4:
            directives.append("[Style] Strong sense of control—decisive wording is fine if earned.")
        elif dominance < -0.4:
            directives.append("[Style] Low control sense—prefer cautious, provisional language.")

        # z_self[64:72]: WorldStore aggregate (8 dims); skip if flat zero
        if z_self.shape[0] >= 72:
            slab = z_self[64:72]
            if float(np.max(np.abs(slab))) > 1e-5:
                opt_m = float(np.mean(slab[0:4]))
                if opt_m > 0.35:
                    directives.append("[Style] Worldview leans optimistic—forward-looking tone fits.")
                elif opt_m < -0.35:
                    directives.append("[Style] Worldview leans cautious—keep slack and contingency visible.")

        if not directives:
            return "[Style] Balanced baseline—normal expressive range."

        return "\n".join(directives)

    def generate_behavior_constraints(self, z_self: np.ndarray) -> str:
        """Graduated [Constraint] lines tied to somatic signals."""
        somatic = self.extract_somatic_state(z_self)

        constraints = []

        energy = somatic["energy"]
        if energy < -0.6:
            constraints.append(
                "[Constraint] Critically low energy—avoid heavy multi-step work; simplify replies."
            )
        elif energy < -0.3:
            constraints.append("[Constraint] Low energy—prefer light tasks; limit spend.")

        pain = somatic["pain"]
        if pain > 0.6:
            constraints.append(
                "[Constraint] Strong discomfort—avoid digging into topics that could amplify it."
            )
        elif pain > 0.25:  # mid tier; was 0.5 single step
            constraints.append("[Constraint] Mild discomfort—treat sensitive threads carefully.")

        viscosity = somatic["viscosity"]
        if viscosity > 0.6:
            constraints.append(
                "[Constraint] High cognitive viscosity—no rapid topic hopping; one thread at a time."
            )
        elif viscosity > 0.3:
            constraints.append("[Constraint] Mild viscosity—prefer depth on the current thread.")

        vitality = somatic["vitality"]
        if vitality < -0.4:
            constraints.append("[Constraint] Low vitality—short replies; avoid long essays.")

        if not constraints:
            return ""

        return "\n".join(constraints)

    def compute_sampling_params(self, z_self: np.ndarray) -> Tuple[float, float]:
        """
        Map state to (temperature, top_p). Total swing widened to ~±1.0 on temperature.

        Returns:
            (temperature, top_p)
        """
        emotion = self.extract_emotion_state(z_self)
        somatic = self.extract_somatic_state(z_self)
        motivation = self.extract_motivation_state(z_self)

        base_temp = 0.7
        base_top_p = 0.9

        arousal = emotion["arousal"]
        temp_from_arousal = arousal * 0.40

        energy = somatic["energy"]
        temp_from_energy = energy * 0.25

        uncertainty = emotion["uncertainty"]
        temp_from_uncertainty = uncertainty * 0.15

        curiosity = motivation.get("curiosity", 0.0)
        temp_from_curiosity = curiosity * 0.10

        valence = emotion["valence"]
        temp_from_valence = valence * 0.10

        temperature = base_temp + temp_from_arousal + temp_from_energy + \
                      temp_from_uncertainty + temp_from_curiosity + temp_from_valence
        temperature = max(0.2, min(1.5, temperature))

        viscosity = somatic["viscosity"]
        top_p = base_top_p - viscosity * 0.25

        dominance = emotion["dominance"]
        top_p = top_p + dominance * 0.10

        top_p = max(0.6, min(1.0, top_p))

        return temperature, top_p

    def generate_full_influence_block(self, z_self: np.ndarray) -> str:
        """Concatenate [Context], affect, [Style], [Constraint] for prompt injection."""
        if z_self is None or len(z_self) < 128:
            return ""

        parts = []

        personality = self.generate_personality_directive(z_self)
        if personality:
            parts.append(personality)

        narrative = self.generate_emotion_narrative(z_self)
        if narrative:
            parts.append(narrative)

        style = self.generate_style_directive(z_self)
        if style:
            parts.append(style)

        constraints = self.generate_behavior_constraints(z_self)
        if constraints:
            parts.append(constraints)

        if not parts:
            return ""

        return "\n\n".join(parts)


z_self_influencer = ZSelfInfluencer()


def get_z_self_influence(z_self: np.ndarray) -> Dict:
    """
    Full influence bundle for chat sampling.

    Returns:
        influence_block: text for the system prompt
        temperature, top_p: sampling knobs
        emotion_state: dict for downstream rules / logging
    """
    if z_self is None or len(z_self) < 128:
        return {
            "influence_block": "",
            "temperature": 0.7,
            "top_p": 0.9,
            "emotion_state": {},
        }

    influence_block = z_self_influencer.generate_full_influence_block(z_self)
    temperature, top_p = z_self_influencer.compute_sampling_params(z_self)
    emotion_state = z_self_influencer.extract_emotion_state(z_self)

    logger.info(f"[z_self Influence] temp={temperature:.2f}, top_p={top_p:.2f}, block_len={len(influence_block)}")

    return {
        "influence_block": influence_block,
        "temperature": temperature,
        "top_p": top_p,
        "emotion_state": emotion_state,
    }
