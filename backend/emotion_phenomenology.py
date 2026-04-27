#!/usr/bin/env python3
"""
Emotion phenomenology (engineering façade).

Maps numeric affect cues into short, first-person phenomenology-style prose for
prompts, introspection, and governance—not a claim about real qualia.

Design notes:
- We keep a strict split between **measurable state** and **reporting text**.
- ``sadness ≈ 0.8`` is a tensor-friendly scalar; this module supplies a readable layer.
- Output is bounded, testable, and auditable subjective-style copy for operators and the LLM.
"""
from typing import List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class EmotionPhenomenology:
    """Turn named emotions into multi-axis phenomenology snippets."""

    def __init__(self):
        # Phenomenology strings are English-first for LLM-facing prompts; keys keep CN aliases for lookup.
        _JOY = {
            "bodily_feeling": "Warmth spreads outward from the chest.",
            "temporal_quality": "Time feels a little lighter.",
            "world_coloring": "The scene feels brighter.",
            "action_tendency": "I want to share and lean in.",
        }
        _SAD = {
            "bodily_feeling": "A weight sits on the chest.",
            "temporal_quality": "Time feels slow and thick.",
            "world_coloring": "Colors feel muted.",
            "action_tendency": "I want to pull back and go quiet.",
        }
        _ANX = {
            "bodily_feeling": "Heart races; muscles feel tight.",
            "temporal_quality": "The future presses on the present.",
            "world_coloring": "Threats feel close at hand.",
            "action_tendency": "I want to escape and check that I am safe.",
        }
        _CUR = {
            "bodily_feeling": "A forward lean, attention tilting ahead.",
            "temporal_quality": "Time fills with anticipation.",
            "world_coloring": "Everything looks like a possible doorway.",
            "action_tendency": "I want to explore and ask more.",
        }
        _ANG = {
            "bodily_feeling": "Heat gathers in the chest; jaw and shoulders tense.",
            "temporal_quality": "Time feels rushed.",
            "world_coloring": "Obstacles glare.",
            "action_tendency": "I want to push through and change what blocks me.",
        }
        _FEAR = {
            "bodily_feeling": "A shrinking grip around the ribs.",
            "temporal_quality": "Time freezes into the now.",
            "world_coloring": "Danger feels everywhere.",
            "action_tendency": "I want to flee and protect myself.",
        }
        _SUR = {
            "bodily_feeling": "A sudden pause in the body.",
            "temporal_quality": "Time skips a beat.",
            "world_coloring": "The familiar turns strange.",
            "action_tendency": "I want to understand and re-orient.",
        }
        _CONF = {
            "bodily_feeling": "A vague unease, hard to place.",
            "temporal_quality": "Time seems to circle.",
            "world_coloring": "Everything feels uncertain.",
            "action_tendency": "I want to sort it out and find an answer.",
        }
        _GRAT = {
            "bodily_feeling": "A warm fullness in the chest.",
            "temporal_quality": "Time feels meaningful.",
            "world_coloring": "The world looks kinder.",
            "action_tendency": "I want to give back and say thanks.",
        }
        _NEU = {
            "bodily_feeling": "Body feels steady and even.",
            "temporal_quality": "Time moves at an ordinary pace.",
            "world_coloring": "Things look as usual.",
            "action_tendency": "I stay open and responsive.",
        }
        self.phenomenology_map = {
            "joy": _JOY,
            "快乐": _JOY,
            "sadness": _SAD,
            "悲伤": _SAD,
            "anxiety": _ANX,
            "焦虑": _ANX,
            "不安": _ANX,
            "curiosity": _CUR,
            "好奇": _CUR,
            "anger": _ANG,
            "愤怒": _ANG,
            "fear": _FEAR,
            "恐惧": _FEAR,
            "surprise": _SUR,
            "惊讶": _SUR,
            "confusion": _CONF,
            "困惑": _CONF,
            "gratitude": _GRAT,
            "感激": _GRAT,
            "neutral": _NEU,
            "中性": _NEU,
        }

    def describe_emotion_phenomenology(
        self,
        emotion_name: str,
        intensity: float
    ) -> str:
        """
        Produce a short phenomenology string for ``emotion_name`` at ``intensity``.

        Instead of a raw score like ``joy = 0.8``, callers get stitched bodily/temporal/world copy,
        e.g. warmth from the chest plus a brighter scene when intensity is high.

        Args:
            emotion_name: Canonical English label, alias, or legacy Chinese token (e.g. ``joy``, ``快乐``).
            intensity: Scalar strength in ``[0.0, 1.0]``.

        Returns:
            A single English sentence or clause suitable for prompts / logs.
        """
        # Normalize labels so both English and legacy Chinese keys resolve.
        emotion_key = self._normalize_emotion_name(emotion_name)

        if emotion_key not in self.phenomenology_map:
            return f"A vague, name-tagged feeling: {emotion_name}."

        phenom = self.phenomenology_map[emotion_key]

        # Gate detail by intensity bands.
        if intensity < 0.3:
            return f"A faint hint: {phenom['bodily_feeling']}"
        if intensity < 0.6:
            return f"{phenom['bodily_feeling']} {phenom['world_coloring']}"
        return (
            f"{phenom['bodily_feeling']} "
            f"{phenom['temporal_quality']} "
            f"{phenom['world_coloring']} "
            f"{phenom['action_tendency']}"
        )

    def generate_emotion_trajectory(
        self,
        emotion_history: List[Tuple[str, float, str]]  # [(emotion, intensity, timestamp), ...]
    ) -> str:
        """
        Summarize how the tagged emotion changes over ``emotion_history``.

        Treats affect as a simple rise/hold/fall narrative derived from consecutive samples.

        Args:
            emotion_history: Series of ``(emotion_name, intensity, iso_timestamp)`` tuples.

        Returns:
            Comma-separated English clauses, or empty string when fewer than two samples exist.
        """
        if len(emotion_history) < 2:
            return ""

        trajectory_parts = []
        for i in range(1, len(emotion_history)):
            prev_emotion, prev_intensity, prev_time = emotion_history[i - 1]
            curr_emotion, curr_intensity, curr_time = emotion_history[i]

            if curr_emotion != prev_emotion:
                trajectory_parts.append(f"shift from {prev_emotion} toward {curr_emotion}")
            elif curr_intensity > prev_intensity:
                trajectory_parts.append(f"{curr_emotion} intensifying")
            elif curr_intensity < prev_intensity:
                trajectory_parts.append(f"{curr_emotion} easing")

        return ", ".join(trajectory_parts) if trajectory_parts else ""

    def _normalize_emotion_name(self, emotion_name: str) -> str:
        """Map mixed-language aliases onto keys present in ``phenomenology_map``."""
        emotion_mapping = {
            "joy": "joy",
            "快乐": "joy",
            "sadness": "sadness",
            "悲伤": "sadness",
            "anxiety": "anxiety",
            "焦虑": "anxiety",
            "不安": "anxiety",
            "curiosity": "curiosity",
            "好奇": "curiosity",
            "anger": "anger",
            "愤怒": "anger",
            "fear": "fear",
            "恐惧": "fear",
            "surprise": "surprise",
            "惊讶": "surprise",
            "confusion": "confusion",
            "困惑": "confusion",
            "gratitude": "gratitude",
            "感激": "gratitude",
            "neutral": "neutral",
            "中性": "neutral",
        }

        if emotion_name in self.phenomenology_map:
            return emotion_name

        normalized = emotion_mapping.get(emotion_name.lower(), emotion_name.lower())
        if normalized in self.phenomenology_map:
            return normalized

        return emotion_name

    def get_full_phenomenology(
        self,
        emotion_name: str,
        intensity: float
    ) -> Dict[str, str]:
        """
        Return the full four-axis phenomenology dict for ``emotion_name``.

        Returns:
            Mapping with keys ``bodily_feeling``, ``temporal_quality``, ``world_coloring``, ``action_tendency``.
        """
        emotion_key = self._normalize_emotion_name(emotion_name)

        if emotion_key not in self.phenomenology_map:
            return {
                "bodily_feeling": f"A vague, name-tagged feeling: {emotion_name}",
                "temporal_quality": "Time moves at an ordinary pace.",
                "world_coloring": "Things look as usual.",
                "action_tendency": "I stay open and responsive.",
            }

        return self.phenomenology_map[emotion_key].copy()
