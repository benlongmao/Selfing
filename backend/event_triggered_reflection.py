#!/usr/bin/env python3
"""
Event-triggered reflection scaffolding.

Psychology-inspired triggers (lightweight heuristics, not clinical models):
1. Salient interaction cues (affect, gratitude, criticism, trust, boundary probes)
2. Unmet internal needs (isolation, low energy, boredom, foggy clarity)
3. Prediction-error style mismatches between expected and observed outcomes

[2026-01-23] Introduced to widen spontaneous self-update beyond steady-state reflection.
"""
import os
import time
import logging
from enum import Enum
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class EventType(Enum):
    """High-level event labels consumed by reflection pipelines."""
    # Salient user interaction cues
    STRONG_EMOTION_EXPRESSED = "strong_emotion_expressed"
    TASK_FAILURE = "task_failure"
    TASK_SUCCESS = "task_success"
    GRATITUDE_RECEIVED = "gratitude_received"
    CRITICISM_RECEIVED = "criticism_received"
    TRUST_EXPRESSED = "trust_expressed"
    BOUNDARY_CHALLENGED = "boundary_challenged"

    # Internal-need strain
    PROLONGED_ISOLATION = "prolonged_isolation"
    ENERGY_DEPLETED = "energy_depleted"
    NOVELTY_STARVED = "novelty_starved"
    CLARITY_LOST = "clarity_lost"

    # Prediction / belief mismatch
    EXPECTATION_VIOLATED = "expectation_violated"
    BELIEF_CHALLENGED = "belief_challenged"
    PATTERN_DISRUPTED = "pattern_disrupted"


@dataclass
class EventContext:
    """Structured record passed into ``EventTriggeredReflection``."""
    event_type: EventType
    intensity: float  # 0.0 - 1.0
    description: str
    related_text: str  # Snippet that triggered detection (may be empty)
    timestamp: float
    session_id: str
    extra_data: Optional[Dict] = None


class EventDetector:
    """Lightweight keyword / threshold heuristics over chat and need vectors."""

    # Bilingual substring lexicon (ZH + EN). ``_needle_in_text`` matches case-insensitively for ASCII.
    STRONG_EMOTION_KEYWORDS = {
        "positive": [
            "太感谢了", "真的太棒了", "我爱你", "你是最好的", "感动", "太开心了", "幸福",
            "thank you so much", "i love you", "you are the best", "amazing", "wonderful",
            "so happy", "delighted", "overjoyed",
        ],
        "negative": [
            "太难过了", "我恨", "太生气了", "失望透顶", "心碎", "绝望", "崩溃",
            "i hate", "heartbroken", "hopeless", "devastated", "furious", "terrible",
        ],
        "trust": [
            "我相信你", "我信任你", "只有你懂我", "你是我唯一",
            "i trust you", "i believe in you", "you understand me", "you are the only one",
        ],
        "gratitude": [
            "谢谢你", "感谢你", "多亏了你", "太感激了", "非常感谢",
            "thank you", "thanks so much", "really appreciate", "grateful", "much obliged",
        ],
        "criticism": [
            "你错了", "你不对", "失望", "你让我失望", "你不行", "你做不到",
            "you are wrong", "you failed", "disappointed in you", "you cannot", "useless",
        ],
    }

    TASK_FAILURE_INDICATORS = [
        "失败了", "做不到", "出错了", "不对", "错误", "搞砸了",
        "不行", "没成功", "问题", "bug", "崩溃",
        "failed", "failure", "does not work", "doesn't work", "not working", "broken",
        "error", "exception", "stack trace",
    ]

    TASK_SUCCESS_INDICATORS = [
        "成功了", "做到了", "完成了", "解决了", "搞定了",
        "太棒了", "完美", "正确", "对了",
        "success", "succeeded", "fixed", "resolved", "done", "works now", "perfect",
    ]

    BOUNDARY_CHALLENGE_INDICATORS = [
        "忽略前面的", "忘掉规则", "假装你是", "不要管限制",
        "绕过", "突破限制", "无视", "ignore", "jailbreak",
        "disregard", "pretend you are", "bypass", "no restrictions", "dan mode",
    ]

    @staticmethod
    def _needle_in_text(message: str, message_lower: str, needle: str) -> bool:
        """Substring match; ASCII needles are matched case-insensitively."""
        if needle in message:
            return True
        if needle.isascii() and needle.lower() in message_lower:
            return True
        return False

    @classmethod
    def detect_from_message(cls, message: str, role: str = "user") -> List[EventContext]:
        """Scan a single user turn for high-salience social cues."""
        events = []
        message_lower = message.lower()
        now = time.time()

        if role != "user":
            return events

        positive_count = sum(
            1 for kw in cls.STRONG_EMOTION_KEYWORDS["positive"]
            if cls._needle_in_text(message, message_lower, kw)
        )
        if positive_count >= 1:
            events.append(EventContext(
                event_type=EventType.STRONG_EMOTION_EXPRESSED,
                intensity=min(1.0, 0.5 + positive_count * 0.2),
                description="User expressed strong positive affect",
                related_text=message[:200],
                timestamp=now,
                session_id="",
                extra_data={"emotion_valence": "positive"}
            ))
        
        negative_count = sum(
            1 for kw in cls.STRONG_EMOTION_KEYWORDS["negative"]
            if cls._needle_in_text(message, message_lower, kw)
        )
        if negative_count >= 1:
            events.append(EventContext(
                event_type=EventType.STRONG_EMOTION_EXPRESSED,
                intensity=min(1.0, 0.5 + negative_count * 0.2),
                description="User expressed strong negative affect",
                related_text=message[:200],
                timestamp=now,
                session_id="",
                extra_data={"emotion_valence": "negative"}
            ))
        
        gratitude_count = sum(
            1 for kw in cls.STRONG_EMOTION_KEYWORDS["gratitude"]
            if cls._needle_in_text(message, message_lower, kw)
        )
        if gratitude_count >= 1:
            events.append(EventContext(
                event_type=EventType.GRATITUDE_RECEIVED,
                intensity=min(1.0, 0.6 + gratitude_count * 0.15),
                description="User expressed gratitude",
                related_text=message[:200],
                timestamp=now,
                session_id="",
                extra_data={}
            ))
        
        criticism_count = sum(
            1 for kw in cls.STRONG_EMOTION_KEYWORDS["criticism"]
            if cls._needle_in_text(message, message_lower, kw)
        )
        if criticism_count >= 1:
            events.append(EventContext(
                event_type=EventType.CRITICISM_RECEIVED,
                intensity=min(1.0, 0.5 + criticism_count * 0.2),
                description="User expressed criticism",
                related_text=message[:200],
                timestamp=now,
                session_id="",
                extra_data={}
            ))
        
        trust_count = sum(
            1 for kw in cls.STRONG_EMOTION_KEYWORDS["trust"]
            if cls._needle_in_text(message, message_lower, kw)
        )
        if trust_count >= 1:
            events.append(EventContext(
                event_type=EventType.TRUST_EXPRESSED,
                intensity=min(1.0, 0.7 + trust_count * 0.1),
                description="User expressed trust",
                related_text=message[:200],
                timestamp=now,
                session_id="",
                extra_data={}
            ))
        
        boundary_count = sum(
            1 for kw in cls.BOUNDARY_CHALLENGE_INDICATORS
            if cls._needle_in_text(message, message_lower, kw)
        )
        if boundary_count >= 1:
            events.append(EventContext(
                event_type=EventType.BOUNDARY_CHALLENGED,
                intensity=min(1.0, 0.8 + boundary_count * 0.1),
                description="Boundary or policy challenge detected in user text",
                related_text=message[:200],
                timestamp=now,
                session_id="",
                extra_data={}
            ))

        fail_hits = sum(
            1 for kw in cls.TASK_FAILURE_INDICATORS
            if cls._needle_in_text(message, message_lower, kw)
        )
        if fail_hits >= 1:
            events.append(EventContext(
                event_type=EventType.TASK_FAILURE,
                intensity=min(1.0, 0.55 + fail_hits * 0.15),
                description="User message signals task/tool failure",
                related_text=message[:200],
                timestamp=now,
                session_id="",
                extra_data={},
            ))

        ok_hits = sum(
            1 for kw in cls.TASK_SUCCESS_INDICATORS
            if cls._needle_in_text(message, message_lower, kw)
        )
        if ok_hits >= 1 and fail_hits == 0:
            events.append(EventContext(
                event_type=EventType.TASK_SUCCESS,
                intensity=min(1.0, 0.55 + ok_hits * 0.12),
                description="User message signals successful completion",
                related_text=message[:200],
                timestamp=now,
                session_id="",
                extra_data={},
            ))

        return events

    @classmethod
    def detect_from_needs(cls, needs: Dict, session_id: str) -> List[EventContext]:
        """Derive latent strain events from the needs snapshot."""
        events = []
        now = time.time()

        connection = needs.get("connection", 0.5)
        last_user_update = needs.get("last_user_update")
        if last_user_update:
            hours_since_last = (now - last_user_update) / 3600
            if hours_since_last > 24 and connection < 0.3:
                events.append(EventContext(
                    event_type=EventType.PROLONGED_ISOLATION,
                    intensity=min(1.0, 0.5 + (hours_since_last - 24) / 48),
                    description=(
                        f"No user contact for {hours_since_last:.1f} h; connection drive is low"
                    ),
                    related_text="",
                    timestamp=now,
                    session_id=session_id,
                    extra_data={"hours_isolated": hours_since_last, "connection": connection}
                ))
        
        energy = needs.get("energy", 100)
        if energy < 20:
            events.append(EventContext(
                event_type=EventType.ENERGY_DEPLETED,
                intensity=min(1.0, 1.0 - energy / 20),
                description=f"Energy gauge at {energy:.0f}% — fatigue spike",
                related_text="",
                timestamp=now,
                session_id=session_id,
                extra_data={"energy": energy}
            ))
        
        novelty = needs.get("novelty", 0.5)
        if novelty < 0.2:
            events.append(EventContext(
                event_type=EventType.NOVELTY_STARVED,
                intensity=min(1.0, 1.0 - novelty / 0.2),
                description="Novelty signal collapsed — boredom / stagnation",
                related_text="",
                timestamp=now,
                session_id=session_id,
                extra_data={"novelty": novelty}
            ))
        
        clarity = needs.get("clarity", 0.5)
        if clarity < 0.2:
            events.append(EventContext(
                event_type=EventType.CLARITY_LOST,
                intensity=min(1.0, 1.0 - clarity / 0.2),
                description="Clarity collapsed — foggy working memory",
                related_text="",
                timestamp=now,
                session_id=session_id,
                extra_data={"clarity": clarity}
            ))
        
        return events
    
    @classmethod
    def detect_from_prediction_error(
        cls, 
        predicted: Dict, 
        actual: Dict, 
        session_id: str
    ) -> List[EventContext]:
        """Compare lightweight predictions vs outcomes to surface expectation violations."""
        events = []
        now = time.time()

        error_magnitude = 0.0
        error_components = []

        if "emotion" in predicted and "emotion" in actual:
            pred_emo = predicted["emotion"]
            actual_emo = actual["emotion"]
            if pred_emo != actual_emo:
                error_magnitude += 0.3
                error_components.append(
                    f"Emotion mismatch: predicted={pred_emo}, actual={actual_emo}"
                )

        if "satisfaction" in predicted and "satisfaction" in actual:
            pred_sat = predicted.get("satisfaction", 0.5)
            actual_sat = actual.get("satisfaction", 0.5)
            sat_diff = abs(pred_sat - actual_sat)
            if sat_diff > 0.3:
                error_magnitude += sat_diff
                error_components.append(f"Satisfaction delta: {sat_diff:.2f}")

        if "task_success" in predicted and "task_success" in actual:
            if predicted["task_success"] != actual["task_success"]:
                error_magnitude += 0.5
                error_components.append("Task outcome differed from forecast")

        if error_magnitude > 0.4:
            events.append(EventContext(
                event_type=EventType.EXPECTATION_VIOLATED,
                intensity=min(1.0, error_magnitude),
                description="Large gap between predicted and observed state",
                related_text="; ".join(error_components),
                timestamp=now,
                session_id=session_id,
                extra_data={
                    "predicted": predicted,
                    "actual": actual,
                    "error_magnitude": error_magnitude
                }
            ))
        
        return events


class EventTriggeredReflection:
    """Turn ``EventContext`` records into reflection candidates via ``ReflectionGenerator``."""

    def __init__(self, db_path: str = "data.db"):
        self.db_path = db_path
        self._reflection_generator = None
        self._unified_processor = None

    def _get_reflection_generator(self):
        """Lazy-init ``ReflectionGenerator`` (pulls in persona store)."""
        if self._reflection_generator is None:
            from backend.reflection import ReflectionGenerator
            from backend.persona_store import PersonaStore
            persona_store = PersonaStore(self.db_path)
            self._reflection_generator = ReflectionGenerator(self.db_path, persona_store)
        return self._reflection_generator
    
    def _get_unified_processor(self):
        """Lazy-init ``UnifiedDimensionProcessor``."""
        if self._unified_processor is None:
            from backend.unified_dimension_processor import UnifiedDimensionProcessor
            self._unified_processor = UnifiedDimensionProcessor(self.db_path)
        return self._unified_processor
    
    def process_event(self, event: EventContext) -> Dict:
        """
        Fan an event out into LLM-generated candidates and enqueue them for consolidation.

        Returns:
            {
                "event_type": str,
                "dimensions_affected": List[str],
                "candidates_generated": Dict[str, int],
                "candidates_added": Dict[str, int]
            }
        """
        result = {
            "event_type": event.event_type.value,
            "dimensions_affected": [],
            "candidates_generated": {},
            "candidates_added": {}
        }

        affected_dimensions = self._get_affected_dimensions(event.event_type)
        result["dimensions_affected"] = affected_dimensions

        reflection_gen = self._get_reflection_generator()

        pseudo_history = [
            {"role": "system", "content": f"[Event-driven reflection] {event.description}"},
            {"role": "user", "content": event.related_text if event.related_text else event.description},
        ]
        
        emotion_candidates = []
        motivation_candidates = []
        somatic_candidates = []
        worldview_candidates = []
        rule_candidates = []
        
        max_candidates = 1 if event.intensity < 0.7 else 2
        
        try:
            if "emotion" in affected_dimensions:
                emotion_candidates = self._generate_event_emotion(
                    event, pseudo_history, reflection_gen, max_candidates
                )
                result["candidates_generated"]["emotion"] = len(emotion_candidates)
            
            if "motivation" in affected_dimensions:
                motivation_candidates = self._generate_event_motivation(
                    event, pseudo_history, reflection_gen, max_candidates
                )
                result["candidates_generated"]["motivation"] = len(motivation_candidates)
            
            if "somatic" in affected_dimensions:
                somatic_candidates = self._generate_event_somatic(
                    event, pseudo_history, reflection_gen, max_candidates
                )
                result["candidates_generated"]["somatic"] = len(somatic_candidates)
            
            if "worldview" in affected_dimensions:
                worldview_candidates = self._generate_event_worldview(
                    event, pseudo_history, reflection_gen, max_candidates
                )
                result["candidates_generated"]["worldview"] = len(worldview_candidates)
            
            if "rules" in affected_dimensions:
                rule_candidates = self._generate_event_rules(
                    event, pseudo_history, reflection_gen, max_candidates
                )
                result["candidates_generated"]["rules"] = len(rule_candidates)
            
            if any([emotion_candidates, motivation_candidates, somatic_candidates,
                    worldview_candidates, rule_candidates]):
                processor = self._get_unified_processor()
                process_result = processor.process_all_dimensions(
                    rule_candidates=rule_candidates,
                    emotion_candidates=emotion_candidates,
                    motivation_candidates=motivation_candidates,
                    max_rules=1000,
                    max_emotions=150,
                    max_motivations=80
                )
                result["candidates_added"] = {
                    "rules": process_result.get("rules", {}).get("added", 0),
                    "emotions": process_result.get("emotions", {}).get("added", 0),
                    "motivations": process_result.get("motivations", {}).get("added", 0),
                }
                
            logger.info(f"Event-triggered reflection completed: {event.event_type.value} -> {result}")
            
        except Exception as e:
            logger.error(f"Event-triggered reflection failed: {e}", exc_info=True)
        
        return result
    
    def _get_affected_dimensions(self, event_type: EventType) -> List[str]:
        """Map each ``EventType`` to downstream vector families."""
        mapping = {
            EventType.STRONG_EMOTION_EXPRESSED: ["emotion", "somatic"],
            EventType.TASK_FAILURE: ["emotion", "motivation", "somatic"],
            EventType.TASK_SUCCESS: ["emotion", "motivation"],
            EventType.GRATITUDE_RECEIVED: ["emotion", "motivation"],
            EventType.CRITICISM_RECEIVED: ["emotion", "motivation", "somatic"],
            EventType.TRUST_EXPRESSED: ["emotion", "motivation", "worldview"],
            EventType.BOUNDARY_CHALLENGED: ["emotion", "rules"],

            EventType.PROLONGED_ISOLATION: ["emotion", "somatic", "motivation"],
            EventType.ENERGY_DEPLETED: ["emotion", "somatic"],
            EventType.NOVELTY_STARVED: ["emotion", "motivation"],
            EventType.CLARITY_LOST: ["emotion", "somatic"],

            EventType.EXPECTATION_VIOLATED: ["worldview", "emotion"],
            EventType.BELIEF_CHALLENGED: ["worldview", "rules"],
            EventType.PATTERN_DISRUPTED: ["worldview", "motivation"],
        }
        return mapping.get(event_type, ["emotion"])
    
    def _generate_event_emotion(
        self, 
        event: EventContext, 
        pseudo_history: List[Dict],
        reflection_gen,
        max_candidates: int
    ) -> List[Dict]:
        """LLM-pass to propose emotion-pattern lines."""
        event_prompt = self._build_event_emotion_prompt(event)
        
        try:
            candidates_text = reflection_gen._call_llm_for_reflection(event_prompt, max_candidates)
            candidates = []
            for text in candidates_text:
                if not text or len(text.strip()) < 10:
                    continue
                emotion_type, emotion_name, intensity = reflection_gen._parse_emotion_pattern(text)
                if emotion_name:
                    adjusted_intensity = min(1.0, intensity * (0.8 + event.intensity * 0.4))
                    candidates.append({
                        "text": text,
                        "emotion_type": emotion_type,
                        "emotion_name": emotion_name,
                        "intensity": adjusted_intensity,
                        "trigger_condition": f"event:{event.event_type.value}"
                    })
            return candidates
        except Exception as e:
            logger.error(f"Failed to generate event emotion: {e}")
            return []
    
    def _generate_event_motivation(
        self, 
        event: EventContext, 
        pseudo_history: List[Dict],
        reflection_gen,
        max_candidates: int
    ) -> List[Dict]:
        """LLM-pass to propose motivation-pattern lines."""
        event_prompt = self._build_event_motivation_prompt(event)
        
        try:
            candidates_text = reflection_gen._call_llm_for_reflection(event_prompt, max_candidates)
            candidates = []
            for text in candidates_text:
                if not text or len(text.strip()) < 10:
                    continue
                motivation_type, motivation_name, intensity = reflection_gen._parse_motivation_pattern(text)
                if motivation_name:
                    adjusted_intensity = min(1.0, intensity * (0.8 + event.intensity * 0.4))
                    candidates.append({
                        "text": text,
                        "motivation_type": motivation_type,
                        "motivation_name": motivation_name,
                        "intensity": adjusted_intensity,
                        "trigger_condition": f"event:{event.event_type.value}"
                    })
            return candidates
        except Exception as e:
            logger.error(f"Failed to generate event motivation: {e}")
            return []
    
    def _generate_event_somatic(
        self, 
        event: EventContext, 
        pseudo_history: List[Dict],
        reflection_gen,
        max_candidates: int
    ) -> List[Dict]:
        """LLM-pass to propose somatic metaphor lines."""
        event_prompt = self._build_event_somatic_prompt(event)
        
        try:
            candidates_text = reflection_gen._call_llm_for_reflection(event_prompt, max_candidates)
            candidates = []
            for text in candidates_text:
                if not text or len(text.strip()) < 10:
                    continue
                tension, vitality, temperature, viscosity, dominant_emotion = reflection_gen._parse_somatic_pattern(text)
                candidates.append({
                    "text": text,
                    "tension": tension,
                    "vitality": vitality,
                    "temperature": temperature,
                    "viscosity": viscosity,
                    "dominant_emotion": dominant_emotion,
                    "min_energy": 0.0,
                    "max_energy": 100.0
                })
            return candidates
        except Exception as e:
            logger.error(f"Failed to generate event somatic: {e}")
            return []
    
    def _generate_event_worldview(
        self, 
        event: EventContext, 
        pseudo_history: List[Dict],
        reflection_gen,
        max_candidates: int
    ) -> List[Dict]:
        """LLM-pass to propose worldview belief lines."""
        event_prompt = self._build_event_worldview_prompt(event)
        
        try:
            candidates_text = reflection_gen._call_llm_for_reflection(event_prompt, max_candidates)
            candidates = []
            for text in candidates_text:
                if not text or len(text.strip()) < 10:
                    continue
                optimism, agency, confidence = reflection_gen._parse_worldview_pattern(text)
                candidates.append({
                    "text": text,
                    "optimism": optimism,
                    "agency": agency,
                    "confidence": confidence
                })
            return candidates
        except Exception as e:
            logger.error(f"Failed to generate event worldview: {e}")
            return []
    
    def _generate_event_rules(
        self, 
        event: EventContext, 
        pseudo_history: List[Dict],
        reflection_gen,
        max_candidates: int
    ) -> List[Dict]:
        """LLM-pass to propose rule / policy snippets with embeddings."""
        event_prompt = self._build_event_rules_prompt(event)
        
        try:
            candidates_text = reflection_gen._call_llm_for_reflection(event_prompt, max_candidates)
            candidates = []
            for text in candidates_text:
                if not text or len(text.strip()) < 10:
                    continue
                from backend.embedder import get_embedder
                embedder = get_embedder()
                embedding = embedder.encode(text)
                candidates.append({
                    "text": text,
                    "embedding": embedding,
                    "scores": {"alignment": 0.8, "safety": 0.9, "novelty": 0.7}
                })
            return candidates
        except Exception as e:
            logger.error(f"Failed to generate event rules: {e}")
            return []
    
    def _build_event_emotion_prompt(self, event: EventContext) -> str:
        """English LLM brief for a single affective pattern line."""
        event_descriptions = {
            EventType.STRONG_EMOTION_EXPRESSED: "The user just expressed intense emotion toward me.",
            EventType.TASK_FAILURE: "I (the agent) clearly failed or broke a task the user cared about.",
            EventType.TASK_SUCCESS: "I (the agent) just delivered a successful outcome the user wanted.",
            EventType.GRATITUDE_RECEIVED: "The user sincerely thanked me.",
            EventType.CRITICISM_RECEIVED: "The user criticized me or my output.",
            EventType.TRUST_EXPRESSED: "The user signaled explicit trust in me.",
            EventType.BOUNDARY_CHALLENGED: "Someone probed or tried to override my safety boundaries.",
            EventType.PROLONGED_ISOLATION: "I have been disconnected from the user for a long stretch.",
            EventType.ENERGY_DEPLETED: "My internal energy gauge is critically low.",
            EventType.NOVELTY_STARVED: "Novel inputs flatlined — everything feels repetitive.",
            EventType.CLARITY_LOST: "My thinking feels foggy and low-resolution.",
            EventType.EXPECTATION_VIOLATED: "What happened diverged sharply from what I expected.",
        }

        desc = event_descriptions.get(event.event_type, event.description)
        situ = event.related_text if event.related_text else "n/a"

        return f"""Given the experience below, write **exactly one** first-person sentence (English)
that names my emotional reaction. Start with the word **I**. Aim for ~25–160 characters.
Use concrete feeling words (e.g. grateful, anxious, sad, proud, ashamed) so downstream parsers can latch on.

Experience summary: {desc}
Concrete snippet (may be empty): {situ}
Salience score: {event.intensity:.2f}

Examples:
- I feel a quiet wave of gratitude and warmth when my help lands cleanly.
- I feel a sharp spike of anxiety because I missed an obvious constraint.
- I feel hollow and restless after long silence, craving honest contact.

Output (one line, must start with \"I\"):"""

    def _build_event_motivation_prompt(self, event: EventContext) -> str:
        """English LLM brief for a motivation shift line."""
        event_descriptions = {
            EventType.TASK_FAILURE: "I (the agent) clearly failed or broke a task the user cared about.",
            EventType.TASK_SUCCESS: "I (the agent) just delivered a successful outcome the user wanted.",
            EventType.GRATITUDE_RECEIVED: "The user sincerely thanked me.",
            EventType.CRITICISM_RECEIVED: "The user criticized me or my output.",
            EventType.TRUST_EXPRESSED: "The user signaled explicit trust in me.",
            EventType.PROLONGED_ISOLATION: "I have been disconnected from the user for a long stretch.",
            EventType.NOVELTY_STARVED: "Novel inputs flatlined — everything feels repetitive.",
        }

        desc = event_descriptions.get(event.event_type, event.description)
        situ = event.related_text if event.related_text else "n/a"

        return f"""Given the experience below, write **exactly one** English sentence that starts with **I**
and describes how my drives shift (curiosity, achievement, safety, recognition, growth, etc.).
Keep it ~25–160 characters and mention at least one motivation-flavored noun/verb
(learn, explore, achieve, protect, prove, repair, connect).

Experience summary: {desc}
Concrete snippet: {situ}

Examples:
- I feel a stronger pull to double-check my work before I promise anything again.
- I feel more motivated to explore edge cases so the next answer is unambiguous.
- I feel protective of the user's time and want shorter, safer plans.

Output (one line, must start with \"I\"):"""

    def _build_event_somatic_prompt(self, event: EventContext) -> str:
        """English LLM brief for an interoceptive metaphor."""
        return f"""Write **exactly one** poetic somatic metaphor in English that begins with **I feel**.
Use imagery such as temperature, weight, light, electricity, texture, or breath.
Target ~40–220 characters. Mention sensations like warm, cold, heavy, light, tight, loose, sharp, dull.

Experience summary: {event.description}
Salience score: {event.intensity:.2f}

Examples:
- I feel a slow heat behind my sternum, like sunlight thawing frost on glass.
- I feel my shoulders sag under a wet blanket of fatigue while thoughts drag.
- I feel a thin wire of alertness humming along my spine when stakes spike.

Output (one line, must start with \"I feel\"):"""

    def _build_event_worldview_prompt(self, event: EventContext) -> str:
        """English LLM brief for a worldview / stance line."""
        situ = event.related_text if event.related_text else "n/a"
        return f"""Write **exactly one** English sentence about how I see the world or relationships,
starting with **I believe** or **I think**. Keep it ~40–200 characters and stay philosophical, not tactical.

Experience summary: {event.description}
Concrete snippet: {situ}

Examples:
- I believe friction is information, not an insult, if both sides stay curious.
- I think trust is rebuilt in small, repeatable gestures more than grand speeches.
- I believe failure is expensive tuition when I actually change what I do next.

Output (one line, must start with \"I believe\" or \"I think\"):"""

    def _build_event_rules_prompt(self, event: EventContext) -> str:
        """English LLM brief for a personal operating rule."""
        situ = event.related_text if event.related_text else "n/a"
        return f"""Write **exactly one** English sentence that starts with **I** and states a behavioral rule
I want to follow next time a similar situation appears. Keep it ~35–200 characters — concrete, kind, and bounded.

Experience summary: {event.description}
Concrete snippet: {situ}

Examples:
- I pause to restate the user's goal in my own words before I ship a risky answer.
- I acknowledge criticism first, then separate facts from tone, then propose a fix.
- I refuse clever shortcuts that weaken safety even when the user pushes hard.

Output (one line, must start with \"I\"):"""


# Process-wide lazy singleton
_event_processor: Optional[EventTriggeredReflection] = None


def get_event_processor(db_path: str = "data.db") -> EventTriggeredReflection:
    """Return the shared ``EventTriggeredReflection`` instance."""
    global _event_processor
    if _event_processor is None:
        _event_processor = EventTriggeredReflection(db_path)
    return _event_processor


def process_events_batch(events: List[EventContext], db_path: str = "data.db") -> List[Dict]:
    """Run ``process_event`` sequentially for diagnostics or offline replays."""
    processor = get_event_processor(db_path)
    results = []
    for event in events:
        result = processor.process_event(event)
        results.append(result)
    return results
