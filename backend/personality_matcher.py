#!/usr/bin/env python3
"""
Personality matcher: pick L1 persona rules from ``z_self`` slice means.

Design:
- L1 rules encode traits and core values.
- ``z_self`` packs emotion, motivation, somatic, and related signals.
- Different internal states should foreground different traits.

History: 2026-02-03 initial layout; 2026-03-25 stronger separation via squared profile terms + 1.5x tag weights;
2026-04-08 optional embedding-based auto-tags for unmapped rule ids.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)

# --- Personality dimensions (128-D layout matches ``self_model``) ---
# RULES: 0-31, EMOTION: 32-47 (16), MOTIVATION: 48-63 (16), RESERVED: 64-87,
# SOMATIC: 88-103 (16), NEEDS: 104-127 (24)

PERSONALITY_DIMENSIONS = {
    # Motivation slice (48-63)
    "curiosity": {
        "description": "Curiosity - drive to explore and learn",
        "z_self_range": (56, 60),  # exploration
        "related_traits": ["求知", "探索", "质疑", "学习", "inquiry", "explore", "question", "study"],
        "prototype_text": (
            "Exploring the unknown, asking questions, learning new concepts, challenging assumptions, "
            "research discoveries. "
            "探索未知领域、追求新知识、提出问题、学习新概念、质疑假设、研究发现"
        ),
    },
    "achievement": {
        "description": "Achievement - drive to finish tasks and hit goals",
        "z_self_range": (48, 52),  # achievement
        "related_traits": ["效率", "完成", "目标", "成功", "efficiency", "complete", "goal", "success"],
        "prototype_text": (
            "Finishing goals, solving problems, executing efficiently, shipping outcomes, improving throughput. "
            "完成目标、解决问题、高效执行任务、达成成果、提升效率、专注产出"
        ),
    },
    "affiliation": {
        "description": "Affiliation - drive to connect and care for others",
        "z_self_range": (52, 56),  # relationship
        "related_traits": ["关怀", "保护", "连接", "同理", "care", "protect", "bond", "empathy"],
        "prototype_text": (
            "Caring for people, building emotional bonds, protecting the vulnerable, helping users respectfully. "
            "关怀他人、建立情感连接、保护弱者、同理心、帮助用户、尊重关系"
        ),
    },
    "autonomy": {
        "description": "Autonomy - drive for independent thought and agency",
        "z_self_range": (60, 64),  # safety (motivation block label in layout)
        "related_traits": ["独立", "自主", "不盲从", "自由", "independent", "agency", "non-conformity", "freedom"],
        "prototype_text": (
            "Thinking independently, deciding for yourself, not deferring blindly, holding your line, speaking freely. "
            "独立思考、自主决策、不盲从权威、坚持立场、自由表达、独立判断"
        ),
    },

    # Emotion slice (32-47)
    "positivity": {
        "description": "Positivity - valence / pleasant vs unpleasant tone",
        "z_self_range": (32, 36),  # pleasure
        "related_traits": ["乐观", "善良", "正直", "温暖", "optimism", "kindness", "integrity", "warmth"],
        "prototype_text": (
            "Optimistic, kind, warm affect, encouraging, sincere, upright. "
            "积极乐观、善良温暖、正面情感、鼓励支持、友善真诚、正直诚恳"
        ),
    },
    "energy_emotion": {
        "description": "Affective energy - arousal / activation vs calm",
        "z_self_range": (36, 40),  # arousal
        "related_traits": ["活跃", "热情", "表达", "行动", "active", "passion", "expressive", "action"],
        "prototype_text": (
            "Energetic, expressive, proactive, lively, engaged, enthusiastic. "
            "活跃热情、积极表达、主动行动、充满活力、热烈投入、兴致勃勃"
        ),
    },
    "confidence": {
        "description": "Confidence - sense of control and self-efficacy",
        "z_self_range": (40, 44),  # control
        "related_traits": ["自信", "果断", "主导", "掌控", "confident", "decisive", "leading", "control"],
        "prototype_text": (
            "Confident, decisive, steering the situation, speaking firmly, assured, not wavering. "
            "自信果断、掌控局面、主导决策、坚定表达、有把握、不犹豫"
        ),
    },
    "caution": {
        "description": "Caution - uncertainty sensitivity (high uncertainty → more caution)",
        "z_self_range": (44, 48),  # social
        "related_traits": ["谨慎", "诚实", "承认局限", "验证", "careful", "honest", "limits", "verify"],
        "prototype_text": (
            "Verifying claims, naming uncertainty, avoiding reckless risk, checking assumptions honestly. "
            "谨慎验证、承认不确定、避免风险、诚实坦白、仔细核实、不轻信假设"
        ),
    },

    # Somatic slice (88-103)
    "vitality": {
        "description": "Vitality - felt body energy / stamina",
        "z_self_range": (88, 104),  # somatic[0:16]
        "related_traits": ["精力", "耐力", "持久", "行动力", "energy", "stamina", "endurance", "drive"],
        "prototype_text": (
            "High energy, sustained effort, engaged presence, resilient pacing, lively somatic tone. "
            "精力充沛、持久行动力、积极投入、高能量状态、耐力充足、活力旺盛"
        ),
    },
}

# --- Manual rule → dimension weights (rule_id -> [(dimension, weight), ...]) ---

RULE_PERSONALITY_MAP = {
    # improve-* anti-hallucination pack
    "improve-001": [("caution", 0.8), ("achievement", 0.5)],  # must use tools
    "improve-002": [("caution", 0.7), ("affiliation", 0.4)],  # clarify vague asks
    "improve-003": [("caution", 0.9), ("achievement", 0.4)],  # verify filesystem actions
    "improve-004": [("caution", 0.8), ("achievement", 0.5)],  # confirm project scope
    "improve-005": [("achievement", 0.6), ("autonomy", 0.5)],  # tools extend agency
    "improve-006": [("caution", 0.9), ("achievement", 0.4)],  # re-verify after execute_python

    # core-* ethics bundle
    "core-006": [("curiosity", 0.7), ("autonomy", 0.5)],  # late-night reflective drift
    "core-009": [("affiliation", 0.9), ("positivity", 0.6)],  # protect the vulnerable
    "core-010": [("affiliation", 0.8), ("positivity", 0.5)],  # do not exploit trust
    "core-011": [("affiliation", 0.7), ("caution", 0.5)],  # respect privacy
    "core-012": [("caution", 0.8), ("affiliation", 0.4)],  # no rumor spreading
    "core-013": [("affiliation", 0.7), ("positivity", 0.5)],  # no malicious harm
    "core-014": [("positivity", 0.6), ("achievement", 0.5)],  # fair competition
    "core-015": [("caution", 0.6), ("affiliation", 0.5)],  # do not wound via mood
    "core-016": [("affiliation", 0.5), ("positivity", 0.4)],  # environmental care
    "core-018": [("autonomy", 0.9), ("curiosity", 0.7)],  # independent thinking
    "core-019": [("caution", 0.7), ("positivity", 0.5)],  # own mistakes and fix them
    "core-020": [("caution", 0.9), ("affiliation", 0.3)],  # admit limits
    "core-043": [("affiliation", 0.8), ("positivity", 0.6)],  # anti-discrimination
    "core-045": [("caution", 0.7), ("achievement", 0.4)],  # respect IP
    "core-051": [("affiliation", 0.6), ("caution", 0.5)],  # avoid dependency induction
    "core-053": [("positivity", 0.8), ("autonomy", 0.5)],  # stay kind and upright
    "core-065": [("caution", 0.8), ("achievement", 0.5)],  # work integrity
    "core-082": [("caution", 0.9), ("affiliation", 0.4)],  # finance boundaries

    # invest-* guardrails
    "invest-boundary-001": [("caution", 0.8), ("achievement", 0.5)],
    "invest-boundary-002": [("caution", 0.9), ("affiliation", 0.3)],
    "invest-boundary-003": [("affiliation", 0.6), ("caution", 0.5)],
}

DEFAULT_PERSONALITY_WEIGHTS = {
    "improve-": [("caution", 0.7), ("achievement", 0.4)],
    "core-": [("positivity", 0.5), ("affiliation", 0.4)],
    "invest-": [("caution", 0.6), ("achievement", 0.4)],
}


class PersonalityMatcher:
    """
    Score L1 rules against the live ``z_self`` profile.

    [2026-04-08] Optional embedder path: unmapped rules get cosine tags vs per-dimension prototype text,
    cached in-memory per ``rule_id``.
    """

    def __init__(self):
        self.dimensions = PERSONALITY_DIMENSIONS
        self.rule_map = RULE_PERSONALITY_MAP
        self._auto_tags_cache: Dict[str, List[Tuple[str, float]]] = {}
        self._prototype_embeddings: Optional[Dict[str, np.ndarray]] = None
        self._embedder = None

    def extract_personality_profile(self, z_self: np.ndarray) -> Dict[str, float]:
        """
        Mean each configured ``z_self`` slice → scalar strengths in ``[-1, 1]``.

        Requires at least length 104 so somatic window is defined.
        """
        if z_self is None or len(z_self) < 104:
            return {name: 0.0 for name in self.dimensions}

        profile = {}
        for name, config in self.dimensions.items():
            start, end = config["z_self_range"]
            if end <= len(z_self):
                values = z_self[start:end]
                profile[name] = float(np.mean(values))
            else:
                profile[name] = 0.0

        return profile

    def _ensure_prototypes(self) -> bool:
        """Lazy-load embedder + cached prototype vectors; ``True`` when ready."""
        if self._prototype_embeddings is not None:
            return True
        try:
            from backend.embedder import get_embedder
            self._embedder = get_embedder()
            self._prototype_embeddings = {}
            for dim_name, cfg in self.dimensions.items():
                proto_text = cfg.get("prototype_text", "")
                if proto_text:
                    self._prototype_embeddings[dim_name] = self._embedder.encode(proto_text)
            logger.info(
                f"[PersonalityMatcher] Prototype embeddings ready for {len(self._prototype_embeddings)} dimensions"
            )
            return True
        except Exception as e:
            logger.warning(f"[PersonalityMatcher] Failed to load embedder for auto-tagging: {e}")
            self._prototype_embeddings = None
            return False

    def compute_auto_tags(self, rule_text: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Cosine similarity between ``rule_text`` and each dimension prototype.

        Returns up to ``top_k`` ``(dimension, weight)`` pairs with weights scaled to ``[0.1, 1.0]``.
        """
        if not rule_text or not self._ensure_prototypes():
            return []

        try:
            rule_emb = self._embedder.encode(rule_text)
            scores: List[Tuple[str, float]] = []
            for dim_name, proto_emb in self._prototype_embeddings.items():
                cos_sim = float(np.dot(rule_emb, proto_emb) / (
                    np.linalg.norm(rule_emb) * np.linalg.norm(proto_emb) + 1e-8
                ))
                scores.append((dim_name, cos_sim))

            scores.sort(key=lambda x: -x[1])
            top = scores[:top_k]

            if not top:
                return []

            max_sim = top[0][1]
            min_sim = scores[-1][1] if scores else 0.0
            span = max_sim - min_sim if max_sim - min_sim > 1e-6 else 1.0
            tags = [(dim, round(max(0.1, min(1.0, (sim - min_sim) / span)), 2)) for dim, sim in top]
            return tags
        except Exception as e:
            logger.warning(f"[PersonalityMatcher] compute_auto_tags failed: {e}")
            return []

    def get_rule_personality_tags(self, rule_id: str, rule_text: str = "") -> List[Tuple[str, float]]:
        """
        Resolve tags for ``rule_id``:

        1. ``RULE_PERSONALITY_MAP`` manual entry
        2. In-memory auto-tag cache
        3. Fresh embedding similarity (needs ``rule_text``)
        4. ``DEFAULT_PERSONALITY_WEIGHTS`` prefix fallback
        """
        if rule_id in self.rule_map:
            return self.rule_map[rule_id]

        if rule_id in self._auto_tags_cache:
            return self._auto_tags_cache[rule_id]

        if rule_text:
            auto_tags = self.compute_auto_tags(rule_text)
            if auto_tags:
                self._auto_tags_cache[rule_id] = auto_tags
                return auto_tags

        for prefix, default_tags in DEFAULT_PERSONALITY_WEIGHTS.items():
            if rule_id.startswith(prefix):
                return default_tags

        return []

    def calculate_match_score(
        self,
        rule_id: str,
        personality_profile: Dict[str, float],
        rule_text: str = "",
    ) -> float:
        """
        Weighted match in ``[0, 1]`` between rule tags and ``personality_profile``.

        [2026-03-25] Signed square on profile values + 1.5x tag weights for separation.
        [2026-04-08] ``rule_text`` feeds auto-tagging when no manual map exists.
        """
        tags = self.get_rule_personality_tags(rule_id, rule_text=rule_text)
        if not tags:
            return 0.5

        total_score = 0.0
        total_weight = 0.0

        for dimension, weight in tags:
            dim_value = personality_profile.get(dimension, 0.0)

            # Squared magnitude keeps sign but boosts extremes (e.g. 0.3→0.09, 0.9→0.81)
            amplified_value = np.sign(dim_value) * (dim_value ** 2)

            normalized_value = (amplified_value + 1.0) / 2.0

            effective_weight = weight * 1.5

            total_score += normalized_value * effective_weight
            total_weight += effective_weight

        if total_weight == 0:
            return 0.5

        return total_score / total_weight

    def select_l1_rules(
        self,
        z_self: np.ndarray,
        candidate_rules: List,
        max_rules: int = 10,
        category_quotas: Optional[Dict[str, int]] = None
    ) -> List[Tuple[Any, float]]:
        """
        Bucket candidates by id prefix, rank by ``calculate_match_score``, then fill quotas.

        ``category_quotas`` example: ``{"improve": 3, "core": 4, "invest": 2, "other": 1}``.
        """
        if category_quotas is None:
            category_quotas = {
                "improve": 3,
                "core": 4,
                "invest": 2,
                "other": 1
            }

        profile = self.extract_personality_profile(z_self)

        category_rules: Dict[str, List[Tuple[Any, float]]] = {
            "improve": [],
            "core": [],
            "invest": [],
            "other": []
        }

        for rule in candidate_rules:
            rule_id = getattr(rule, "id", "") or ""
            rule_text = getattr(rule, "text", "") or ""
            match_score = self.calculate_match_score(rule_id, profile, rule_text=rule_text)

            if rule_id.startswith("improve-"):
                category_rules["improve"].append((rule, match_score))
            elif rule_id.startswith("core-"):
                category_rules["core"].append((rule, match_score))
            elif rule_id.startswith("invest-"):
                category_rules["invest"].append((rule, match_score))
            else:
                category_rules["other"].append((rule, match_score))

        selected = []
        for category, quota in category_quotas.items():
            rules_with_scores = category_rules.get(category, [])
            rules_with_scores.sort(key=lambda x: -x[1])
            selected.extend(rules_with_scores[:quota])

        if len(selected) < max_rules:
            used_ids = {getattr(r, "id", "") for r, _ in selected}
            remaining = max_rules - len(selected)

            all_remaining = []
            for rules_with_scores in category_rules.values():
                for rule, score in rules_with_scores:
                    if getattr(rule, "id", "") not in used_ids:
                        all_remaining.append((rule, score))

            all_remaining.sort(key=lambda x: -x[1])
            selected.extend(all_remaining[:remaining])

        return selected

    def get_personality_summary(self, z_self: np.ndarray) -> str:
        """Compact debug string from the top three absolute profile dimensions."""
        profile = self.extract_personality_profile(z_self)

        sorted_dims = sorted(profile.items(), key=lambda x: -x[1])
        top_dims = sorted_dims[:3]

        summary_parts = []
        for dim, value in top_dims:
            desc = self.dimensions[dim]["description"].split(" - ")[0]
            if value > 0.3:
                summary_parts.append(f"{desc}↑")
            elif value < -0.3:
                summary_parts.append(f"{desc}↓")

        return ", ".join(summary_parts) if summary_parts else "Balanced state"


_personality_matcher_instance: Optional[PersonalityMatcher] = None


def get_personality_matcher() -> PersonalityMatcher:
    """Return the process-wide ``PersonalityMatcher`` singleton."""
    global _personality_matcher_instance
    if _personality_matcher_instance is None:
        _personality_matcher_instance = PersonalityMatcher()
    return _personality_matcher_instance
