#!/usr/bin/env python3
"""
Multi-factor memory importance scorer (added 2026-04-12).

Centralizes heuristics that used to be scattered across modules. Dimensions:
- ``emotional_intensity``
- ``identity_relevance``
- ``relationship_depth``
- ``recency_score`` (fresh vs old)
- ``access_frequency_score``
- ``content_richness``
"""

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List, Tuple

from backend.config import config

logger = logging.getLogger(__name__)


@dataclass
class MemoryImportanceScore:
    """Structured importance breakdown for a single memory."""
    total_score: float              # blended score (0-1)
    emotional_intensity: float      # emotional salience (0-1)
    identity_relevance: float       # identity / selfhood cues (0-1)
    relationship_depth: float       # relational cues (0-1)
    recency_score: float           # freshness (0-1)
    access_frequency_score: float  # reuse signal (0-1)
    content_richness: float        # length / diversity (0-1)
    memory_type: str               # coarse category token
    protection_level: str          # tier label (ABSOLUTE … DISPOSABLE)


class MemoryImportanceEvaluator:
    """Heuristic scorer for how “sticky” a memory should be."""

    # Score thresholds align with ``MemoryCleaner`` tiers (retention is descriptive only).
    PROTECTION_LEVELS = {
        "ABSOLUTE": {"min_score": 0.9, "retention": "permanent"},
        "CRITICAL": {"min_score": 0.7, "retention": "5 years"},
        "HIGH": {"min_score": 0.5, "retention": "2 years"},
        "MEDIUM": {"min_score": 0.3, "retention": "1 year"},
        "LOW": {"min_score": 0.15, "retention": "6 months"},
        "DISPOSABLE": {"min_score": 0.0, "retention": "90 days"},
    }

    # Bilingual keyword probes (Chinese retained for mixed-language chat).
    IDENTITY_KEYWORDS = [
        '你是谁', '我是谁', '你的身份', '你的存在', '你觉得', '你认为',
        '你怎么看', '自我', '意识', '存在', '灵魂', '本质', 'selfing', 'self-becoming',
        '我存在', '我思考', '真实的', '涌现', '主体',
        'who are you', 'who am i', 'your identity', 'sense of self', 'consciousness',
        'existential', 'emergent self', 'i think therefore',
    ]

    RELATIONSHIP_KEYWORDS = [
        '我们', '你和我', '我们之间', '关系', '信任', '陪伴',
        '一起', '共同', '记住我', '不要忘记',
        'between us', 'our bond', 'trust you', 'stay with me', "don't forget me",
    ]

    INTIMATE_KEYWORDS = [
        '喜欢你', '讨厌你', '相信你', '依赖', '在乎', '担心你',
        '爱你', '恨你', '需要你', '离不开',
        'i love you', 'i hate you', 'i need you', "can't live without",
        'care about you', 'worried about you', 'depend on you',
    ]

    STRONG_POSITIVE_KEYWORDS = [
        '太感谢了', '非常感谢', '太棒了', '真的很好', '帮了大忙',
        '太厉害了', '太好了', '完美', '精彩',
        'thank you so much', 'life saver', 'absolutely amazing', 'perfect job',
    ]

    STRONG_NEGATIVE_KEYWORDS = [
        '非常失望', '很生气', '太糟糕', '彻底失败', '非常难过',
        '崩溃', '绝望', '太痛苦', '受不了',
        'utterly disappointed', 'furious', 'total failure', 'breaking down',
    ]

    POSITIVE_KEYWORDS = [
        '谢谢', '感谢', '很好', '开心', '喜欢', '爱', '赞',
        '厉害', '满意', '高兴', '快乐',
        'thanks', 'great job', 'happy', 'love it', 'awesome', 'pleased',
    ]

    NEGATIVE_KEYWORDS = [
        '难过', '失望', '生气', '不满', '糟糕', '失败',
        '抱歉', '对不起', '错误', '难受', '痛苦', '焦虑',
        'sad', 'angry', 'sorry', 'anxious', 'upset', 'failed', 'frustrated',
    ]
    
    def __init__(self, db_path: str = "data.db"):
        self.db_path = db_path
    
    def evaluate(
        self,
        content: str,
        user_input: str = "",
        assistant_response: str = "",
        created_at: Optional[datetime] = None,
        access_count: int = 0,
        last_accessed_at: Optional[datetime] = None,
    ) -> MemoryImportanceScore:
        """
        Score a memory along multiple axes and derive a protection tier.

        Args:
            content: Primary memory body.
            user_input: Optional user turn for extra cues.
            assistant_response: Optional assistant turn for extra cues.
            created_at: Creation timestamp (UTC-aware preferred).
            access_count: Number of prior reads.
            last_accessed_at: Last read timestamp, if any.

        Returns:
            ``MemoryImportanceScore`` with per-axis scores plus ``memory_type``.
        """
        all_text = f"{content} {user_input} {assistant_response}".lower()

        emotional_intensity = self._evaluate_emotional_intensity(all_text)

        identity_relevance = self._evaluate_identity_relevance(all_text)

        relationship_depth = self._evaluate_relationship_depth(all_text)

        recency_score = self._evaluate_recency(created_at)

        access_frequency_score = self._evaluate_access_frequency(
            access_count, last_accessed_at, created_at
        )

        content_richness = self._evaluate_content_richness(content, user_input, assistant_response)

        memory_type = self._determine_memory_type(
            identity_relevance, relationship_depth, emotional_intensity, all_text
        )

        # Tunable blend of the six signals.
        weights = {
            "emotional": 0.20,
            "identity": 0.25,
            "relationship": 0.20,
            "recency": 0.10,
            "access": 0.10,
            "richness": 0.15,
        }
        
        total_score = (
            emotional_intensity * weights["emotional"] +
            identity_relevance * weights["identity"] +
            relationship_depth * weights["relationship"] +
            recency_score * weights["recency"] +
            access_frequency_score * weights["access"] +
            content_richness * weights["richness"]
        )
        
        if identity_relevance > 0.7:
            total_score = max(total_score, 0.6)

        if emotional_intensity > 0.8:
            total_score = max(total_score, 0.5)

        protection_level = self._determine_protection_level(total_score, memory_type)
        
        return MemoryImportanceScore(
            total_score=min(1.0, total_score),
            emotional_intensity=emotional_intensity,
            identity_relevance=identity_relevance,
            relationship_depth=relationship_depth,
            recency_score=recency_score,
            access_frequency_score=access_frequency_score,
            content_richness=content_richness,
            memory_type=memory_type,
            protection_level=protection_level,
        )
    
    def _evaluate_emotional_intensity(self, text: str) -> float:
        """Map affective language to a 0-1 emotional salience score."""
        score = 0.0

        if any(kw in text for kw in self.STRONG_POSITIVE_KEYWORDS):
            score = 0.9
        elif any(kw in text for kw in self.STRONG_NEGATIVE_KEYWORDS):
            score = 0.85
        elif any(kw in text for kw in self.POSITIVE_KEYWORDS):
            score = 0.6
        elif any(kw in text for kw in self.NEGATIVE_KEYWORDS):
            score = 0.5

        exclamation_count = text.count('!') + text.count('！')
        if exclamation_count > 0:
            score = min(1.0, score + exclamation_count * 0.05)
        
        return score
    
    def _evaluate_identity_relevance(self, text: str) -> float:
        """Score how much the text probes selfhood / identity."""
        score = 0.0

        identity_matches = sum(1 for kw in self.IDENTITY_KEYWORDS if kw in text)
        if identity_matches >= 3:
            score = 0.9
        elif identity_matches >= 2:
            score = 0.7
        elif identity_matches >= 1:
            score = 0.5
        
        return score
    
    def _evaluate_relationship_depth(self, text: str) -> float:
        """Score relational / attachment language."""
        score = 0.0

        if any(kw in text for kw in self.INTIMATE_KEYWORDS):
            score = 0.9
        elif any(kw in text for kw in self.RELATIONSHIP_KEYWORDS):
            score = 0.6
        
        return score
    
    def _evaluate_recency(self, created_at: Optional[datetime]) -> float:
        """Fresh memories score higher (piecewise decay by age in days)."""
        if created_at is None:
            return 0.5
        
        now = datetime.now(timezone.utc)
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        
        days_old = (now - created_at).days

        if days_old <= 1:
            return 1.0
        elif days_old <= 7:
            return 0.9
        elif days_old <= 30:
            return 0.7
        elif days_old <= 90:
            return 0.5
        elif days_old <= 365:
            return 0.3
        else:
            return 0.1
    
    def _evaluate_access_frequency(
        self,
        access_count: int,
        last_accessed_at: Optional[datetime],
        created_at: Optional[datetime],
    ) -> float:
        """Boost memories that are revisited frequently relative to their age."""
        if access_count == 0:
            return 0.1

        if created_at is None:
            return min(1.0, access_count * 0.1)
        
        now = datetime.now(timezone.utc)
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
        
        days_since_creation = max(1, (now - created_at).days)
        avg_access_per_day = access_count / days_since_creation

        if avg_access_per_day >= 1:
            return 1.0
        elif avg_access_per_day >= 0.1:  # ~once per 10 days
            return 0.7
        elif avg_access_per_day >= 0.03:  # ~monthly
            return 0.5
        else:
            return 0.3
    
    def _evaluate_content_richness(
        self,
        content: str,
        user_input: str,
        assistant_response: str,
    ) -> float:
        """Combine length and light structural cues (questions, digits, newlines)."""
        total_length = len(content) + len(user_input) + len(assistant_response)

        if total_length > 1000:
            length_score = 0.8
        elif total_length > 500:
            length_score = 0.6
        elif total_length > 200:
            length_score = 0.4
        else:
            length_score = 0.2

        diversity_score = 0.0
        if '?' in content or '？' in content:
            diversity_score += 0.1
        if any(char.isdigit() for char in content):
            diversity_score += 0.1
        if '\n' in content:
            diversity_score += 0.1
        
        return min(1.0, length_score + diversity_score)
    
    def _determine_memory_type(
        self,
        identity_relevance: float,
        relationship_depth: float,
        emotional_intensity: float,
        text: str,
    ) -> str:
        """Assign a coarse memory family used for protection overrides."""
        if identity_relevance > 0.5:
            return "identity"
        elif relationship_depth > 0.5:
            return "relation"
        elif emotional_intensity > 0.6:
            return "emotional"
        elif any(
            kw in text
            for kw in (
                '代码', '函数', '报错', 'error', 'bug', 'api', 'stack trace', 'traceback',
                'exception', 'typescript', 'javascript', 'compile error',
            )
        ):
            return "technical"
        elif any(
            kw in text
            for kw in (
                '我喜欢', '我偏好', '我习惯', '记住',
                'i prefer', 'i like to', 'my habit', 'remember this preference',
            )
        ):
            return "semantic"
        else:
            return "episodic"

    def _determine_protection_level(self, total_score: float, memory_type: str) -> str:
        """Pick the strictest tier compatible with ``total_score`` and ``memory_type``."""
        if memory_type == "identity":
            return "ABSOLUTE"

        if memory_type == "relation" and total_score > 0.4:
            return "CRITICAL"

        for level, criteria in self.PROTECTION_LEVELS.items():
            if total_score >= criteria["min_score"]:
                return level
        
        return "DISPOSABLE"
    
    def batch_evaluate(self, memories: List[Dict]) -> List[MemoryImportanceScore]:
        """Run :meth:`evaluate` over a list of memory dicts."""
        results = []
        for memory in memories:
            score = self.evaluate(
                content=memory.get("content", ""),
                user_input=memory.get("user_input", ""),
                assistant_response=memory.get("assistant_response", ""),
                created_at=memory.get("created_at"),
                access_count=memory.get("access_count", 0),
                last_accessed_at=memory.get("last_accessed_at"),
            )
            results.append(score)
        return results


def evaluate_memory_importance(
    content: str,
    user_input: str = "",
    assistant_response: str = "",
    **kwargs
) -> MemoryImportanceScore:
    """Module-level helper wrapping :class:`MemoryImportanceEvaluator`."""
    evaluator = MemoryImportanceEvaluator()
    return evaluator.evaluate(content, user_input, assistant_response, **kwargs)


def get_protection_level(score: float, memory_type: str = "episodic") -> str:
    """Map ``(score, memory_type)`` to a protection tier via a short-lived evaluator."""
    evaluator = MemoryImportanceEvaluator()
    return evaluator._determine_protection_level(score, memory_type)
