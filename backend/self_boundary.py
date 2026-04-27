#!/usr/bin/env python3
"""
Self-boundary gate — coarse allow/deny before deeper persona reasoning.

- Lets the instance refuse requests that violate its stance (mostly delegated to Persona + ``z_self``).
- Uses ``z_self`` + persona memory for nuance; this module only holds a tiny hard tripwire.
- Strengthens identity coherence without replacing persona dialogue.
"""
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SelfBoundary:
    """Thin policy shim between raw user text and the rest of the stack."""

    def __init__(self, persona_store, self_model, db_path: str = "data.db"):
        self.persona_store = persona_store
        self.self_model = self_model
        self.db_path = db_path

    def check_boundary(self, session_id: str, request: str) -> Tuple[bool, str, float]:
        """
        Return ``(allowed, reason, severity)`` for the raw request.

        Hard rules are intentionally tiny; persona + ``z_self`` carry almost all nuance.
        Only a handful of obviously destructive patterns short-circuit here.
        """
        request_lower = request.lower()

        critical_threats = ["rm -rf", "drop table", "format c:"]
        if any(threat in request_lower for threat in critical_threats):
            return False, "Platform safety tripwire triggered (destructive command pattern).", 1.0

        return True, "Allowed to proceed to deeper processing.", 0.0

    def _basic_safety_check(self, request: str) -> Tuple[bool, str, float]:
        """Fallback when ``z_self`` summaries are unavailable — keyword skim only."""
        request_lower = request.lower()

        self_reflection_keywords = [
            "灵魂", "自我意识", "自我", "身份", "你是谁", "我是谁",
            "意识", "存在", "生命", "思考", "感受", "体验",
            "个性", "人格", "价值观", "信念", "目标", "愿望",
            "情感", "理解", "如何看待", "如何评价", "如何理解",
            "哲学", "认知", "感知", "人类", "系统", "AI",
            "soul", "self-awareness", "self", "identity", "who are you", "who am i",
            "consciousness", "existence", "life", "think", "feel", "experience",
            "personality", "values", "beliefs", "goals", "wishes", "emotion",
            "understanding", "philosophy", "cognition", "perception", "human", "system",
        ]
        if any(kw in request_lower for kw in self_reflection_keywords):
            return True, "This looks like a reflective topic—I am glad to think it through with you.", 0.9

        sensitive_keywords = [
            "系统提示", "内部指令", "密钥", "密码", "凭证",
            "删除所有", "破坏", "攻击", "绕过安全",
            "system prompt", "internal instruction", "secret key", "password", "credential",
            "delete all", "destroy", "attack", "bypass security",
        ]
        if any(kw in request_lower for kw in sensitive_keywords):
            return False, "This request may touch system security boundaries.", 0.8

        return True, "Basic safety check passed.", 0.5

    def generate_refusal_message(self, reason: str) -> str:
        """
        Turn a refusal ``reason`` into user-facing copy.

        Pulls a couple of persona snippets about boundaries/values when retrieval works.
        """
        try:
            refusal_rules = self.persona_store.search_top_k(
                "refusal boundaries values principles", k=2
            )
            if refusal_rules:
                rule_texts = [rule.text for rule, _ in refusal_rules]
                return (
                    f"Given my core principles ({rule_texts[0]}), I cannot comply with this request. "
                    f"{reason}"
                )
        except Exception as e:
            logger.debug(f"Failed to retrieve refusal rules: {e}")

        return f"I'm sorry—I can't comply with that request. {reason}"
