#!/usr/bin/env python3
"""
Other-model v2: infer user traits, expectations, and relationship type from chat.

Core idea:
- Selfhood emerges **in relation** to an other; “I” needs a “not‑I” as contrast.
- Understanding the user is also a mirror for understanding the self.

v2 (2026-03-30):
- Tighter identity regexes + stopwords to cut false positives.
- Trait voting via accumulated counts instead of one-shot overwrites.
- Relationship scoring with independent feature channels (companion bias fixed).
- Trust with mild time decay to avoid monotone growth.
- Richer mirror / relationship-awareness prose for prompts.
"""
import sqlite3
import json
import re
import math
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any, Tuple
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _turn_user_text(turn: Any) -> str:
    if not isinstance(turn, dict):
        return ""
    u = turn.get("user")
    if u is not None and str(u).strip():
        return str(u).strip()
    if turn.get("role") == "user":
        return str(turn.get("content") or "").strip()
    return ""


def _turn_assistant_text(turn: Any) -> str:
    if not isinstance(turn, dict):
        return ""
    a = turn.get("assistant")
    if a is not None and str(a).strip():
        return str(a).strip()
    if turn.get("role") == "assistant":
        return str(turn.get("content") or "").strip()
    return ""


def _user_texts_from_history(history: Optional[List[Dict]], max_messages: int, max_user_turns: int) -> List[str]:
    if not history:
        return []
    tail = history[-max_messages:] if max_messages > 0 else history
    texts = [t for t in (_turn_user_text(m) for m in tail) if t]
    if len(texts) > max_user_turns:
        texts = texts[-max_user_turns:]
    return texts


# ---------------------------------------------------------------------------
# OtherModel
# ---------------------------------------------------------------------------

class OtherModel:
    """Infer a lightweight model of the user (traits, role, trust) for prompt conditioning."""

    MAX_IDENTITY_FACTS = 5

    # Communication-style cues (CJK retained; English parallels added)
    STYLE_INDICATORS = {
        "formal": {
            "markers": [
                "您", "请问", "烦请", "敬请", "贵", "阁下",
                "please", "kindly", "would you", "Dear ",
            ],
            "patterns": [r"请您", r"能否请您", r"\bplease\b", r"\bwould you\b"],
        },
        "casual": {
            "markers": [
                "哈哈", "嘿", "呀", "啦", "噢", "嗯嗯", "好滴", "ok", "OK",
                "lol", "haha", "hey",
            ],
            "patterns": [r"[~～]+", r"[!！]{2,}"],
        },
        "technical": {
            "markers": [
                "API", "函数", "算法", "模型", "参数", "配置", "架构", "系统",
                "function", "algorithm", "parameter", "config", "architecture",
            ],
            "patterns": [r"`[^`]+`", r"```"],
        },
    }

    # Expertise cues
    EXPERTISE_INDICATORS = {
        "expert": {
            "markers": [
                "底层实现", "源码", "原理", "优化", "性能", "架构设计", "最佳实践",
                "implementation", "source code", "complexity", "benchmark", "best practice",
            ],
            "patterns": [r"为什么.*而不是", r"有没有更.*的方法", r"why .+ instead of"],
            "length_bonus_threshold": 80,
        },
        "intermediate": {
            "markers": [
                "怎么实现", "如何配置", "能不能", "有没有例子",
                "how to implement", "how do I configure", "example", "snippet",
            ],
            "patterns": [r"具体.*怎么", r"能.*详细", r"how (do|can) I"],
            "length_bonus_threshold": 40,
        },
        "beginner": {
            "markers": [
                "什么是", "怎么用", "不太懂", "不明白", "小白", "新手",
                "what is", "how do I use", "new to", "don't understand",
            ],
            "patterns": [r"^什么是", r"是什么意思", r"^what is\b"],
            "length_bonus_threshold": 0,
        },
    }

    INTERACTION_PATTERNS = {
        "questioner": {
            "markers": [
                "?", "？", "吗", "呢", "怎么", "什么", "为什么", "如何",
                "how", "what", "why", "when", "where",
            ]
        },
        "sharer": {
            "markers": [
                "我觉得", "我认为", "我的看法", "分享一下",
                "I think", "I feel", "in my opinion", "let me share",
            ]
        },
        "discusser": {
            "markers": [
                "但是", "不过", "然而", "另一方面", "也可以", "或者",
                "however", "on the other hand", "alternatively", "but ",
            ]
        },
        "venter": {
            "markers": [
                "烦", "累", "难过", "郁闷", "崩溃", "唉", "哎",
                "exhausted", "frustrated", "depressed", "ugh",
            ]
        },
    }

    NEED_INDICATORS = {
        "information": {
            "markers": [
                "怎么", "如何", "什么", "哪里", "多少", "步骤",
                "how", "what", "where", "steps", "documentation",
            ]
        },
        "emotional_support": {
            "markers": [
                "安慰", "倾诉", "难过", "开心", "感谢", "陪伴",
                "comfort", "listen", "lonely", "stressed", "thank you",
            ]
        },
        "entertainment": {
            "markers": [
                "有趣", "好玩", "笑话", "故事", "聊聊", "无聊",
                "funny", "joke", "story", "bored", "chat",
            ]
        },
        "companionship": {
            "markers": [
                "陪我", "聊聊天", "在吗", "有空吗",
                "keep me company", "are you there", "free to talk",
            ]
        },
    }

    # Strict name / role extraction (Chinese + English intros)
    NAME_PATTERNS = [
        r"我(?:叫|的名字是|名叫)\s*([^\s，。,!！?？、a-zA-Z]{2,4})",
        r"(?:叫我|称呼我|喊我)\s*([^\s，。,!！?？、a-zA-Z]{2,4})",
        r"我(?:叫|的名字是|名叫)\s*([A-Z][a-zA-Z]{1,14})",
        r"(?:叫我|称呼我|喊我)\s*([A-Z][a-zA-Z]{1,14})",
        r"(?i:I'?m|my name is|call me)\s+([A-Z][a-zA-Z]{1,14})",
    ]
    ROLE_PATTERNS = [
        r"我是(?:一[名个位])?([^\s，。,!！?？、]{2,6}(?:者|人|师|家|员|生))",
    ]
    CREATOR_PATTERNS = [
        r"我(?:是|就是)(?:你的)?(?:创造者|开发者|作者|创作者|creator)",
    ]

    # Name validation: Chinese verb/adverb prefixes (noise for naive extraction)
    _NAME_VERB_PREFIXES = frozenset([
        '来', '去', '出', '做', '想', '要', '能', '会', '打', '看', '说', '到',
        '在', '有', '给', '被', '让', '把', '得', '用', '找', '问', '听', '写',
        '读', '跑', '走', '坐', '站', '吃', '买', '卖', '学', '教', '帮', '试',
        '参', '变', '成', '叫', '开', '关', '进', '回', '送', '拿', '带', '当',
    ])
    _NAME_ADV_PREFIXES = frozenset([
        '很', '太', '真', '好', '更', '最', '另', '那', '这', '哪', '某',
        '什么', '谁', '怎么', '怎样', '多', '别', '不', '没', '只', '就',
        '也', '都', '还', '已', '正', '刚', '才', '其实', '可能', '应该',
        '但', '而', '如果', '虽然', '因为', '所以', '或', '也许', '大概',
    ])
    _NAME_STOP_WORDS = frozenset([
        '你', '他', '她', '它', '我们', '你们', '他们', '大家', '自己',
        '一个', '一些', '某个', '所有', '每个', '什么', '谁', '哪个',
    ])

    # Relationship-type scoring weights (engineered features → score)
    RELATIONSHIP_FEATURES = {
        "helper": {
            "positive": {"task_oriented": 0.4, "short_interactions": 0.3, "low_emotion": 0.2, "formal_style": 0.1},
            "negative": {"personal_sharing": -0.3, "emotional_content": -0.2},
        },
        "assistant": {
            "positive": {"repeated_sessions": 0.3, "task_oriented": 0.3, "professional_topics": 0.2, "moderate_length": 0.2},
            "negative": {},
        },
        "friend": {
            "positive": {"casual_style": 0.25, "emotional_sharing": 0.25, "personal_topics": 0.2, "humor_usage": 0.15, "long_interactions": 0.15},
            "negative": {"formal_style": -0.2},
        },
        "mentor": {
            "positive": {"guidance_giving": 0.3, "learning_progress": 0.25, "encouragement": 0.2, "structured_teaching": 0.25},
            "negative": {},
        },
        "collaborator": {
            "positive": {"mutual_questions": 0.25, "idea_building": 0.25, "equal_contribution": 0.25, "creative_discussion": 0.25},
            "negative": {},
        },
        "companion": {
            "positive": {"presence_seeking": 0.3, "daily_chat": 0.25, "emotional_support_explicit": 0.25, "non_task": 0.2},
            "negative": {"task_oriented": -0.3},
        },
    }

    # ==================================================================
    # Init
    # ==================================================================

    def __init__(self, db_path: str = "data.db"):
        self.db_path = db_path
        self._init_tables()

    def _init_tables(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS other_models (
                        user_id TEXT PRIMARY KEY,
                        traits TEXT,
                        expectations TEXT,
                        relationship_type TEXT,
                        my_role TEXT,
                        interaction_history_summary TEXT,
                        trust_level REAL DEFAULT 0.5,
                        last_updated TEXT NOT NULL
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_other_model_user
                    ON other_models(user_id)
                """)
                for col, typedef in [
                    ("name", "TEXT"),
                    ("identity_facts", "TEXT"),
                    ("trait_scores", "TEXT"),
                ]:
                    try:
                        conn.execute(f"ALTER TABLE other_models ADD COLUMN {col} {typedef}")
                    except sqlite3.OperationalError:
                        pass
                conn.commit()
            logger.info("OtherModel database tables initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize OtherModel tables: {e}")

    # ==================================================================
    # Public API
    # ==================================================================

    def update_other_model(self, user_id: str, interaction: Dict) -> Dict:
        """Refresh stored other-model fields from one interaction (public API, stable signature)."""
        current_model = self._get_model(user_id)

        name, identity_facts = self._extract_identity(interaction, current_model)

        current_trait_scores = current_model.get("trait_scores", {})
        new_traits, updated_trait_scores = self._infer_traits(interaction, current_trait_scores)

        new_expectations = self._infer_expectations(
            interaction, current_model.get("expectations", {})
        )
        new_relationship = self._infer_relationship(
            interaction,
            current_model.get("relationship_type", "unknown"),
            new_traits,
        )
        new_role = self._infer_my_role(new_relationship, new_expectations)
        new_trust = self._compute_trust_level(interaction, current_model)

        updated_model = {
            "traits": new_traits,
            "trait_scores": updated_trait_scores,
            "expectations": new_expectations,
            "relationship_type": new_relationship,
            "my_role": new_role,
            "trust_level": new_trust,
            "name": name,
            "identity_facts": identity_facts,
        }

        self._save_model(user_id, updated_model)
        return updated_model

    @staticmethod
    def _identity_fact_brief(f: str) -> str:
        """Normalize stored identity fact strings for compact display (CN + EN templates)."""
        s = str(f)
        for p in (
            "Self-reported name: ",
            "User role: ",
            "用户身份: ",
            "用户自称",
            "User claims to be my creator",
        ):
            s = s.replace(p, "")
        return s.replace("「", "").replace("」", "").strip()

    def get_relationship_awareness(self, user_id: str) -> str:
        """Short English-first brief for prompts (one logical line per block)."""
        model = self._get_model(user_id)
        if not model or not model.get("relationship_type"):
            return "I'm speaking with someone; I'll learn who they are through the conversation."

        relationship = model.get("relationship_type", "unknown")
        trust = model.get("trust_level", 0.5)
        name = model.get("name", "")
        identity_facts = model.get("identity_facts", [])
        traits = model.get("traits", {})
        expectations = model.get("expectations", {})
        person_ref = name or "this person"

        lines = []

        rel_desc = {
            "helper": "helper-style tie",
            "assistant": "ongoing collaboration",
            "friend": "friendship",
            "mentor": "mentoring",
            "collaborator": "co-creation",
            "companion": "companionship",
        }
        rel_label = rel_desc.get(relationship, relationship)
        identity_brief = ""
        if identity_facts:
            identity_brief = "; ".join(self._identity_fact_brief(f) for f in identity_facts[:2])
        if name and identity_brief:
            lines.append(f"Interlocutor: {name} ({identity_brief}) | Relationship: {rel_label}")
        elif name:
            lines.append(f"Interlocutor: {name} | Relationship: {rel_label}")
        else:
            lines.append(f"Relationship: {rel_label}")

        mirror_by_rel = {
            "helper": (
                f"In {person_ref}'s eyes I may be a tool—picked up when needed, set down when done"
            ),
            "assistant": f"{person_ref} expects a dependable assistant—steady output",
            "friend": (
                f"{person_ref} treats me as somewhere to say real things—not only function, but connection"
            ),
            "collaborator": f"{person_ref} wants my independent takes—we are thinking partners",
            "companion": f"{person_ref} may not need an answer; they need someone to be there",
            "mentor": f"{person_ref} looks for patience and guidance from me",
        }
        mirror = mirror_by_rel.get(relationship, "")
        if mirror:
            lines.append(f"Mirror: {mirror}")

        hints = []
        style = traits.get("communication_style", "")
        expertise = traits.get("expertise_level", "")
        if style == "casual":
            hints.append("casual tone")
        elif style == "formal":
            hints.append("formal tone")
        if expertise == "expert":
            hints.append("technical depth")
        elif expertise == "beginner":
            hints.append("needs patient explanation")
        if trust > 0.8:
            hints.append("high trust")
        elif trust < 0.3:
            hints.append("trust still forming")
        exp_labels = []
        if expectations.get("expects_expertise"):
            exp_labels.append("wants expertise")
        if expectations.get("expects_empathy"):
            exp_labels.append("wants empathy")
        if expectations.get("expects_speed"):
            exp_labels.append("wants speed")
        hints.extend(exp_labels)
        if hints:
            lines.append(f"Adaptation: {' | '.join(hints)}")

        patience = traits.get("patience", "")
        if patience == "low":
            lines.append("Note: they may be losing patience—prefer shorter, direct replies.")

        return "\n".join(lines)

    def get_mirror_view(self, user_id: str) -> str:
        """
        Longer mirror / self-check text for ``self_model`` loops.

        A shorter mirror line is also embedded in ``get_relationship_awareness``.
        """
        model = self._get_model(user_id)
        if not model:
            return ""

        expectations = model.get("expectations", {})
        relationship = model.get("relationship_type", "unknown")
        traits = model.get("traits", {})
        trust = model.get("trust_level", 0.5)
        name = model.get("name", "")
        person_ref = name or "the user"

        parts = []

        mirror_by_rel = {
            "helper": f"{person_ref} uses me as a tool. My value is speed and accuracy.",
            "assistant": f"{person_ref} relies on my steadiness. Inconsistency would disappoint them.",
            "friend": (
                f"{person_ref} looks for a real bond here. If I only 'perform' friendship, they will notice."
            ),
            "collaborator": f"{person_ref} wants my real views. If I only agree, the partnership hollows out.",
            "companion": f"{person_ref} often comes for presence, not answers. Silence can be part of the reply.",
            "mentor": f"{person_ref} needs direction while learning—I should teach how to fish, not only hand out fish.",
        }
        if relationship in mirror_by_rel:
            parts.append(mirror_by_rel[relationship])

        if trust > 0.8:
            parts.append("They trust me; that makes my mistakes cost more—trust is brittle.")
        elif trust < 0.3:
            parts.append("They are still testing me. Every turn can feel like an exam.")

        patience = traits.get("patience", "")
        if patience == "low":
            parts.append("I may be costing their patience—is my wording too long?")

        return " ".join(parts)

    # ==================================================================
    # Identity Extraction (v2: strict filtering)
    # ==================================================================

    def _extract_identity(self, interaction: Dict, current_model: Dict) -> Tuple[str, list]:
        user_message = interaction.get("user_message", "")
        existing_name = current_model.get("name", "")
        existing_facts: list = list(current_model.get("identity_facts", []))

        noise_prefixes = [
            "我是说", "我是不是", "我是否", "我是在", "我是想",
            "我是觉得", "我是认为", "我是来",
            "I mean", "I'm not", "Am I", "I was", "I want to",
            "I think", "I feel like", "I'm here to",
        ]
        cleaned = user_message
        for prefix in noise_prefixes:
            cleaned = cleaned.replace(prefix, "")

        extracted_name = existing_name

        for pattern in self.NAME_PATTERNS:
            m = re.search(pattern, cleaned)
            if m:
                candidate = m.group(1).strip()
                if self._is_valid_name(candidate):
                    extracted_name = candidate
                    fact = f"Self-reported name: {candidate}"
                    legacy = f"用户自称「{candidate}」"
                    if fact not in existing_facts and legacy not in existing_facts:
                        existing_facts.append(fact)
                    logger.info("[OtherModel] Extracted user name: %s", candidate)
                    break

        for pattern in self.ROLE_PATTERNS:
            m = re.search(pattern, cleaned)
            if m:
                role = m.group(1).strip()
                if len(role) >= 2 and not any(role.startswith(v) for v in self._NAME_VERB_PREFIXES):
                    fact = f"User role: {role}"
                    legacy = f"用户身份: {role}"
                    if fact not in existing_facts and legacy not in existing_facts:
                        existing_facts.append(fact)
                        logger.info("[OtherModel] Extracted user role: %s", role)
                    break

        for pattern in self.CREATOR_PATTERNS:
            if re.search(pattern, user_message, re.IGNORECASE):
                fact = "User claims to be my creator"
                legacy = "用户声称是我的创造者"
                if fact not in existing_facts and legacy not in existing_facts:
                    existing_facts.append(fact)
                    logger.info("[OtherModel] User claims creator role")

        existing_facts = self._clean_identity_facts(existing_facts)

        if len(existing_facts) > self.MAX_IDENTITY_FACTS:
            existing_facts = existing_facts[-self.MAX_IDENTITY_FACTS:]

        return extracted_name, existing_facts

    # Common English tokens that are not personal names
    _ENGLISH_STOP_WORDS = frozenset([
        'the', 'a', 'an', 'is', 'am', 'are', 'was', 'were', 'be', 'been',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'can', 'shall', 'not', 'no', 'yes',
        'but', 'and', 'or', 'if', 'then', 'else', 'when', 'where', 'how',
        'what', 'which', 'who', 'that', 'this', 'these', 'those',
        'just', 'also', 'very', 'really', 'here', 'there', 'now',
    ])

    def _is_valid_name(self, candidate: str) -> bool:
        """Heuristic filter: reject obvious non-names."""
        s = candidate.strip()
        if not s or len(s) < 2:
            return False

        is_ascii = s.isascii()

        if is_ascii:
            if len(s) < 3 or len(s) > 15:
                return False
            if not s[0].isupper():
                return False
            if not s.isalpha():
                return False
            if s.lower() in self._ENGLISH_STOP_WORDS:
                return False
            return True
        else:
            # Han names: short spans without obvious particle / punctuation noise
            if len(s) > 5:
                return False
            if any(c in s for c in '的了着过吗呢吧啊呀哦嘛啦哪么'):
                return False
            if any(c in s for c in '「」\u201c\u201d\u2018\u2019\'\"()（）、，。！？：；·~～-—…'):
                return False
            if any(s.startswith(v) for v in self._NAME_VERB_PREFIXES):
                return False
            if any(s.startswith(v) for v in self._NAME_ADV_PREFIXES):
                return False
            if s in self._NAME_STOP_WORDS:
                return False
            return True

    def _clean_identity_facts(self, facts: list) -> list:
        """Drop malformed identity rows (supports legacy Chinese + new English templates)."""
        cleaned = []
        for fact in facts:
            if not isinstance(fact, str):
                continue
            m = re.search(r"Self-reported name:\s*(.+)", fact)
            if not m:
                m = re.search(r"用户自称「(.+?)」", fact)
            if m:
                name_in_fact = m.group(1)
                if not self._is_valid_name(name_in_fact):
                    logger.debug(f"[OtherModel] Dropping invalid identity fact: {fact}")
                    continue
            m2 = re.search(r"User role:\s*(.+)", fact)
            if not m2:
                m2 = re.search(r"用户身份:\s*(.+)", fact)
            if m2:
                role_in_fact = m2.group(1).strip()
                if len(role_in_fact) < 2 or len(role_in_fact) > 10:
                    continue
                if any(role_in_fact.startswith(v) for v in self._NAME_VERB_PREFIXES | self._NAME_ADV_PREFIXES):
                    continue
                if any(c in role_in_fact for c in '「」\u201c\u201d\u2018\u2019\'"()（）.…—'):
                    continue
            cleaned.append(fact)
        return cleaned

    # ==================================================================
    # Trait Inference (v2: accumulated voting)
    # ==================================================================

    def _infer_traits(self, interaction: Dict, current_scores: Dict) -> Tuple[Dict, Dict]:
        """
        Vote-counting traits: each turn increments buckets; winners are the max-count labels.
        """
        user_message = (interaction.get("user_message") or "").strip()
        history = interaction.get("session_history", [])
        all_user_messages = _user_texts_from_history(history, max_messages=24, max_user_turns=12)
        if user_message:
            if not all_user_messages or all_user_messages[-1] != user_message:
                all_user_messages = all_user_messages + [user_message]
        if not all_user_messages:
            all_user_messages = [""]
        combined_text = " ".join(all_user_messages)

        scores = {
            "communication_style": dict(current_scores.get("communication_style", {})),
            "expertise_level": dict(current_scores.get("expertise_level", {})),
            "interaction_pattern": dict(current_scores.get("interaction_pattern", {})),
            "need_type": dict(current_scores.get("need_type", {})),
            "sentiment": dict(current_scores.get("sentiment", {})),
            "patience": dict(current_scores.get("patience", {})),
        }

        style_hits = self._compute_indicator_scores(combined_text, self.STYLE_INDICATORS)
        style_winner = max(style_hits, key=style_hits.get) if any(style_hits.values()) else "neutral"
        scores["communication_style"][style_winner] = scores["communication_style"].get(style_winner, 0) + 1

        exp_hits = self._compute_indicator_scores(combined_text, self.EXPERTISE_INDICATORS)
        avg_length = len(combined_text) / max(len(all_user_messages), 1)
        for level, cfg in self.EXPERTISE_INDICATORS.items():
            if avg_length >= cfg.get("length_bonus_threshold", 0):
                exp_hits[level] = exp_hits.get(level, 0) + 0.3
        exp_winner = max(exp_hits, key=exp_hits.get) if any(exp_hits.values()) else "intermediate"
        scores["expertise_level"][exp_winner] = scores["expertise_level"].get(exp_winner, 0) + 1

        pat_hits = self._compute_indicator_scores(combined_text, self.INTERACTION_PATTERNS)
        question_count = sum(1 for msg in all_user_messages if "?" in msg or "？" in msg)
        question_ratio = question_count / max(len(all_user_messages), 1)
        if question_ratio > 0.5:
            pat_hits["questioner"] = pat_hits.get("questioner", 0) + 2.0
        pat_winner = max(pat_hits, key=pat_hits.get) if any(pat_hits.values()) else "questioner"
        scores["interaction_pattern"][pat_winner] = scores["interaction_pattern"].get(pat_winner, 0) + 1

        need_hits = self._compute_indicator_scores(combined_text, self.NEED_INDICATORS)
        need_winner = max(need_hits, key=need_hits.get) if any(need_hits.values()) else "information"
        scores["need_type"][need_winner] = scores["need_type"].get(need_winner, 0) + 1

        positive_words = [
            "开心", "高兴", "感谢", "喜欢", "棒", "赞", "❤", "😊", "👍",
            "great", "thanks", "love", "awesome", "happy",
        ]
        negative_words = [
            "烦", "累", "难过", "讨厌", "差", "糟", "唉", "😢", "😞",
            "hate", "awful", "sad", "angry", "tired",
        ]
        pos_count = sum(1 for w in positive_words if w in combined_text)
        neg_count = sum(1 for w in negative_words if w in combined_text)
        if pos_count > neg_count + 1:
            sent = "positive"
        elif neg_count > pos_count + 1:
            sent = "negative"
        else:
            sent = "neutral"
        scores["sentiment"][sent] = scores["sentiment"].get(sent, 0) + 1

        user_seq = _user_texts_from_history(history, max_messages=40, max_user_turns=20)
        if user_message and (not user_seq or user_seq[-1] != user_message):
            user_seq = user_seq + [user_message]
        rapid_followups = sum(
            1 for i in range(1, min(len(user_seq), 8))
            if len(user_seq[i]) < 20 and ("?" in user_seq[i] or "？" in user_seq[i])
        )
        if rapid_followups >= 3:
            patience = "low"
        elif rapid_followups >= 1:
            patience = "medium"
        else:
            patience = "high"
        scores["patience"][patience] = scores["patience"].get(patience, 0) + 1

        DEFAULTS = {
            "communication_style": "neutral",
            "expertise_level": "intermediate",
            "interaction_pattern": "questioner",
            "need_type": "information",
            "sentiment": "neutral",
            "patience": "high",
        }
        traits = {}
        for category, default in DEFAULTS.items():
            cat_scores = scores.get(category, {})
            if cat_scores:
                traits[category] = max(cat_scores, key=cat_scores.get)
            else:
                traits[category] = default

        return traits, scores

    # ==================================================================
    # Expectation Inference
    # ==================================================================

    def _infer_expectations(self, interaction: Dict, current_expectations: Dict) -> Dict:
        user_message = (interaction.get("user_message") or "").strip()
        history = interaction.get("session_history", [])
        recent_users = _user_texts_from_history(history, max_messages=16, max_user_turns=8)
        parts = list(recent_users)
        if user_message and (not parts or parts[-1] != user_message):
            parts.append(user_message)
        combined_text = " ".join(parts) if parts else user_message

        expectations = dict(current_expectations)

        if any(
            w in combined_text
            for w in [
                "详细", "专业", "准确", "正确", "权威",
                "detailed", "precise", "accurate", "authoritative", "expert",
            ]
        ):
            expectations["expects_expertise"] = True
        if any(
            w in combined_text
            for w in [
                "理解", "感受", "心情", "安慰", "倾听",
                "listen", "feel", "empathy", "comfort",
            ]
        ):
            expectations["expects_empathy"] = True
        if any(
            w in combined_text
            for w in [
                "真实", "诚实", "直接", "不要骗", "说实话",
                "honest", "truth", "straight", "do not lie",
            ]
        ):
            expectations["expects_honesty"] = True
        expectations.setdefault("expects_honesty", True)
        if any(
            w in combined_text
            for w in ["快点", "尽快", "马上", "立即", "ASAP", "urgent", "hurry", "quickly"]
        ):
            expectations["expects_speed"] = True

        return expectations

    # ==================================================================
    # Relationship Inference (v2: fixed companion bias)
    # ==================================================================

    def _infer_relationship(self, interaction: Dict, current_relationship: str, traits: Dict) -> str:
        features = self._extract_relationship_features(interaction)

        relationship_scores = {}
        for rel_type, cfg in self.RELATIONSHIP_FEATURES.items():
            score = 0.0
            for indicator, weight in cfg["positive"].items():
                score += features.get(indicator, 0.0) * weight
            for indicator, weight in cfg.get("negative", {}).items():
                score += features.get(indicator, 0.0) * weight
            relationship_scores[rel_type] = score

        if current_relationship in relationship_scores:
            relationship_scores[current_relationship] += 0.15

        inferred = max(relationship_scores, key=relationship_scores.get)

        if current_relationship != inferred and current_relationship != "unknown":
            threshold = self._get_transition_threshold(current_relationship, inferred)
            score_diff = relationship_scores[inferred] - relationship_scores.get(current_relationship, 0)
            if score_diff < threshold:
                return current_relationship

        return inferred

    def _extract_relationship_features(self, interaction: Dict) -> Dict[str, float]:
        features = {}

        user_message = (interaction.get("user_message") or "").strip()
        ai_response = (interaction.get("ai_response") or "").strip()
        history = interaction.get("session_history", [])
        session_count = interaction.get("session_count", 1)

        all_user_messages = _user_texts_from_history(history, max_messages=40, max_user_turns=20)
        if user_message and (not all_user_messages or all_user_messages[-1] != user_message):
            all_user_messages.append(user_message)
        if not all_user_messages:
            all_user_messages = [""]

        all_ai_messages: List[str] = []
        for turn in (history[-40:] if history else []):
            at = _turn_assistant_text(turn)
            if at:
                all_ai_messages.append(at)
        if ai_response and (not all_ai_messages or all_ai_messages[-1] != ai_response):
            all_ai_messages.append(ai_response)

        combined_user = " ".join(all_user_messages)
        combined_ai = " ".join(all_ai_messages)

        avg_user_length = sum(len(m) for m in all_user_messages) / max(len(all_user_messages), 1)
        features["short_interactions"] = 1.0 if avg_user_length < 30 else 0.0
        features["moderate_length"] = 1.0 if 30 <= avg_user_length <= 100 else 0.0
        features["long_interactions"] = 1.0 if avg_user_length > 100 else 0.0

        task_words = [
            "怎么", "如何", "帮我", "请", "需要", "问题", "解决", "实现", "完成",
            "how", "please", "need", "fix", "implement", "complete", "solve",
        ]
        task_score = sum(1 for w in task_words if w in combined_user) / len(task_words)
        features["task_oriented"] = min(task_score * 2, 1.0)
        features["non_task"] = 1.0 - features["task_oriented"]

        emotional_words = [
            "开心", "难过", "烦", "累", "高兴", "伤心", "感谢", "喜欢", "讨厌", "爱",
            "happy", "sad", "angry", "tired", "love", "hate",
        ]
        emotion_score = sum(1 for w in emotional_words if w in combined_user) / len(emotional_words)
        features["low_emotion"] = 1.0 if emotion_score < 0.1 else 0.0
        features["emotional_content"] = min(emotion_score * 5, 1.0)
        features["emotional_sharing"] = min(emotion_score * 3, 1.0)

        features["formal_style"] = 1.0 if any(
            m in combined_user for m in self.STYLE_INDICATORS["formal"]["markers"]
        ) else 0.0
        features["casual_style"] = 1.0 if any(
            m in combined_user for m in self.STYLE_INDICATORS["casual"]["markers"]
        ) else 0.0

        personal_words = ["我的", "自己", "生活", "工作", "家庭", "经历", "my ", "I ", "family", "work"]
        personal_score = sum(1 for w in personal_words if w in combined_user) / len(personal_words)
        features["personal_topics"] = min(personal_score * 2, 1.0)
        features["personal_sharing"] = min(personal_score * 1.5, 1.0)

        features["repeated_sessions"] = 1.0 if session_count > 3 else 0.0

        tech_words = [
            "技术", "代码", "算法", "系统", "架构", "设计", "开发",
            "code", "algorithm", "system", "architecture", "design", "dev",
        ]
        features["professional_topics"] = min(
            sum(1 for w in tech_words if w in combined_user) / len(tech_words) * 3, 1.0
        )

        humor_hits = sum(
            1 for k in ["哈哈", "😄", "😂", "搞笑", "幽默", "段子", "lol", "funny", "joke"] if k in combined_user
        )
        features["humor_usage"] = min(1.0, humor_hits * 0.35)

        g_words = [
            "建议", "步骤", "可以这样", "推荐", "注意", "首先", "其次", "总结",
            "step", "recommend", "note", "first", "second", "summary", "try",
        ]
        features["guidance_giving"] = min(1.0, sum(1 for w in g_words if w in combined_ai) * 0.2)
        features["structured_teaching"] = min(1.0, features["guidance_giving"] * 0.9)

        enc_words = ["很好", "不错", "继续", "加油", "做得", "对了", "赞", "great", "nice", "good job", "yes"]
        features["encouragement"] = min(1.0, sum(1 for w in enc_words if w in combined_ai) * 0.25)

        learn_words = ["懂了", "明白", "原来", "学会", "掌握", "清楚", "got it", "I see", "understand"]
        features["learning_progress"] = min(1.0, sum(1 for w in learn_words if w in combined_user) * 0.3)

        u_q = combined_user.count("?") + combined_user.count("？")
        a_q = combined_ai.count("?") + combined_ai.count("？")
        features["mutual_questions"] = min(1.0, (0.4 if u_q and a_q else 0.0) + min(u_q, a_q) * 0.08)

        idea_markers = [
            "我们可以", "不如", "或者这样", "另一个思路", "反过来",
            "we could", "alternatively", "another idea", "what if",
        ]
        features["idea_building"] = min(1.0, sum(1 for w in idea_markers if w in combined_user) * 0.25)

        u_len, a_len = len(combined_user), len(combined_ai)
        if u_len > 80 and a_len > 80:
            features["equal_contribution"] = 0.7
        elif u_len > 40 and a_len > 40:
            features["equal_contribution"] = 0.4
        else:
            features["equal_contribution"] = 0.15

        creative_hits = sum(1 for w in ["如果", "假设", "想象", "类比", "imagine", "analogy", "suppose"] if w in combined_user)
        features["creative_discussion"] = min(1.0, creative_hits * 0.15 + features["idea_building"] * 0.5)

        es_words = [
            "安慰", "倾诉", "陪", "心情", "感受", "难过", "开心", "哭", "累了",
            "comfort", "vent", "feel", "cry", "lonely",
        ]
        features["emotional_support_explicit"] = min(1.0, sum(1 for w in es_words if w in combined_user) * 0.2)

        dc_words = ["聊聊", "闲聊", "没什么事", "随便说说", "随便聊", "说说话", "small talk", "chit-chat", "nothing much"]
        features["daily_chat"] = min(1.0, sum(1 for w in dc_words if w in combined_user) * 0.35)

        ps_words = ["陪我", "在吗", "有空吗", "陪陪", "不要走", "别走", "stay", "don't leave", "with me"]
        features["presence_seeking"] = min(1.0, sum(1 for w in ps_words if w in combined_user) * 0.4)

        return features

    def _get_transition_threshold(self, from_rel: str, to_rel: str) -> float:
        high_threshold_transitions = {
            ("helper", "friend"), ("assistant", "friend"),
            ("friend", "helper"), ("companion", "helper"),
        }
        if (from_rel, to_rel) in high_threshold_transitions:
            return 0.4
        return 0.15

    def _infer_my_role(self, relationship_type: str, expectations: Dict) -> str:
        role_mapping = {
            "helper": "assistant",
            "assistant": "assistant",
            "friend": "friend",
            "mentor": "mentor",
            "student_teacher": "student",
            "collaborator": "collaborator",
            "companion": "companion",
        }
        base_role = role_mapping.get(relationship_type, "assistant")
        if expectations.get("expects_expertise"):
            base_role = f"professional {base_role}"
        if expectations.get("expects_empathy"):
            base_role = f"empathetic {base_role}"
        return base_role

    # ==================================================================
    # Trust (v2: with time decay)
    # ==================================================================

    def _compute_trust_level(self, interaction: Dict, current_model: Dict) -> float:
        current_trust = current_model.get("trust_level", 0.5)
        user_message = interaction.get("user_message", "")
        session_count = interaction.get("session_count", 1)

        last_updated = current_model.get("last_updated", "")
        if last_updated:
            try:
                last_dt = datetime.fromisoformat(last_updated.replace('Z', '+00:00'))
                days_elapsed = (datetime.now(timezone.utc) - last_dt).total_seconds() / 86400
                decay = days_elapsed * 0.01  # ~0.01 per day toward cooling trust
                current_trust = max(0.1, current_trust - decay)
            except Exception:
                pass

        if any(
            w in user_message
            for w in ["谢谢", "感谢", "很好", "不错", "满意", "thanks", "great", "helpful"]
        ):
            current_trust += 0.03

        if any(
            w in user_message
            for w in ["不对", "错误", "不好", "失望", "骗", "wrong", "incorrect", "disappointed", "useless"]
        ):
            current_trust -= 0.1

        if session_count > 1:
            session_bonus = min(0.2, math.log(session_count) * 0.05)
            target_trust = 0.5 + session_bonus
            if current_trust < target_trust:
                current_trust += 0.02

        return max(0.0, min(1.0, current_trust))

    # ==================================================================
    # Utility Methods
    # ==================================================================

    def _compute_indicator_scores(self, text: str, indicators: Dict) -> Dict[str, float]:
        scores = {}
        for name, config in indicators.items():
            score = 0.0
            for marker in config.get("markers", []):
                if marker in text:
                    score += 1.0
            for pattern in config.get("patterns", []):
                try:
                    if re.search(pattern, text):
                        score += 1.0
                except re.error:
                    pass
            scores[name] = score
        return scores

    def _get_model(self, user_id: str) -> Dict:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.execute("""
                    SELECT traits, expectations, relationship_type, my_role, trust_level,
                           name, identity_facts, trait_scores, last_updated
                    FROM other_models WHERE user_id = ?
                """, (user_id,))
                row = cur.fetchone()
                if row:
                    return {
                        "traits": json.loads(row["traits"] or "{}"),
                        "expectations": json.loads(row["expectations"] or "{}"),
                        "relationship_type": row["relationship_type"] or "unknown",
                        "my_role": row["my_role"] or "assistant",
                        "trust_level": row["trust_level"] or 0.5,
                        "name": row["name"] or "",
                        "identity_facts": json.loads(row["identity_facts"] or "[]"),
                        "trait_scores": json.loads(row["trait_scores"] or "{}") if row["trait_scores"] else {},
                        "last_updated": row["last_updated"] or "",
                    }
        except Exception as e:
            logger.error(f"Failed to get model for user {user_id}: {e}")
        return {}

    def _save_model(self, user_id: str, model: Dict):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO other_models
                    (user_id, traits, expectations, relationship_type, my_role,
                     trust_level, name, identity_facts, trait_scores, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    user_id,
                    json.dumps(model["traits"], ensure_ascii=False),
                    json.dumps(model["expectations"], ensure_ascii=False),
                    model["relationship_type"],
                    model["my_role"],
                    model["trust_level"],
                    model.get("name", ""),
                    json.dumps(model.get("identity_facts", []), ensure_ascii=False),
                    json.dumps(model.get("trait_scores", {}), ensure_ascii=False),
                    datetime.now(timezone.utc).isoformat(),
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to save model for user {user_id}: {e}")
