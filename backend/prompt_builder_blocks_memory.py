from typing import Optional, Dict, Any, Tuple, Union, List

from backend.config import config
from backend.prompt_builder_core import logger


def _salient_chat_match_limit() -> int:
    """Max salient chat snippets pulled from chat_turns (see self_narrative.retrieve_salient_chat_turn)."""
    v = int(config.get("parameters.memory.salient_chat_matches", 3) or 3)
    return max(1, min(8, v))


# [2026-02-26] Daily narrative retrieval gate
def _contains_any_keyword(text: str, keywords: List[str]) -> bool:
    lower = text.lower()
    for kw in keywords:
        if kw.isascii():
            if kw in lower:
                return True
        elif kw in text:
            return True
    return False


def _needs_daily_narrative(user_input: str) -> bool:
    """True when the user likely refers to past calendar time (daily narrative retrieval)."""
    time_keywords = [
        "昨天",
        "前天",
        "前几天",
        "上周",
        "之前",
        "过去几天",
        "这周",
        "最近",
        "前段时间",
        "那天",
        "上次",
        "2月",
        "1月",
        "几天前",
        "一周前",
        "yesterday",
        "last week",
        "few days ago",
        "the other day",
        "earlier this week",
        "last month",
        "a week ago",
        "days ago",
    ]
    return _contains_any_keyword(user_input, time_keywords)


# [2026-02-26] Build daily narrative block
def build_daily_narrative_block(
    user_input: str,
    session_id: str = "default",
    db_path: str = "data.db",
    limit: int = 2
) -> str:
    """
    检索并构建日叙事上下文块
    
    当用户问到过去的事情时，注入相关的日叙事
    """
    if not _needs_daily_narrative(user_input):
        return ""
    
    try:
        from backend.daily_narrative import get_daily_narrative_generator
        daily_gen = get_daily_narrative_generator(db_path)
        
        # Fetch relevant daily narratives
        narratives = daily_gen.retrieve_relevant_narratives(session_id, user_input, limit=limit)
        
        if not narratives:
            return ""
        
        lines = ["[Past journal entries]"]
        for n in narratives:
            date = n.get("date", "")
            narrative = n.get("narrative", "")
            themes = n.get("themes", [])
            theme_str = ", ".join(themes[:2]) if themes else ""
            
            lines.append(f"• {date}: {narrative}")
            if theme_str:
                lines.append(f"  (Themes: {theme_str})")
        
        lines.append("(Curated journal excerpts you may reference.)")
        
        return "\n".join(lines) + "\n"
        
    except Exception as e:
        logger.debug(f"Failed to build daily narrative block: {e}")
        return ""


# [2026-02-22 P2] Dynamic memory retrieval limits (config-driven)
# Memory/relation/identity intents get higher default caps
MEMORY_INTENT_KEYWORDS = [
    "忘了",
    "还记得",
    "我们",
    "关系",
    "谁",
    "暗号",
    "约定",
    "身份",
    "最早",
    "第一次",
    "之前",
    "上次",
    "刚才",
    "记得",
    "说过",
    "聊过",
    "我们讨论",
    "你提到",
    "过去",
    "历史",
    "曾经",
    "以前",
    "那时候",
    "回忆",
    "remember",
    "forgot",
    "forget",
    "relationship",
    "identity",
    "who said",
    "earlier you",
    "last time",
    "we discussed",
    "you mentioned",
    "recall",
    "remind me",
    "history",
]


def _is_memory_or_relation_query(user_input: str) -> bool:
    """Memory / relationship / identity-ish asks → higher retrieval budget + query expansion."""
    if not user_input:
        return False
    return _contains_any_keyword(user_input, MEMORY_INTENT_KEYWORDS)


def _estimate_memory_need(user_input: str) -> int:
    """
    根据用户输入估算需要的记忆数量。

    策略：
    - 记忆/关系/身份类 → memory_intent_limit（默认 7）
    - 明确引用历史 → 5条
    - 涉及过去的问题 → 4条
    - 技术/任务型问题 → 2条
    - 简单问候/闲聊 → 1条
    - 默认 → default_limit（默认 5）
    """
    if not user_input:
        return int(config.get("parameters.memory.default_limit", 5) or 5)
    default_limit = int(config.get("parameters.memory.default_limit", 5) or 5)
    memory_intent_limit = int(config.get("parameters.memory.memory_intent_limit", 7) or 7)

    if _is_memory_or_relation_query(user_input):
        return memory_intent_limit

    history_keywords = [
        "之前",
        "上次",
        "刚才",
        "记得",
        "说过",
        "聊过",
        "我们讨论",
        "你提到",
        "earlier",
        "last time",
        "you said",
        "we talked",
        "mentioned before",
    ]
    if _contains_any_keyword(user_input, history_keywords):
        return 5

    past_keywords = [
        "过去",
        "历史",
        "曾经",
        "以前",
        "那时候",
        "回忆",
        "in the past",
        "back then",
        "used to",
    ]
    if _contains_any_keyword(user_input, past_keywords):
        return 4

    greeting_keywords = [
        "你好",
        "早上好",
        "晚上好",
        "hi",
        "hello",
        "嗨",
        "在吗",
        "你在",
        "hey",
        "good morning",
    ]
    if _contains_any_keyword(user_input, greeting_keywords) and len(user_input) < 20:
        return 1

    task_keywords = [
        "代码",
        "文件",
        "创建",
        "修改",
        "删除",
        "运行",
        "执行",
        "搜索",
        "查找",
        "code",
        "file",
        "create",
        "delete",
        "run",
        "execute",
        "search",
        "find",
        "implement",
        "refactor",
    ]
    if _contains_any_keyword(user_input, task_keywords):
        return 2

    return default_limit


def build_memory_block(
    self_narrative,
    user_input: str,
    session_id: str = "default",
    *,
    distress_level: Optional[float] = None,
    limit: int = 5,
    return_signal: bool = False,
    auto_adjust_limit: bool = True,  # [P2] auto-tune retrieval count
) -> Union[str, Tuple[str, Dict[str, Any]]]:
    """
    构建叙事记忆提示块。

    新增：
    - distress_level：高破坏性过载时，自动收紧/减少记忆注入，避免 prompt 污染与跑题
    - return_signal：返回 (memory_block, signal_dict)，用于把“记忆命中/模式”写回 needs → z_self
    """
    if not self_narrative:
        return ("", {"strength": 0.0, "mode": "none"}) if return_signal else ""
    try:
        # Prefer unified memory bus when available
        try:
            from backend.unified_memory import UnifiedMemoryBus

            bus = UnifiedMemoryBus(getattr(self_narrative, "db_path", "data.db"))
            unified = bus.retrieve_for_prompt(
                query=user_input,
                session_id=session_id,
                limit=limit,
                distress_level=distress_level,
            )
            unified_block = unified.get("block") or ""
            unified_signal = unified.get("signal") or {}
            if unified_block:
                return (unified_block, unified_signal) if return_signal else unified_block
        except Exception as unified_err:
            logger.debug(f"Unified memory route unavailable, fallback to legacy path: {unified_err}")

        # Default signal; branches overwrite fields
        signal: Dict[str, Any] = {"strength": 0.0, "mode": "none"}

        # High distress: disable weak/extra association blocks
        distress = float(distress_level) if distress_level is not None else None
        restrict = bool(distress is not None and distress >= 0.60)

        # [P2] Dynamic limit from intent heuristics
        try:
            limit_i = int(limit)
        except Exception:
            limit_i = 5
        
        if auto_adjust_limit:
            estimated = _estimate_memory_need(user_input)
            limit_i = max(limit_i, estimated)  # floor for memory-ish queries

        # Clamp limit
        limit_i = max(1, min(12, limit_i))

        # Expand retrieval_query only (display user_input unchanged)
        retrieval_query = user_input
        if _is_memory_or_relation_query(user_input):
            retrieval_query = user_input.rstrip() + " identity relationship agreement user profile"
        memories = self_narrative.retrieve_related_memory(retrieval_query, limit=limit_i)

        profile_block = ""
        try:
            from backend.user_fact_capture import format_user_profile_block_for_prompt

            profile_block = format_user_profile_block_for_prompt(
                getattr(self_narrative, "db_path", "data.db"),
                session_id,
            )
        except Exception:
            profile_block = ""

        if memories:
            # Distress cap on injected rows (config, default 3)
            cap = int(config.get("parameters.memory.distress_memory_cap", 3) or 3)
            use_memories = memories[:cap] if restrict else memories
            memory_text = "\n".join([f"• {m}" for m in use_memories])
            signal = {"strength": 0.85, "mode": "strict", "count": len(use_memories)}
            # Strict mode: at most one extra association block (event > chat)
            extra_block = ""
            # Distress: skip extra association block
            if not restrict:
                try:
                    salient_lines = ""
                    if hasattr(self_narrative, "retrieve_salient_event"):
                        salient = self_narrative.retrieve_salient_event(session_id, user_input, limit=1)
                        if salient:
                            salient_lines = "\n".join([f"• {s}" for s in salient])
                    if salient_lines:
                        # [2026-02-22 P0] compact block
                        extra_block = f"""
[Recent events] {salient_lines}
"""
                        signal = {**signal, "extra": "salient_event"}
                    else:
                        chat_snippet_lines = ""
                        if hasattr(self_narrative, "retrieve_salient_chat_turn"):
                            snippets = self_narrative.retrieve_salient_chat_turn(
                                session_id, user_input, limit=_salient_chat_match_limit()
                            )
                            if snippets:
                                chat_snippet_lines = "\n".join([f"• {s}" for s in snippets])
                        if chat_snippet_lines:
                            extra_block = f"""

[Related chat snippets — use only if truly relevant]
{chat_snippet_lines}
(Reminds you we may have touched similar ground; ignore if off-topic—do not force-fit.)
"""
                            signal = {**signal, "extra": "salient_chat_turn"}
                except Exception:
                    extra_block = ""

            # [2026-02-22 P0] compact memory lines
            # [2026-02-24] nudge explicit recall when useful
            # [2026-04] fallback: profile + note on missing per-line timestamps
            block = f"""{profile_block}[Memory]
{memory_text}{extra_block}
(Use when relevant; ignore otherwise. Fallback narrative retrieval has no per-line timestamps; if this conflicts with [USER profile] or the user's current message, prefer profile / current utterance.)
Tip: for richer history, call recall_memory (dialogue/diary/knowledge/rules); for “what happened recently”, use get_recent_context.
"""
            return (block, signal) if return_signal else block
        # No strict hits → try weak association (max 1, hedged wording)
        if restrict:
            # Distress: skip weak / fallback association
            block = (
                "\n[Memory hint]\n"
                "If the user refers to the past but nothing matched retrieval, say honestly that recall is fuzzy "
                "or that you do not remember—do not invent.\n"
            )
            if profile_block:
                block = profile_block + block
            signal = {"strength": 0.15, "mode": "restricted_none"}
            return (block, signal) if return_signal else block

        assoc = []
        try:
            if hasattr(self_narrative, "retrieve_associative_memory"):
                assoc = self_narrative.retrieve_associative_memory(user_input, limit=1)
        except Exception:
            assoc = []
        if assoc:
            assoc_text = "\n".join([f"• {m}" for m in assoc])
            signal = {"strength": 0.45, "mode": "weak_association", "count": len(assoc)}
            # [2026-02-22 P0] compact weak-assoc block
            block = f"""{profile_block}
[Weak association]
{assoc_text}
(May be related; if unsure, say you only vaguely recall.)
"""
            return (block, signal) if return_signal else block
        # Weak assoc empty → single salient fallback: event > chat
        try:
            salient_only = []
            if hasattr(self_narrative, "retrieve_salient_event"):
                salient_only = self_narrative.retrieve_salient_event(session_id, user_input, limit=1)
            if salient_only:
                salient_text = "\n".join([f"• {s}" for s in salient_only])
                signal = {"strength": 0.35, "mode": "salient_event", "count": len(salient_only)}
                # [2026-02-22 P0] compact block
                block = f"""{profile_block}
[Recent events]
{salient_text}
(Mention lightly only when relevant.)
"""
                return (block, signal) if return_signal else block

            chat_only = []
            if hasattr(self_narrative, "retrieve_salient_chat_turn"):
                chat_only = self_narrative.retrieve_salient_chat_turn(
                    session_id, user_input, limit=_salient_chat_match_limit()
                )
            if chat_only:
                chat_text = "\n".join([f"• {s}" for s in chat_only])
                signal = {"strength": 0.30, "mode": "salient_chat_turn", "count": len(chat_only)}
                # [2026-02-22 P0] compact block
                block = f"""{profile_block}
[Recent conversation]
{chat_text}
(Mention lightly only when relevant.)
"""
                return (block, signal) if return_signal else block
        except Exception:
            pass
        # Nothing retrieved → honest fuzzy-recall hint (profile still first)
        block = (
            "\n[Memory hint]\n"
            "If the user refers to the past but nothing matched retrieval, say honestly recall is fuzzy "
            "or that you do not remember—do not invent.\n"
        )
        if profile_block:
            block = profile_block + block
        signal = {"strength": 0.20, "mode": "none"}
        return (block, signal) if return_signal else block
    except Exception as e:
        logger.debug(f"Failed to retrieve memory: {e}")
        block = (
            "\n[Memory hint]\n"
            "If the user refers to the past but nothing matched retrieval, say honestly recall is fuzzy.\n"
        )
        signal = {"strength": 0.0, "mode": "error"}
        return (block, signal) if return_signal else block


def build_relevant_identity_block(self_model, session_id: str, user_input: str) -> str:
    """检索相关的身份叙事块。身份/关系/回忆类问题时提高 limit 以增强曝光。"""
    if not (self_model and getattr(self_model, "narrative_identity", None)):
        return ""
    try:
        identity_limit = (
            int(config.get("parameters.memory.identity_intent_limit", 6) or 6)
            if _is_memory_or_relation_query(user_input)
            else 3
        )
        relevant_narratives = self_model.narrative_identity.get_relevant_narratives(
            session_id, user_input, limit=identity_limit
        )
        if not relevant_narratives:
            return ""
        narrative_texts = [f"- [{n['type'].upper()}] {n['content']}" for n in relevant_narratives]
        # [2026-02-22 P0] compact identity lines
        return "\n[Identity narratives]\n" + "\n".join(narrative_texts) + "\n"
    except Exception as e:
        logger.debug(f"Failed to retrieve relevant identity narratives: {e}")
        return ""


# [2026-02-07] Knowledge base retrieval block
def build_knowledge_context_block(
    user_input: str,
    session_id: str = "selfing-session",
    top_k: int = 3,
    db_path: str = "data.db"
) -> str:
    """
    根据用户输入检索相关知识，注入到 prompt
    
    [2026-02-07] AGI持续学习能力：让Agent能利用学到的知识
    
    Args:
        user_input: 用户输入
        session_id: 会话ID
        top_k: 返回数量
        db_path: 数据库路径
    
    Returns:
        知识上下文块
    """
    try:
        from backend.knowledge_base import get_knowledge_base
        
        kb = get_knowledge_base(db_path)
        results = kb.search_knowledge(
            query=user_input,
            top_k=top_k,
            session_id=session_id
        )
        
        if not results:
            return ""
        
        # Drop low-similarity rows
        relevant = [r for r in results if r.get("similarity", 0) > 0.3]
        
        if not relevant:
            return ""
        
        lines = ["[Learned knowledge (retrieved)]"]
        for item in relevant:
            confidence_emoji = "🟢" if item["confidence"] > 0.7 else "🟡" if item["confidence"] > 0.4 else "🔴"
            # Truncate long bodies
            content = item["content"]
            if len(content) > 200:
                content = content[:200] + "..."
            lines.append(f"• {confidence_emoji} **{item['title']}**: {content}")
        
        # [2026-02-22 P0] minimal prose around list

        return "\n".join(lines) + "\n"
        
    except ImportError:
        logger.debug("Knowledge base not available")
        return ""
    except Exception as e:
        logger.debug(f"Failed to build knowledge context: {e}")
        return ""

