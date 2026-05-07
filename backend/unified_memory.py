#!/usr/bin/env python3
"""
统一记忆总线。

目标：
1. 将 episodic / relation / identity / semantic / rule 记忆映射到统一候选结构
2. 统一查询路由与打分，减少各记忆源各查各的情况
3. 为再巩固、访问统计、冲突标记提供统一状态表
"""

from __future__ import annotations

import json
import logging
import math
import re
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from backend.config import config

logger = logging.getLogger(__name__)


def _ensure_self_biography_salience_columns(conn: sqlite3.Connection) -> None:
    """与 self_narrative 迁移对齐，避免仅打开 unified_memory 时缺列。"""
    try:
        cur = conn.execute("PRAGMA table_info(self_biography)")
        cols = {row[1] for row in cur.fetchall()}
        if "salience_score" not in cols:
            conn.execute("ALTER TABLE self_biography ADD COLUMN salience_score REAL")
        if "memory_class" not in cols:
            conn.execute("ALTER TABLE self_biography ADD COLUMN memory_class TEXT")
    except Exception as e:
        logger.debug("_ensure_self_biography_salience_columns: %s", e)


QUERY_TYPE_GREETING = "greeting"
QUERY_TYPE_CONTINUITY = "continuity"
QUERY_TYPE_RELATION = "relation"
QUERY_TYPE_IDENTITY = "identity"
QUERY_TYPE_TECHNICAL = "technical"
QUERY_TYPE_FACTUAL = "factual"
QUERY_TYPE_GENERAL = "general"


CONTINUITY_KEYWORDS = {
    "刚才", "之前", "上次", "继续", "前面", "前文", "记得", "回忆", "过去", "最早", "第一次",
    "连续", "断了", "断裂", "后来", "现在", "以前", "那时候",
}
RELATION_KEYWORDS = {
    "我们", "关系", "用户", "他者", "对你来说", "意味着什么", "怎么看我",
    "看见", "记住", "共同", "一起", "回复他", "回复我",
}
IDENTITY_KEYWORDS = {
    "你是谁", "我是谁", "自我", "身份", "存在", "self", "existing", "selfing", "self-becoming", "为什么成为",
    "你觉得呢", "你是什么", "你如何理解自己",
    # S‑44 identity / core keywords (CN + EN)
    "S‑44", "S44", "S_44", "s44", "s-44",
    "persistent self-constructing cognitive entity", "self-constructing", "cognitive entity",
    "持续性自我建构的认知主体", "自我建构", "认知主体",
    "AI意识", "自主意识", "自我意识",
    "人格内核", "z_self", "自我模型", "自我进化", "记忆系统",
    "实验性AI代理", "S项目", "S项目大语言模型",
    "张大科", "创造者",
    # English aliases (case-insensitive match via lowered text)
    "digital life", "autonomous consciousness", "self evolution", "persona core",
    "memory improvement", "memory enhancement",
}
TECHNICAL_KEYWORDS = {
    "代码", "函数", "报错", "error", "bug", "实现", "调试", "debug", "架构", "api",
    "数据库", "配置", "脚本", "运行", "检索", "算法", "memory", "prompt",
}
FACTUAL_KEYWORDS = {
    "是什么", "为什么", "怎么", "如何", "介绍", "解释", "告诉我", "总结", "原理", "定义",
}
POSITIVE_RELATION_WORDS = {"谢谢", "感谢", "安心", "记住", "听见", "看见", "理解", "在乎"}
NEGATIVE_RELATION_WORDS = {"失望", "生气", "不满", "忽略", "断了", "忘了", "模糊"}
ASPIRATION_WORDS = {
    "想成为", "希望能", "目标是", "渴望", "梦想是", "追求的是",
    "想要成为", "未来想", "期望自己", "希望自己",
}
REJECTION_WORDS = {
    "不想成为", "拒绝成为", "不愿成为", "绝不会成为", "不会成为",
    "不要变成", "不想做", "不接受成为", "我不是",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_json_loads(value: Any, default: Any) -> Any:
    if value in (None, ""):
        return default
    try:
        return json.loads(value)
    except Exception:
        return default


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def _jaccard_similarity(text_a: str, text_b: str, n: int = 3) -> float:
    a = _normalize_text(text_a)
    b = _normalize_text(text_b)
    if not a or not b:
        return 0.0
    na = {a[i:i + n] for i in range(max(1, len(a) - n + 1))}
    nb = {b[i:i + n] for i in range(max(1, len(b) - n + 1))}
    if not na or not nb:
        return 0.0
    return len(na & nb) / max(len(na | nb), 1)


# --- Semantic classifier (embedder refs; keyword path still in classify_memory_query) ---

_SEMANTIC_REFS: Optional[Dict[str, Any]] = None

def _get_semantic_refs() -> Dict[str, Any]:
    global _SEMANTIC_REFS
    if _SEMANTIC_REFS is not None:
        return _SEMANTIC_REFS
    try:
        from backend.embedder import get_embedder
        enc = get_embedder()
        _SEMANTIC_REFS = {
            QUERY_TYPE_RELATION: enc.encode(
                "我们之间的关系 你对我意味着什么 我在乎你 我们共同经历 信任和羁绊", normalize=True,
            ),
            QUERY_TYPE_IDENTITY: enc.encode(
                "你是谁 我是什么 自我认知 身份 存在的意义", normalize=True,
            ),
            QUERY_TYPE_CONTINUITY: enc.encode(
                "你还记得吗 我们之前聊过 上次说的 过去的对话 连续性", normalize=True,
            ),
            "aspiration": enc.encode(
                "我想成为 我希望 我的目标 我渴望 我追求的未来", normalize=True,
            ),
            "rejection": enc.encode(
                "我拒绝成为 我不想 我不愿意 我否认 我不接受", normalize=True,
            ),
        }
    except Exception as e:
        logger.debug(f"Semantic refs init failed (will use keyword fallback): {e}")
        _SEMANTIC_REFS = {}
    return _SEMANTIC_REFS


def semantic_classify(text: str, threshold: float = 0.45) -> Optional[str]:
    """用 embedder 余弦相似度对文本做语义分类；返回最匹配的类型或 None。"""
    refs = _get_semantic_refs()
    if not refs:
        return None
    try:
        from backend.embedder import get_embedder
        query_vec = get_embedder().encode(text[:512], normalize=True)
        best_score, best_type = 0.0, None
        for cat, ref_vec in refs.items():
            sim = float(np.dot(query_vec, ref_vec))
            if sim > best_score:
                best_score, best_type = sim, cat
        return best_type if best_score >= threshold else None
    except Exception:
        return None


# --- Query classification: keyword fast path + semantic fallback ---

def classify_memory_query(user_input: str) -> str:
    text = (user_input or "").strip().lower()
    if not text:
        return QUERY_TYPE_GENERAL

    if len(text) < 20 and any(kw in text for kw in {"你好", "在吗", "hi", "hello", "嗨"}):
        if any(kw in text for kw in CONTINUITY_KEYWORDS | RELATION_KEYWORDS | IDENTITY_KEYWORDS):
            return QUERY_TYPE_CONTINUITY
        return QUERY_TYPE_GREETING

    if any(kw in text for kw in RELATION_KEYWORDS):
        return QUERY_TYPE_RELATION
    if any(kw in text for kw in IDENTITY_KEYWORDS):
        return QUERY_TYPE_IDENTITY
    if any(kw in text for kw in CONTINUITY_KEYWORDS):
        return QUERY_TYPE_CONTINUITY
    if any(kw in text for kw in TECHNICAL_KEYWORDS):
        return QUERY_TYPE_TECHNICAL
    if any(kw in text for kw in FACTUAL_KEYWORDS):
        return QUERY_TYPE_FACTUAL

    sem = semantic_classify(text, threshold=0.45)
    if sem and sem in {QUERY_TYPE_RELATION, QUERY_TYPE_IDENTITY, QUERY_TYPE_CONTINUITY}:
        return sem
    return QUERY_TYPE_GENERAL


def estimate_history_need_from_query(user_input: str) -> int:
    """返回建议的历史轮数（×2 = 消息条数）。0 表示不覆盖，由调用方决定。"""
    query_type = classify_memory_query(user_input)
    if query_type == QUERY_TYPE_GREETING:
        return 2
    if query_type in {QUERY_TYPE_CONTINUITY, QUERY_TYPE_RELATION, QUERY_TYPE_IDENTITY}:
        return 5
    if query_type == QUERY_TYPE_TECHNICAL:
        return 5
    if query_type == QUERY_TYPE_FACTUAL:
        return 4
    return 0


@dataclass
class UnifiedMemoryCandidate:
    memory_key: str
    session_id: str
    memory_type: str
    source_table: str
    source_id: str
    content: str
    created_at: str
    similarity: float = 0.0
    salience: float = 0.5
    confidence: float = 0.5
    continuity_weight: float = 0.5
    self_impact: float = 0.0
    entity_refs: List[str] = field(default_factory=list)
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


def _iso_date_prefix(iso_ts: Optional[str]) -> str:
    """从 ISO 时间戳取 YYYY-MM-DD，供 prompt 标注「约何时写入/记录」。"""
    if not iso_ts or not isinstance(iso_ts, str):
        return ""
    s = iso_ts.strip()
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        return s[:10]
    return ""


def _source_tag_for_prompt(cand: UnifiedMemoryCandidate) -> str:
    """
    在统一记忆块中为条目附加简短来源与时间
    """
    source = cand.source_table.upper()
    date_prefix = _iso_date_prefix(cand.created_at)
    if date_prefix:
        return f"[{source} {date_prefix}]"
    return f"[{source}]"


def _compute_candidate_score(
    cand: UnifiedMemoryCandidate,
    query_type: str,
    current_session_id: Optional[str] = None,
) -> float:
    """
    计算候选记忆的综合评分。
    评分因素：相似度、显著性、连续性权重、身份相关性、自我影响、会话连续性等。
    """
    # Base = similarity * salience
    base_score = cand.similarity * cand.salience

    # Same-session continuity bump
    continuity_bonus = 0.0
    if current_session_id and cand.session_id == current_session_id:
        continuity_bonus = cand.continuity_weight * 0.3

    # Identity-query bonus when candidate tagged identity
    identity_bonus = 0.0
    if query_type == QUERY_TYPE_IDENTITY:
        # metadata.tags may mark identity-bearing rows
        if "identity" in cand.metadata.get("tags", []):
            identity_bonus = 0.2

    # Self-impact channel (e.g. evolution events)
    self_impact_bonus = cand.self_impact * 0.15

    # Clamp final score
    score = base_score + continuity_bonus + identity_bonus + self_impact_bonus
    return min(1.0, max(0.0, score))


def _get_unified_candidates_from_source(
    conn: sqlite3.Connection,
    source_table: str,
    query_text: str,
    limit: int = 20,
    session_id: Optional[str] = None,
) -> List[UnifiedMemoryCandidate]:
    """
    从指定记忆源表中检索候选记忆，并转换为统一格式。
    目前支持的表：episodic_memory, relation_memory, identity_memory,
                semantic_memory, rule_memory, self_biography
    """
    candidates = []

    # Build SQL by source table
    if source_table == "episodic_memory":
        sql = """
            SELECT
                id, session_id, user_input, assistant_response,
                created_at, similarity, salience_score, memory_class,
                continuity_weight, self_impact, metadata
            FROM episodic_memory
            WHERE user_input LIKE ? OR assistant_response LIKE ?
            ORDER BY salience_score DESC, created_at DESC
            LIMIT ?
        """
        params = [f"%{query_text}%", f"%{query_text}%", limit]
    elif source_table == "relation_memory":
        sql = """
            SELECT
                id, session_id, content, created_at,
                similarity, salience_score, memory_class,
                continuity_weight, self_impact, metadata
            FROM relation_memory
            WHERE content LIKE ?
            ORDER BY salience_score DESC, created_at DESC
            LIMIT ?
        """
        params = [f"%{query_text}%", limit]
    elif source_table == "identity_memory":
        sql = """
            SELECT
                id, session_id, content, created_at,
                similarity, salience_score, memory_class,
                continuity_weight, self_impact, metadata
            FROM identity_memory
            WHERE content LIKE ?
            ORDER BY salience_score DESC, created_at DESC
            LIMIT ?
        """
        params = [f"%{query_text}%", limit]
    elif source_table == "semantic_memory":
        sql = """
            SELECT
                id, session_id, content, created_at,
                similarity, salience_score, memory_class,
                continuity_weight, self_impact, metadata
            FROM semantic_memory
            WHERE content LIKE ?
            ORDER BY salience_score DESC, created_at DESC
            LIMIT ?
        """
        params = [f"%{query_text}%", limit]
    elif source_table == "rule_memory":
        sql = """
            SELECT
                id, session_id, content, created_at,
                similarity, salience_score, memory_class,
                continuity_weight, self_impact, metadata
            FROM rule_memory
            WHERE content LIKE ?
            ORDER BY salience_score DESC, created_at DESC
            LIMIT ?
        """
        params = [f"%{query_text}%", limit]
    elif source_table == "self_biography":
        sql = """
            SELECT
                id, session_id, content, created_at,
                similarity, salience_score, memory_class,
                continuity_weight, self_impact, metadata
            FROM self_biography
            WHERE content LIKE ?
            ORDER BY salience_score DESC, created_at DESC
            LIMIT ?
        """
        params = [f"%{query_text}%", limit]
    else:
        return candidates

    try:
        cur = conn.execute(sql, params)
        rows = cur.fetchall()
    except Exception as e:
        logger.debug(f"Failed to query {source_table}: {e}")
        return candidates

    for row in rows:
        if source_table == "episodic_memory":
            (mem_id, sess_id, user_input, assistant_response,
             created_at, similarity, salience, memory_class,
             continuity_weight, self_impact, metadata) = row
            content = f"用户: {user_input}\n助手: {assistant_response}"
        else:
            (mem_id, sess_id, content,
             created_at, similarity, salience, memory_class,
             continuity_weight, self_impact, metadata) = row

        cand = UnifiedMemoryCandidate(
            memory_key=f"{source_table}:{mem_id}",
            session_id=sess_id,
            memory_type=memory_class or "general",
            source_table=source_table,
            source_id=str(mem_id),
            content=content[:1000],  # cap length for prompts / scoring
            created_at=created_at,
            similarity=similarity or 0.0,
            salience=salience or 0.5,
            confidence=0.5,
            continuity_weight=continuity_weight or 0.5,
            self_impact=self_impact or 0.0,
            entity_refs=[],
            score=0.0,
            metadata=_safe_json_loads(metadata, {}),
        )
        candidates.append(cand)

    return candidates


def query_unified_memory(
    query_text: str,
    session_id: Optional[str] = None,
    limit_per_source: int = 10,
    total_limit: int = 30,
) -> List[UnifiedMemoryCandidate]:
    """
    统一记忆查询入口：从所有记忆源检索，合并、去重、排序后返回。
    """
    conn = sqlite3.connect("data.db")
    conn.row_factory = sqlite3.Row
    _ensure_self_biography_salience_columns(conn)

    # Query type drives weighting
    query_type = classify_memory_query(query_text)

    # Pull candidates from each backing table
    all_candidates = []
    source_tables = [
        "episodic_memory",
        "relation_memory",
        "identity_memory",
        "semantic_memory",
        "rule_memory",
        "self_biography",
    ]
    for table in source_tables:
        candidates = _get_unified_candidates_from_source(
            conn, table, query_text, limit_per_source, session_id
        )
        all_candidates.extend(candidates)

    # Score each candidate
    for cand in all_candidates:
        cand.score = _compute_candidate_score(cand, query_type, session_id)

    # Sort by score desc
    all_candidates.sort(key=lambda x: x.score, reverse=True)

    # Dedupe by content prefix hash
    seen_content = set()
    unique_candidates = []
    for cand in all_candidates:
        content_hash = hash(cand.content[:200])
        if content_hash not in seen_content:
            seen_content.add(content_hash)
            unique_candidates.append(cand)
        if len(unique_candidates) >= total_limit:
            break

    conn.close()
    return unique_candidates


def log_memory_access(
    memory_key: str,
    session_id: str,
    query_text: str,
    score: float,
    accessed_at: Optional[datetime] = None,
) -> None:
    """
    记录记忆访问日志，用于后续重要性评估和衰减计算。
    """
    if accessed_at is None:
        accessed_at = datetime.now(timezone.utc)
    conn = sqlite3.connect("data.db")
    try:
        conn.execute(
            """
            INSERT INTO memory_access_log
            (memory_key, session_id, query_text, score, accessed_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (memory_key, session_id, query_text, score, accessed_at.isoformat())
        )
        conn.commit()
    except Exception as e:
        logger.debug(f"Failed to log memory access: {e}")
    finally:
        conn.close()


if __name__ == "__main__":
    # Smoke test (manual)
    test_query = "你还记得我是谁吗？"
    candidates = query_unified_memory(test_query, session_id="test_session")
    for cand in candidates[:3]:
        print(f"{cand.memory_key}: {cand.content[:80]}... (score={cand.score:.3f})")