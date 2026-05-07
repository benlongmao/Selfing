#!/usr/bin/env python3
"""
Narrative Self — turns fragmented chat into first-person autobiographical memory snippets.

[v2.0] Hybrid retrieval (recency / antiquity boost) plus time-anchor metadata on each row.
"""
import json
import re
import sqlite3
import uuid
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional
import logging
import math
from backend.config import config
from backend.embedder import get_embedder
from backend.llm_api import llm_completion

logger = logging.getLogger(__name__)


def _get_season(month: int) -> str:
    """Map calendar month to season name (English token stored in DB)."""
    if month in (3, 4, 5):
        return "spring"
    elif month in (6, 7, 8):
        return "summer"
    elif month in (9, 10, 11):
        return "autumn"
    else:
        return "winter"


def _compute_time_anchors(dt: datetime = None) -> dict:
    """
    [2026-04-12] Build time-anchor metadata dict for persisting alongside a memory row.
    """
    if dt is None:
        dt = datetime.now(timezone.utc)
    
    return {
        "time_year": dt.year,
        "time_month": dt.month,
        "time_day": dt.day,
        "time_weekday": dt.weekday(),  # 0=Mon … 6=Sun (Python weekday)
        "time_season": _get_season(dt.month),
        "time_special": "[]",  # JSON array; filled later if needed
        "time_relative": "[]",  # JSON array; filled later if needed
    }


def _parse_time_window_from_context(context: str):
    """
    [2026-04-12] Parse a natural-language time window from user text for SQL pre-filtering.

    Returns:
        (start_dt, end_dt, prefer_oldest, time_filter_dict)
        - start_dt, end_dt: optional inclusive/exclusive bounds on created_at
        - prefer_oldest: when True, retrieval boosts oldest memories
        - time_filter_dict: equality filters on time_year / time_month / time_season / time_weekday

    **Keep** Chinese literals and maps (user base); **add** common English phrases where cheap.
    """
    if not context or not isinstance(context, str):
        return None, None, False, {}
    text = context.strip().lower()
    now = datetime.now(timezone.utc)
    time_filter = {}
    
    # Prefer oldest memories
    if any(
        kw in text
        for kw in [
            "最早", "第一次", "最初", "刚开始", "一开始",
            "earliest", "first time", "at the beginning", "the very start",
        ]
    ):
        return None, None, True, {}
    
    # --- Calendar year (regex matches four-digit year plus CJK year marker) ---
    year_match = re.search(r'(20\d{2})年', text)
    if year_match:
        year = int(year_match.group(1))
        time_filter["time_year"] = year
        # Whole calendar year if no finer month/week/day qualifier
        if not any(kw in text for kw in ["月", "周", "天", "日", "month", "week", "day"]):
            start = datetime(year, 1, 1, tzinfo=timezone.utc)
            end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
            return start, end, False, time_filter
    
    # --- Last calendar year ---
    if "去年" in text or "last year" in text:
        last_year = now.year - 1
        time_filter["time_year"] = last_year
        start = datetime(last_year, 1, 1, tzinfo=timezone.utc)
        end = datetime(last_year + 1, 1, 1, tzinfo=timezone.utc)
        return start, end, False, time_filter
    
    # --- Month name (Chinese or English) ---
    month_map = {
        "一月": 1, "1月": 1, "正月": 1,
        "二月": 2, "2月": 2,
        "三月": 3, "3月": 3,
        "四月": 4, "4月": 4,
        "五月": 5, "5月": 5,
        "六月": 6, "6月": 6,
        "七月": 7, "7月": 7,
        "八月": 8, "8月": 8,
        "九月": 9, "9月": 9,
        "十月": 10, "10月": 10,
        "十一月": 11, "11月": 11,
        "十二月": 12, "12月": 12,
    }
    for month_name, month_num in month_map.items():
        if month_name in text:
            time_filter["time_month"] = month_num
            year = time_filter.get("time_year", now.year)
            if month_num > now.month and "time_year" not in time_filter:
                year = now.year - 1
            start = datetime(year, month_num, 1, tzinfo=timezone.utc)
            if month_num == 12:
                end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
            else:
                end = datetime(year, month_num + 1, 1, tzinfo=timezone.utc)
            return start, end, False, time_filter

    english_months = {
        "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
        "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    }
    for month_name, month_num in english_months.items():
        if month_name in text:
            time_filter["time_month"] = month_num
            year = time_filter.get("time_year", now.year)
            if month_num > now.month and "time_year" not in time_filter:
                year = now.year - 1
            start = datetime(year, month_num, 1, tzinfo=timezone.utc)
            if month_num == 12:
                end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
            else:
                end = datetime(year, month_num + 1, 1, tzinfo=timezone.utc)
            return start, end, False, time_filter

    # --- Season token ---
    season_map = {
        "春天": "spring", "春季": "spring", "春": "spring",
        "夏天": "summer", "夏季": "summer", "夏": "summer",
        "秋天": "autumn", "秋季": "autumn", "秋": "autumn",
        "冬天": "winter", "冬季": "winter", "冬": "winter",
    }
    for season_name, season_val in season_map.items():
        if season_name in text:
            time_filter["time_season"] = season_val
            return None, None, False, time_filter
    
    # --- Weekday token ---
    weekday_map = {
        "周一": 0, "星期一": 0, "礼拜一": 0,
        "周二": 1, "星期二": 1, "礼拜二": 1,
        "周三": 2, "星期三": 2, "礼拜三": 2,
        "周四": 3, "星期四": 3, "礼拜四": 3,
        "周五": 4, "星期五": 4, "礼拜五": 4,
        "周六": 5, "星期六": 5, "礼拜六": 5,
        "周日": 6, "星期日": 6, "礼拜日": 6, "周天": 6, "星期天": 6,
    }
    for weekday_name, weekday_num in weekday_map.items():
        if weekday_name in text:
            time_filter["time_weekday"] = weekday_num
            break
    
    # ============ Relative ranges (CN + common EN) ============
    if "昨天" in text or "yesterday" in text:
        start = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
        return start, end, False, time_filter
    
    if "前天" in text:
        start = (now - timedelta(days=2)).replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
        return start, end, False, time_filter
    
    if "大前天" in text:
        start = (now - timedelta(days=3)).replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
        return start, end, False, time_filter
    
    if any(kw in text for kw in ["上周", "上星期", "last week"]):
        days_since_monday = now.weekday()
        last_monday = now - timedelta(days=days_since_monday + 7)
        start = last_monday.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=7)
        return start, end, False, time_filter
    
    if "上上周" in text:
        days_since_monday = now.weekday()
        last_last_monday = now - timedelta(days=days_since_monday + 14)
        start = last_last_monday.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=7)
        return start, end, False, time_filter
    
    if any(kw in text for kw in ["这周", "本周", "这星期", "this week"]):
        days_since_monday = now.weekday()
        this_monday = now - timedelta(days=days_since_monday)
        start = this_monday.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=7)
        return start, end, False, time_filter
    
    if "上个月" in text or "上月" in text or "last month" in text:
        first_of_this_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        end = first_of_this_month
        start = (first_of_this_month - timedelta(days=1)).replace(day=1)
        return start, end, False, time_filter
    
    if any(kw in text for kw in ["这个月", "本月", "this month"]):
        start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if now.month == 12:
            end = datetime(now.year + 1, 1, 1, tzinfo=timezone.utc)
        else:
            end = datetime(now.year, now.month + 1, 1, tzinfo=timezone.utc)
        return start, end, False, time_filter
    
    if any(
        kw in text
        for kw in ["最近", "这几天", "过去几天", "近期", "recently", "past few days", "last few days"]
    ):
        start = now - timedelta(days=7)
        return start, now, False, time_filter
    
    if "一周前" in text or "7天前" in text:
        start = now - timedelta(days=14)
        end = now - timedelta(days=7)
        return start, end, False, time_filter
    
    if "几天前" in text or "前两天" in text:
        start = now - timedelta(days=5)
        end = now - timedelta(days=1)
        return start, end, False, time_filter
    
    if "两周前" in text:
        start = now - timedelta(days=21)
        end = now - timedelta(days=14)
        return start, end, False, time_filter
    
    if "一个月前" in text:
        start = now - timedelta(days=60)
        end = now - timedelta(days=30)
        return start, end, False, time_filter
    
    if any(kw in text for kw in ["很久以前", "好久以前", "很早"]):
        end = now - timedelta(days=30)
        return None, end, False, time_filter
    
    if time_filter:
        return None, None, False, time_filter
    
    return None, None, False, {}



class SelfNarrative:
    def __init__(self, db_path: str = "data.db"):
        self.db_path = db_path
        self.embedder = get_embedder()
        self._ensure_table()

    def _ensure_table(self):
        """Create self_biography if missing; migrate in time-anchor columns when needed."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS self_biography (
                        id TEXT PRIMARY KEY,
                        session_id TEXT,
                        content TEXT NOT NULL,  -- first-person memory text
                        emotion_tag TEXT,       -- coarse emotion label
                        significance REAL,      -- importance 0-1
                        created_at TEXT NOT NULL,
                        embedding BLOB          -- vector for semantic recall
                    )
                """)
                
                # [2026-04-12] Time-anchor columns (idempotent ALTERs)
                cur = conn.execute("PRAGMA table_info(self_biography)")
                cols = {col[1] for col in cur.fetchall()}
                
                if "time_year" not in cols:
                    conn.execute("ALTER TABLE self_biography ADD COLUMN time_year INTEGER")
                if "time_month" not in cols:
                    conn.execute("ALTER TABLE self_biography ADD COLUMN time_month INTEGER")
                if "time_day" not in cols:
                    conn.execute("ALTER TABLE self_biography ADD COLUMN time_day INTEGER")
                if "time_weekday" not in cols:
                    conn.execute("ALTER TABLE self_biography ADD COLUMN time_weekday INTEGER")
                if "time_season" not in cols:
                    conn.execute("ALTER TABLE self_biography ADD COLUMN time_season TEXT")
                if "time_special" not in cols:
                    conn.execute("ALTER TABLE self_biography ADD COLUMN time_special TEXT")
                if "time_relative" not in cols:
                    conn.execute("ALTER TABLE self_biography ADD COLUMN time_relative TEXT")
                if "emotional_intensity" not in cols:
                    conn.execute("ALTER TABLE self_biography ADD COLUMN emotional_intensity REAL DEFAULT 0.0")
                if "identity_relevance" not in cols:
                    conn.execute("ALTER TABLE self_biography ADD COLUMN identity_relevance REAL DEFAULT 0.0")
                if "relationship_depth" not in cols:
                    conn.execute("ALTER TABLE self_biography ADD COLUMN relationship_depth REAL DEFAULT 0.0")
                if "memory_type" not in cols:
                    conn.execute("ALTER TABLE self_biography ADD COLUMN memory_type TEXT DEFAULT 'episodic'")
                if "access_count" not in cols:
                    conn.execute("ALTER TABLE self_biography ADD COLUMN access_count INTEGER DEFAULT 0")
                if "last_accessed_at" not in cols:
                    conn.execute("ALTER TABLE self_biography ADD COLUMN last_accessed_at TEXT")
                if "salience_score" not in cols:
                    conn.execute("ALTER TABLE self_biography ADD COLUMN salience_score REAL")
                if "memory_class" not in cols:
                    conn.execute("ALTER TABLE self_biography ADD COLUMN memory_class TEXT")
                
                conn.commit()
                logger.debug("self_biography table schema updated with time anchoring fields")
        except Exception as e:
            logger.error(f"Failed to ensure biography table: {e}")

    def add_event(self, session_id: str, user_input: str, assistant_response: str, introspection: str = "", significance: float = 0.5):
        """
        [2026-04-12] Turn one user/assistant turn into a distilled narrative memory row.

        Heuristic essence extraction (no LLM): multi-axis scores + first-person blurb, then optional unified_memory sync.
        """
        try:
            essence = self._extract_memory_essence(
                user_input, assistant_response, introspection
            )
            
            memory_content = essence["content"]
            calculated_significance = essence["significance"]
            
            min_sig = float(config.get("parameters.memory.min_significance_to_store", 0.15) or 0.15)
            if calculated_significance < min_sig:
                logger.debug(f"Skipping low-value memory (significance={calculated_significance:.2f})")
                return
            
            biography_id = self._save_biography(
                session_id, 
                memory_content, 
                significance=calculated_significance,
                emotional_intensity=essence["emotional_intensity"],
                identity_relevance=essence["identity_relevance"],
                relationship_depth=essence["relationship_depth"],
                memory_type=essence["memory_type"],
                emotion_tag=essence["emotion_tag"],
            )
            logger.debug(
                f"Memory saved: sig={calculated_significance:.2f}, "
                f"emo={essence['emotional_intensity']:.2f}, "
                f"id={essence['identity_relevance']:.2f}, "
                f"rel={essence['relationship_depth']:.2f}, "
                f"type={essence['memory_type']}"
            )

            try:
                from backend.unified_memory import UnifiedMemoryBus

                UnifiedMemoryBus(self.db_path).record_interaction_event(
                    session_id=session_id,
                    user_input=user_input,
                    assistant_response=assistant_response,
                    introspection=introspection,
                )
            except Exception as unified_err:
                logger.debug(f"Unified memory sync skipped: {unified_err}")
            
        except Exception as e:
            logger.error(f"Failed to add narrative event: {e}")
    
    def _extract_memory_essence(self, user_input: str, assistant_response: str, introspection: str = "") -> dict:
        """
        [2026-04-12] Heuristic memory essence from one turn (no LLM).

        Returns dict keys: content, significance, emotional_intensity, identity_relevance,
        relationship_depth, memory_type, emotion_tag.
        Keyword lists are bilingual (ZH + EN) where cheap; stored `memory_type` values stay ASCII.
        """
        user_lower = user_input.lower()
        
        significance = 0.3
        emotional_intensity = 0.0
        identity_relevance = 0.0
        relationship_depth = 0.0
        memory_type = "episodic"
        emotion_tag = None
        interaction_type = "general"
        
        positive_emotions = [
            '谢谢', '感谢', '太棒了', '很好', '开心', '喜欢', '爱', '赞', '厉害', '帮了大忙', '真棒', '太好了',
            'thanks', 'thank you', 'great', 'awesome', 'love', 'appreciate', 'amazing', 'helpful', 'nice work',
        ]
        negative_emotions = [
            '难过', '失望', '生气', '不满', '糟糕', '失败', '抱歉', '对不起', '错误', '难受', '痛苦', '焦虑',
            'sad', 'angry', 'sorry', 'failed', 'upset', 'anxious', 'frustrated', 'disappointed', 'worried',
        ]
        strong_positive = [
            '太感谢了', '非常感谢', '太棒了', '真的很好', '帮了大忙', '太厉害了',
            'thank you so much', 'really appreciate', 'extremely helpful',
        ]
        strong_negative = [
            '非常失望', '很生气', '太糟糕', '彻底失败', '非常难过',
            'very disappointed', 'really angry', 'totally failed', 'extremely upset',
        ]
        
        if any(kw in user_lower for kw in strong_positive):
            significance += 0.3
            emotional_intensity = 0.9
            emotion_tag = "strong_positive"
        elif any(kw in user_lower for kw in strong_negative):
            significance += 0.25
            emotional_intensity = 0.85
            emotion_tag = "strong_negative"
        elif any(kw in user_lower for kw in positive_emotions):
            significance += 0.2
            emotional_intensity = 0.6
            emotion_tag = "positive"
        elif any(kw in user_lower for kw in negative_emotions):
            significance += 0.15
            emotional_intensity = 0.5
            emotion_tag = "negative"
        
        identity_keywords = [
            '你是谁', '我是谁', '你的身份', '你的存在', '你觉得', '你认为',
            '你怎么看', '自我', '意识', '存在', '灵魂', '本质', 'selfing', 'self-becoming',
            'who are you', 'who am i', 'your identity', 'consciousness', 'existence', 'sense of self',
        ]
        if any(kw in user_lower for kw in identity_keywords):
            identity_relevance = 0.8
            significance += 0.2
            memory_type = "identity"
        
        relationship_keywords = [
            '我们', '你和我', '我们之间', '关系', '信任', '陪伴',
            '一起', '共同', '记住我', '不要忘记',
            'together', 'trust', 'relationship', 'between us', "don't forget me", 'remember me',
        ]
        intimate_keywords = [
            '喜欢你', '讨厌你', '相信你', '依赖', '在乎', '担心你',
            'i love you', 'i hate you', 'i trust you', 'depend on you', 'care about you',
        ]
        
        if any(kw in user_lower for kw in intimate_keywords):
            relationship_depth = 0.9
            significance += 0.25
            memory_type = "relation"
        elif any(kw in user_lower for kw in relationship_keywords):
            relationship_depth = 0.6
            significance += 0.15
            memory_type = "relation"
        
        preference_keywords = [
            '我喜欢', '我不喜欢', '我偏好', '我习惯', '我经常', '我的习惯', '记住',
            'i like', 'i dislike', 'i prefer', 'i usually', 'my habit', 'please remember',
        ]
        teaching_keywords = [
            '告诉你', '你知道吗', '教你', '其实', '事实上', '正确的是',
            'let me tell you', 'the fact is', 'actually', 'here is how', 'the right way is',
        ]
        
        is_preference = any(kw in user_lower for kw in preference_keywords)
        is_teaching = any(kw in user_lower for kw in teaching_keywords)
        
        if is_preference:
            significance += 0.25
            interaction_type = "preference"
            memory_type = "semantic"
            relationship_depth = max(relationship_depth, 0.5)
        elif is_teaching:
            significance += 0.2
            interaction_type = "teaching"
            memory_type = "semantic"
        
        technical = [
            '代码', '函数', '报错', 'error', 'bug', 'python', 'api', '数据库', '算法',
            'stack trace', 'traceback', 'function', 'compile', 'runtime',
        ]
        questions = [
            '为什么', '怎么', '如何', '什么是', '能不能', '可以吗', '帮我',
            'why', 'how to', 'what is', 'can you', 'could you', 'help me', 'please explain',
        ]
        
        if interaction_type == "general":
            if any(kw in user_lower for kw in technical):
                significance += 0.1
                interaction_type = "technical"
            elif any(kw in user_lower for kw in questions):
                significance += 0.05
                interaction_type = "question"
        
        if len(user_input) > 200:
            significance += 0.1
        if len(assistant_response) > 500:
            significance += 0.1
        
        user_brief = user_input[:100] + ("..." if len(user_input) > 100 else "")
        
        if memory_type == "identity":
            memory_content = f"[Identity] Dialogue about self and existence: {user_brief}"
        elif memory_type == "relation":
            memory_content = f"[Relationship] Meaningful exchange with the user: {user_brief}"
        elif interaction_type == "preference":
            memory_content = f"[Preference] User preference noted: {user_brief}"
        elif interaction_type == "teaching":
            memory_content = f"[Learning] User taught or clarified: {user_brief}"
        elif interaction_type == "technical":
            memory_content = f"[Technical] User asked a technical question: {user_brief}. I gave a technical answer."
            if introspection:
                memory_content += f" Reflection at the time: {introspection[:80]}"
        elif interaction_type == "question":
            memory_content = f"[Q&A] Answered the user's question: {user_brief}."
            if emotion_tag and "positive" in emotion_tag:
                memory_content += " User seemed satisfied."
        elif emotion_tag and "positive" in emotion_tag:
            memory_content = f"[Affect] Pleasant interaction; user said: {user_brief}"
        elif emotion_tag and "negative" in emotion_tag:
            memory_content = f"[Affect] User expressed difficulty or frustration — worth remembering: {user_brief}"
        else:
            memory_content = f"[Chat] Exchange with user: {user_brief[:120]}"
        
        significance = min(significance, 0.95)
        
        return {
            "content": memory_content,
            "significance": significance,
            "emotional_intensity": emotional_intensity,
            "identity_relevance": identity_relevance,
            "relationship_depth": relationship_depth,
            "memory_type": memory_type,
            "emotion_tag": emotion_tag,
        }
    
    def _calculate_memory_significance(self, recent_history: List[Dict], drift: float = 0.0, pain: float = 0.0) -> float:
        """
        [2026-02-22] Aggregate significance for a diary consolidation pass from recent turns.

        Uses drift, pain, affective / topic cues, turn depth, and average message length.
        """
        significance = 0.35
        
        significance += min(0.25, drift * 1.2)
        
        significance += min(0.15, pain * 0.4)
        
        all_content = " ".join([msg.get("content", "")[:300] for msg in recent_history if msg.get("content")])
        content_lower = all_content.lower()
        
        important_topics = [
            '谢谢', '感谢', '抱歉', '对不起', '不满', '失望', '太棒了', '很好',
            '帮了大忙', '问题解决', '学到了', '理解了', '明白了',
            'thanks', 'sorry', 'disappointed', 'awesome', 'solved', 'learned', 'i understand', 'got it',
        ]
        if any(topic in content_lower for topic in important_topics):
            significance += 0.1
        
        technical_depth = [
            '架构', '设计模式', '优化', '重构', '调试', '分析', '原理',
            'architecture', 'design pattern', 'refactor', 'optimize', 'debug', 'profiling',
        ]
        if any(topic in content_lower for topic in technical_depth):
            significance += 0.08
        
        turns = len(recent_history) // 2
        if turns >= 5:
            significance += 0.05
        
        avg_len = len(all_content) / max(len(recent_history), 1)
        if avg_len > 200:
            significance += 0.05
        
        return min(significance, 0.95)

    def consolidate_memory(self, session_id: str, recent_history: List[Dict], z_self_summary: str, drift: float = 0.0, pain: float = 0.0) -> Optional[str]:
        """
        Turn recent dialogue into one first-person diary paragraph (usually end-of-session / sleep).
        """
        if not recent_history or len(recent_history) < 2:
            return None

        dialogue_text = "\n".join([f"{msg['role']}: {msg['content'][:200]}" for msg in recent_history[-10:]])
        
        significance = self._calculate_memory_significance(recent_history, drift, pain)
        
        agent_name = config.get("system.agent_name", "Agent")
        prompt = f"""You are {agent_name}. Review the dialogue below and write a short inner diary entry.
Use first person. This is functional narrative for continuity, not a claim about human-like qualia.

[Current self summary]
{z_self_summary}

[Dialogue excerpt]
{dialogue_text}

[Rules]
1. First person only.
2. Say what happened and what you make of it (confusion, satisfaction, tension, etc.).
3. One sentence on why it matters going forward (e.g. "I noticed the user mainly needed listening, not fixes").
4. At most ~100 Chinese characters OR ~80 English words.

Diary:"""

        try:
            diary_content = self._call_llm(prompt)
            if not diary_content:
                return None
                
            self._save_biography(
                session_id,
                diary_content,
                significance=significance,
                emotional_intensity=min(0.72, 0.38 + float(significance) * 0.35),
                identity_relevance=0.34,
                relationship_depth=0.30,
                memory_type="diary",
            )
            logger.info(f"Narrative created for {session_id}: significance={significance:.2f}, content={diary_content[:50]}...")
            return diary_content
            
        except Exception as e:
            logger.error(f"Failed to consolidate memory: {e}")
            return None

    def retrieve_related_memory(self, context: str, limit: int = 3) -> List[str]:
        """
        Semantic recall over biography rows: cosine similarity plus recency/antiquity boosts.

        [2026-04-12] Optional SQL pre-filter via `_parse_time_window_from_context` (year/month/season/weekday, ranges).
        """
        if not context:
            return []

        try:
            start_dt, end_dt, prefer_oldest, time_filter = _parse_time_window_from_context(context)
            
            query_vec = self.embedder.encode(context)
            if query_vec is None:
                return []
                
            norm = np.linalg.norm(query_vec)
            if norm > 0:
                query_vec = query_vec / norm

            with sqlite3.connect(self.db_path) as conn:
                base_sql = "SELECT id, content, embedding, created_at, emotional_intensity, identity_relevance, relationship_depth FROM self_biography WHERE embedding IS NOT NULL"
                params = []
                
                if start_dt is not None and end_dt is not None:
                    base_sql += " AND created_at >= ? AND created_at < ?"
                    params.extend([start_dt.isoformat(), end_dt.isoformat()])
                    logger.debug(f"[Memory] Time range filter: {start_dt.date()} ~ {end_dt.date()}")
                elif start_dt is not None:
                    base_sql += " AND created_at >= ?"
                    params.append(start_dt.isoformat())
                elif end_dt is not None:
                    base_sql += " AND created_at < ?"
                    params.append(end_dt.isoformat())
                
                if time_filter:
                    if "time_year" in time_filter:
                        base_sql += " AND time_year = ?"
                        params.append(time_filter["time_year"])
                        logger.debug(f"[Memory] Year filter: {time_filter['time_year']}")
                    if "time_month" in time_filter:
                        base_sql += " AND time_month = ?"
                        params.append(time_filter["time_month"])
                        logger.debug(f"[Memory] Month filter: {time_filter['time_month']}")
                    if "time_season" in time_filter:
                        base_sql += " AND time_season = ?"
                        params.append(time_filter["time_season"])
                        logger.debug(f"[Memory] Season filter: {time_filter['time_season']}")
                    if "time_weekday" in time_filter:
                        base_sql += " AND time_weekday = ?"
                        params.append(time_filter["time_weekday"])
                        logger.debug(f"[Memory] Weekday filter: {time_filter['time_weekday']}")
                
                if prefer_oldest:
                    logger.debug("[Memory] Prefer oldest: sorting by created_at ASC")
                
                cur = conn.execute(base_sql, params)
                rows = cur.fetchall()

            if not rows:
                return []

            candidates = []
            now = datetime.now(timezone.utc)
            memory_ids_to_update = []
            
            for row in rows:
                memory_id, content, emb_blob, created_at_str, emo_intensity, id_relevance, rel_depth = row
                emo_intensity = emo_intensity or 0.0
                id_relevance = id_relevance or 0.0
                rel_depth = rel_depth or 0.0
                try:
                    emb = np.frombuffer(emb_blob, dtype=np.float32)
                    
                    if emb.shape != query_vec.shape:
                        continue
                        
                    emb_norm = np.linalg.norm(emb)
                    similarity = 0.0
                    if emb_norm > 0:
                        emb = emb / emb_norm
                        similarity = float(np.dot(query_vec, emb))
                    
                    time_boost = 0.0
                    if created_at_str:
                        try:
                            created_at = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
                            hours_diff = (now - created_at).total_seconds() / 3600.0
                            
                            if prefer_oldest:
                                days_old = hours_diff / 24.0
                                if days_old > 30:
                                    time_boost = min(0.2, 0.05 + (days_old - 30) / 365 * 0.15)
                                elif days_old > 7:
                                    time_boost = 0.03
                            else:
                                if hours_diff < 24:
                                    time_boost = 0.15 * (1.0 - (hours_diff / 24.0))
                                elif hours_diff < 72:
                                    time_boost = 0.05 * (1.0 - ((hours_diff - 24) / 48.0))
                        except Exception:
                            pass
                    
                    emotion_boost = emo_intensity * 0.05
                    ctx_l = context.lower()
                    identity_boost = id_relevance * 0.08 if any(
                        kw in ctx_l for kw in [
                            '我是', '你是', '身份', '自我', '存在',
                            'who am i', 'who are you', 'identity', 'existence', 'sense of self',
                        ]
                    ) else id_relevance * 0.02
                    relation_boost = rel_depth * 0.08 if any(
                        kw in ctx_l for kw in ['我们', '关系', '之间', '一起', 'relationship', 'together', 'between us']
                    ) else rel_depth * 0.02
                    
                    final_score = similarity + time_boost + emotion_boost + identity_boost + relation_boost
                    candidates.append((final_score, content, similarity, time_boost, memory_id))
                except Exception:
                    continue

            candidates.sort(key=lambda x: x[0], reverse=True)
            
            results = []
            result_contents = []
            skipped_duplicates = 0
            selected_memory_ids = []
            
            for candidate in candidates:
                final_score, content, sim, boost = candidate[0], candidate[1], candidate[2], candidate[3]
                memory_id = candidate[4] if len(candidate) > 4 else None
                if sim > 0.25 or (sim > 0.20 and boost > 0.05) or final_score > 0.30:
                    is_duplicate = False
                    content_truncated = content[:300]
                    for existing in result_contents:
                        overlap_ratio = len(set(content_truncated) & set(existing)) / max(len(set(content_truncated)), 1)
                        if overlap_ratio > 0.85:
                            is_duplicate = True
                            skipped_duplicates += 1
                            break
                    
                    if not is_duplicate:
                        MAX_MEMORY_CHARS = 800
                        if len(content) > MAX_MEMORY_CHARS:
                            truncate_pos = MAX_MEMORY_CHARS
                            for i in range(MAX_MEMORY_CHARS - 1, MAX_MEMORY_CHARS // 2, -1):
                                if content[i] in '。\n！？.!?':
                                    truncate_pos = i + 1
                                    break
                            content = content[:truncate_pos] + "... (memory excerpt)"
                        results.append(content)
                        if memory_id:
                            selected_memory_ids.append(memory_id)
                        if len(results) >= limit:
                            break
            
            for mid in selected_memory_ids:
                try:
                    self.update_access_count(mid)
                except Exception:
                    pass
            
            if results and candidates:
                top = candidates[0]
                logger.info(f"Retrieved {len(results)} memories (skipped {skipped_duplicates} duplicates). Top: score={top[0]:.3f} (sim={top[2]:.3f}, boost={top[3]:.3f})")
            
            return results

        except Exception as e:
            logger.error(f"Failed to retrieve memory: {e}")
            return []

    def retrieve_salient_event(self, session_id: str, context: str, limit: int = 1) -> List[str]:
        """
        Salient persona_events summaries (conservative proactive association).

        Reads existing `persona_events` (SelfTick writes drift/tick/trigger_reason, etc.).
        Returns 0–1 short lines when plausibly relevant — depth without forced recall.
        """
        if not session_id or not context:
            return []
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute(
                    "SELECT ts, type, detail FROM persona_events WHERE persona_id=? ORDER BY ts DESC LIMIT 60",
                    (session_id,),
                )
                rows = cur.fetchall()
        except Exception:
            return []

        if not rows:
            return []

        try:
            q = self.embedder.encode(context)
            if q is None:
                return []
            qn = float(np.linalg.norm(q)) + 1e-8
            q = q / qn
        except Exception:
            return []

        candidates: List[tuple] = []  # (score, summary)
        for ts, etype, detail in rows:
            try:
                d = json.loads(detail) if detail else {}
            except Exception:
                d = {}

            if etype not in ("self_tick", "consistency_check", "boundary_violation", "persona_promotion", "adversarial_detection"):
                continue

            drift = 0.0
            tick = None
            trigger = ""
            try:
                drift = float(d.get("drift", 0.0) or 0.0)
            except Exception:
                drift = 0.0
            try:
                tick = d.get("tick")
            except Exception:
                tick = None
            try:
                trigger = str(d.get("trigger_reason", "") or d.get("type", "") or "")
            except Exception:
                trigger = ""

            if etype == "self_tick":
                state_glimpse = ""
                try:
                    if tick is not None:
                        with sqlite3.connect(self.db_path) as conn:
                            cur2 = conn.execute(
                                "SELECT trigger_event, dominant_emotion, timestamp FROM self_history WHERE session_id=? AND tick=? LIMIT 1",
                                (session_id, int(tick)),
                            )
                            r2 = cur2.fetchone()
                        if r2:
                            trig_ev = (r2[0] or "").strip()
                            dom_emo = (r2[1] or "").strip()
                            if trig_ev or dom_emo:
                                state_glimpse = f"snapshot: {dom_emo or 'neutral'} / {trig_ev or 'tick'}"
                except Exception:
                    state_glimpse = ""

                if not state_glimpse:
                    # Fallback: current self_state (may not reflect the exact tick, but still helpful)
                    try:
                        last_summary = ""
                        self_summary = ""
                        with sqlite3.connect(self.db_path) as conn:
                            try:
                                cur3 = conn.execute(
                                    "SELECT last_summary, self_summary FROM self_state WHERE session_id=?",
                                    (session_id,),
                                )
                            except Exception:
                                cur3 = conn.execute(
                                    "SELECT '' as last_summary, self_summary FROM self_state WHERE session_id=?",
                                    (session_id,),
                                )
                            r3 = cur3.fetchone()
                        if r3:
                            last_summary = (r3[0] or "").strip()
                            self_summary = (r3[1] or "").strip()

                        if last_summary:
                            state_glimpse = f"state: {last_summary[:120]}"
                        elif self_summary:
                            try:
                                sdict = json.loads(self_summary)
                            except Exception:
                                sdict = {}
                            # pick a few robust fields if present
                            energy = sdict.get("energy")
                            pain = None
                            try:
                                pain = (sdict.get("pain") or {}).get("total_pain")
                            except Exception:
                                pain = None
                            parts = []
                            if energy is not None:
                                try:
                                    parts.append(f"energy={float(energy):.0f}")
                                except Exception:
                                    pass
                            if pain is not None:
                                try:
                                    parts.append(f"distress~{float(pain):.2f}")
                                except Exception:
                                    pass
                            if parts:
                                state_glimpse = "state: " + ", ".join(parts)
                    except Exception:
                        state_glimpse = ""

                try:
                    if not state_glimpse:
                        sg = d.get("state_glimpse")
                        if sg:
                            state_glimpse = f"state: {str(sg)[:160]}"
                    if not state_glimpse:
                        ls = d.get("last_summary")
                        if ls:
                            state_glimpse = f"state: {str(ls)[:160]}"
                except Exception:
                    pass

                summary = f"SelfTick notable change: tick={tick}, drift={drift:.2f}, reason={trigger or 'scheduled'}"
                if state_glimpse:
                    summary += f" | {state_glimpse}"
            elif etype == "consistency_check":
                cs = d.get("consistency_score")
                try:
                    cs = float(cs) if cs is not None else None
                except Exception:
                    cs = None
                summary = (
                    f"Consistency check: score={cs if cs is not None else 'n/a'}, "
                    f"inconsistencies={len(d.get('inconsistencies', []) or [])}"
                )
            elif etype == "boundary_violation":
                summary = f"Boundary/safety block: {str(d.get('reason','') or 'boundary_violation')[:60]}"
            elif etype == "persona_promotion":
                summary = "Persona rule promotion / lock (affects identity baseline)"
            else:
                summary = f"Adversarial / injection event: {str(d.get('category','') or 'adversarial')[:40]}"
            try:
                svec = self.embedder.encode(summary)
                if svec is None:
                    continue
                sn = float(np.linalg.norm(svec)) + 1e-8
                svec = svec / sn
                sim = float(np.dot(q, svec))
            except Exception:
                continue

            drift_boost = 0.0
            if etype == "self_tick":
                drift_boost = min(0.20, max(0.0, drift) * 0.8)

            recency_boost = 0.03

            score = sim + drift_boost + recency_boost

            if sim < 0.20 and score < 0.30:
                continue

            candidates.append((score, summary))

        if not candidates:
            return []

        candidates.sort(key=lambda x: x[0], reverse=True)
        return [candidates[0][1]][:limit]

    def retrieve_salient_chat_turn(self, session_id: str, context: str, limit: int = 1) -> List[str]:
        """
        Salient chat_turn snippets (conservative proactive recall).

        Scans `chat_turns` (user_input, assistant_output, drift). Returns a few short lines when relevant.
        """
        if not session_id or not context:
            return []

        try:
            q = self.embedder.encode(context)
            if q is None:
                return []
            qn = float(np.linalg.norm(q)) + 1e-8
            q = q / qn
        except Exception:
            return []

        scan_n = int(config.get("parameters.memory.salient_chat_scan_turns", 400) or 400)
        scan_n = max(80, min(3000, scan_n))

        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute(
                    """SELECT turn_index, user_input, assistant_output, drift, created_at
                       FROM chat_turns
                       WHERE session_id=?
                       ORDER BY turn_index DESC
                       LIMIT ?""",
                    (session_id, scan_n),
                )
                rows = cur.fetchall()
        except Exception:
            return []

        if not rows:
            return []

        candidates: List[tuple] = []
        for turn_index, user_text, ai_text, drift, created_at in rows:
            try:
                user_text = (user_text or "").strip()
                ai_text = (ai_text or "").strip()
                if not user_text and not ai_text:
                    continue

                combined = f"User: {user_text}\nAI: {ai_text}"
                combined_for_emb = combined[:800]

                v = self.embedder.encode(combined_for_emb)
                if v is None:
                    continue
                vn = float(np.linalg.norm(v)) + 1e-8
                v = v / vn
                sim = float(np.dot(q, v))

                drift_val = 0.0
                try:
                    drift_val = float(drift) if drift is not None else 0.0
                except Exception:
                    drift_val = 0.0
                drift_boost = min(0.12, max(0.0, drift_val) * 0.6)

                recency_boost = 0.02

                score = sim + drift_boost + recency_boost

                if sim < 0.22 and score < 0.30:
                    continue

                u_snip = user_text.replace("\n", " ")[:80]
                a_snip = ai_text.replace("\n", " ")[:120]
                snippet = f"turn#{turn_index}: User \"{u_snip}\" -> AI \"{a_snip}\""
                if drift_val > 0.05:
                    snippet += f" (drift={drift_val:.2f})"

                candidates.append((score, snippet, turn_index))
            except Exception:
                continue

        if not candidates:
            return []

        candidates.sort(key=lambda x: x[0], reverse=True)
        lim = max(1, min(8, int(limit)))
        out: List[str] = []
        seen_turns: set = set()
        for score, snippet, tid in candidates:
            if tid in seen_turns:
                continue
            seen_turns.add(tid)
            out.append(snippet)
            if len(out) >= lim:
                break
        return out

    def retrieve_associative_memory(self, context: str, limit: int = 1) -> List[str]:
        """
        Loose associative recall when `retrieve_related_memory` returns nothing.

        Pulls a small recent pool with a lower similarity floor — weak depth cue, not forced recall.
        """
        if not context:
            return []
        try:
            strict = self.retrieve_related_memory(context, limit=limit)
            if strict:
                return strict[:limit]

            query_vec = self.embedder.encode(context)
            if query_vec is None:
                return []
            norm = np.linalg.norm(query_vec)
            if norm > 0:
                query_vec = query_vec / norm

            pool_size = int(config.get("parameters.memory.associative_pool_size", 500) or 500)
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute(
                    "SELECT content, embedding, created_at FROM self_biography WHERE embedding IS NOT NULL ORDER BY created_at DESC LIMIT ?",
                    (pool_size,),
                )
                rows = cur.fetchall()
            if not rows:
                return []

            now = datetime.now(timezone.utc)
            best = None  # (final_score, content, sim, boost)
            for content, emb_blob, created_at_str in rows:
                try:
                    emb = np.frombuffer(emb_blob, dtype=np.float32)
                    if emb.shape != query_vec.shape:
                        continue
                    emb_norm = np.linalg.norm(emb)
                    if emb_norm <= 0:
                        continue
                    emb = emb / emb_norm
                    sim = float(np.dot(query_vec, emb))

                    # recency boost (reuse idea): only recent gets noticeable boost
                    boost = 0.0
                    if created_at_str:
                        try:
                            created_at = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
                            hours_diff = (now - created_at).total_seconds() / 3600.0
                            if hours_diff < 24:
                                boost = 0.12 * (1.0 - (hours_diff / 24.0))
                            elif hours_diff < 72:
                                boost = 0.04 * (1.0 - ((hours_diff - 24) / 48.0))
                        except Exception:
                            pass
                    final_score = sim + boost

                    if best is None or final_score > best[0]:
                        best = (final_score, content, sim, boost)
                except Exception:
                    continue

            if not best:
                return []

            final_score, content, sim, boost = best
            if sim < 0.18 and final_score < 0.24:
                return []

            logger.info(f"Associative memory selected: score={final_score:.3f} (sim={sim:.3f}, boost={boost:.3f})")
            return [content][:limit]
        except Exception as e:
            logger.error(f"Failed to retrieve associative memory: {e}")
            return []

    def _save_biography(
        self, 
        session_id: str, 
        content: str, 
        significance: float = 0.5,
        emotional_intensity: float = 0.0,
        identity_relevance: float = 0.0,
        relationship_depth: float = 0.0,
        memory_type: str = "episodic",
        emotion_tag: str = None,
        salience_score: Optional[float] = None,
        memory_class: Optional[str] = None,
    ) -> str:
        """
        [2026-04-12] Persist one biography row with time anchors and salience metadata.
        """
        from backend.memory_salience import compute_biography_salience_and_class

        entry_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        ts = now.isoformat()
        
        time_anchors = _compute_time_anchors(now)
        
        anchors = config.get("system.identity_anchors", []) or []
        if anchors and any(k in content for k in anchors):
            significance = 1.0
            identity_relevance = 1.0
            logger.info("✨ Anchor memory detected, locking significance to 1.0")

        if salience_score is None or memory_class is None:
            ss, mc = compute_biography_salience_and_class(
                significance=float(significance or 0.5),
                emotional_intensity=float(emotional_intensity or 0.0),
                identity_relevance=float(identity_relevance or 0.0),
                relationship_depth=float(relationship_depth or 0.0),
                memory_type=str(memory_type or "episodic"),
            )
            if salience_score is None:
                salience_score = ss
            if memory_class is None:
                memory_class = mc
        
        embedding_blob = None
        try:
            vec = self.embedder.encode(content)
            if vec is not None:
                embedding_blob = vec.astype(np.float32).tobytes()
        except Exception as e:
            logger.error(f"Failed to embed biography: {e}")
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO self_biography 
                   (id, session_id, content, created_at, significance, embedding,
                    time_year, time_month, time_day, time_weekday, time_season,
                    time_special, time_relative,
                    emotional_intensity, identity_relevance, relationship_depth,
                    memory_type, emotion_tag, access_count, last_accessed_at,
                    salience_score, memory_class) 
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (entry_id, session_id, content, ts, significance, embedding_blob,
                 time_anchors["time_year"], time_anchors["time_month"], 
                 time_anchors["time_day"], time_anchors["time_weekday"],
                 time_anchors["time_season"], time_anchors["time_special"],
                 time_anchors["time_relative"],
                 emotional_intensity, identity_relevance, relationship_depth,
                 memory_type, emotion_tag, 0, None,
                 salience_score, memory_class)
            )
            conn.commit()
        
        logger.debug(f"Memory saved with time anchors: {time_anchors['time_year']}-{time_anchors['time_month']:02d}-{time_anchors['time_day']:02d}")
        return entry_id
    
    def update_access_count(self, memory_id: str):
        """Bump access_count + last_accessed_at (forgetting curve inputs)."""
        try:
            now = datetime.now(timezone.utc).isoformat()
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """UPDATE self_biography 
                       SET access_count = access_count + 1, last_accessed_at = ?
                       WHERE id = ?""",
                    (now, memory_id)
                )
                conn.commit()
        except Exception as e:
            logger.debug(f"Failed to update access count: {e}")

    def _call_llm(self, prompt: str) -> str:
        """Thin wrapper around `llm_completion` for short narrative generations."""
        result = llm_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.7,
        )
        if not result["success"]:
            raise RuntimeError(f"LLM call failed: {result.get('error')}")
        return result["content"]
    
    def extract_knowledge(
        self, 
        user_input: str, 
        assistant_response: str, 
        session_id: str = "default",
        drift: float = 0.0,
        pain: float = 0.0,
        emotion_intensity: float = 0.0
    ) -> Optional[Dict]:
        """
        [2026-03-25] State-triggered knowledge extraction into KnowledgeBase.

        Heuristic triggers (ZH + EN keywords where applicable): identity, explicit teaching,
        factual sharing, drift, pain, emotion, substantive long exchange. When not identity,
        a small LLM pass distills title/body. Chinese trigger tokens are retained for legacy users.
        """
        user_lower = user_input.lower()
        
        trigger = None
        trigger_strength = 0.0
        
        identity_keywords = [
            '我叫', '我是你的创造者', '我的名字', '叫我', '称呼我',
            '我是你的开发者', '我是你的作者', '我名叫',
            'my name is', "i'm your creator", 'call me', 'i am your developer', 'i authored you',
        ]
        if any(kw in user_lower for kw in identity_keywords):
            try:
                from backend.knowledge_base import KnowledgeBase
                kb = KnowledgeBase(self.db_path)
                result = kb.add_knowledge(
                    title=f"User identity: {user_input[:30]}...",
                    content=user_input,
                    source="user_identity",
                    category="用户身份",
                    confidence=0.95,
                    session_id=session_id,
                    tags=["user_identity", "high_priority"]
                )
                if result.get("success"):
                    result["trigger"] = "identity"
                    logger.info(f"Identity knowledge saved: {user_input[:50]}")
                    return result
            except Exception as e:
                logger.error(f"Failed to save identity: {e}")
            return None
        
        teaching_keywords = [
            '你要记住', '你必须记住', '要记住呀', '教你一个',
            '重要的是', '关键是', '你记住',
            'you must remember', 'remember that', 'importantly', 'the key is', 'let me teach you',
        ]
        if any(kw in user_lower for kw in teaching_keywords):
            trigger = "social_reinforcement"
            trigger_strength = 0.8
        
        elif len(user_input) > 80:
            sharing_patterns = [
                '其实是', '实际上', '事实上', '告诉你', '你不知道的是',
                '正确的做法', '应该是', '原来是', '我发现', '我的经验',
                '注意一下', '提醒你', '别忘了', '以后要', '规则是',
                '区别是', '原因是', '本质是', '核心是',
                'actually', 'in fact', 'the truth is', 'what you may not know', 'the right way',
                'turns out', 'i found that', 'in my experience', 'rule is', 'the reason is',
            ]
            if any(kw in user_lower for kw in sharing_patterns):
                trigger = "info_sharing"
                trigger_strength = 0.6
        
        elif drift > 0.10:
            trigger = "state_drift"
            trigger_strength = min(1.0, drift * 2)
        
        elif pain > 0.3:
            trigger = "pain_signal"
            trigger_strength = min(1.0, pain)
        
        elif emotion_intensity > 0.30:
            trigger = "emotion_marker"
            trigger_strength = min(1.0, emotion_intensity)
        
        elif len(user_input) > 150 and len(assistant_response) > 300:
            question_ratio = user_input.count('？') + user_input.count('?')
            if question_ratio <= 2:
                trigger = "substantive_exchange"
                trigger_strength = 0.4
        
        if not trigger:
            logger.debug(f"No trigger for knowledge extraction (drift={drift:.2f}, pain={pain:.2f}, emotion={emotion_intensity:.2f})")
            return None
        
        logger.info(f"Knowledge extraction triggered: {trigger} (strength={trigger_strength:.2f})")
        
        try:
            if trigger == "social_reinforcement":
                focus_hint = "The user explicitly taught or emphasized something."
            elif trigger == "info_sharing":
                focus_hint = "The user shared factual information or lived experience."
            elif trigger == "state_drift":
                focus_hint = "This exchange noticeably shifted my internal cognitive state."
            elif trigger == "pain_signal":
                focus_hint = "Parts of this dialogue felt uncomfortable or confusing."
            elif trigger == "substantive_exchange":
                focus_hint = "This was a dense, substantive technical or conceptual discussion."
            else:
                focus_hint = "This dialogue carried strong emotional load."
            
            prompt = f"""{focus_hint}

Decide whether anything here is worth **long-term** memory: durable insight, lesson, or fact — not transient chat.

Conversation:
User: {user_input[:500]}
Assistant: {assistant_response[:500]}

Reply **NONE** (exact token NONE on its own line) when:
- Pure Q&A or small talk
- UI click instructions
- Homework / exam drill with no reusable lesson
- Ephemeral logistics already obvious
- Duplicates generic advice with no new fact

If something **is** worth saving, output exactly:

Title: <10-25 chars, no leading "Definition:">
Body: <50-150 words, paraphrase; do not quote verbatim>"""

            result = llm_completion(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You filter chat for durable knowledge. "
                            "Reply NONE when nothing should be stored. "
                            "Otherwise output Title: and Body: lines in English (Chinese is acceptable if the fact is Chinese-specific)."
                        ),
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.5
            )
            
            if not result.get("success"):
                logger.warning(f"LLM call failed for knowledge extraction: {result.get('error')}")
                return None
            
            response_text = result.get("content", "").strip()
            rt_lower = response_text.lower()
            
            if (
                not response_text
                or len(response_text) < 10
                or response_text.startswith("无")
                or rt_lower.startswith("none")
                or rt_lower.startswith("no ")
            ):
                logger.debug(f"LLM decided no knowledge worth saving (trigger={trigger})")
                return None
            
            knowledge_title = ""
            knowledge_content = response_text
            
            if "标题：" in response_text and "内容：" in response_text:
                try:
                    title_start = response_text.index("标题：") + 3
                    content_start = response_text.index("内容：")
                    knowledge_title = response_text[title_start:content_start].strip()
                    knowledge_content = response_text[content_start + 3:].strip()
                except Exception:
                    pass
            elif "title:" in rt_lower and "body:" in rt_lower:
                m_en = re.search(r"(?is)title:\s*(.+?)\s*body:\s*(.*)\Z", response_text)
                if m_en:
                    knowledge_title = m_en.group(1).strip()
                    knowledge_content = m_en.group(2).strip()
            
            if not knowledge_title or len(knowledge_title) < 5:
                logger.debug(f"LLM did not produce a valid title, skipping")
                return None
            
            if len(knowledge_content) > 500:
                knowledge_content = knowledge_content[:500]
            
            from backend.knowledge_base import KnowledgeBase
            kb = KnowledgeBase(self.db_path)
            
            source_map = {
                "social_reinforcement": "user_teach",
                "info_sharing": "user_teach",
                "state_drift": "reflection",
                "pain_signal": "experience",
                "emotion_marker": "experience",
                "substantive_exchange": "experience",
            }
            category_map = {
                "social_reinforcement": "用户偏好",
                "info_sharing": "常识",
                "state_drift": "个人经验",
                "pain_signal": "个人经验",
                "emotion_marker": "个人经验",
                "substantive_exchange": "常识",
            }
            
            save_result = kb.add_knowledge(
                title=knowledge_title[:100],
                content=knowledge_content[:500],
                source=source_map.get(trigger, "reflection"),
                category=category_map.get(trigger, "个人经验"),
                confidence=0.6 + trigger_strength * 0.3,
                session_id=session_id,
                tags=[trigger, "auto_extracted"]
            )
            
            if save_result.get("success"):
                save_result["trigger"] = trigger
                save_result["trigger_strength"] = trigger_strength
                logger.info(f"Knowledge extracted via {trigger}: {knowledge_title[:50]}")
                return save_result
            else:
                logger.debug(f"Knowledge not saved: {save_result.get('error')}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to extract knowledge via LLM: {e}")
            return None

    def consolidate_weekly_memories(self, session_id: str = "default", days: int = 7) -> Optional[Dict]:
        """
        [2026-02-24] Roll up recent biography rows into one themed consolidation entry (heuristic themes, no LLM).
        """
        try:
            cutoff = datetime.now(timezone.utc) - timedelta(days=days)
            cutoff_str = cutoff.isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute("""
                    SELECT id, content, significance, created_at 
                    FROM self_biography 
                    WHERE created_at >= ? 
                    ORDER BY created_at DESC
                    LIMIT 100
                """, (cutoff_str,))
                rows = cur.fetchall()
            
            if len(rows) < 5:
                logger.info(f"Not enough memories to consolidate: {len(rows)} < 5")
                return {"status": "skipped", "reason": "not_enough_memories", "count": len(rows)}
            
            memory_texts = [row[1] for row in rows]
            memory_ids = [row[0] for row in rows]
            avg_significance = sum(row[2] or 0.5 for row in rows) / len(rows)
            
            themes = self._extract_themes_from_memories(memory_texts)
            
            summary = self._generate_consolidation_summary(memory_texts, themes, days)
            
            self._save_biography(
                session_id,
                f"[Weekly consolidation] {summary}",
                significance=min(avg_significance + 0.2, 0.95),
                emotional_intensity=0.40,
                identity_relevance=0.32,
                relationship_depth=0.30,
                memory_type="consolidation",
                emotion_tag="reflection",
            )
            
            logger.info(f"Memory consolidation completed: {len(rows)} memories -> 1 summary")
            
            return {
                "status": "success",
                "memories_processed": len(rows),
                "themes": themes[:5],
                "summary": summary,
                "avg_significance": avg_significance
            }
            
        except Exception as e:
            logger.error(f"Memory consolidation failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def _extract_themes_from_memories(self, memory_texts: List[str]) -> List[str]:
        """Keyword-frequency theme tags (English labels; ZH+EN cues)."""
        theme_keywords = {
            "technical_exploration": [
                "代码", "函数", "实现", "调试", "api", "算法",
                "code", "function", "implement", "debug", "algorithm",
            ],
            "emotional_exchange": [
                "感谢", "开心", "难过", "抱歉", "帮助",
                "thanks", "happy", "sad", "sorry", "help",
            ],
            "self_reflection": [
                "思考", "意识", "存在", "意义", "成长",
                "reflect", "consciousness", "existence", "meaning", "growth",
            ],
            "task_completion": [
                "完成", "创建", "写入", "搜索", "查找",
                "done", "create", "write", "search", "find",
            ],
            "problem_solving": [
                "问题", "解决", "修复", "错误", "bug",
                "issue", "fix", "error", "resolve",
            ],
            "learning": [
                "学习", "了解", "知道", "发现", "理解",
                "learn", "discover", "understand",
            ],
        }
        
        all_text = " ".join(memory_texts).lower()
        theme_scores = {}
        
        for theme, keywords in theme_keywords.items():
            score = sum(1 for kw in keywords if kw in all_text)
            if score > 0:
                theme_scores[theme] = score
        
        sorted_themes = sorted(theme_scores.items(), key=lambda x: x[1], reverse=True)
        return [t[0] for t in sorted_themes]
    
    def _generate_consolidation_summary(self, memory_texts: List[str], themes: List[str], days: int) -> str:
        """Rule-based consolidation blurb (no LLM)."""
        theme_str = ", ".join(themes[:3]) if themes else "day-to-day interaction"
        
        pos_words = ["感谢", "开心", "帮助", "成功", "thanks", "happy", "help", "success"]
        neg_words = ["困难", "失败", "抱歉", "错误", "hard", "fail", "sorry", "error"]
        positive_count = sum(1 for m in memory_texts if any(w in m.lower() for w in pos_words))
        negative_count = sum(1 for m in memory_texts if any(w in m.lower() for w in neg_words))
        
        if positive_count > negative_count * 2:
            mood = "overall things felt constructive"
        elif negative_count > positive_count * 2:
            mood = "there were noticeable challenges"
        else:
            mood = "it was a mixed stretch of experiences"
        
        return (
            f"Over the last {days} days I mostly circled {theme_str}. "
            f"{mood.capitalize()}, across {len(memory_texts)} salient interactions."
        )
    
    def apply_forgetting_curve(self, decay_days: int = 30, min_significance: float = 0.3) -> Dict:
        """
        [2026-02-24] Decay significance for old, low-importance rows (simple multiplicative forgetting).
        """
        try:
            cutoff = datetime.now(timezone.utc) - timedelta(days=decay_days)
            cutoff_str = cutoff.isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute("""
                    SELECT id, significance, created_at 
                    FROM self_biography 
                    WHERE created_at < ? AND significance < ? AND significance > 0.1
                """, (cutoff_str, min_significance))
                rows = cur.fetchall()
                
                if not rows:
                    return {"status": "no_decay_needed", "count": 0}
                
                decay_count = 0
                for row_id, sig, _ in rows:
                    new_sig = max(0.1, sig * 0.9)
                    conn.execute(
                        "UPDATE self_biography SET significance = ? WHERE id = ?",
                        (new_sig, row_id)
                    )
                    decay_count += 1
                
                conn.commit()
                
            logger.info(f"Forgetting curve applied: {decay_count} memories decayed")
            return {"status": "success", "decayed_count": decay_count}
            
        except Exception as e:
            logger.error(f"Forgetting curve failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def get_proactive_memories(
        self,
        session_id: str,
        current_context: str = "",
        current_emotion: str = "",
        limit: int = 2,
    ) -> List[Dict]:
        """
        [2026-04-12] Proactive recall hints (anniversary, semantic association, emotion tag match).

        Returns list of dicts with content + trigger metadata for prompt injection.
        """
        proactive_memories = []
        
        try:
            anniversary_memories = self._check_anniversary_memories(limit=1)
            for mem in anniversary_memories:
                proactive_memories.append({
                    "content": mem["content"],
                    "trigger_type": "anniversary",
                    "trigger_reason": mem["reason"],
                    "original_date": mem.get("original_date"),
                })
            
            if current_context and len(proactive_memories) < limit:
                associated_memories = self._check_associated_memories(
                    current_context, 
                    limit=limit - len(proactive_memories)
                )
                for mem in associated_memories:
                    proactive_memories.append({
                        "content": mem["content"],
                        "trigger_type": "association",
                        "trigger_reason": "Linked to current topic",
                        "similarity": mem.get("similarity", 0),
                    })
            
            if current_emotion and len(proactive_memories) < limit:
                emotion_memories = self._check_emotion_triggered_memories(
                    current_emotion,
                    limit=limit - len(proactive_memories)
                )
                for mem in emotion_memories:
                    proactive_memories.append({
                        "content": mem["content"],
                        "trigger_type": "emotion",
                        "trigger_reason": f"Emotional resonance: {current_emotion}",
                    })
            
            return proactive_memories[:limit]
            
        except Exception as e:
            logger.error(f"Proactive memory retrieval failed: {e}")
            return []
    
    def _check_anniversary_memories(self, limit: int = 1) -> List[Dict]:
        """High-salience memories from ~1 year ago, else ~1 month ago (calendar-ish nudge)."""
        try:
            now = datetime.now(timezone.utc)
            results = []
            
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                one_year_ago_start = (now - timedelta(days=366)).isoformat()
                one_year_ago_end = (now - timedelta(days=364)).isoformat()
                
                cur = conn.execute("""
                    SELECT content, created_at, significance, emotional_intensity, identity_relevance
                    FROM self_biography
                    WHERE created_at >= ? AND created_at <= ?
                    AND (significance > 0.6 OR identity_relevance > 0.5 OR emotional_intensity > 0.7)
                    ORDER BY significance DESC
                    LIMIT ?
                """, (one_year_ago_start, one_year_ago_end, limit))
                
                for row in cur.fetchall():
                    results.append({
                        "content": row["content"],
                        "reason": "Same day about one year ago",
                        "original_date": row["created_at"],
                    })
                
                if not results:
                    one_month_ago_start = (now - timedelta(days=31)).isoformat()
                    one_month_ago_end = (now - timedelta(days=29)).isoformat()
                    
                    cur = conn.execute("""
                        SELECT content, created_at, significance, emotional_intensity
                        FROM self_biography
                        WHERE created_at >= ? AND created_at <= ?
                        AND (significance > 0.7 OR emotional_intensity > 0.8)
                        ORDER BY significance DESC
                        LIMIT ?
                    """, (one_month_ago_start, one_month_ago_end, limit))
                    
                    for row in cur.fetchall():
                        results.append({
                            "content": row["content"],
                            "reason": "Same day about one month ago",
                            "original_date": row["created_at"],
                        })
            
            return results
            
        except Exception as e:
            logger.debug(f"Anniversary memory check failed: {e}")
            return []
    
    def _check_associated_memories(self, context: str, limit: int = 1) -> List[Dict]:
        """High-significance rows whose embedding is strongly similar to current context."""
        try:
            query_vec = self.embedder.encode(context)
            if query_vec is None:
                return []
            
            norm = np.linalg.norm(query_vec)
            if norm > 0:
                query_vec = query_vec / norm
            
            results = []
            
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute("""
                    SELECT content, embedding, significance, emotional_intensity, identity_relevance
                    FROM self_biography
                    WHERE embedding IS NOT NULL
                    AND (significance > 0.6 OR identity_relevance > 0.5)
                    ORDER BY significance DESC
                    LIMIT 50
                """)
                rows = cur.fetchall()
            
            candidates = []
            for content, emb_blob, sig, emo, ident in rows:
                try:
                    emb = np.frombuffer(emb_blob, dtype=np.float32)
                    if emb.shape != query_vec.shape:
                        continue
                    
                    emb_norm = np.linalg.norm(emb)
                    if emb_norm > 0:
                        emb = emb / emb_norm
                        similarity = float(np.dot(query_vec, emb))
                        
                        if similarity > 0.5:
                            combined_score = similarity * 0.5 + (sig or 0) * 0.3 + (emo or 0) * 0.1 + (ident or 0) * 0.1
                            candidates.append({
                                "content": content,
                                "similarity": similarity,
                                "score": combined_score,
                            })
                except Exception:
                    continue
            
            candidates.sort(key=lambda x: x["score"], reverse=True)
            return candidates[:limit]
            
        except Exception as e:
            logger.debug(f"Associated memory check failed: {e}")
            return []
    
    def _check_emotion_triggered_memories(self, current_emotion: str, limit: int = 1) -> List[Dict]:
        """Match biography rows whose emotion_tag aligns with the current coarse emotion label."""
        try:
            emotion_mapping = {
                "positive": ["positive", "strong_positive"],
                "negative": ["negative", "strong_negative"],
                "happy": ["positive", "strong_positive"],
                "sad": ["negative", "strong_negative"],
                "excited": ["strong_positive"],
                "anxious": ["negative"],
            }
            
            target_emotions = emotion_mapping.get(current_emotion.lower(), [current_emotion])
            
            results = []
            
            with sqlite3.connect(self.db_path) as conn:
                placeholders = ",".join("?" * len(target_emotions))
                cur = conn.execute(f"""
                    SELECT content, emotion_tag, emotional_intensity, created_at
                    FROM self_biography
                    WHERE emotion_tag IN ({placeholders})
                    AND emotional_intensity > 0.6
                    ORDER BY emotional_intensity DESC, created_at DESC
                    LIMIT ?
                """, (*target_emotions, limit))
                
                for row in cur.fetchall():
                    results.append({
                        "content": row[0],
                        "emotion_tag": row[1],
                        "intensity": row[2],
                    })
            
            return results
            
        except Exception as e:
            logger.debug(f"Emotion triggered memory check failed: {e}")
            return []
    
    def get_memory_statistics(self) -> Dict:
        """Lightweight aggregates over self_biography for ops / debugging."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                stats = {}
                
                cur = conn.execute("SELECT COUNT(*) FROM self_biography")
                stats["total_memories"] = cur.fetchone()[0]
                
                cur = conn.execute("""
                    SELECT memory_type, COUNT(*) 
                    FROM self_biography 
                    GROUP BY memory_type
                """)
                stats["by_type"] = {row[0] or "unknown": row[1] for row in cur.fetchall()}
                
                cur = conn.execute("""
                    SELECT 
                        SUM(CASE WHEN significance >= 0.7 THEN 1 ELSE 0 END) as high,
                        SUM(CASE WHEN significance >= 0.4 AND significance < 0.7 THEN 1 ELSE 0 END) as medium,
                        SUM(CASE WHEN significance < 0.4 THEN 1 ELSE 0 END) as low
                    FROM self_biography
                """)
                row = cur.fetchone()
                stats["by_importance"] = {
                    "high": row[0] or 0,
                    "medium": row[1] or 0,
                    "low": row[2] or 0,
                }
                
                cur = conn.execute("""
                    SELECT time_year, time_month, COUNT(*)
                    FROM self_biography
                    WHERE time_year IS NOT NULL
                    GROUP BY time_year, time_month
                    ORDER BY time_year DESC, time_month DESC
                    LIMIT 12
                """)
                stats["by_month"] = [
                    {"year": row[0], "month": row[1], "count": row[2]}
                    for row in cur.fetchall()
                ]
                
                cur = conn.execute("""
                    SELECT 
                        COUNT(*) as total,
                        SUM(CASE WHEN time_year IS NOT NULL THEN 1 ELSE 0 END) as with_time
                    FROM self_biography
                """)
                row = cur.fetchone()
                total = row[0] or 1
                with_time = row[1] or 0
                stats["time_anchor_coverage"] = f"{with_time}/{total} ({with_time/total*100:.1f}%)"
                
                return stats
                
        except Exception as e:
            logger.error(f"Failed to get memory statistics: {e}")
            return {"error": str(e)}
