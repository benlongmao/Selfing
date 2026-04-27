#!/usr/bin/env python3
"""
Daily narrative rollup (rule-based, no extra LLM call in this module).

- Weaves same-day ``self_biography`` snippets into a short first-person summary.
- Persists rows for later embedding retrieval during chat.
"""

import sqlite3
import json
import uuid
import logging
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Tuple
from collections import Counter

logger = logging.getLogger(__name__)


class DailyNarrativeGenerator:
    """Builds and stores one stitched narrative per UTC calendar day."""
    
    def __init__(self, db_path: str = "data.db"):
        self.db_path = db_path
        self._ensure_tables()
        self._embedder = None
    
    @property
    def embedder(self):
        if self._embedder is None:
            from backend.embedder import get_embedder
            self._embedder = get_embedder()
        return self._embedder
    
    def _ensure_tables(self):
        """Create SQLite tables used by this helper (idempotent)."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS daily_narratives (
                        id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        date TEXT NOT NULL,              -- YYYY-MM-DD (UTC)
                        narrative TEXT NOT NULL,         -- first-person summary
                        themes TEXT,                     -- JSON list of theme labels
                        emotional_arc TEXT,              -- JSON mood arc
                        causal_links TEXT,               -- JSON link objects
                        memory_count INTEGER,            -- biography rows merged
                        significance REAL DEFAULT 0.7,
                        created_at TEXT NOT NULL,
                        embedding BLOB                   -- float32 vector bytes
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS theme_tracking (
                        id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        theme_name TEXT NOT NULL,
                        first_seen TEXT NOT NULL,
                        last_seen TEXT NOT NULL,
                        occurrence_count INTEGER DEFAULT 1,
                        evolution_notes TEXT,
                        current_status TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_daily_narratives_session_date 
                    ON daily_narratives(session_id, date)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_theme_tracking_session 
                    ON theme_tracking(session_id, theme_name)
                """)
                
                conn.commit()
                logger.info("DailyNarrativeGenerator tables initialized")
        except Exception as e:
            logger.error(f"Failed to ensure daily narrative tables: {e}")
    
    def check_and_generate_daily_narrative(
        self, 
        session_id: str, 
        force: bool = False
    ) -> Optional[Dict]:
        """
        If yesterday’s rollup is missing, generate it (typically at chat start).

        Args:
            session_id: session key
            force: rebuild even when a row already exists (tests)

        Returns:
            Result dict or ``None`` when nothing was generated.
        """
        try:
            today = datetime.now(timezone.utc).date()
            yesterday = today - timedelta(days=1)
            yesterday_str = yesterday.isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute("""
                    SELECT id FROM daily_narratives 
                    WHERE session_id = ? AND date = ?
                """, (session_id, yesterday_str))
                exists = cur.fetchone()
            
            if exists and not force:
                logger.debug(f"Daily narrative for {yesterday_str} already exists")
                return None
            
            return self.generate_daily_narrative(session_id, yesterday_str)
            
        except Exception as e:
            logger.error(f"Failed to check/generate daily narrative: {e}")
            return None
    
    def generate_daily_narrative(
        self, 
        session_id: str, 
        date_str: str
    ) -> Optional[Dict]:
        """
        Produce the rollup for ``date_str`` (YYYY-MM-DD, UTC).
        """
        try:
            memories = self._get_day_memories(session_id, date_str)
            
            if len(memories) < 3:
                logger.info(f"Not enough memories for {date_str}: {len(memories)} < 3")
                return {"status": "skipped", "reason": "not_enough_memories", "count": len(memories)}
            
            themes = self._extract_themes(memories)
            emotional_arc = self._analyze_emotional_arc(memories)
            causal_links = self._detect_causal_links(memories)
            narrative = self._compose_narrative(
                date_str, memories, themes, emotional_arc, causal_links
            )
            narrative_id = self._save_narrative(
                session_id, date_str, narrative, themes, 
                emotional_arc, causal_links, len(memories)
            )
            self._update_theme_tracking(session_id, date_str, themes)
            
            logger.info(f"Daily narrative generated for {date_str}: {len(memories)} memories -> 1 narrative")
            
            return {
                "status": "success",
                "id": narrative_id,
                "date": date_str,
                "memory_count": len(memories),
                "themes": themes,
                "narrative_preview": narrative[:100] + "..."
            }
            
        except Exception as e:
            logger.error(f"Failed to generate daily narrative for {date_str}: {e}")
            return {"status": "error", "error": str(e)}
    
    def _get_day_memories(self, session_id: str, date_str: str) -> List[Dict]:
        """Pull ``self_biography`` rows for the UTC window of ``date_str``."""
        memories = []
        try:
            date = datetime.fromisoformat(date_str)
            start = datetime(date.year, date.month, date.day, 0, 0, 0, tzinfo=timezone.utc)
            end = start + timedelta(days=1)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.execute("""
                    SELECT content, significance, created_at, emotion_tag
                    FROM self_biography
                    WHERE created_at >= ? AND created_at < ?
                    ORDER BY created_at ASC
                """, (start.isoformat(), end.isoformat()))
                
                for row in cur.fetchall():
                    memories.append({
                        "content": row["content"],
                        "significance": row["significance"] or 0.5,
                        "created_at": row["created_at"],
                        "emotion_tag": row["emotion_tag"]
                    })
                    
        except Exception as e:
            logger.error(f"Failed to get day memories: {e}")
            
        return memories
    
    def _extract_themes(self, memories: List[Dict]) -> List[str]:
        """Keyword bag → ranked English theme labels (bilingual keyword lists)."""
        theme_keywords = {
            "Tech exploration": [
                "代码", "函数", "实现", "调试", "算法", "编程", "开发",
                "code", "function", "implementation", "debug", "api", "algorithm", "programming",
            ],
            "Self-reflection": [
                "思考", "意识", "存在", "意义", "成长", "自我", "身份",
                "reflect", "consciousness", "existence", "meaning", "identity", "growth",
            ],
            "Emotional rapport": [
                "感谢", "开心", "难过", "抱歉", "帮助", "关心", "理解",
                "thanks", "happy", "sorry", "help", "care", "understand", "grateful",
            ],
            "Task collaboration": [
                "完成", "创建", "写入", "搜索", "查找", "修改", "整理",
                "complete", "create", "write", "search", "find", "modify", "organize",
            ],
            "Problem solving": [
                "问题", "解决", "修复", "错误", "排查", "分析",
                "problem", "solve", "fix", "error", "bug", "analyze", "troubleshoot",
            ],
            "Learning & growth": [
                "学习", "了解", "知道", "发现", "理解", "掌握", "研究",
                "learn", "study", "discover", "understand", "research", "master",
            ],
            "Creative expression": [
                "想法", "创意", "设计", "灵感", "创作", "构思",
                "idea", "creative", "design", "inspiration", "draft", "brainstorm",
            ],
            "Social dialogue": [
                "对话", "交流", "讨论", "聊天", "沟通", "分享",
                "dialogue", "discuss", "chat", "share", "conversation", "talk",
            ],
        }
        
        all_text = " ".join([m["content"] for m in memories]).lower()
        theme_scores: Dict[str, int] = {}
        
        for theme, keywords in theme_keywords.items():
            score = sum(all_text.count(kw.lower()) for kw in keywords)
            if score > 0:
                theme_scores[theme] = score
        
        sorted_themes = sorted(theme_scores.items(), key=lambda x: x[1], reverse=True)
        return [t[0] for t in sorted_themes[:4]]
    
    def _analyze_emotional_arc(self, memories: List[Dict]) -> Dict:
        """Slice memories into coarse mood segments + overall trend."""
        positive_words = [
            "感谢", "开心", "成功", "满意", "喜欢", "好", "棒", "赞",
            "thanks", "great", "success", "happy", "love", "awesome", "good",
        ]
        negative_words = [
            "困难", "失败", "抱歉", "错误", "问题", "难过", "失望",
            "fail", "sorry", "error", "hard", "sad", "disappointed", "stuck",
        ]
        
        segments = []
        segment_size = max(1, len(memories) // 3)
        
        for i in range(0, len(memories), segment_size):
            segment = memories[i:i+segment_size]
            text = " ".join([m["content"] for m in segment]).lower()
            
            pos_count = sum(text.count(w.lower()) for w in positive_words)
            neg_count = sum(text.count(w.lower()) for w in negative_words)
            
            if pos_count > neg_count * 1.5:
                mood = "positive"
            elif neg_count > pos_count * 1.5:
                mood = "negative"
            else:
                mood = "neutral"
            
            segments.append(mood)
        
        if len(segments) >= 2:
            if segments[-1] == "positive" and segments[0] != "positive":
                trend = "improving"
            elif segments[-1] == "negative" and segments[0] != "negative":
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "stable"
        
        return {
            "segments": segments,
            "trend": trend,
            "dominant": max(set(segments), key=segments.count) if segments else "neutral"
        }
    
    def _detect_causal_links(self, memories: List[Dict]) -> List[Dict]:
        """Lightweight cue scan for causal / continuation language."""
        causal_words = [
            "因为", "所以", "导致", "因此", "于是", "结果", "由于",
            "because", "therefore", "thus", "hence", "led to", "as a result",
        ]
        continuation_words = [
            "继续", "接着", "然后", "之后", "进一步",
            "continue", "then", "next", "afterward", "further",
        ]
        
        links = []
        
        for i, mem in enumerate(memories):
            content = mem["content"]
            content_lower = content.lower()
            
            for word in causal_words:
                if word.lower() in content_lower or word in content:
                    links.append({
                        "type": "causal",
                        "index": i,
                        "keyword": word,
                        "snippet": content[:50]
                    })
                    break
            
            for word in continuation_words:
                if (word.lower() in content_lower or word in content) and i > 0:
                    links.append({
                        "type": "continuation",
                        "from_index": i - 1,
                        "to_index": i,
                        "keyword": word
                    })
                    break
        
        return links[:5]
    
    def _compose_narrative(
        self, 
        date_str: str, 
        memories: List[Dict], 
        themes: List[str],
        emotional_arc: Dict,
        causal_links: List[Dict]
    ) -> str:
        """Rule-based English first-person stitch (no extra LLM call here)."""
        date = datetime.fromisoformat(date_str)
        date_display = date.strftime("%Y-%m-%d")
        
        theme_str = ", ".join(themes[:3]) if themes else "day-to-day interaction"
        
        mood_map = {
            "positive": "the tone felt mostly upbeat",
            "negative": "the day carried some friction or worry",
            "neutral": "the tone stayed even-keeled",
        }
        trend_map = {
            "improving": "things seemed to brighten toward the end",
            "declining": "energy dipped as the day went on",
            "stable": "the rhythm stayed steady",
        }
        mood_desc = mood_map.get(emotional_arc.get("dominant", "neutral"), "the tone stayed even-keeled")
        trend_desc = trend_map.get(emotional_arc.get("trend", "stable"), "")
        
        important_memories = sorted(memories, key=lambda x: x["significance"], reverse=True)[:3]
        key_events = []
        for m in important_memories:
            content = m["content"]
            if len(content) > 60:
                content = content[:60] + "..."
            key_events.append(content)
        
        parts = [f"[{date_display}]"]
        parts.append(f"Most of the day clustered around: {theme_str}.")
        
        if key_events:
            parts.append(f"A line that stayed with me: {key_events[0]}")
        
        parts.append(f"Overall, {mood_desc}; {trend_desc}." if trend_desc else f"Overall, {mood_desc}.")
        
        if causal_links:
            parts.append("I can see a few moments chaining cause and effect.")
        
        parts.append(f"({len(memories)} substantive interactions woven together.)")
        
        return " ".join(parts)
    
    def _save_narrative(
        self,
        session_id: str,
        date_str: str,
        narrative: str,
        themes: List[str],
        emotional_arc: Dict,
        causal_links: List[Dict],
        memory_count: int
    ) -> str:
        """Insert the rollup row and return its id."""
        narrative_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        
        embedding_blob = None
        try:
            vec = self.embedder.encode(narrative)
            if vec is not None:
                embedding_blob = vec.astype(np.float32).tobytes()
        except Exception as e:
            logger.warning(f"Failed to embed narrative: {e}")
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO daily_narratives 
                (id, session_id, date, narrative, themes, emotional_arc, 
                 causal_links, memory_count, created_at, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                narrative_id,
                session_id,
                date_str,
                narrative,
                json.dumps(themes, ensure_ascii=False),
                json.dumps(emotional_arc, ensure_ascii=False),
                json.dumps(causal_links, ensure_ascii=False),
                memory_count,
                now,
                embedding_blob
            ))
            conn.commit()
        
        return narrative_id
    
    def _update_theme_tracking(self, session_id: str, date_str: str, themes: List[str]):
        """Upsert ``theme_tracking`` rows for the themes we just emitted."""
        now = datetime.now(timezone.utc).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            for theme in themes:
                cur = conn.execute("""
                    SELECT id, occurrence_count FROM theme_tracking
                    WHERE session_id = ? AND theme_name = ?
                """, (session_id, theme))
                row = cur.fetchone()
                
                if row:
                    conn.execute("""
                        UPDATE theme_tracking
                        SET last_seen = ?, occurrence_count = ?, updated_at = ?
                        WHERE id = ?
                    """, (date_str, row[1] + 1, now, row[0]))
                else:
                    theme_id = str(uuid.uuid4())
                    conn.execute("""
                        INSERT INTO theme_tracking
                        (id, session_id, theme_name, first_seen, last_seen, 
                         occurrence_count, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, 1, ?, ?)
                    """, (theme_id, session_id, theme, date_str, date_str, now, now))
            
            conn.commit()
    
    def retrieve_relevant_narratives(
        self, 
        session_id: str, 
        query: str, 
        limit: int = 3
    ) -> List[Dict]:
        """Embedding + recency score over recent narratives for prompt injection."""
        try:
            query_vec = self.embedder.encode(query)
            if query_vec is None:
                return []
            
            norm = np.linalg.norm(query_vec)
            if norm > 0:
                query_vec = query_vec / norm
            
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.execute("""
                    SELECT id, date, narrative, themes, emotional_arc, 
                           memory_count, embedding
                    FROM daily_narratives
                    WHERE session_id = ? AND embedding IS NOT NULL
                    ORDER BY date DESC
                    LIMIT 30
                """, (session_id,))
                rows = cur.fetchall()
            
            if not rows:
                return []
            
            candidates = []
            for row in rows:
                try:
                    emb = np.frombuffer(row["embedding"], dtype=np.float32)
                    if emb.shape != query_vec.shape:
                        continue
                    
                    emb_norm = np.linalg.norm(emb)
                    if emb_norm > 0:
                        emb = emb / emb_norm
                    
                    sim = float(np.dot(query_vec, emb))
                    
                    date = datetime.fromisoformat(row["date"])
                    days_ago = (datetime.now(timezone.utc).date() - date.date()).days
                    recency_boost = max(0, 0.1 * (1 - days_ago / 30))
                    
                    final_score = sim + recency_boost
                    
                    if sim > 0.2 or final_score > 0.25:
                        candidates.append({
                            "date": row["date"],
                            "narrative": row["narrative"],
                            "themes": json.loads(row["themes"]) if row["themes"] else [],
                            "memory_count": row["memory_count"],
                            "similarity": sim,
                            "score": final_score
                        })
                except Exception:
                    continue
            
            candidates.sort(key=lambda x: x["score"], reverse=True)
            return candidates[:limit]
            
        except Exception as e:
            logger.error(f"Failed to retrieve relevant narratives: {e}")
            return []
    
    def get_theme_evolution(self, session_id: str, theme_name: str) -> Optional[Dict]:
        """Return tracking metadata for a single theme label."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.execute("""
                    SELECT * FROM theme_tracking
                    WHERE session_id = ? AND theme_name = ?
                """, (session_id, theme_name))
                row = cur.fetchone()
            
            if not row:
                return None
            
            return {
                "theme": row["theme_name"],
                "first_seen": row["first_seen"],
                "last_seen": row["last_seen"],
                "occurrence_count": row["occurrence_count"],
                "current_status": row["current_status"],
                "evolution_notes": json.loads(row["evolution_notes"]) if row["evolution_notes"] else []
            }
            
        except Exception as e:
            logger.error(f"Failed to get theme evolution: {e}")
            return None
    
    def get_recent_narratives(self, session_id: str, days: int = 7) -> List[Dict]:
        """Return recent rows for dashboards or debugging."""
        try:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).date().isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.execute("""
                    SELECT date, narrative, themes, emotional_arc, memory_count
                    FROM daily_narratives
                    WHERE session_id = ? AND date >= ?
                    ORDER BY date DESC
                """, (session_id, cutoff))
                
                return [dict(row) for row in cur.fetchall()]
                
        except Exception as e:
            logger.error(f"Failed to get recent narratives: {e}")
            return []


_generator_instance = None

def get_daily_narrative_generator(db_path: str = "data.db") -> DailyNarrativeGenerator:
    """Process-wide singleton accessor."""
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = DailyNarrativeGenerator(db_path)
    return _generator_instance
