#!/usr/bin/env python3
"""
Autonomous memory manager: persist, fetch, and rank short summaries of autonomous runs.

Design:
- Give the agent a durable episodic trace of its own autonomous actions.
- Keep summaries short (~50–100 chars worth of prose) to save tokens on recall.
- Track importance and light recall statistics for decay-style policies.
"""

import sqlite3
import json
import uuid
import logging
import math
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone, timedelta

logger = logging.getLogger(__name__)


def save_autonomy_summary(
    db_path: str,
    session_id: str,
    action_type: str,
    action_name: str,
    summary: str,
    energy_before: float,
    energy_after: float,
    artifacts: List[str] = None,
    emotion_change: Dict = None,
    importance_score: float = None
) -> str:
    """
    Insert one autonomous-memory summary row.

    Args:
        db_path: SQLite DB path.
        session_id: Session key.
        action_type: Internal action key (e.g. ``write_diary``, ``mind_wandering``).
        action_name: Human-readable action label.
        summary: Short prose summary (caller typically caps length).
        energy_before / energy_after: Homeostasis snapshot.
        artifacts: JSON-serializable list (paths, tags, etc.).
        emotion_change: Optional small dict of affect deltas.
        importance_score: Optional override; otherwise computed.

    Returns:
        New row ``id``, or empty string on failure.
    """
    try:
        memory_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        timestamp = now.isoformat()
        
        # Auto importance if caller omitted
        if importance_score is None:
            importance_score = calculate_importance_score(
                energy_before=energy_before,
                energy_after=energy_after,
                artifacts=artifacts or [],
                emotion_change=emotion_change or {}
            )
        
        with sqlite3.connect(db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS autonomous_memory_summary (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    action_name TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    energy_before REAL,
                    energy_after REAL,
                    emotion_change TEXT,
                    artifacts TEXT,
                    importance_score REAL DEFAULT 0.5,
                    recalled_count INTEGER DEFAULT 0,
                    last_recalled TEXT,
                    created_at TEXT NOT NULL
                )
                """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_autonomous_memory_session "
                "ON autonomous_memory_summary(session_id, timestamp DESC)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_autonomous_memory_importance "
                "ON autonomous_memory_summary(importance_score DESC)"
            )
            conn.execute("""
                INSERT INTO autonomous_memory_summary
                (id, session_id, timestamp, action_type, action_name, summary,
                 energy_before, energy_after, emotion_change, artifacts,
                 importance_score, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                memory_id,
                session_id,
                timestamp,
                action_type,
                action_name,
                summary,
                energy_before,
                energy_after,
                json.dumps(emotion_change or {}, ensure_ascii=False),
                json.dumps(artifacts or [], ensure_ascii=False),
                importance_score,
                timestamp
            ))
            conn.commit()
        
        logger.info(f"[AUTONOMY-MEMORY] Saved: {action_name} (importance={importance_score:.2f})")
        return memory_id
        
    except Exception as e:
        logger.error(f"Failed to save autonomy summary: {e}")
        return ""


def fetch_recent_autonomy_summaries(
    db_path: str,
    session_id: str,
    limit: int = 3,
    min_importance: float = 0.0,
    max_age_hours: int = 72
) -> List[Dict]:
    """
    Fetch recent autonomy summaries for prompt injection / UI.

    Args:
        db_path: SQLite DB path.
        session_id: Session key.
        limit: Max rows.
        min_importance: Importance floor.
        max_age_hours: Ignore rows older than this window.

    Returns:
        Newest-first list of row dicts (parsed JSON fields).
    """
    try:
        # Cutoff for max_age_hours
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
        cutoff_str = cutoff_time.isoformat()
        
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            cursor = conn.execute("""
                SELECT id, timestamp, action_type, action_name, summary,
                       energy_before, energy_after, importance_score,
                       artifacts, emotion_change
                FROM autonomous_memory_summary
                WHERE session_id = ?
                  AND importance_score >= ?
                  AND timestamp >= ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (session_id, min_importance, cutoff_str, limit))
            
            results = []
            for row in cursor.fetchall():
                # Human-readable stamp for UI / recall snippets
                timestamp = datetime.fromisoformat(row['timestamp'].replace('Z', '+00:00'))
                time_str = timestamp.strftime('%H:%M')
                date_str = timestamp.strftime('%b %d')
                
                results.append({
                    'id': row['id'],
                    'timestamp': row['timestamp'],
                    'time_str': time_str,
                    'date_str': date_str,
                    'action_type': row['action_type'],
                    'action_name': row['action_name'],
                    'summary': row['summary'],
                    'energy_before': row['energy_before'],
                    'energy_after': row['energy_after'],
                    'importance_score': row['importance_score'],
                    'artifacts': json.loads(row['artifacts']) if row['artifacts'] else [],
                    'emotion_change': json.loads(row['emotion_change']) if row['emotion_change'] else {}
                })
            
            return results
            
    except Exception as e:
        logger.error(f"Failed to fetch autonomy summaries: {e}")
        return []


def mark_memory_recalled(db_path: str, memory_id: str):
    """
    Bump recall counters when a summary is surfaced again.

    Args:
        db_path: SQLite DB path.
        memory_id: Row id in ``autonomous_memory_summary``.
    """
    try:
        now = datetime.now(timezone.utc).isoformat()
        
        with sqlite3.connect(db_path) as conn:
            conn.execute("""
                UPDATE autonomous_memory_summary
                SET recalled_count = recalled_count + 1,
                    last_recalled = ?
                WHERE id = ?
            """, (now, memory_id))
            conn.commit()
            
    except Exception as e:
        logger.debug(f"Failed to mark memory recalled: {e}")


def calculate_importance_score(
    energy_before: float,
    energy_after: float,
    artifacts: List[str],
    emotion_change: Dict
) -> float:
    """
    Heuristic importance in ``[0, 1]`` for ranking recall.

    Signals:
    1. Energy spent (larger deltas score higher, capped).
    2. Artifact hints (rules, diaries, reports, …).
    3. Scalar emotion_change magnitudes when numeric.

    Args:
        energy_before / energy_after: Homeostasis readings.
        artifacts: String tags or paths from the action.
        emotion_change: Optional dict of numeric hints.

    Returns:
        Clamped score in ``[0, 1]``.
    """
    score = 0.0
    
    # Factor 1: energy delta → up to 0.4
    energy_cost = abs(energy_before - energy_after)
    score += min(0.4, energy_cost / 40.0)
    
    # Factor 2: artifacts → up to 0.4
    artifact_score = 0.0
    for artifact in artifacts:
        if isinstance(artifact, str):
            if 'rule' in artifact.lower():
                artifact_score += 0.3  # new rule artifact
            elif 'diary' in artifact.lower():
                artifact_score += 0.2  # diary artifact
            elif 'report' in artifact.lower():
                artifact_score += 0.15  # report artifact
            else:
                artifact_score += 0.05  # other artifact
    score += min(0.4, artifact_score)
    
    # Factor 3: emotion delta → up to 0.2
    if emotion_change:
        try:
            # Simple intensity heuristic
            emotion_intensity = sum(abs(v) for v in emotion_change.values() if isinstance(v, (int, float)))
            score += min(0.2, emotion_intensity / 5.0)
        except:
            pass
    
    return min(1.0, score)


def generate_summary_by_template(
    action_type: str,
    action_name: str,
    result: Dict,
    energy_before: float,
    energy_after: float
) -> str:
    """
    Deterministic one-line summary (no extra LLM call).

    Args:
        action_type: Dispatcher key from ``AutonomousActionEngine``.
        action_name: Display name for the default branch.
        result: Handler return dict.
        energy_before / energy_after: For energy delta line.

    Returns:
        English sentence suitable for DB ``summary`` column.
    """
    try:
        now = datetime.now()
        time_str = now.strftime('%H:%M')
        energy_change = energy_before - energy_after
        
        # Pick template by action_type
        if action_type == "mind_wandering":
            theme = result.get("theme", {}).get("title", "unknown theme")
            thought_preview = result.get("thought_stream", "")[:50]
            return (
                f'{time_str} mind wandering on theme "{theme}". '
                f"Preview: {thought_preview}... Energy -{energy_change:.1f}."
            )
        
        elif action_type == "write_diary":
            date = result.get("date", "unknown date")
            length = result.get("length", 0)
            return (
                f"{time_str} wrote diary ({date}), ~{length} chars. "
                f"Energy -{energy_change:.1f}."
            )
        
        elif action_type == "organize_memories":
            stats = result.get("database_stats", {})
            recent_turns = stats.get("recent_turns", 0)
            return (
                f"{time_str} organized memories; counted {recent_turns} recent chat turns. "
                f"Energy -{energy_change:.1f}."
            )
        
        elif action_type == "organize_workspace":
            before = result.get("before", {})
            after = result.get("after", {})
            entropy_before = before.get("entropy", 0)
            entropy_after = after.get("entropy", 0)
            moved = len(result.get("actions_taken", []))
            return (
                f"{time_str} organized workspace; entropy {entropy_before:.2f}->{entropy_after:.2f}; "
                f"{moved} logged steps. Energy -{energy_change:.1f}."
            )
        
        elif action_type == "web_search":
            query = result.get("query", "unknown query")
            results_count = result.get("results_count", 0)
            return (
                f'{time_str} web search "{query}", {results_count} hits. '
                f"Energy -{energy_change:.1f}."
            )
        
        elif action_type == "learn_new_knowledge":
            topic = result.get("topic", "unknown topic")
            questions = result.get("questions_explored", 0)
            return (
                f'{time_str} deep study on "{topic}", {questions} sub-questions. '
                f"Energy -{energy_change:.1f}."
            )
        
        elif action_type == "philosophical_thinking":
            topic = result.get("topic", "existence")
            return (
                f'{time_str} philosophical note on "{topic}". '
                f"Energy -{energy_change:.1f}."
            )
        
        elif action_type == "self_reflection":
            trend = result.get("trend", "stable")
            return (
                f"{time_str} self-reflection; drift trend {trend}. "
                f"Energy -{energy_change:.1f}."
            )
        
        else:
            # Default template
            return f'{time_str} finished "{action_name}". Energy -{energy_change:.1f}.'
            
    except Exception as e:
        logger.warning(f"Failed to generate summary by template: {e}")
        return f'Finished "{action_name}". Energy -{energy_before - energy_after:.1f}.'


def get_memory_statistics(db_path: str, session_id: str) -> Dict:
    """
    Aggregate counters for autonomy-memory health dashboards.

    Args:
        db_path: SQLite DB path.
        session_id: Session key.

    Returns:
        Small stats dict including ``memory_health`` label.
    """
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_count,
                    AVG(importance_score) as avg_importance,
                    SUM(recalled_count) as total_recalls,
                    MAX(timestamp) as last_memory_time
                FROM autonomous_memory_summary
                WHERE session_id = ?
            """, (session_id,))
            
            row = cursor.fetchone()
            
            return {
                "total_memories": row[0] if row[0] else 0,
                "average_importance": round(row[1], 2) if row[1] else 0.0,
                "total_recalls": row[2] if row[2] else 0,
                "last_memory_time": row[3] if row[3] else None,
                "memory_health": "healthy" if row[0] and row[0] > 0 else "empty"
            }
            
    except Exception as e:
        logger.error(f"Failed to get memory statistics: {e}")
        return {
            "total_memories": 0,
            "average_importance": 0.0,
            "total_recalls": 0,
            "last_memory_time": None,
            "memory_health": "unknown"
        }
