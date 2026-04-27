#!/usr/bin/env python3
"""
Emotion memory: learned emotion patterns, trigger history, and a 16-D state vector (4 PAD+N subspaces).

Canonical ``emotion_name`` labels in stored rows remain **Chinese** for compatibility with
``emotion_phenomenology`` and reflection parsers; English aliases are accepted where noted.
"""
import os
import json
import sqlite3
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime, timezone, timedelta
import math

try:
    from backend.embedder import get_embedder
except Exception as e:
    import logging
    logging.warning(f"Failed to import real embedder: {e}, using fallback")
    from backend.embedder_fallback import get_embedder_fallback as get_embedder

logger = logging.getLogger(__name__)

# 16-D layout: 4 subspaces × 4 dims — PAD (Mehrabian & Russell, 1974) + novelty (Fontaine et al., 2007)
EMOTION_DIM = 16
EMOTION_SUBSPACE_DIMS = {
    "pleasure": (0, 4),      # P: valence (pleasant / unpleasant)
    "arousal": (4, 8),       # A: activation (calm / activated)
    "dominance": (8, 12),    # D: dominance (in control / powerless); legacy key: control
    "novelty": (12, 16),     # N: unpredictability (novel / expected); legacy key: social
}
# Legacy aliases (older code used control/social)
EMOTION_SUBSPACE_DIMS["control"] = EMOTION_SUBSPACE_DIMS["dominance"]
EMOTION_SUBSPACE_DIMS["social"] = EMOTION_SUBSPACE_DIMS["novelty"]

# Canonical CJK emotion labels (reference / prompts); vectors use the same names as keys below
EMOTION_TYPES = {
    "basic": ["快乐", "悲伤", "愤怒", "恐惧", "惊讶", "厌恶"],
    "complex": ["自豪", "羞愧", "嫉妒", "感激", "同情", "困惑", "不安"]
}

MAX_EMOTION_PATTERNS = int(os.environ.get("MAX_EMOTION_PATTERNS", "60"))
# Emotion patterns churn faster than rules — default replacement slack is 0.03
EMOTION_REPLACEMENT_THRESHOLD = float(os.environ.get("EMOTION_REPLACEMENT_THRESHOLD", "0.03"))
EMOTION_STALE_DAYS = int(os.environ.get("EMOTION_STALE_DAYS", "45"))


class EmotionCapacityError(RuntimeError):
    """Raised when the emotion-pattern store is at capacity and the new row cannot replace a weaker one."""
    pass

try:
    from backend.base_dimension_store import BaseDimensionStore
    BASE_STORE_AVAILABLE = True
except ImportError:
    BASE_STORE_AVAILABLE = False
    logger.warning("BaseDimensionStore not available, using standalone implementation")

@dataclass
class EmotionPattern:
    """One persisted emotion pattern row (text + embedding + optional 16-D vector)."""
    id: str
    text: str  # Human-readable trigger description (any locale)
    emotion_type: str  # "basic" | "complex"
    emotion_name: str  # Canonical label (typically Chinese; see _generate_emotion_vector)
    intensity: float  # 0..1
    embedding: Optional[np.ndarray] = None
    trigger_condition: str = ""
    evidence_count: int = 0
    created_at: str = ""
    last_seen_at: str = ""
    status: str = "active"  # active | archived
    is_core: int = 0
    locked: int = 0
    emotion_vector: Optional[np.ndarray] = None  # 16 floats, PAD+N layout

@dataclass
class EmotionState:
    """Per-session aggregate emotion vector and dominant label."""
    session_id: str
    emotion_vector: np.ndarray
    dominant_emotion: str = ""  # Canonical CJK label from _analyze_emotion_vector / DB
    intensity: float = 0.0
    updated_at: str = ""

class EmotionStore(BaseDimensionStore if BASE_STORE_AVAILABLE else object):
    """SQLite-backed emotion pattern store and session state updates."""
    
    def __init__(self, db_path: str = "data.db"):
        self.db_path = db_path
        
        # Optional phenomenology helper (English-first descriptions)
        try:
            from backend.emotion_phenomenology import EmotionPhenomenology
            self.phenomenology = EmotionPhenomenology()
            logger.info("EmotionPhenomenology initialized")
        except ImportError as e:
            logger.warning(f"EmotionPhenomenology not available: {e}")
            self.phenomenology = None
        except Exception as e:
            logger.warning(f"EmotionPhenomenology initialization failed: {e}")
            self.phenomenology = None
        
        if BASE_STORE_AVAILABLE:
            super().__init__(
                db_path=db_path,
                table_name="emotion_patterns",
                pattern_prefix="emotion-",
                max_patterns=MAX_EMOTION_PATTERNS,
                replacement_threshold=EMOTION_REPLACEMENT_THRESHOLD,
                stale_days=EMOTION_STALE_DAYS
            )
        else:
            self.embedder = get_embedder()
            self.max_patterns = MAX_EMOTION_PATTERNS
            self.replacement_threshold = EMOTION_REPLACEMENT_THRESHOLD
            # Dynamic intensity weights (same spirit as persona rule dynamic score)
            self.DYNAMIC_INTENSITY_BASE_WEIGHT = 0.3
            self.DYNAMIC_INTENSITY_EVIDENCE_WEIGHT = 0.4
            self.DYNAMIC_INTENSITY_RECENCY_WEIGHT = 0.2
            self.DYNAMIC_INTENSITY_CONTEXT_WEIGHT = 0.1  # reserved
            self.EVIDENCE_DECAY_HALF_LIFE_DAYS = 30.0
        
        self.embedder = get_embedder()
        self.dim = EMOTION_DIM
        self._ensure_tables()
    
    def _ensure_tables(self):
        """Create emotion tables and indexes if missing (idempotent)."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS emotion_patterns (
                        id TEXT PRIMARY KEY,
                        text TEXT NOT NULL,
                        emotion_type TEXT NOT NULL,
                        emotion_name TEXT NOT NULL,
                        intensity REAL DEFAULT 0.5,
                        embedding BLOB,
                        trigger_condition TEXT,
                        evidence_count INTEGER DEFAULT 0,
                        created_at TEXT NOT NULL,
                        last_seen_at TEXT NOT NULL,
                        status TEXT DEFAULT 'active',
                        is_core INTEGER DEFAULT 0,
                        locked INTEGER DEFAULT 0,
                        emotion_vector BLOB
                    )
                """)
                for column, default in (("is_core", "0"), ("locked", "0")):
                    try:
                        conn.execute(f"ALTER TABLE emotion_patterns ADD COLUMN {column} INTEGER DEFAULT {default}")
                    except sqlite3.OperationalError:
                        pass
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS emotion_states (
                        session_id TEXT PRIMARY KEY,
                        emotion_vector BLOB NOT NULL,
                        dominant_emotion TEXT,
                        intensity REAL DEFAULT 0.0,
                        updated_at TEXT NOT NULL
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS emotion_triggers (
                        id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        pattern_id TEXT,
                        trigger_source TEXT,  -- rule/user_feedback/task_difficulty/social_interaction
                        emotion_delta BLOB,  -- delta vector (16-D)
                        intensity_delta REAL,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY(pattern_id) REFERENCES emotion_patterns(id)
                    )
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_emotion_patterns_status 
                    ON emotion_patterns(status, emotion_type)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_emotion_triggers_session 
                    ON emotion_triggers(session_id, created_at)
                """)
                
                conn.commit()
                logger.info("Emotion tables ensured")
        except sqlite3.Error as e:
            logger.error(f"Database error ensuring emotion tables: {e}")
        except Exception as e:
            logger.error(f"Unexpected error ensuring emotion tables: {e}", exc_info=True)
    
    def add_emotion_pattern(
        self,
        text: str,
        emotion_type: str,
        emotion_name: str,
        intensity: float = 0.5,
        trigger_condition: str = "",
        is_core: bool = False,
        locked: bool = False
    ) -> EmotionPattern:
        """Insert a new active emotion pattern after capacity checks."""
        self._ensure_capacity(intensity)
        pattern_id = f"emotion-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
        created_at = datetime.now(timezone.utc).isoformat()
        
        embedding = self.embedder.encode(text)
        emotion_vector = self._generate_emotion_vector(emotion_type, emotion_name, intensity)
        
        pattern = EmotionPattern(
            id=pattern_id,
            text=text,
            emotion_type=emotion_type,
            emotion_name=emotion_name,
            intensity=intensity,
            embedding=embedding,
            trigger_condition=trigger_condition,
            evidence_count=0,
            created_at=created_at,
            last_seen_at=created_at,
            status="active",
            is_core=1 if is_core else 0,
            locked=1 if locked else 0,
            emotion_vector=emotion_vector
        )
        
        with sqlite3.connect(self.db_path) as conn:
            emb_blob = embedding.astype(np.float32).tobytes()
            emotion_vec_blob = emotion_vector.astype(np.float32).tobytes()
            
            conn.execute("""
                INSERT INTO emotion_patterns 
                (id, text, emotion_type, emotion_name, intensity, embedding, 
                 trigger_condition, evidence_count, created_at, last_seen_at, status, is_core, locked, emotion_vector)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pattern.id, pattern.text, pattern.emotion_type, pattern.emotion_name,
                pattern.intensity, emb_blob, pattern.trigger_condition,
                pattern.evidence_count, pattern.created_at, pattern.last_seen_at,
                pattern.status, pattern.is_core, pattern.locked, emotion_vec_blob
            ))
            conn.commit()
        
        logger.info(f"Added emotion pattern: {pattern_id} ({emotion_name}, intensity={intensity:.2f})")
        return pattern

    def _ensure_capacity(self, new_intensity: float) -> None:
        """Enforce max pattern count; archive lowest dynamic-intensity row if the newcomer is stronger."""
        if BASE_STORE_AVAILABLE:
            try:
                super()._ensure_capacity(new_intensity)
            except RuntimeError as e:
                raise EmotionCapacityError(str(e))
        else:
            with sqlite3.connect(self.db_path) as conn:
                current = self._count_active_patterns(conn)
                if current < self.max_patterns:
                    return
                
                candidate = self._select_replacement_candidate(conn)
                if not candidate:
                    raise EmotionCapacityError("Emotion pattern limit reached and no replacement candidate found.")
                
                candidate_dynamic_intensity = candidate["dynamic_intensity"]
                if new_intensity <= candidate_dynamic_intensity + self.replacement_threshold:
                    raise EmotionCapacityError(
                        f"Emotion pattern limit reached ({current} >= {self.max_patterns}) "
                        f"and new pattern intensity ({new_intensity:.3f}) not significantly higher than "
                        f"candidate dynamic intensity ({candidate_dynamic_intensity:.3f})."
                    )
                
                logger.info(
                    f"Emotion limit reached. Archiving pattern {candidate['id']} "
                    f"(base_intensity={candidate['intensity']:.3f}, dynamic_intensity={candidate_dynamic_intensity:.3f}) "
                    f"to insert new pattern (intensity={new_intensity:.3f})."
                )
                self._archive_pattern(conn, candidate["id"])

    def _count_active_patterns(self, conn: sqlite3.Connection) -> int:
        """Count active rows (BaseDimensionStore hook)."""
        cur = conn.execute("SELECT COUNT(*) FROM emotion_patterns WHERE status='active'")
        row = cur.fetchone()
        return row[0] if row else 0

    def _select_replacement_candidate(self, conn: sqlite3.Connection) -> Optional[Dict]:
        """Pick the weakest replaceable pattern (stale first), using dynamic intensity."""
        if BASE_STORE_AVAILABLE:
            return super()._select_replacement_candidate(conn)
        else:
            stale_cutoff = (datetime.now(timezone.utc) - timedelta(days=EMOTION_STALE_DAYS)).isoformat()

            cur = conn.execute(
                """
                SELECT id, intensity, evidence_count, last_seen_at, created_at
                FROM emotion_patterns
                WHERE status='active' AND locked=0 AND last_seen_at <= ?
                """,
                (stale_cutoff,)
            )
            candidates = cur.fetchall()
            
            if not candidates:
                cur = conn.execute(
                    """
                    SELECT id, intensity, evidence_count, last_seen_at, created_at
                    FROM emotion_patterns
                    WHERE status='active' AND locked=0
                    """
                )
                candidates = cur.fetchall()
            
            if not candidates:
                return None
            
            best_candidate = None
            lowest_dynamic_intensity = float('inf')
            
            for row in candidates:
                pattern_id = row[0]
                base_intensity = float(row[1])
                evidence_count = row[2]
                last_seen_at = row[3]
                created_at = row[4]
                
                dynamic_intensity = self._calculate_dynamic_intensity(
                    base_intensity=base_intensity,
                    evidence_count=evidence_count,
                    last_seen_at=last_seen_at,
                    created_at=created_at,
                    pattern_id=pattern_id
                )
                
                if dynamic_intensity < lowest_dynamic_intensity:
                    lowest_dynamic_intensity = dynamic_intensity
                    best_candidate = {
                        "id": pattern_id,
                        "intensity": base_intensity,
                        "dynamic_intensity": dynamic_intensity,
                        "evidence_count": evidence_count,
                        "last_seen_at": last_seen_at,
                        "created_at": created_at
                    }
            
            return best_candidate

    def _archive_pattern(self, conn: sqlite3.Connection, pattern_id: str) -> None:
        """Mark a pattern archived (BaseDimensionStore hook)."""
        conn.execute(
            "UPDATE emotion_patterns SET status='archived', last_seen_at=? WHERE id=?",
            (datetime.now(timezone.utc).isoformat(), pattern_id)
        )
        conn.commit()

    def _update_pattern_intensity(
        self,
        pattern_id: str,
        new_intensity: float,
        emotion_type: str,
        emotion_name: str
    ) -> bool:
        """Update stored intensity and recompute the 16-D vector (non-locked rows only)."""
        try:
            new_vec = self._generate_emotion_vector(emotion_type, emotion_name, new_intensity)
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    UPDATE emotion_patterns
                    SET intensity=?, emotion_vector=?, last_seen_at=?
                    WHERE id=? AND locked=0
                    """,
                    (
                        new_intensity,
                        new_vec.astype(np.float32).tobytes(),
                        datetime.now(timezone.utc).isoformat(),
                        pattern_id
                    )
                )
                conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to update emotion pattern intensity ({pattern_id}): {e}")
            return False
    
    def _generate_emotion_vector(
        self,
        emotion_type: str,
        emotion_name: str,
        intensity: float
    ) -> np.ndarray:
        """
        Build a 16-D PAD+N vector from ``emotion_name`` and scalar ``intensity``.

        Subspaces: pleasure (valence), arousal, dominance, novelty — each gets ``intensity`` scaling.
        Keys are **canonical Chinese** labels; common English aliases are normalized first.
        """
        vec = np.zeros(EMOTION_DIM, dtype=np.float32)

        en_to_canonical = {
            "joy": "快乐", "sadness": "悲伤", "anger": "愤怒", "fear": "恐惧",
            "surprise": "惊讶", "disgust": "厌恶",
            "pride": "自豪", "shame": "羞愧", "jealousy": "嫉妒", "gratitude": "感激",
            "sympathy": "同情", "confusion": "困惑", "unease": "不安", "anxiety": "焦虑",
            "fatigue": "疲惫", "insight": "顿悟", "emptiness": "空虚", "loneliness": "孤独",
            "awe": "敬畏", "detachment": "疏离",
        }
        name_key = (emotion_name or "").strip()
        name_key = en_to_canonical.get(name_key.lower(), name_key)

        emotion_mapping = {
            "快乐":  {"pleasure": 1.0,  "arousal": 0.5,  "dominance": 0.7,  "novelty": 0.3},
            "悲伤":  {"pleasure": -0.8, "arousal": -0.3, "dominance": -0.5, "novelty": -0.4},
            "愤怒":  {"pleasure": -0.5, "arousal": 1.0,  "dominance": 0.3,  "novelty": 0.4},
            "恐惧":  {"pleasure": -0.7, "arousal": 0.9,  "dominance": -0.8, "novelty": 0.7},
            "惊讶":  {"pleasure": 0.3,  "arousal": 0.8,  "dominance": 0.0,  "novelty": 1.0},
            "厌恶":  {"pleasure": -0.9, "arousal": 0.3,  "dominance": 0.2,  "novelty": -0.2},
            "自豪":  {"pleasure": 0.9,  "arousal": 0.6,  "dominance": 0.9,  "novelty": 0.2},
            "羞愧":  {"pleasure": -0.6, "arousal": 0.2,  "dominance": -0.7, "novelty": 0.5},
            "嫉妒":  {"pleasure": -0.5, "arousal": 0.4,  "dominance": -0.4, "novelty": 0.1},
            "感激":  {"pleasure": 0.8,  "arousal": 0.4,  "dominance": 0.6,  "novelty": 0.3},
            "同情":  {"pleasure": 0.2,  "arousal": 0.3,  "dominance": 0.3,  "novelty": 0.1},
            "困惑":  {"pleasure": -0.2, "arousal": 0.1,  "dominance": -0.6, "novelty": 0.8},
            "不安":  {"pleasure": -0.4, "arousal": 0.5,  "dominance": -0.7, "novelty": 0.6},
            "疲惫":  {"pleasure": -0.3, "arousal": -0.8, "dominance": -0.6, "novelty": -0.5},
            "顿悟":  {"pleasure": 0.9,  "arousal": 0.7,  "dominance": 0.8,  "novelty": 0.9},
            "空虚":  {"pleasure": -0.6, "arousal": -0.2, "dominance": -0.4, "novelty": -0.6},
            "孤独":  {"pleasure": -0.5, "arousal": -0.1, "dominance": -0.3, "novelty": -0.3},
            "敬畏":  {"pleasure": 0.5,  "arousal": 0.6,  "dominance": -0.2, "novelty": 0.8},
            "疏离":  {"pleasure": -0.3, "arousal": -0.2, "dominance": 0.2,  "novelty": -0.5},
        }
        
        if name_key in emotion_mapping:
            mapping = emotion_mapping[name_key]
            for dim_name in ("pleasure", "arousal", "dominance", "novelty"):
                s, e = EMOTION_SUBSPACE_DIMS[dim_name]
                vec[s:e] = mapping[dim_name] * intensity
        
        vec = np.clip(vec, -1.0, 1.0)
        return vec
    
    def get_emotion_state(self, session_id: str) -> Optional[EmotionState]:
        """Return session emotion state, or a neutral default row if none exists."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "SELECT emotion_vector, dominant_emotion, intensity, updated_at "
                "FROM emotion_states WHERE session_id = ?",
                (session_id,)
            )
            row = cur.fetchone()
            
            if row is None:
                return EmotionState(
                    session_id=session_id,
                    emotion_vector=np.zeros(EMOTION_DIM, dtype=np.float32),
                    dominant_emotion="中性",
                    intensity=0.0,
                    updated_at=datetime.now(timezone.utc).isoformat()
                )
            
            emotion_vec = np.frombuffer(row[0], dtype=np.float32)
            # Match z_self[32:48]: coerce to 16-D for legacy / corrupt blobs
            if emotion_vec.shape[0] != EMOTION_DIM:
                vec = np.zeros(EMOTION_DIM, dtype=np.float32)
                n = min(emotion_vec.shape[0], EMOTION_DIM)
                vec[:n] = emotion_vec[:n]
                emotion_vec = vec
            return EmotionState(
                session_id=session_id,
                emotion_vector=emotion_vec,
                dominant_emotion=row[1] or "中性",
                intensity=row[2] or 0.0,
                updated_at=row[3]
            )
    
    def update_emotion(
        self,
        session_id: str,
        emotion_delta: np.ndarray,
        trigger_source: str = "unknown",
        pattern_id: Optional[str] = None
    ) -> EmotionState:
        """
        EMA-style merge of ``emotion_delta`` into the session vector with inertia (stronger mood resists change).
        """
        current_state = self.get_emotion_state(session_id)

        # Inertia: higher prior intensity -> higher alpha -> less weight on new delta (Kuppens-style minute-scale mood)
        base_alpha = 0.55
        prev_intensity = min(1.0, current_state.intensity)
        dynamic_alpha = base_alpha + (0.15 * prev_intensity)
        
        new_emotion_vec = dynamic_alpha * current_state.emotion_vector + (1 - dynamic_alpha) * emotion_delta

        # Anti-flip: opposing-sign deltas are damped within the same turn
        for i in range(EMOTION_DIM):
            if current_state.emotion_vector[i] * emotion_delta[i] < -0.1:
                new_emotion_vec[i] = dynamic_alpha * current_state.emotion_vector[i] + (1 - dynamic_alpha) * (emotion_delta[i] * 0.5)

        new_emotion_vec = np.clip(new_emotion_vec, -1.0, 1.0)
        dominant_emotion, intensity = self._analyze_emotion_vector(new_emotion_vec)
        updated_at = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            emotion_vec_blob = new_emotion_vec.astype(np.float32).tobytes()
            conn.execute("""
                INSERT INTO emotion_states (session_id, emotion_vector, dominant_emotion, intensity, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    emotion_vector=excluded.emotion_vector,
                    dominant_emotion=excluded.dominant_emotion,
                    intensity=excluded.intensity,
                    updated_at=excluded.updated_at
            """, (session_id, emotion_vec_blob, dominant_emotion, intensity, updated_at))
            
            trigger_id = f"trigger-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
            delta_blob = emotion_delta.astype(np.float32).tobytes()
            intensity_delta = float(np.linalg.norm(emotion_delta))
            conn.execute("""
                INSERT INTO emotion_triggers 
                (id, session_id, pattern_id, trigger_source, emotion_delta, intensity_delta, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (trigger_id, session_id, pattern_id, trigger_source, delta_blob, intensity_delta, updated_at))
            
            conn.commit()
        
        logger.info(
            f"Updated emotion state for session {session_id}: "
            f"{dominant_emotion} (intensity={intensity:.2f})"
        )
        
        return EmotionState(
            session_id=session_id,
            emotion_vector=new_emotion_vec,
            dominant_emotion=dominant_emotion,
            intensity=intensity,
            updated_at=updated_at
        )
    
    def _analyze_emotion_vector(self, emotion_vec: np.ndarray) -> Tuple[str, float]:
        """Coarse discrete dominant label from PAD+N means + L2-derived intensity in [0,1]."""
        raw_intensity = float(np.linalg.norm(emotion_vec))
        intensity = raw_intensity / np.sqrt(len(emotion_vec))
        intensity = np.clip(intensity, 0.0, 1.0)
        logger.debug(
            f"Emotion intensity: raw={raw_intensity:.3f}, "
            f"normalized={intensity:.3f}, dim={len(emotion_vec)}"
        )
        
        pleasure = np.mean(emotion_vec[EMOTION_SUBSPACE_DIMS["pleasure"][0]:EMOTION_SUBSPACE_DIMS["pleasure"][1]])
        arousal = np.mean(emotion_vec[EMOTION_SUBSPACE_DIMS["arousal"][0]:EMOTION_SUBSPACE_DIMS["arousal"][1]])
        dominance = np.mean(emotion_vec[EMOTION_SUBSPACE_DIMS["dominance"][0]:EMOTION_SUBSPACE_DIMS["dominance"][1]])
        novelty = np.mean(emotion_vec[EMOTION_SUBSPACE_DIMS["novelty"][0]:EMOTION_SUBSPACE_DIMS["novelty"][1]])
        
        # Discrete labels (canonical CJK for phenomenology / reflection)
        if pleasure > 0.5 and dominance > 0.5:
            dominant = "自豪"
        elif pleasure > 0.5:
            dominant = "快乐"
        elif pleasure < -0.5 and arousal > 0.5 and dominance > 0:
            dominant = "愤怒"
        elif pleasure < -0.5 and dominance < -0.5:
            dominant = "焦虑"
        elif pleasure < -0.5 and arousal < -0.3:
            dominant = "悲伤"
        elif dominance < -0.5 and novelty > 0.5:
            dominant = "困惑"
        elif pleasure < -0.3:
            dominant = "不安"
        else:
            dominant = "中性"
        
        return dominant, intensity
    
    def get_emotion_phenomenology(
        self,
        session_id: str
    ) -> Optional[str]:
        """
        Short English-first phenomenology line for prompts (via ``EmotionPhenomenology``).

        Returns:
            Text or ``None`` if phenomenology is unavailable.
        """
        if not self.phenomenology:
            return None
        
        emotion_state = self.get_emotion_state(session_id)
        if emotion_state is None:
            return None
        
        phenomenology_text = self.phenomenology.describe_emotion_phenomenology(
            emotion_state.dominant_emotion,
            emotion_state.intensity
        )
        
        return phenomenology_text
    
    def get_emotion_trajectory(
        self,
        session_id: str,
        limit: int = 10
    ) -> str:
        """
        Build a short trajectory string from recent ``emotion_states`` rows (via phenomenology).
        """
        if not self.phenomenology:
            return ""
        
        emotion_history = []
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute("""
                    SELECT dominant_emotion, intensity, updated_at
                    FROM emotion_states
                    WHERE session_id = ?
                    ORDER BY updated_at DESC
                    LIMIT ?
                """, (session_id, limit))
                
                for row in cur.fetchall():
                    emotion_history.append((
                        row[0] or "中性",
                        row[1] or 0.0,
                        row[2] or ""
                    ))
        except Exception as e:
            logger.error(f"Failed to get emotion history for session {session_id}: {e}")
            return ""
        
        emotion_history.reverse()
        
        return self.phenomenology.generate_emotion_trajectory(emotion_history)
    
    def _calculate_dynamic_intensity(
        self,
        base_intensity: float,
        evidence_count: int,
        last_seen_at: Optional[str],
        created_at: Optional[str],
        pattern_id: str = ""
    ) -> float:
        """
        Evidence- and recency-weighted intensity used for ranking and replacement.

        Core locked rows (``is_core=1`` and ``locked=1``) short-circuit to ``1.0``.
        """
        now = datetime.now(timezone.utc)

        if evidence_count > 0:
            max_evidence = 100.0
            evidence_strength = min(1.0, math.log(1 + evidence_count) / math.log(1 + max_evidence))
        else:
            evidence_strength = 0.0
        
        if last_seen_at:
            try:
                last_seen = datetime.fromisoformat(last_seen_at.replace('Z', '+00:00'))
                if last_seen.tzinfo is None:
                    last_seen = last_seen.replace(tzinfo=timezone.utc)
                days_since_last_use = (now - last_seen).total_seconds() / 86400.0
                recency_factor = math.exp(-days_since_last_use / self.EVIDENCE_DECAY_HALF_LIFE_DAYS)
            except (ValueError, AttributeError) as e:
                logger.debug(f"Failed to parse last_seen_at '{last_seen_at}': {e}")
                recency_factor = 0.5
            except Exception as e:
                logger.warning(f"Unexpected error parsing last_seen_at '{last_seen_at}': {e}")
                recency_factor = 0.5
        else:
            recency_factor = 0.3
        
        context_relevance = 0.5

        if pattern_id and (pattern_id.startswith("emotion-") or pattern_id.startswith("emotion_")):
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cur = conn.execute(
                        "SELECT is_core, locked FROM emotion_patterns WHERE id = ?",
                        (pattern_id,)
                    )
                    row = cur.fetchone()
                    if row and row[0] == 1 and row[1] == 1:
                        return 1.0
            except sqlite3.Error as e:
                logger.debug(f"Database error checking core pattern: {e}")
            except Exception as e:
                logger.warning(f"Unexpected error checking core pattern: {e}")
        
        dynamic_intensity = (
            base_intensity * self.DYNAMIC_INTENSITY_BASE_WEIGHT +
            evidence_strength * self.DYNAMIC_INTENSITY_EVIDENCE_WEIGHT +
            recency_factor * self.DYNAMIC_INTENSITY_RECENCY_WEIGHT +
            context_relevance * self.DYNAMIC_INTENSITY_CONTEXT_WEIGHT
        )
        dynamic_intensity = max(0.0, min(1.0, dynamic_intensity))
        
        return dynamic_intensity
    
    def get_all_patterns(self, status: str = "active", limit: int = 100) -> List[EmotionPattern]:
        """
        Active patterns sorted by dynamic intensity; pinned core+locked row first when present.
        """
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "SELECT id, text, emotion_type, emotion_name, intensity, embedding, "
                "trigger_condition, evidence_count, created_at, last_seen_at, status, is_core, locked, emotion_vector "
                "FROM emotion_patterns WHERE status = ? ORDER BY created_at ASC",
                (status,)
            )
            rows = cur.fetchall()
        
        patterns_with_dynamic_intensity = []
        for row in rows:
            embedding = np.frombuffer(row[5], dtype=np.float32) if row[5] else None
            emotion_vec = np.frombuffer(row[13], dtype=np.float32) if row[13] else None
            
            dynamic_intensity = self._calculate_dynamic_intensity(
                base_intensity=row[4],
                evidence_count=row[7] or 0,
                last_seen_at=row[9],
                created_at=row[8],
                pattern_id=row[0]
            )
            
            pattern = EmotionPattern(
                id=row[0],
                text=row[1],
                emotion_type=row[2],
                emotion_name=row[3],
                intensity=row[4],
                embedding=embedding,
                trigger_condition=row[6],
                evidence_count=row[7],
                created_at=row[8],
                last_seen_at=row[9],
                status=row[10],
                is_core=row[11] or 0,
                locked=row[12] or 0,
                emotion_vector=emotion_vec
            )
            patterns_with_dynamic_intensity.append((pattern, dynamic_intensity))
        
        patterns_with_dynamic_intensity.sort(key=lambda x: x[1], reverse=True)
        patterns = [pattern for pattern, _ in patterns_with_dynamic_intensity[:limit]]
        first_pattern = None
        other_patterns = []
        for pattern in patterns:
            if pattern.is_core == 1 and pattern.locked == 1:
                first_pattern = pattern
            else:
                other_patterns.append(pattern)
        
        if first_pattern:
            return [first_pattern] + other_patterns
        else:
            return patterns
    
    def search_matching_patterns(
        self,
        evidence_text: str,
        trigger_condition: Optional[str] = None,
        top_k: int = 5,
        similarity_threshold: float = 0.6
    ) -> List[Tuple[EmotionPattern, float]]:
        """
        Cosine similarity over pattern embeddings; rerank with ``similarity * 0.6 + dynamic_intensity * 0.4``.
        """
        try:
            all_patterns = self.get_all_patterns(status="active", limit=200)

            if trigger_condition:
                all_patterns = [p for p in all_patterns if p.trigger_condition == trigger_condition]
            
            if not all_patterns:
                return []
            
            evidence_emb = self.embedder.encode(evidence_text)
            matches = []
            for pattern in all_patterns:
                if pattern.embedding is None:
                    continue

                similarity = np.dot(evidence_emb, pattern.embedding) / (
                    np.linalg.norm(evidence_emb) * np.linalg.norm(pattern.embedding) + 1e-8
                )
                
                if similarity >= similarity_threshold:
                    matches.append((pattern, float(similarity)))
            
            scored_matches = []
            for pattern, similarity in matches:
                dynamic_intensity = self._calculate_dynamic_intensity(
                    base_intensity=pattern.intensity,
                    evidence_count=pattern.evidence_count,
                    last_seen_at=pattern.last_seen_at,
                    created_at=pattern.created_at,
                    pattern_id=pattern.id
                )
                combined_score = similarity * 0.6 + dynamic_intensity * 0.4
                scored_matches.append((pattern, similarity, combined_score))
            
            scored_matches.sort(key=lambda x: x[2], reverse=True)
            return [(pattern, similarity) for pattern, similarity, _ in scored_matches[:top_k]]
        except Exception as e:
            logger.error(f"Failed to search matching emotion patterns: {e}", exc_info=True)
            return []
    
    def delete(self, pattern_id: str) -> bool:
        """Archive a pattern (soft delete)."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "UPDATE emotion_patterns SET status='archived' WHERE id=?",
                    (pattern_id,)
                )
                conn.commit()
            logger.info(f"Archived emotion pattern: {pattern_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete emotion pattern: {e}")
            return False
    
    def evolve_emotions(self, session_id: Optional[str] = None) -> Dict:
        """
        Periodic reinforcement / decay pass over unlocked patterns (evidence + staleness heuristics).

        Returns:
            ``patterns_analyzed``, ``patterns_evolved``, ``evolution_details``, ``evolution_summary``.
        """
        try:
            patterns = self.get_all_patterns(status="active", limit=100)

            if not patterns:
                return {
                    "patterns_analyzed": 0,
                    "patterns_evolved": 0,
                    "evolution_summary": "No active emotion patterns to evolve.",
                }
            
            evolved_count = 0
            evolution_details = []
            
            for pattern in patterns:
                if pattern.locked:
                    continue
                if pattern.evidence_count > 10:
                    if pattern.intensity < 0.9:
                        new_intensity = min(0.9, pattern.intensity + 0.05)
                        if self._update_pattern_intensity(
                            pattern.id,
                            new_intensity,
                            pattern.emotion_type,
                            pattern.emotion_name
                        ):
                            evolution_details.append({
                                "pattern_id": pattern.id,
                                "emotion_name": pattern.emotion_name,
                                "action": "reinforce",
                                "old_intensity": pattern.intensity,
                                "new_intensity": new_intensity,
                                "reason": f"High trigger rate (evidence_count={pattern.evidence_count})",
                                "applied": True
                            })
                            evolved_count += 1
                
                elif pattern.evidence_count < 3 and pattern.last_seen_at:
                    from datetime import datetime, timezone
                    try:
                        last_seen = datetime.fromisoformat(pattern.last_seen_at.replace('Z', '+00:00'))
                        days_since = (datetime.now(timezone.utc) - last_seen).days

                        if days_since > 30:
                            new_intensity = max(0.1, pattern.intensity - 0.1)
                            if self._update_pattern_intensity(
                                pattern.id,
                                new_intensity,
                                pattern.emotion_type,
                                pattern.emotion_name
                            ):
                                evolution_details.append({
                                    "pattern_id": pattern.id,
                                    "emotion_name": pattern.emotion_name,
                                    "action": "decay",
                                    "old_intensity": pattern.intensity,
                                    "new_intensity": new_intensity,
                                    "reason": f"Stale (no strong evidence, {days_since} days since last_seen)",
                                    "applied": True
                                })
                                evolved_count += 1
                    except Exception:
                        pass
            
            summary = f"Analyzed {len(patterns)} emotion pattern(s); {evolved_count} updated."
            if evolution_details:
                summary += f" Examples: {', '.join([d['emotion_name'] for d in evolution_details[:3]])}."
            
            return {
                "patterns_analyzed": len(patterns),
                "patterns_evolved": evolved_count,
                "evolution_details": evolution_details,
                "evolution_summary": summary
            }
        except Exception as e:
            logger.error(f"Failed to evolve emotions: {e}", exc_info=True)
            return {
                "patterns_analyzed": 0,
                "patterns_evolved": 0,
                "evolution_summary": f"Emotion evolution failed: {str(e)}",
            }

    # --- Derived anxiety (P2, 2026-03-30) ---

    def calculate_anxiety(self, session_id: str) -> float:
        """
        Derived anxiety score in ``[-1, 1]`` from dominance + arousal means.

        ``anxiety ≈ -dominance * 0.6 + arousal * 0.4`` (low control + high activation).
        """
        state = self.get_emotion_state(session_id)
        if state is None:
            return 0.0
        
        vec = state.emotion_vector
        dominance = np.mean(vec[EMOTION_SUBSPACE_DIMS["dominance"][0]:EMOTION_SUBSPACE_DIMS["dominance"][1]])
        arousal = np.mean(vec[EMOTION_SUBSPACE_DIMS["arousal"][0]:EMOTION_SUBSPACE_DIMS["arousal"][1]])
        
        return self._calculate_anxiety_from_subspaces(dominance, arousal)
    
    def _calculate_anxiety_from_subspaces(self, dominance: float, arousal: float) -> float:
        """Linear blend of low dominance and high arousal into a clipped anxiety scalar."""
        anxiety = -dominance * 0.6 + arousal * 0.4
        return float(np.clip(anxiety, -1.0, 1.0))
    
    def get_anxiety_description(self, anxiety: float) -> str:
        """Map ``calculate_anxiety`` output to a short first-person English line for prompts."""
        if anxiety > 0.6:
            return "A strong restless edge; thoughts keep racing and won't settle."
        elif anxiety > 0.4:
            return "Noticeably on edge—heart up, attention hard to hold in one place."
        elif anxiety > 0.2:
            return "A mild tightness—present, but still steerable."
        elif anxiety > -0.3:
            return "Mostly steady; no clear anxiety signal."
        else:
            return "Calm and grounded—breathing feels even and the mind is quiet."
    
    def get_extended_emotion_analysis(self, session_id: str) -> Dict:
        """Return subspace means plus derived ``anxiety`` and current dominant label."""
        state = self.get_emotion_state(session_id)
        if state is None:
            return {
                "pleasure": 0.0,
                "arousal": 0.0,
                "control": 0.0,
                "social": 0.0,
                "anxiety": 0.0,
                "dominant": "中性",
                "intensity": 0.0,
            }
        
        vec = state.emotion_vector
        pleasure = float(np.mean(vec[EMOTION_SUBSPACE_DIMS["pleasure"][0]:EMOTION_SUBSPACE_DIMS["pleasure"][1]]))
        arousal = float(np.mean(vec[EMOTION_SUBSPACE_DIMS["arousal"][0]:EMOTION_SUBSPACE_DIMS["arousal"][1]]))
        dominance = float(np.mean(vec[EMOTION_SUBSPACE_DIMS["dominance"][0]:EMOTION_SUBSPACE_DIMS["dominance"][1]]))
        novelty = float(np.mean(vec[EMOTION_SUBSPACE_DIMS["novelty"][0]:EMOTION_SUBSPACE_DIMS["novelty"][1]]))
        anxiety = self._calculate_anxiety_from_subspaces(dominance, arousal)
        
        return {
            "pleasure": pleasure,
            "arousal": arousal,
            "dominance": dominance,
            "novelty": novelty,
            "control": dominance,   # legacy alias for dominance
            "social": novelty,      # legacy alias for novelty
            "anxiety": anxiety,
            "dominant": state.dominant_emotion,
            "intensity": state.intensity,
        }

