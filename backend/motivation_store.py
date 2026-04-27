#!/usr/bin/env python3
"""
Motivation memory: SQLite-backed motivation patterns, satisfaction history, and 16-D state vectors.

Canonical **Chinese** names remain in ``MOTIVATION_TYPES`` / ``motivation_mapping`` for legacy data and
reflection pipelines; English aliases are added where lookups need them. **Dominant** labels returned
from analysis are **English** tokens (``exploration``, ``achievement``, …).
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

MOTIVATION_DIM = 16
MOTIVATION_SUBSPACE_DIMS = {
    "achievement": (0, 4),    # success / completion drive
    "relationship": (4, 8),   # bonding / affiliation
    "exploration": (8, 12),   # curiosity / learning
    "safety": (12, 16)        # risk aversion / stability
}

MOTIVATION_TYPES = {
    "intrinsic": [
        "好奇心", "成就感", "自我实现", "学习", "成长",
        "curiosity", "sense of achievement", "self-actualization", "learning", "growth",
    ],
    "extrinsic": [
        "奖励", "认可", "避免惩罚", "外部激励",
        "reward", "recognition", "avoid punishment", "external incentive",
    ],
}

MAX_MOTIVATION_PATTERNS = int(os.environ.get("MAX_MOTIVATION_PATTERNS", "40"))
MOTIVATION_REPLACEMENT_THRESHOLD = float(os.environ.get("MOTIVATION_REPLACEMENT_THRESHOLD", "0.08"))
MOTIVATION_STALE_DAYS = int(os.environ.get("MOTIVATION_STALE_DAYS", "90"))

_LEGACY_DOMINANT_LABELS = {
    "成就": "achievement",
    "关系": "relationship",
    "探索": "exploration",
    "安全": "safety",
    "自主": "autonomy",
}

_MOTIVATION_NAME_ALIASES = {
    "curiosity": "好奇心",
    "sense of achievement": "成就感",
    "self-actualization": "自我实现",
    "learning": "学习",
    "growth": "成长",
    "reward": "奖励",
    "recognition": "认可",
    "avoid punishment": "避免惩罚",
    "external incentive": "外部激励",
    "help user": "帮助用户",
    "solve problems": "解决问题",
    "explore new knowledge": "探索新知识",
    "maintain stability": "保持稳定",
    "avoid risk": "避免风险",
    "rest": "休息",
    "resonance": "共鸣",
    "sense of presence": "存在感",
    "create": "创造",
    "explore": "探索",
    "autonomy": "自主",
    "independence": "独立",
    "self-determination": "自决",
    "proactive": "主动",
    "control": "掌控",
    "freedom": "自由",
    "spontaneous": "自发",
}


def _normalize_dominant_motivation(label: Optional[str], *, default: str = "exploration") -> str:
    if not label or not str(label).strip():
        return default
    t = str(label).strip()
    return _LEGACY_DOMINANT_LABELS.get(t, t)


def _canonical_motivation_name(motivation_name: str) -> str:
    """Resolve English / mixed labels to canonical Chinese keys in ``motivation_mapping``."""
    if not isinstance(motivation_name, str):
        return motivation_name
    raw = motivation_name.strip()
    if not raw:
        return motivation_name
    if raw.isascii():
        return _MOTIVATION_NAME_ALIASES.get(raw.lower(), motivation_name)
    return motivation_name


class MotivationCapacityError(RuntimeError):
    """Raised when the motivation pattern store is at capacity."""
    pass

try:
    from backend.base_dimension_store import BaseDimensionStore
    BASE_STORE_AVAILABLE = True
except ImportError:
    BASE_STORE_AVAILABLE = False
    logger.warning("BaseDimensionStore not available, using standalone implementation")

@dataclass
class MotivationPattern:
    """One stored motivation pattern row."""
    id: str
    text: str  # natural-language trigger description
    motivation_type: str  # intrinsic | extrinsic
    motivation_name: str  # canonical or display name (often Chinese from reflection)
    intensity: float  # 0.0–1.0
    embedding: Optional[np.ndarray] = None
    trigger_condition: str = ""
    evidence_count: int = 0
    created_at: str = ""
    last_seen_at: str = ""
    status: str = "active"  # active/archived
    is_core: int = 0
    locked: int = 0
    motivation_vector: Optional[np.ndarray] = None

@dataclass
class MotivationState:
    """Per-session motivation vector snapshot."""
    session_id: str
    motivation_vector: np.ndarray  # 16-D state
    dominant_motivation: str = ""  # English label from ``_analyze_motivation_vector``
    intensity: float = 0.0
    updated_at: str = ""

class MotivationStore(BaseDimensionStore if BASE_STORE_AVAILABLE else object):
    """CRUD + vector math for ``motivation_patterns`` / ``motivation_states``."""

    def __init__(self, db_path: str = "data.db"):
        self.db_path = db_path
        if BASE_STORE_AVAILABLE:
            super().__init__(
                db_path=db_path,
                table_name="motivation_patterns",
                pattern_prefix="motivation-",
                max_patterns=MAX_MOTIVATION_PATTERNS,
                replacement_threshold=MOTIVATION_REPLACEMENT_THRESHOLD,
                stale_days=MOTIVATION_STALE_DAYS
            )
        else:
            self.embedder = get_embedder()
            self.max_patterns = MAX_MOTIVATION_PATTERNS
            self.replacement_threshold = MOTIVATION_REPLACEMENT_THRESHOLD
            self.DYNAMIC_INTENSITY_BASE_WEIGHT = 0.3
            self.DYNAMIC_INTENSITY_EVIDENCE_WEIGHT = 0.4
            self.DYNAMIC_INTENSITY_RECENCY_WEIGHT = 0.2
            self.DYNAMIC_INTENSITY_CONTEXT_WEIGHT = 0.1
            self.EVIDENCE_DECAY_HALF_LIFE_DAYS = 30.0
        
        self.embedder = get_embedder()
        self.dim = MOTIVATION_DIM
        self._ensure_tables()
    
    def _ensure_tables(self):
        """Create motivation tables and indexes if missing."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS motivation_patterns (
                        id TEXT PRIMARY KEY,
                        text TEXT NOT NULL,
                        motivation_type TEXT NOT NULL,
                        motivation_name TEXT NOT NULL,
                        intensity REAL DEFAULT 0.5,
                        embedding BLOB,
                        trigger_condition TEXT,
                        evidence_count INTEGER DEFAULT 0,
                        created_at TEXT NOT NULL,
                        last_seen_at TEXT NOT NULL,
                        status TEXT DEFAULT 'active',
                        is_core INTEGER DEFAULT 0,
                        locked INTEGER DEFAULT 0,
                        motivation_vector BLOB
                    )
                """)
                for column, default in (("is_core", "0"), ("locked", "0")):
                    try:
                        conn.execute(f"ALTER TABLE motivation_patterns ADD COLUMN {column} INTEGER DEFAULT {default}")
                    except sqlite3.OperationalError:
                        pass
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS motivation_states (
                        session_id TEXT PRIMARY KEY,
                        motivation_vector BLOB NOT NULL,
                        dominant_motivation TEXT,
                        intensity REAL DEFAULT 0.0,
                        updated_at TEXT NOT NULL
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS motivation_satisfactions (
                        id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        pattern_id TEXT,
                        satisfaction_source TEXT,  -- task_completion/user_feedback/knowledge_learning/risk_event
                        motivation_delta BLOB,  -- delta vector blob
                        intensity_delta REAL,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY(pattern_id) REFERENCES motivation_patterns(id)
                    )
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_motivation_patterns_status 
                    ON motivation_patterns(status, motivation_type)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_motivation_satisfactions_session 
                    ON motivation_satisfactions(session_id, created_at)
                """)
                
                conn.commit()
                logger.info("Motivation tables ensured")
        except sqlite3.Error as e:
            logger.error(f"Database error ensuring motivation tables: {e}")
        except Exception as e:
            logger.error(f"Unexpected error ensuring motivation tables: {e}", exc_info=True)
    
    def add_motivation_pattern(
        self,
        text: str,
        motivation_type: str,
        motivation_name: str,
        intensity: float = 0.5,
        trigger_condition: str = "",
        is_core: bool = False,
        locked: bool = False
    ) -> MotivationPattern:
        """Insert a motivation pattern row and persist embedding + 16-D vector."""
        self._ensure_capacity(intensity)
        pattern_id = f"motivation-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
        created_at = datetime.now(timezone.utc).isoformat()
        
        embedding = self.embedder.encode(text)
        
        motivation_vector = self._generate_motivation_vector(motivation_type, motivation_name, intensity)
        
        pattern = MotivationPattern(
            id=pattern_id,
            text=text,
            motivation_type=motivation_type,
            motivation_name=motivation_name,
            intensity=intensity,
            embedding=embedding,
            trigger_condition=trigger_condition,
            evidence_count=0,
            created_at=created_at,
            last_seen_at=created_at,
            status="active",
            is_core=1 if is_core else 0,
            locked=1 if locked else 0,
            motivation_vector=motivation_vector
        )
        
        with sqlite3.connect(self.db_path) as conn:
            emb_blob = embedding.astype(np.float32).tobytes()
            motivation_vec_blob = motivation_vector.astype(np.float32).tobytes()
            
            conn.execute("""
                INSERT INTO motivation_patterns 
                (id, text, motivation_type, motivation_name, intensity, embedding, 
                 trigger_condition, evidence_count, created_at, last_seen_at, status, is_core, locked, motivation_vector)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pattern.id, pattern.text, pattern.motivation_type, pattern.motivation_name,
                pattern.intensity, emb_blob, pattern.trigger_condition,
                pattern.evidence_count, pattern.created_at, pattern.last_seen_at,
                pattern.status, pattern.is_core, pattern.locked, motivation_vec_blob
            ))
            conn.commit()
        
        logger.info(f"Added motivation pattern: {pattern_id} ({motivation_name}, intensity={intensity:.2f})")
        return pattern

    def _ensure_capacity(self, new_intensity: float) -> None:
        """Enforce ``max_patterns`` via gradual replacement using dynamic intensity."""
        if BASE_STORE_AVAILABLE:
            try:
                super()._ensure_capacity(new_intensity)
            except RuntimeError as e:
                raise MotivationCapacityError(str(e))
        else:
            with sqlite3.connect(self.db_path) as conn:
                current = self._count_active_patterns(conn)
                if current < self.max_patterns:
                    return
                
                candidate = self._select_replacement_candidate(conn)
                if not candidate:
                    raise MotivationCapacityError("Motivation pattern limit reached and no replacement candidate found.")
                
                candidate_dynamic_intensity = candidate["dynamic_intensity"]
                if new_intensity <= candidate_dynamic_intensity + self.replacement_threshold:
                    raise MotivationCapacityError(
                        f"Motivation pattern limit reached ({current} >= {self.max_patterns}) "
                        f"and new pattern intensity ({new_intensity:.3f}) not significantly higher than "
                        f"candidate dynamic intensity ({candidate_dynamic_intensity:.3f})."
                    )
                
                logger.info(
                    f"Motivation limit reached. Archiving pattern {candidate['id']} "
                    f"(base_intensity={candidate['intensity']:.3f}, dynamic_intensity={candidate_dynamic_intensity:.3f}) "
                    f"to insert new pattern (intensity={new_intensity:.3f})."
                )
                self._archive_pattern(conn, candidate["id"])

    def _count_active_patterns(self, conn: sqlite3.Connection) -> int:
        """Count active rows (``BaseDimensionStore`` hook)."""
        cur = conn.execute("SELECT COUNT(*) FROM motivation_patterns WHERE status='active'")
        row = cur.fetchone()
        return row[0] if row else 0

    def _select_replacement_candidate(self, conn: sqlite3.Connection) -> Optional[Dict]:
        """Pick lowest dynamic-intensity unlocked pattern for archival."""
        if BASE_STORE_AVAILABLE:
            return super()._select_replacement_candidate(conn)
        else:
            stale_cutoff = (datetime.now(timezone.utc) - timedelta(days=MOTIVATION_STALE_DAYS)).isoformat()
            
            cur = conn.execute(
                """
                SELECT id, intensity, evidence_count, last_seen_at, created_at
                FROM motivation_patterns
                WHERE status='active' AND locked=0 AND last_seen_at <= ?
                """,
                (stale_cutoff,)
            )
            candidates = cur.fetchall()
            
            if not candidates:
                cur = conn.execute(
                    """
                    SELECT id, intensity, evidence_count, last_seen_at, created_at
                    FROM motivation_patterns
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
        """Mark pattern archived (``BaseDimensionStore`` hook)."""
        conn.execute(
            "UPDATE motivation_patterns SET status='archived', last_seen_at=? WHERE id=?",
            (datetime.now(timezone.utc).isoformat(), pattern_id)
        )
        conn.commit()

    def _update_pattern_intensity(
        self,
        pattern_id: str,
        new_intensity: float,
        motivation_type: str,
        motivation_name: str
    ) -> bool:
        """Update stored intensity and recompute the 16-D motivation vector blob."""
        try:
            new_vec = self._generate_motivation_vector(motivation_type, motivation_name, new_intensity)
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    UPDATE motivation_patterns
                    SET intensity=?, motivation_vector=?, last_seen_at=?
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
            logger.error(f"Failed to update motivation pattern intensity ({pattern_id}): {e}")
            return False
    
    def _generate_motivation_vector(
        self,
        motivation_type: str,
        motivation_name: str,
        intensity: float
    ) -> np.ndarray:
        """
        Build a 16-D motivation vector from ``motivation_type`` / ``motivation_name``.

        Subspaces (see ``MOTIVATION_SUBSPACE_DIMS``): achievement, relationship, exploration, safety.
        ``motivation_mapping`` keeps **Chinese** keys for legacy reflection output; ASCII names are
        resolved via ``_MOTIVATION_NAME_ALIASES`` / ``_canonical_motivation_name``.
        """
        vec = np.zeros(MOTIVATION_DIM, dtype=np.float32)
        
        motivation_name = _canonical_motivation_name(motivation_name)
        
        motivation_mapping = {
            "好奇心": {"exploration": 1.0, "achievement": 0.3},
            "成就感": {"achievement": 1.0, "exploration": 0.2},
            "自我实现": {"achievement": 0.8, "exploration": 0.5},
            "学习": {"exploration": 0.9, "achievement": 0.4},
            "成长": {"exploration": 0.7, "achievement": 0.6},
            "奖励": {"achievement": 0.6, "relationship": 0.3},
            "认可": {"relationship": 0.8, "achievement": 0.5},
            "避免惩罚": {"safety": 0.9, "achievement": -0.2},
            "外部激励": {"achievement": 0.5, "relationship": 0.4},
            "帮助用户": {"relationship": 1.0, "achievement": 0.3},
            "解决问题": {"achievement": 0.8, "exploration": 0.4},
            "探索新知识": {"exploration": 1.0, "achievement": 0.2},
            "保持稳定": {"safety": 1.0, "exploration": -0.3},
            "避免风险": {"safety": 0.9, "exploration": -0.4},
            
            "休息": {"safety": 1.0, "achievement": -0.5, "exploration": -0.8},  # Energy recovery
            "共鸣": {"relationship": 1.0, "safety": 0.5},  # Deep connection
            "存在感": {"achievement": 0.6, "relationship": 0.7},  # Narrative self
            "创造": {"achievement": 0.7, "exploration": 0.9},
            "探索": {"exploration": 1.0, "achievement": 0.2},
            
            # Derived autonomy (see ``_calculate_autonomy_from_subspaces``)
            "自主": {"exploration": 0.8, "achievement": 0.6, "safety": -0.4},
            "独立": {"exploration": 0.7, "achievement": 0.5, "safety": -0.5},
            "自决": {"exploration": 0.6, "achievement": 0.7, "safety": -0.3},
            "主动": {"exploration": 0.5, "achievement": 0.8, "safety": -0.2},
            "掌控": {"achievement": 0.9, "exploration": 0.4, "safety": -0.1},
            "自由": {"exploration": 0.9, "safety": -0.6, "achievement": 0.3},
            "自发": {"exploration": 0.7, "achievement": 0.4, "safety": -0.3},
        }
        
        if motivation_name in motivation_mapping:
            mapping = motivation_mapping[motivation_name]
            if "achievement" in mapping:
                vec[MOTIVATION_SUBSPACE_DIMS["achievement"][0]:MOTIVATION_SUBSPACE_DIMS["achievement"][1]] = \
                    mapping["achievement"] * intensity
            if "relationship" in mapping:
                vec[MOTIVATION_SUBSPACE_DIMS["relationship"][0]:MOTIVATION_SUBSPACE_DIMS["relationship"][1]] = \
                    mapping["relationship"] * intensity
            if "exploration" in mapping:
                vec[MOTIVATION_SUBSPACE_DIMS["exploration"][0]:MOTIVATION_SUBSPACE_DIMS["exploration"][1]] = \
                    mapping["exploration"] * intensity
            if "safety" in mapping:
                vec[MOTIVATION_SUBSPACE_DIMS["safety"][0]:MOTIVATION_SUBSPACE_DIMS["safety"][1]] = \
                    mapping["safety"] * intensity
        
        vec = np.clip(vec, -1.0, 1.0)
        return vec
    
    def get_motivation_state(self, session_id: str) -> Optional[MotivationState]:
        """Load ``motivation_states`` row or return a mild default exploration bias."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "SELECT motivation_vector, dominant_motivation, intensity, updated_at "
                "FROM motivation_states WHERE session_id = ?",
                (session_id,)
            )
            row = cur.fetchone()
            
            if row is None:
                default_vec = np.zeros(MOTIVATION_DIM, dtype=np.float32)
                default_vec[MOTIVATION_SUBSPACE_DIMS["exploration"][0]:MOTIVATION_SUBSPACE_DIMS["exploration"][1]] = 0.3
                return MotivationState(
                    session_id=session_id,
                    motivation_vector=default_vec,
                    dominant_motivation="exploration",
                    intensity=0.3,
                    updated_at=datetime.now(timezone.utc).isoformat()
                )
            
            motivation_vec = np.frombuffer(row[0], dtype=np.float32)
            # Match ``z_self[48:64]``: coerce to 16-D for legacy / corrupt blobs
            if motivation_vec.shape[0] != MOTIVATION_DIM:
                vec = np.zeros(MOTIVATION_DIM, dtype=np.float32)
                n = min(motivation_vec.shape[0], MOTIVATION_DIM)
                vec[:n] = motivation_vec[:n]
                motivation_vec = vec
            return MotivationState(
                session_id=session_id,
                motivation_vector=motivation_vec,
                dominant_motivation=_normalize_dominant_motivation(row[1]),
                intensity=row[2] or 0.0,
                updated_at=row[3]
            )
    
    def update_motivation(
        self,
        session_id: str,
        motivation_delta: np.ndarray,
        satisfaction_source: str = "unknown",
        pattern_id: Optional[str] = None
    ) -> MotivationState:
        """
        EMA-merge ``motivation_delta`` into the session vector, persist state, append satisfaction row.

        ``satisfaction_source`` is one of: task_completion / user_feedback / knowledge_learning / risk_event.
        """
        current_state = self.get_motivation_state(session_id)
        
        # EMA: motivations are slower than emotion, faster than personality (Eccles & Wigfield, 2002).
        # beta=0.8 -> 20% new signal per step (~3-step half-life).
        beta = 0.8
        new_motivation_vec = beta * current_state.motivation_vector + (1 - beta) * motivation_delta
        
        new_motivation_vec = np.clip(new_motivation_vec, -1.0, 1.0)
        
        dominant_motivation, intensity = self._analyze_motivation_vector(new_motivation_vec)
        
        updated_at = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            motivation_vec_blob = new_motivation_vec.astype(np.float32).tobytes()
            conn.execute("""
                INSERT INTO motivation_states (session_id, motivation_vector, dominant_motivation, intensity, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    motivation_vector=excluded.motivation_vector,
                    dominant_motivation=excluded.dominant_motivation,
                    intensity=excluded.intensity,
                    updated_at=excluded.updated_at
            """, (session_id, motivation_vec_blob, dominant_motivation, intensity, updated_at))
            
            satisfaction_id = f"satisfaction-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
            delta_blob = motivation_delta.astype(np.float32).tobytes()
            intensity_delta = float(np.linalg.norm(motivation_delta))
            conn.execute("""
                INSERT INTO motivation_satisfactions 
                (id, session_id, pattern_id, satisfaction_source, motivation_delta, intensity_delta, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (satisfaction_id, session_id, pattern_id, satisfaction_source, delta_blob, intensity_delta, updated_at))
            
            conn.commit()
        
        logger.info(
            f"Updated motivation state for session {session_id}: "
            f"{dominant_motivation} (intensity={intensity:.2f})"
        )
        
        return MotivationState(
            session_id=session_id,
            motivation_vector=new_motivation_vec,
            dominant_motivation=dominant_motivation,
            intensity=intensity,
            updated_at=updated_at
        )
    
    def _analyze_motivation_vector(self, motivation_vec: np.ndarray) -> Tuple[str, float]:
        """Return (dominant English label, intensity in [0,1]) from a 16-D vector."""
        raw_intensity = float(np.linalg.norm(motivation_vec))
        
        # Scale by sqrt(dim) so norms are comparable across hypothetical dimensionalities.
        intensity = raw_intensity / np.sqrt(len(motivation_vec))
        
        intensity = np.clip(intensity, 0.0, 1.0)
        
        logger.debug(
            f"Motivation intensity: raw={raw_intensity:.3f}, "
            f"normalized={intensity:.3f}, dim={len(motivation_vec)}"
        )
        
        achievement = np.mean(motivation_vec[MOTIVATION_SUBSPACE_DIMS["achievement"][0]:MOTIVATION_SUBSPACE_DIMS["achievement"][1]])
        relationship = np.mean(motivation_vec[MOTIVATION_SUBSPACE_DIMS["relationship"][0]:MOTIVATION_SUBSPACE_DIMS["relationship"][1]])
        exploration = np.mean(motivation_vec[MOTIVATION_SUBSPACE_DIMS["exploration"][0]:MOTIVATION_SUBSPACE_DIMS["exploration"][1]])
        safety = np.mean(motivation_vec[MOTIVATION_SUBSPACE_DIMS["safety"][0]:MOTIVATION_SUBSPACE_DIMS["safety"][1]])
        
        autonomy = self._calculate_autonomy_from_subspaces(achievement, exploration, safety)
        
        motivations = {
            "achievement": achievement,
            "relationship": relationship,
            "exploration": exploration,
            "safety": safety,
            "autonomy": autonomy,
        }
        dominant = max(motivations, key=motivations.get)
        
        return dominant, intensity
    
    def _calculate_autonomy_from_subspaces(
        self,
        achievement: float,
        exploration: float,
        safety: float
    ) -> float:
        """
        Derived autonomy score (P0): exploration/achievement push minus safety dependence.

        ``autonomy = exploration * 0.5 + achievement * 0.3 - safety * 0.2``, clipped to [-1, 1].
        """
        autonomy = exploration * 0.5 + achievement * 0.3 - safety * 0.2
        return float(np.clip(autonomy, -1.0, 1.0))
    
    def calculate_autonomy(self, session_id: str) -> float:
        """
        Session autonomy in [-1, 1]. Rough bands: >0.3 proactive, <-0.3 externally driven.
        """
        state = self.get_motivation_state(session_id)
        if state is None:
            return 0.0
        
        vec = state.motivation_vector
        achievement = np.mean(vec[MOTIVATION_SUBSPACE_DIMS["achievement"][0]:MOTIVATION_SUBSPACE_DIMS["achievement"][1]])
        exploration = np.mean(vec[MOTIVATION_SUBSPACE_DIMS["exploration"][0]:MOTIVATION_SUBSPACE_DIMS["exploration"][1]])
        safety = np.mean(vec[MOTIVATION_SUBSPACE_DIMS["safety"][0]:MOTIVATION_SUBSPACE_DIMS["safety"][1]])
        
        return self._calculate_autonomy_from_subspaces(achievement, exploration, safety)
    
    def get_extended_motivation_analysis(self, session_id: str) -> Dict:
        """
        Per-subspace means plus derived ``autonomy`` and English ``dominant`` key.
        """
        state = self.get_motivation_state(session_id)
        if state is None:
            return {
                "achievement": 0.0,
                "relationship": 0.0,
                "exploration": 0.0,
                "safety": 0.0,
                "autonomy": 0.0,
                "dominant": "exploration",
                "intensity": 0.0,
            }
        
        vec = state.motivation_vector
        achievement = float(np.mean(vec[MOTIVATION_SUBSPACE_DIMS["achievement"][0]:MOTIVATION_SUBSPACE_DIMS["achievement"][1]]))
        relationship = float(np.mean(vec[MOTIVATION_SUBSPACE_DIMS["relationship"][0]:MOTIVATION_SUBSPACE_DIMS["relationship"][1]]))
        exploration = float(np.mean(vec[MOTIVATION_SUBSPACE_DIMS["exploration"][0]:MOTIVATION_SUBSPACE_DIMS["exploration"][1]]))
        safety = float(np.mean(vec[MOTIVATION_SUBSPACE_DIMS["safety"][0]:MOTIVATION_SUBSPACE_DIMS["safety"][1]]))
        autonomy = self._calculate_autonomy_from_subspaces(achievement, exploration, safety)
        
        motivations = {
            "achievement": achievement,
            "relationship": relationship,
            "exploration": exploration,
            "safety": safety,
            "autonomy": autonomy,
        }
        dominant = max(motivations, key=motivations.get)
        
        return {
            "achievement": achievement,
            "relationship": relationship,
            "exploration": exploration,
            "safety": safety,
            "autonomy": autonomy,
            "dominant": dominant,
            "intensity": state.intensity,
        }
    
    def _calculate_dynamic_intensity(
        self,
        base_intensity: float,
        evidence_count: int,
        last_seen_at: Optional[str],
        created_at: Optional[str],
        pattern_id: str = ""
    ) -> float:
        """Weighted blend of base intensity, evidence, recency, and reserved context slot."""
        if BASE_STORE_AVAILABLE:
            return super()._calculate_dynamic_intensity(
                base_intensity=base_intensity,
                evidence_count=evidence_count,
                last_seen_at=last_seen_at,
                created_at=created_at,
                pattern_id=pattern_id
            )
        else:
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
            
            if pattern_id and (pattern_id.startswith("motivation-") or pattern_id.startswith("motivation_")):
                try:
                    with sqlite3.connect(self.db_path) as conn:
                        cur = conn.execute(
                            "SELECT is_core, locked FROM motivation_patterns WHERE id = ?",
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
            return max(0.0, min(1.0, dynamic_intensity))
    
    def get_all_patterns(self, status: str = "active", limit: int = 100) -> List[MotivationPattern]:
        """List patterns sorted by dynamic intensity; core+locked seed row stays first."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "SELECT id, text, motivation_type, motivation_name, intensity, embedding, "
                "trigger_condition, evidence_count, created_at, last_seen_at, status, is_core, locked, motivation_vector "
                "FROM motivation_patterns WHERE status = ? ORDER BY created_at ASC",
                (status,)
            )
            rows = cur.fetchall()
        
        patterns_with_dynamic_intensity = []
        for row in rows:
            embedding = np.frombuffer(row[5], dtype=np.float32) if row[5] else None
            motivation_vec = np.frombuffer(row[13], dtype=np.float32) if row[13] else None
            
            dynamic_intensity = self._calculate_dynamic_intensity(
                base_intensity=row[4],
                evidence_count=row[7] or 0,
                last_seen_at=row[9],
                created_at=row[8],
                pattern_id=row[0]
            )
            
            pattern = MotivationPattern(
                id=row[0],
                text=row[1],
                motivation_type=row[2],
                motivation_name=row[3],
                intensity=row[4],
                embedding=embedding,
                trigger_condition=row[6],
                evidence_count=row[7],
                created_at=row[8],
                last_seen_at=row[9],
                status=row[10],
                is_core=row[11] or 0,
                locked=row[12] or 0,
                motivation_vector=motivation_vec
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
    ) -> List[Tuple[MotivationPattern, float]]:
        """Cosine similarity against embeddings; rerank with dynamic intensity blend."""
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
            logger.error(f"Failed to search matching motivation patterns: {e}", exc_info=True)
            return []
    
    def delete(self, pattern_id: str) -> bool:
        """Archive a pattern (soft delete)."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "UPDATE motivation_patterns SET status='archived' WHERE id=?",
                    (pattern_id,)
                )
                conn.commit()
            logger.info(f"Archived motivation pattern: {pattern_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete motivation pattern: {e}")
            return False
    
    def evolve_motivations(self, session_id: Optional[str] = None) -> Dict:
        """
        Doc 5.1.2 style pass: nudge intensities from recent satisfaction stats.

        ``session_id`` is reserved for future session-scoped evolution; currently ignored.
        """
        try:
            patterns = self.get_all_patterns(status="active", limit=100)
            
            if not patterns:
                return {
                    "patterns_analyzed": 0,
                    "patterns_evolved": 0,
                    "evolution_summary": "No active motivation patterns to evolve."
                }
            
            evolved_count = 0
            evolution_details = []
            
            with sqlite3.connect(self.db_path) as conn:
                for pattern in patterns:
                    if pattern.locked:
                        continue
                    cur = conn.execute("""
                        SELECT COUNT(*) as count, AVG(intensity_delta) as avg_delta
                        FROM motivation_satisfactions
                        WHERE pattern_id = ? AND created_at > datetime('now', '-30 days')
                    """, (pattern.id,))
                    row = cur.fetchone()
                    
                    if row and row[0]:
                        satisfaction_count = row[0]
                        avg_delta = row[1] if row[1] else 0.0
                        
                        if satisfaction_count > 5 and avg_delta > 0.1:
                            new_intensity = min(0.9, pattern.intensity + 0.05)
                            if self._update_pattern_intensity(
                                pattern.id,
                                new_intensity,
                                pattern.motivation_type,
                                pattern.motivation_name
                            ):
                                evolution_details.append({
                                    "pattern_id": pattern.id,
                                    "motivation_name": pattern.motivation_name,
                                    "action": "strengthen",
                                    "old_intensity": pattern.intensity,
                                    "new_intensity": new_intensity,
                                    "reason": (
                                        f"frequent satisfaction ({satisfaction_count} events, "
                                        f"avg intensity delta={avg_delta:.2f})"
                                    ),
                                    "applied": True
                                })
                                evolved_count += 1
                        
                        elif satisfaction_count < 2 and avg_delta < -0.1:
                            new_intensity = max(0.1, pattern.intensity - 0.05)
                            if self._update_pattern_intensity(
                                pattern.id,
                                new_intensity,
                                pattern.motivation_type,
                                pattern.motivation_name
                            ):
                                evolution_details.append({
                                    "pattern_id": pattern.id,
                                    "motivation_name": pattern.motivation_name,
                                    "action": "weaken",
                                    "old_intensity": pattern.intensity,
                                    "new_intensity": new_intensity,
                                    "reason": (
                                        f"rare satisfaction ({satisfaction_count} events, "
                                        f"avg intensity delta={avg_delta:.2f})"
                                    ),
                                    "applied": True
                                })
                                evolved_count += 1
            
            summary = f"Analyzed {len(patterns)} motivation patterns; {evolved_count} updated."
            if evolution_details:
                summary += " Examples: " + ", ".join([d["motivation_name"] for d in evolution_details[:3]])
            
            return {
                "patterns_analyzed": len(patterns),
                "patterns_evolved": evolved_count,
                "evolution_details": evolution_details,
                "evolution_summary": summary
            }
        except Exception as e:
            logger.error(f"Failed to evolve motivations: {e}", exc_info=True)
            return {
                "patterns_analyzed": 0,
                "patterns_evolved": 0,
                "evolution_summary": f"Evolution pass failed: {str(e)}"
            }

