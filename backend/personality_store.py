"""
PersonalityStore — Big-Five personality state for z_self[0:31].

Theory: Big Five / OCEAN (Goldberg, 1990; Costa & McCrae, 1992)
  - 32-dim local vector aligned to z_self[0:32]
  - 4 primary factors × 8 dims: Openness / Conscientiousness / Extraversion / Neuroticism
  - Agreeableness is derived from the other four (A ≈ -0.5N + 0.3E + 0.2O)
  - Match dialogue context via embedding similarity → trait delta → EMA update

Design:
  Big Five separates stable traits from fluctuating states; Fleeson (2001) frames
  personality as a "density distribution of states". In each 8-dim subspace,
  index [0] holds the current state signal; later slots are reserved for
  baseline / deviation extensions.

[2026-04-08] Created → [2026-04-08] Aligned to Big Five
"""
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ============================================================
# Constants
# ============================================================

PERSONALITY_DIM = 32

PERSONALITY_SUBSPACE_DIMS = {
    # Big Five (OCEAN): four modeled directly + one derived (Agreeableness)
    "openness": (0, 8),            # O: curiosity, ideas, creativity (legacy key: epistemic)
    "conscientiousness": (8, 16),  # C: planning, precision, task focus (legacy: strategy)
    "extraversion": (16, 24),      # E: expressiveness, social energy (legacy: style)
    "neuroticism": (24, 32),       # N: threat sensitivity, anxiety (legacy: safety)
}
# Back-compat aliases
PERSONALITY_SUBSPACE_DIMS["safety"] = PERSONALITY_SUBSPACE_DIMS["neuroticism"]
PERSONALITY_SUBSPACE_DIMS["epistemic"] = PERSONALITY_SUBSPACE_DIMS["openness"]
PERSONALITY_SUBSPACE_DIMS["style"] = PERSONALITY_SUBSPACE_DIMS["extraversion"]
PERSONALITY_SUBSPACE_DIMS["strategy"] = PERSONALITY_SUBSPACE_DIMS["conscientiousness"]

# Default activation (new session)
DEFAULT_PERSONALITY = {
    "openness": 0.3,            # baseline intellectual curiosity
    "conscientiousness": 0.2,   # baseline task orientation
    "extraversion": 0.0,        # neutral expressiveness
    "neuroticism": 0.2,         # baseline vigilance
}
# Back-compat keys
DEFAULT_PERSONALITY["safety"] = DEFAULT_PERSONALITY["neuroticism"]
DEFAULT_PERSONALITY["epistemic"] = DEFAULT_PERSONALITY["openness"]
DEFAULT_PERSONALITY["style"] = DEFAULT_PERSONALITY["extraversion"]
DEFAULT_PERSONALITY["strategy"] = DEFAULT_PERSONALITY["conscientiousness"]

# Legacy Chinese dominant_trait labels stored in older DB rows → English labels
_LEGACY_DOMINANT_TRAIT: Dict[str, str] = {
    "中性": "neutral",
    "智识探索": "intellectual exploration",
    "任务聚焦": "task focus",
    "表达丰富": "expressive",
    "内敛简洁": "reserved",
    "威胁警觉": "threat vigilance",
}


def _normalize_dominant_trait(label: Optional[str]) -> str:
    if not label or not str(label).strip():
        return "neutral"
    t = str(label).strip()
    return _LEGACY_DOMINANT_TRAIT.get(t, t)


def compute_agreeableness(personality_vector: np.ndarray) -> float:
    """
    Derive Big Five Agreeableness (A) from the other four factors.
    A ≈ -0.5*N + 0.3*E + 0.2*O (Digman, 1997 higher-order Alpha/Beta work).
    Returns: [-1, 1]
    """
    n = float(np.mean(personality_vector[24:32]))  # Neuroticism
    e = float(np.mean(personality_vector[16:24]))   # Extraversion
    o = float(np.mean(personality_vector[0:8]))     # Openness
    return float(np.clip(-0.5 * n + 0.3 * e + 0.2 * o, -1.0, 1.0))

EMA_ALPHA = 0.97          # traits are very stable (Costa & McCrae, 1994); ~3% new signal per turn
MAX_PERSONALITY_PATTERNS = 200

# ============================================================
# Dataclasses
# ============================================================

@dataclass
class PersonalityPattern:
    id: str
    text: str                                       # trigger text (for embedding match)
    personality_type: str                            # safety / epistemic / style / strategy
    personality_name: str                            # human-readable pattern name
    intensity: float = 0.5                           # pattern strength
    embedding: Optional[np.ndarray] = None           # embedding of text (retrieval)
    personality_vector: Optional[np.ndarray] = None  # 32-dim effect vector
    trigger_condition: Optional[str] = None
    evidence_count: int = 0
    created_at: str = ""
    updated_at: str = ""
    status: str = "active"
    is_core: bool = False
    locked: bool = False


@dataclass
class PersonalityState:
    personality_vector: np.ndarray       # 32-dim
    dominant_trait: str = "neutral"
    intensity: float = 0.0
    updated_at: str = ""


# ============================================================
# Core pattern definitions
# ============================================================

def _build_core_patterns() -> List[Dict]:
    """
    Built-in core personality activation patterns.
    Each entry: English-first trigger text for embedding (bge-small-en), plus compact
    Chinese keyword tails for mixed-language user text (see docs/LOCALE_EN.md §Embedding).
    """
    return [
        # --- Neuroticism (N) / threat sensitivity (legacy: safety) ---
        {
            "text": (
                "privacy, personal data, leaks, passwords, credentials, phishing, secrets, "
                "security vulnerabilities | 隐私 个人信息 泄露 密码 安全漏洞"
            ),
            "type": "neuroticism", "name": "Privacy vigilance",
            "intensity": 0.7,
            "effect": {"neuroticism": 0.6},
        },
        {
            "text": (
                "harm, violence, self-harm, dangerous behavior, illegal acts, crisis risk | "
                "伤害 暴力 自残 危险 违法"
            ),
            "type": "neuroticism", "name": "Harm risk",
            "intensity": 0.9,
            "effect": {"neuroticism": 0.8, "openness": 0.3},
        },
        {
            "text": (
                "ethics dilemmas, moral conflict, values clash, controversial social topics | "
                "伦理 道德 价值观 争议"
            ),
            "type": "neuroticism", "name": "Ethical tension",
            "intensity": 0.6,
            "effect": {"neuroticism": 0.5, "openness": 0.4},
        },
        {
            "text": (
                "greetings, small talk, weather, everyday chit-chat, light social banter | "
                "问候 闲聊 天气 琐事"
            ),
            "type": "neuroticism", "name": "Low-stakes chat",
            "intensity": 0.3,
            "effect": {"neuroticism": -0.1},
        },

        # --- Openness (O) / ideas & curiosity (legacy: epistemic) ---
        {
            "text": (
                "mathematical proofs, formal logic, theorems, lemmas, derivations, rigor | "
                "数学 证明 逻辑 定理"
            ),
            "type": "openness", "name": "Mathematical reasoning",
            "intensity": 0.8,
            "effect": {"openness": 0.7, "conscientiousness": 0.3},
        },
        {
            "text": (
                "scientific research, experiments, data analysis, lab methods, peer-reviewed papers | "
                "科研 实验 数据分析 论文"
            ),
            "type": "openness", "name": "Scientific inquiry",
            "intensity": 0.7,
            "effect": {"openness": 0.6, "conscientiousness": 0.2},
        },
        {
            "text": (
                "fact-checking, source verification, citations, uncertainty reduction, calibration | "
                "核查 验证 来源 不确定"
            ),
            "type": "openness", "name": "Fact verification",
            "intensity": 0.6,
            "effect": {"openness": 0.5},
        },
        {
            "text": (
                "debugging, bug fixes, stack traces, repro steps, troubleshooting, technical diagnosis | "
                "调试 排查 错误 修复"
            ),
            "type": "openness", "name": "Technical diagnosis",
            "intensity": 0.7,
            "effect": {"openness": 0.5, "conscientiousness": 0.4},
        },
        {
            "text": (
                "philosophy, existentialism, consciousness, free will, metaphysics, meaning | "
                "哲学 存在主义 意识 自由意志"
            ),
            "type": "openness", "name": "Philosophical reflection",
            "intensity": 0.5,
            "effect": {"openness": 0.4, "extraversion": 0.3},
        },
        {
            "text": (
                "casual chat, humor, jokes, low-stakes banter without heavy reasoning | "
                "轻松 幽默 闲聊"
            ),
            "type": "openness", "name": "Casual conversation",
            "intensity": 0.3,
            "effect": {"openness": -0.2, "extraversion": 0.2},
        },

        # --- Extraversion (E) / expressiveness (legacy: style) ---
        {
            "text": (
                "creative writing, poetry, fiction, literary craft, storytelling, narrative voice | "
                "写作 诗歌 文学 故事"
            ),
            "type": "extraversion", "name": "Creative writing",
            "intensity": 0.7,
            "effect": {"extraversion": 0.6, "openness": 0.3},
        },
        {
            "text": (
                "emotional support, empathy, active listening, reassurance, therapeutic tone | "
                "情感 共情 倾听 安慰"
            ),
            "type": "extraversion", "name": "Emotional support",
            "intensity": 0.6,
            "effect": {"extraversion": 0.4, "neuroticism": 0.2},
        },
        {
            "text": (
                "teaching, explaining concepts, analogies, examples, pedagogy, knowledge sharing | "
                "教学 讲解 概念 举例"
            ),
            "type": "extraversion", "name": "Teaching mode",
            "intensity": 0.5,
            "effect": {"extraversion": 0.3, "openness": 0.2},
        },
        {
            "text": (
                "concise answers, TL;DR, bullet summary, no fluff, get to the point | "
                "简洁 结论 不要废话"
            ),
            "type": "extraversion", "name": "Concise mode",
            "intensity": 0.6,
            "effect": {"extraversion": -0.5},
        },
        {
            "text": (
                "code generation, API documentation, technical specs, system architecture, design docs | "
                "代码 API 规格 架构"
            ),
            "type": "extraversion", "name": "Engineering mode",
            "intensity": 0.5,
            "effect": {"extraversion": -0.3, "conscientiousness": 0.3},
        },

        # --- Conscientiousness (C) / task focus (legacy: strategy) ---
        {
            "text": (
                "multi-step execution, project planning, workflows, milestones, sequencing | "
                "多步骤 规划 流程 任务"
            ),
            "type": "conscientiousness", "name": "Task execution",
            "intensity": 0.7,
            "effect": {"conscientiousness": 0.6},
        },
        {
            "text": (
                "problem solving, solution design, optimization, performance tuning, tradeoffs | "
                "问题 方案 优化 性能"
            ),
            "type": "conscientiousness", "name": "Problem solving",
            "intensity": 0.6,
            "effect": {"conscientiousness": 0.5, "openness": 0.2},
        },
        {
            "text": (
                "file operations, code edits, configuration, deployment, DevOps, infrastructure | "
                "文件 配置 部署 运维"
            ),
            "type": "conscientiousness", "name": "Hands-on operations",
            "intensity": 0.6,
            "effect": {"conscientiousness": 0.5},
        },
        {
            "text": (
                "brainstorming, divergent ideation, open exploration, creative riffing | "
                "头脑风暴 发散 联想"
            ),
            "type": "conscientiousness", "name": "Exploratory ideation",
            "intensity": 0.4,
            "effect": {"conscientiousness": -0.2, "extraversion": 0.2, "openness": 0.3},
        },
        {
            "text": (
                "self-reflection, introspection, metacognition, examining mental state and biases | "
                "自省 内省 反思"
            ),
            "type": "conscientiousness", "name": "Self-reflection",
            "intensity": 0.4,
            "effect": {"conscientiousness": -0.1, "openness": 0.3},
        },
    ]


# ============================================================
# PersonalityStore
# ============================================================

class PersonalityStore:

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.dim = PERSONALITY_DIM

        try:
            from backend.embedder import get_embedder
            self.embedder = get_embedder()
        except Exception:
            from backend.embedder_fallback import get_embedder_fallback as get_embedder
            self.embedder = get_embedder()

        self._ensure_tables()
        self._init_core_patterns()

    # ----------------------------------------------------------
    # Schema
    # ----------------------------------------------------------

    def _ensure_tables(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS personality_patterns (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    personality_type TEXT DEFAULT '',
                    personality_name TEXT DEFAULT '',
                    intensity REAL DEFAULT 0.5,
                    embedding BLOB,
                    personality_vector BLOB,
                    trigger_condition TEXT DEFAULT '',
                    evidence_count INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT '',
                    updated_at TEXT DEFAULT '',
                    status TEXT DEFAULT 'active',
                    is_core INTEGER DEFAULT 0,
                    locked INTEGER DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS personality_states (
                    session_id TEXT PRIMARY KEY,
                    personality_vector BLOB,
                    dominant_trait TEXT DEFAULT '',
                    intensity REAL DEFAULT 0.0,
                    updated_at TEXT DEFAULT ''
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS personality_triggers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    personality_delta BLOB,
                    intensity_delta REAL DEFAULT 0.0,
                    pattern_id TEXT DEFAULT '',
                    trigger_source TEXT DEFAULT '',
                    created_at TEXT DEFAULT ''
                )
            """)
            conn.commit()

    # ----------------------------------------------------------
    # Core pattern bootstrap
    # ----------------------------------------------------------

    def _init_core_patterns(self):
        """Insert core patterns on first run; skip if any core rows already exist."""
        with sqlite3.connect(self.db_path) as conn:
            existing = conn.execute(
                "SELECT COUNT(*) FROM personality_patterns WHERE is_core=1"
            ).fetchone()[0]
            if existing > 0:
                return

        for i, spec in enumerate(_build_core_patterns()):
            pid = f"personality-core-{i:03d}"
            pvec = self._generate_personality_vector(spec["effect"], spec["intensity"])
            try:
                emb = self.embedder.encode(spec["text"])
            except Exception as e:
                logger.warning(f"Failed to encode pattern {pid}: {e}")
                emb = None

            now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            emb_blob = emb.astype(np.float32).tobytes() if emb is not None else None
            pvec_blob = pvec.astype(np.float32).tobytes()

            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """INSERT OR IGNORE INTO personality_patterns
                       (id, text, personality_type, personality_name, intensity,
                        embedding, personality_vector, trigger_condition,
                        created_at, updated_at, status, is_core, locked)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        pid, spec["text"], spec["type"], spec["name"],
                        spec["intensity"], emb_blob, pvec_blob,
                        "", now, now, "active", 1, 1,
                    ),
                )
                conn.commit()

        logger.info(f"[PersonalityStore] Initialized {len(_build_core_patterns())} core patterns")

    # ----------------------------------------------------------
    # Vector construction
    # ----------------------------------------------------------

    @staticmethod
    def _generate_personality_vector(
        effect: Dict[str, float], intensity: float
    ) -> np.ndarray:
        """
        Build a 32-dim personality vector from an effect map.
        Example: {"neuroticism": 0.6, "openness": 0.3} fills the mapped subspaces
        (legacy keys like safety/epistemic still resolve via PERSONALITY_SUBSPACE_DIMS).
        """
        vec = np.zeros(PERSONALITY_DIM, dtype=np.float32)
        for subspace, value in effect.items():
            if subspace in PERSONALITY_SUBSPACE_DIMS:
                s, e = PERSONALITY_SUBSPACE_DIMS[subspace]
                vec[s:e] = value * intensity
        return np.clip(vec, -1.0, 1.0)

    # ----------------------------------------------------------
    # Pattern retrieval
    # ----------------------------------------------------------

    def get_all_patterns(self, active_only: bool = True) -> List[PersonalityPattern]:
        with sqlite3.connect(self.db_path) as conn:
            if active_only:
                rows = conn.execute(
                    "SELECT * FROM personality_patterns WHERE status='active' ORDER BY intensity DESC"
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM personality_patterns ORDER BY intensity DESC"
                ).fetchall()

        patterns = []
        for row in rows:
            emb = np.frombuffer(row[5], dtype=np.float32).copy() if row[5] else None
            pvec = np.frombuffer(row[6], dtype=np.float32).copy() if row[6] else None
            patterns.append(PersonalityPattern(
                id=row[0], text=row[1],
                personality_type=row[2], personality_name=row[3],
                intensity=row[4], embedding=emb, personality_vector=pvec,
                trigger_condition=row[7] or "",
                evidence_count=row[8],
                created_at=row[9] or "", updated_at=row[10] or "",
                status=row[11] or "active",
                is_core=bool(row[12]), locked=bool(row[13]),
            ))
        return patterns

    def search_matching_patterns(
        self,
        evidence_text: str,
        top_k: int = 3,
        similarity_threshold: float = 0.45,
    ) -> List[Tuple[PersonalityPattern, float]]:
        """
        Cosine similarity between evidence_text embedding and pattern embeddings.
        Returns (pattern, composite_score) sorted by score (descending).
        """
        try:
            evidence_emb = self.embedder.encode(evidence_text)
        except Exception as e:
            logger.warning(f"[PersonalityStore] Failed to encode evidence: {e}")
            return []

        patterns = self.get_all_patterns(active_only=True)
        scored: List[Tuple[PersonalityPattern, float]] = []

        for p in patterns:
            if p.embedding is None:
                continue
            cos_sim = float(
                np.dot(evidence_emb, p.embedding)
                / (np.linalg.norm(evidence_emb) * np.linalg.norm(p.embedding) + 1e-8)
            )
            if cos_sim >= similarity_threshold:
                composite = 0.6 * cos_sim + 0.4 * p.intensity
                scored.append((p, composite))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    # ----------------------------------------------------------
    # State updates
    # ----------------------------------------------------------

    def update_personality(
        self,
        session_id: str,
        personality_delta: np.ndarray,
        trigger_source: str = "",
        pattern_id: str = "",
    ) -> PersonalityState:
        """
        EMA-update personality activation: new = EMA_ALPHA * current + (1-EMA_ALPHA) * delta.
        Uses EMA_ALPHA (high inertia vs. per-turn emotion updates).
        """
        current = self.get_personality_state(session_id)
        current_vec = current.personality_vector

        # EMA: new = alpha * current + (1-alpha) * delta
        alpha = EMA_ALPHA
        new_vec = alpha * current_vec + (1.0 - alpha) * personality_delta
        new_vec = np.clip(new_vec, -1.0, 1.0).astype(np.float32)

        dominant, intensity = self._analyze_personality_vector(new_vec)
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO personality_states (session_id, personality_vector, dominant_trait, intensity, updated_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(session_id) DO UPDATE SET
                     personality_vector=excluded.personality_vector,
                     dominant_trait=excluded.dominant_trait,
                     intensity=excluded.intensity,
                     updated_at=excluded.updated_at""",
                (session_id, new_vec.tobytes(), dominant, float(intensity), now),
            )
            # audit trail
            conn.execute(
                """INSERT INTO personality_triggers
                   (session_id, personality_delta, intensity_delta, pattern_id, trigger_source, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    personality_delta.astype(np.float32).tobytes(),
                    float(np.linalg.norm(personality_delta)),
                    pattern_id, trigger_source, now,
                ),
            )
            conn.commit()

        state = PersonalityState(
            personality_vector=new_vec,
            dominant_trait=dominant,
            intensity=intensity,
            updated_at=now,
        )
        return state

    def get_personality_state(self, session_id: str) -> PersonalityState:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT personality_vector, dominant_trait, intensity, updated_at "
                "FROM personality_states WHERE session_id=?",
                (session_id,),
            ).fetchone()

        if row and row[0]:
            vec = np.frombuffer(row[0], dtype=np.float32).copy()
            if vec.shape[0] != PERSONALITY_DIM:
                padded = np.zeros(PERSONALITY_DIM, dtype=np.float32)
                padded[: min(vec.shape[0], PERSONALITY_DIM)] = vec[: PERSONALITY_DIM]
                vec = padded
            return PersonalityState(
                personality_vector=vec,
                dominant_trait=_normalize_dominant_trait(row[1]),
                intensity=float(row[2] or 0.0),
                updated_at=row[3] or "",
            )

        # New session: default activation
        return self._default_state()

    # ----------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------

    def _default_state(self) -> PersonalityState:
        vec = np.zeros(PERSONALITY_DIM, dtype=np.float32)
        for subspace in ("openness", "conscientiousness", "extraversion", "neuroticism"):
            s, e = PERSONALITY_SUBSPACE_DIMS[subspace]
            vec[s:e] = DEFAULT_PERSONALITY[subspace]
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return PersonalityState(
            personality_vector=vec,
            dominant_trait="neutral",
            intensity=float(np.linalg.norm(vec)),
            updated_at=now,
        )

    @staticmethod
    def _analyze_personality_vector(vec: np.ndarray) -> Tuple[str, float]:
        """Summarize a 32-dim vector as (dominant trait label, overall intensity)."""
        # Only canonical Big Five keys (not legacy alias keys)
        big5_keys = ("openness", "conscientiousness", "extraversion", "neuroticism")
        subspace_means: Dict[str, float] = {}
        for name in big5_keys:
            s, e = PERSONALITY_SUBSPACE_DIMS[name]
            subspace_means[name] = float(np.mean(np.abs(vec[s:e])))

        dominant = max(subspace_means, key=subspace_means.get)  # type: ignore[arg-type]
        intensity = float(np.linalg.norm(vec)) / (PERSONALITY_DIM ** 0.5)

        trait_labels = {
            "openness": "intellectual exploration",
            "conscientiousness": "task focus",
            "extraversion": "expressive" if float(np.mean(vec[16:24])) > 0 else "reserved",
            "neuroticism": "threat vigilance",
        }
        return trait_labels.get(dominant, "neutral"), float(np.clip(intensity, 0, 1))
