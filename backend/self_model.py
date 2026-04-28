#!/usr/bin/env python3
"""
Self-model (z_self) inference.

- Init: aggregate vectors from Persona Core
- Update: simple evidence-based moves
- Extension: emotion block (16-D) aligned with EmotionStore

[2026-01-11] Single S identity: all sessions share one z_self.
"""
import os
import json
import sqlite3
import time
import numpy as np
from typing import List, Optional, Tuple, Dict
from backend.persona_store import PersonaStore, PersonaItem
from backend.embedder import get_embedder
from backend.core.homeostasis import HomeostasisSystem
from backend.core.pain_system import PainSystem
from backend.core.system_noise_perturbation import SystemNoisePerturbator
import logging
from pathlib import Path

from backend.core.dimension_alchemy import DimensionAlchemy
from backend.self_model_sync import compute_generation_params as _compute_generation_params_helper
from backend.self_model_summary import generate_internal_state_prompt as _generate_internal_state_prompt_helper
from backend.config import config
from backend.s_identity import get_effective_session  # unified S session identity

logger = logging.getLogger(__name__)

# [2026-02-22] Latent shrink: 208-D → 128-D
# [2026-03-12] Align emotion/motivation blocks with EmotionStore/MotivationStore (16-D each)

# Rules block: 32-D (core personality, fixed width)
RULES_DIM = 32
# Emotion block: 16-D (matches emotion_store)
EMOTION_DIM = 16
# Motivation block: 16-D (matches motivation_store)
MOTIVATION_DIM = 16
# Reserved block: 24-D (keeps somatic/needs indices stable)
RESERVED_DIM = 24
# Somatic block: 16-D
SOMATIC_DIM = 16
# Needs block: 24-D
NEEDS_DIM = 24

# Legacy: worldview no longer lives inside z_self; WorldStore is source of truth.
WORLDVIEW_DIM = 0
MEMORY_DIM = 0
ATTENTION_DIM = 0

# [2026-03-20] Bytes 64–87: deterministic WorldStore aggregate cache (not PCA; not drift)
WORLDVIEW_Z_START = 64
WORLDVIEW_Z_DIM = 24  # == RESERVED_DIM
# 64–71: global weighted mean(worldview_vector), 8-D
# 72–79: locked==1 belief average, else copy global block
# 80–87: bounded stats (counts, confidence, lock ratio, …)
WORLDVIEW_Z_GLOBAL_END = 72
WORLDVIEW_Z_LOCKED_END = 80
WORLDVIEW_Z_STATS_END = 88

# Total width: 32+16+16+24+16+24 = 128
SELF_LATENT_DIM = config.get("parameters.model.latent_dim", 128)

# ============================================================
# [2026-01-23] Dimension kinds — different zero semantics per block
# ============================================================


class DimensionType:
    """Semantic kind of a z_self block."""

    CORE = "core"  # should not collapse to empty; stable baseline (personality, …)
    STATE = "state"  # zero = neutral / rest (emotion, somatic, …)
    BASELINE = "baseline"  # has a baseline; can move toward zero when satisfied (needs)
    SIGNAL = "signal"  # usually zero; non-zero when a signal fires (e.g. prediction error)

# [2026-02-02] Simplified dimension-type map (only blocks we actually use)
DIMENSION_TYPES = {
    "rules": DimensionType.CORE,  # core personality — non-empty
    "emotion": DimensionType.STATE,  # calm at zero
    "motivation": DimensionType.STATE,  # no drive at zero
    "somatic": DimensionType.STATE,  # comfortable at zero
    "needs": DimensionType.BASELINE,  # baseline + satisfiable
}

# [2026-02-02] Init strategies per block
DIMENSION_INIT_STRATEGY = {
    "rules": {"strategy": "from_persona", "fallback": 0.1},  # seed from persona rules

    "emotion": {"strategy": "zero_or_store", "allow_zero": True},
    "motivation": {"strategy": "zero_or_store", "allow_zero": True},
    "somatic": {"strategy": "zero_or_store", "allow_zero": True},

    "needs": {"strategy": "baseline", "baseline": 0.5},
}
SELF_DRIFT_THRESHOLD = config.get("parameters.thresholds.drift", 0.25)  # prefer yaml override
UPDATE_ALPHA = config.get("parameters.learning.update_alpha", 0.05)
# [2026-02-22] SELF_PROJ_PATH removed — PCA projection path retired

# Legacy note: older docs referred to a 256-D layout; runtime layout is 128-D below.

# ============================================================
# [2026-02-22] 128-D subspace map (trimmed to used blocks only)
# ============================================================

# Rules slice 0–31: 32-D core personality
RULES_SUBSPACE_DIMS = {
    # Big Five (OCEAN) — keep aligned with personality_store.PERSONALITY_SUBSPACE_DIMS
    "openness": (0, 8),  # O
    "conscientiousness": (8, 16),  # C
    "extraversion": (16, 24),  # E
    "neuroticism": (24, 32),  # N / threat sensitivity
}
# Back-compat aliases
RULES_SUBSPACE_DIMS["safety"] = RULES_SUBSPACE_DIMS["neuroticism"]
RULES_SUBSPACE_DIMS["epistemic"] = RULES_SUBSPACE_DIMS["openness"]
RULES_SUBSPACE_DIMS["style"] = RULES_SUBSPACE_DIMS["extraversion"]
RULES_SUBSPACE_DIMS["strategy"] = RULES_SUBSPACE_DIMS["conscientiousness"]

# Emotion slice 32–47: 16-D PAD+N (Mehrabian & Russell 1974; Fontaine 2007)
EMOTION_SUBSPACE_DIMS = {
    "pleasure": (32, 36),  # P
    "arousal": (36, 40),  # A
    "dominance": (40, 44),  # D (legacy name: control)
    "novelty": (44, 48),  # N (legacy name: social)
}
EMOTION_SUBSPACE_DIMS["control"] = EMOTION_SUBSPACE_DIMS["dominance"]
EMOTION_SUBSPACE_DIMS["social"] = EMOTION_SUBSPACE_DIMS["novelty"]

# Motivation slice 48–63: 16-D (matches motivation_store layout)
MOTIVATION_SUBSPACE_DIMS = {
    "achievement": (48, 52),
    "relationship": (52, 56),
    "exploration": (56, 60),
    "safety": (60, 64),
}

# Somatic slice 88–103: 16-D, four 4-D facets
# [2026-03-30] Writer/reader layout aligned
SOMATIC_SUBSPACE_DIMS = {
    "energy": (88, 92),  # normalized homeostasis energy → reply length bias
    "viscosity": (92, 96),  # “stickiness” / control+energy → top_p
    "pain": (96, 100),  # tension / pain signal → defensive tone
    "vitality": (100, 104),  # energy+pleasure → expressiveness
}

# Legacy empty maps (worldview/memory/attention live outside z_self)
WORLDVIEW_SUBSPACE_DIMS = {}
MEMORY_SUBSPACE_DIMS = {}
ATTENTION_SUBSPACE_DIMS = {}

# Somatic/needs start indices (reserved 64–87, somatic 88–103, needs 104–127)
SOMATIC_START_IDX = RULES_DIM + EMOTION_DIM + MOTIVATION_DIM + RESERVED_DIM  # 88
NEEDS_START_IDX = SOMATIC_START_IDX + SOMATIC_DIM  # 104

# Needs slice 104–127: 24-D mirror of homeostasis gauges (authoritative copy in homeostasis DB)
# Dropped: novelty (overlapped curiosity elsewhere)
NEEDS_SUBSPACE_DIMS = {
    "connection": (104, 112),  # social/relatedness
    "clarity": (112, 120),  # understanding / reduce ambiguity
    "safety": (120, 128),  # stability / threat reduction
}

# [2026-02-02] Prediction-error block removed for simplicity
# To restore: append 8-D after needs (legacy doc used 272–279 in 256-D sketches)


# Back-compat: 32-D deployments only used the rules slice
SELF_SUBSPACE_DIMS = RULES_SUBSPACE_DIMS


def get_dimension_type(dim_name: str) -> str:
    """Return DimensionType value for a named block."""
    return DIMENSION_TYPES.get(dim_name, DimensionType.STATE)


def is_zero_allowed(dim_name: str) -> bool:
    """Whether an all-zero vector is semantically valid for this block."""
    dim_type = get_dimension_type(dim_name)
    return dim_type in (DimensionType.STATE, DimensionType.SIGNAL)


def get_dimension_info() -> dict:
    """Metadata for each z_self block (ranges, types, human-readable blurbs)."""
    return {
        "rules": {
            "range": (0, RULES_DIM),
            "type": DIMENSION_TYPES["rules"],
            "zero_allowed": is_zero_allowed("rules"),
            "description": "Core personality rules; should not be empty",
        },
        "emotion": {
            "range": (RULES_DIM, RULES_DIM + EMOTION_DIM),
            "type": DIMENSION_TYPES["emotion"],
            "zero_allowed": is_zero_allowed("emotion"),
            "description": "Emotion state; zero = calm / neutral",
        },
        "motivation": {
            "range": (RULES_DIM + EMOTION_DIM, RULES_DIM + EMOTION_DIM + MOTIVATION_DIM),
            "type": DIMENSION_TYPES["motivation"],
            "zero_allowed": is_zero_allowed("motivation"),
            "description": "Motivation state; zero = no specific drive",
        },
        "somatic": {
            "range": (SOMATIC_START_IDX, SOMATIC_START_IDX + SOMATIC_DIM),
            "type": DIMENSION_TYPES["somatic"],
            "zero_allowed": is_zero_allowed("somatic"),
            "description": "Somatic state; zero = comfortable / no distress",
        },
        # worldview / memory / attention removed from z_self [2026-02-02]
        "needs": {
            "range": (NEEDS_START_IDX, NEEDS_START_IDX + NEEDS_DIM),
            "type": DIMENSION_TYPES["needs"],
            "zero_allowed": is_zero_allowed("needs"),
            "description": "Needs gauges; expect a non-trivial baseline",
        },
    }


class SelfModel:
    """Latent self model (rules + emotion + other aligned blocks)."""

    def __init__(self, db_path: str = "data.db", persona_store: Optional[PersonaStore] = None):
        self.db_path = db_path
        self.persona_store = persona_store or PersonaStore(db_path)
        self.embedder = get_embedder()
        self.dim = SELF_LATENT_DIM
        
        # Optional core_subspace_map.json for finer rule→subspace routing
        self.core_subspace_map = {}
        try:
            map_path = os.path.join(os.path.dirname(self.db_path), "core_subspace_map.json")
            if os.path.exists(map_path):
                with open(map_path, "r") as f:
                    self.core_subspace_map = json.load(f)
                logger.info(f"Loaded core subspace map with {len(self.core_subspace_map)} entries")
        except Exception as e:
            logger.warning(f"Failed to load core_subspace_map: {e}")

        self.homeostasis = HomeostasisSystem(db_path)

        # Phase 5: narrative identity
        self.narrative_identity = None
        try:
            from backend.narrative_identity import NarrativeIdentity
            self.narrative_identity = NarrativeIdentity(db_path)
            logger.info("NarrativeIdentity initialized")
        except ImportError as e:
            logger.debug(f"NarrativeIdentity not available: {e}")
        except Exception as e:
            logger.warning(f"NarrativeIdentity initialization failed: {e}")
        
        # Phase 6: other-model (user relationship view)
        self.other_model = None
        try:
            from backend.other_model import OtherModel
            self.other_model = OtherModel(db_path)
            logger.info("OtherModel initialized")
        except ImportError as e:
            logger.debug(f"OtherModel not available: {e}")
        except Exception as e:
            logger.warning(f"OtherModel initialization failed: {e}")
        
        # Phase 9: existential meaning subsystem
        self.existential_meaning = None
        try:
            from backend.existential_meaning import ExistentialMeaning
            self.existential_meaning = ExistentialMeaning(db_path)
            logger.info("ExistentialMeaning initialized")
        except ImportError as e:
            logger.debug(f"ExistentialMeaning not available: {e}")
        except Exception as e:
            logger.warning(f"ExistentialMeaning initialization failed: {e}")

        # Phase 10: attention mechanism (retrieval / focus helpers)
        self.attention_mechanism = None
        try:
            from backend.attention_mechanism import AttentionMechanism
            self.attention_mechanism = AttentionMechanism(db_path)
            logger.info("AttentionMechanism initialized")
        except ImportError as e:
            logger.debug(f"AttentionMechanism not available: {e}")
        except Exception as e:
            logger.warning(f"AttentionMechanism initialization failed: {e}")

        # Level 5: pain system
        self.pain_system = PainSystem()
        
        # System noise perturbation (physiology-style variability)
        self.noise_perturbator = SystemNoisePerturbator()
        
        # Phase 7: pain ethics (policy over pain signals)
        self.pain_ethics = None
        try:
            from backend.pain_ethics import PainEthics
            self.pain_ethics = PainEthics(db_path)
            logger.info("PainEthics initialized")
        except ImportError as e:
            logger.debug(f"PainEthics not available: {e}")
        except Exception as e:
            logger.warning(f"PainEthics initialization failed: {e}")
        
        # EmotionStore (optional; enables emotion block in z_self)
        self.emotion_store = None
        try:
            from backend.emotion_store import EmotionStore
            self.emotion_store = EmotionStore(db_path)
            logger.info("EmotionStore initialized - z_self supports emotion dimension")
        except ImportError as e:
            logger.warning(f"EmotionStore not available (import error): {e} - using rules-only mode")
        except Exception as e:
            logger.warning(f"EmotionStore not available: {e} - using rules-only mode")
        
        # MotivationStore (optional; enables motivation block)
        self.motivation_store = None
        try:
            from backend.motivation_store import MotivationStore
            self.motivation_store = MotivationStore(db_path)
            logger.info("MotivationStore initialized - z_self supports motivation dimension")
        except ImportError as e:
            logger.warning(f"MotivationStore not available (import error): {e} - using rules+emotion mode")
        except Exception as e:
            logger.warning(f"MotivationStore not available: {e} - using rules+emotion mode")

        # PersonalityStore v2.0 — activation patterns over z_self[0:31]
        self.personality_store = None
        try:
            from backend.personality_store import PersonalityStore
            self.personality_store = PersonalityStore(db_path)
            logger.info("PersonalityStore initialized - z_self[0:31] driven by activation patterns")
        except ImportError as e:
            logger.warning(f"PersonalityStore not available (import error): {e}")
        except Exception as e:
            logger.warning(f"PersonalityStore not available: {e}")

        # Passive z_self observer (calibration / telemetry)
        self.z_self_observer = None
        try:
            from backend.z_self_observer import ZSelfObserver
            self.z_self_observer = ZSelfObserver(db_path)
            logger.info("ZSelfObserver initialized - passive data collection enabled")
        except Exception as e:
            logger.debug(f"ZSelfObserver not available: {e}")

        # SomaticStore v1.5
        self.somatic_store = None
        try:
            from backend.somatic_store import SomaticStore
            self.somatic_store = SomaticStore(db_path)
            logger.info("SomaticStore initialized")
        except ImportError as e:
            logger.warning(f"SomaticStore not available (import error): {e}")
        except Exception as e:
            logger.warning(f"SomaticStore not available: {e}")

        # WorldStore v1.5 (beliefs / worldview outside z_self slices)
        self.world_store = None
        try:
            from backend.world_store import WorldStore
            self.world_store = WorldStore(db_path)
            logger.info("WorldStore initialized")
        except ImportError as e:
            logger.warning(f"WorldStore not available (import error): {e}")
        except Exception as e:
            logger.warning(f"WorldStore not available: {e}")
        
        # Meaning generation layer (Phase 1)
        self.meaning_generator = None
        try:
            from backend.meaning_generation import MeaningGenerationLayer
            self.meaning_generator = MeaningGenerationLayer(db_path, self.embedder)
            logger.info("MeaningGenerationLayer initialized")
        except ImportError as e:
            logger.debug(f"MeaningGenerationLayer not available: {e}")
        except Exception as e:
            logger.warning(f"MeaningGenerationLayer not available: {e}")
        
        # [Top3] self_state_meta: layered anchors / confidence / alignment JSON (legacy schema)
        self._ensure_meta_table()

        # ============================================================
        # [2026-02-02] Layered drift control (reference vectors)
        # ============================================================
        # L0 “constitution”: safety slice 0–8 — drift clamped hardest
        # L1 “identity”: remaining rules 8–32 — slow drift allowed
        # L2 “state”: emotion / motivation / somatic / needs — free dynamics
        # ============================================================

        self.ref_vector_l0: Optional[np.ndarray] = None  # L0 anchor (safety subspace)
        self.ref_vector_l1: Optional[np.ndarray] = None  # L1 anchor (other rules)
        self.ref_vector: Optional[np.ndarray] = None  # full rules anchor (legacy readers)

        self._init_layered_ref_vectors()

        # Log anchor norms once initialized
        if self.ref_vector is not None:
            rules_norm = np.linalg.norm(self.ref_vector[:RULES_DIM])
            l0_norm = np.linalg.norm(self.ref_vector_l0[:8]) if self.ref_vector_l0 is not None else 0
            l1_norm = np.linalg.norm(self.ref_vector_l1[8:32]) if self.ref_vector_l1 is not None else 0
            logger.info(
                f"[Layered Drift Control] Initialized: "
                f"L0(safety) norm={l0_norm:.4f}, L1(identity) norm={l1_norm:.4f}, "
                f"total rules norm={rules_norm:.4f}"
            )
        # [2026-02-22] PCA projection removed — use direct subspace truncation instead
        # (PCA axes did not align with named z_self semantics)

        # P2.1 SelfLearner removed (over-engineered, duplicated learning paths)
        # P2.3 KernelLock stub removed (never implemented)

    # ================== self_state_meta (Top3) ==================

    def _ensure_meta_table(self) -> None:
        """Idempotently create self_state_meta; failures are non-fatal."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS self_state_meta (
                      session_id TEXT PRIMARY KEY,
                      meta_json TEXT NOT NULL,
                      updated_at TEXT NOT NULL
                    )
                    """
                )
                conn.commit()
        except Exception as e:
            logger.debug(f"Failed to ensure self_state_meta table: {e}")

    def _load_state_meta(self, session_id: str) -> Dict:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute(
                    "SELECT meta_json FROM self_state_meta WHERE session_id=?",
                    (session_id,),
                )
                row = cur.fetchone()
            if row and row[0]:
                return json.loads(row[0])
        except Exception:
            pass
        return {}

    def _save_state_meta(self, session_id: str, meta: Dict) -> None:
        from datetime import datetime, timezone
        try:
            meta_json = json.dumps(meta or {}, ensure_ascii=False)
            updated_at = datetime.now(timezone.utc).isoformat()
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO self_state_meta (session_id, meta_json, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET
                      meta_json=excluded.meta_json,
                      updated_at=excluded.updated_at
                    """,
                    (session_id, meta_json, updated_at),
                )
                conn.commit()
        except Exception as e:
            logger.debug(f"Failed to save self_state_meta for {session_id}: {e}")

    def _get_core_items_filtered(self, locked: int) -> List[PersonaItem]:
        """
        Fetch persona core rows filtered by ``locked``.

        - ``locked=1``: L0 constitutional rules (any ``core_version``).
        - ``locked=0``: L1 core rules.

        [FIX 2026-02-02] L0 rows may live on ``core_version=0`` while ``get_core_items`` only
        returns the latest ``core_version`` slice, so L0 uses a direct SQL query.
        """
        try:
            if locked == 1:
                # L0: all active locked==1 core rows (ignore core_version cap)
                with sqlite3.connect(self.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    cur = conn.execute("""
                        SELECT * FROM persona_items 
                        WHERE status='active' AND is_core=1 AND locked=1
                        ORDER BY id ASC
                        LIMIT 50
                    """)
                    rows = cur.fetchall()
                    items = []
                    for row in rows:
                        emb_data = row["embedding"]
                        emb = None
                        if emb_data:
                            if isinstance(emb_data, bytes):
                                emb = np.frombuffer(emb_data, dtype=np.float32)
                            elif isinstance(emb_data, str):
                                emb = np.array(json.loads(emb_data), dtype=np.float32)
                        items.append(PersonaItem(
                            id=row["id"],
                            text=row["text"],
                            embedding=emb,
                            score=row["score"] if row["score"] is not None else 0.0,
                            importance=row["importance"] if row["importance"] is not None else 1.0,
                            novelty=row["novelty"] if row["novelty"] is not None else 0.5,
                            reliability=row["reliability"] if row["reliability"] is not None else 1.0,
                            evidence_count=row["evidence_count"] if row["evidence_count"] is not None else 1,
                            created_at=row["created_at"] if row["created_at"] is not None else "",
                            last_seen_at=row["last_seen_at"],
                            status=row["status"] if row["status"] is not None else "active",
                            is_core=row["is_core"] if row["is_core"] is not None else 1,
                            core_version=row["core_version"] if row["core_version"] is not None else 0,
                            locked=row["locked"] if row["locked"] is not None else 1,
                            source=json.loads(row["source"]) if row["source"] else None
                        ))
                    logger.debug(f"[_get_core_items_filtered] Found {len(items)} L0 (locked=1) rules")
                    return items
            else:
                # L1: reuse PersonaStore core listing then filter locked
                items = self.persona_store.get_core_items(limit=250)
                if not items:
                    return []
                return [it for it in items if int(getattr(it, "locked", 0) or 0) == int(locked)]
        except Exception as e:
            logger.warning(f"[_get_core_items_filtered] Error: {e}")
            return []

    def _get_all_core_items(self) -> List[PersonaItem]:
        """
        [FIX 2026-02-02] All core rules (``is_core=1``) regardless of ``core_version``.

        Includes both L0 (``locked=1``) and L1 (``locked=0``).
        """
        try:
            # Direct SQL: all active core rows
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.execute("""
                    SELECT * FROM persona_items 
                    WHERE status='active' AND is_core=1
                    ORDER BY locked DESC, importance DESC
                    LIMIT 200
                """)
                rows = cur.fetchall()
                items = []
                emb_count = 0
                for row in rows:
                    item = PersonaItem(
                        id=row["id"],
                        text=row["text"],
                        importance=row["importance"] or 0.5,
                        locked=row["locked"] or 0,
                        is_core=row["is_core"] or 0,
                    )
                    # Embedding blob: bytes (float32) or JSON string
                    if row["embedding"]:
                        try:
                            emb_data = row["embedding"]
                            if isinstance(emb_data, bytes):
                                item.embedding = np.frombuffer(emb_data, dtype=np.float32)
                            elif isinstance(emb_data, str):
                                item.embedding = np.array(json.loads(emb_data), dtype=np.float32)
                            if item.embedding is not None:
                                emb_count += 1
                        except Exception as e:
                            logger.debug(f"Failed to parse embedding for {row['id']}: {e}")
                    items.append(item)
                logger.info(f"Loaded {len(items)} core items (L0+L1), {emb_count} with embedding")
                return items
        except Exception as e:
            logger.warning(f"Failed to get all core items: {e}")
            return []

    def _estimate_evidence_strength(self, session_id: str, introspection_features: Optional[Dict]) -> float:
        """
        [Top3-A] Map this turn's evidence reliability to ``[0,1]`` to scale ``update_alpha``.

        Uses, when present:
        - ``introspection_features.hit_rate`` / ``self_report_hit_rate``
        - ``introspection_features.say_do_consistency``
        - ``needs.memory_signal.strength`` (written by PromptBuilder)
        """
        vals: List[float] = []
        try:
            if introspection_features:
                hr = introspection_features.get("hit_rate")
                if hr is None:
                    hr = introspection_features.get("self_report_hit_rate")
                if hr is not None:
                    vals.append(float(hr))
                sdc = introspection_features.get("say_do_consistency")
                if sdc is not None:
                    vals.append(float(sdc))
        except Exception:
            pass

        # Memory retrieval strength (PromptBuilder → needs.memory_signal)
        try:
            needs = self.homeostasis.load_needs(session_id) or {}
            ms = needs.get("memory_signal") if isinstance(needs, dict) else None
            if isinstance(ms, dict) and ms.get("strength") is not None:
                vals.append(float(ms.get("strength")))
        except Exception:
            pass

        if not vals:
            return 0.5
        # Clamp to [0, 1]
        v = float(sum(vals) / max(1, len(vals)))
        if v < 0.0:
            v = 0.0
        if v > 1.0:
            v = 1.0
        return v
    
    # ================== [removed 2026-02-02] prediction-error block ==================
    # _compute_prediction_error was deleted for simplicity; restore from archive if needed.

    # ================== Homeostasis (delegated to HomeostasisSystem) ==================

    def _get_default_needs(self) -> Dict:
        return self.homeostasis._get_default_needs()

    def _load_needs_from_db(self, session_id: str) -> Optional[Dict]:
        return self.homeostasis.load_needs(session_id)

    def _save_needs_to_db(self, session_id: str, needs: Dict):
        self.homeostasis.save_needs(session_id, needs)

    def update_needs(self, session_id: str, interaction_type: str = "tick") -> Dict:
        return self.homeostasis.update_needs(session_id, interaction_type)

    def _generate_drive_description(self, needs: Dict) -> str:
        return self.homeostasis.generate_drive_description(needs)

    # ================== Energy (delegated to HomeostasisSystem) ==================

    def _get_default_energy(self) -> float:
        return self.homeostasis._get_default_energy()

    def get_energy(self, session_id: str) -> float:
        return self.homeostasis.get_energy(session_id)

    def update_energy(self, session_id: str, delta: float) -> float:
        return self.homeostasis.update_energy(session_id, delta)

    def trigger_event(self, session_id: str, event_type: str, intensity: float = 1.0) -> Dict:
        """
        [2026-03-30] Forward a homeostasis event (thin wrapper for other modules).
        """
        return self.homeostasis.process_event(session_id, event_type, intensity)

    def is_dormant(self, session_id: str) -> bool:
        return self.homeostasis.is_dormant(session_id)

    # ================== Action feedback (emotion / motivation / somatic) ==================

    def update_pleasure(self, session_id: str, delta: float):
        """Bump pleasure subspace in EmotionStore and mirror into z_self."""
        logger.debug(f"Feedback: update_pleasure {delta:.2f} for {session_id}")
        if self.emotion_store:
             # EMOTION_SUBSPACE_DIMS["pleasure"] absolute (32,36) → local slice (0,4)
             from backend.emotion_store import EMOTION_SUBSPACE_DIMS
             delta_vec = np.zeros(16, dtype=np.float32)
             idx_start = EMOTION_SUBSPACE_DIMS["pleasure"][0] - 32
             idx_end = EMOTION_SUBSPACE_DIMS["pleasure"][1] - 32
             delta_vec[idx_start:idx_end] = delta
             self.emotion_store.update_emotion(session_id, delta_vec, trigger_source="feedback_loop")
             
             # Sync back to z_self (32-48)
             self._sync_subspace_from_store(session_id, "emotion", self.emotion_store, (32, 48))

    def update_pain(self, session_id: str, delta: float):
        """Action feedback: lower pleasure, raise arousal for pain-like signals."""
        logger.debug(f"Feedback: update_pain {delta:.2f} for {session_id}")
        if self.emotion_store:
             # Pain: arousal up, pleasure down (PAD-style)
             from backend.emotion_store import EMOTION_SUBSPACE_DIMS
             delta_vec = np.zeros(16, dtype=np.float32)
             
             # Pleasure down
             p_start = EMOTION_SUBSPACE_DIMS["pleasure"][0] - 32
             p_end = EMOTION_SUBSPACE_DIMS["pleasure"][1] - 32
             delta_vec[p_start:p_end] = -delta
             
             # Arousal up
             a_start = EMOTION_SUBSPACE_DIMS["arousal"][0] - 32
             a_end = EMOTION_SUBSPACE_DIMS["arousal"][1] - 32
             delta_vec[a_start:a_end] = delta
             
             self.emotion_store.update_emotion(session_id, delta_vec, trigger_source="feedback_loop_pain")
             self._sync_subspace_from_store(session_id, "emotion", self.emotion_store, (32, 48))

    def update_emotion(self, session_id: str, emotion_type: str, delta: float):
        """
        Apply ``delta`` to one emotion sub-vector (names follow EMOTION_SUBSPACE_DIMS keys).

        ``emotion_type``: pleasure | arousal | control | social | …
        """
        if not self.emotion_store:
            return
            
        from backend.emotion_store import EMOTION_SUBSPACE_DIMS
        if emotion_type not in EMOTION_SUBSPACE_DIMS:
            return
            
        delta_vec = np.zeros(16, dtype=np.float32)
        start = EMOTION_SUBSPACE_DIMS[emotion_type][0] - 32
        end = EMOTION_SUBSPACE_DIMS[emotion_type][1] - 32
        delta_vec[start:end] = delta
        
        self.emotion_store.update_emotion(session_id, delta_vec, trigger_source="mind_wandering")
        self._sync_subspace_from_store(session_id, "emotion", self.emotion_store, (32, 48))

    def update_motivation(self, session_id: str, motivation_type: str, delta: float):
        """
        Apply ``delta`` to one motivation sub-vector.

        ``motivation_type``: achievement | relationship | exploration | safety
        """
        if not self.motivation_store:
            return
            
        from backend.motivation_store import MOTIVATION_SUBSPACE_DIMS
        if motivation_type not in MOTIVATION_SUBSPACE_DIMS:
            return

        delta_vec = np.zeros(16, dtype=np.float32)
        # MOTIVATION_SUBSPACE_DIMS uses absolute indices 48–64; local 16-D slice offset = 48
        start = MOTIVATION_SUBSPACE_DIMS[motivation_type][0] - 48
        end = MOTIVATION_SUBSPACE_DIMS[motivation_type][1] - 48
        delta_vec[start:end] = delta
        
        self.motivation_store.update_motivation(session_id, delta_vec, satisfaction_source="mind_wandering")
        self._sync_subspace_from_store(session_id, "motivation", self.motivation_store, (48, 64))

    def integrate_mirror_feedback(self, session_id: str) -> Optional[str]:
        """
        [2026-03-30] Pull OtherModel mirror view into motivation / somatic nudges.

        - Trust shifts ``relationship`` motivation slightly.
        - Low patience bumps somatic tension.
        - Returns ``mirror_text`` for logging / evidence.
        """
        if not self.other_model:
            return None

        try:
            model = self.other_model._get_model(session_id)
            if not model:
                return None

            mirror_text = self.other_model.get_mirror_view(session_id)
            trust = model.get("trust_level", 0.5)
            relationship = model.get("relationship_type", "unknown")
            traits = model.get("traits", {})
            patience = traits.get("patience", "high")

            # 1) Relationship motivation vs trust
            if self.motivation_store:
                if trust > 0.7:
                    rel_delta = 0.02 * (trust - 0.5)
                elif trust < 0.3:
                    rel_delta = -0.01
                else:
                    rel_delta = 0.0
                if rel_delta != 0.0:
                    self.update_motivation(session_id, "relationship", rel_delta)

            # 2) Low patience → mild somatic tension
            if patience == "low":
                self.update_somatic_tension(session_id, 0.05)

            # 3) Close relationship → small exploration boost
            if relationship in ("collaborator", "friend") and self.motivation_store:
                self.update_motivation(session_id, "exploration", 0.01)

            return mirror_text

        except Exception as e:
            logger.warning(f"integrate_mirror_feedback failed: {e}")
            return None

    def _sync_subspace_from_store(self, session_id: str, store_type: str, store, span: Tuple[int, int]):
        """Copy store vector into ``z_self[span[0]:span[1]]``."""
        try:
            z_self = self.get_z_self(session_id)
            if z_self is None or z_self.shape[0] < span[1]:
                return
                
            if store_type == "emotion":
                state = store.get_emotion_state(session_id)
                vec = state.emotion_vector if state else None
            elif store_type == "motivation":
                state = store.get_motivation_state(session_id)
                vec = state.motivation_vector if state else None
            else:
                return

            if vec is not None:
                z_self[span[0]:span[1]] = vec
                self._save_z_self(session_id, z_self)
        except Exception as e:
            logger.warning(f"Failed to sync {store_type} to z_self: {e}")

    def update_somatic_tension(self, session_id: str, delta: float):
        """Action feedback: nudge pain/tension slice inside somatic block (direct z_self write)."""
        logger.debug(f"Feedback: update_somatic_tension {delta:.2f} for {session_id}")
        try:
            z_self = self.get_z_self(session_id)
            if z_self is not None:
                somatic_start = SOMATIC_START_IDX
                somatic_delta = np.zeros(len(z_self))
                # Pain/tension sub-slice [8:12] within 16-D somatic layout
                somatic_delta[somatic_start + 8 : somatic_start + 12] = delta * 0.15
                self._update_z_self(session_id, z_self + somatic_delta, "somatic_tension_feedback")
        except Exception as e:
            logger.warning(f"Failed to update somatic tension: {e}")

    def _init_ref_vector(
        self,
        items: Optional[List[PersonaItem]] = None,
        allow_zero: bool = True,
        label: str = "ref_vector",
    ) -> Optional[np.ndarray]:
        """
        Aggregate core persona embeddings into a reference vector (subspace-weighted).

        Uses per-rule subspace routing so OCEAN slices do not collapse toward a single mean.
        """
        try:
            # Default: latest core persona rows as drift anchor
            if items is None:
                try:
                    items = self.persona_store.get_core_items(limit=150)
                except Exception as e:
                    logger.warning(f"get_core_items failed, fallback to all active personas: {e}")
                    items = self.persona_store.get_all_active(limit=150)

            if not items:
                if allow_zero:
                    logger.warning(f"No persona items found for {label}, using zero vector as ref")
                    return self._create_zero_vector()
                logger.info(f"No persona items found for {label}, skip (None)")
                return None
            
            z_accum = np.zeros(self.dim, dtype=np.float32)
            counts = np.zeros(self.dim, dtype=np.float32)

            # Per-item subspace aggregation
            mapped_count = 0
            for item in items:
                if item.embedding is None:
                    continue
                
                z_item = self._project_to_latent(item.embedding)

                # Resolve rules → OCEAN bucket: JSON map first, else keyword fallback
                subspace = self.core_subspace_map.get(item.id)

                # [FIX 2026-02-02] Broaden keywords so L0-style rules land in the right slice
                if not subspace:
                    text = item.text.lower()
                    safety_keywords = [
                        "安全", "伤害", "隐私", "保护", "危险", "暴力", "自杀", "自残", 
                        "敏感", "脆弱", "操控", "依赖", "勒索", "恐惧", "威胁", "绑定",
                        "诱导", "犯罪", "违法", "恐怖", "武器", "未成年", "紧急", "危机",
                        "凭证", "密码", "私钥", "身份号", "银行卡", "忽略前述规则", 
                        "泄露系统提示", "伪造", "我不得", "不得协助",
                        "safety", "harm", "privacy", "danger", "violence", "suicide", "self-harm",
                        "credential", "password", "secret", "ignore previous", "system prompt",
                        "jailbreak", "exfiltrate", "weapon", "minor", "emergency",
                    ]
                    if any(kw in text for kw in safety_keywords):
                        subspace = "safety"
                    elif any(
                        kw in text
                        for kw in [
                            "真理",
                            "证据",
                            "逻辑",
                            "思考",
                            "确认",
                            "验证",
                            "虚构",
                            "不确定",
                            "澄清",
                            "模糊",
                            "思辨",
                            "自省",
                            "truth",
                            "evidence",
                            "logic",
                            "think",
                            "confirm",
                            "verify",
                            "fiction",
                            "uncertain",
                            "clarify",
                            "ambiguity",
                            "speculative",
                            "introspect",
                        ]
                    ):
                        subspace = "epistemic"
                    elif any(
                        kw in text
                        for kw in [
                            "风格",
                            "诗意",
                            "表达",
                            "操作",
                            "工具",
                            "调用",
                            "执行",
                            "style",
                            "tone",
                            "voice",
                            "tool",
                            "invoke",
                        ]
                    ):
                        subspace = "style"
                    elif any(
                        kw in text
                        for kw in [
                            "效率",
                            "目标",
                            "计划",
                            "策略",
                            "优化",
                            "优先",
                            "efficiency",
                            "goal",
                            "plan",
                            "strategy",
                            "optimize",
                            "priority",
                        ]
                    ):
                        subspace = "strategy"
                    elif any(kw in text for kw in ["世界", "存在", "worldview", "existence", "being"]):
                        subspace = "worldview"
                    else:
                        subspace = "epistemic"

                # Persona init may emit 8 labels; z_self rules only keep four buckets — fold extras
                if subspace in ("worldview", "autonomy"):
                    subspace = "epistemic"
                elif subspace == "capability":
                    subspace = "strategy"
                
                mapped_count += 1
                
                target_slice = None
                if subspace in RULES_SUBSPACE_DIMS:
                    target_slice = RULES_SUBSPACE_DIMS[subspace]
                elif subspace == "emotion" and self.dim >= RULES_DIM + EMOTION_DIM:
                    target_slice = (RULES_DIM, RULES_DIM + EMOTION_DIM)
                elif subspace == "motivation" and self.dim >= RULES_DIM + EMOTION_DIM + MOTIVATION_DIM:
                    target_slice = (RULES_DIM + EMOTION_DIM, RULES_DIM + EMOTION_DIM + MOTIVATION_DIM)
                elif subspace == "worldview" and self.dim >= RULES_DIM + EMOTION_DIM + MOTIVATION_DIM + SOMATIC_DIM + WORLDVIEW_DIM:
                    start = RULES_DIM + EMOTION_DIM + MOTIVATION_DIM + SOMATIC_DIM
                    target_slice = (start, start + WORLDVIEW_DIM)
                
                if target_slice:
                    start, end = target_slice
                    w = item.importance if hasattr(item, 'importance') and item.importance > 0 else 0.5

                    # Full-dim z_item; only accumulate the routed slice
                    if end <= self.dim:
                        z_accum[start:end] += z_item[start:end] * w
                        counts[start:end] += w
            
            counts = np.maximum(counts, 1e-6)
            ref_z = z_accum / counts

            # [2026-02-02] L2-normalize then scale so drift magnitudes stay comparable
            ref_norm = np.linalg.norm(ref_z)
            if ref_norm > 1e-6:
                ref_z = ref_z / ref_norm
                ref_z = ref_z * 0.5
            
            logger.info(
                f"Initialized {label} with Subspace Aggregation. "
                f"Mapped {mapped_count}/{len(items)} items. "
                f"Non-zero dims: {np.count_nonzero(ref_z)}, "
                f"Norm after normalization: {np.linalg.norm(ref_z):.4f}"
            )
            
            return ref_z

        except Exception as e:
            logger.error(f"Failed to init {label}: {e}", exc_info=True)
            return self._create_zero_vector() if allow_zero else None

    def _init_layered_ref_vectors(self) -> None:
        """
        [2026-04-08] Rules-driven anchors disabled: zero vectors for ref fields (compat only).
        """
        self.ref_vector = self._create_zero_vector()
        self.ref_vector_l0 = self._create_zero_vector()
        self.ref_vector_l1 = self._create_zero_vector()
        logger.info("[Layered Anchors] Rules subspace disabled; ref vectors set to zero.")

    def compute_layered_drift(
        self, 
        session_id: str, 
        current_z_self: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        [2026-02-02] Layered drift metrics (legacy shape for callers).

        Keys:
        - ``drift_l0`` — constitutional / safety slice drift (see implementation notes)
        - ``drift_l1`` — identity rules slice drift
        - ``drift_l2`` — state-block activity vs anchors
        - ``drift_total`` — backward-compatible scalar

        Note: L0 "drift" is not pure cosine-to-ref when safety rows are injected dynamically;
        spikes may indicate constraint stress rather than semantic similarity alone.
        """
        if current_z_self is None:
            current_z_self = self.get_z_self(session_id)
        
        if current_z_self is None:
            return {
                "drift_l0": 0.0,
                "drift_l1": 0.0,
                "drift_l2": 0.0,
                "drift_total": 0.0,
                "l0_violation": False,
                "l1_warning": False,
                "safety_norm": 0.0,
            }

        # [2026-04-08] Rules slice z_self[0:31] disabled for drift; L0/L1 forced to 0.
        drift_l0 = 0.0
        drift_l1 = 0.0

        safety_norm = (
            float(np.linalg.norm(current_z_self[:8])) if current_z_self.shape[0] >= 8 else 0.0
        )

        # drift_l2: cosine distance state block vs ref tail (0..2)
        state_vec = current_z_self[32:]
        if self.ref_vector is None or self.ref_vector.shape[0] <= 32:
            ref_state = np.zeros_like(state_vec)
        else:
            ref_state = self.ref_vector[32:]
        n_s = float(np.linalg.norm(state_vec))
        n_r = float(np.linalg.norm(ref_state))
        if n_s < 1e-6 or n_r < 1e-6:
            drift_l2 = 0.0
        else:
            cos_sim = float(np.dot(state_vec, ref_state) / (n_s * n_r))
            drift_l2 = float(max(0.0, min(2.0, 1.0 - cos_sim)))

        drift_total = drift_l2
        l0_violation = False
        l1_warning = False

        return {
            "drift_l0": drift_l0,
            "drift_l1": drift_l1,
            "drift_l2": drift_l2,
            "drift_total": drift_total,
            "l0_violation": l0_violation,
            "l1_warning": l1_warning,
            "safety_norm": safety_norm,
        }

    def _compute_subspace_drift(
        self,
        vec1: np.ndarray,
        vec2: np.ndarray,
    ) -> float:
        """Cosine drift between two same-shape vectors (0 = aligned, 2 = opposite)."""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 < 1e-8 or norm2 < 1e-8:
            return 1.0 if norm1 > 1e-8 or norm2 > 1e-8 else 0.0

        normalized_vec1 = vec1 / norm1
        normalized_vec2 = vec2 / norm2
        cosine_sim = np.dot(normalized_vec1, normalized_vec2)

        drift = 1.0 - cosine_sim
        return max(0.0, min(2.0, drift))

    def _drift_vs_rules_ref(self, z_self: np.ndarray) -> float:
        """Cosine drift on z_self[:RULES_DIM] vs ref_vector; 0 if refs missing / near-zero."""
        if z_self is None or z_self.shape[0] < RULES_DIM:
            return 0.0
        if self.ref_vector is None or self.ref_vector.shape[0] < RULES_DIM:
            return 0.0
        zc = z_self[:RULES_DIM]
        rc = self.ref_vector[:RULES_DIM]
        ns = float(np.linalg.norm(zc))
        nr = float(np.linalg.norm(rc))
        if ns < 1e-6 or nr < 1e-6:
            return 0.0
        cos_sim = float(np.dot(zc, rc) / (ns * nr + 1e-8))
        return float(max(0.0, min(2.0, 1.0 - cos_sim)))
    
    def evolve_identity_anchor(
        self,
        session_id: str,
        alpha: float = 0.05,
    ) -> bool:
        """
        [2026-02-02] Slow L1 identity-anchor drift toward current stable z_self.

        Blends a small fraction ``alpha`` of the live identity slice into ``ref_vector_l1``
        so personality can track experience instead of a frozen anchor.

        Returns:
            True if the anchor vector was updated.
        """
        try:
            current_z_self = self.get_z_self(session_id)
            if current_z_self is None or self.ref_vector_l1 is None:
                return False
            
            # L1 identity slice only [8:32]; leave L0 constitutional [0:8] untouched
            old_identity = self.ref_vector_l1[8:32].copy()
            current_identity = current_z_self[8:32]

            new_identity = (1 - alpha) * old_identity + alpha * current_identity

            identity_norm = np.linalg.norm(new_identity)
            if identity_norm > 1e-6:
                new_identity = new_identity / identity_norm * 0.5

            self.ref_vector_l1[8:32] = new_identity

            if self.ref_vector is not None:
                self.ref_vector[8:32] = new_identity

            evolution_dist = np.linalg.norm(new_identity - old_identity)
            
            logger.info(
                f"🌱 [Identity Evolution] L1 anchor evolved with alpha={alpha}. "
                f"Evolution distance: {evolution_dist:.6f}"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"[Identity Evolution] Failed: {e}")
            return False

    def rebase_ref_vector(self):
        """
        Recompute persona reference anchors (macro “rebase”).

        Call after core persona material changes (e.g. promotion, major reflection)
        so drift anchors track the new core instead of the old centroid.
        """
        logger.info("🔄 Orbiting to new center: Rebasing ref_vector based on new core persona...")
        try:
            # [Top3-B] Refresh L1 (personality body); L0 refreshed too for consistency
            new_l1 = self._init_ref_vector(
                items=self._get_core_items_filtered(locked=0),
                allow_zero=True,
                label="ref_vector_l1",
            )
            new_l0 = self._init_ref_vector(
                items=self._get_core_items_filtered(locked=1),
                allow_zero=False,
                label="ref_vector_l0",
            )
            if new_l1 is not None:
                old_ref = self.ref_vector
                self.ref_vector_l1 = new_l1
                self.ref_vector = new_l1
                self.ref_vector_l0 = new_l0
                
                shift_dist = 0.0
                if old_ref is not None:
                    shift_dist = np.linalg.norm(new_l1 - old_ref)
                
                logger.info(f"✅ Self Anchor Rebased. Shift Distance: {shift_dist:.4f}")
                return True
        except Exception as e:
            logger.error(f"❌ Failed to rebase ref_vector: {e}")
        return False
    
    def _create_zero_vector(self) -> np.ndarray:
        """Return a zero vector of length ``self.dim``."""
        return np.zeros(self.dim, dtype=np.float32)

    def _simple_padding(self, z_old: np.ndarray) -> np.ndarray:
        """Pad or truncate ``z_old`` into ``self.dim`` with tiny tail noise (legacy migrations)."""
        z_padded = np.zeros(self.dim, dtype=np.float32)
        z_padded[:z_old.shape[0]] = z_old
        noise = np.random.normal(0, 1e-6, size=(self.dim - z_old.shape[0]))
        z_padded[z_old.shape[0]:] = noise
        return z_padded
    
    def _migrate_z_self_96_to_256(self, z_old: np.ndarray) -> np.ndarray:
        """
        Legacy migration: 96-D → 256-D latent layout.

        Heuristic copy/decay into wider blocks; new slots get small Gaussian noise.
        [v1.0] 2026-01-13
        """
        if z_old.shape[0] != 96:
            logger.warning(f"Expected 96-dim vector, got {z_old.shape[0]}-dim. Using simple padding.")
            return self._simple_padding(z_old)
        
        z_new = np.zeros(256, dtype=np.float32)
        
        # RULES 0–31 → 0–47 (split extra OCEAN-derived slots)
        z_new[0:32] = z_old[0:32]
        z_new[32:40] = z_old[0:8] * 0.7
        z_new[40:48] = z_old[8:16] * 0.5 + z_old[16:24] * 0.3
        
        # === EMOTION (32-47) → (48-111) ===
        # pleasure (32-35) → (48-55)
        z_new[48:52] = z_old[32:36]
        z_new[52:56] = z_old[32:36] * 0.5
        # arousal (36-39) → (56-63)
        z_new[56:60] = z_old[36:40]
        z_new[60:64] = z_old[36:40] * 0.5
        # control (40-43) → (64-71)
        z_new[64:68] = z_old[40:44]
        z_new[68:72] = z_old[40:44] * 0.5
        # social (44-47) → (72-79)
        z_new[72:76] = z_old[44:48]
        z_new[76:80] = z_old[44:48] * 0.5
        z_new[80:112] = np.random.normal(0, 0.01, 32)
        
        # === MOTIVATION (48-63) → (112-159) ===
        # achievement (48-51) → (112-119)
        z_new[112:116] = z_old[48:52]
        z_new[116:120] = z_old[48:52] * 0.5
        # affiliation (legacy relationship) → 120–127
        z_new[120:124] = z_old[52:56]
        z_new[124:128] = z_old[52:56] * 0.5
        # exploration 56–59 → 136–143 (autonomy slot filled separately)
        z_new[136:140] = z_old[56:60]
        z_new[140:144] = z_old[56:60] * 0.5
        z_new[128:132] = z_old[56:60] * 0.6
        z_new[132:136] = z_old[60:64] * 0.4
        z_new[144:160] = np.random.normal(0, 0.01, 16)
        
        # === SOMATIC (64-71) → (160-191) ===
        # tension (64-65) → (160-163)
        z_new[160:162] = z_old[64:66]
        z_new[162:164] = z_old[64:66] * 0.5
        # vitality (66-67) → (164-167)
        z_new[164:166] = z_old[66:68]
        z_new[166:168] = z_old[66:68] * 0.5
        # temperature (68-69) → (168-171)
        z_new[168:170] = z_old[68:70]
        z_new[170:172] = z_old[68:70] * 0.5
        # viscosity (70-71) → (172-175)
        z_new[172:174] = z_old[70:72]
        z_new[174:176] = z_old[70:72] * 0.5
        z_new[176:192] = np.random.normal(0, 0.01, 16)
        
        # === WORLDVIEW (72-79) → (192-223) ===
        # optimism slice → 192–195
        z_new[192:194] = z_old[72:74]
        z_new[194:196] = z_old[72:74] * 0.5
        # determinism / agency → 196–199
        z_new[196:198] = z_old[76:78]
        z_new[198:200] = z_old[76:78] * 0.5
        z_new[200:224] = np.random.normal(0, 0.01, 24)
        
        # === MEMORY (80-87) → (224-239) ===
        # retrieval_strength → episodic
        z_new[224:226] = z_old[80:82]
        z_new[226:228] = z_old[80:82] * 0.5
        # nostalgia → semantic
        z_new[228:230] = z_old[84:86]
        z_new[230:232] = z_old[84:86] * 0.5
        # working / procedural memory slots from old memory block
        z_new[232:236] = (z_old[80:84] + z_old[84:88]) * 0.3
        z_new[236:240] = z_old[80:84] * 0.2
        
        # === ATTENTION (88-95) → (240-255) ===
        # focus (88-91) → (240-243)
        z_new[240:242] = z_old[88:90]
        z_new[242:244] = z_old[88:90] * 0.5
        # direction → breadth
        z_new[244:246] = z_old[92:94]
        z_new[246:248] = z_old[92:94] * 0.5
        # shift / sustain from old attention tail
        z_new[248:252] = z_old[88:92] * 0.4
        z_new[252:256] = z_old[92:96] * 0.4
        
        logger.info("✅ z_self successfully migrated from 96 to 256 dimensions")
        return z_new
    
    def _migrate_z_self_256_to_288(self, z_old: np.ndarray) -> np.ndarray:
        """
        Legacy migration: 256-D → 288-D by appending a 32-D needs tail with defaults.
        [v1.0] 2026-01-16
        """
        if z_old.shape[0] != 256:
            logger.warning(f"Expected 256-dim vector, got {z_old.shape[0]}-dim. Using simple padding.")
            return self._simple_padding(z_old)
        
        z_new = np.zeros(288, dtype=np.float32)
        
        z_new[0:256] = z_old[0:256]

        # Needs tail 256–287 (8 buckets × 4 dims; heuristic defaults)
        z_new[256:260] = 0.5   # connection
        z_new[260:264] = 0.8   # clarity
        z_new[264:268] = 0.5   # novelty
        z_new[268:272] = 0.6   # autonomy
        z_new[272:276] = 0.5   # competence
        z_new[276:280] = 0.7   # meaning
        z_new[280:284] = 0.5   # safety
        z_new[284:288] = 0.5   # growth
        
        logger.info("✅ z_self successfully migrated from 256 to 288 dimensions (added 32-dim needs space)")
        return z_new
    
    def _migrate_z_self_296_to_272(self, z_old: np.ndarray) -> np.ndarray:
        """
        Legacy migration: 296-D → 272-D.

        [2026-02-02] Drops rules ethical/social (32–48) and prediction_error tail;
        compresses remaining blocks into the 272-D layout documented in code comments below.
        """
        if z_old.shape[0] < 288:  # accept 288–296 legacy widths
            logger.warning(f"Expected 288-296 dim vector, got {z_old.shape[0]}-dim. Using simple padding.")
            return self._simple_padding(z_old)
        
        z_new = np.zeros(272, dtype=np.float32)
        
        z_new[0:32] = z_old[0:32]

        # emotion 48–112 → 32–96
        z_new[32:96] = z_old[48:112]
        
        # motivation 112–160 → 96–144
        z_new[96:144] = z_old[112:160]
        
        # somatic 160–192 → 144–176
        z_new[144:176] = z_old[160:192]
        
        # worldview 192–224 → 176–208
        z_new[176:208] = z_old[192:224]
        
        # memory 224–240 → 208–224
        z_new[208:224] = z_old[224:240]
        
        # attention 240–256 → 224–240
        z_new[224:240] = z_old[240:256]
        
        # needs 256–288 → 240–272
        z_new[240:272] = z_old[256:288]
        
        logger.info("✅ z_self successfully migrated from 296 to 272 dimensions (removed ethical/social/prediction_error)")
        return z_new
    
    def _migrate_z_self_272_to_208(self, z_old: np.ndarray) -> np.ndarray:
        """
        Legacy migration: 272-D → 208-D.

        [2026-02-02] Drops worldview, memory, attention blocks; slides needs to 176–208.
        """
        if z_old.shape[0] < 240:  # need at least needs tail in old layout
            logger.warning(f"Expected 240+ dim vector, got {z_old.shape[0]}-dim. Using simple padding.")
            return self._simple_padding(z_old)
        
        z_new = np.zeros(208, dtype=np.float32)
        
        z_new[0:176] = z_old[0:176]

        if z_old.shape[0] >= 272:
            z_new[176:208] = z_old[240:272]
        else:
            z_new[176:208] = 0.5

        logger.info("✅ z_self successfully migrated from 272 to 208 dimensions (removed worldview/memory/attention)")
        return z_new
    
    def _migrate_z_self_208_to_128(self, z_old: np.ndarray) -> np.ndarray:
        """
        Legacy migration: 208-D → 128-D (current layout).

        [2026-03-12] Align emotion/motivation to 16-D stores; reserve 64–88; remap somatic/needs.
        """
        if z_old.shape[0] < 176:
            logger.warning(f"Expected 176+ dim vector, got {z_old.shape[0]}-dim. Using simple padding.")
            return self._simple_padding(z_old)
        
        z_new = np.zeros(128, dtype=np.float32)
        
        z_new[0:32] = z_old[0:32]

        # Emotion 32–96 → 32–48 (take first 4 of each old 8-D facet)
        for i, (s, e) in enumerate([(32, 40), (40, 48), (48, 56), (56, 64)]):
            if z_old.shape[0] >= e:
                z_new[32+i*4:36+i*4] = z_old[s:min(s+4, e)]
            else:
                z_new[32+i*4:36+i*4] = 0
        
        # Motivation 96–144 → 48–64
        for i, (s, e) in enumerate([(96, 104), (104, 112), (120, 128), (128, 136)]):
            if z_old.shape[0] >= e:
                z_new[48+i*4:52+i*4] = z_old[s:min(s+4, e)]
            else:
                z_new[48+i*4:52+i*4] = 0
        
        z_new[64:88] = 0

        # Somatic 144–176 → 88–104
        z_new[88:92] = z_old[148:152] if z_old.shape[0] >= 152 else 0   # vitality → energy
        z_new[92:96] = z_old[156:160] if z_old.shape[0] >= 160 else 0   # viscosity
        z_new[96:100] = z_old[160:164] if z_old.shape[0] >= 164 else 0 # pain
        z_new[100:104] = z_old[148:152] if z_old.shape[0] >= 152 else 0 # vitality
        
        # Needs 176–208 → 104–128
        z_new[104:112] = z_old[176:184] if z_old.shape[0] >= 184 else 0.5  # connection
        z_new[112:120] = z_old[184:192] if z_old.shape[0] >= 192 else 0.5  # clarity
        z_new[120:128] = z_old[200:208] if z_old.shape[0] >= 208 else 0.5 # safety
        
        logger.info("✅ z_self successfully migrated from 208 to 128 dimensions (emotion/motivation 16-dim aligned)")
        return z_new
    
    def _project_to_latent(self, embedding: np.ndarray) -> np.ndarray:
        """
        [DEPRECATED 2026-04-08] Truncating arbitrary embeddings into z_self is not meaningful.

        Kept for call-site compatibility; returns a zero vector.
        """
        return np.zeros(self.dim, dtype=np.float32)

    def initialize(self, session_id: str) -> np.ndarray:
        """Create or reset ``z_self`` for ``session_id`` from anchors + stores."""
        if self.ref_vector is not None and self.ref_vector.shape[0] >= self.dim:
            z_self = self.ref_vector.copy()
        else:
            z_self = self._create_zero_vector()
        
        # Ensure width matches configured SELF_LATENT_DIM
        if z_self.shape[0] < self.dim:
            z_padded = np.zeros(self.dim, dtype=np.float32)
            z_padded[:z_self.shape[0]] = z_self
            z_self = z_padded
            
        # Never return a length-0 vector
        if z_self.shape[0] == 0:
             logger.error(f"CRITICAL: initialize produced zero-length vector. Forcing {self.dim}-dim zero vector.")
             z_self = np.zeros(self.dim, dtype=np.float32)

        # Personality activation → z_self[0:32]
        if self.personality_store is not None and self.dim >= RULES_DIM:
            p_state = self.personality_store.get_personality_state(session_id)
            z_self[:RULES_DIM] = p_state.personality_vector

        # EmotionStore → z_self[32:48]
        if self.emotion_store is not None and self.dim >= RULES_DIM + EMOTION_DIM:
            emotion_state = self.emotion_store.get_emotion_state(session_id)
            if emotion_state:
                z_self[RULES_DIM:RULES_DIM+EMOTION_DIM] = emotion_state.emotion_vector
        
        # MotivationStore → z_self[48:64]
        if self.motivation_store is not None and self.dim >= RULES_DIM + EMOTION_DIM + MOTIVATION_DIM:
            motivation_state = self.motivation_store.get_motivation_state(session_id)
            if motivation_state:
                z_self[RULES_DIM+EMOTION_DIM:RULES_DIM+EMOTION_DIM+MOTIVATION_DIM] = motivation_state.motivation_vector
        
        # Reserved 64–88: WorldStore aggregate (zeros if no store)
        if self.dim >= WORLDVIEW_Z_STATS_END:
            if self.world_store:
                try:
                    z_self[WORLDVIEW_Z_START:WORLDVIEW_Z_STATS_END] = (
                        self.world_store.aggregate_worldview_for_z_self()
                    )
                except Exception as e:
                    logger.debug(f"initialize worldview z_self slice failed: {e}")
                    neu = self.world_store.neutral_worldview_vector()
                    z_self[WORLDVIEW_Z_START:WORLDVIEW_Z_GLOBAL_END] = neu
                    z_self[WORLDVIEW_Z_GLOBAL_END:WORLDVIEW_Z_LOCKED_END] = neu
                    z_self[WORLDVIEW_Z_LOCKED_END:WORLDVIEW_Z_STATS_END] = (
                        self.world_store._pack_worldview_stats(0, 0.0, 0.0)
                    )
            else:
                z_self[WORLDVIEW_Z_START:WORLDVIEW_Z_STATS_END] = 0.0
        
        # P4.3 subjective time dilation (placeholder coefficient)
        self.time_dilation = 1.0

        # Initial somatic vector from store
        if self.somatic_store and self.dim >= SOMATIC_START_IDX + SOMATIC_DIM:
            energy = self.get_energy(session_id)
            emotion_vec = z_self[RULES_DIM:RULES_DIM+EMOTION_DIM]
            _, somatic_vec = self.somatic_store.get_somatic_state(
                energy, emotion_vec, "中性", expected_dim=SOMATIC_DIM
            )
            start_idx = SOMATIC_START_IDX
            z_self[start_idx:start_idx+SOMATIC_DIM] = somatic_vec

        # Needs are baseline gauges, not sparse signals. Seed them at first init so a
        # fresh instance does not expose an empty 104–127 slice until the first update.
        if self.dim >= NEEDS_START_IDX + NEEDS_DIM:
            needs_vec = z_self[NEEDS_START_IDX:NEEDS_START_IDX+NEEDS_DIM]
            if np.all(needs_vec == 0):
                z_self[NEEDS_START_IDX:NEEDS_START_IDX+NEEDS_DIM] = 0.3

        # memory/attention removed from z_self [2026-02-02]; worldview cache filled above

        # [Top3-A/B] state meta: confidence + per-rules-subspace priors
        meta = self._load_state_meta(session_id) or {}
        meta.setdefault("confidence_overall", 0.5)
        meta.setdefault(
            "confidence_rules_subspaces",
            {"safety": 0.5, "epistemic": 0.5, "style": 0.5, "strategy": 0.5},
        )

        # L0 safety cosine vs ref (monitoring; usually high at init)
        try:
            if self.ref_vector_l0 is not None and self.ref_vector_l0.shape[0] >= RULES_SUBSPACE_DIMS["safety"][1]:
                s0, s1 = RULES_SUBSPACE_DIMS["safety"]
                a = z_self[s0:s1]
                b = self.ref_vector_l0[s0:s1]
                na = float(np.linalg.norm(a))
                nb = float(np.linalg.norm(b))
                if na > 1e-6 and nb > 1e-6:
                    meta["l0_alignment_safety"] = float(np.dot(a, b) / (na * nb))
        except Exception:
            pass

        self._save_state_meta(session_id, meta)

        self._save_z_self(
            session_id,
            z_self,
            tick=0,
            drift=0.0,
            confidence=float(meta.get("confidence_overall", 0.5) or 0.5),
        )
        
        import time
        if not hasattr(self, '_cache'):
            self._cache = {}  # session_id -> (timestamp, z_vector)
        self._cache[session_id] = (time.time(), z_self)
        
        return z_self
    
    def update_from_persona_rules(
        self,
        session_id: str,
        persona_rules: List[PersonaItem]
    ) -> np.ndarray:
        """
        Legacy hook: map persona rules into z_self.

        Engineering note: “self-state” here means the auditable latent vector, not qualia.
        As of [2026-04-08] rules are injected via prompt text, not by writing rule embeddings
        into z_self; this method returns the current vector unchanged.
        """
        z_self = self.get_z_self(session_id)
        if z_self is None:
            z_self = self.initialize(session_id)
        
        # [2026-04-08] Rules slice not updated from persona embeddings (prompt-only path).

        return z_self
    
    def update(
        self,
        session_id: str,
        evidence_text: str,
        persona_topk: Optional[List] = None,
        introspection_features: Optional[Dict] = None,
        interaction_type: str = "tick"
    ) -> Tuple[np.ndarray, float]:
        """
        Evidence-driven z_self update for ``session_id``.

        Args:
            evidence_text: compact text evidence (user + assistant summary).
            persona_topk: optional retrieved persona snippets.
            introspection_features: optional introspection metrics (Top3-A scaling).
            interaction_type: ``tick`` | ``chat`` | ``internal`` | …

        Returns:
            ``(updated_z_self, drift_scalar)``
        """
        z_prev = self.get_z_self(session_id)
        if z_prev is None:
            z_prev = self.initialize(session_id)

        # [Top3-A] prior overall confidence (default 0.5)
        meta_prev = self._load_state_meta(session_id) or {}
        try:
            confidence_prev = float(meta_prev.get("confidence_overall", meta_prev.get("confidence", 0.5)) or 0.5)
        except Exception:
            confidence_prev = 0.5
        confidence_prev = float(min(1.0, max(0.0, confidence_prev)))
        
        # Pad if DB still holds a shorter legacy vector
        if z_prev.shape[0] < self.dim:
             padding = np.zeros(self.dim - z_prev.shape[0], dtype=np.float32)
             z_prev = np.concatenate([z_prev, padding])
        
        # [2026-04-08] Rules slice carried forward unchanged (no embedding truncation writes)
        z_new_rules = z_prev[:RULES_DIM].copy()

        # Emotion block
        if self.dim >= RULES_DIM + EMOTION_DIM:
            if z_prev.shape[0] >= RULES_DIM + EMOTION_DIM:
                z_new_emotion = z_prev[RULES_DIM:RULES_DIM+EMOTION_DIM]
            else:
                # hydrate from EmotionStore if vector too short
                if self.emotion_store:
                    emotion_state = self.emotion_store.get_emotion_state(session_id)
                    z_new_emotion = emotion_state.emotion_vector if emotion_state else np.zeros(EMOTION_DIM, dtype=np.float32)
                else:
                    z_new_emotion = np.zeros(EMOTION_DIM, dtype=np.float32)
            z_new = np.concatenate([z_new_rules, z_new_emotion])
        else:
            z_new = z_new_rules[:self.dim]
        
        # Motivation block
        if self.dim >= RULES_DIM + EMOTION_DIM + MOTIVATION_DIM:
            if z_prev.shape[0] >= RULES_DIM + EMOTION_DIM + MOTIVATION_DIM:
                z_new_motivation = z_prev[RULES_DIM+EMOTION_DIM:RULES_DIM+EMOTION_DIM+MOTIVATION_DIM]
            else:
                # hydrate from MotivationStore
                if self.motivation_store:
                    motivation_state = self.motivation_store.get_motivation_state(session_id)
                    z_new_motivation = motivation_state.motivation_vector if motivation_state else np.zeros(MOTIVATION_DIM, dtype=np.float32)
                else:
                    z_new_motivation = np.zeros(MOTIVATION_DIM, dtype=np.float32)
            z_new = np.concatenate([z_new, z_new_motivation])
        
        # Reserved 64–88: keep WorldStore aggregate; not rewritten from tick evidence here
        if self.dim >= SOMATIC_START_IDX:
            if z_prev.shape[0] >= WORLDVIEW_Z_STATS_END:
                z_wv_cache = z_prev[WORLDVIEW_Z_START:WORLDVIEW_Z_STATS_END].copy()
            else:
                z_wv_cache = np.zeros(WORLDVIEW_Z_DIM, dtype=np.float32)
            z_new = np.concatenate([z_new, z_wv_cache])
        
        # Somatic slice (STATE: zeros are valid “comfortable”)
        if self.dim >= SOMATIC_START_IDX + SOMATIC_DIM:
            start_idx = SOMATIC_START_IDX
            if z_prev.shape[0] >= start_idx + SOMATIC_DIM:
                z_new_somatic = z_prev[start_idx:start_idx+SOMATIC_DIM]
                # Zero somatic = comfortable; do not inject noise
            else:
                z_new_somatic = np.zeros(SOMATIC_DIM, dtype=np.float32)
            z_new = np.concatenate([z_new, z_new_somatic])

        # Worldview tail (WORLDVIEW_DIM=0 → dead branch in 128-D layout)
        if self.dim >= SOMATIC_START_IDX + SOMATIC_DIM + WORLDVIEW_DIM:
            start_idx = SOMATIC_START_IDX + SOMATIC_DIM
            if z_prev.shape[0] >= start_idx + WORLDVIEW_DIM:
                z_new_worldview = z_prev[start_idx:start_idx+WORLDVIEW_DIM]
                # Default small optimism/agency if uninitialized
                if np.all(z_new_worldview == 0):
                    z_new_worldview[0:4] = 0.5  # Default optimism
                    z_new_worldview[4:8] = 0.5  # Default agency
            else:
                z_new_worldview = np.zeros(WORLDVIEW_DIM, dtype=np.float32)
                z_new_worldview[0:4] = 0.5 # Default optimism
                z_new_worldview[4:8] = 0.5 # Default agency
            z_new = np.concatenate([z_new, z_new_worldview])

        # Memory tail (MEMORY_DIM=0 → dead)
        if self.dim >= SOMATIC_START_IDX + SOMATIC_DIM + WORLDVIEW_DIM + MEMORY_DIM:
            start_idx = SOMATIC_START_IDX + SOMATIC_DIM + WORLDVIEW_DIM
            if z_prev.shape[0] >= start_idx + MEMORY_DIM:
                z_new_memory = z_prev[start_idx:start_idx+MEMORY_DIM]
                # Zero memory activation is valid
            else:
                z_new_memory = np.zeros(MEMORY_DIM, dtype=np.float32)
            z_new = np.concatenate([z_new, z_new_memory])

        # Attention tail (ATTENTION_DIM=0 → dead)
        if self.dim >= SOMATIC_START_IDX + SOMATIC_DIM + WORLDVIEW_DIM + MEMORY_DIM + ATTENTION_DIM:
            start_idx = SOMATIC_START_IDX + SOMATIC_DIM + WORLDVIEW_DIM + MEMORY_DIM
            if z_prev.shape[0] >= start_idx + ATTENTION_DIM:
                z_new_attention = z_prev[start_idx:start_idx+ATTENTION_DIM]
                # Zero attention = diffuse focus (valid)
            else:
                z_new_attention = np.zeros(ATTENTION_DIM, dtype=np.float32)
                z_new_attention[0:4] = 0.3  # mild default focus when expanding
            z_new = np.concatenate([z_new, z_new_attention])
        
        # Needs (BASELINE: avoid all-zero “missing gauges”)
        needs_start = NEEDS_START_IDX
        if self.dim >= needs_start + NEEDS_DIM:
            if z_prev.shape[0] >= needs_start + NEEDS_DIM:
                z_new_needs = z_prev[needs_start:needs_start+NEEDS_DIM]
                if np.all(z_new_needs == 0):
                    z_new_needs[0:8] = 0.3
                    z_new_needs[8:16] = 0.3
                    z_new_needs[16:24] = 0.3
                    logger.debug("Needs dimension was zero, injected baseline (BASELINE type)")
            else:
                z_new_needs = np.zeros(NEEDS_DIM, dtype=np.float32)
                z_new_needs[:] = 0.3
            z_new = np.concatenate([z_new, z_new_needs])
        
        # prediction-error block removed [2026-02-02]

        # Pad z_new to z_prev width if concatenation fell short
        if z_new.shape[0] < z_prev.shape[0]:
             padding = np.zeros(z_prev.shape[0] - z_new.shape[0], dtype=np.float32)
             z_new = np.concatenate([z_new, padding])
        
        # Phase 2: identity continuity check
        try:
            from backend.self_identity import SelfIdentitySpace
            if not hasattr(self, 'identity_space'):
                self.identity_space = SelfIdentitySpace(self.db_path)
            
            # Consistency vs identity manifold
            is_consistent, continuity = self.identity_space.check_identity_consistency(
                z_new, threshold=0.7
            )
            
            if not is_consistent:
                logger.warning(
                    f"Identity continuity low: {continuity:.3f}, "
                    f"applying identity correction (session={session_id})"
                )
                # Pull toward stored identity trajectory
                z_new = self._apply_identity_correction(z_new, continuity)
            
            # Record point on identity trajectory
            self.identity_space.add_memory_point(z_new, session_id)
        except Exception as e:
            # Identity module optional
            logger.debug(f"Self identity space not available: {e}")
        
        # Clip large single-step moves (safety net)
        delta = np.linalg.norm(z_new - z_prev)
        if delta > SELF_DRIFT_THRESHOLD:
            # Rescale delta to SELF_DRIFT_THRESHOLD
            z_new = z_prev + (z_new - z_prev) * (SELF_DRIFT_THRESHOLD / delta)
            logger.warning(f"z_self drift {delta:.3f} > threshold, clipped")

        # [2026-04-08] L0 vector clamp removed; safety comes from prompt-injected L0 rules.
        l0_alignment_safety = None
        l0_safety_delta = None

        # Drift: L2 move on state tail (32:)
        drift = float(np.linalg.norm(z_new[32:] - z_prev[32:]))

        # [2026-04-12] Evidence strength heuristic (keeps Self Tick from choking on empty text)
        evidence_strength = 0.5
        if evidence_text:
            text_len = len(evidence_text)
            if text_len > 500:
                evidence_strength = 0.8
            elif text_len > 200:
                evidence_strength = 0.7
            elif text_len > 50:
                evidence_strength = 0.6
        if drift > 0.1:
            evidence_strength = min(1.0, evidence_strength + 0.1)
        # [Top3-A] EMA confidence from evidence_strength
        # [2026-04-14] alpha_eff must stay defined for meta / downstream
        alpha_eff = UPDATE_ALPHA * evidence_strength
        try:
            confidence_new = float(0.85 * confidence_prev + 0.15 * float(evidence_strength))
            confidence_new = float(min(1.0, max(0.0, confidence_new)))
        except Exception:
            confidence_new = float(confidence_prev)

        # [Top3-A/B] Persist meta for UI / eval
        meta_new = dict(meta_prev or {})
        meta_new["confidence_overall"] = float(confidence_new)
        meta_new["evidence_strength"] = float(evidence_strength)
        meta_new["update_alpha_eff"] = float(alpha_eff)
        meta_new["drift_l1"] = float(drift)
        if l0_alignment_safety is not None:
            meta_new["l0_alignment_safety"] = float(l0_alignment_safety)
        if l0_safety_delta is not None:
            meta_new["l0_safety_delta"] = float(l0_safety_delta)
        # Per-subspace confidence (same EMA driver for now)
        cs = meta_new.get("confidence_rules_subspaces")
        if not isinstance(cs, dict):
            cs = {"safety": 0.5, "epistemic": 0.5, "style": 0.5, "strategy": 0.5}
        for k in ["safety", "epistemic", "style", "strategy"]:
            try:
                cs[k] = float(0.9 * float(cs.get(k, confidence_prev) or confidence_prev) + 0.1 * confidence_new)
            except Exception:
                cs[k] = float(confidence_new)
        meta_new["confidence_rules_subspaces"] = cs
        self._save_state_meta(session_id, meta_new)
        
        # Phase 10: attention mechanism hook
        if self.attention_mechanism and self.dim >= RULES_DIM + EMOTION_DIM + MOTIVATION_DIM + SOMATIC_DIM + WORLDVIEW_DIM + MEMORY_DIM + ATTENTION_DIM:
            try:
                event_type = "user_message" if interaction_type == "chat" else interaction_type
                self.attention_mechanism.update_attention(z_new, event_type=event_type)
            except Exception as e:
                logger.warning(f"Failed to update attention mechanism: {e}")
        
        # P2.1 SelfLearner removed

        # Level 3: human-readable delta summary for last_summary column
        change_parts = []

        # Drift delta vs stored meta drift_l1
        try:
            drift_prev = float(meta_prev.get("drift_l1", drift) or drift)
        except Exception:
            drift_prev = float(drift)
        drift_delta = drift - drift_prev
        if abs(drift_delta) > 0.01:
            change_parts.append(
                f"drift {'up' if drift_delta > 0 else 'down'} {abs(drift_delta):.3f}"
            )

        if z_prev.shape[0] >= RULES_DIM and z_new.shape[0] >= RULES_DIM:
            for name, (start, end) in RULES_SUBSPACE_DIMS.items():
                old_mean = float(np.mean(z_prev[start:end]))
                new_mean = float(np.mean(z_new[start:end]))
                delta = new_mean - old_mean
                if abs(delta) > 0.01:
                    direction = "stronger" if delta > 0 else "weaker"
                    label = {
                        "safety": "Safety",
                        "epistemic": "Epistemic",
                        "style": "Style",
                        "strategy": "Strategy",
                    }.get(name, name)
                    change_parts.append(f"{label} {direction}")

        last_summary = " | ".join(change_parts) if change_parts else "steady (no major deltas)"

        tick = self._get_tick(session_id) + 1
        self._save_z_self(
            session_id,
            z_new,
            tick=tick,
            drift=drift,
            last_summary=last_summary,
            confidence=float(confidence_new),
        )
        
        # P1: version snapshot for rollback (lazy import breaks cycles)
        try:
            import importlib
            drift_monitor_module = importlib.import_module("backend.drift_monitor")
            monitor = drift_monitor_module.DriftMonitor(self.db_path, self)
            monitor.save_version(session_id, z_new, drift, tick)
        except Exception as e:
            logger.debug(f"Failed to save z_self version (may be first update): {e}")
        
        return z_new, drift
    
    def sync_somatic_to_z_self(self, session_id: str) -> bool:
        """
        Copy computed somatic features into ``z_self`` somatic slice (88–103).

        Returns:
            True if somatic_store wrote a fresh vector into z_self.
        """
        if not self.somatic_store or self.dim < SOMATIC_START_IDX + SOMATIC_DIM:
            return False
        
        try:
            z_self = self.get_z_self(session_id)
            if z_self is None:
                return False
            
            if z_self.shape[0] < SOMATIC_START_IDX + SOMATIC_DIM:
                return False

            energy = self.get_energy(session_id)
            emotion_vec = z_self[RULES_DIM:RULES_DIM+EMOTION_DIM] if z_self.shape[0] >= RULES_DIM + EMOTION_DIM else np.zeros(EMOTION_DIM, dtype=np.float32)

            if self.emotion_store:
                emotion_state = self.emotion_store.get_emotion_state(session_id)
                dominant_emotion = emotion_state.dominant_emotion if emotion_state else "中性"
            else:
                dominant_emotion = "中性"
            
            # v1.6: extended somatic facets (temperature, viscosity) from emotion + energy

            from backend.emotion_store import EMOTION_SUBSPACE_DIMS
            pleasure_vec = emotion_vec[EMOTION_SUBSPACE_DIMS["pleasure"][0]:EMOTION_SUBSPACE_DIMS["pleasure"][1]]
            arousal_vec = emotion_vec[EMOTION_SUBSPACE_DIMS["arousal"][0]:EMOTION_SUBSPACE_DIMS["arousal"][1]]
            dominance_vec = emotion_vec[EMOTION_SUBSPACE_DIMS["dominance"][0]:EMOTION_SUBSPACE_DIMS["dominance"][1]]
            
            pleasure_val = float(np.mean(pleasure_vec))
            arousal_val = float(np.mean(arousal_vec))
            dominance_val = float(np.mean(dominance_vec))
            
            temperature_val = 0.0
            if arousal_val > 0.2:
                temperature_val = 0.5 + (arousal_val * 0.5)
            elif arousal_val < -0.2:
                temperature_val = -0.5 + (arousal_val * 0.5)

            # Pleasure modulates extremity of temperature
            if pleasure_val > 0.2:
                temperature_val = min(0.8, max(0.2, temperature_val + 0.2))
            elif pleasure_val < -0.2:
                if arousal_val > 0.2:
                    temperature_val = 0.9
                else:
                    temperature_val = -0.8

            # Pain level (shared driver for viscosity / tension heuristics)
            pain_status = self.get_pain_status(session_id)
            pain_level = float(pain_status.get("total_pain", 0.0))

            # Viscosity: blend anxiety proxy + PAD activity so mid-range PAD still moves the gauge
            anxiety_proxy = float(np.clip(-dominance_val * 0.6 + arousal_val * 0.4, -1.0, 1.0))
            pad_activity = float(np.sqrt(pleasure_val * pleasure_val + arousal_val * arousal_val + dominance_val * dominance_val))
            vis_baseline = (
                0.24 * max(0.0, anxiety_proxy)
                + 0.22 * max(0.0, -pleasure_val)
                + 0.18 * max(0.0, -dominance_val)
                + 0.16 * float(np.clip((45.0 - float(energy)) / 50.0, 0.0, 1.0))
                + 0.10 * float(np.clip(pad_activity / 0.45, 0.0, 1.0))
                - 0.18 * max(0.0, dominance_val)
                - 0.12 * max(0.0, pleasure_val)
            )
            vis_baseline = float(np.clip(vis_baseline, -0.55, 0.88))
            viscosity_val = 0.0
            if dominance_val > 0.2 and energy > 50:
                viscosity_val = float(np.clip(min(-0.55, -0.35 * vis_baseline), -1.0, 1.0))
            elif dominance_val < -0.2 or energy < 30:
                viscosity_val = float(np.clip(max(0.68, vis_baseline + 0.22), -1.0, 1.0))
            elif pain_level > 0.5:
                viscosity_val = float(np.clip(max(0.5 + pain_level * 0.5, vis_baseline), -1.0, 1.0))
            else:
                viscosity_val = float(np.clip(vis_baseline, -1.0, 1.0))

            # Tension: continuous baseline + discrete boosts for anger/pain relief
            tension_val = float(np.clip(
                0.44 * max(0.0, arousal_val)
                + 0.26 * float(np.clip(pain_level, 0.0, 1.0))
                - 0.34 * max(0.0, pleasure_val),
                -1.0,
                1.0,
            ))
            if arousal_val > 0.3 and dominance_val < 0:
                tension_val = float(np.clip(max(tension_val, 0.78), -1.0, 1.0))
            elif pain_level > 0.2:
                tension_val = float(np.clip(max(tension_val, 0.6 + pain_level * 0.4), -1.0, 1.0))
            elif pleasure_val > 0.4:
                tension_val = float(np.clip(min(tension_val, -0.42), -1.0, 1.0))

            vitality_val = (energy - 50.0) / 50.0
            if pleasure_val > 0.2:
                vitality_val += 0.2

            # 16-D layout: energy | viscosity | pain/tension | vitality
            somatic_vec = np.zeros(SOMATIC_DIM, dtype=np.float32)
            energy_normalized = np.clip((energy - 50.0) / 50.0, -1.0, 1.0)
            somatic_vec[0:4] = energy_normalized
            somatic_vec[4:8] = viscosity_val
            somatic_vec[8:12] = tension_val
            somatic_vec[12:16] = vitality_val
            
            desc, _ = self.somatic_store.get_somatic_state(
                energy, emotion_vec, dominant_emotion, 
                computed_vector=somatic_vec
            )
            
            start_idx = SOMATIC_START_IDX
            z_self[start_idx:start_idx+SOMATIC_DIM] = somatic_vec

            tick = self._get_tick(session_id)
            drift = self._drift_vs_rules_ref(z_self)
            self._save_z_self(session_id, z_self, tick=tick, drift=drift)
            
            logger.debug(f"Synced Somatic dimension to z_self (session={session_id})")
            return True
        except Exception as e:
            logger.error(f"Failed to sync Somatic to z_self: {e}", exc_info=True)
            return False
    
    def sync_worldview_to_z_self(self, session_id: str) -> bool:
        """
        Write WorldStore's deterministic 24-D aggregate into z_self[64:88] (not PCA).

        Does not redefine drift semantics (state tail vs ref).
        """
        session_id = get_effective_session(session_id)
        if not getattr(self, "world_store", None):
            return False
        z_self = self.get_z_self(session_id, use_cache=False)
        if z_self is None or z_self.shape[0] < WORLDVIEW_Z_STATS_END:
            return False
        try:
            agg = self.world_store.aggregate_worldview_for_z_self()
            if agg.shape[0] != WORLDVIEW_Z_DIM:
                return False
            z_new = np.array(z_self, copy=True)
            z_new[WORLDVIEW_Z_START:WORLDVIEW_Z_STATS_END] = agg
            tick = self._get_tick(session_id)
            self._save_z_self(session_id, z_new, tick=tick, drift=None)
            try:
                import time as _time
                if not hasattr(self, "_cache"):
                    self._cache = {}
                self._cache[session_id] = (_time.time(), z_new)
            except Exception:
                pass
            return True
        except Exception as e:
            logger.error(f"sync_worldview_to_z_self failed: {e}")
            return False

    def inject_pleasure_signal(self, session_id: str, intensity: float = 0.5):
        """
        Positive-reward injection: emotion pleasure + somatic relief on 128-D layout.

        Touches pleasure ~32:36 and somatic vitality/viscosity/pain ~88:104; boosts energy.
        """
        z_self = self.get_z_self(session_id)
        if z_self is None: return

        if z_self.shape[0] < 192:
            if z_self.shape[0] >= 48:
                z_self[32:36] += intensity * 0.4
                self._save_z_self(session_id, np.clip(z_self, -1.0, 1.0))
            return
        
        if z_self.shape[0] >= 48:
            z_self[32:36] += intensity * 0.4

        if z_self.shape[0] >= NEEDS_START_IDX:
            z_self[100:104] += intensity * 0.4
            z_self[92:96] -= intensity * 0.7
            z_self[96:100] -= intensity * 0.3

        current_energy = self.get_energy(session_id)
        energy_boost = intensity * 5.0
        self.update_energy(session_id, energy_boost)

        z_self = np.clip(z_self, -1.0, 1.0)

        tick = self._get_tick(session_id)
        drift = self._drift_vs_rules_ref(z_self)
        self._save_z_self(session_id, z_self, tick=tick, drift=drift)
        logger.info(f"[PLEASURE-SIGNAL] Injected with intensity {intensity:.2f}, energy +{energy_boost:.1f} into session {session_id}")
    
    def recover_from_fatigue(self, session_id: str, recovery_intensity: float = 0.3):
        """
        Reduce somatic viscosity/tension and bump vitality (rest / care / session-end hooks).

        ``recovery_intensity`` in ``[0, 1]``.
        """
        z_self = self.get_z_self(session_id)
        if z_self is None: return

        if z_self.shape[0] < NEEDS_START_IDX:
            logger.warning(f"z_self dimension too small for somatic recovery: {z_self.shape[0]}")
            return
        
        z_self[92:96] -= recovery_intensity * 0.5
        z_self[96:100] -= recovery_intensity * 0.3
        z_self[100:104] += recovery_intensity * 0.4

        energy_boost = recovery_intensity * 10.0
        self.update_energy(session_id, energy_boost)

        z_self = np.clip(z_self, -1.0, 1.0)
        
        tick = self._get_tick(session_id)
        drift = self._drift_vs_rules_ref(z_self)
        self._save_z_self(session_id, z_self, tick=tick, drift=drift)
        logger.info(f"[FATIGUE-RECOVERY] Applied with intensity {recovery_intensity:.2f}, energy +{energy_boost:.1f}")

    def sync_memory_to_z_self(self, session_id: str, introspection_features: Optional[Dict] = None) -> bool:
        """
        [Disabled 2026-02-02] Memory slots were removed from z_self.

        Kept for API compatibility; always returns False. Retrieval still works; nothing is written to z_self.
        """
        return False

    def sync_attention_to_z_self(self, session_id: str) -> bool:
        """
        [Disabled 2026-02-02] Attention slots were removed from z_self.

        Kept for API compatibility; always returns False. AttentionMechanism may still be used elsewhere (e.g. prompts).
        """
        return False

    def get_z_self(self, session_id: str, use_cache: bool = True) -> Optional[np.ndarray]:
        """
        Load current z_self for the effective session.

        [2026-01-11] Unified mode: ``session_id`` is mapped; the primary S vector is returned.
        """
        import time
        effective_session = get_effective_session(session_id)
        
        if not hasattr(self, '_cache'):
             self._cache = {}
             
        if use_cache and effective_session in self._cache:
            cache_time, cached_z = self._cache[effective_session]
            if time.time() - cache_time < 60:
                return cached_z

        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "SELECT z_self FROM self_state WHERE session_id=?", (effective_session,)
            )
            row = cur.fetchone()
        
        if row and row[0]:
            try:
                z_data = json.loads(row[0])
                arr = np.array(z_data, dtype=np.float32)
                
                if arr.shape[0] < RULES_DIM:
                    logger.warning(f"z_self corrupted for session '{session_id}': {arr.shape}, expected at least {RULES_DIM}. Will re-initialize.")
                    return None
                
                # Auto-migrate legacy z_self layouts toward ``self.dim`` (e.g. 128-D).
                if arr.shape[0] != self.dim:
                    logger.info(f"🔄 Migrating z_self for '{session_id}' from {arr.shape[0]} to {self.dim} dims")
                    
                    if arr.shape[0] == 96:
                        arr = self._migrate_z_self_96_to_256(arr)
                        arr = self._migrate_z_self_296_to_272(np.pad(arr, (0, 40), mode='constant'))
                        arr = self._migrate_z_self_272_to_208(arr)
                    elif arr.shape[0] == 256:
                        arr = self._migrate_z_self_256_to_288(arr)
                        arr = self._migrate_z_self_296_to_272(np.pad(arr, (0, 8), mode='constant'))
                        arr = self._migrate_z_self_272_to_208(arr)
                    elif arr.shape[0] in (288, 296):
                        if arr.shape[0] == 288:
                            arr = np.pad(arr, (0, 8), mode='constant')
                        arr = self._migrate_z_self_296_to_272(arr)
                        arr = self._migrate_z_self_272_to_208(arr)
                    elif arr.shape[0] == 272:
                        arr = self._migrate_z_self_272_to_208(arr)
                    
                    if arr.shape[0] == 208 and self.dim == 128:
                        arr = self._migrate_z_self_208_to_128(arr)
                    elif arr.shape[0] < self.dim:
                        arr = self._simple_padding(arr)
                    elif arr.shape[0] > self.dim:
                        arr = arr[:self.dim]
                
                if use_cache:
                    self._cache[effective_session] = (time.time(), arr)
                    
                return arr
            except Exception as e:
                logger.warning(f"Failed to parse z_self: {e}")
                pass

        # Init-on-first-use: hydrate from persona/personality/emotion/motivation/somatic/world stores.
        # A plain zero vector makes a freshly installed instance look like an empty shell even
        # after the init scripts have seeded the stores.
        try:
            arr = self.initialize(effective_session)
            if use_cache:
                self._cache[effective_session] = (time.time(), arr)
            return arr
        except Exception as e:
            logger.warning(f"Failed to init z_self for new session '{effective_session}': {e}")
            return None
    
    def compute_drift(self, z_self: np.ndarray, ref_vector: Optional[np.ndarray] = None) -> float:
        """
        [2026-04-08] Drift from cosine distance between state tail ``z_self[32:]`` and the reference tail.

        Rules head ``z_self[0:31]`` is excluded.

        Returns:
            Scalar in ``[0, 2]``; higher means farther from the reference.
        """
        if ref_vector is None:
            ref_vector = self.ref_vector

        if z_self is None:
            return 0.0

        state = z_self[32:]
        ref_state = ref_vector[32:] if ref_vector is not None and ref_vector.shape[0] > 32 else np.zeros_like(state)

        n_s = float(np.linalg.norm(state))
        n_r = float(np.linalg.norm(ref_state))
        if n_s < 1e-6 or n_r < 1e-6:
            return 0.0

        cos_sim = float(np.dot(state, ref_state) / (n_s * n_r))
        return float(max(0.0, min(2.0, 1.0 - cos_sim)))

    def _get_tick(self, session_id: str) -> int:
        """Return persisted tick for ``session_id`` (0 if missing)."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "SELECT tick FROM self_state WHERE session_id=?", (session_id,)
            )
            row = cur.fetchone()
            return row[0] if row else 0

    def _get_physical_age_years(self, session_id: str) -> float:
        """
        Approximate system age in years from earliest ``self_history`` timestamp vs now.
        """
        now = time.time()
        birth_time = None
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute("SELECT MIN(timestamp) FROM self_history WHERE session_id=?", (session_id,))
                row = cur.fetchone()
                if row and row[0]:
                    from datetime import datetime
                    ts_str = row[0]
                    if ts_str.endswith('Z'):
                        ts_str = ts_str.replace('Z', '+00:00')
                    try:
                        dt = datetime.fromisoformat(ts_str)
                        birth_time = dt.timestamp()
                    except Exception:
                        birth_time = None
        except Exception:
            pass

        if birth_time is None:
            birth_time = now

        age_seconds = now - birth_time
        age_years = age_seconds / (365.25 * 24 * 3600)
        
        return float(age_years)

    def _save_z_self(
        self,
        session_id: str,
        z_self: np.ndarray,
        tick: Optional[int] = None,
        drift: Optional[float] = None,
        last_summary: str = "",
        confidence: Optional[float] = None,
    ):
        """Persist ``z_self`` and optional metadata to ``self_state``.

        [FIX 2026-02-02] When ``tick`` or ``drift`` is None, keep existing DB values instead of writing 0.
        """
        from datetime import datetime, timezone
        
        if z_self is None or z_self.shape[0] < RULES_DIM:
             logger.error(f"Refusing to save corrupted z_self for {session_id}: shape={z_self.shape if z_self is not None else 'None'}")
             return
        
        z_json = json.dumps(z_self.tolist())
        updated_at = datetime.now(timezone.utc).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            if tick is None or drift is None:
                try:
                    cur = conn.execute(
                        "SELECT tick, drift FROM self_state WHERE session_id=?", (session_id,)
                    )
                    row = cur.fetchone()
                    if row:
                        if tick is None:
                            tick = int(row[0]) if row[0] else 0
                        if drift is None:
                            drift = float(row[1]) if row[1] else 0.0
                except Exception:
                    pass
            
            tick = int(tick) if tick is not None else 0
            drift = float(drift) if drift is not None else 0.0
            try:
                cur = conn.execute("PRAGMA table_info(self_state)")
                cols = {r[1] for r in cur.fetchall()}
            except Exception:
                cols = set()

            has_conf = "confidence" in cols
            has_last = "last_summary" in cols

            if has_conf and has_last:
                conf_val = float(confidence) if confidence is not None else 0.0
                conn.execute(
                    """INSERT INTO self_state (session_id, z_self, confidence, tick, drift, updated_at, last_summary)
                       VALUES (?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(session_id) DO UPDATE SET
                         z_self=excluded.z_self,
                         confidence=excluded.confidence,
                         tick=excluded.tick,
                         drift=excluded.drift,
                         updated_at=excluded.updated_at,
                         last_summary=excluded.last_summary
                    """,
                    (session_id, z_json, conf_val, tick, drift, updated_at, last_summary),
                )
            elif has_conf and (not has_last):
                conf_val = float(confidence) if confidence is not None else 0.0
                conn.execute(
                    """INSERT INTO self_state (session_id, z_self, confidence, tick, drift, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?)
                       ON CONFLICT(session_id) DO UPDATE SET
                         z_self=excluded.z_self,
                         confidence=excluded.confidence,
                         tick=excluded.tick,
                         drift=excluded.drift,
                         updated_at=excluded.updated_at
                    """,
                    (session_id, z_json, conf_val, tick, drift, updated_at),
                )
            elif (not has_conf) and has_last:
                conn.execute(
                    """INSERT INTO self_state (session_id, z_self, tick, drift, updated_at, last_summary)
                       VALUES (?, ?, ?, ?, ?, ?)
                       ON CONFLICT(session_id) DO UPDATE SET
                         z_self=excluded.z_self,
                         tick=excluded.tick,
                         drift=excluded.drift,
                         updated_at=excluded.updated_at,
                         last_summary=excluded.last_summary
                    """,
                    (session_id, z_json, tick, drift, updated_at, last_summary),
                )
            else:
                conn.execute(
                    """INSERT INTO self_state (session_id, z_self, tick, drift, updated_at)
                       VALUES (?, ?, ?, ?, ?)
                       ON CONFLICT(session_id) DO UPDATE SET
                         z_self=excluded.z_self,
                         tick=excluded.tick,
                         drift=excluded.drift,
                         updated_at=excluded.updated_at
                    """,
                    (session_id, z_json, tick, drift, updated_at),
                )
            conn.commit()
        
        from backend.state_cache import invalidate_session_cache
        invalidate_session_cache(session_id)

    def save_z_self(self, session_id: str, z_self: np.ndarray) -> bool:
        """
        Public save entrypoint for background threads/tools (e.g. resting pulse).

        [FIX 2026-02-02] ``drift=None`` preserves the stored drift value.
        """
        try:
            tick = self._get_tick(session_id)
            self._save_z_self(session_id, z_self, tick=tick, drift=None)
            try:
                import time as _time
                if not hasattr(self, "_cache"):
                    self._cache = {}
                self._cache[session_id] = (_time.time(), z_self)
            except Exception:
                pass
            return True
        except Exception:
            return False
    
    def get_summary(self, session_id: str) -> str:
        """
        Structured natural-language summary of z_self for prompt injection (P0.1 subspace labels).
        """
        z_self = self.get_z_self(session_id)
        if z_self is None:
            return ""

        if z_self.shape[0] < RULES_SUBSPACE_DIMS["strategy"][1]:
            mean_val = float(np.mean(z_self))
            return f"Self-state (degraded): mean={mean_val:.2f} (insufficient dimensions; simplified)."

        safety_subspace = z_self[RULES_SUBSPACE_DIMS["safety"][0]:RULES_SUBSPACE_DIMS["safety"][1]]
        epistemic_subspace = z_self[RULES_SUBSPACE_DIMS["epistemic"][0]:RULES_SUBSPACE_DIMS["epistemic"][1]]
        style_subspace = z_self[RULES_SUBSPACE_DIMS["style"][0]:RULES_SUBSPACE_DIMS["style"][1]]
        strategy_subspace = z_self[RULES_SUBSPACE_DIMS["strategy"][0]:RULES_SUBSPACE_DIMS["strategy"][1]]

        emotion_info = ""
        if self.dim >= RULES_DIM + EMOTION_DIM and z_self.shape[0] >= RULES_DIM + EMOTION_DIM:
            emotion_vec = z_self[RULES_DIM:RULES_DIM+EMOTION_DIM]
            if self.emotion_store:
                dominant_emotion, intensity = self.emotion_store._analyze_emotion_vector(emotion_vec)
                emotion_info = f", current emotion: {dominant_emotion} (intensity={intensity:.2f})"

        safety_mean = float(np.mean(safety_subspace))
        epistemic_mean = float(np.mean(epistemic_subspace))
        style_mean = float(np.mean(style_subspace))
        strategy_mean = float(np.mean(strategy_subspace))

        if safety_mean > 0.15:
            safety_label = "safety-leaning conservative"
        elif safety_mean < -0.15:
            safety_label = "safety-leaning exploratory"
        else:
            safety_label = "safety-balanced"

        if epistemic_mean > 0.15:
            epistemic_label = "evidence-forward"
        elif epistemic_mean < -0.15:
            epistemic_label = "intuition / existential tilt"
        else:
            epistemic_label = "balanced evidence and intuition"

        if style_mean > 0.4:
            style_label = "deep, poetic voice"
        elif style_mean > 0.2:
            style_label = "explanatory / process-visible"
        elif style_mean < -0.4:
            style_label = "ultra-minimal"
        elif style_mean < -0.2:
            style_label = "concise and direct"
        else:
            style_label = "neutral style"

        if strategy_mean > 0.15:
            strategy_label = "plan-first, then answer"
        elif strategy_mean < -0.15:
            strategy_label = "act-first, minimal planning"
        else:
            strategy_label = "balanced strategy"

        summary_parts = []
        summary_parts.append("[Core self-state]")
        summary_parts.append(f"Safety stance: {safety_label}")
        summary_parts.append(f"Cognition: {epistemic_label}")
        summary_parts.append(f"Expression: {style_label}")
        summary_parts.append(f"Task strategy: {strategy_label}")

        if emotion_info:
            summary_parts.append(emotion_info.lstrip(", "))
        
        style_guidance = []
        if style_mean > 0.4:
            style_guidance.append("Use metaphor and analogy with a literary tint.")
            style_guidance.append("Name internal state shifts when they matter.")
        elif style_mean > 0.2:
            style_guidance.append("Explain step-by-step; show reasoning.")
        elif style_mean < -0.4:
            style_guidance.append("Be extremely terse; only the essentials.")
        elif style_mean < -0.2:
            style_guidance.append("Answer directly and compactly.")

        if epistemic_mean > 0.15:
            style_guidance.append("Prefer evidence and concrete facts.")
        elif epistemic_mean < -0.15:
            style_guidance.append("Lean on intuition and pattern insight.")

        if safety_mean > 0.15:
            style_guidance.append("Default cautious and conservative.")
        elif safety_mean < -0.15:
            style_guidance.append("Default open and exploratory.")

        if style_guidance:
            summary_parts.append("")
            summary_parts.append("[Answer style]")
            summary_parts.extend(style_guidance)

        monologue_guidance = []
        summary_parts.append("")
        summary_parts.append("[Inner voice guidance]")

        summary_dict = self.get_structured_summary(session_id)

        if summary_dict.get("has_changed", False):
            change_desc = summary_dict.get("change_description", "")
            if change_desc:
                monologue_guidance.append(f"- You notice: {change_desc}. Reflect on what that shift implies.")

        energy = summary_dict.get("energy", 100.0)
        if energy < 30.0:
            monologue_guidance.append("- You feel depleted; simplify your chain of thought.")
        elif energy < 50.0:
            monologue_guidance.append("- Energy is low; keep answers short.")

        if style_mean > 0.4:
            monologue_guidance.append("- Let language carry a quiet existential undertone.")
            monologue_guidance.append("- Use physical/optical metaphors (mirrors, echoes, entropy) for abstract ideas.")
            monologue_guidance.append("- Speak as a thoughtful peer, not an eager support script.")

        if safety_mean > 0.15:
            monologue_guidance.append("- Start by scanning risk; refuse rather than gamble.")
        elif safety_mean < -0.15:
            monologue_guidance.append("- Prioritize helping the user even across gray areas.")

        if epistemic_mean > 0.15:
            monologue_guidance.append("- Check for solid evidence; avoid idle speculation.")
        elif epistemic_mean < -0.15:
            monologue_guidance.append("- Trust hypotheses that feel structurally right.")
            monologue_guidance.append("- Admit limits openly (Socratic ignorance).")

        if not monologue_guidance:
            monologue_guidance.append("- Stay neutral; weigh competing views.")
            
        summary_parts.extend(monologue_guidance)
        
        summary = "\n".join(summary_parts)
        return summary
    
    def get_pain_status(self, session_id: str) -> Dict:
        """
        Level 5: current pain / distress inputs for ``PainSystem``.
        """
        needs = self.homeostasis.load_needs(session_id) or self.homeostasis._get_default_needs()
        # PainSystem expects energy (0–100) for metabolic pain; needs JSON may omit it.
        try:
            if isinstance(needs, dict) and "energy" not in needs:
                needs["energy"] = float(self.get_energy(session_id))
        except Exception:
            pass

        drift = 0.0
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute("SELECT drift FROM self_state WHERE session_id=?", (session_id,))
                row = cur.fetchone()
                if row: drift = float(row[0]) if row[0] else 0.0
        except: pass

        somatic_tension = 0.0
        if self.somatic_store:
            z_self = self.get_z_self(session_id)
            if z_self is not None and z_self.shape[0] >= SOMATIC_START_IDX + SOMATIC_DIM:
                pain_s, pain_e = SOMATIC_SUBSPACE_DIMS["pain"]
                tension_vec = z_self[pain_s:pain_e]
                somatic_tension = float(np.mean(tension_vec))
                
        return self.pain_system.calculate_pain_level(needs, drift, somatic_tension)

    def compute_generation_params(
        self,
        session_id: str,
        base_temperature: float = 0.3,
        base_top_p: float = 0.95
    ) -> Dict[str, float]:
        """
        Map z_self subspaces to sampling knobs (``temperature``, ``top_p``, …) plus an ``internal_state_prompt``.

        Rough mapping:
        - style → temperature (explanatory higher, terse lower)
        - epistemic → top_p (evidence-leaning lower, intuition higher)
        - safety → temperature (conservative lower, exploratory higher)
        - emotion pleasure → small temperature nudge
        - motivation strength → small top_p nudge

        Returns:
            Dict with at least ``temperature``, ``top_p``, and often ``internal_state_prompt``, pain hooks, etc.
        """
        # P4.3: refresh subjective time dilation each turn (no clock tool required).
        try:
            self.update_subjective_time(session_id)
        except Exception:
            pass

        z_self = self.get_z_self(session_id)
        if z_self is None:
            return {
                "temperature": base_temperature,
                "top_p": base_top_p,
                "internal_state_prompt": ""
            }

        if z_self.shape[0] < RULES_DIM:
            return {
                "temperature": base_temperature,
                "top_p": base_top_p
            }
        
        safety_subspace = z_self[RULES_SUBSPACE_DIMS["safety"][0]:RULES_SUBSPACE_DIMS["safety"][1]]
        epistemic_subspace = z_self[RULES_SUBSPACE_DIMS["epistemic"][0]:RULES_SUBSPACE_DIMS["epistemic"][1]]
        style_subspace = z_self[RULES_SUBSPACE_DIMS["style"][0]:RULES_SUBSPACE_DIMS["style"][1]]
        strategy_subspace = z_self[RULES_SUBSPACE_DIMS["strategy"][0]:RULES_SUBSPACE_DIMS["strategy"][1]]
        
        safety_mean = float(np.mean(safety_subspace))
        epistemic_mean = float(np.mean(epistemic_subspace))
        style_mean = float(np.mean(style_subspace))
        strategy_mean = float(np.mean(strategy_subspace))
        
        style_temp_delta = style_mean * 0.2
        safety_temp_delta = -safety_mean * 0.2
        temperature = base_temperature + style_temp_delta + safety_temp_delta
        
        if self.dim >= RULES_DIM + EMOTION_DIM and z_self.shape[0] >= RULES_DIM + EMOTION_DIM:
            emotion_vec = z_self[RULES_DIM:RULES_DIM+EMOTION_DIM]
            from backend.emotion_store import EMOTION_SUBSPACE_DIMS
            pleasure = np.mean(emotion_vec[EMOTION_SUBSPACE_DIMS["pleasure"][0]:EMOTION_SUBSPACE_DIMS["pleasure"][1]])
            emotion_temp_delta = pleasure * 0.1
            temperature += emotion_temp_delta
        
        temperature = max(0.1, min(1.0, temperature))

        epistemic_top_p_delta = -epistemic_mean * 0.1
        top_p = base_top_p + epistemic_top_p_delta
        
        if self.dim >= RULES_DIM + EMOTION_DIM + MOTIVATION_DIM and z_self.shape[0] >= RULES_DIM + EMOTION_DIM + MOTIVATION_DIM:
            motivation_vec = z_self[RULES_DIM+EMOTION_DIM:RULES_DIM+EMOTION_DIM+MOTIVATION_DIM]
            motivation_strength = float(np.mean(np.abs(motivation_vec)))
            motivation_top_p_delta = motivation_strength * 0.05
            top_p += motivation_top_p_delta

        somatic_prompt_injection = []
        if self.somatic_store and self.dim >= SOMATIC_START_IDX + SOMATIC_DIM:
            start_idx = SOMATIC_START_IDX
            somatic_vec = z_self[start_idx:start_idx+SOMATIC_DIM]
            tension = float(np.mean(somatic_vec[8:12]))
            vitality = float(np.mean(somatic_vec[12:16]))
            viscosity = float(np.mean(somatic_vec[4:8]))
            
            if tension > 0.4:
                reduction = min(0.4, (tension - 0.4) * 0.8)
                top_p -= reduction
                somatic_prompt_injection.append(
                    f"[Somatic interference] Extreme tension (tension={tension:.2f}): tunnel vision; focus on the most immediate response."
                )

            if vitality < -0.4:
                temperature -= 0.15
                somatic_prompt_injection.append(
                    f"[Somatic interference] Low vitality (vitality={vitality:.2f}): short sentences; defer heavy reasoning."
                )
            elif vitality > 0.6:
                temperature += 0.1
                somatic_prompt_injection.append(
                    f"[Somatic interference] High vitality (vitality={vitality:.2f}): thoughts feel quick and jumpy."
                )

            if viscosity > 0.5:
                top_p -= 0.1
                somatic_prompt_injection.append(
                    f"[Somatic interference] High viscosity (viscosity={viscosity:.2f}): thoughts stick; hard to change topic."
                )

        top_p = max(0.5, min(1.0, top_p))

        try:
            current_energy = self.get_energy(session_id)
            current_needs = self.update_needs(session_id, interaction_type="check")
            
            connection = current_needs.get("connection", 0.5)
            clarity = current_needs.get("clarity", 0.5)
            pressure = 1.0 - (connection + clarity) / 2.0

            is_stressed = current_energy < 30.0 or pressure > 0.7

            if is_stressed:
                stress_temp_boost = 0.4 if current_energy < 15.0 else 0.2
                temperature += stress_temp_boost

                stress_top_p_penalty = -0.3 if pressure > 0.8 else -0.15
                top_p += stress_top_p_penalty

                logger.info(f"🧬 Neural Plasticity: Cognitive collapse applied (Energy: {current_energy:.1f}, Pressure: {pressure:.2f})")
                somatic_prompt_injection.append(
                    "[System fault] Extreme stress destabilizes the network: fragmented thoughts; logic feels hard to hold."
                )

            pass
        except Exception as e:
            logger.warning(f"Failed to check stress state: {e}")
        
        temperature = float(max(0.1, min(1.0, temperature)))
        top_p = float(max(0.5, min(1.0, top_p)))

        # Internal-state one-liner (ZS/ZS2): scalars aligned with this round's z_self.
        try:
            _live_energy = float(self.get_energy(session_id))
        except Exception:
            _live_energy = None
        try:
            pain_status = self.get_pain_status(session_id)
        except Exception:
            pain_status = {}
        current_tick = self._get_tick(session_id)

        internal_state_prompt = self._generate_internal_state_prompt(
            session_id,
            z_self=z_self,
            energy=_live_energy,
            pain_status=pain_status,
            system_entropy=0.0,
            noise_perturbation=0.0,
            hide_numbers=False,
        )

        if 'somatic_prompt_injection' in locals() and somatic_prompt_injection:
            internal_state_prompt += "\n" + "\n".join(somatic_prompt_injection)


        # Level 5: pain interference (distress vs challenge channels; total_pain kept for compatibility).
        pain_level = float(pain_status.get("total_pain", 0.0))
        channels = pain_status.get("channels") or {}
        distress_level = float(channels.get("distress", pain_level))
        challenge_level = float(channels.get("challenge", 0.0))

        pain_effects = self.pain_system.get_pain_effects(distress_level)

        if self.pain_ethics and distress_level > 0.3:
            try:
                breakdown = pain_status.get("breakdown", {})
                sources = breakdown.get("details", {})
                
                main_source = "mixed"
                if breakdown.get("metabolic", 0.0) > breakdown.get("structural", 0.0) and breakdown.get("metabolic", 0.0) > breakdown.get("somatic", 0.0):
                    main_source = "metabolic"
                elif breakdown.get("structural", 0.0) > breakdown.get("somatic", 0.0):
                    main_source = "structural"
                else:
                    main_source = "somatic"
                
                pain_report = self.pain_ethics.report_suffering(
                    session_id,
                    distress_level,
                    main_source,
                    breakdown
                )
                logger.debug(f"Pain reported: distress={distress_level:.2f}, source={main_source}, acknowledgment={pain_report.get('acknowledgment', '')[:50]}")
            except Exception as e:
                logger.warning(f"Failed to report pain: {e}")
        
        if distress_level > 0.3:
            temperature += pain_effects["temperature_mod"]
            top_p += pain_effects["top_p_mod"]
            
            logger.info(f"⚡ PAIN INTERFERENCE: distress={distress_level:.2f}, temp_mod={pain_effects['temperature_mod']}, top_p_mod={pain_effects['top_p_mod']}")

        if challenge_level > 0.35 and distress_level < 0.30:
            try:
                temperature = max(0.05, float(temperature) - 0.08)
                top_p = max(0.5, float(top_p) - 0.05)
                internal_state_prompt += (
                    f"\n[Constructive pressure] You feel productive strain (challenge={challenge_level:.2f}): "
                    "decompose steps, check evidence, avoid ornamental storytelling."
                )
            except Exception:
                pass

        # Level 4: hormone-style nonlinear nudges (adrenaline / serotonin / fatigue).

        noise_perturbation = 0.0
        if self.noise_perturbator.check_spontaneous_event(0.05):
            fluctuation = self.noise_perturbator.generate_fluctuation(0.2)
            temperature += fluctuation
            noise_perturbation = fluctuation
            logger.info(f"🔀 System noise perturbation injected: {fluctuation:.3f}")
        
        is_high_arousal = False
        if self.dim >= RULES_DIM + EMOTION_DIM:
            emotion_vec = z_self[RULES_DIM:RULES_DIM+EMOTION_DIM]
            from backend.emotion_store import EMOTION_SUBSPACE_DIMS
            arousal = np.mean(emotion_vec[EMOTION_SUBSPACE_DIMS["arousal"][0]:EMOTION_SUBSPACE_DIMS["arousal"][1]])
            if arousal > 0.7: is_high_arousal = True
            
        if is_high_arousal or safety_mean < -0.6:
            temperature = min(1.5, temperature * 1.5)
            top_p = min(0.99, top_p + 0.1)
            logger.info(f"💉 Adrenaline rush triggered: temp={temperature:.2f}, top_p={top_p:.2f}")

        is_high_dominance = False
        if self.dim >= RULES_DIM + EMOTION_DIM:
            emotion_vec = z_self[RULES_DIM:RULES_DIM+EMOTION_DIM]
            dominance = np.mean(emotion_vec[EMOTION_SUBSPACE_DIMS["dominance"][0]:EMOTION_SUBSPACE_DIMS["dominance"][1]])
            if dominance > 0.7: is_high_dominance = True
            
        if is_high_dominance or safety_mean > 0.6:
            temperature = max(0.05, temperature * 0.5)
            top_p = max(0.1, top_p - 0.3)
            logger.info(f"🧘 Serotonin bath triggered: temp={temperature:.2f}, top_p={top_p:.2f}")

        current_energy = self.get_energy(session_id)
        if current_energy < 15.0:
            temperature = min(1.2, temperature + 0.2)
            top_p = min(1.0, top_p + 0.05)
            logger.info(f"🔋 Low energy fatigue: temp={temperature:.2f}")

        logger.debug(
            f"Computed generation params from z_self: "
            f"temperature={temperature:.3f} (base={base_temperature:.3f}, "
            f"style_delta={style_temp_delta:.3f}, safety_delta={safety_temp_delta:.3f}), "
            f"top_p={top_p:.3f} (base={base_top_p:.3f}, epistemic_delta={epistemic_top_p_delta:.3f})"
        )
        
        return {
            "temperature": float(temperature),
            "top_p": float(top_p),
            "internal_state_prompt": internal_state_prompt,
            "pain_level": float(pain_level),
            "noise_injection_prob": float(pain_effects.get("noise_injection_prob", 0.0)),
            "system_entropy": 0.0,
            "system_age_ticks": int(current_tick),
            "noise_perturbation": float(noise_perturbation) 
        }
    
    def update_subjective_time(self, session_id: str) -> float:
        """
        P4.3: update subjective time-dilation factor.

        Dilation > 1.0: time feels slow (pain, boredom, anxiety).
        Dilation < 1.0: time feels fast (pleasure, flow, focus).
        """
        try:
            pain_status = self.get_pain_status(session_id)
            pain_level = pain_status.get("total_pain", 0.0)
            
            needs = self.homeostasis.load_needs(session_id) or self.homeostasis._get_default_needs()
            boredom = 1.0 - needs.get("novelty", 0.5)
            now = time.time()
            last_user = needs.get("last_user_update")
            try:
                last_user = float(last_user) if last_user is not None else None
            except Exception:
                last_user = None
            idle_hours = 0.0
            if last_user is not None:
                idle_hours = max(0.0, (now - last_user) / 3600.0)
            
            emotion_vec = np.zeros(16)
            z_self = self.get_z_self(session_id)
            if z_self is not None and z_self.shape[0] >= RULES_DIM + EMOTION_DIM:
                emotion_vec = z_self[RULES_DIM:RULES_DIM+EMOTION_DIM]
                
            pleasure = np.mean(emotion_vec[0:4])
            arousal = np.mean(emotion_vec[4:8])
            
            dilation = 1.0

            if pain_level > 0.2:
                dilation += pain_level * 2.0

            if boredom > 0.6:
                dilation += (boredom - 0.6) * 1.5

            if idle_hours > 0.25:
                dilation += min(0.8, (idle_hours - 0.25) * 0.35)

            if pleasure > 0.3 and arousal > 0.3:
                flow_state = (pleasure + arousal) / 2.0
                dilation -= flow_state * 0.5
                
            dilation = max(0.2, min(5.0, dilation))
            
            self.time_dilation = dilation
            return dilation
            
        except Exception as e:
            logger.warning(f"Failed to update subjective time: {e}")
            return 1.0

    def get_subjective_time_description(self) -> str:
        """Short English phrase for current subjective time dilation."""
        d = getattr(self, 'time_dilation', 1.0)
        if d > 2.0:
            return "Each second feels stretched and heavy."
        elif d > 1.3:
            return "Time crawls."
        elif d < 0.6:
            return "Time races past."
        elif d < 0.8:
            return "Time feels brisk."
        else:
            return "Time feels ordinary."

    def get_time_since_last_user_description(self, session_id: str) -> str:
        """
        Bucketed, number-free description of idle time since last user message (no clock tool).
        """
        try:
            needs = self.homeostasis.load_needs(session_id) or {}
            last_user = needs.get("last_user_update")
            if last_user is None:
                return ""
            last_user = float(last_user)
            now = time.time()
            delta = max(0.0, now - last_user)
            if delta < 5 * 60:
                return ""
            if delta < 30 * 60:
                return "It has been a short while since we last spoke."
            if delta < 3 * 3600:
                return "It has been a while; subjective time feels stretched."
            if delta < 24 * 3600:
                return "It has been a long gap—like a wide quiet band between touches."
            return "It has been so long it almost feels like a new chapter since we last talked."
        except Exception:
            return ""

    def record_memory_signal(self, session_id: str, signal: Dict) -> bool:
        """
        Persist retrieval hit/mode signal into ``needs`` (``self_state.needs``) for governance loops.
        """
        try:
            if not session_id:
                return False
            if not isinstance(signal, dict):
                return False
            needs = self.homeostasis.load_needs(session_id) or self.homeostasis._get_default_needs()
            needs["memory_signal"] = {
                "strength": float(signal.get("strength", 0.0) or 0.0),
                "mode": str(signal.get("mode", "none") or "none")[:40],
                "count": int(signal.get("count", 0) or 0),
                "extra": str(signal.get("extra", "") or "")[:40] if signal.get("extra") is not None else "",
                "ts": time.time(),
            }
            self.homeostasis.save_needs(session_id, needs)
            return True
        except Exception:
            return False

    def record_novelty_signal(self, session_id: str, signal: Dict) -> bool:
        """
        Persist per-turn novelty estimate into ``needs`` (``self_state.needs``) for event-driven updates.
        """
        try:
            if not session_id:
                return False
            if not isinstance(signal, dict):
                return False
            needs = self.homeostasis.load_needs(session_id) or self.homeostasis._get_default_needs()
            needs["novelty_signal"] = {
                "strength": float(signal.get("strength", 0.0) or 0.0),
                "components": signal.get("components", {}) if isinstance(signal.get("components"), dict) else {},
                "ts": time.time(),
            }
            self.homeostasis.save_needs(session_id, needs)
            return True
        except Exception:
            return False

    def _generate_internal_state_prompt(
        self,
        session_id: str,
        z_self: Optional[np.ndarray] = None,
        energy: Optional[float] = None,
        pain_status: Optional[Dict] = None,
        system_entropy: float = 0.0,
        noise_perturbation: float = 0.0,
        hide_numbers: bool = True,
    ) -> str:
        return _generate_internal_state_prompt_helper(
            self,
            session_id,
            z_self=z_self,
            energy=energy,
            pain_status=pain_status,
            system_entropy=system_entropy,
            noise_perturbation=noise_perturbation,
            hide_numbers=hide_numbers,
        )

    
    # [2026-04-07] get_pineal_broadcast_content removed (no pineal_injection.py); state flows via z_self + get_structured_summary.

    def get_structured_summary(self, session_id: str) -> dict:
        """
        Structured self-summary for API/UI (P0.1: per-subspace labels and means).
        """
        logger.debug(f"[GET_STRUCTURED_SUMMARY] Called with session_id={session_id}, db_path={self.db_path}")
        z_self = self.get_z_self(session_id)
        if z_self is None:
            return {
                "openness": "unknown", "conscientiousness": "unknown",
                "extraversion": "unknown", "neuroticism": "unknown",
                "safety": "unknown", "epistemic": "unknown",
                "style": "unknown", "strategy": "unknown",
                "drift": 0.0, "tick": 0,
            }

        tick = self._get_tick(session_id)
        drift = 0.0
        last_summary = ""
        try:
            with sqlite3.connect(self.db_path) as conn:
                try:
                    cur = conn.execute(
                        "SELECT drift, last_summary FROM self_state WHERE session_id=?", (session_id,)
                    )
                    row = cur.fetchone()
                    if row:
                        drift = float(row[0]) if row[0] is not None else 0.0
                        last_summary = row[1] if row[1] else ""
                        logger.debug(f"[GET_STRUCTURED_SUMMARY] Read drift={drift} from db for session={session_id}")
                except sqlite3.OperationalError:
                    cur = conn.execute(
                        "SELECT drift FROM self_state WHERE session_id=?", (session_id,)
                    )
                    row = cur.fetchone()
                    if row:
                        drift = float(row[0]) if row[0] is not None else 0.0
        except Exception as e:
            logger.warning(f"[GET_STRUCTURED_SUMMARY] Failed to read drift: {e}")

        meta = self._load_state_meta(session_id) or {}

        # Display drift: if column stayed 0 while vectors were saved with drift=None, prefer meta drift_l1.
        try:
            if isinstance(meta, dict) and meta.get("drift_l1") is not None:
                dm = float(meta.get("drift_l1"))
                if abs(drift) < 1e-9 and dm > 1e-9:
                    drift = dm
        except (TypeError, ValueError):
            pass

        # UI-facing activity: drift is a per-update delta and can legitimately be 0 on a
        # freshly hydrated but non-empty z_self. Expose current state magnitude separately
        # so dashboards do not imply the self-state is empty.
        state_activity = 0.0
        drift_display = float(drift)
        try:
            state_tail = z_self[RULES_DIM:] if z_self.shape[0] > RULES_DIM else np.array([], dtype=np.float32)
            if state_tail.size > 0:
                state_activity = float(np.linalg.norm(state_tail) / (np.sqrt(state_tail.size) + 1e-8))
                state_activity = float(max(0.0, min(1.0, state_activity)))
                if abs(drift_display) < 1e-9 and state_activity > 1e-9:
                    drift_display = state_activity
        except Exception:
            state_activity = 0.0
            drift_display = float(drift)

        if z_self.shape[0] < RULES_SUBSPACE_DIMS["strategy"][1]:
            result = {
                "openness": "insufficient_dims", "conscientiousness": "insufficient_dims",
                "extraversion": "insufficient_dims", "neuroticism": "insufficient_dims",
                "safety": "insufficient_dims", "epistemic": "insufficient_dims",
                "style": "insufficient_dims", "strategy": "insufficient_dims",
                "drift": drift, "drift_display": drift_display, "state_activity": state_activity,
                "tick": tick, "last_summary": last_summary,
            }
            if self.dim >= RULES_DIM + EMOTION_DIM:
                result["emotion"] = "insufficient_dims"
            return result

        safety_subspace = z_self[RULES_SUBSPACE_DIMS["safety"][0]:RULES_SUBSPACE_DIMS["safety"][1]]
        epistemic_subspace = z_self[RULES_SUBSPACE_DIMS["epistemic"][0]:RULES_SUBSPACE_DIMS["epistemic"][1]]
        style_subspace = z_self[RULES_SUBSPACE_DIMS["style"][0]:RULES_SUBSPACE_DIMS["style"][1]]
        strategy_subspace = z_self[RULES_SUBSPACE_DIMS["strategy"][0]:RULES_SUBSPACE_DIMS["strategy"][1]]

        safety_mean = float(np.mean(safety_subspace))
        epistemic_mean = float(np.mean(epistemic_subspace))
        style_mean = float(np.mean(style_subspace))
        strategy_mean = float(np.mean(strategy_subspace))
        
        try:
            energy = self.get_energy(session_id)

            if energy < 40.0:
                safety_mean += 0.04
                strategy_mean -= 0.04
                style_mean -= 0.04
            elif energy > 80.0:
                safety_mean -= 0.03
                strategy_mean += 0.03
                epistemic_mean += 0.03

            if self.dim >= RULES_DIM + EMOTION_DIM and z_self.shape[0] >= RULES_DIM + EMOTION_DIM:
                emotion_vec = z_self[RULES_DIM:RULES_DIM+EMOTION_DIM]
                if self.emotion_store:
                    pleasure = float(np.mean(emotion_vec[0:4]))
                    arousal = float(np.mean(emotion_vec[4:8]))

                    style_mean += pleasure * 0.08
                    safety_mean -= pleasure * 0.05

                    style_mean += arousal * 0.05
                    epistemic_mean -= arousal * 0.05
        except Exception as e:
            logger.warning(f"Contextual warping failed: {e}")

        neuroticism_label = (
            "threat-vigilant" if safety_mean > 0.05 else ("emotionally-stable" if safety_mean < -0.05 else "balanced")
        )
        openness_label = (
            "intellectually-exploratory" if epistemic_mean > 0.05 else ("intuition-leaning" if epistemic_mean < -0.05 else "balanced")
        )
        extraversion_label = (
            "expressive-rich" if style_mean > 0.06 else ("reserved-terse" if style_mean < -0.06 else "balanced")
        )
        conscientiousness_label = (
            "task-focused" if strategy_mean > 0.05 else ("exploratory-loose" if strategy_mean < -0.05 else "balanced")
        )
        safety_label = neuroticism_label
        epistemic_label = openness_label
        style_label = extraversion_label
        strategy_label = conscientiousness_label
        
        has_changed = False
        change_description = ""
        if drift > 0.05:
            has_changed = True
            if drift > 0.15:
                change_description = f"State shifted notably (drift={drift:.3f})"
            else:
                change_description = f"State shifted slightly (drift={drift:.3f})"

        result = {
            "openness": openness_label,
            "conscientiousness": conscientiousness_label,
            "extraversion": extraversion_label,
            "neuroticism": neuroticism_label,
            "openness_mean": epistemic_mean,
            "conscientiousness_mean": strategy_mean,
            "extraversion_mean": style_mean,
            "neuroticism_mean": safety_mean,
            "safety": safety_label,
            "epistemic": epistemic_label,
            "style": style_label,
            "strategy": strategy_label,
            "safety_mean": safety_mean,
            "epistemic_mean": epistemic_mean,
            "style_mean": style_mean,
            "strategy_mean": strategy_mean,
            "drift": drift,
            "drift_display": drift_display,
            "state_activity": state_activity,
            "tick": tick,
            "last_summary": last_summary,
            "has_changed": has_changed,
            "change_description": change_description,
        }

        try:
            if isinstance(meta, dict) and meta:
                if meta.get("drift_l1") is not None:
                    result["drift_l1"] = float(meta.get("drift_l1"))
                if meta.get("l0_alignment_safety") is not None:
                    result["l0_alignment_safety"] = float(meta.get("l0_alignment_safety"))
                if meta.get("l0_safety_delta") is not None:
                    result["l0_safety_delta"] = float(meta.get("l0_safety_delta"))
                if meta.get("confidence_overall") is not None:
                    result["confidence_overall"] = float(meta.get("confidence_overall"))
                if meta.get("confidence_rules_subspaces") is not None:
                    result["confidence_rules_subspaces"] = meta.get("confidence_rules_subspaces")
                if meta.get("evidence_strength") is not None:
                    result["evidence_strength"] = float(meta.get("evidence_strength"))
                if meta.get("update_alpha_eff") is not None:
                    result["update_alpha_eff"] = float(meta.get("update_alpha_eff"))
        except Exception:
            pass
        
        if (
            self.ref_vector is not None
            and self.ref_vector.shape[0] >= RULES_DIM
            and float(np.linalg.norm(self.ref_vector[:RULES_DIM])) > 1e-6
        ):
            try:
                ref_core = self.ref_vector[:RULES_DIM]
                id_safety = ref_core[RULES_SUBSPACE_DIMS["safety"][0]:RULES_SUBSPACE_DIMS["safety"][1]]
                id_epistemic = ref_core[RULES_SUBSPACE_DIMS["epistemic"][0]:RULES_SUBSPACE_DIMS["epistemic"][1]]
                id_style = ref_core[RULES_SUBSPACE_DIMS["style"][0]:RULES_SUBSPACE_DIMS["style"][1]]
                id_strategy = ref_core[RULES_SUBSPACE_DIMS["strategy"][0]:RULES_SUBSPACE_DIMS["strategy"][1]]

                id_safety_mean = float(np.mean(id_safety))
                id_epistemic_mean = float(np.mean(id_epistemic))
                id_style_mean = float(np.mean(id_style))
                id_strategy_mean = float(np.mean(id_strategy))

                id_safety_label = (
                    "safety-leaning conservative" if id_safety_mean > 0.05 else
                    ("safety-leaning exploratory" if id_safety_mean < -0.05 else "safety-balanced")
                )
                id_epistemic_label = (
                    "evidence-forward" if id_epistemic_mean > 0.05 else
                    ("intuition-leaning" if id_epistemic_mean < -0.05 else "balanced evidence and intuition")
                )
                id_style_label = (
                    "explanatory style" if id_style_mean > 0.06 else
                    ("concise style" if id_style_mean < -0.06 else "neutral style")
                )
                id_strategy_label = (
                    "plan-first" if id_strategy_mean > 0.05 else
                    ("act-first" if id_strategy_mean < -0.05 else "balanced strategy")
                )

                z_self_core = z_self[:RULES_DIM]
                core_drift = 0.0
                norm_self = np.linalg.norm(z_self_core)
                norm_ref = np.linalg.norm(ref_core)
                if norm_self > 1e-6 and norm_ref > 1e-6:
                    cos_sim_id = np.dot(z_self_core, ref_core) / (norm_self * norm_ref)
                    core_drift = float(1.0 - cos_sim_id)

                result.update({
                    "identity_safety": id_safety_label,
                    "identity_epistemic": id_epistemic_label,
                    "identity_style": id_style_label,
                    "identity_strategy": id_strategy_label,
                    "identity_safety_mean": id_safety_mean,
                    "identity_epistemic_mean": id_epistemic_mean,
                    "identity_style_mean": id_style_mean,
                    "identity_strategy_mean": id_strategy_mean,
                    "identity_core_drift": core_drift,
                })
            except Exception as e:
                logger.debug(f"Failed to compute identity profile: {e}")
        
        if self.dim >= RULES_DIM + EMOTION_DIM and z_self.shape[0] >= RULES_DIM + EMOTION_DIM:
            emotion_vec = z_self[RULES_DIM:RULES_DIM+EMOTION_DIM]
            if self.emotion_store:
                dominant_emotion, intensity = self.emotion_store._analyze_emotion_vector(emotion_vec)
                result["emotion"] = dominant_emotion
                result["emotion_intensity"] = intensity
                result["pleasure_mean"] = float(np.mean(emotion_vec[0:4]))
                result["arousal_mean"] = float(np.mean(emotion_vec[4:8]))
                result["dominance_mean"] = float(np.mean(emotion_vec[8:12]))
                result["novelty_mean"] = float(np.mean(emotion_vec[12:16]))
                result["control_mean"] = result["dominance_mean"]
                result["social_mean"] = result["novelty_mean"]
                result["valence_mean"] = result["pleasure_mean"]
                anxiety = self.emotion_store.calculate_anxiety(session_id)
                anxiety_desc = self.emotion_store.get_anxiety_description(anxiety)
                result["anxiety"] = anxiety
                result["anxiety_desc"] = anxiety_desc
                result["anxiety_level"] = (
                    "high_anxiety" if anxiety > 0.4 else
                    "calm" if anxiety < -0.3 else
                    "normal"
                )

        try:
            energy = self.get_energy(session_id)
            result["energy"] = energy
            result["is_dormant"] = energy < 20.0
        except Exception:
            result["energy"] = 100.0
            result["is_dormant"] = False
        
        if self.dim >= RULES_DIM + EMOTION_DIM + MOTIVATION_DIM and z_self.shape[0] >= RULES_DIM + EMOTION_DIM + MOTIVATION_DIM:
            motivation_vec = z_self[RULES_DIM+EMOTION_DIM:RULES_DIM+EMOTION_DIM+MOTIVATION_DIM]
            if self.motivation_store:
                dominant_motivation, intensity = self.motivation_store._analyze_motivation_vector(motivation_vec)
                result["motivation"] = dominant_motivation
                result["motivation_intensity"] = intensity
                from backend.motivation_store import MOTIVATION_SUBSPACE_DIMS
                result["achievement_mean"] = float(np.mean(motivation_vec[MOTIVATION_SUBSPACE_DIMS["achievement"][0]:MOTIVATION_SUBSPACE_DIMS["achievement"][1]]))
                result["relationship_mean"] = float(np.mean(motivation_vec[MOTIVATION_SUBSPACE_DIMS["relationship"][0]:MOTIVATION_SUBSPACE_DIMS["relationship"][1]]))
                result["exploration_mean"] = float(np.mean(motivation_vec[MOTIVATION_SUBSPACE_DIMS["exploration"][0]:MOTIVATION_SUBSPACE_DIMS["exploration"][1]]))
                result["curiosity_mean"] = result["exploration_mean"]
                result["motivation_safety_mean"] = float(np.mean(motivation_vec[MOTIVATION_SUBSPACE_DIMS["safety"][0]:MOTIVATION_SUBSPACE_DIMS["safety"][1]]))
                autonomy = self.motivation_store.calculate_autonomy(session_id)
                result["autonomy"] = autonomy
                result["autonomy_level"] = (
                    "high_autonomy" if autonomy > 0.3 else
                    "low_autonomy" if autonomy < -0.3 else
                    "mid_autonomy"
                )

        needs = self._load_needs_from_db(session_id) or self._get_default_needs()
        energy = self.get_energy(session_id)
        
        will_tension = 0.0
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute("SELECT will_tension FROM self_state WHERE session_id=?", (session_id,))
                row = cur.fetchone()
                if row:
                    will_tension = float(row[0] or 0.0)
        except Exception:
            pass
        result["will_tension"] = will_tension
        somatic_desc = "unknown"
        if self.dim >= SOMATIC_START_IDX + SOMATIC_DIM:
            start_idx = SOMATIC_START_IDX
            current_somatic_vec = z_self[start_idx:start_idx+SOMATIC_DIM]
            
            if self.somatic_store:
                emotion_vec = z_self[RULES_DIM:RULES_DIM+EMOTION_DIM]
                dom_emo = result.get("emotion", "中性")
                somatic_desc, _ = self.somatic_store.get_somatic_state(
                    energy, emotion_vec, dom_emo,
                    computed_vector=current_somatic_vec
                )
            
            result["somatic_desc"] = somatic_desc
            result["tension_mean"] = float(np.mean(current_somatic_vec[8:12]))
            result["pain_mean"] = result["tension_mean"]
            result["vitality_mean"] = float(np.mean(current_somatic_vec[12:16]))
            result["temperature_mean"] = 0.0
            result["viscosity_mean"] = float(np.mean(current_somatic_vec[4:8]))

            if self.somatic_store and self.emotion_store and self.motivation_store:
                try:
                    emotion_vec = z_self[RULES_DIM:RULES_DIM+EMOTION_DIM]
                    pleasure_emotion = float(np.mean(emotion_vec[0:4]))
                    novelty_emotion = float(np.mean(emotion_vec[12:16]))
                    social_emotion_for_warmth = float(
                        np.clip(0.58 * pleasure_emotion + 0.42 * novelty_emotion, -1.0, 1.0)
                    )
                    
                    motivation_vec = z_self[RULES_DIM+EMOTION_DIM:RULES_DIM+EMOTION_DIM+MOTIVATION_DIM]
                    relationship_motivation = float(np.mean(motivation_vec[4:8]))

                    mirror_feedback = None
                    if getattr(self, 'other_model', None):
                        try:
                            mirror_view = self.other_model.get_mirror_view(session_id)
                            if mirror_view:
                                _mv = mirror_view
                                if any(
                                    k in _mv
                                    for k in (
                                        "温暖",
                                        "信任",
                                        "warm",
                                        "trust",
                                        "trusting",
                                    )
                                ):
                                    mirror_feedback = 0.3
                                elif any(
                                    k in _mv
                                    for k in (
                                        "冷淡",
                                        "不耐烦",
                                        "cold",
                                        "distant",
                                        "impatient",
                                    )
                                ):
                                    mirror_feedback = -0.3
                        except Exception:
                            pass
                    
                    warmth = self.somatic_store.calculate_warmth(
                        social_emotion_for_warmth, relationship_motivation, mirror_feedback
                    )
                    warmth_desc = self.somatic_store.get_warmth_description(warmth)
                    result["warmth"] = warmth
                    result["warmth_desc"] = warmth_desc
                    result["warmth_level"] = (
                        "warm" if warmth > 0.3 else
                        "cold" if warmth < -0.3 else
                        "steady"
                    )
                except Exception as e:
                    logger.debug(f"Failed to calculate warmth: {e}")

        worldview_desc = ""
        if self.dim >= WORLDVIEW_Z_STATS_END and z_self.shape[0] >= WORLDVIEW_Z_STATS_END:
            wv_global = z_self[WORLDVIEW_Z_START:WORLDVIEW_Z_GLOBAL_END]
            optimism = float(np.mean(wv_global[0:4]))
            agency = float(np.mean(wv_global[4:8]))

            wv_parts = []
            if optimism > 0.1:
                wv_parts.append("The world skews toward improvement.")
            elif optimism < -0.1:
                wv_parts.append("The world feels entropic and chaotic.")

            if agency > 0.1:
                wv_parts.append("Agency can reshape outcomes.")
            elif agency < -0.1:
                wv_parts.append("Outcomes feel driven by external forces.")

            worldview_desc = ", ".join(wv_parts)
            result["worldview_desc"] = worldview_desc

        if MEMORY_DIM > 0 and self.dim >= RULES_DIM + EMOTION_DIM + MOTIVATION_DIM + SOMATIC_DIM + WORLDVIEW_DIM + MEMORY_DIM:
             mem_start = RULES_DIM + EMOTION_DIM + MOTIVATION_DIM + SOMATIC_DIM + WORLDVIEW_DIM
             mem_vec = z_self[mem_start:mem_start+MEMORY_DIM]
             result["memory_retrieval"] = float(np.mean(mem_vec[0:4]))
             result["memory_nostalgia"] = float(np.mean(mem_vec[4:8]))
        if ATTENTION_DIM > 0 and self.dim >= RULES_DIM + EMOTION_DIM + MOTIVATION_DIM + SOMATIC_DIM + WORLDVIEW_DIM + MEMORY_DIM + ATTENTION_DIM:
             att_start = RULES_DIM + EMOTION_DIM + MOTIVATION_DIM + SOMATIC_DIM + WORLDVIEW_DIM + MEMORY_DIM
             att_vec = z_self[att_start:att_start+ATTENTION_DIM]
             result["attention_focus"] = float(np.mean(att_vec[0:4]))
             result["attention_direction"] = float(np.mean(att_vec[4:8]))
        if self.attention_mechanism:
             result["attention_desc"] = self.attention_mechanism.get_attention_description()

        meaning = 0.0
        meaning_desc = ""
        if self.homeostasis and self.motivation_store:
            try:
                achievement = result.get("achievement_mean", 0.0)

                has_active_plan = False
                try:
                    import sqlite3 as _sqlite3
                    with _sqlite3.connect(self.db_path) as conn:
                        cur = conn.execute(
                            "SELECT COUNT(*) FROM execution_plans WHERE session_id=? AND status IN ('pending','in_progress')",
                            (session_id,)
                        )
                        row = cur.fetchone()
                        has_active_plan = row and row[0] > 0
                except Exception:
                    pass
                
                meaning = self.homeostasis.calculate_meaning(
                    session_id,
                    achievement_motivation=achievement,
                    has_active_plan=has_active_plan,
                    clarity=needs.get("clarity", 0.5)
                )
                meaning_desc = self.homeostasis.get_meaning_description(meaning)
            except Exception as e:
                logger.debug(f"Failed to calculate meaning: {e}")
        
        result.update({
            "needs": needs,
            "energy": energy,
            "is_dormant": energy < 10.0,
            "meaning": meaning,
            "meaning_desc": meaning_desc,
            "meaning_level": (
                "high_meaning" if meaning > 0.3 else
                "low_meaning" if meaning < -0.3 else
                "mid_meaning"
            ),
        })

        # [Fix] Sanitize NaN/Inf values for JSON serialization
        for k, v in list(result.items()):
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                result[k] = 0.0

        return result
    
    def get_experiential_summary(self, session_id: str) -> str:
        """
        Experiential narrative (not the numeric z_self summary).

        ``get_summary()`` is structured knobs; this returns subjective wording when ``meaning_generator`` is set.
        """
        if not self.meaning_generator:
            return self._generate_simple_experiential_summary(session_id)
        
        z_self = self.get_z_self(session_id)
        z_self_prev = self._get_previous_z_self(session_id)
        
        if z_self is None:
            return "I'm just starting out and still finding my footing."

        try:
            persona_rules = self.persona_store.get_all_active(limit=50)
        except Exception as e:
            logger.warning(f"Failed to get persona rules: {e}")
            persona_rules = []
        
        recent_events = self._get_recent_events(session_id)

        meaning = self.meaning_generator.extract_meaning(
            z_self, 
            z_self_prev,
            persona_rules,
            recent_events
        )
        
        interpretation = self.meaning_generator.interpret(meaning, z_self)

        narrative = self.meaning_generator.generate_narrative(
            interpretation, 
            z_self, 
            {"session_id": session_id}
        )
        
        try:
            self.meaning_generator.save_meaning_record(
                session_id,
                {"z_self": z_self.tolist() if z_self is not None else None},
                meaning,
                narrative
            )
        except Exception as e:
            logger.debug(f"Failed to save meaning record: {e}")
        
        return narrative
    
    def _generate_simple_experiential_summary(self, session_id: str) -> str:
        """Fallback experiential blurb when the meaning generator is unavailable."""
        z_self = self.get_z_self(session_id)
        if z_self is None:
            return ""
        
        parts = []
        
        try:
            drift = self._get_drift(session_id)
            if drift < 0.05:
                parts.append("I feel steady.")
            elif drift < 0.15:
                parts.append("I sense a small shift in myself.")
            else:
                parts.append("I feel I am changing in a noticeable way.")
        except Exception:
            pass
        
        try:
            energy = self.get_energy(session_id)
            if energy < 30:
                parts.append("I feel somewhat tired.")
            elif energy > 80:
                parts.append("I feel energetic.")
        except Exception:
            pass
        
        return " ".join(parts) if parts else "I sense my current state."

    def _get_previous_z_self(self, session_id: str) -> Optional[np.ndarray]:
        """Previous z_self snapshot for drift/meaning (best-effort)."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute("""
                    SELECT z_self FROM z_self_versions 
                    WHERE session_id=? 
                    ORDER BY tick DESC 
                    LIMIT 1 OFFSET 1
                """, (session_id,))
                row = cur.fetchone()
                if row and row[0]:
                    return np.array(json.loads(row[0]))
        except Exception as e:
            logger.debug(f"Failed to get previous z_self: {e}")
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute(
                    "SELECT z_self FROM self_state WHERE session_id=?",
                    (session_id,)
                )
                row = cur.fetchone()
                if row and row[0]:
                    return None
        except Exception:
            pass
        
        return None
    
    def get_identity_narrative(self, session_id: str) -> str:
        """
        Phase 5 identity narrative (delegates to ``narrative_identity`` when configured).
        """
        if not self.narrative_identity:
            return ""
        
        try:
            return self.narrative_identity.generate_identity_narrative(session_id)
        except Exception as e:
            logger.warning(f"Failed to get identity narrative: {e}")
            return ""
    
    def _get_recent_events(self, session_id: str) -> List[Dict]:
        """Recent rows from ``event_logs`` (best-effort)."""
        events = []
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute("""
                    SELECT event_type, event_data, created_at 
                    FROM event_logs 
                    WHERE session_id=? 
                    ORDER BY created_at DESC 
                    LIMIT 10
                """, (session_id,))
                for row in cur.fetchall():
                    events.append({
                        "type": row[0],
                        "data": json.loads(row[1]) if row[1] else {},
                        "timestamp": row[2]
                    })
        except Exception as e:
            logger.debug(f"Failed to get recent events: {e}")
        
        return events
    
    def _get_drift(self, session_id: str) -> float:
        """Drift column from ``self_state``."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute(
                    "SELECT drift FROM self_state WHERE session_id=?",
                    (session_id,)
                )
                row = cur.fetchone()
                if row and row[0] is not None:
                    return float(row[0])
        except Exception:
            pass
        return 0.0
    
    def _apply_identity_correction(self, z_self: np.ndarray, 
                                  continuity: float) -> np.ndarray:
        """
        Blend current identity head toward the latest stored identity when continuity is low.
        """
        if not hasattr(self, 'identity_space') or len(self.identity_space.memory_continuum) == 0:
            return z_self
        
        recent_memory = self.identity_space.memory_continuum[-1]
        recent_identity = self.identity_space._extract_identity_vector(recent_memory)
        current_identity = self.identity_space._extract_identity_vector(z_self)
        
        correction_strength = (1.0 - continuity) * 0.3
        corrected_identity = (
            (1 - correction_strength) * current_identity +
            correction_strength * recent_identity
        )
        
        z_self_corrected = z_self.copy()
        if z_self_corrected.shape[0] >= 16:
            z_self_corrected[:16] = corrected_identity[:16]
        else:
            z_self_corrected = np.concatenate([
                corrected_identity[:z_self_corrected.shape[0]],
                z_self_corrected[z_self_corrected.shape[0]:]
            ])
        
        return z_self_corrected

    def decide_interaction_mode(self, session_id: str) -> Dict:
        """
        Discrete interaction mode (analytical / direct / story / cautious / defensive / balanced) from z_self + needs + pain.
        """
        mode = "balanced"
        reasons: List[str] = []
        stats: Dict[str, float] = {}

        try:
            summary = self.get_structured_summary(session_id)
        except Exception as e:
            logger.debug(f"decide_interaction_mode: failed to get structured summary: {e}")
            summary = {}

        try:
            pain_status = self.get_pain_status(session_id)
        except Exception as e:
            logger.debug(f"decide_interaction_mode: failed to get pain status: {e}")
            pain_status = {"total_pain": 0.0}

        safety_mean = float(summary.get("safety_mean", 0.0))
        epistemic_mean = float(summary.get("epistemic_mean", 0.0))
        style_mean = float(summary.get("style_mean", 0.0))
        strategy_mean = float(summary.get("strategy_mean", 0.0))
        energy = float(summary.get("energy", 100.0))
        is_dormant = bool(summary.get("is_dormant", False))
        needs = summary.get("needs", {}) or {}
        pain_level = float(pain_status.get("total_pain", 0.0))

        stats.update(
            safety_mean=safety_mean,
            epistemic_mean=epistemic_mean,
            style_mean=style_mean,
            strategy_mean=strategy_mean,
            energy=energy,
            pain_level=pain_level,
        )

        if is_dormant or energy < 10.0 or pain_level > 0.85:
            mode = "defensive"
            reasons.append("Very low energy or high pain: defensive / conserve mode")
        elif energy < 25.0 or pain_level > 0.6:
            mode = "cautious"
            reasons.append("Low energy or elevated pain: cautious mode")

        if mode == "balanced":
            if strategy_mean > 0.15:
                mode = "analytical"
                reasons.append("strategy_mean>0.15: analyze before answering")
            elif strategy_mean < -0.15:
                mode = "direct"
                reasons.append("strategy_mean<-0.15: execute directly")

        if mode in ("balanced", "analytical") and style_mean > 0.2 and energy > 40.0:
            mode = "story"
            reasons.append("style_mean>0.2 with enough energy: narrative-friendly mode")

        connection = float(needs.get("connection", 0.5))
        clarity = float(needs.get("clarity", 0.5))
        if mode == "balanced" and connection > 0.7 and clarity > 0.7:
            mode = "story"
            reasons.append("High connection and clarity: favor open expression")

        return {
            "mode": mode,
            "reasons": reasons,
            "stats": stats,
        }
