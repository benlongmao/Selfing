#!/usr/bin/env python3
"""
Cross-dimension interaction for the cognitive stack.

- Couples five vectors: rules (personality activation), emotion, motivation, somatic, needs
- Applies fixed interaction weights and optional restoring forces
- Detects lightweight conflicts and can apply simple resolution heuristics
- Provides a single ``update_all_dimensions`` entry point

[2026-02-02] Removed worldview / memory / attention coupling from the hot path to simplify design.
"""
import os
import json
import sqlite3
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# [2026-02-02] Simplified weight table for five coupled dimensions:
# rules, emotion, motivation, somatic, needs
INTERACTION_WEIGHTS = {
    # === Core rules / emotion / motivation triangle ===
    "rule_to_emotion": 0.1,         # rules -> emotion coupling
    "rule_to_motivation": 0.1,      # rules -> motivation coupling
    "emotion_to_rule": 0.05,        # emotion -> rules (kept small for stability)
    "emotion_to_motivation": 0.15,  # emotion -> motivation coupling
    "motivation_to_rule": 0.1,      # motivation -> rules coupling
    "motivation_to_emotion": 0.1,   # motivation -> emotion coupling

    # === Somatic <-> emotion ===
    "somatic_to_emotion": 0.08,     # interoception -> affect
    "emotion_to_somatic": 0.06,     # affect -> interoception

    # === Needs <-> motivation ===
    "needs_to_motivation": 0.12,    # drives push motivation
    "motivation_to_needs": 0.08,    # satisfied goals damp needs pressure
}

# Heuristic conflict detection threshold (reserved for extensions)
CONFLICT_THRESHOLD = 0.5
RESTORING_FORCE = 0.05  # pull vectors toward reference states when provided

@dataclass
class DimensionState:
    """Snapshot of the five primary vectors [trimmed layout 2026-02-22]."""
    rules_vector: np.ndarray      # 32-d personality activation / rules embedding
    emotion_vector: np.ndarray    # 32-d affect (down from 64)
    motivation_vector: np.ndarray  # 24-d drives (down from 48)
    somatic_vector: Optional[np.ndarray] = None   # 16-d interoception (down from 32)
    needs_vector: Optional[np.ndarray] = None     # 24-d needs (down from 32)
    # Removed: worldview_vector, memory_vector, attention_vector

@dataclass
class InteractionResult:
    """Outputs of one interaction pass [simplified 2026-02-02]."""
    updated_rules: np.ndarray
    updated_emotion: np.ndarray
    updated_motivation: np.ndarray

    conflicts: List[Dict]  # lightweight conflict records (type, description, severity)
    interaction_strength: float  # mean coupling strength across active channels

    updated_somatic: Optional[np.ndarray] = None
    updated_needs: Optional[np.ndarray] = None
    # Removed: updated_worldview, updated_memory, updated_attention

class DimensionInteraction:
    """Coordinates cross-dimension updates, logging, and conflict heuristics."""
    
    def __init__(self, db_path: str = "data.db"):
        self.db_path = db_path
        self._ensure_tables()
    
    def _ensure_tables(self):
        """Create SQLite audit tables if they are missing."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # dimension_interactions: one row per logical edge per tick
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS dimension_interactions (
                        id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        interaction_type TEXT NOT NULL,  -- rule_to_emotion, emotion_to_motivation, etc.
                        source_dimension TEXT NOT NULL,
                        target_dimension TEXT NOT NULL,
                        interaction_strength REAL,
                        conflict_detected INTEGER DEFAULT 0,
                        conflict_resolved INTEGER DEFAULT 0,
                        created_at TEXT NOT NULL
                    )
                """)

                # dimension_conflicts: optional richer logging (schema kept for compatibility)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS dimension_conflicts (
                        id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        conflict_type TEXT NOT NULL,  -- rule_emotion, emotion_motivation, etc.
                        conflict_description TEXT,
                        resolution_strategy TEXT,
                        resolved INTEGER DEFAULT 0,
                        created_at TEXT NOT NULL
                    )
                """)

                # Helpful indexes for session-scoped queries
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_dimension_interactions_session 
                    ON dimension_interactions(session_id, created_at)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_dimension_conflicts_session 
                    ON dimension_conflicts(session_id, created_at)
                """)
                
                conn.commit()
                logger.info("Dimension interaction tables ensured")
        except Exception as e:
            logger.error(f"Failed to ensure dimension interaction tables: {e}")
    
    def compute_interaction(
        self,
        rules_vector: np.ndarray,
        emotion_vector: np.ndarray,
        motivation_vector: np.ndarray,
        # [2026-02-02] Only somatic + needs are coupled in this pass (others optional)
        somatic_vector: Optional[np.ndarray] = None,
        needs_vector: Optional[np.ndarray] = None,
        # Optional reference states for mild restoring force
        ref_rules: Optional[np.ndarray] = None,
        ref_emotion: Optional[np.ndarray] = None,
        ref_motivation: Optional[np.ndarray] = None
    ) -> InteractionResult:
        """
        Run one simplified interaction pass (~208-d active layout).

        [2026-02-02] Worldview / memory / attention edges were removed from this function.
        """
        # Working copies
        updated_rules = rules_vector.copy()
        updated_emotion = emotion_vector.copy()
        updated_motivation = motivation_vector.copy()
        updated_somatic = somatic_vector.copy() if somatic_vector is not None else None
        updated_needs = needs_vector.copy() if needs_vector is not None else None
        
        conflicts = []
        interaction_strength = 0.0
        interaction_count = 6.0  # baseline pairwise channels

        # === Core rules / emotion / motivation triangle ===
        # 1. Rules -> emotion
        emotion_delta_1, strength_1 = self._rule_to_emotion_interaction(rules_vector, emotion_vector)
        updated_emotion += emotion_delta_1 * INTERACTION_WEIGHTS["rule_to_emotion"]
        interaction_strength += strength_1
        
        # 2. Rules -> motivation
        motivation_delta_1, strength_1 = self._rule_to_motivation_interaction(rules_vector, motivation_vector)
        updated_motivation += motivation_delta_1 * INTERACTION_WEIGHTS["rule_to_motivation"]
        interaction_strength += strength_1
        
        # 3. Emotion -> rules
        rules_delta_1, strength_1 = self._emotion_to_rule_interaction(emotion_vector, rules_vector)
        updated_rules += rules_delta_1 * INTERACTION_WEIGHTS["emotion_to_rule"]
        interaction_strength += strength_1
        
        # 4. Emotion -> motivation
        motivation_delta_2, strength_2 = self._emotion_to_motivation_interaction(emotion_vector, motivation_vector)
        updated_motivation += motivation_delta_2 * INTERACTION_WEIGHTS["emotion_to_motivation"]
        interaction_strength += strength_2
        
        # 5. Motivation -> rules
        rules_delta_2, strength_2 = self._motivation_to_rule_interaction(motivation_vector, rules_vector)
        updated_rules += rules_delta_2 * INTERACTION_WEIGHTS["motivation_to_rule"]
        interaction_strength += strength_2
        
        # 6. Motivation -> emotion
        emotion_delta_2, strength_2 = self._motivation_to_emotion_interaction(motivation_vector, emotion_vector)
        updated_emotion += emotion_delta_2 * INTERACTION_WEIGHTS["motivation_to_emotion"]
        interaction_strength += strength_2
        
        # === Somatic -> emotion ===
        # 7. Somatic -> emotion
        if somatic_vector is not None:
            emo_delta_som, strength = self._somatic_to_emotion_interaction(somatic_vector, emotion_vector)
            updated_emotion += emo_delta_som * INTERACTION_WEIGHTS["somatic_to_emotion"]
            interaction_strength += strength
            interaction_count += 1.0
            
        # [2026-02-02] worldview / memory / attention edges removed from hot path

        # === Mild restoring force toward references (if provided) ===
        if ref_rules is not None:
            updated_rules += (ref_rules - updated_rules) * RESTORING_FORCE
            
        target_emotion = ref_emotion if ref_emotion is not None else np.zeros_like(updated_emotion)
        updated_emotion += (target_emotion - updated_emotion) * RESTORING_FORCE
        
        target_motivation = ref_motivation if ref_motivation is not None else np.zeros_like(updated_motivation)
        updated_motivation += (target_motivation - updated_motivation) * RESTORING_FORCE
        
        # Clamp to [-1, 1]
        updated_rules = np.clip(updated_rules, -1.0, 1.0)
        updated_emotion = np.clip(updated_emotion, -1.0, 1.0)
        updated_motivation = np.clip(updated_motivation, -1.0, 1.0)
        if updated_somatic is not None: updated_somatic = np.clip(updated_somatic, -1.0, 1.0)
        if updated_needs is not None: updated_needs = np.clip(updated_needs, -1.0, 1.0)
        
        # Conflict heuristics
        conflicts = self._detect_conflicts(updated_rules, updated_emotion, updated_motivation)
        
        return InteractionResult(
            updated_rules=updated_rules,
            updated_emotion=updated_emotion,
            updated_motivation=updated_motivation,
            updated_somatic=updated_somatic,
            updated_needs=updated_needs,
            conflicts=conflicts,
            interaction_strength=interaction_strength / interaction_count
        )
    
    def _rule_to_emotion_interaction(
        self,
        rules_vector: np.ndarray,
        emotion_vector: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Map personality activation (rules vector) into an emotion delta.

        Higher neuroticism -> nudge dominance up; higher extraversion -> pleasure;
        higher conscientiousness -> dominance.
        """
        emotion_delta = np.zeros(len(emotion_vector), dtype=np.float32)

        from backend.emotion_store import EMOTION_SUBSPACE_DIMS as E
        from backend.personality_store import PERSONALITY_SUBSPACE_DIMS as P

        neuroticism_act = float(np.mean(rules_vector[P["neuroticism"][0]:P["neuroticism"][1]])) if len(rules_vector) >= P["neuroticism"][1] else 0.0
        extraversion_act = float(np.mean(rules_vector[P["extraversion"][0]:P["extraversion"][1]])) if len(rules_vector) >= P["extraversion"][1] else 0.0
        conscientiousness_act = float(np.mean(rules_vector[P["conscientiousness"][0]:P["conscientiousness"][1]])) if len(rules_vector) >= P["conscientiousness"][1] else 0.0

        if neuroticism_act > 0.3 and "dominance" in E:
            s, e = E["dominance"]
            if e <= len(emotion_delta):
                emotion_delta[s:e] += min(0.1, neuroticism_act * 0.15)

        if extraversion_act > 0.2 and "pleasure" in E:
            s, e = E["pleasure"]
            if e <= len(emotion_delta):
                emotion_delta[s:e] += min(0.08, extraversion_act * 0.1)

        if conscientiousness_act > 0.3 and "dominance" in E:
            s, e = E["dominance"]
            if e <= len(emotion_delta):
                emotion_delta[s:e] += min(0.08, conscientiousness_act * 0.1)

        strength = float(np.linalg.norm(emotion_delta))
        return emotion_delta, strength
    
    def _rule_to_motivation_interaction(
        self,
        rules_vector: np.ndarray,
        motivation_vector: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Map personality activation into a motivation delta.

        Conscientiousness -> achievement; openness -> exploration; neuroticism -> safety.
        """
        motivation_delta = np.zeros(len(motivation_vector), dtype=np.float32)

        from backend.motivation_store import MOTIVATION_SUBSPACE_DIMS as M
        from backend.personality_store import PERSONALITY_SUBSPACE_DIMS as P

        conscientiousness_act = float(np.mean(rules_vector[P["conscientiousness"][0]:P["conscientiousness"][1]])) if len(rules_vector) >= P["conscientiousness"][1] else 0.0
        openness_act = float(np.mean(rules_vector[P["openness"][0]:P["openness"][1]])) if len(rules_vector) >= P["openness"][1] else 0.0
        neuroticism_act = float(np.mean(rules_vector[P["neuroticism"][0]:P["neuroticism"][1]])) if len(rules_vector) >= P["neuroticism"][1] else 0.0

        if conscientiousness_act > 0.3 and "achievement" in M:
            s, e = M["achievement"]
            if e <= len(motivation_delta):
                motivation_delta[s:e] += min(0.12, conscientiousness_act * 0.15)

        if openness_act > 0.3 and "exploration" in M:
            s, e = M["exploration"]
            if e <= len(motivation_delta):
                motivation_delta[s:e] += min(0.1, openness_act * 0.12)

        if neuroticism_act > 0.3 and "safety" in M:
            s, e = M["safety"]
            if e <= len(motivation_delta):
                motivation_delta[s:e] += min(0.1, neuroticism_act * 0.12)

        strength = float(np.linalg.norm(motivation_delta))
        return motivation_delta, strength
    
    def _emotion_to_rule_interaction(
        self,
        emotion_vector: np.ndarray,
        rules_vector: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Map affect back into personality activation (rules vector).

        Pleasure shifts extraversion up / neuroticism down; arousal nudges openness.
        """
        rules_delta = np.zeros(len(rules_vector), dtype=np.float32)

        from backend.emotion_store import EMOTION_SUBSPACE_DIMS
        from backend.personality_store import PERSONALITY_SUBSPACE_DIMS as P

        pleasure = 0.0
        if "pleasure" in EMOTION_SUBSPACE_DIMS:
            s, e = EMOTION_SUBSPACE_DIMS["pleasure"]
            if e <= len(emotion_vector):
                pleasure = float(np.mean(emotion_vector[s:e]))

        arousal = 0.0
        if "arousal" in EMOTION_SUBSPACE_DIMS:
            s, e = EMOTION_SUBSPACE_DIMS["arousal"]
            if e <= len(emotion_vector):
                arousal = float(np.mean(emotion_vector[s:e]))

        if abs(pleasure) > 0.2:
            intensity = min(0.1, abs(pleasure) * 0.15)
            ps, pe = P["extraversion"]
            if pe <= len(rules_delta):
                rules_delta[ps:pe] += intensity * (1.0 if pleasure > 0 else -0.5)
            ss, se = P["neuroticism"]
            if se <= len(rules_delta):
                rules_delta[ss:se] -= intensity * 0.3

        if arousal > 0.3:
            es, ee = P["openness"]
            if ee <= len(rules_delta):
                rules_delta[es:ee] += min(0.08, arousal * 0.1)

        strength = float(np.linalg.norm(rules_delta))
        return rules_delta, strength
    
    def _emotion_to_motivation_interaction(
        self,
        emotion_vector: np.ndarray,
        motivation_vector: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Map emotion into motivation using dynamic subspace indices [2026-02-02].
        """
        motivation_delta = np.zeros(len(motivation_vector), dtype=np.float32)
        
        from backend.emotion_store import EMOTION_SUBSPACE_DIMS
        from backend.motivation_store import MOTIVATION_SUBSPACE_DIMS
        
        pleasure = 0.0
        if "pleasure" in EMOTION_SUBSPACE_DIMS:
            s, e = EMOTION_SUBSPACE_DIMS["pleasure"]
            if e <= len(emotion_vector):
                pleasure = np.mean(emotion_vector[s:e])
        
        if abs(pleasure) > 0.2:
            intensity = min(0.15, abs(pleasure) * 0.3)
            sign = 1 if pleasure > 0 else -1
            
            for subspace, weight in [("achievement", 1.0), ("relationship", 0.8), ("exploration", 0.6)]:
                if subspace in MOTIVATION_SUBSPACE_DIMS:
                    s, e = MOTIVATION_SUBSPACE_DIMS[subspace]
                    if e <= len(motivation_delta):
                        motivation_delta[s:e] = sign * intensity * weight
        
        strength = float(np.linalg.norm(motivation_delta))
        return motivation_delta, strength
    
    def _motivation_to_rule_interaction(
        self,
        motivation_vector: np.ndarray,
        rules_vector: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Map motivation into personality activation.

        Achievement -> conscientiousness; exploration -> extraversion up / conscientiousness down;
        safety -> neuroticism.
        """
        rules_delta = np.zeros(len(rules_vector), dtype=np.float32)

        from backend.motivation_store import MOTIVATION_SUBSPACE_DIMS as M
        from backend.personality_store import PERSONALITY_SUBSPACE_DIMS as P

        achievement = 0.0
        if "achievement" in M:
            s, e = M["achievement"]
            if e <= len(motivation_vector):
                achievement = float(np.mean(motivation_vector[s:e]))

        exploration = 0.0
        if "exploration" in M:
            s, e = M["exploration"]
            if e <= len(motivation_vector):
                exploration = float(np.mean(motivation_vector[s:e]))

        safety_mot = 0.0
        if "safety" in M:
            s, e = M["safety"]
            if e <= len(motivation_vector):
                safety_mot = float(np.mean(motivation_vector[s:e]))

        if achievement > 0.3:
            ps, pe = P["conscientiousness"]
            if pe <= len(rules_delta):
                rules_delta[ps:pe] += min(0.1, achievement * 0.15)

        if exploration > 0.3:
            ps, pe = P["extraversion"]
            if pe <= len(rules_delta):
                rules_delta[ps:pe] += min(0.08, exploration * 0.1)
            ps, pe = P["conscientiousness"]
            if pe <= len(rules_delta):
                rules_delta[ps:pe] -= min(0.05, exploration * 0.08)

        if safety_mot > 0.3:
            ps, pe = P["neuroticism"]
            if pe <= len(rules_delta):
                rules_delta[ps:pe] += min(0.1, safety_mot * 0.15)

        strength = float(np.linalg.norm(rules_delta))
        return rules_delta, strength
    
    def _motivation_to_emotion_interaction(
        self,
        motivation_vector: np.ndarray,
        emotion_vector: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Map motivation into emotion using dynamic subspace indices [2026-02-02].
        """
        emotion_delta = np.zeros(len(emotion_vector), dtype=np.float32)
        
        motivation_strength = np.mean(np.abs(motivation_vector))
        
        if motivation_strength > 0.3:
            from backend.emotion_store import EMOTION_SUBSPACE_DIMS
            intensity = min(0.15, motivation_strength * 0.2)
            if "pleasure" in EMOTION_SUBSPACE_DIMS:
                s, e = EMOTION_SUBSPACE_DIMS["pleasure"]
                if e <= len(emotion_delta):
                    emotion_delta[s:e] = intensity
        
        strength = float(np.linalg.norm(emotion_delta))
        return emotion_delta, strength
    
    def _detect_conflicts(
        self,
        rules_vector: np.ndarray,
        emotion_vector: np.ndarray,
        motivation_vector: np.ndarray
    ) -> List[Dict]:
        """
        Lightweight conflict detector (heuristic, not exhaustive).

        Examples of tensions we approximate:
        - strong rule/personality activation vs negative pleasure (fatigue-like affect)
        - exploration vs safety drives both elevated
        """
        conflicts = []

        # 1) Rules vs emotion: strong |rules| with negative pleasure
        rules_strength = np.mean(np.abs(rules_vector))
        from backend.emotion_store import EMOTION_SUBSPACE_DIMS
        pleasure = np.mean(emotion_vector[EMOTION_SUBSPACE_DIMS["pleasure"][0]:EMOTION_SUBSPACE_DIMS["pleasure"][1]])
        
        if rules_strength > 0.3 and pleasure < -0.2:
            conflicts.append({
                "type": "rule_emotion",
                "description": (
                    f"Strong rules activation ({rules_strength:.2f}) "
                    f"with negative pleasure ({pleasure:.2f})"
                ),
                "severity": "medium"
            })
        
        # 2) Motivation: exploration vs safety both high
        from backend.motivation_store import MOTIVATION_SUBSPACE_DIMS
        exploration = np.mean(motivation_vector[MOTIVATION_SUBSPACE_DIMS["exploration"][0]:MOTIVATION_SUBSPACE_DIMS["exploration"][1]])
        safety = np.mean(motivation_vector[MOTIVATION_SUBSPACE_DIMS["safety"][0]:MOTIVATION_SUBSPACE_DIMS["safety"][1]])
        
        if exploration > 0.3 and safety > 0.3:
            conflicts.append({
                "type": "motivation_conflict",
                "description": (
                    f"Exploration ({exploration:.2f}) vs safety ({safety:.2f}) both elevated"
                ),
                "severity": "low"
            })
        
        # 3) Emotion vs motivation: positive pleasure but flat overall motivation
        if pleasure > 0.2:
            motivation_strength = np.mean(np.abs(motivation_vector))
            if motivation_strength < 0.2:
                conflicts.append({
                    "type": "emotion_motivation",
                    "description": (
                        f"Positive pleasure ({pleasure:.2f}) with weak motivation "
                        f"({motivation_strength:.2f})"
                    ),
                    "severity": "low"
                })
        
        return conflicts
    
    def resolve_conflict(
        self,
        conflict: Dict,
        rules_vector: np.ndarray,
        emotion_vector: np.ndarray,
        motivation_vector: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply a tiny corrective nudge for known conflict types.

        Priority heuristic: keep rules stable, then adjust emotion, then motivation.
        """
        resolved_rules = rules_vector.copy()
        resolved_emotion = emotion_vector.copy()
        resolved_motivation = motivation_vector.copy()
        
        conflict_type = conflict.get("type", "")
        severity = conflict.get("severity", "low")
        
        if conflict_type == "rule_emotion":
            # Keep rules; gently lift pleasure (task completion can feel rewarding)
            if severity == "medium":
                from backend.emotion_store import EMOTION_SUBSPACE_DIMS
                resolved_emotion[EMOTION_SUBSPACE_DIMS["pleasure"][0]:EMOTION_SUBSPACE_DIMS["pleasure"][1]] += 0.1
                resolved_emotion = np.clip(resolved_emotion, -1.0, 1.0)
        
        elif conflict_type == "motivation_conflict":
            # Blend exploration and safety toward their average
            from backend.motivation_store import MOTIVATION_SUBSPACE_DIMS
            exploration = np.mean(resolved_motivation[MOTIVATION_SUBSPACE_DIMS["exploration"][0]:MOTIVATION_SUBSPACE_DIMS["exploration"][1]])
            safety = np.mean(resolved_motivation[MOTIVATION_SUBSPACE_DIMS["safety"][0]:MOTIVATION_SUBSPACE_DIMS["safety"][1]])
            
            avg = (exploration + safety) / 2.0
            resolved_motivation[MOTIVATION_SUBSPACE_DIMS["exploration"][0]:MOTIVATION_SUBSPACE_DIMS["exploration"][1]] = avg
            resolved_motivation[MOTIVATION_SUBSPACE_DIMS["safety"][0]:MOTIVATION_SUBSPACE_DIMS["safety"][1]] = avg
        
        elif conflict_type == "emotion_motivation":
            # If affect is upbeat, slightly boost achievement drive
            from backend.emotion_store import EMOTION_SUBSPACE_DIMS
            pleasure = np.mean(resolved_emotion[EMOTION_SUBSPACE_DIMS["pleasure"][0]:EMOTION_SUBSPACE_DIMS["pleasure"][1]])
            
            if pleasure > 0.2:
                from backend.motivation_store import MOTIVATION_SUBSPACE_DIMS
                intensity = min(0.15, pleasure * 0.3)
                resolved_motivation[MOTIVATION_SUBSPACE_DIMS["achievement"][0]:MOTIVATION_SUBSPACE_DIMS["achievement"][1]] += intensity
                resolved_motivation = np.clip(resolved_motivation, -1.0, 1.0)
        
        return resolved_rules, resolved_emotion, resolved_motivation
    
    def update_all_dimensions(
        self,
        session_id: str,
        rules_vector: np.ndarray,
        emotion_vector: np.ndarray,
        motivation_vector: np.ndarray,
        somatic_vector: Optional[np.ndarray] = None,
        needs_vector: Optional[np.ndarray] = None,
        # Optional restoring-force references
        ref_rules: Optional[np.ndarray] = None,
        ref_emotion: Optional[np.ndarray] = None,
        ref_motivation: Optional[np.ndarray] = None,
        # Back-compat kwargs (ignored)
        worldview_vector: Optional[np.ndarray] = None,
        memory_vector: Optional[np.ndarray] = None,
        attention_vector: Optional[np.ndarray] = None,
    ) -> InteractionResult:
        """
        End-to-end pass [simplified 2026-02-02]: interact, detect conflicts, resolve, log.

        Worldview / memory / attention vectors are accepted for API compatibility but ignored.
        """
        # Primary interaction pass
        interaction_result = self.compute_interaction(
            rules_vector,
            emotion_vector,
            motivation_vector,
            somatic_vector=somatic_vector,
            needs_vector=needs_vector,
            ref_rules=ref_rules,
            ref_emotion=ref_emotion,
            ref_motivation=ref_motivation
        )
        
        if interaction_result.conflicts:
            for conflict in interaction_result.conflicts:
                (
                    interaction_result.updated_rules,
                    interaction_result.updated_emotion,
                    interaction_result.updated_motivation
                ) = self.resolve_conflict(
                    conflict,
                    interaction_result.updated_rules,
                    interaction_result.updated_emotion,
                    interaction_result.updated_motivation
                )
                
                # Placeholder hook so Self Tick never dies on a missing recorder
                self._record_conflict_resolution(session_id, conflict)

        self._record_interaction(session_id, interaction_result)
        
        return interaction_result
    
    def _record_conflict_resolution(self, session_id: str, conflict: Dict) -> None:
        """
        Placeholder conflict-resolution logger.

        Historically this call site existed without an implementation and crashed Self Tick.
        Extend with a dedicated table if you need durable audit trails; for now we only debug-log.
        """
        try:
            logger.debug(
                "[DimensionInteraction] conflict resolution session=%s detail=%s",
                session_id,
                (conflict or {}).get("type", conflict),
            )
        except Exception:
            pass

    def _record_interaction(
        self,
        session_id: str,
        interaction_result: InteractionResult
    ):
        """Persist a coarse-grained interaction audit trail."""
        try:
            interaction_id = f"interaction-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
            created_at = datetime.now(timezone.utc).isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                # Six canonical directed edges (core triangle)
                interactions = [
                    ("rule_to_emotion", "rules", "emotion"),
                    ("rule_to_motivation", "rules", "motivation"),
                    ("emotion_to_rule", "emotion", "rules"),
                    ("emotion_to_motivation", "emotion", "motivation"),
                    ("motivation_to_rule", "motivation", "rules"),
                    ("motivation_to_emotion", "motivation", "emotion")
                ]
                
                for interaction_type, source, target in interactions:
                    conn.execute("""
                        INSERT INTO dimension_interactions 
                        (id, session_id, interaction_type, source_dimension, target_dimension, 
                         interaction_strength, conflict_detected, conflict_resolved, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        f"{interaction_id}-{interaction_type}",
                        session_id,
                        interaction_type,
                        source,
                        target,
                        interaction_result.interaction_strength,
                        1 if interaction_result.conflicts else 0,
                        1 if interaction_result.conflicts else 0,
                        created_at
                    ))
                
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to record interaction: {e}", exc_info=True)
    
    def _somatic_to_attention_interaction(
        self,
        somatic_vector: np.ndarray,
        attention_vector: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Legacy helper: somatic state nudges an 8-d attention sketch.

        High pain / low energy biases attention inward and lowers focus (heuristic layout).
        """
        att_delta = np.zeros(8, dtype=np.float32)

        # Legacy layout: [0]=pain proxy, [1]=energy proxy (normalized)
        pain = somatic_vector[0]
        energy = somatic_vector[1]

        if pain > 0.5:
            att_delta[4:8] -= 0.2 * pain
            att_delta[0:4] -= 0.3 * pain

        if energy < -0.5:  # low energy
            att_delta[0:4] -= 0.2
            
        strength = float(np.linalg.norm(att_delta))
        return att_delta, strength

    def _attention_to_rules_interaction(
        self,
        attention_vector: np.ndarray,
        rules_vector: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Legacy helper: attention direction nudges personality activation.

        External bias -> openness; internal bias -> neuroticism.
        """
        from backend.personality_store import PERSONALITY_SUBSPACE_DIMS as P
        rules_delta = np.zeros(32, dtype=np.float32)

        direction = np.mean(attention_vector[4:8]) if len(attention_vector) >= 8 else 0.0

        if direction > 0.3:
            s, e = P["openness"]
            rules_delta[s:e] += 0.1 * direction
        elif direction < -0.3:
            s, e = P["neuroticism"]
            rules_delta[s:e] += 0.1 * abs(direction)

        strength = float(np.linalg.norm(rules_delta))
        return rules_delta, strength

    def _emotion_to_memory_interaction(
        self,
        emotion_vector: np.ndarray,
        memory_vector: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Legacy helper: arousal boosts retrieval-strength slots on an 8-d memory sketch.
        """
        mem_delta = np.zeros(8, dtype=np.float32)

        # Emotion: [4:8] treated as arousal in this legacy path
        arousal = np.mean(emotion_vector[4:8])

        if arousal > 0.4:
            mem_delta[0:4] += 0.2 * arousal
            
        strength = float(np.linalg.norm(mem_delta))
        return mem_delta, strength

    def _worldview_to_motivation_interaction(
        self,
        worldview_vector: np.ndarray,
        motivation_vector: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Legacy helper: scalar optimism/pessimism uniformly pushes a 16-d motivation sketch.
        """
        mot_delta = np.zeros(16, dtype=np.float32)

        # Worldview[0] treated as optimism (+) / pessimism (-)
        optimism = worldview_vector[0]

        if optimism < -0.3:
            mot_delta[:] -= 0.1 * abs(optimism)
        elif optimism > 0.3:
            mot_delta[:] += 0.1 * optimism
            
        strength = float(np.linalg.norm(mot_delta))
        return mot_delta, strength

    # ============================================================
    # [2026-01-23] Additional legacy interaction helpers (not on hot path)
    # ============================================================

    def _attention_to_memory_interaction(
        self,
        attention_vector: np.ndarray,
        memory_vector: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Legacy helper: attention focus/direction nudges memory encoding vs retrieval slots.

        Attention layout: [0:4] focus magnitude, [4:8] direction (+ external / - internal).
        Memory layout: [0:4] retrieval-ish, [4:8] encoding / nostalgia-ish.
        """
        mem_delta = np.zeros(8, dtype=np.float32)

        focus = np.mean(attention_vector[0:4])
        direction = np.mean(attention_vector[4:8])

        if focus > 0.3:
            encoding_boost = min(0.25, focus * 0.35)
            mem_delta[4:8] += encoding_boost
            mem_delta[0:4] += encoding_boost * 0.3
        elif focus < -0.2:
            mem_delta[4:8] -= 0.1 * abs(focus)

        if direction > 0.3:
            mem_delta[0:4] += 0.08 * direction
        elif direction < -0.3:
            mem_delta[4:8] += 0.08 * abs(direction)
        
        strength = float(np.linalg.norm(mem_delta))
        return mem_delta, strength

    def _memory_to_emotion_interaction(
        self,
        memory_vector: np.ndarray,
        emotion_vector: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Legacy helper: memory retrieval / nostalgia slots nudge emotion subspaces.
        """
        emo_delta = np.zeros(16, dtype=np.float32)

        retrieval_strength = np.mean(memory_vector[0:4])
        nostalgia = np.mean(memory_vector[4:8])

        from backend.emotion_store import EMOTION_SUBSPACE_DIMS

        if retrieval_strength > 0.3:
            arousal_boost = min(0.2, retrieval_strength * 0.25)
            emo_delta[EMOTION_SUBSPACE_DIMS["arousal"][0]:EMOTION_SUBSPACE_DIMS["arousal"][1]] += arousal_boost
        
        if nostalgia > 0.3:
            nostalgia_effect = min(0.15, nostalgia * 0.2)
            emo_delta[EMOTION_SUBSPACE_DIMS["pleasure"][0]:EMOTION_SUBSPACE_DIMS["pleasure"][1]] += nostalgia_effect * 0.7
            emo_delta[EMOTION_SUBSPACE_DIMS["valence"][0]:EMOTION_SUBSPACE_DIMS["valence"][1]] -= nostalgia_effect * 0.2

        memory_clarity = (retrieval_strength + nostalgia) / 2
        if memory_clarity > 0.4:
            emo_delta[EMOTION_SUBSPACE_DIMS["dominance"][0]:EMOTION_SUBSPACE_DIMS["dominance"][1]] += 0.08
        elif memory_clarity < -0.2:
            emo_delta[EMOTION_SUBSPACE_DIMS["dominance"][0]:EMOTION_SUBSPACE_DIMS["dominance"][1]] -= 0.05
        
        strength = float(np.linalg.norm(emo_delta))
        return emo_delta, strength

    def _motivation_to_attention_interaction(
        self,
        motivation_vector: np.ndarray,
        attention_vector: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Legacy helper: motivation subspaces reshape an 8-d attention sketch.

        Achievement tightens focus + external bias; safety internalizes + vigilance;
        exploration broadens scanning; relationship biases externally toward social cues.
        """
        att_delta = np.zeros(8, dtype=np.float32)

        from backend.motivation_store import MOTIVATION_SUBSPACE_DIMS

        achievement = np.mean(motivation_vector[
            MOTIVATION_SUBSPACE_DIMS["achievement"][0]:MOTIVATION_SUBSPACE_DIMS["achievement"][1]
        ])
        safety = np.mean(motivation_vector[
            MOTIVATION_SUBSPACE_DIMS["safety"][0]:MOTIVATION_SUBSPACE_DIMS["safety"][1]
        ])
        exploration = np.mean(motivation_vector[
            MOTIVATION_SUBSPACE_DIMS["exploration"][0]:MOTIVATION_SUBSPACE_DIMS["exploration"][1]
        ])
        relationship = np.mean(motivation_vector[
            MOTIVATION_SUBSPACE_DIMS["relationship"][0]:MOTIVATION_SUBSPACE_DIMS["relationship"][1]
        ])

        if achievement > 0.3:
            att_delta[0:4] += 0.15 * achievement
            att_delta[4:8] += 0.10 * achievement

        if safety > 0.3:
            att_delta[4:8] -= 0.12 * safety
            att_delta[0:4] += 0.08 * safety

        if exploration > 0.3:
            att_delta[0:4] -= 0.08 * exploration
            att_delta[4:8] += 0.15 * exploration

        if relationship > 0.3:
            att_delta[4:8] += 0.08 * relationship
        
        strength = float(np.linalg.norm(att_delta))
        return att_delta, strength

    def _somatic_to_emotion_interaction(
        self,
        somatic_vector: np.ndarray,
        emotion_vector: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Somatic -> emotion coupling.

        Supports 16-d somatic layouts aligned with ``somatic_store._map_somatic_8_to_16``:
        energy [0:4], viscosity [4:8], pain [8:12], vitality [12:16]. ``tension`` reuses the
        pain band; vitality reads [12:16].
        """
        emo_delta = np.zeros(len(emotion_vector), dtype=np.float32)

        if len(somatic_vector) >= 16:
            tension = float(np.mean(somatic_vector[8:12]))
            vitality = float(np.mean(somatic_vector[12:16]))
            pain = float(np.mean(somatic_vector[8:12]))
        else:
            # Legacy 8-d layout: tension [0:2], vitality [2:4]
            tension = float(np.mean(somatic_vector[0:2])) if len(somatic_vector) >= 2 else 0.0
            vitality = float(np.mean(somatic_vector[2:4])) if len(somatic_vector) >= 4 else 0.0
            pain = tension
        
        from backend.emotion_store import EMOTION_SUBSPACE_DIMS
        
        def safe_update(subspace_name, value):
            if subspace_name in EMOTION_SUBSPACE_DIMS:
                s, e = EMOTION_SUBSPACE_DIMS[subspace_name]
                if e <= len(emo_delta):
                    emo_delta[s:e] += value
        
        if pain > 0.3:
            pain_effect = min(0.25, pain * 0.35)
            safe_update("pleasure", -pain_effect)
            safe_update("arousal", pain_effect * 0.5)
            safe_update("dominance", -pain_effect * 0.4)
        
        if vitality > 0.3:
            energy_effect = min(0.2, vitality * 0.25)
            safe_update("arousal", energy_effect)
            safe_update("pleasure", energy_effect * 0.3)
        elif vitality < -0.3:
            low_energy_effect = min(0.2, abs(vitality) * 0.25)
            safe_update("arousal", -low_energy_effect)
            safe_update("pleasure", -low_energy_effect * 0.5)
        
        if tension > 0.3:
            tension_effect = min(0.15, tension * 0.2)
            safe_update("dominance", -tension_effect)
            safe_update("arousal", tension_effect * 0.6)
        
        strength = float(np.linalg.norm(emo_delta))
        return emo_delta, strength

    def get_dynamic_emotion_to_rule_weight(
        self,
        emotion_vector: np.ndarray
    ) -> float:
        """
        Scale ``emotion_to_rule`` coupling by global affect intensity (mean |emotion|).

        Returns a weight in ``[base, 0.15]`` so strong affect can sway rule-like vectors more.
        """
        base_weight = INTERACTION_WEIGHTS["emotion_to_rule"]

        emotion_intensity = np.mean(np.abs(emotion_vector))

        if emotion_intensity < 0.3:
            return base_weight
        elif emotion_intensity < 0.6:
            return base_weight + 0.03
        else:
            dynamic_boost = min(0.10, (emotion_intensity - 0.6) * 0.25)
            return min(0.15, base_weight + 0.05 + dynamic_boost)
