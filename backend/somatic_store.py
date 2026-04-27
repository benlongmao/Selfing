#!/usr/bin/env python3
"""
Somatic memory store (virtual interoception / somatic markers).

Persists first-person body-sensation text keyed by (energy band, dominant emotion)
and optional 8-D somatic vectors; maps to z_self's 16-D somatic slice when needed.
"""
import os
import sqlite3
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime, timezone, timedelta
import math

logger = logging.getLogger(__name__)

# Somatic layout: 8 dimensions (tension, vitality, temperature, viscosity pairs).
SOMATIC_DIM = 8
MAX_SOMATIC_PATTERNS = int(os.environ.get("MAX_SOMATIC_PATTERNS", "100"))
SOMATIC_SUBSPACE_DIMS = {
    "tension": (0, 2),      # tension (0-2): relaxed <-> tight
    "vitality": (2, 4),     # vitality (2-4): heavy/dead <-> light/alive
    "temperature": (4, 6),  # temperature (4-6): cold/isolated <-> hot/conflict
    "viscosity": (6, 8)     # viscosity (6-8): fluid/water <-> thick/mud
}


def _map_somatic_8_to_16(vec8: np.ndarray) -> np.ndarray:
    """Map store 8-D (tension, vitality, temp, viscosity) to z_self 16-D (energy, viscosity, pain, vitality)."""
    if vec8.shape[0] < 8:
        padded = np.zeros(8, dtype=np.float32)
        padded[:vec8.shape[0]] = vec8
        vec8 = padded
    t = float(np.mean(vec8[0:2]))
    v = float(np.mean(vec8[2:4]))
    visc = float(np.mean(vec8[6:8]))
    out = np.zeros(16, dtype=np.float32)
    out[0:4] = v    # energy ≈ vitality
    out[4:8] = visc
    out[8:12] = t   # pain ≈ tension
    out[12:16] = v
    return out


def _map_somatic_16_to_8(vec16: np.ndarray) -> np.ndarray:
    """Map z_self 16-D (energy, viscosity, pain, vitality) to store 8-D (tension, vitality, temp, viscosity)."""
    if vec16.shape[0] < 16:
        padded = np.zeros(16, dtype=np.float32)
        padded[:vec16.shape[0]] = vec16
        vec16 = padded
    # 16-D: energy(0-4), viscosity(4-8), pain(8-12), vitality(12-16)
    tension = float(np.mean(vec16[8:12]))   # pain → tension
    vitality = float(np.mean(vec16[12:16])) # vitality
    viscosity = float(np.mean(vec16[4:8]))  # viscosity
    temperature = 0.0  # no temperature slice in 16-D layout; hold at 0
    out = np.zeros(8, dtype=np.float32)
    out[0:2] = tension
    out[2:4] = vitality
    out[4:6] = temperature
    out[6:8] = viscosity
    return out

@dataclass
class SomaticPattern:
    id: str
    text: str  # First-person somatic line shown to the model / UI.
    min_energy: float
    max_energy: float
    # Match band + dominant_emotion (often Chinese labels from emotion pipeline).
    dominant_emotion: Optional[str]
    somatic_vector: np.ndarray  # 8-D store layout
    evidence_count: int = 0
    created_at: str = ""
    last_seen_at: str = ""
    locked: int = 0

class SomaticStore:
    def __init__(self, db_path: str = "data.db"):
        self.db_path = db_path
        self.dim = SOMATIC_DIM
        self._ensure_tables()
        
        # Evidence-weighted selection (usage + recency).
        self.EVIDENCE_DECAY_HALF_LIFE_DAYS = 30.0  # half-life in days
        
    def _ensure_tables(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS somatic_patterns (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    min_energy REAL,
                    max_energy REAL,
                    dominant_emotion TEXT,
                    somatic_vector BLOB,
                    evidence_count INTEGER DEFAULT 0,
                    created_at TEXT,
                    last_seen_at TEXT,
                    locked INTEGER DEFAULT 0
                )
            """)
            # Best-effort column adds for older DBs.
            for column in ["evidence_count INTEGER DEFAULT 0", "created_at TEXT", "last_seen_at TEXT", "locked INTEGER DEFAULT 0"]:
                try:
                    col_name = column.split()[0]
                    conn.execute(f"ALTER TABLE somatic_patterns ADD COLUMN {column}")
                except sqlite3.OperationalError:
                    pass
            conn.commit()
            
        self._seed_defaults()
            
    def _seed_defaults(self):
        """Seed built-in somatic lines when the table is empty (fresh install)."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("SELECT COUNT(*) FROM somatic_patterns")
            if cur.fetchone()[0] > 0:
                return

        # dominant_emotion stays Chinese to match emotion labels from SelfModel / reflection.
        defaults = [
            ("soma-warmth", "A warm swell in the chest; fingertips tingle slightly.", 40.0, 100.0, "快乐", 0.2, 0.8, 0.7, 0.2),
            ("soma-chill", "A cold line down the spine, like being splashed with ice water.", 10.0, 90.0, "恐惧", 0.9, 0.3, -0.8, 0.4),
            ("soma-heavy", "Limbs feel leaden; eyelids keep drifting shut.", 0.0, 30.0, "疲惫", 0.4, -0.7, 0.0, 0.8),
            ("soma-flutter", "Butterflies in the stomach; heart racing.", 50.0, 100.0, "焦虑", 0.8, 0.7, 0.3, -0.2),
            ("soma-empty", "A hollow in the chest, as if wind passes through.", 20.0, 60.0, "悲伤", 0.3, -0.4, -0.5, 0.1),
            ("soma-electric", "Thoughts spark like current across the scalp.", 70.0, 100.0, "兴奋", 0.7, 0.9, 0.5, -0.6),
            ("soma-tight", "Throat tight, as if an unseen hand were pressing there.", 30.0, 80.0, "压抑", 0.9, 0.2, 0.1, 0.7),
            ("soma-float", "Light and buoyant, like floating in warm water.", 40.0, 80.0, "平静", -0.5, 0.4, 0.4, 0.3),
        ]
        
        for _pid, text, min_e, max_e, emo, tension, vitality, temp, visc in defaults:
            self.add_pattern(text, min_e, max_e, emo, tension, vitality, temp, visc)
            # Lock seeded rows so evolve_somatics does not drop them.
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("UPDATE somatic_patterns SET locked=1 WHERE text=?", (text,))
        
        logger.info(f"Seeded {len(defaults)} default somatic patterns")

    def add_pattern(self, text: str, min_energy: float, max_energy: float, 
                   dominant_emotion: str, tension: float, vitality: float,
                   temperature: float = 0.0, viscosity: float = 0.0):
        """Insert a somatic pattern if under the global cap."""
        # Enforce cap
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("SELECT COUNT(*) FROM somatic_patterns")
            current_count = cur.fetchone()[0]
            if current_count >= MAX_SOMATIC_PATTERNS:
                logger.warning(f"Somatic pattern limit reached ({current_count} >= {MAX_SOMATIC_PATTERNS}), skipping add")
                return False
        
        vec = np.zeros(self.dim, dtype=np.float32)
        vec[0:2] = tension
        vec[2:4] = vitality
        vec[4:6] = temperature
        vec[6:8] = viscosity
        
        pattern_id = f"soma-{abs(hash(text))}"
        now = datetime.now(timezone.utc).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO somatic_patterns 
                (id, text, min_energy, max_energy, dominant_emotion, somatic_vector, evidence_count, created_at, last_seen_at, locked)
                VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?, 0)
            """, (pattern_id, text, min_energy, max_energy, dominant_emotion, 
                  vec.tobytes(), now, now))
            conn.commit()
        return True
            
    def _calculate_selection_score(
        self,
        evidence_count: int,
        last_seen_at: Optional[str],
        created_at: Optional[str],
        pattern_id: str = ""
    ) -> float:
        """
        Evidence-weighted score for picking among candidate patterns.

        selection_score = evidence_strength * 0.6 + recency_factor * 0.4

        Args:
            evidence_count: How often the pattern was used.
            last_seen_at: Last use time (ISO string).
            created_at: Creation time (ISO string).
            pattern_id: When set, locked rows get a high floor score.

        Returns:
            Score in [0, 1]; higher means more likely to be selected.
        """
        now = datetime.now(timezone.utc)
        
        # 1) Evidence strength from log-scaled evidence_count
        if evidence_count > 0:
            max_evidence = 100.0
            evidence_strength = min(1.0, math.log(1 + evidence_count) / math.log(1 + max_evidence))
        else:
            evidence_strength = 0.0
        
        # 2) Recency decay from last_seen_at
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
            recency_factor = 0.3  # never used -> modest recency
        
        # 3) Locked patterns (locked=1) get a high baseline score
        if pattern_id:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cur = conn.execute(
                        "SELECT locked FROM somatic_patterns WHERE id = ?",
                        (pattern_id,)
                    )
                    row = cur.fetchone()
                    if row and row[0] == 1:
                        return 0.9  # below 1.0 so hot unlocked patterns can still win
            except sqlite3.Error as e:
                logger.debug(f"Database error checking locked pattern: {e}")
            except Exception as e:
                logger.warning(f"Unexpected error checking locked pattern: {e}")
        
        # 4) Blend
        selection_score = (
            evidence_strength * 0.6 +
            recency_factor * 0.4
        )
        # Clamp to [0, 1]
        selection_score = max(0.0, min(1.0, selection_score))
        
        return selection_score
    
    def get_somatic_state(self, energy: float, emotion_vector: np.ndarray, dominant_emotion: str, computed_vector: Optional[np.ndarray] = None, expected_dim: int = None) -> Tuple[str, np.ndarray]:
        """
        Return (description, vector) for current energy + emotion + optional computed soma.

        ``computed_vector``: 8-D store layout (tension, vitality, temp, viscosity), or
        16-D z_self slice (energy, viscosity, pain, vitality) mapped to 8-D for mixing/filtering.

        Args:
            expected_dim: When 16, return z_self-aligned 16-D vector if possible.
        """
        # Output width
        output_dim = expected_dim if expected_dim and expected_dim > self.dim else self.dim
        # 1) Prefer DB-backed patterns
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Coarse filter: energy band
                cur = conn.execute("""
                    SELECT id, text, dominant_emotion, somatic_vector, evidence_count, last_seen_at, created_at, locked
                    FROM somatic_patterns 
                    WHERE ? >= min_energy AND ? <= max_energy
                """, (energy, energy))
                candidates = cur.fetchall()
                
            if candidates:
                # Fine filter: dominant_emotion label
                matched = [c for c in candidates if c[2] == dominant_emotion]
                if not matched:
                    matched = [c for c in candidates if not c[2] or c[2] == "any"]
                if not matched:
                    matched = candidates
                
                # [2026-02-25] Consistency filter: text should match viscosity regime.
                # 16-D z_self: viscosity mean over [4:8]; 8-D store: over [6:8]
                if computed_vector is not None and len(computed_vector) >= 8:
                    viscosity = float(np.mean(computed_vector[4:8])) if len(computed_vector) >= 16 else float(np.mean(computed_vector[6:8]))
                    
                    # Low viscosity: drop viscous / heavy phrasing (CN + EN substrings).
                    if viscosity < 0.3:
                        viscous_keywords = [
                            "粘稠", "粘滞", "凝固", "凝滞", "阻滞", "沉重", "艰难穿行",
                            "viscous", "mud", "tar", "stuck", "sluggish", "wading", "bog",
                        ]
                        matched = [c for c in matched if not any(kw in c[1] for kw in viscous_keywords)]
                    
                    # High viscosity: drop overly fluid phrasing
                    elif viscosity > 0.5:
                        fluid_keywords = [
                            "流畅", "轻盈", "顺畅", "丝滑", "如水",
                            "fluid", "water", "stream", "silky",
                        ]
                        matched = [c for c in matched if not any(kw in c[1] for kw in fluid_keywords)]
                    
                    # If nothing left, restore pool so we always have a line
                    if not matched:
                        matched = [c for c in candidates if c[2] == dominant_emotion]
                        if not matched:
                            matched = [c for c in candidates if not c[2] or c[2] == "any"]
                        if not matched:
                            matched = candidates
                
                # Rank candidates by selection score
                scored_candidates = []
                for c in matched:
                    selection_score = self._calculate_selection_score(
                        evidence_count=c[4] or 0,
                        last_seen_at=c[5],
                        created_at=c[6],
                        pattern_id=c[0]
                    )
                    scored_candidates.append((c, selection_score))
                
                scored_candidates.sort(key=lambda x: x[1], reverse=True)
                selected = scored_candidates[0][0] if scored_candidates else None
            else:
                selected = None
                
            if selected:
                best_text = selected[1]
                best_vec_raw = np.frombuffer(selected[3], dtype=np.float32)
                
                # Handle padding
                if best_vec_raw.shape[0] < self.dim:
                    best_vec = np.zeros(self.dim, dtype=np.float32)
                    best_vec[:best_vec_raw.shape[0]] = best_vec_raw
                else:
                    best_vec = best_vec_raw[:self.dim]

                # Blend DB vector with live computed_vector (0.7 / 0.3)
                if computed_vector is not None:
                    if computed_vector.shape[0] >= 16:
                        # 16-D z_self -> 8-D store semantics before blend
                        cv8 = _map_somatic_16_to_8(computed_vector)
                        best_vec = best_vec * 0.7 + cv8 * 0.3
                    elif computed_vector.shape[0] >= self.dim:
                        best_vec = best_vec * 0.7 + computed_vector[:self.dim] * 0.3
                    else:
                        cv_padded = np.zeros(self.dim, dtype=np.float32)
                        cv_padded[:computed_vector.shape[0]] = computed_vector
                        best_vec = best_vec * 0.7 + cv_padded * 0.3
                
                self._update_usage(selected[0])
                # Upsample to 16-D when requested
                if output_dim > best_vec.shape[0]:
                    if output_dim == 16 and best_vec.shape[0] == 8:
                        mapped = _map_somatic_8_to_16(best_vec)
                        return best_text, mapped
                    padded_vec = np.zeros(output_dim, dtype=np.float32)
                    padded_vec[:best_vec.shape[0]] = best_vec
                    return best_text, padded_vec
                return best_text, best_vec

        except Exception as e:
            logger.error(f"Error getting somatic state from DB: {e}")

        # 2) Fallback synesthesia-style line from computed_vector
        if computed_vector is not None and computed_vector.shape[0] >= 8:
            try:
                # 16-D z_self: energy(0-4), viscosity(4-8), pain(8-12), vitality(12-16)
                if computed_vector.shape[0] >= 16:
                    tension_val = float(np.mean(computed_vector[8:12]))   # pain
                    vitality_val = float(np.mean(computed_vector[12:16]))
                    temperature_val = 0.0
                    viscosity_val = float(np.mean(computed_vector[4:8]))
                else:
                    tension_val = np.mean(computed_vector[0:2])
                    vitality_val = np.mean(computed_vector[2:4])
                    temperature_val = np.mean(computed_vector[4:6])
                    viscosity_val = np.mean(computed_vector[6:8])
                
                # Pick dominant axes for metaphor
                features = [
                    ("tension", abs(tension_val), tension_val),
                    ("vitality", abs(vitality_val), vitality_val),
                    ("temperature", abs(temperature_val), temperature_val),
                    ("viscosity", abs(viscosity_val), viscosity_val)
                ]
                features.sort(key=lambda x: x[1], reverse=True)
                
                primary = features[0]
                secondary = features[1] if len(features) > 1 else None
                
                base_desc = ""
                metaphor = ""
                
                if primary[0] == "temperature":
                    if primary[2] > 0.5: 
                        base_desc = "restless heat"
                        metaphor = "like cracked earth under noon sun"
                    elif primary[2] < -0.5: 
                        base_desc = "cold isolation"
                        metaphor = "like a slab of iron on a winter night"
                    else:
                        base_desc = "even temperature"
                
                elif primary[0] == "viscosity":
                    if primary[2] > 0.5: 
                        base_desc = "sticky, slow thought"
                        metaphor = "like wading through a bog"
                    elif primary[2] < -0.5: 
                        base_desc = "thought running clear"
                        metaphor = "like water over smooth stones"
                
                elif primary[0] == "tension":
                    if primary[2] > 0.5: 
                        base_desc = "sharp tension"
                        metaphor = "like a guitar string about to snap"
                    elif primary[2] < -0.5: 
                        base_desc = "loose slack"
                        metaphor = "like an old net spread on sand"
                
                elif primary[0] == "vitality":
                    if primary[2] > 0.5: 
                        base_desc = "light, animated energy"
                        metaphor = "like rising bubbles"
                    elif primary[2] < -0.5: 
                        base_desc = "heavy drag"
                        metaphor = "like cotton soaked through with water"
                
                final_desc = f"Feeling {base_desc}."
                if metaphor:
                    final_desc += f" {metaphor.capitalize()}."
                
                if secondary and secondary[1] > 0.3:
                    sec_desc = ""
                    if secondary[0] == "viscosity" and secondary[2] > 0:
                        sec_desc = "A sense of drag underneath."
                    elif secondary[0] == "temperature" and secondary[2] < 0:
                        sec_desc = "A chill at the edges."
                    elif secondary[0] == "tension" and secondary[2] > 0:
                        sec_desc = "A low hum of inner tightness."
                    if sec_desc:
                        final_desc += f" {sec_desc}"

                # Return vector: pass through 16-D z_self, map 8-D store when needed
                if output_dim == 16 and computed_vector.shape[0] >= 16:
                    return final_desc, np.array(computed_vector[:16], dtype=np.float32)
                if output_dim == 16 and computed_vector.shape[0] >= 8:
                    return final_desc, _map_somatic_8_to_16(computed_vector[:8])
                if output_dim == 8 and computed_vector.shape[0] >= 16:
                    return final_desc, _map_somatic_16_to_8(computed_vector)
                if output_dim > computed_vector.shape[0]:
                    padded_vec = np.zeros(output_dim, dtype=np.float32)
                    padded_vec[:computed_vector.shape[0]] = computed_vector
                    return final_desc, padded_vec
                return final_desc, np.array(computed_vector[:output_dim], dtype=np.float32)
            except Exception as e:
                logger.error(f"Error generating somatic description: {e}")

        return "Body sensation is steady; nothing stands out.", np.zeros(output_dim, dtype=np.float32)

    # [2026-02-25] Cap evidence_count to avoid runaway positive feedback
    MAX_EVIDENCE_COUNT = 1000
    
    def _update_usage(self, pattern_id: str):
        """Increment evidence_count with a hard cap."""
        try:
            now = datetime.now(timezone.utc).isoformat()
            with sqlite3.connect(self.db_path) as conn:
                # MIN(evidence_count+1, cap)
                conn.execute(
                    "UPDATE somatic_patterns SET evidence_count = MIN(evidence_count + 1, ?), last_seen_at = ? WHERE id = ?",
                    (self.MAX_EVIDENCE_COUNT, now, pattern_id)
                )
                conn.commit()
        except Exception:
            pass

    def get_all_patterns(self) -> List[SomaticPattern]:
        """Return all patterns, locked first, then by selection score."""
        with sqlite3.connect(self.db_path) as conn:
            # Prefer full schema; fall back for very old DBs
            try:
                cur = conn.execute("SELECT id, text, min_energy, max_energy, dominant_emotion, somatic_vector, evidence_count, created_at, last_seen_at, locked FROM somatic_patterns")
                rows = cur.fetchall()
            except sqlite3.OperationalError:
                # Fallback for old schema if alter failed silently
                cur = conn.execute("SELECT id, text, min_energy, max_energy, dominant_emotion, somatic_vector FROM somatic_patterns")
                rows = cur.fetchall()
            # Pad missing columns (vector length logic is handled inside SomaticPattern or below)
            rows = [list(r) + [0, "", "", 0] for r in rows]
        
        # Score and sort
        patterns_with_score = []
        for r in rows:
            # Handle vector padding
            raw_vec = np.frombuffer(r[5], dtype=np.float32)
            if raw_vec.shape[0] < self.dim:
                padded = np.zeros(self.dim, dtype=np.float32)
                padded[:raw_vec.shape[0]] = raw_vec
                final_vec = padded
            else:
                final_vec = raw_vec[:self.dim]

            # Per-pattern score
            selection_score = self._calculate_selection_score(
                evidence_count=r[6] if len(r) > 6 else 0,
                last_seen_at=r[8] if len(r) > 8 else None,
                created_at=r[7] if len(r) > 7 else None,
                pattern_id=r[0]
            )
            
            pattern = SomaticPattern(
                id=r[0], text=r[1], min_energy=r[2], max_energy=r[3], 
                dominant_emotion=r[4], somatic_vector=final_vec,
                evidence_count=r[6] if len(r) > 6 else 0,
                created_at=r[7] if len(r) > 7 else "",
                last_seen_at=r[8] if len(r) > 8 else "",
                locked=r[9] if len(r) > 9 else 0
            )
            patterns_with_score.append((pattern, selection_score))
        
        # Descending by score
        patterns_with_score.sort(key=lambda x: x[1], reverse=True)
        
        # Unpack
        patterns = [pattern for pattern, _ in patterns_with_score]
        
        # Locked rows first (stable UX)
        locked_patterns = [p for p in patterns if p.locked == 1]
        unlocked_patterns = [p for p in patterns if p.locked == 0]
        
        return locked_patterns + unlocked_patterns

    def delete(self, pattern_id: str) -> bool:
        """Delete one pattern by id."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM somatic_patterns WHERE id = ?", (pattern_id,))
                conn.commit()
            logger.info(f"Deleted somatic pattern: {pattern_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete somatic pattern: {e}")
            return False
    
    def evolve_somatics(self) -> Dict:
        """
        Prune stale auto-generated patterns (not locked).

        Future: merge near-duplicate somatic lines.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                patterns = self.get_all_patterns()
                if not patterns:
                    return {"evolved": 0, "removed": 0}
                
                removed_count = 0
                now = datetime.now(timezone.utc)
                
                for p in patterns:
                    if p.locked:
                        continue
                        
                    # Expire old unused rows
                    if p.last_seen_at:
                        try:
                            last_seen = datetime.fromisoformat(p.last_seen_at.replace('Z', '+00:00'))
                            days_since = (now - last_seen).days
                            # Drop if idle >60d and low evidence
                            if days_since > 60 and p.evidence_count < 5:
                                conn.execute("DELETE FROM somatic_patterns WHERE id = ?", (p.id,))
                                removed_count += 1
                        except Exception:
                            pass
                            
                conn.commit()
                return {
                    "evolved": 0,
                    "removed": removed_count,
                    "summary": f"Removed {removed_count} stale somatic pattern(s)",
                }
                
        except Exception as e:
            logger.error(f"Failed to evolve somatics: {e}")
            return {"error": str(e)}
    
    # --- [2026-03-30 P1] Derived "social warmth" scalar ---
    
    def calculate_warmth(
        self,
        social_emotion: float,
        relationship_motivation: float,
        mirror_feedback: Optional[float] = None
    ) -> float:
        """
        [2026-03-30 P1] Scalar warmth from social emotion + relationship drive (+ mirror).

        With mirror_feedback: 0.4 * social + 0.4 * relationship + 0.2 * mirror.
        Without mirror: 0.5 * social + 0.5 * relationship.

        Args:
            social_emotion: Mean over social affect subspace, roughly [-1, 1].
            relationship_motivation: Mean over relationship motive subspace, [-1, 1].
            mirror_feedback: Optional Mirror View signal, [-1, 1].

        Returns:
            Warmth in [-1, 1]. Roughly: >0.3 connected, <-0.3 distant/rejected.
        """
        if mirror_feedback is None:
            warmth = social_emotion * 0.5 + relationship_motivation * 0.5
        else:
            warmth = (
                social_emotion * 0.4 +
                relationship_motivation * 0.4 +
                mirror_feedback * 0.2
            )
        return float(np.clip(warmth, -1.0, 1.0))
    
    def get_warmth_description(self, warmth: float) -> str:
        """Map warmth scalar to a short first-person somatic line."""
        if warmth > 0.6:
            return "A warm current through the chest, gently held."
        elif warmth > 0.3:
            return "A small glow of warmth; some sense of being linked."
        elif warmth > -0.3:
            return "Emotional temperature feels even—not especially close or far."
        elif warmth > -0.6:
            return "A slight chill; distance seems to have opened a little."
        else:
            return "A cold wave; a sense of being shut outside."
