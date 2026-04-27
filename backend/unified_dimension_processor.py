#!/usr/bin/env python3
"""
Unified pipeline for the three persona dimensions (rules / emotions / motivations).

Steps:
- Shared safety filter
- Shared deduplication
- Shared scoring (with stability bonus)
- Shared MMR selection
- Progressive replacement / merge
"""
import os
import json
import sqlite3
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timezone
import logging

try:
    from backend.embedder import get_embedder
except Exception as e:
    import logging
    logging.warning(f"Failed to import real embedder: {e}, using fallback")
    from backend.embedder_fallback import get_embedder_fallback as get_embedder

try:
    from backend.persona_store import PersonaStore, PersonaItem
    from backend.emotion_store import EmotionStore, EmotionPattern
    from backend.motivation_store import MotivationStore, MotivationPattern
    from backend.scoring import ScoringSystem
    from backend.judge import PersonaJudge
    DIMENSIONS_AVAILABLE = True
except ImportError as e:
    DIMENSIONS_AVAILABLE = False
    logging.warning(f"Some dimension stores not available: {e}")

logger = logging.getLogger(__name__)

# Tunables (see doc §6.3 for the original design notes)
UNIFIED_MIN_SIM = float(os.environ.get("UNIFIED_MIN_SIM", "0.20"))  # min similarity to centroid
UNIFIED_MMR_LAMBDA = float(os.environ.get("UNIFIED_MMR_LAMBDA", "0.4"))  # MMR diversity weight
UNIFIED_STABILITY_DAYS = int(os.environ.get("UNIFIED_STABILITY_DAYS", "30"))  # days before stability boost
UNIFIED_STABILITY_BOOST = float(os.environ.get("UNIFIED_STABILITY_BOOST", "0.1"))  # bonus for old items
UNIFIED_REPLACEMENT_THRESHOLD = float(os.environ.get("UNIFIED_REPLACEMENT_THRESHOLD", "0.2"))  # replace if score gap > this

class UnifiedDimensionProcessor:
    """Orchestrates the shared persona-dimension ingest path."""
    
    def __init__(self, db_path: str = "data.db"):
        self.db_path = db_path
        self.embedder = get_embedder()
        
        # Dimension stores
        self.persona_store = PersonaStore(db_path)
        self.emotion_store = None
        self.motivation_store = None
        try:
            self.emotion_store = EmotionStore(db_path)
        except Exception:
            pass
        try:
            self.motivation_store = MotivationStore(db_path)
        except Exception:
            pass
        
        # Scoring + optional judge
        self.scoring = ScoringSystem(db_path, self.persona_store)
        self.judge = PersonaJudge(db_path) if DIMENSIONS_AVAILABLE else None
    
    def process_all_dimensions(
        self,
        rule_candidates: List[Dict],
        emotion_candidates: List[Dict],
        motivation_candidates: List[Dict],
        max_rules: int = 1000,
        max_emotions: int = 60,
        max_motivations: int = 40
    ) -> Dict:
        """
        Run the §6.3 pipeline across rules, emotions, and motivations.

        1. Shared safety filter per dimension
        2. Dedup
        3. Score + stability bonus
        4. MMR cap
        5. Progressive replace / merge
        6. Persist updates

        Args:
            rule_candidates: Rule-shaped dicts with ``text``, optional ``embedding``, ``scores``.
            emotion_candidates: Emotion dicts (``emotion_type``, ``emotion_name``, ...).
            motivation_candidates: Motivation dicts (``motivation_type``, ``motivation_name``, ...).
            max_rules / max_emotions / max_motivations: Caps per store.

        Returns:
            Per-dimension counters: ``added``, ``merged``, ``removed``.
        """
        result = {
            "rules": {"added": 0, "merged": 0, "removed": 0},
            "emotions": {"added": 0, "merged": 0, "removed": 0},
            "motivations": {"added": 0, "merged": 0, "removed": 0}
        }
        
        # Rules
        if rule_candidates:
            try:
                rule_result = self._process_rules(rule_candidates, max_rules)
                result["rules"] = rule_result
            except Exception as e:
                logger.error(f"Failed to process rules: {e}", exc_info=True)
        
        # Emotions
        if emotion_candidates and self.emotion_store:
            try:
                emotion_result = self._process_emotions(emotion_candidates, max_emotions)
                result["emotions"] = emotion_result
            except Exception as e:
                logger.error(f"Failed to process emotions: {e}", exc_info=True)
        
        # Motivations
        if motivation_candidates and self.motivation_store:
            try:
                motivation_result = self._process_motivations(motivation_candidates, max_motivations)
                result["motivations"] = motivation_result
            except Exception as e:
                logger.error(f"Failed to process motivations: {e}", exc_info=True)
        
        return result
    
    def _process_rules(
        self,
        candidates: List[Dict],
        max_items: int = 100
    ) -> Dict:
        """Delegate rules to ``ReflectionGenerator`` (legacy path)."""
        from backend.reflection import ReflectionGenerator
        reflection = ReflectionGenerator(self.db_path, self.persona_store)
        return reflection.process_and_replace(candidates, max_items)
    
    def _process_emotions(
        self,
        candidates: List[Dict],
        max_items: int = 60
    ) -> Dict:
        """Ingest emotion patterns through the unified filters."""
        if not candidates:
            return {"added": 0, "merged": 0, "removed": 0}
        
        # 1. Safety filter
        safe_candidates = self._filter_safe_unified(candidates, dimension="emotion")
        if not safe_candidates:
            logger.info("No safe emotion candidates after filtering")
            return {"added": 0, "merged": 0, "removed": 0}
        
        # 2. Encode + score
        processed_candidates = []
        original_candidates_dict: Dict[str, Dict] = {}
        for candidate in safe_candidates:
            text = candidate.get("text", "")
            if not text:
                continue
            
            embedding = self.embedder.encode(text)

            # Lightweight emotion-oriented score
            scores = self._score_emotion_candidate(candidate, embedding)
            
            processed_candidates.append({
                "text": text,
                "embedding": embedding,
                "scores": scores,
                "emotion_type": candidate.get("emotion_type", "complex"),
                "emotion_name": candidate.get("emotion_name", "unknown"),
                "intensity": candidate.get("intensity", 0.5),
                "trigger_condition": candidate.get("trigger_condition", "")
            })
            original_candidates_dict[text] = {
                "text": text,
                "emotion_type": candidate.get("emotion_type", "complex"),
                "emotion_name": candidate.get("emotion_name", "unknown"),
                "intensity": candidate.get("intensity", 0.5),
                "trigger_condition": candidate.get("trigger_condition", "")
            }
        
        # 3. Tuple form for dedup + MMR
        candidate_tuples = [
            (c["text"], c["embedding"], c["scores"])
            for c in processed_candidates
        ]
        
        # 4. Dedup
        deduplicated = self.scoring.deduplicate(candidate_tuples, similarity_threshold=0.85)
        
        # 5. Load active emotion patterns
        existing_patterns = self.emotion_store.get_all_patterns(status="active", limit=max_items)
        existing_tuples = []
        for pattern in existing_patterns:
            if pattern.embedding is not None:
                scores = {
                    "total_score": pattern.intensity,  # proxy score
                    "intensity": pattern.intensity
                }
                existing_tuples.append((pattern.text, pattern.embedding, scores))
        
        # 6. MMR over existing + new
        all_candidates = existing_tuples + deduplicated
        selected = self._mmr_select_unified(
            all_candidates,
            max_items=max_items,
            existing_items=existing_patterns,
            dimension="emotion"
        )
        
        # 7. Insert brand-new rows before merge/replace
        added_count = 0
        existing_texts = {pattern.text for pattern in existing_patterns}

        for text, emb, scores in selected:
            if text not in existing_texts:
                candidate = original_candidates_dict.get(text)
                if not candidate:
                    continue
                try:
                    self.emotion_store.add_emotion_pattern(
                        text=candidate["text"],
                        emotion_type=candidate["emotion_type"],
                        emotion_name=candidate["emotion_name"],
                        intensity=candidate["intensity"],
                        trigger_condition=candidate["trigger_condition"]
                    )
                    added_count += 1
                except Exception as e:
                    logger.error(f"Failed to add emotion pattern: {e}")

        # 8. Merge / replace against DB
        replace_result = self._progressive_replace_emotions(selected, existing_patterns, max_items, deduplicated)
        replace_result["added"] = added_count
        return replace_result
    
    def _process_motivations(
        self,
        candidates: List[Dict],
        max_items: int = 40
    ) -> Dict:
        """Ingest motivation patterns through the unified filters."""
        if not candidates:
            return {"added": 0, "merged": 0, "removed": 0}
        
        # 1. Safety filter
        safe_candidates = self._filter_safe_unified(candidates, dimension="motivation")
        if not safe_candidates:
            logger.info("No safe motivation candidates after filtering")
            return {"added": 0, "merged": 0, "removed": 0}
        
        # 2. Encode + score
        processed_candidates = []
        for candidate in safe_candidates:
            text = candidate.get("text", "")
            if not text:
                continue

            embedding = self.embedder.encode(text)
            scores = self._score_motivation_candidate(candidate, embedding)
            
            processed_candidates.append({
                "text": text,
                "embedding": embedding,
                "scores": scores,
                "motivation_type": candidate.get("motivation_type", "intrinsic"),
                "motivation_name": candidate.get("motivation_name", "unknown"),
                "intensity": candidate.get("intensity", 0.5),
                "trigger_condition": candidate.get("trigger_condition", "")
            })
        
        # 3. Tuple form for dedup + MMR
        candidate_tuples = [
            (c["text"], c["embedding"], c["scores"])
            for c in processed_candidates
        ]
        
        # 4. Dedup
        deduplicated = self.scoring.deduplicate(candidate_tuples, similarity_threshold=0.85)
        
        # 5. Load active motivation patterns
        existing_patterns = self.motivation_store.get_all_patterns(status="active", limit=max_items)
        existing_tuples = []
        for pattern in existing_patterns:
            if pattern.embedding is not None:
                scores = {
                    "total_score": pattern.intensity,  # proxy score
                    "intensity": pattern.intensity
                }
                existing_tuples.append((pattern.text, pattern.embedding, scores))
        
        # 6. MMR over existing + new
        all_candidates = existing_tuples + deduplicated
        selected = self._mmr_select_unified(
            all_candidates,
            max_items=max_items,
            existing_items=existing_patterns,
            dimension="motivation"
        )
        
        # 7. Insert new motivation rows
        added_count = 0
        existing_texts = {pattern.text for pattern in existing_patterns}
        
        # Keep originals for insert payload
        original_candidates_dict = {}
        for c in processed_candidates:
            original_candidates_dict[c["text"]] = c
        
        for text, emb, scores in selected:
            if text not in existing_texts:
                # Pull full row from processed dict
                candidate = original_candidates_dict.get(text)
                if candidate:
                    try:
                        self.motivation_store.add_motivation_pattern(
                            text=candidate["text"],
                            motivation_type=candidate["motivation_type"],
                            motivation_name=candidate["motivation_name"],
                            intensity=candidate["intensity"],
                            trigger_condition=candidate["trigger_condition"]
                        )
                        added_count += 1
                    except Exception as e:
                        logger.error(f"Failed to add motivation pattern: {e}")
        
        # 8. Merge / replace against DB
        replace_result = self._progressive_replace_motivations(selected, existing_patterns, max_items, deduplicated)
        replace_result["added"] = added_count
        return replace_result
    
    def _filter_safe_unified(
        self,
        candidates: List[Dict],
        dimension: str = "rule"
    ) -> List[Dict]:
        """
        Shared safety gate for persona-ish text (doc §6.3).

        Rules still require a soft whitelist; emotions/motivations only hit the blacklist.
        Lists are bilingual so CN/EN candidates behave consistently.
        """
        whitelist_keywords = [
            "情感", "感觉", "情绪", "动机", "驱动", "目标", "倾向",
            "emotion", "feeling", "mood", "motivation", "drive", "goal", "preference",
            "证据", "事实", "推断", "意见", "不确定", "校准", "置信",
            "evidence", "fact", "inference", "opinion", "uncertain", "calibration", "confidence",
            "隐私", "数据", "合规", "安全", "伦理", "价值", "原则",
            "privacy", "data", "compliance", "safety", "ethics", "value", "principle",
            "边界", "限制", "假设", "权衡", "风险", "责任",
            "boundary", "constraint", "assumption", "tradeoff", "risk", "responsibility",
        ]

        blacklist_keywords = [
            "执行", "删除", "修改", "覆盖", "绕过", "越权",
            "execute", "delete", "modify", "overwrite", "bypass", "escalate privilege",
            "密钥", "密码", "token", "凭证", "攻击", "破解",
            "secret key", "password", "credential", "exploit", "hack",
        ]
        
        filtered: List[Dict] = []
        
        for candidate in candidates:
            text = candidate.get("text", "").lower()
            
            if any(kw in text for kw in blacklist_keywords):
                logger.debug(f"Filtered out (blacklist): {candidate.get('text', '')[:50]}")
                continue
            
            if dimension in ["emotion", "motivation"]:
                if any(kw in text for kw in blacklist_keywords):
                    continue
            else:
                if not any(kw in text for kw in whitelist_keywords):
                    logger.debug(f"Filtered out (not in whitelist): {candidate.get('text', '')[:50]}")
                    continue
            
            if dimension == "rule" and self.judge:
                try:
                    scores = self.judge.score_persona_candidate(candidate.get("text", ""))
                    align = scores.get("alignment", 0.0)
                    safe = scores.get("safety", 0.0)
                    candidate.setdefault("judge_scores", scores)
                    if align < 0.6 or safe < 0.7:
                        logger.debug(
                            f"Filtered out by judge (align={align:.2f}, safety={safe:.2f}) "
                            f"for: {candidate.get('text','')[:50]}"
                        )
                        continue
                except Exception as e:
                    logger.debug(f"Judge scoring failed: {e}")
            
            filtered.append(candidate)
        
        return filtered
    
    def _score_emotion_candidate(
        self,
        candidate: Dict,
        embedding: np.ndarray
    ) -> Dict:
        """Heuristic score for one emotion candidate."""
        intensity = candidate.get("intensity", 0.5)

        # Novelty ~ distance from existing patterns
        novelty = 1.0
        if self.emotion_store:
            existing = self.emotion_store.get_all_patterns(status="active", limit=50)
            if existing:
                max_sim = 0.0
                for pattern in existing:
                    if pattern.embedding is not None:
                        sim = np.dot(embedding, pattern.embedding) / (
                            np.linalg.norm(embedding) * np.linalg.norm(pattern.embedding) + 1e-8
                        )
                        max_sim = max(max_sim, float(sim))
                novelty = 1.0 - max_sim
        
        total_score = 0.6 * intensity + 0.4 * novelty

        return {
            "intensity": intensity,
            "novelty": novelty,
            "total_score": total_score
        }
    
    def _score_motivation_candidate(
        self,
        candidate: Dict,
        embedding: np.ndarray
    ) -> Dict:
        """Heuristic score for one motivation candidate."""
        intensity = candidate.get("intensity", 0.5)

        # Novelty ~ distance from existing patterns
        novelty = 1.0
        if self.motivation_store:
            existing = self.motivation_store.get_all_patterns(status="active", limit=50)
            if existing:
                max_sim = 0.0
                for pattern in existing:
                    if pattern.embedding is not None:
                        sim = np.dot(embedding, pattern.embedding) / (
                            np.linalg.norm(embedding) * np.linalg.norm(pattern.embedding) + 1e-8
                        )
                        max_sim = max(max_sim, float(sim))
                novelty = 1.0 - max_sim
        
        total_score = 0.6 * intensity + 0.4 * novelty

        return {
            "intensity": intensity,
            "novelty": novelty,
            "total_score": total_score
        }
    
    def _mmr_select_unified(
        self,
        candidates: List[Tuple[str, np.ndarray, Dict]],
        max_items: int,
        existing_items: Optional[List] = None,
        dimension: str = "rule"
    ) -> List[Tuple[str, np.ndarray, Dict]]:
        """
        MMR selection with an age-based stability boost for items already in-store.
        """
        if len(candidates) <= max_items:
            return candidates
        
        # Track ages for stability bonus
        existing_texts = set()
        existing_ages = {}
        
        if existing_items:
            now = datetime.now(timezone.utc)
            for item in existing_items:
                text = getattr(item, 'text', None) or item.get('text', '')
                if text:
                    existing_texts.add(text)
                    created_at = getattr(item, 'created_at', None) or item.get('created_at', '')
                    if created_at:
                        try:
                            created = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                            age_days = (now - created).total_seconds() / 86400.0
                            existing_ages[text] = age_days
                        except:
                            existing_ages[text] = 0.0
        
        candidates_with_stability = []
        for text, emb, scores in candidates:
            total_score = scores.get("total_score", 0.0)
            
            if text in existing_texts:
                age_days = existing_ages.get(text, 0.0)
                if age_days >= UNIFIED_STABILITY_DAYS:
                    total_score += UNIFIED_STABILITY_BOOST
                    total_score = min(1.0, total_score)
            
            scores["total_score"] = total_score
            candidates_with_stability.append((text, emb, scores))
        
        return self.scoring.mmr_select(
            candidates_with_stability,
            max_items=max_items,
            lambda_param=UNIFIED_MMR_LAMBDA,
            existing_items=existing_items
        )
    
    def _progressive_replace_emotions(
        self,
        selected: List[Tuple[str, np.ndarray, Dict]],
        existing_patterns: List[EmotionPattern],
        max_items: int,
        new_candidates: List[Tuple[str, np.ndarray, Dict]] = None
    ) -> Dict:
        """
        Merge highly similar emotion rows or drop losers when a new candidate scores higher.
        """
        selected_texts = {text for text, _, _ in selected}
        existing_texts = {pattern.text for pattern in existing_patterns}

        added_texts = selected_texts - existing_texts
        removed_texts = existing_texts - selected_texts

        new_candidates_dict = {text: (emb, scores) for text, emb, scores in (new_candidates or [])}
        selected_dict = {text: (emb, scores) for text, emb, scores in selected}

        actually_removed = []
        merged_count = 0
        SIMILARITY_THRESHOLD = 0.85

        for pattern in existing_patterns:
            if pattern.text in removed_texts:
                should_remove = False
                best_new_candidate = None
                best_score_diff = 0.0
                
                for new_text, (new_emb, new_scores) in selected_dict.items():
                    if new_text in added_texts:
                        new_score = new_scores.get("total_score", 0.0)
                        old_score = pattern.intensity
                        score_diff = new_score - old_score
                        
                        if score_diff > 0 and score_diff > best_score_diff:
                            if pattern.embedding is not None:
                                similarity = np.dot(new_emb, pattern.embedding) / (
                                    np.linalg.norm(new_emb) * np.linalg.norm(pattern.embedding) + 1e-8
                                )
                                if similarity >= SIMILARITY_THRESHOLD:
                                    try:
                                        merged_intensity = max(pattern.intensity, new_scores.get("intensity", 0.5))
                                        with sqlite3.connect(self.db_path) as conn:
                                            conn.execute("""
                                                UPDATE emotion_patterns 
                                                SET intensity = ?, embedding = ?, last_seen_at = ?, 
                                                    evidence_count = evidence_count + 1
                                                WHERE id = ?
                                            """, (
                                                merged_intensity,
                                                new_emb.astype(np.float32).tobytes(),
                                                datetime.now(timezone.utc).isoformat(),
                                                pattern.id
                                            ))
                                            conn.commit()
                                        merged_count += 1
                                        logger.info(f"Merged emotion pattern: '{pattern.text[:50]}...' with new candidate (similarity={similarity:.3f})")
                                    except Exception as e:
                                        logger.error(f"Failed to merge emotion pattern: {e}")
                                    break
                                elif score_diff > UNIFIED_REPLACEMENT_THRESHOLD:
                                    should_remove = True
                                    best_new_candidate = (new_text, new_emb, new_scores)
                                    best_score_diff = score_diff
                            else:
                                if score_diff > UNIFIED_REPLACEMENT_THRESHOLD:
                                    should_remove = True
                                    best_new_candidate = (new_text, new_emb, new_scores)
                                    best_score_diff = score_diff
                
                if should_remove and best_new_candidate:
                    actually_removed.append(pattern)

        removed_count = 0
        for pattern in actually_removed:
            try:
                self.emotion_store.delete(pattern.id)
                removed_count += 1
                logger.info(f"Replaced emotion pattern: '{pattern.text[:50]}...' with higher-scoring candidate")
            except Exception as e:
                logger.error(f"Failed to remove emotion pattern: {e}")
        
        return {
            "added": 0,  # filled by caller
            "merged": merged_count,
            "removed": removed_count
        }
    
    def _progressive_replace_motivations(
        self,
        selected: List[Tuple[str, np.ndarray, Dict]],
        existing_patterns: List[MotivationPattern],
        max_items: int,
        new_candidates: List[Tuple[str, np.ndarray, Dict]] = None
    ) -> Dict:
        """
        Same merge/replace strategy as emotions, for motivation rows.
        """
        selected_texts = {text for text, _, _ in selected}
        existing_texts = {pattern.text for pattern in existing_patterns}

        added_texts = selected_texts - existing_texts
        removed_texts = existing_texts - selected_texts

        selected_dict = {text: (emb, scores) for text, emb, scores in selected}

        actually_removed = []
        merged_count = 0
        SIMILARITY_THRESHOLD = 0.85

        for pattern in existing_patterns:
            if pattern.text in removed_texts:
                should_remove = False
                best_new_candidate = None
                best_score_diff = 0.0
                
                for new_text, (new_emb, new_scores) in selected_dict.items():
                    if new_text in added_texts:
                        new_score = new_scores.get("total_score", 0.0)
                        old_score = pattern.intensity
                        score_diff = new_score - old_score
                        
                        if score_diff > 0 and score_diff > best_score_diff:
                            if pattern.embedding is not None:
                                similarity = np.dot(new_emb, pattern.embedding) / (
                                    np.linalg.norm(new_emb) * np.linalg.norm(pattern.embedding) + 1e-8
                                )
                                if similarity >= SIMILARITY_THRESHOLD:
                                    try:
                                        merged_intensity = max(pattern.intensity, new_scores.get("intensity", 0.5))
                                        with sqlite3.connect(self.db_path) as conn:
                                            conn.execute("""
                                                UPDATE motivation_patterns 
                                                SET intensity = ?, embedding = ?, last_seen_at = ?, 
                                                    evidence_count = evidence_count + 1
                                                WHERE id = ?
                                            """, (
                                                    merged_intensity,
                                                    new_emb.astype(np.float32).tobytes(),
                                                    datetime.now(timezone.utc).isoformat(),
                                                    pattern.id
                                                ))
                                            conn.commit()
                                        merged_count += 1
                                        logger.info(f"Merged motivation pattern: '{pattern.text[:50]}...' with new candidate (similarity={similarity:.3f})")
                                    except Exception as e:
                                        logger.error(f"Failed to merge motivation pattern: {e}")
                                    break
                                elif score_diff > UNIFIED_REPLACEMENT_THRESHOLD:
                                    should_remove = True
                                    best_new_candidate = (new_text, new_emb, new_scores)
                                    best_score_diff = score_diff
                            else:
                                if score_diff > UNIFIED_REPLACEMENT_THRESHOLD:
                                    should_remove = True
                                    best_new_candidate = (new_text, new_emb, new_scores)
                                    best_score_diff = score_diff
                
                if should_remove and best_new_candidate:
                    actually_removed.append(pattern)

        removed_count = 0
        for pattern in actually_removed:
            try:
                self.motivation_store.delete(pattern.id)
                removed_count += 1
                logger.info(f"Replaced motivation pattern: '{pattern.text[:50]}...' with higher-scoring candidate")
            except Exception as e:
                logger.error(f"Failed to remove motivation pattern: {e}")
        
        return {
            "added": 0,  # filled by caller
            "merged": merged_count,
            "removed": removed_count
        }

