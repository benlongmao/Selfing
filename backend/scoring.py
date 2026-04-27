#!/usr/bin/env python3
"""
Heuristic scoring for persona candidates: importance, novelty, reliability, coreness.
"""
import os
import numpy as np
from typing import List, Optional, Tuple, Dict
from datetime import datetime, timezone
from backend.persona_store import PersonaStore, PersonaItem
from backend.embedder import get_embedder
import logging

logger = logging.getLogger(__name__)

# Default blend weights (overridable via env)
IMPORTANCE_WEIGHT = float(os.environ.get("SCORING_IMPORTANCE_WEIGHT", "0.5"))
NOVELTY_WEIGHT = float(os.environ.get("SCORING_NOVELTY_WEIGHT", "0.3"))
RELIABILITY_WEIGHT = float(os.environ.get("SCORING_RELIABILITY_WEIGHT", "0.2"))

# Time decay half-life (days)
DECAY_HALF_LIFE_DAYS = float(os.environ.get("DECAY_HALF_LIFE_DAYS", "30.0"))

class ScoringSystem:
    """Embedding-aware persona candidate scorer."""
    
    def __init__(self, db_path: str = "data.db", persona_store: Optional[PersonaStore] = None):
        self.db_path = db_path
        self.persona_store = persona_store or PersonaStore(db_path)
        self.embedder = get_embedder()
    
    def _score_importance(self, text: str) -> float:
        """
        Lightweight importance heuristic (length + modal keywords).

        Production stacks may swap in LLM judges; this stays deterministic.
        """
        length_score = min(len(text) / 50.0, 1.0)

        strong_keywords = [
            "必须", "应该", "优先", "禁止", "避免", "确保", "保护", "永远", "总是", "原则", "核心",
            "must", "should", "prioritize", "forbid", "avoid", "ensure", "protect", "always", "never",
            "principle", "core",
        ]
        weak_keywords = [
            "通常", "倾向", "可能", "建议", "尝试", "喜欢", "偏好", "关注", "重视",
            "usually", "tend", "might", "suggest", "try", "prefer", "care about", "value",
        ]
        
        strong_count = sum(1 for kw in strong_keywords if kw in text)
        weak_count = sum(1 for kw in weak_keywords if kw in text)
        
        # Strong modality bumps the ceiling quickly
        keyword_score = min((strong_count * 1.0 + weak_count * 0.5) / 2.0, 1.0)
        
        # Baseline so empty-ish lines still rank above noise
        base_importance = 0.2
        
        importance = base_importance + 0.4 * length_score + 0.4 * keyword_score
        return float(min(1.0, importance))
    
    def _score_coreness(self, text: str, embedding: np.ndarray) -> float:
        """
        ``coreness`` — how central / distilled a rule feels.

        Blends importance, generality (penalises hyper-specific strings), abstraction cues,
        and ideal length (20–50 graphemes is the sweet spot for this heuristic).

        Args:
            text: Rule body
            embedding: Encoded vector (currently unused but kept for future signals)

        Returns:
            Score in ``[0, 1]``.
        """
        # 1) Importance prior
        importance = self._score_importance(text)
        
        # 2) Generality — penalise obviously specific artefacts
        specific_indicators = [
            r'\d+',
            r'[A-Z][a-z]+ [A-Z][a-z]+',
            r'http[s]?://',
            r'@\w+',
        ]
        import re
        specificity_count = sum(1 for pattern in specific_indicators if re.search(pattern, text))
        # More specificity hits => lower generality
        generality = max(0.0, 1.0 - specificity_count * 0.3)
        
        # 3) Abstraction cues (ZH + EN)
        abstract_keywords = [
            "原则", "价值", "伦理", "方法", "策略", "理念", "准则",
            "应该", "必须", "优先", "避免", "确保", "保护",
            "principle", "value", "ethic", "method", "strategy", "guideline",
            "should", "must", "prioritize", "avoid", "ensure", "protect",
        ]
        abstract_count = sum(1 for kw in abstract_keywords if kw in text)
        abstraction_level = min(1.0, abstract_count / 3.0)
        
        # 4) Length — prefer concise statements without being empty
        length = len(text)
        if 20 <= length <= 50:
            length_score = 1.0
        elif length < 20:
            length_score = length / 20.0
        else:
            length_score = max(0.0, 1.0 - (length - 50) / 100.0)

        # Weighted blend
        coreness = (
            0.3 * importance +
            0.25 * generality +
            0.25 * abstraction_level +
            0.2 * length_score
        )
        
        return float(coreness)
    
    def score_candidate(
        self,
        candidate_text: str,
        candidate_embedding: Optional[np.ndarray] = None,
        evidence_count: int = 1
    ) -> Dict:
        """
        Aggregate score dict for a candidate rule row.
        """
        if candidate_embedding is None:
            candidate_embedding = self.embedder.encode(candidate_text)
        
        # Importance
        importance = self._score_importance(candidate_text)
        
        # Novelty vs active store
        novelty = self._score_novelty(candidate_embedding)
        
        # Reliability from evidence count × optional quality
        reliability = self._score_reliability(evidence_count, evidence_quality=1.0)
        
        # Coreness
        coreness = self._score_coreness(candidate_text, candidate_embedding)
        
        # Weighted total (importance 0.35, novelty 0.25, reliability 0.15, coreness 0.25)
        total_score = (
            0.35 * importance +
            0.25 * novelty +
            0.15 * reliability +
            0.25 * coreness
        )
        
        return {
            "importance": importance,
            "novelty": novelty,
            "reliability": reliability,
            "coreness": coreness,
            "total_score": total_score
        }
    
    def _score_novelty(self, candidate_embedding: np.ndarray) -> float:
        """``1 - max_cosine`` against active persona embeddings."""
        try:
            items = self.persona_store.get_all_active(limit=100)
            if not items:
                return 1.0
            
            max_similarity = 0.0
            for item in items:
                if item.embedding is not None:
                    similarity = np.dot(candidate_embedding, item.embedding) / (
                        np.linalg.norm(candidate_embedding) * np.linalg.norm(item.embedding) + 1e-8
                    )
                    max_similarity = max(max_similarity, float(similarity))
            
            novelty = 1.0 - max_similarity
            return max(0.0, novelty)
        except Exception as e:
            logger.warning(f"Failed to compute novelty: {e}")
            return 0.5
    
    def _score_reliability(self, evidence_count: int, evidence_quality: float = 1.0) -> float:
        """Saturating curve over evidence count scaled by ``evidence_quality``."""
        # reliability = (1 - exp(-evidence_count / threshold)) * evidence_quality
        threshold = 5.0
        count_factor = 1.0 - np.exp(-evidence_count / threshold)
        reliability = count_factor * evidence_quality
        return float(max(0.0, min(1.0, reliability)))
    
    def apply_time_decay(self, score: float, created_at: str) -> float:
        """Exponential half-life decay from ``created_at``."""
        try:
            created = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            now = datetime.now(timezone.utc)
            days_old = (now - created).total_seconds() / 86400.0
            
            decay_factor = np.exp(-days_old * np.log(2) / DECAY_HALF_LIFE_DAYS)
            return float(score * decay_factor)
        except Exception as e:
            logger.warning(f"Failed to apply time decay: {e}")
            return score
    
    def deduplicate(
        self,
        candidates: List[Tuple[str, np.ndarray, Dict]],
        similarity_threshold: float = 0.85
    ) -> List[Tuple[str, np.ndarray, Dict]]:
        """Greedy dedupe by cosine similarity (keep higher ``total_score`` first)."""
        if not candidates:
            return []
        
        candidates_sorted = sorted(candidates, key=lambda x: x[2].get("total_score", 0), reverse=True)
        
        deduplicated = []
        for text, emb, scores in candidates_sorted:
            is_duplicate = False
            for existing_text, existing_emb, existing_scores in deduplicated:
                similarity = np.dot(emb, existing_emb) / (
                    np.linalg.norm(emb) * np.linalg.norm(existing_emb) + 1e-8
                )
                if similarity >= similarity_threshold:
                    is_duplicate = True
                    logger.debug(f"Merged duplicate: '{text[:50]}...' with '{existing_text[:50]}...' (similarity={similarity:.3f})")
                    break
            
            if not is_duplicate:
                deduplicated.append((text, emb, scores))
        
        return deduplicated
    
    def mmr_select(
        self,
        candidates: List[Tuple[str, np.ndarray, Dict]],
        max_items: int = 100,
        lambda_param: float = 0.4,
        existing_items: Optional[List] = None,
        attention_focus: float = 0.5,
    ) -> List[Tuple[str, np.ndarray, Dict]]:
        """
        MMR selection with optional attention gating.

        High ``attention_focus`` (>0.7) tightens the candidate pool; low focus (<0.3)
        nudges ``lambda`` toward diversity.
        """
        if not candidates:
            return []

        # Attention pre-filter
        filtered_candidates = candidates
        if attention_focus > 0.7:
            # Keep items >= 60% of the current max total_score
            max_score = max((c[2].get("total_score", 0) for c in candidates), default=1.0)
            threshold = max_score * 0.6
            filtered_candidates = [c for c in candidates if c[2].get("total_score", 0) >= threshold]
            
            # Always keep at least three candidates
            if len(filtered_candidates) < 3:
                candidates_sorted = sorted(candidates, key=lambda x: x[2].get("total_score", 0), reverse=True)
                filtered_candidates = candidates_sorted[:3]
                
            logger.debug(f"Attention Gating (Focus={attention_focus:.2f}): Filtered {len(candidates)} -> {len(filtered_candidates)}")
        
        if len(filtered_candidates) <= max_items:
            return filtered_candidates
        
        # Stability bonus for long-lived rows
        from datetime import datetime, timezone
        STABILITY_DAYS = 30
        STABILITY_BOOST = 0.1
        
        existing_texts = set()
        existing_ages = {}  # text -> age_in_days
        
        if existing_items:
            now = datetime.now(timezone.utc)
            for item in existing_items:
                if hasattr(item, 'text') and hasattr(item, 'created_at'):
                    existing_texts.add(item.text)
                    try:
                        created = datetime.fromisoformat(item.created_at.replace('Z', '+00:00'))
                        age_days = (now - created).total_seconds() / 86400.0
                        existing_ages[item.text] = age_days
                    except Exception:
                        existing_ages[item.text] = 0.0

        if attention_focus < 0.3:
            lambda_param = min(0.7, lambda_param + 0.2)
        elif attention_focus > 0.7:
            lambda_param = max(0.2, lambda_param - 0.1)

        candidates_with_stability = []
        for text, emb, scores in filtered_candidates:
            stability_boost = 0.0
            if text in existing_ages:
                age_days = existing_ages[text]
                if age_days >= STABILITY_DAYS:
                    stability_boost = STABILITY_BOOST * min(1.0, age_days / 90.0)
            
            adjusted_scores = scores.copy()
            adjusted_scores["total_score"] = scores.get("total_score", 0) + stability_boost
            adjusted_scores["stability_boost"] = stability_boost
            
            candidates_with_stability.append((text, emb, adjusted_scores))
        
        candidates_sorted = sorted(candidates_with_stability, key=lambda x: x[2].get("total_score", 0), reverse=True)
        
        selected = []
        remaining = candidates_sorted.copy()
        
        if remaining:
            selected.append(remaining.pop(0))
        
        while len(selected) < max_items and remaining:
            best_idx = 0
            best_mmr = -float('inf')
            
            for i, (text, emb, scores) in enumerate(remaining):
                relevance = scores.get("total_score", 0)
                
                min_similarity = 1.0
                for sel_text, sel_emb, sel_scores in selected:
                    similarity = np.dot(emb, sel_emb) / (
                        np.linalg.norm(emb) * np.linalg.norm(sel_emb) + 1e-8
                    )
                    min_similarity = min(min_similarity, float(similarity))
                
                diversity = 1.0 - min_similarity
                
                mmr = lambda_param * relevance + (1 - lambda_param) * diversity
                
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_idx = i
            
            selected.append(remaining.pop(best_idx))
        
        return selected

