import logging
import numpy as np
from typing import Dict, List

from backend.embedder import get_embedder


def process_somatic_with_mmr(chat_service, candidates: List[Dict], session_id: str) -> Dict:
    """Run MMR selection for somatic pattern candidates vs the live store."""
    from backend.somatic_store import MAX_SOMATIC_PATTERNS
    from backend.scoring import ScoringSystem

    if not candidates:
        return {"added": 0, "merged": 0, "removed": 0}

    log = getattr(chat_service, "logger", None) or logging.getLogger(__name__)
    somatic_store = chat_service.self_model.somatic_store
    scoring = ScoringSystem(chat_service.db_path)
    embedder = get_embedder()

    # 1) Score + embed each candidate row
    processed_candidates = []
    original_candidates_dict = {}
    for c in candidates:
        text = c.get("text", "")
        if not text:
            continue
        embedding = embedder.encode(text)
        score = c.get("confidence", 0.7)
        scores = {"total_score": score, "confidence": score}
        processed_candidates.append((text, embedding, scores))
        original_candidates_dict[text] = c

    # 2) Near-duplicate collapse
    deduplicated = scoring.deduplicate(processed_candidates, similarity_threshold=0.85)

    # 3) Pull existing patterns and embed them for comparison
    existing_patterns = somatic_store.get_all_patterns()
    existing_patterns = [p for p in existing_patterns if not p.locked][:MAX_SOMATIC_PATTERNS]
    existing_tuples = []
    for pattern in existing_patterns:
        emb = embedder.encode(pattern.text)
        scores = {"total_score": min(1.0, pattern.evidence_count / 10.0)}
        existing_tuples.append((pattern.text, emb, scores))

    # 4) MMR mix of legacy + new candidates
    all_candidates = existing_tuples + deduplicated
    selected = scoring.mmr_select(
        all_candidates,
        max_items=MAX_SOMATIC_PATTERNS,
        lambda_param=0.4,
        existing_items=existing_patterns,
    )

    # 5) Diff selected vs existing
    existing_texts = {text for text, _, _ in existing_tuples}
    selected_texts = {text for text, _, _ in selected}
    to_add = [t for t in selected if t[0] not in existing_texts]
    to_remove = [t for t in existing_tuples if t[0] not in selected_texts]

    # 6) Persist additions
    added_count = 0
    for text, emb, scores in to_add:
        c = original_candidates_dict.get(text)
        if not c:
            continue
        try:
            somatic_store.add_pattern(
                c["text"],
                c["min_energy"],
                c["max_energy"],
                c["dominant_emotion"],
                c["tension"],
                c["vitality"],
                c.get("temperature", 0.0),
                c.get("viscosity", 0.0),
            )
            added_count += 1
        except Exception as e:
            log.error(f"Failed to add somatic pattern: {e}")

    # 7) Gradual eviction of low-value rows when replacements score higher
    removed_count = 0
    if to_remove:
        avg_new_score = (
            np.mean([scores.get("total_score", 0) for _, _, scores in to_add]) if to_add else 0.0
        )
        SCORE_DIFF_THRESHOLD = 0.2

        for text, _, scores in to_remove:
            old_score = scores.get("total_score", 0)
            if avg_new_score - old_score <= SCORE_DIFF_THRESHOLD:
                continue
            for pattern in existing_patterns:
                if pattern.text != text:
                    continue
                try:
                    somatic_store.delete(pattern.id)
                    removed_count += 1
                except Exception as e:
                    log.error(f"Failed to remove somatic pattern: {e}")
                break

    return {"added": added_count, "merged": 0, "removed": removed_count}


def process_worldview_with_mmr(chat_service, candidates: List[Dict], session_id: str) -> Dict:
    """Run MMR selection for worldview beliefs vs the active store."""
    from backend.world_store import MAX_WORLDVIEW_BELIEFS
    from backend.scoring import ScoringSystem

    if not candidates:
        return {"added": 0, "merged": 0, "removed": 0}

    log = getattr(chat_service, "logger", None) or logging.getLogger(__name__)
    world_store = chat_service.self_model.world_store
    scoring = ScoringSystem(chat_service.db_path)
    embedder = get_embedder()

    # 1) Score + embed incoming beliefs
    processed_candidates = []
    original_candidates_dict = {}
    for c in candidates:
        text = c.get("text", "")
        if not text:
            continue
        embedding = embedder.encode(text)
        score = c.get("confidence", 0.7)
        scores = {"total_score": score, "confidence": score}
        processed_candidates.append((text, embedding, scores))
        original_candidates_dict[text] = c

    # 2) Collapse duplicates
    deduplicated = scoring.deduplicate(processed_candidates, similarity_threshold=0.85)

    # 3) Embed current active beliefs
    existing_beliefs = world_store.get_all_beliefs(status="active", limit=MAX_WORLDVIEW_BELIEFS)
    existing_tuples = []
    for belief in existing_beliefs:
        emb = belief.embedding
        if emb is None or len(emb) < 16:
            emb = embedder.encode(belief.text)
        scores = {"total_score": belief.confidence}
        existing_tuples.append((belief.text, emb, scores))

    # 4) MMR pool
    all_candidates = existing_tuples + deduplicated
    selected = scoring.mmr_select(
        all_candidates,
        max_items=MAX_WORLDVIEW_BELIEFS,
        lambda_param=0.4,
        existing_items=existing_beliefs,
    )

    # 5) Compute add/remove sets
    existing_texts = {text for text, _, _ in existing_tuples}
    selected_texts = {text for text, _, _ in selected}
    to_add = [t for t in selected if t[0] not in existing_texts]
    to_remove = [t for t in existing_tuples if t[0] not in selected_texts]

    # 6) Insert new beliefs
    added_count = 0
    for text, emb, scores in to_add:
        c = original_candidates_dict.get(text)
        if not c:
            continue
        try:
            world_store.add_belief(
                c.get("text", ""),
                c.get("confidence", 0.7),
                c.get("optimism", 0.5),
                c.get("agency", 0.5),
            )
            added_count += 1
        except Exception as e:
            log.error(f"Failed to add worldview belief: {e}")

    # 7) Deactivate losers when replacements are materially stronger
    removed_count = 0
    if to_remove:
        avg_new_score = (
            np.mean([scores.get("total_score", 0) for _, _, scores in to_add]) if to_add else 0.0
        )
        SCORE_DIFF_THRESHOLD = 0.2

        for text, _, scores in to_remove:
            old_score = scores.get("total_score", 0)
            if avg_new_score - old_score <= SCORE_DIFF_THRESHOLD:
                continue
            for belief in existing_beliefs:
                if belief.text != text:
                    continue
                try:
                    world_store.deactivate_belief(belief.id)
                    removed_count += 1
                except Exception as e:
                    log.error(f"Failed to remove worldview belief: {e}")
                break

    return {"added": added_count, "merged": 0, "removed": removed_count}
