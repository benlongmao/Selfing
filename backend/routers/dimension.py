from fastapi import APIRouter, Query, HTTPException
from typing import Optional
import logging
from backend.database import DB_PATH
from backend.emotion_store import EmotionStore
from backend.motivation_store import MotivationStore
from backend.persona_store import PersonaStore
from backend.dimension_metrics import DimensionMetrics
from backend.self_model import SelfModel

router = APIRouter(prefix="/dimension", tags=["Dimension"])
logger = logging.getLogger(__name__)
# DB_PATH imported from backend.database

@router.get("/metrics")
def get_dimension_metrics(sessionId: str = Query(...)):
    try:
        metrics_calculator = DimensionMetrics(DB_PATH)
        emotion_store = EmotionStore(DB_PATH)
        motivation_store = MotivationStore(DB_PATH)
        persona_store = PersonaStore(DB_PATH)
        
        emotion_metrics = metrics_calculator.compute_emotion_consistency(
            sessionId, emotion_store, persona_store
        )
        motivation_metrics = metrics_calculator.compute_motivation_consistency(
            sessionId, motivation_store, persona_store
        )
        interaction_metrics = metrics_calculator.compute_dimension_interaction_effectiveness(
            sessionId
        )
        
        return {
            "sessionId": sessionId,
            "emotionConsistency": emotion_metrics,
            "motivationConsistency": motivation_metrics,
            "dimensionInteraction": interaction_metrics
        }
    except Exception as e:
        logger.error(f"Failed to get dimension metrics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/state")
def get_dimension_state(sessionId: str = Query(...)):
    try:
        persona_store = PersonaStore(DB_PATH)
        self_model = SelfModel(DB_PATH, persona_store)
        
        z_self = self_model.get_z_self(sessionId)
        if not z_self:
            raise HTTPException(status_code=404, detail="Self state not found")
        
        summary = self_model.get_structured_summary(sessionId)
        
        result = {
            "sessionId": sessionId,
            "dimension": self_model.dim,
            "zSelf": z_self.tolist(),
            "summary": summary
        }
        
        if self_model.emotion_store and z_self.shape[0] >= 48:
            emotion_state = self_model.emotion_store.get_emotion_state(sessionId)
            if emotion_state:
                result["emotion"] = {
                    "dominantEmotion": emotion_state.dominant_emotion,
                    "intensity": float(emotion_state.intensity),
                    "vector": emotion_state.emotion_vector.tolist()
                }
        
        if self_model.motivation_store and z_self.shape[0] >= 64:
            motivation_state = self_model.motivation_store.get_motivation_state(sessionId)
            if motivation_state:
                result["motivation"] = {
                    "dominantMotivation": motivation_state.dominant_motivation,
                    "intensity": float(motivation_state.intensity),
                    "vector": motivation_state.motivation_vector.tolist()
                }
        
        return result
    except Exception as e:
        logger.error(f"Failed to get dimension state: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/patterns")
def get_dimension_patterns(limit: int = Query(50, ge=1, le=200)):
    try:
        result = {
            "rules": [],
            "emotions": [],
            "motivations": []
        }
        
        persona_store = PersonaStore(DB_PATH)
        rules = persona_store.get_all_active(limit=limit)
        result["rules"] = [
            {
                "id": rule.id,
                "text": rule.text,
                "score": float(rule.score),
                "importance": float(rule.importance),
                "novelty": float(rule.novelty),
                "reliability": float(rule.reliability),
                "evidence_count": rule.evidence_count,
                "is_core": bool(rule.is_core),
                "locked": bool(rule.locked),
                "created_at": rule.created_at,
                "last_seen_at": rule.last_seen_at
            }
            for rule in rules
        ]
        
        try:
            emotion_store = EmotionStore(DB_PATH)
            emotions = emotion_store.get_all_patterns(status="active", limit=limit)
            result["emotions"] = [
                {
                    "id": pattern.id,
                    "text": pattern.text,
                    "emotion_type": pattern.emotion_type,
                    "emotion_name": pattern.emotion_name,
                    "intensity": float(pattern.intensity),
                    "trigger_condition": pattern.trigger_condition,
                    "evidence_count": pattern.evidence_count,
                    "created_at": pattern.created_at,
                    "last_seen_at": pattern.last_seen_at
                }
                for pattern in emotions
            ]
        except Exception as e:
            logger.warning(f"Failed to get emotion patterns: {e}")
        
        try:
            motivation_store = MotivationStore(DB_PATH)
            motivations = motivation_store.get_all_patterns(status="active", limit=limit)
            result["motivations"] = [
                {
                    "id": pattern.id,
                    "text": pattern.text,
                    "motivation_type": pattern.motivation_type,
                    "motivation_name": pattern.motivation_name,
                    "intensity": float(pattern.intensity),
                    "trigger_condition": pattern.trigger_condition,
                    "evidence_count": pattern.evidence_count,
                    "created_at": pattern.created_at,
                    "last_seen_at": pattern.last_seen_at
                }
                for pattern in motivations
            ]
        except Exception as e:
            logger.warning(f"Failed to get motivation patterns: {e}")
            
        result["statistics"] = {
            "rules_count": len(result["rules"]),
            "emotions_count": len(result["emotions"]),
            "motivations_count": len(result["motivations"]),
            "total_count": len(result["rules"]) + len(result["emotions"]) + len(result["motivations"])
        }
        
        return result
    except Exception as e:
        logger.error(f"Failed to get dimension patterns: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Standalone Emotion / Motivation HTTP helpers live here too.
@router.get("/emotion/state", tags=["Emotion"])
def get_emotion_state(sessionId: str = Query(...)):
    try:
        emotion_store = EmotionStore(DB_PATH)
        emotion_state = emotion_store.get_emotion_state(sessionId)
        if not emotion_state:
            raise HTTPException(status_code=404, detail="Emotion state not found")
        return {
            "sessionId": sessionId,
            "dominantEmotion": emotion_state.dominant_emotion,
            "intensity": float(emotion_state.intensity),
            "emotionVector": emotion_state.emotion_vector.tolist(),
            "updatedAt": emotion_state.updated_at
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/motivation/state", tags=["Motivation"])
def get_motivation_state(sessionId: str = Query(...)):
    try:
        motivation_store = MotivationStore(DB_PATH)
        motivation_state = motivation_store.get_motivation_state(sessionId)
        if not motivation_state:
            raise HTTPException(status_code=404, detail="Motivation state not found")
        return {
            "sessionId": sessionId,
            "dominantMotivation": motivation_state.dominant_motivation,
            "intensity": float(motivation_state.intensity),
            "motivationVector": motivation_state.motivation_vector.tolist(),
            "updatedAt": motivation_state.updated_at
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

