import uuid
import json
import os
import numpy as np
import sqlite3
from fastapi import APIRouter, HTTPException, Request
from typing import Dict, Any, Optional
from backend.database import get_db, now_iso, DB_PATH
from backend.chat_service import ChatService
from backend.schemas import ChatRequest, FeedbackRequest
from backend.persona_store import PersonaStore
from backend.s_identity import get_effective_session
from backend.config import config
from backend.llm_api import llm_completion
import logging
import requests

router = APIRouter(tags=["Chat"])
logger = logging.getLogger(__name__)
# DB_PATH imported from backend.database

# ChatService singleton so session_history survives across requests
_chat_service_instance: Optional[ChatService] = None

def get_chat_service() -> ChatService:
    global _chat_service_instance
    if _chat_service_instance is None:
        _chat_service_instance = ChatService(DB_PATH)
        logger.info("Created new ChatService instance (singleton)")
    return _chat_service_instance

def get_persona_store() -> PersonaStore:
    return PersonaStore(DB_PATH)

def _convert_to_json_serializable(obj):
    """Make values JSON-safe (NaN/Infinity -> None, numpy -> native)."""
    import math
    
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        val = float(obj)
        # NaN / Inf
        if math.isnan(val):
            return None
        elif math.isinf(val):
            return None if val < 0 else None  # fold infinities to None
        return val
    elif isinstance(obj, float):
        # Native float NaN/Inf
        if math.isnan(obj):
            return None
        elif math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_to_json_serializable(item) for item in obj]
    else:
        return obj

@router.post("/api/chat")
def chat(request: ChatRequest, http_request: Request):
    result = None  # for error logging
    try:
        # [Session Unification] normalize session id
        request.sessionId = get_effective_session(request.sessionId)
        
        # [2026-02-27] WebSocket "thinking" ping for immediate UI feedback
        try:
            from backend.websocket_manager import get_websocket_manager, create_ws_message
            ws_manager = get_websocket_manager()
            agent_name = config.get("system.agent_name", "Agent")
            thinking_msg = create_ws_message(
                msg_type="thinking",
                content=f"{agent_name} is thinking...",
                session_id=request.sessionId
            )
            ws_manager.queue_message(request.sessionId, thinking_msg)
            logger.debug(f"[CHAT] Sent thinking status for session {request.sessionId}")
        except Exception as ws_err:
            logger.debug(f"[CHAT] Failed to send thinking status: {ws_err}")
        
        service = get_chat_service()
        result = service.chat(
            user_input=request.message,
            session_id=request.sessionId,
            temperature=request.temperature,
            ab_test={
                "disable_persona": bool(request.ab_disable_persona),
                "disable_identity": bool(request.ab_disable_identity),
                "disable_core_anchor": bool(request.ab_disable_core_anchor),
                "disable_collective_resonance": bool(request.ab_disable_collective_resonance),
                "raw_mode": bool(request.ab_raw_mode),
            },
        )
        # [2026-01-20] success path: debug level only
        logger.debug("[CHAT] result type=%s", type(result))
        if isinstance(result, dict):
            logger.debug("[CHAT] result keys=%s", list(result.keys()))
        else:
            logger.debug("[CHAT] result value=%s", result)
        
        cleaned_result = _convert_to_json_serializable(result)
        # [2026-03-18] expose agent_name for the UI
        if isinstance(cleaned_result, dict):
            cleaned_result["agent_name"] = config.get("system.agent_name", "Agent")
        return cleaned_result
    except HTTPException:
        raise
    except requests.RequestException as e:
        logger.error(f"Network error in chat: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail="Service temporarily unavailable")
    except (KeyError, ValueError) as e:
        import traceback
        logger.error(f"Invalid response format in chat: {e}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")
        logger.error(f"Result type: {type(result)}, Result: {str(result)[:500]}")
        raise HTTPException(status_code=500, detail=f"Invalid response format: {str(e)}")
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        logger.error(f"Result: {str(result)[:500] if result else 'None'}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/feedback", tags=["Feedback"])
def submit_feedback(body: FeedbackRequest):
    body.sessionId = get_effective_session(body.sessionId)
    if body.feedbackType not in ["positive", "negative", "neutral"]:
        raise HTTPException(status_code=400, detail="feedbackType must be 'positive', 'negative', or 'neutral'")
    
    store = get_persona_store()
    POSITIVE_BOOST = float(os.environ.get("FEEDBACK_POSITIVE_BOOST", "0.1"))
    NEGATIVE_PENALTY = float(os.environ.get("FEEDBACK_NEGATIVE_PENALTY", "0.1"))
    
    updated_count = 0
    
    try:
        if body.personaId:
            item = store.get_by_id(body.personaId)
            if not item:
                raise HTTPException(status_code=404, detail=f"Persona {body.personaId} not found")
            
            # Feedback adjusts importance/reliability/novelty then recomputes base score
            if body.feedbackType == "positive":
                item.reliability = min(1.0, item.reliability + POSITIVE_BOOST)
                item.importance = min(1.0, item.importance + POSITIVE_BOOST * 0.5)
            elif body.feedbackType == "negative":
                item.reliability = max(0.0, item.reliability - NEGATIVE_PENALTY)
                item.importance = max(0.0, item.importance - NEGATIVE_PENALTY * 0.5)
            
            # Recompute base score
            from backend.scoring import IMPORTANCE_WEIGHT, NOVELTY_WEIGHT, RELIABILITY_WEIGHT
            item.score = (
                IMPORTANCE_WEIGHT * item.importance +
                NOVELTY_WEIGHT * item.novelty +
                RELIABILITY_WEIGHT * item.reliability
            )
            # item.score is base_score; dynamic scorer may adjust at retrieval time
            store.add_or_update(item, update_embedding=False)
            updated_count = 1
            
        else:
            with get_db() as conn:
                # Older DBs may lack chat_turns.retrieved_persona_ids
                has_retrieved_col = False
                try:
                    cols = [r[1] for r in conn.execute("PRAGMA table_info(chat_turns)").fetchall()]
                    has_retrieved_col = "retrieved_persona_ids" in cols
                except Exception:
                    has_retrieved_col = False

                row = None
                if has_retrieved_col:
                    try:
                        if body.turnIndex is not None:
                            cur = conn.execute(
                                "SELECT id, retrieved_persona_ids FROM chat_turns WHERE session_id=? AND turn_index=?",
                                (body.sessionId, body.turnIndex)
                            )
                        else:
                            cur = conn.execute(
                                "SELECT id, retrieved_persona_ids FROM chat_turns WHERE session_id=? ORDER BY turn_index DESC LIMIT 1",
                                (body.sessionId,)
                            )
                        row = cur.fetchone()
                    except sqlite3.OperationalError:
                        row = None
                
                retrieved_ids = []
                if row:
                    retrieved_ids_str = row[1] if len(row) > 1 else None
                    if retrieved_ids_str:
                        try:
                            retrieved_ids = json.loads(retrieved_ids_str)
                        except (json.JSONDecodeError, TypeError, ValueError):
                            pass
                
                if not retrieved_ids:
                    items = store.get_all_active(limit=20)
                    retrieved_ids = [item.id for item in items[:10]]
                
                if retrieved_ids:
                    for rule_id in retrieved_ids:
                        item = store.get_by_id(rule_id)
                        if item and body.feedbackType != "neutral":
                            # Feedback path: tweak reliability then recompute base score
                            session_boost = POSITIVE_BOOST * 0.5 if body.feedbackType == "positive" else -NEGATIVE_PENALTY * 0.5
                            item.reliability = max(0.0, min(1.0, item.reliability + session_boost))
                            # Recompute base score
                            from backend.scoring import IMPORTANCE_WEIGHT, NOVELTY_WEIGHT, RELIABILITY_WEIGHT
                            item.score = (
                                IMPORTANCE_WEIGHT * item.importance +
                                NOVELTY_WEIGHT * item.novelty +
                                RELIABILITY_WEIGHT * item.reliability
                            )
                            # item.score is base_score; dynamic scorer may adjust at retrieval time
                            store.add_or_update(item, update_embedding=False)
                            updated_count += 1
        
        with get_db() as conn:
            conn.execute(
                """INSERT INTO persona_events (id, ts, type, persona_id, detail)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    str(uuid.uuid4()),
                    now_iso(),
                    "feedback",
                    body.personaId or body.sessionId,
                    json.dumps({
                        "sessionId": body.sessionId,
                        "personaId": body.personaId,
                        "turnIndex": body.turnIndex,
                        "feedbackType": body.feedbackType,
                        "comment": body.comment,
                        "updatedCount": updated_count
                    }, ensure_ascii=False)
                )
            )
            conn.commit()
        
        # P1: optional energy bump from positive interaction feedback
        if body.feedbackType in ["positive", "negative"]:
            try:
                service = get_chat_service()
                if service.self_model:
                    # [2026-03-30] feedback via event path
                    if body.feedbackType == "positive":
                        service.self_model.trigger_event(body.sessionId, "positive_feedback", intensity=1.0)
                        logger.info(f"Positive feedback event triggered (session={body.sessionId})")
                    elif body.feedbackType == "negative":
                        service.self_model.trigger_event(body.sessionId, "negative_feedback", intensity=1.0)
                        logger.info(f"Negative feedback event triggered (session={body.sessionId})")
            except Exception as e:
                logger.warning(f"Failed to update energy from feedback: {e}")
        
        return {
            "ok": True,
            "sessionId": body.sessionId,
            "feedbackType": body.feedbackType,
            "updatedCount": updated_count
        }
        
    except Exception as e:
        logger.error(f"Feedback processing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/api/chat/history")
def get_chat_history(sessionId: str, limit: int = 20, http_request: Request = None):
    """Return recent chat turns for a session."""
    try:
        sessionId = get_effective_session(sessionId)
        service = get_chat_service()
        agent_name = config.get("system.agent_name", "Agent")
        # Prefer in-process buffer
        if sessionId in service.session_history and service.session_history[sessionId]:
            history = service.session_history[sessionId]
            return {"history": history[-limit:], "agent_name": agent_name}
        
        # Else load from sensory buffer snapshot
        if service.sensory_buffer:
            history = service.sensory_buffer.get_conversation_history(sessionId, limit=limit)
            if history:
                return {"history": history, "agent_name": agent_name}
        
        return {"history": [], "agent_name": agent_name}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get chat history: {e}")
        return {"history": [], "agent_name": config.get("system.agent_name", "Agent"), "error": str(e)}

@router.get("/api/chat/last_reflection")
def get_last_reflection(sessionId: str):
    """Last reflection snapshot for async reflection pipeline (cached)."""
    try:
        sessionId = get_effective_session(sessionId)
        service = get_chat_service()
        
        # Cached reflection payload
        if hasattr(service, 'session_reflection_cache') and sessionId in service.session_reflection_cache:
            cached = service.session_reflection_cache[sessionId]
            return {
                "sessionId": sessionId,
                "exists": True,
                "reflection": cached["result"],
                "turn_index": cached["turn_index"],
                "timestamp": cached["timestamp"]
            }
        else:
            return {
                "sessionId": sessionId,
                "exists": False,
                "message": "No reflection run yet, or result not cached."
            }
    except Exception as e:
        logger.error(f"Failed to get last reflection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/chat_baseline")
def chat_baseline(request: ChatRequest):
    try:
        with get_db() as conn:
            conn.row_factory = import_sqlite3_row_factory()
            cur = conn.execute(
                "SELECT text FROM persona_items WHERE status='active' AND is_core=1 "
                "ORDER BY core_version DESC, score DESC, created_at ASC LIMIT 100"
            )
            rows = cur.fetchall()
        lines = [f"- {r['text']}" for r in rows] if rows else []
        persona_block = "\n".join(lines)
        system_prompt = "[Persona core]\n" + persona_block + "\n\n[Task]\n" \
            "Follow the values above; when uncertain, state confidence and limits."

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request.message},
        ]

        result = llm_completion(
            messages=messages,
            temperature=request.temperature or 0.3,
            max_tokens=2000,
            use_lite=False,
        )
        if not result["success"]:
            raise RuntimeError(f"Baseline LLM call failed: {result.get('error')}")
        return {"response": result["content"]}
    except Exception as e:
        logger.error(f"Baseline chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

def import_sqlite3_row_factory():
    import sqlite3
    return sqlite3.Row
