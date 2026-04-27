import os
from fastapi import APIRouter, Query
from backend.schemas import WorldStateUpdate
from backend.world_state import WorldState
from backend.database import DB_PATH

router = APIRouter(prefix="/world", tags=["World"])

@router.get("/state")
def get_world_state(
    sessionId: str = Query(..., description="Session id (normalized by callers when applicable)"),
):
    """Return persisted world/task snapshot for a session, if any."""
    world_state = WorldState(DB_PATH)
    state = world_state.get_state(sessionId)
    if not state:
        return {"sessionId": sessionId, "exists": False}
    return {
        "sessionId": sessionId,
        "taskStage": state["task_stage"],
        "envSummary": state["env_summary"],
        "lastAction": state["last_action"],
        "updatedAt": state["updated_at"],
        "exists": True
    }

@router.post("/state")
def update_world_state(body: WorldStateUpdate):
    """Persist coarse world/task stage hints for a session."""
    world_state = WorldState(DB_PATH)
    world_state.update_state(
        session_id=body.sessionId,
        task_stage=body.taskStage,
        env_summary=body.envSummary,
        last_action=body.lastAction
    )
    return {"ok": True, "sessionId": body.sessionId}

