from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class PersonaCreate(BaseModel):
    text: str
    score: Optional[float] = 0.0
    importance: Optional[float] = 0.0
    novelty: Optional[float] = 0.0
    reliability: Optional[float] = 0.0
    status: Optional[str] = "active"

class PersonaUpdate(BaseModel):
    score: Optional[float] = None
    importance: Optional[float] = None
    novelty: Optional[float] = None
    reliability: Optional[float] = None
    status: Optional[str] = None

class SelfStateUpsert(BaseModel):
    sessionId: str
    zSelf: List[float] = Field(default_factory=list)
    confidence: Optional[float] = 0.0
    limits: Optional[List[str]] = Field(default_factory=list)
    tick: Optional[int] = 0
    # None => do not overwrite self_state.drift (prevents accidental 0 when the client omits the field)
    drift: Optional[float] = None
    calibrationECE: Optional[float] = 0.0

class PersonaItemOut(BaseModel):
    """Persona row returned in list/core APIs."""
    id: str
    text: str
    score: float
    importance: float
    novelty: float
    reliability: float
    evidence_count: int
    created_at: str
    last_seen_at: str
    status: str
    is_core: int = 0
    core_version: int = 0
    locked: int = 0

class PersonaListResponse(BaseModel):
    items: List[PersonaItemOut]
    count: int

class PersonaSearchItem(BaseModel):
    id: str
    text: str
    score: float
    similarity: float
    status: str
    is_core: int
    core_version: int
    locked: int

class PersonaSearchResponse(BaseModel):
    items: List[PersonaSearchItem]
    count: int

class PromoteResult(BaseModel):
    ok: bool
    promoted: int
    core_version: Optional[int] = None
    ids: Optional[List[str]] = None

class ModifyRuleRequest(BaseModel):
    newText: str
    reason: Optional[str] = None

class SearchBody(BaseModel):
    query: str
    limit: Optional[int] = 10

class PromoteBody(BaseModel):
    ids: List[str]
    lock: Optional[bool] = True
    boost_score: Optional[float] = 0.0

class WorldStateUpdate(BaseModel):
    sessionId: str
    taskStage: Optional[str] = None
    envSummary: Optional[Dict[str, Any]] = None
    lastAction: Optional[str] = None

class RollbackBody(BaseModel):
    sessionId: str
    targetVersion: Optional[int] = None  # None => roll back to the previous version

class ChatRequest(BaseModel):
    message: str
    sessionId: Optional[str] = "default"
    temperature: Optional[float] = 0.3
    # A/B toggles for single-request ablation (prompt-only effect checks)
    ab_disable_persona: Optional[bool] = False
    ab_disable_identity: Optional[bool] = False
    ab_disable_core_anchor: Optional[bool] = False
    ab_disable_collective_resonance: Optional[bool] = False
    ab_raw_mode: Optional[bool] = False

class FeedbackRequest(BaseModel):
    sessionId: str
    personaId: Optional[str] = None  # When set, feedback targets that rule id
    turnIndex: Optional[int] = None  # When set, feedback targets that chat turn index
    feedbackType: str  # "positive" | "negative" | "neutral"
    comment: Optional[str] = None  # Optional free-text note

