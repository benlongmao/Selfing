import uuid
import json
from fastapi import APIRouter, Query, HTTPException
from typing import List, Dict, Any, Optional
from backend.database import get_db, now_iso, DB_PATH
from backend.persona_store import PersonaStore
from backend.promotion import PromotionGate
from backend.schemas import (
    PersonaListResponse, PersonaCreate, PersonaUpdate, ModifyRuleRequest,
    PersonaSearchResponse, SearchBody, PromoteResult, PromoteBody
)

router = APIRouter(prefix="/persona", tags=["Persona"])
# DB_PATH imported from backend.database

def get_persona_store() -> PersonaStore:
    return PersonaStore(DB_PATH)

@router.get("", response_model=PersonaListResponse)
def list_persona(limit: int = Query(100, ge=1, le=1000)):
    """List active persona_items (includes core/dynamic flags; not split by layer)."""
    with get_db() as conn:
        cur = conn.execute(
            "SELECT id, text, score, importance, novelty, reliability, evidence_count, created_at, last_seen_at, status, "
            "COALESCE(is_core,0) AS is_core, COALESCE(core_version,0) AS core_version, COALESCE(locked,0) AS locked "
            "FROM persona_items WHERE status='active' ORDER BY score DESC LIMIT ?",
            (limit,),
        )
        rows = [dict(r) for r in cur.fetchall()]
    return {"items": rows, "count": len(rows)}

@router.post("", response_model=Dict[str, Any])
def create_persona(body: PersonaCreate):
    pid = str(uuid.uuid4())
    ts = now_iso()
    with get_db() as conn:
        conn.execute(
            "INSERT INTO persona_items (id, text, score, importance, novelty, reliability, evidence_count, created_at, last_seen_at, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                pid,
                body.text,
                body.score or 0.0,
                body.importance or 0.0,
                body.novelty or 0.0,
                body.reliability or 0.0,
                0,
                ts,
                ts,
                body.status or "active",
            ),
        )
        conn.commit()
    return {"id": pid, "createdAt": ts}

@router.put("/{persona_id}")
def update_persona(persona_id: str, body: PersonaUpdate):
    fields: Dict[str, Any] = {}
    for k in ("score", "importance", "novelty", "reliability", "status"):
        v = getattr(body, k)
        if v is not None:
            fields[k] = v
    if not fields:
        return {"ok": True, "updated": 0}
    sets = ", ".join(f"{k}=?" for k in fields.keys())
    with get_db() as conn:
        cur = conn.execute(f"UPDATE persona_items SET {sets}, last_seen_at=? WHERE id=?", (*fields.values(), now_iso(), persona_id))
        conn.commit()
        updated = cur.rowcount
    return {"ok": True, "updated": updated}

@router.put("/{persona_id}/modify")
def modify_persona_rule(persona_id: str, body: ModifyRuleRequest):
    store = get_persona_store()
    result = store.modify_rule(persona_id, body.newText, body.reason)
    
    if result.get("success"):
        return {
            "ok": True,
            "ruleId": persona_id,
            "oldText": result.get("old_text"),
            "newText": result.get("new_text")
        }
    else:
        raise HTTPException(status_code=404, detail=result.get("error", "Rule modification failed"))

@router.get("/{persona_id}/history")
def get_persona_history(persona_id: str, limit: int = Query(10, ge=1, le=50)):
    store = get_persona_store()
    history = store.get_rule_history(persona_id, limit=limit)
    return {
        "ruleId": persona_id,
        "history": history,
        "count": len(history)
    }

@router.delete("/{persona_id}")
def delete_persona(persona_id: str):
    with get_db() as conn:
        cur = conn.execute(
            "SELECT locked FROM persona_items WHERE id=?", (persona_id,)
        )
        row = cur.fetchone()
        if row and row[0] == 1:
            return {"ok": False, "error": "L0 constitutional rule is protected and cannot be archived"}
        cur = conn.execute(
            "UPDATE persona_items SET status='archived', last_seen_at=? WHERE id=? AND locked=0",
            (now_iso(), persona_id),
        )
        conn.commit()
        updated = cur.rowcount
    return {"ok": True, "updated": updated}

@router.post("/search", response_model=PersonaSearchResponse)
def persona_search(body: SearchBody):
    store = get_persona_store()
    results = store.search_top_k(body.query, k=body.limit or 20)
    
    items = []
    for item, similarity in results:
        items.append({
            "id": item.id,
            "text": item.text,
            "score": item.score,
            "similarity": similarity,
            "status": item.status,
            "is_core": getattr(item, "is_core", 0),
            "core_version": getattr(item, "core_version", 0),
            "locked": getattr(item, "locked", 0)
        })
    return {"items": items, "count": len(items)}

# Core Persona Endpoints
@router.get("/core", response_model=PersonaListResponse, tags=["CorePersona"])
def list_core_persona(limit: int = Query(100, ge=1, le=1000)):
    """List core persona rules (is_core=1)."""
    with get_db() as conn:
        cur = conn.execute(
            "SELECT id, text, score, importance, novelty, reliability, evidence_count, created_at, last_seen_at, status, "
            "COALESCE(is_core,1) AS is_core, COALESCE(core_version,0) AS core_version, COALESCE(locked,0) AS locked "
            "FROM persona_items WHERE status='active' AND is_core=1 "
            "ORDER BY score DESC, core_version DESC LIMIT ?",
            (limit,),
        )
        rows = [dict(r) for r in cur.fetchall()]
    return {"items": rows, "count": len(rows)}

@router.post("/core/promote", response_model=PromoteResult, tags=["CorePersona"])
def promote_to_core(body: PromoteBody):
    if not body.ids:
        raise HTTPException(status_code=400, detail="ids is required")
    with get_db() as conn:
        cur = conn.execute("SELECT MAX(core_version) FROM persona_items WHERE is_core=1")
        row = cur.fetchone()
        next_version = (row[0] or 0) + 1
        ts = now_iso()
        updated = 0
        for pid in body.ids:
            cur2 = conn.execute(
                "UPDATE persona_items SET is_core=1, core_version=?, locked=?, status='active', last_seen_at=?, score=score+? WHERE id=?",
                (next_version, 1 if body.lock else 0, ts, (body.boost_score or 0.0), pid)
            )
            updated += cur2.rowcount
        conn.commit()
    return {"ok": True, "promoted": updated, "core_version": next_version, "ids": body.ids}

@router.post("/core/auto_promote", response_model=PromoteResult, tags=["CorePersona"])
def auto_promote_core():
    try:
        gate = PromotionGate(DB_PATH)
        result = gate.auto_promote()
        return {
            "ok": bool(result.get("ok", True)),
            "promoted": int(result.get("promoted", 0)),
            "core_version": result.get("core_version"),
            "ids": result.get("ids") or [],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/compress")
def compress_persona_rules(
    similarity_threshold: Optional[float] = Query(None, ge=0.0, le=1.0),
    max_items: int = Query(100, ge=1, le=200)
):
    from backend.rule_compressor import RuleCompressor
    compressor = RuleCompressor(DB_PATH)
    result = compressor.compress_rules(max_items=max_items, similarity_threshold=similarity_threshold)
    return {
        "ok": True,
        "compressed": result["compressed"],
        "merged_groups": result["merged_groups"],
        "new_rules": result["new_rules"]
    }

@router.get("/{persona_id}")
def get_persona_by_id(persona_id: str):
    store = get_persona_store()
    item = store.get_by_id(persona_id)
    if not item:
        raise HTTPException(status_code=404, detail=f"Persona rule {persona_id} not found")
    return {
        "id": item.id,
        "text": item.text,
        "score": item.score,
        "importance": item.importance,
        "novelty": item.novelty,
        "reliability": item.reliability,
        "evidence_count": item.evidence_count,
        "created_at": item.created_at,
        "last_seen_at": item.last_seen_at,
        "status": item.status,
        "is_core": getattr(item, "is_core", 0),
        "core_version": getattr(item, "core_version", 0),
        "locked": getattr(item, "locked", 0)
    }

@router.get("/{persona_id}/source")
def get_persona_source(persona_id: str):
    store = get_persona_store()
    source = store.get_rule_source(persona_id)
    if source:
        return {"ruleId": persona_id, "source": source}
    else:
        raise HTTPException(status_code=404, detail=f"Source information not found for rule {persona_id}")

@router.get("/evolution")
def analyze_rule_evolution(
    ruleId: Optional[str] = Query(None),
    daysBack: int = Query(30, ge=1, le=365),
    limit: Optional[int] = Query(None, ge=1, le=200)
):
    store = get_persona_store()
    return store.analyze_rule_evolution(rule_id=ruleId, days_back=daysBack, limit=limit)

@router.get("/evolution/summary")
def get_evolution_summary(daysBack: int = Query(30, ge=1, le=365)):
    store = get_persona_store()
    return store.get_evolution_summary(days_back=daysBack)

