import json
import math
import os
import logging
from fastapi import APIRouter, Query, HTTPException
from typing import Dict
from pydantic import BaseModel
import sqlite3
from backend.database import get_db, now_iso, DB_PATH
from backend.schemas import SelfStateUpsert, RollbackBody
from backend.self_model import SelfModel
from backend.persona_store import PersonaStore
from backend.drift_monitor import DriftMonitor
from backend.soul_consistency import SoulConsistencyChecker
from backend.rule_compressor import RuleCompressor
from backend.s_identity import get_effective_session

router = APIRouter(prefix="/self", tags=["Self"])
# DB_PATH imported from backend.database
logger = logging.getLogger(__name__)

class SleepBody(BaseModel):
    sessionId: str

@router.post("/tick")
def trigger_self_tick(body: SleepBody):
    """
    Manually run SelfTick (heartbeat tick):
    aggregate evidence, refresh z_self, schedule background mind-wandering hooks.
    """
    try:
        body.sessionId = get_effective_session(body.sessionId)
        from backend.routers.chat import get_chat_service
        chat_service = get_chat_service()
        
        if not chat_service.self_tick:
            raise HTTPException(status_code=500, detail="SelfTick not initialized")
            
        result = chat_service.self_tick.trigger(
            session_id=body.sessionId,
            self_model=chat_service.self_model,
            persona_store=chat_service.persona_store,
            trigger_reason="manual_heartbeat"
        )
        return result
    except Exception as e:
        logger.error(f"Manual tick failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Tick failed: {str(e)}")

@router.post("/sleep")
def trigger_sleep_consolidation(body: SleepBody):
    """
    Sleep consolidation: memory metabolism (compress/forget) and recharge energy to 100%.
    """
    try:
        body.sessionId = get_effective_session(body.sessionId)
        # 1) Memory metabolism
        compressor = RuleCompressor(DB_PATH)
        
        # Optional sensory buffer (if chat_service initialized it)
        sensory_buffer = None
        try:
            from backend.routers.chat import get_chat_service
            chat_service = get_chat_service()
            sensory_buffer = getattr(chat_service, 'sensory_buffer', None)
        except Exception as e:
            logger.warning(f"Failed to get sensory buffer: {e}")
        
        metabolism_result = compressor.sleep_consolidation(
            session_id=body.sessionId,
            sensory_buffer=sensory_buffer
        )
        
        # 1.5) Narrative consolidation into long-term memory
        try:
            from backend.self_narrative import SelfNarrative
            from backend.routers.chat import get_chat_service
            chat_service = get_chat_service()
            
            if chat_service.self_narrative:
                current_history = chat_service.session_history.get(body.sessionId, [])
                if len(current_history) >= 2:
                    # z_self snapshot for consolidation
                    z_self_summary = ""
                    drift = 0.0
                    pain = 0.0
                    if chat_service.self_model:
                        struct_summary = chat_service.self_model.get_structured_summary(body.sessionId)
                        z_self_summary = chat_service.self_model.get_summary(body.sessionId)
                        drift = struct_summary.get("drift", 0.0)
                        pain_status = chat_service.self_model.get_pain_status(body.sessionId)
                        pain = pain_status.get("total_pain", 0.0)
                    
                    consolidated = chat_service.self_narrative.consolidate_memory(
                        session_id=body.sessionId,
                        recent_history=current_history[-20:],  # last 20 turns
                        z_self_summary=z_self_summary,
                        drift=drift,
                        pain=pain
                    )
                    if consolidated:
                        logger.info(f"Memory consolidated during sleep for session {body.sessionId}")
                        metabolism_result["narrative_consolidated"] = True
        except Exception as e:
            logger.warning(f"Failed to consolidate narrative memory during sleep: {e}")
        
        # 2) Force energy recharge
        recharged = False
        with get_db() as conn:
            # Row must exist
            cur = conn.execute("SELECT self_summary FROM self_state WHERE session_id=?", (body.sessionId,))
            row = cur.fetchone()
            if row:
                current_summary = json.loads(row[0]) if row[0] else {}
                current_summary["energy"] = 100.0
                current_summary["is_dormant"] = False
                
                # Persist summary + energy column
                conn.execute(
                    "UPDATE self_state SET self_summary=?, energy=100.0, updated_at=? WHERE session_id=?",
                    (json.dumps(current_summary), now_iso(), body.sessionId)
                )
                conn.commit()
                recharged = True
        
        return {
            "ok": True,
            "metabolism": metabolism_result,
            "energy_recharged": recharged,
            "message": "AI has successfully slept and recharged."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sleep failed: {str(e)}")

@router.get("/versions")
def get_self_versions(sessionId: str, limit: int = Query(10, ge=1, le=50)):
    sessionId = get_effective_session(sessionId)
    persona_store = PersonaStore(DB_PATH)
    self_model = SelfModel(DB_PATH, persona_store)
    monitor = DriftMonitor(DB_PATH, self_model)
    versions = monitor.get_versions(sessionId, limit)
    return {"sessionId": sessionId, "versions": versions, "count": len(versions)}

@router.get("/state")
def get_self_state(sessionId: str):
    sessionId = get_effective_session(sessionId)
    with get_db() as conn:
        # Prefer row with energy + needs when schema supports it
        try:
            cur = conn.execute("SELECT session_id, z_self, confidence, limits, tick, drift, calibration_ece, self_summary, updated_at, energy, needs FROM self_state WHERE session_id=?", (sessionId,))
        except Exception:
            # Older schema without energy/needs columns
            cur = conn.execute("SELECT session_id, z_self, confidence, limits, tick, drift, calibration_ece, self_summary, updated_at FROM self_state WHERE session_id=?", (sessionId,))
            
        row = cur.fetchone()
        if not row:
            return {"sessionId": sessionId, "exists": False}
        data = dict(row)
        try:
            data["z_self"] = json.loads(data["z_self"])
        except Exception:
            data["z_self"] = []
        try:
            data["limits"] = json.loads(data["limits"]) if data["limits"] else []
        except Exception:
            data["limits"] = []
        try:
            data["self_summary"] = json.loads(data["self_summary"]) if data["self_summary"] else {}
        except Exception:
            data["self_summary"] = {}
            
        # Mirror DB energy/needs into selfSummary for clients
        if "energy" in data and data["energy"] is not None:
            energy_val = float(data["energy"])
            # Drop invalid floats
            if not (math.isnan(energy_val) or math.isinf(energy_val)):
                data["self_summary"]["energy"] = energy_val
                data["self_summary"]["is_dormant"] = energy_val < 10.0
            
        if "needs" in data and data["needs"] is not None:
            try:
                needs_obj = json.loads(data["needs"])
                data["self_summary"]["needs"] = needs_obj
            except (json.JSONDecodeError, TypeError, ValueError):
                pass

        def sanitize_floats(obj):
            """Strip NaN/Infinity; decode stray bytes so JSON encoding never crashes."""
            if isinstance(obj, float):
                if math.isnan(obj) or math.isinf(obj):
                    return None
                return obj
            elif isinstance(obj, (bytes, bytearray)):
                # Rare legacy rows store bytes; decode defensively
                try:
                    return obj.decode("utf-8")
                except Exception:
                    try:
                        return obj.decode("utf-8", errors="ignore")
                    except Exception:
                        return None
            elif isinstance(obj, dict):
                return {k: sanitize_floats(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [sanitize_floats(item) for item in obj]
            return obj

        # self_summary refresh lags SelfTick; drift column may be stale while JSON drift_l1 is fresh
        self_summary = data["self_summary"] if isinstance(data.get("self_summary"), dict) else {}
        try:
            col_drift_f = float(data["drift"]) if data.get("drift") is not None else 0.0
        except (TypeError, ValueError):
            col_drift_f = 0.0
        effective_drift = col_drift_f
        if isinstance(self_summary, dict) and self_summary:
            try:
                snap = float(self_summary.get("drift", 0.0) or 0.0)
            except (TypeError, ValueError):
                snap = 0.0
            dl1_f = None
            try:
                if self_summary.get("drift_l1") is not None:
                    dl1_f = float(self_summary.get("drift_l1"))
            except (TypeError, ValueError):
                dl1_f = None
            merged = dict(self_summary)
            if dl1_f is not None and abs(col_drift_f) < 1e-9 and dl1_f > 1e-9:
                effective_drift = dl1_f
            merged["drift"] = effective_drift
            if abs(snap - col_drift_f) > 1e-5:
                merged["drift_stale_snapshot"] = snap
            self_summary = merged
        elif not isinstance(self_summary, dict):
            self_summary = {}

        # If still zero drift, pull drift_l1 from self_state_meta (same source as structured summary)
        if abs(effective_drift) < 1e-9:
            try:
                cur_m = conn.execute(
                    "SELECT meta_json FROM self_state_meta WHERE session_id=?",
                    (sessionId,),
                )
                row_m = cur_m.fetchone()
                if row_m and row_m[0]:
                    mj = json.loads(row_m[0])
                    if isinstance(mj, dict) and mj.get("drift_l1") is not None:
                        dm = float(mj["drift_l1"])
                        if dm > 1e-9:
                            effective_drift = dm
                            if isinstance(self_summary, dict):
                                self_summary = dict(self_summary)
                                self_summary["drift"] = effective_drift
            except Exception:
                pass

        state_activity = 0.0
        try:
            z_vals = data.get("z_self") if isinstance(data.get("z_self"), list) else []
            state_tail = [float(v) for v in z_vals[32:] if isinstance(v, (int, float))]
            if state_tail:
                norm = math.sqrt(sum(v * v for v in state_tail))
                state_activity = max(0.0, min(1.0, norm / (math.sqrt(len(state_tail)) + 1e-8)))
        except Exception:
            state_activity = 0.0

        drift_display = effective_drift
        if isinstance(self_summary, dict):
            if self_summary.get("state_activity") is None:
                self_summary = dict(self_summary)
                self_summary["state_activity"] = state_activity
        if abs(drift_display) < 1e-9 and state_activity > 1e-9:
            drift_display = state_activity
        if isinstance(self_summary, dict):
            self_summary = dict(self_summary)
            self_summary["drift_display"] = drift_display

        result = {
            "sessionId": data["session_id"],
            "zSelf": data["z_self"],
            "confidence": data["confidence"],
            "limits": data["limits"],
            "tick": data["tick"],
            "drift": effective_drift,
            "driftDisplay": drift_display,
            "stateActivity": state_activity,
            "calibrationECE": data["calibration_ece"],
            "selfSummary": self_summary,
            "updatedAt": data["updated_at"],
            "exists": True
        }
        return sanitize_floats(result)


@router.get("/last_tick")
def get_last_tick(sessionId: str):
    """
    Latest real SelfTick snapshot from `self_history`, with drift/tick hints from `self_state`.
    """
    sessionId = get_effective_session(sessionId)
    with get_db() as conn:
        conn.row_factory = sqlite3.Row  # type: ignore
        # 1) latest self_history row (real tick)
        last = None
        try:
            cur = conn.execute(
                """
                SELECT session_id, tick, trigger_event, dominant_emotion, timestamp
                FROM self_history
                WHERE session_id=?
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                (sessionId,),
            )
            last = cur.fetchone()
        except Exception:
            last = None

        # 2) latest self_state drift/tick for display
        state = None
        try:
            cur2 = conn.execute(
                "SELECT tick, drift, updated_at FROM self_state WHERE session_id=?",
                (sessionId,),
            )
            state = cur2.fetchone()
        except Exception:
            state = None

        if not last and not state:
            return {"sessionId": sessionId, "exists": False}

        out = {"sessionId": sessionId, "exists": True}
        if state:
            try:
                out["self_state"] = {
                    "tick": int(state["tick"]) if state["tick"] is not None else 0,
                    "drift": float(state["drift"]) if state["drift"] is not None else 0.0,
                    "updated_at": state["updated_at"],
                }
            except Exception:
                pass
        if last:
            out["last_tick"] = {
                "tick": int(last["tick"]) if last["tick"] is not None else 0,
                "timestamp": last["timestamp"],
                "trigger_event": last["trigger_event"],
                "dominant_emotion": last["dominant_emotion"],
            }
        return out

@router.post("/state")
def upsert_self_state(body: SelfStateUpsert):
    body.sessionId = get_effective_session(body.sessionId)
    with get_db() as conn:
        if body.drift is None:
            cur = conn.execute(
                "SELECT drift FROM self_state WHERE session_id=?",
                (body.sessionId,),
            )
            row = cur.fetchone()
            if row is not None and row[0] is not None:
                drift_val = float(row[0])
            else:
                drift_val = 0.0
        else:
            drift_val = float(body.drift)

        conn.execute(
            """INSERT INTO self_state (session_id, z_self, confidence, limits, tick, drift, calibration_ece, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(session_id) DO UPDATE SET
                 z_self=excluded.z_self,
                 confidence=excluded.confidence,
                 limits=excluded.limits,
                 tick=excluded.tick,
                 drift=excluded.drift,
                 calibration_ece=excluded.calibration_ece,
                 updated_at=excluded.updated_at
            """,
            (
                body.sessionId,
                json.dumps(body.zSelf or []),
                body.confidence or 0.0,
                json.dumps(body.limits or []),
                body.tick or 0,
                drift_val,
                body.calibrationECE or 0.0,
                now_iso(),
            ),
        )
        conn.commit()
    return {"ok": True, "sessionId": body.sessionId}

@router.post("/rollback")
def self_rollback(body: RollbackBody):
    body.sessionId = get_effective_session(body.sessionId)
    persona_store = PersonaStore(DB_PATH)
    self_model = SelfModel(DB_PATH, persona_store)
    monitor = DriftMonitor(DB_PATH, self_model)
    result = monitor.rollback(body.sessionId, body.targetVersion)
    
    if result.get("success"):
        return {
            "ok": True,
            "sessionId": body.sessionId,
            "rolledBackTo": result.get("rolled_back_to")
        }
    else:
        raise HTTPException(status_code=400, detail=result.get("error", "Rollback failed"))

@router.get("/summary")
def get_self_summary(sessionId: str):
    sessionId = get_effective_session(sessionId)
    persona_store = PersonaStore(DB_PATH)
    self_model = SelfModel(DB_PATH, persona_store)
    summary = self_model.get_structured_summary(sessionId)
    # Compact line mirrors what generation sees in structured summaries
    internal_state_compact = ""
    try:
        from backend.self_model_summary import generate_internal_state_prompt

        try:
            _live_e = float(self_model.get_energy(sessionId))
        except Exception:
            _live_e = None
        try:
            _ps = self_model.get_pain_status(sessionId)
        except Exception:
            _ps = {}
        internal_state_compact = generate_internal_state_prompt(
            self_model,
            sessionId,
            energy=_live_e,
            pain_status=_ps,
            system_entropy=0.0,
            noise_perturbation=0.0,
            hide_numbers=True,
        )
    except Exception as e:
        logger.warning(
            f"internal_state_compact for monitor failed (sessionId={sessionId}): {e}",
            exc_info=True,
        )

    # If summary has scalars but compact is empty, log version skew—not necessarily missing z_self
    try:
        if not (internal_state_compact or "").strip():
            if isinstance(summary, dict) and (
                summary.get("arousal_mean") is not None
                or summary.get("energy") is not None
            ):
                logger.warning(
                    "[/self/summary] internal_state_compact is empty but summary has z_self-derived fields; "
                    "see warning above if generate_internal_state_prompt raised, or verify deployed backend version."
                )
    except Exception:
        pass

    return {
        "sessionId": sessionId,
        "summary": summary,
        "internal_state_compact": internal_state_compact,
    }

@router.get("/trajectory")
def get_self_trajectory(sessionId: str, limit: int = Query(20, ge=1, le=100)):
    """
    Self evolution trajectory: prefer `self_history` (vector snapshots), else `persona_events` summaries.
    """
    sessionId = get_effective_session(sessionId)
    history_items = []
    
    with get_db() as conn:
        # 1) self_history (mirror system)
        try:
            cur = conn.execute(
                "SELECT tick, z_self_vector, trigger_event, dominant_emotion, timestamp FROM self_history WHERE session_id=? ORDER BY tick DESC LIMIT ?",
                (sessionId, limit)
            )
            rows = cur.fetchall()
            if rows:
                for row in rows:
                    history_items.append({
                        "tick": row["tick"],
                        "z_self_vector": row["z_self_vector"], # JSON string
                        "trigger_event": row["trigger_event"],
                        "dominant_emotion": row["dominant_emotion"],
                        "timestamp": row["timestamp"],
                        "source": "self_history"
                    })
                # Ascending tick for charts
                history_items.sort(key=lambda x: x["tick"])
                return {"sessionId": sessionId, "history": history_items, "count": len(history_items)}
        except Exception as e:
            # Missing table → fall back
            pass

        # 2) Legacy persona_events
        cur = conn.execute(
            "SELECT id, ts, detail FROM persona_events WHERE type='self_tick' AND persona_id=? ORDER BY ts DESC LIMIT ?",
            (sessionId, limit)
        )
        rows = cur.fetchall()
    
    # Normalize legacy rows
    for row in rows:
        detail = {}
        try:
            detail = json.loads(row["detail"]) if row["detail"] else {}
        except Exception:
            detail = {}
        history_items.append({
            "tick": detail.get("tick", 0),
            "z_self_vector": "[]",  # legacy table lacks vectors
            "trigger_event": detail.get("trigger_reason", "unknown"),
            "dominant_emotion": "unknown",
            "timestamp": row["ts"],
            "drift": detail.get("drift"),
            "source": "persona_events"
        })
    
    history_items.sort(key=lambda x: x["tick"])
    return {"sessionId": sessionId, "history": history_items, "count": len(history_items)}

@router.get("/consistency")
def check_soul_consistency(sessionId: str = Query(..., description="Session id"), autoFix: bool = Query(False)):
    sessionId = get_effective_session(sessionId)
    persona_store = PersonaStore(DB_PATH)
    self_model = SelfModel(DB_PATH, persona_store)
    checker = SoulConsistencyChecker(DB_PATH, persona_store, self_model)
    return checker.check_consistency(sessionId, auto_fix=autoFix)

@router.get("/consistency/history")
def get_consistency_history(sessionId: str = Query(...), limit: int = Query(10, ge=1, le=50)):
    sessionId = get_effective_session(sessionId)
    checker = SoulConsistencyChecker(DB_PATH)
    history = checker.get_consistency_history(sessionId, limit=limit)
    return {"sessionId": sessionId, "history": history, "count": len(history)}

@router.post("/sync-rules")
def sync_persona_rules_to_zself(body: SleepBody):
    """Push all active persona rules into z_self (manual repair / refresh)."""
    try:
        body.sessionId = get_effective_session(body.sessionId)
        persona_store = PersonaStore(DB_PATH)
        self_model = SelfModel(DB_PATH, persona_store)
        
        all_rules = persona_store.get_all_active(limit=300)
        
        # Write aggregated persona signal into z_self
        updated_z_self = self_model.update_from_persona_rules(body.sessionId, all_rules)
        
        return {
            "ok": True,
            "sessionId": body.sessionId,
            "rules_count": len(all_rules),
            "z_self_dim": len(updated_z_self) if updated_z_self is not None else 0,
            "message": f"Successfully synced {len(all_rules)} rules to z_self"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sync failed: {str(e)}")

@router.post("/wander")
def trigger_mind_wandering(body: SleepBody):
    """Trigger mind wandering (energy spend, introspective drift, may mint L2 rules)."""
    try:
        body.sessionId = get_effective_session(body.sessionId)
        # Reuse chat_service singleton
        from backend.routers.chat import get_chat_service
        chat_service = get_chat_service()
        
        if not chat_service.mind_wandering:
            # Lazily attach MindWandering if older process skipped init
            try:
                from backend.mind_wandering import MindWandering
                chat_service.mind_wandering = MindWandering(DB_PATH, chat_service)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"MindWandering init failed: {e}")
            
        result = chat_service.mind_wandering.trigger_wandering(body.sessionId)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Mind wandering failed: {str(e)}")

@router.post("/shadow-theater")
def trigger_shadow_theater(body: SleepBody):
    """Shadow theater: opposing agents debate; may update z_self; costs ~20 energy."""
    try:
        body.sessionId = get_effective_session(body.sessionId)
        from backend.routers.chat import get_chat_service
        chat_service = get_chat_service()
        
        if not chat_service.mind_wandering:
            try:
                from backend.mind_wandering import MindWandering
                chat_service.mind_wandering = MindWandering(DB_PATH, chat_service)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"MindWandering init failed: {e}")
        
        if not chat_service.mind_wandering.shadow_theater:
            raise HTTPException(status_code=500, detail="ShadowTheater not initialized")
            
        result = chat_service.mind_wandering.trigger_shadow_theater(body.sessionId)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Shadow theater failed: {str(e)}")

