import json
import logging
from typing import Any, Dict, Optional

from backend.chat_service_mmr import process_somatic_with_mmr, process_worldview_with_mmr
from backend.reflection import REFLECTION_MAX_RULES, REFLECTION_MIN_EVIDENCE

logger = logging.getLogger(__name__)


def _reflection_summary(
    *,
    status: str,
    reason: Optional[str] = None,
    estimated_turns: int = 0,
    pipeline_ran: bool = False,
    rules: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Stable JSON shape for ``session_reflection_cache`` / ``last_reflection`` polling.

    Top-level ``added`` / ``merged`` / ``updated`` / ``removed`` mirror frontend ``reflectionCounts``.
    """
    out: Dict[str, Any] = {
        "status": status,
        "reason": reason,
        "estimated_turns": estimated_turns,
        "min_evidence": REFLECTION_MIN_EVIDENCE,
        "pipeline_ran": pipeline_ran,
    }
    if error is not None:
        out["error"] = error
    if rules is not None:
        out["rules"] = rules
        r = rules
    else:
        out["rules"] = None
        r = {}
    out["added"] = int(r.get("added", 0) or 0)
    out["merged"] = int(r.get("merged", 0) or 0)
    out["updated"] = int(r.get("updated", 0) or 0)
    out["removed"] = int(r.get("removed", 0) or 0)
    return out


def run_reflection(chat_service, session_id: str, turn_index: int) -> Dict[str, Any]:
    """
    End-to-end reflection pipeline: persona rules plus somatic/worldview dimensions.

    Always returns a JSON-serializable summary (``status`` / ``reason``) for UI polling.

    [2026-03-25] Adds **state triggers** so intense drift/pain/arousal can fire reflection
    even when the conversation is still short—closer to “deep chat” human behavior.
    """
    if not chat_service.session_history.get(session_id):
        return _reflection_summary(
            status="skipped",
            reason="no_session_history",
            estimated_turns=0,
            pipeline_ran=False,
            rules=None,
        )

    reflection_result = None
    had_candidates = False
    estimated_turns = 0

    try:
        conversation_turns = len(chat_service.session_history[session_id]) // 2
        # Include the in-flight turn so we do not under-count before history flushes
        estimated_turns = max(conversation_turns, turn_index)

        state_triggered = False
        state_trigger_reason = None

        if chat_service.self_model:
            try:
                struct_summary = chat_service.self_model.get_structured_summary(session_id)
                reflect_drift = float(struct_summary.get("drift", 0.0) or 0.0)
                reflect_arousal = abs(float(struct_summary.get("arousal", 0.0) or 0.0))

                pain_status = chat_service.self_model.get_pain_status(session_id)
                reflect_pain = float(pain_status.get("total_pain", 0.0) or 0.0)

                if reflect_drift > 0.15:
                    state_triggered = True
                    state_trigger_reason = f"drift={reflect_drift:.2f}"
                elif reflect_pain > 0.3:
                    state_triggered = True
                    state_trigger_reason = f"pain={reflect_pain:.2f}"
                elif reflect_arousal > 0.5:
                    state_triggered = True
                    state_trigger_reason = f"arousal={reflect_arousal:.2f}"

                if state_triggered:
                    logger.info(f"[REFLECTION] State-triggered reflection: {state_trigger_reason}")
            except Exception as e:
                logger.debug(f"Failed to get state for reflection trigger: {e}")

        if estimated_turns < REFLECTION_MIN_EVIDENCE and not state_triggered:
            return _reflection_summary(
                status="skipped",
                reason="insufficient_evidence",
                estimated_turns=estimated_turns,
                pipeline_ran=False,
                rules=None,
            )

        # Persona rule candidates (turn_index forwarded for provenance)
        candidates = chat_service.reflection.generate_candidates(
            chat_service.session_history[session_id],
            max_candidates=3,
            session_id=session_id,
            turn_index=turn_index,
        )
        had_candidates = bool(candidates)
        if candidates:
            reflection_result = chat_service.reflection.process_and_replace(
                candidates, max_items=REFLECTION_MAX_RULES
            )

            chat_service.event_logger.log_event(
                session_id,
                "reflection",
                f"Generated {len(candidates)} candidates, result: {json.dumps(reflection_result, ensure_ascii=False)}",
            )

            # P3: feed structured outcomes into the meta rule learner
            if chat_service.meta_rule_learner:
                try:
                    if not isinstance(reflection_result, dict):
                        reflection_result = {}
                    success = reflection_result.get("added", 0) > 0 or reflection_result.get("merged", 0) > 0
                    outcome = "success" if success else "failure"

                    learning_event = {
                        "type": "rule_creation",
                        "context": {
                            "conversation_turns": estimated_turns,
                            "rule_type": "reflection",
                            "candidates_count": len(candidates),
                        },
                        "outcome": outcome,
                        "feedback": {
                            "positive": success,
                            "added": reflection_result.get("added", 0),
                            "merged": reflection_result.get("merged", 0),
                            "comment": f"Added {reflection_result.get('added', 0)} rules, Merged {reflection_result.get('merged', 0)} rules",
                        },
                    }

                    chat_service.meta_rule_learner.learn_from_experience(learning_event)
                except Exception as e:
                    logger.warning(f"Meta-learning failed: {e}")

            # [2026-01-26] Immediately project new rules into z_self (parity with mind wandering)
            if chat_service.self_model and (
                reflection_result.get("added", 0) > 0
                or reflection_result.get("merged", 0) > 0
                or reflection_result.get("updated", 0) > 0
            ):
                try:
                    all_rules = chat_service.persona_store.get_all_active(limit=1000)
                    chat_service.self_model.update_from_persona_rules(session_id, all_rules)
                    logger.info(
                        f"[REFLECTION] Synced {len(all_rules)} rules to z_self "
                        f"(added={reflection_result.get('added', 0)}, "
                        f"merged={reflection_result.get('merged', 0)}, "
                        f"updated={reflection_result.get('updated', 0)})"
                    )
                except Exception as e:
                    logger.error(f"[REFLECTION] Failed to sync rules to z_self: {e}", exc_info=True)

            if chat_service.self_model:
                if reflection_result.get("added", 0) > 0:
                    chat_service.self_model.trigger_event(session_id, "learning_moment", intensity=1.2)
                elif reflection_result.get("merged", 0) > 0:
                    chat_service.self_model.trigger_event(session_id, "learning_moment", intensity=0.6)

        emotion_candidates = []
        motivation_candidates = []
        somatic_candidates = []
        worldview_candidates = []
        quick_somatic_added = False
        quick_worldview_added = False

        if chat_service.self_model and getattr(chat_service.self_model, "emotion_store", None):
            emotion_candidates = chat_service.reflection.generate_emotion_candidates(
                chat_service.session_history[session_id], session_id=session_id
            )
        if chat_service.self_model and getattr(chat_service.self_model, "motivation_store", None):
            motivation_candidates = chat_service.reflection.generate_motivation_candidates(
                chat_service.session_history[session_id], session_id=session_id
            )
        if chat_service.self_model and getattr(chat_service.self_model, "somatic_store", None):
            current_z_self = chat_service.self_model.get_z_self(session_id)
            somatic_candidates = chat_service.reflection.generate_somatic_candidates(
                chat_service.session_history[session_id], session_id=session_id, z_self=current_z_self
            )
        if chat_service.self_model and getattr(chat_service.self_model, "world_store", None):
            worldview_candidates = chat_service.reflection.generate_worldview_candidates(
                chat_service.session_history[session_id], session_id=session_id
            )

        if not somatic_candidates and getattr(chat_service.self_model, "somatic_store", None):
            quick_somatic_added = chat_service._quick_add_somatic_from_state(session_id)
        if not worldview_candidates and getattr(chat_service.self_model, "world_store", None):
            quick_worldview_added = chat_service._quick_add_worldview_from_state(session_id)

        if chat_service.unified_processor and (emotion_candidates or motivation_candidates):
            chat_service.unified_processor.process_all_dimensions(
                rule_candidates=[],
                emotion_candidates=emotion_candidates,
                motivation_candidates=motivation_candidates,
            )

        if somatic_candidates and getattr(chat_service.self_model, "somatic_store", None):
            somatic_result = process_somatic_with_mmr(chat_service, somatic_candidates, session_id)
            if somatic_result.get("added", 0) > 0 or somatic_result.get("removed", 0) > 0:
                chat_service.self_model.sync_somatic_to_z_self(session_id)
        elif getattr(chat_service.self_model, "somatic_store", None) and not quick_somatic_added:
            quick_somatic_added = chat_service._quick_add_somatic_from_state(session_id)

        if worldview_candidates and getattr(chat_service.self_model, "world_store", None):
            worldview_result = process_worldview_with_mmr(chat_service, worldview_candidates, session_id)
            if worldview_result.get("added", 0) > 0 or worldview_result.get("removed", 0) > 0:
                chat_service.self_model.sync_worldview_to_z_self(session_id)
        elif getattr(chat_service.self_model, "world_store", None) and not quick_worldview_added:
            quick_worldview_added = chat_service._quick_add_worldview_from_state(session_id)
            if quick_worldview_added:
                try:
                    chat_service.self_model.sync_worldview_to_z_self(session_id)
                except Exception:
                    pass

        return _reflection_summary(
            status="ok",
            reason=None if had_candidates else "no_rule_candidates",
            estimated_turns=estimated_turns,
            pipeline_ran=True,
            rules=reflection_result,
        )

    except Exception as e:
        logger.error(f"Reflection failed: {e}", exc_info=True)
        return _reflection_summary(
            status="error",
            reason="exception",
            estimated_turns=estimated_turns,
            pipeline_ran=estimated_turns >= REFLECTION_MIN_EVIDENCE,
            rules=None,
            error=str(e),
        )
