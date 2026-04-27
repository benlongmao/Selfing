import logging
import numpy as np
from backend.emotion_store import EMOTION_SUBSPACE_DIMS


def apply_tool_feedback(self_model, tool_result, session_id: str, logger: logging.Logger) -> None:
    """
    Feed tool outcomes back into energy / emotion / somatic state.

    [2026-03-30] Prefer the homeostasis event path when available.
    """
    if self_model is None:
        return

    is_success = "error" not in tool_result
    try:
        if is_success:
            # [2026-03-30] Success path via events when homeostasis is present
            if hasattr(self_model, 'homeostasis'):
                self_model.homeostasis.process_event(session_id, "task_completed", intensity=0.8)
            else:
                self_model.update_energy(session_id, 1.0)
            
            if self_model.emotion_store:
                emotion_delta = np.zeros(16, dtype=np.float32)
                emotion_delta[EMOTION_SUBSPACE_DIMS["pleasure"][0]:EMOTION_SUBSPACE_DIMS["pleasure"][1]] = 0.05
                emotion_delta[EMOTION_SUBSPACE_DIMS["dominance"][0]:EMOTION_SUBSPACE_DIMS["dominance"][1]] = 0.05
                self_model.emotion_store.update_emotion(
                    session_id, emotion_delta, trigger_source="action_success"
                )
        else:
            # [2026-03-30] Failure path via events when homeostasis is present
            if hasattr(self_model, 'homeostasis'):
                self_model.homeostasis.process_event(session_id, "tool_error", intensity=1.0)
            else:
                self_model.update_energy(session_id, -3.0)
            
            if self_model.emotion_store:
                emotion_delta = np.zeros(16, dtype=np.float32)
                emotion_delta[EMOTION_SUBSPACE_DIMS["pleasure"][0]:EMOTION_SUBSPACE_DIMS["pleasure"][1]] = -0.3
                self_model.emotion_store.update_emotion(
                    session_id, emotion_delta, trigger_source="action_failed"
                )

                if self_model.somatic_store:
                    z_self = self_model.get_z_self(session_id)
                    if z_self is not None and z_self.shape[0] >= 100:
                        z_self[96:100] += 0.3
                        self_model._save_z_self(session_id, z_self)
    except Exception as e:
        logger.warning(f"Failed to update state from tool execution: {e}")

