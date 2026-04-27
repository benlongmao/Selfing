import logging
from typing import Tuple, Dict, List
from backend.config import config


def prepare_prompt_and_introspection(
    chat_service,
    user_input: str,
    z_self_summary: str,
    session_id: str,
    interaction_mode: Dict,
    concise_mode: bool,
    logger: logging.Logger,
    force_functional: bool = False,
    user_profile: Dict = None,
    turn_index: int = 0,
    ab_test: Dict = None,
) -> Tuple[str, Dict, str, List, bool]:
    """
    Decide introspection cadence, inject lightweight time context, refresh endogenous urges,
    and call ``PromptBuilder.build_with_introspection_prompt``.

    Returns:
        ``(system_prompt, introspection_config, user_input, urges, require_introspection)``
    """
    # Introspection cadence: off in concise mode; honor introspection_enabled / introspection_freq
    # - freq=0: disabled
    # - freq=1: every turn
    # - freq=N: every N-th turn (0-based turn_index)
    try:
        enabled = bool(config.get("system.introspection_enabled", True))
        freq = int(config.get("system.introspection_freq", 1) or 1)
    except Exception:
        enabled = True
        freq = 1

    if (not enabled) or concise_mode or freq <= 0:
        require_introspection = False
    elif freq == 1:
        require_introspection = True
    else:
        # When turn_index starts at 0, turns 0, N, 2N, ... fire introspection
        try:
            require_introspection = (int(turn_index) % int(freq) == 0)
        except Exception:
            require_introspection = True

    # [Removed 2026-03] MetaCognition: template-only layer with no behavioral value

    # [2026-02-28] Time questions: append factual clock string (informational, not an instruction)
    if user_input and any(
        kw in user_input.lower()
        for kw in [
            "几点",
            "时间",
            "time",
            "clock",
            "date",
            "日期",
            "what time",
            "current time",
            "today's date",
            "timezone",
        ]
    ):
        try:
            real_time = chat_service.clock_tool.get_current_time()
            time_str = real_time.get("local_readable", "Unknown")
            user_input += f"\n\n[Time context]\n{time_str}"
            logger.info(f"Injected real-time context for time query: {time_str}")
        except Exception as e:
            logger.warning(f"Failed to inject real-time context: {e}")

    urges = []
    if chat_service.endogenous_system:
        try:
            urges = chat_service.endogenous_system.update(session_id, user_input)
            if urges:
                logger.info(f"Endogenous urges triggered: {urges}")
        except Exception as e:
            logger.warning(f"Failed to update endogenous system: {e}")

    # Functional routing: testing / shipping / falsification questions force functional framing
    if force_functional:
        base_mode = interaction_mode if (interaction_mode and isinstance(interaction_mode, dict)) else {}
        reasons = list(base_mode.get("reasons", [])) if isinstance(base_mode.get("reasons", []), list) else []
        reasons.append("force_functional: apply framework to concrete system")
        interaction_mode = {
            **base_mode,
            "mode": base_mode.get("mode", "analytical"),
            "reasons": reasons,
            "force_functional": True,
        }

    system_prompt, introspection_config = chat_service.prompt_builder.build_with_introspection_prompt(
        user_input,
        z_self_summary,
        require_introspection=require_introspection,
        session_id=session_id,
        interaction_mode=interaction_mode,
        urges=urges,
        concise_mode=concise_mode,
        user_profile=user_profile,
        ab_disable_persona=bool((ab_test or {}).get("disable_persona")),
        ab_disable_identity=bool((ab_test or {}).get("disable_identity")),
        ab_disable_core_anchor=bool((ab_test or {}).get("disable_core_anchor")),
        ab_disable_collective_resonance=bool((ab_test or {}).get("disable_collective_resonance")),
        ab_raw_mode=bool((ab_test or {}).get("raw_mode")),
    )

    # Pass through for message_builder (e.g. somatic / pineal broadcast toggles)
    try:
        introspection_config["force_functional"] = bool(force_functional)
    except Exception:
        pass

    return system_prompt, introspection_config, user_input, urges, require_introspection
