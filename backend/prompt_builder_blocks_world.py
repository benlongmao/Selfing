"""
Inject readable world context into prompts: WorldState (situation) and WorldStore (beliefs).

Complements the numeric z_self cache slice [64:87] with human-readable text.
"""
from __future__ import annotations

from typing import List, Optional, Set, TYPE_CHECKING

from backend.prompt_builder_core import logger
from backend.world_state import WorldState
from backend.s_identity import get_effective_session

if TYPE_CHECKING:
    from backend.world_store import WorldStore


def build_world_situation_block(db_path: str, session_id: str) -> str:
    """Task phase, environment summary, and recent actions from WorldState."""
    try:
        sid = get_effective_session(session_id)
        ws = WorldState(db_path)
        text = ws.get_state_text(sid)
        if not text or not text.strip():
            return ""
        return f"""[Where I am in the world (task situation)]
{text}
"""
    except Exception as e:
        logger.debug(f"build_world_situation_block failed: {e}")
        return ""


def build_world_beliefs_block(
    world_store: "WorldStore",
    user_input: str,
    top_k: int = 3,
    max_locked: int = 1,
    concise: bool = False,
) -> str:
    """Semantic retrieval plus up to one locked core belief; in concise mode, one-line summary only."""
    if not user_input or not user_input.strip():
        user_input = "current conversation"
    lines: List[str] = []
    seen_ids: Set[str] = set()

    try:
        locked_added = 0
        if max_locked > 0:
            all_b = world_store.get_all_beliefs(status="active", limit=80)
            for b in all_b:
                if getattr(b, "locked", 0) and b.id not in seen_ids:
                    lines.append(f"- (core belief) {b.text}")
                    seen_ids.add(b.id)
                    locked_added += 1
                    if locked_added >= max_locked:
                        break

        k = 1 if concise else top_k
        hits = world_store.search_beliefs(user_input.strip(), top_k=k)
        for b in hits:
            if b.id in seen_ids:
                continue
            lines.append(f"- {b.text}")
            seen_ids.add(b.id)

        if not lines:
            return ""

        if concise:
            body = lines[0] if len(lines) == 1 else f"{lines[0]}; other related beliefs omitted for brevity."
        else:
            body = "\n".join(lines)

        return f"""[Worldview beliefs (relevant to this topic)]
{body}
"""
    except Exception as e:
        logger.debug(f"build_world_beliefs_block failed: {e}")
        return ""


def build_world_context_blocks(
    db_path: str,
    session_id: str,
    user_input: str,
    world_store: Optional["WorldStore"],
    concise_mode: bool = False,
) -> str:
    situation = build_world_situation_block(db_path, session_id)
    if concise_mode:
        return situation
    beliefs = ""
    if world_store:
        beliefs = build_world_beliefs_block(
            world_store, user_input, top_k=3, max_locked=1, concise=False
        )
    parts = [p for p in (situation, beliefs) if p]
    return "\n".join(parts)
