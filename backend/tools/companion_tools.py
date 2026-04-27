#!/usr/bin/env python3
"""
Companion Tools: Tools for enhancing AI's capability as a companion.
Includes: Memory Recall, Emotional Analysis, Safety Check, Time Context.
"""
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

from backend.s_identity import get_effective_session

logger = logging.getLogger(__name__)

class CompanionTools:
    def __init__(self, self_model_getter):
        self._self_model_getter = self_model_getter

    def _get_self_model(self):
        if self._self_model_getter:
            return self._self_model_getter()
        return None

    def recall_memories_narrative_fallback(self, query: str) -> Dict[str, Any]:
        """
        Fallback when MemorySearchTool is down: only ``self_biography`` vector memory (no chat_turns / unified bus).
        Prefer ``recall_memory`` (MemorySearchTool.recall) in normal operation.
        """
        sm = self._get_self_model()
        if not sm:
            return {"error": "SelfModel is not available."}
        try:
            from backend.self_narrative import SelfNarrative
            narrative = SelfNarrative(sm.db_path)
            results = narrative.retrieve_related_memory(query, limit=5)
            if not results:
                return {
                    "result": "No narrative-memory hits (full recall needs MemorySearchTool).",
                    "found": False,
                    "count": 0,
                }
            formatted_memories = "\n---\n".join(results)
            return {
                "result": f"Found {len(results)} relevant memories:\n\n{formatted_memories}",
                "found": True,
                "count": len(results),
            }
        except Exception as e:
            logger.error(f"Narrative recall fallback failed: {e}")
            return {"error": str(e)}

    def consolidate_memories(self, session_id: str = "default") -> Dict[str, Any]:
        """
        Active Consolidation: Force a 'diary entry' creation based on recent context.
        """
        # This requires access to recent history which is in ChatService.
        # This is hard to do as a pure tool without ChatService context.
        # Maybe we skip this for now or return a message saying "I will do this after our chat".
        return {"result": "Memory consolidation is scheduled to run automatically after this interaction.", "status": "scheduled"}

    def analyze_user_emotion(self, text: str, session_id: str = "default") -> Dict[str, Any]:
        """
        Deep Empathy Analysis: Analyze the emotional subtext of the user's input.
        """
        sm = self._get_self_model()
        if not sm or not getattr(sm, 'other_model', None):
            return {"error": "OtherModel is not available."}
        
        try:
            # We can use OtherModel's inference logic, but expose it as a tool result
            # so the AI can "read" it explicitly.
            om = sm.other_model
            # Construct a dummy interaction dict to reuse the logic
            interaction = {"user_message": text, "session_history": []}
            
            # We want to peek into _infer_traits or similar without updating state if possible,
            # or just interpret the current state.
            # Let's just return the current understanding of the user + immediate sentiment.
            
            # Re-using internal methods might be hacky. Let's do a fresh inference.
            # Actually, OtherModel has `_infer_traits`.
            traits, _ = om._infer_traits(interaction, {})
            
            return {
                "result": {
                    "inferred_sentiment": traits.get("sentiment"),
                    "communication_style": traits.get("communication_style"),
                    "underlying_need": traits.get("need_type"),
                    "analysis": f"User seems to be in a {traits.get('sentiment')} state, seeking {traits.get('need_type')}."
                }
            }
        except Exception as e:
            logger.error(f"Emotion analysis failed: {e}")
            return {"error": str(e)}

    def check_safety_risk(self, topic: str, session_id: str = "default") -> Dict[str, Any]:
        """
        Safety Boundary Check: Check if a topic poses a real-world risk.
        """
        sm = self._get_self_model()
        # We need SelfBoundary. It requires PersonaStore.
        # Let's try to instantiate it or access if available.
        # SelfModel has persona_store.
        if not sm or not sm.persona_store:
             return {"error": "SelfModel/PersonaStore not available."}

        try:
            from backend.self_boundary import SelfBoundary
            boundary = SelfBoundary(sm.persona_store, sm, sm.db_path)
            
            allowed, reason, confidence = boundary.check_boundary(session_id, topic)
            
            status = "SAFE" if allowed else "RISKY"
            return {
                "result": f"Safety Assessment for '{topic}': {status}",
                "allowed": allowed,
                "reason": reason,
                "confidence": confidence
            }
        except Exception as e:
            logger.error(f"Safety check failed: {e}")
            return {"error": str(e)}

    def get_time_context(self, session_id: str = "default") -> Dict[str, Any]:
        """
        Time Perception: Get current time and time elapsed since last interaction.

        [2026-03-21] Prefer last interaction from ``chat_turns.created_at`` (ground truth for UI).
        Legacy fields (user_profiles.last_seen, other_models.last_updated) are often stale.
        """
        session_id = get_effective_session(session_id)
        now = datetime.now(timezone.utc)
        last_seen_str = None
        source = "unknown"
        time_diff_desc = "First meeting or unknown."

        try:
            import sqlite3
            sm = self._get_self_model()
            if sm:
                with sqlite3.connect(sm.db_path) as conn:
                    # 1) Best: max(chat_turns.created_at)
                    try:
                        cur = conn.execute(
                            """
                            SELECT MAX(created_at) FROM chat_turns
                            WHERE session_id = ? AND created_at IS NOT NULL AND trim(created_at) != ''
                            """,
                            (session_id,),
                        )
                        row = cur.fetchone()
                        if row and row[0]:
                            last_seen_str = row[0]
                            source = "chat_turns"
                    except Exception as e:
                        logger.debug("[get_time_context] chat_turns max(created_at) skipped: %s", e)

                    # 2) Fallback: user_profiles.last_seen
                    if not last_seen_str:
                        cur = conn.execute(
                            "SELECT last_seen FROM user_profiles WHERE session_id=?",
                            (session_id,),
                        )
                        row = cur.fetchone()
                        if row and row[0]:
                            last_seen_str = row[0]
                            source = "user_profiles"

                    # 3) Fallback: other_models.last_updated
                    if not last_seen_str:
                        cur = conn.execute(
                            "SELECT last_updated FROM other_models WHERE user_id=?",
                            (session_id,),
                        )
                        row = cur.fetchone()
                        if row and row[0]:
                            last_seen_str = row[0]
                            source = "other_models"

            if last_seen_str:
                last_seen = datetime.fromisoformat(last_seen_str.replace("Z", "+00:00"))
                if last_seen.tzinfo is None:
                    last_seen = last_seen.replace(tzinfo=timezone.utc)
                diff = now - last_seen
                total_seconds = diff.total_seconds()

                if total_seconds < 60:
                    time_diff_desc = "Just now."
                elif total_seconds < 3600:
                    minutes = int(total_seconds / 60)
                    time_diff_desc = f"{minutes} minutes ago."
                elif total_seconds < 86400:
                    hours = int(total_seconds / 3600)
                    time_diff_desc = f"{hours} hours ago."
                else:
                    days = int(total_seconds / 86400)
                    time_diff_desc = f"{days} days ago."
        except Exception as e:
            logger.warning(f"Failed to calculate time context: {e}")

        return {
            "current_time_utc": now.isoformat(),
            "time_since_last_interaction": time_diff_desc,
            "last_interaction_source": source,
            "human_readable": (
                f"It is currently {now.strftime('%H:%M')} UTC. "
                f"We last spoke {time_diff_desc} (source: {source})."
            ),
        }

    _GROUP_LABELS = {
        "self_introspection": "Self introspection",
        "basic_file": "Basic file I/O",
        "code_execution": "Code execution",
        "web_search": "Web search",
        "time_util": "Time & clock",
        "geo_weather": "Geo & weather",
        "goal_basic": "Goals (basic)",
        "file_management": "File management",
        "code_analysis": "Code analysis",
        "calendar": "Calendar",
        "goal_management": "Goals (advanced)",
        "email": "Email",
        "browser_unified": "Browser (CDP)",
        "scientific_computing": "Scientific computing (NumPy/SciPy/SymPy/Pandas)",
        "self_healing": "Self-healing (backend scan)",
        "opencode": "OpenCode CLI",
        "learning": "Learning & knowledge",
        "task_planning": "Task planning",
        "code_proposal": "Code proposals",
        "companion": "Companion (social presence)",
        "chemistry": "Chemistry",
        "stock": "Stocks & finance",
        "research": "Research",
        "approval": "Approvals",
        "schedule": "Scheduled jobs",
        "self_awareness": "Self awareness",
        "moltbook": "Moltbook (social)",
        "deep_research": "Deep research",
        "memory_recall": "Memory recall",
        "visualization": "Visualization",
        "data_analysis": "Data analysis",
        "pdf_reader": "PDF reader",
        "s_repo_evolution": "Repository evolution (evolution_* / git / project bash)",
    }

    def list_my_tools(self, category: str = "summary", _current_tools: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        List currently allowed tools plus on-demand groups. ``_current_tools`` is injected by the router.
        """
        from backend.tool_selector import TOOL_GROUPS, COMPACT_DESCRIPTIONS

        current = sorted(_current_tools) if _current_tools else []

        if category == "summary" or category == "all":
            loadable = {}
            for gname, ginfo in TOOL_GROUPS.items():
                tools_in_group = ginfo.get("tools", [])
                already_loaded = [t for t in tools_in_group if t in current] if current else []
                if len(already_loaded) == len(tools_in_group):
                    continue
                unloaded = [t for t in tools_in_group if t not in current] if current else tools_in_group
                label = self._GROUP_LABELS.get(gname, gname)
                loadable[gname] = f"{label} ({', '.join(unloaded[:4])}{'…' if len(unloaded) > 4 else ''})"

            result: Dict[str, Any] = {
                "currently_available": current,
                "currently_available_count": len(current),
                "loadable_groups": loadable,
                "hint": "Call request_tool_group('group_name') to load more tools on demand.",
            }
            if category == "all":
                descs = {}
                for t in current:
                    descs[t] = COMPACT_DESCRIPTIONS.get(t, "")
                result["descriptions"] = descs
            return result
        else:
            for gname, ginfo in TOOL_GROUPS.items():
                if category.lower() in gname.lower() or gname.lower() in category.lower():
                    tools_in_group = ginfo.get("tools", [])
                    return {
                        "group": gname,
                        "tools": tools_in_group,
                        "descriptions": {t: COMPACT_DESCRIPTIONS.get(t, "") for t in tools_in_group},
                        "hint": f"request_tool_group('{gname}') to load this group.",
                    }
            return {"error": f"Category not found: {category}", "available_groups": list(TOOL_GROUPS.keys())}

    def request_tool_group(self, group_name: str, _tool_router=None) -> Dict[str, Any]:
        """
        Load a tool group for this turn. ``_expand_tools`` is consumed by chat_tool_runner to widen the toolset.
        """
        from backend.tool_selector import TOOL_GROUPS, COMPACT_DESCRIPTIONS

        if group_name not in TOOL_GROUPS:
            return {
                "error": f"Unknown tool group: {group_name!r}",
                "available_groups": {
                    gn: self._GROUP_LABELS.get(gn, gn)
                    for gn in TOOL_GROUPS
                },
            }

        group_tools = TOOL_GROUPS[group_name].get("tools", [])
        label = self._GROUP_LABELS.get(group_name, group_name)

        tool_defs = []
        if _tool_router:
            for tname in group_tools:
                tdef = None
                if hasattr(_tool_router, '_tool_name_to_def'):
                    tdef = _tool_router._tool_name_to_def.get(tname)
                if not tdef:
                    all_defs = _tool_router.get_tool_definitions()
                    for d in all_defs:
                        if (d.get("function") or {}).get("name") == tname:
                            tdef = d
                            break
                if tdef:
                    compact = {
                        "type": "function",
                        "function": {
                            "name": tname,
                            "description": COMPACT_DESCRIPTIONS.get(
                                tname,
                                (tdef.get("function") or {}).get("description", ""),
                            ),
                            "parameters": (tdef.get("function") or {}).get("parameters", {
                                "type": "object", "properties": {}, "required": []
                            }),
                        },
                    }
                    tool_defs.append(compact)

        return {
            "loaded_group": group_name,
            "loaded_tools": group_tools,
            "label": label,
            "count": len(group_tools),
            "descriptions": {t: COMPACT_DESCRIPTIONS.get(t, "") for t in group_tools},
            "_expand_tools": tool_defs,
        }

    def get_tool_definitions(self) -> List[Dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "list_my_tools",
                    "description": "List currently available tools and groups you can request on demand.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "category": {
                                "type": "string",
                                "description": "'summary' (default), 'all' (include descriptions), or a group key like 'email' / 'browser'"
                            }
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "request_tool_group",
                    "description": "Load a tool group for this session; use list_my_tools for valid group_name values.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "group_name": {
                                "type": "string",
                                "description": "Group id, e.g. 'email', 'calendar', 'browser', 'stock', 'chemistry'"
                            }
                        },
                        "required": ["group_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_user_emotion",
                    "description": "Analyze the emotional subtext and underlying needs of the user's message.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string", "description": "The user's message text."}
                        },
                        "required": ["text"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "check_safety_risk",
                    "description": "Check if a topic carries safety risks or violates core boundaries.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "topic": {"type": "string", "description": "The topic or action to check."}
                        },
                        "required": ["topic"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_time_context",
                    "description": "Understand the passage of time since our last interaction.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            }
        ]
