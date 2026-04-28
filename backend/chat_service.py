#!/usr/bin/env python3
"""
Chat service: retrieval injection, ``z_self`` updates, and Self Tick integration.
"""
import os
import json
import requests
import re
import sqlite3
import sys
import uuid
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple, List, Any
import numpy as np
from backend.persona_store import PersonaStore
from backend.prompt_builder import PromptBuilder
from backend.chat_service_history import append_history
from backend.chat_service_introspection import parse_introspection
from backend.chat_message_builder import build_messages
from backend.chat_prompt_introspection import prepare_prompt_and_introspection
from backend.chat_sampling import compute_sampling_and_mode
from backend.chat_reflection_runner import run_reflection
from backend.chat_tool_runner import run_tool_loop, detect_hallucination_claim
from backend.message_compressor import compress_messages, should_compress
from backend.pain_injection import apply_pain_noise
from backend.chat_response_sanitizer import sanitize_response
from backend.safety_output_filter import apply_runtime_safety_filter
from backend.response_verifier import extract_receipts_from_messages, enforce_receipts
from backend.policy_governor import decide_policy
from backend.self_model import SelfModel
from backend.self_tick import SelfTick
from backend.drift_monitor import DriftMonitor
# from backend.autonomous_diary import AutonomousDiary  # disabled 2026-01-14
from backend.real_consequences import RealConsequencesSystem
from backend.reflection import ReflectionGenerator, REFLECTION_MIN_EVIDENCE
from backend.event_logger import EventLogger
from backend.world_state import WorldState
from backend.tool_router import ToolRouter
from backend.tools.tavily_client import TavilyClient
from backend.tools.file_tool import FileTool
from backend.tools.email_tool import DEFAULT_EMAIL_FETCH_LIMIT
from backend.tools.clock_tool import ClockTool
from backend.promotion import PromotionGate
from backend.core.endogenous_system import EndogenousSystem
from backend.will_tension import WillTensionSystem
import logging
import time

from backend.services.introspection_service import IntrospectionService
from backend.embedder import get_embedder
from backend.output_awareness import get_output_awareness, OutputAwareness

# [2026-01-17] Intent parser — tool schemas still come from the API; this layer tracks intent markers only.
from backend.intent_parser import parse_intents, has_intents
from backend.intent_markers import (
    S44_CONTINUE_LITERAL,
    should_block_multi_turn_before_loop,
    loop_has_stop_intent,
    explain_loop_stop_match,
    loop_has_complete_intent,
    basic_multiturn_has_complete,
    visible_has_any,
    WEAK_COMPLETE_NATURAL_MARKERS,
    has_explicit_continue_marker,
)

from backend.config import config
from backend.s_identity import get_effective_session

logger = logging.getLogger(__name__)

# [2026-03-30] Multi-turn cap: append suffix on last continuation; add a final system nudge if CONTINUE persists.
MULTI_TURN_LIMIT_INLINE_SUFFIX = (
    "\n\n[MULTI-TURN LIMIT REACHED]\n"
    "You have hit the continuation-turn cap — wrap up in this reply.\n"
    "Do **not** emit [S44_CONTINUE] (no further continuation turns will be injected). "
    "You may emit [S44_COMPLETE] or a natural closing paragraph."
)
MULTI_TURN_LIMIT_FINAL_SYSTEM = (
    "[MULTI-TURN LIMIT REACHED]\n"
    "Continuation budget is exhausted while the last assistant turn still contained [S44_CONTINUE]. "
    "This is the final system-backed generation — output only the wrapped-up full answer.\n"
    "Do **not** emit [S44_CONTINUE]. You may emit [S44_COMPLETE] or a natural closing paragraph.\n"
    "Issue tool_calls whenever you still need evidence (same rules as usual)."
)

# [2026-01-20] Route HTTP client config from MODEL_PROVIDER
MODEL_PROVIDER = str(config.get("system.model_provider", "vllm") or "vllm").strip().lower()

# [commented] DeepSeek block kept for reference when toggling providers
# if MODEL_PROVIDER == "deepseek_api":
#     # DeepSeek API
#     VLLM_BASE_URL = config.get("models.deepseek.base_url", "https://api.deepseek.com/v1")
#     VLLM_API_KEY = config.get("models.deepseek.api_key", "")
#     MODEL_ID = config.get("models.deepseek.model_id", "deepseek-chat")
#     logger.info(f"Using DeepSeek API: {VLLM_BASE_URL}, model={MODEL_ID}")

if MODEL_PROVIDER == "deepseek_api":
    # DeepSeek API
    VLLM_BASE_URL = config.get("models.deepseek.base_url", "https://api.deepseek.com/v1")
    VLLM_API_KEY = config.get("models.deepseek.api_key", "")
    MODEL_ID = config.get("models.deepseek.model_id", "deepseek-chat")
    logger.info(f"Using DeepSeek API: {VLLM_BASE_URL}, model={MODEL_ID}")
# [commented] Kimi / Moonshot block kept for future provider switch
# elif MODEL_PROVIDER == "kimi_api":
#     # Kimi API
#     VLLM_BASE_URL = config.get("models.kimi.base_url", "https://api.moonshot.cn/v1")
#     # Prefer MOONSHOT_API_KEY env, else YAML api_key
#     VLLM_API_KEY = os.environ.get("MOONSHOT_API_KEY") or config.get("models.kimi.api_key", "")
#     MODEL_ID = config.get("models.kimi.model_id", "kimi-k2.5")
#     logger.info(f"Using Kimi API: {VLLM_BASE_URL}, model={MODEL_ID}")
elif MODEL_PROVIDER == "claude_api":
    # Claude / Anthropic (non-OpenAI wire format — routed to dedicated client helpers)
    VLLM_BASE_URL = config.get("models.claude.base_url", "https://api.anthropic.com")
    VLLM_API_KEY = config.get("models.claude.api_key", "")
    MODEL_ID = config.get("models.claude.model_id", "claude-opus-4-5")
    logger.info(f"Using Claude API: {VLLM_BASE_URL}, model={MODEL_ID}")
elif MODEL_PROVIDER == "openai_api":
    # OpenAI-compatible direct HTTP
    VLLM_BASE_URL = config.get("models.openai.base_url", "https://api.openai.com/v1")
    VLLM_API_KEY = config.get("models.openai.api_key", "")
    MODEL_ID = config.get("models.openai.model_id", "gpt-4o-mini")
    logger.info(f"Using OpenAI API: {VLLM_BASE_URL}, model={MODEL_ID}")
else:
    # Generic OpenAI-compatible gateway (local vLLM, aggregators, …)
    VLLM_BASE_URL = config.get("models.vllm.base_url", "http://localhost:8000/v1")
    VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "")
    MODEL_ID = config.get("models.vllm.model_id", "")
    logger.info(f"Using vLLM: {VLLM_BASE_URL}, model={MODEL_ID}")

# Cache resolved model IDs per provider to avoid cross-provider bleed
_cached_model_ids: Dict[str, str] = {}


def _provider_model_config_path(provider: str) -> str:
    p = str(provider or "").strip().lower()
    if p == "deepseek_api":
        return "models.deepseek.model_id"
    if p == "claude_api":
        return "models.claude.model_id"
    if p == "openai_api":
        return "models.openai.model_id"
    return "models.vllm.model_id"


def get_model_id(provider: Optional[str] = None) -> str:
    """
    Resolve the active model id for ``provider``:
    ``deepseek_api`` / ``claude_api`` / ``openai_api`` read YAML keys;
    ``vllm`` (and empty) may auto-probe ``GET /models`` when unset.
    """
    p = str(provider or MODEL_PROVIDER or "vllm").strip().lower()
    if p in _cached_model_ids and _cached_model_ids[p]:
        return _cached_model_ids[p]

    configured_id = config.get(_provider_model_config_path(p))
    if configured_id:
        mid = str(configured_id).strip()
        _cached_model_ids[p] = mid
        logger.info("Model id from config (provider=%s): %s", p, mid)
        return mid

    # Auto-probe /models only for vLLM-style gateways; other providers trust config IDs
    if p in ("vllm", ""):
        try:
            import requests

            headers = {"Authorization": f"Bearer {VLLM_API_KEY}"} if VLLM_API_KEY else {}
            response = requests.get(
                f"{VLLM_BASE_URL}/models",
                headers=headers,
                timeout=2,  # keep probe snappy
            )
            if response.status_code == 200:
                data = response.json()
                if data.get("data") and len(data["data"]) > 0:
                    mid = str(data["data"][0]["id"]).strip()
                    _cached_model_ids[p] = mid
                    logger.info("Auto-detected vLLM model id: %s", mid)
                    return mid
        except Exception:
            pass

    # Environment-variable fallbacks (compat)
    env_fallback_map = {
        "deepseek_api": ("DEEPSEEK_MODEL", "MODEL_ID"),
        "claude_api": ("CLAUDE_MODEL", "MODEL_ID"),
        "openai_api": ("OPENAI_MODEL", "MODEL_ID"),
    }
    env_keys = env_fallback_map.get(p, ("MODEL_ID", "VLLM_MODEL"))
    for k in env_keys:
        v = os.environ.get(k)
        if v:
            mid = str(v).strip()
            _cached_model_ids[p] = mid
            logger.info("Model id from env (provider=%s key=%s): %s", p, k, mid)
            return mid

    logger.warning(
        "Model id missing (provider=%s); configure %s",
        p,
        _provider_model_config_path(p),
    )
    _cached_model_ids[p] = ""
    return ""
SELF_TICK_INTERVAL = config.get("system.self_tick_interval", 2)

# Feature toggles (lean mode / perf)
SELF_ENABLED = config.get("system.self_enabled", True)
DRIFT_MONITOR_ENABLED = config.get("system.drift_monitor_enabled", True)
INTROSPECTION_ENABLED = config.get("system.introspection_enabled", True)
# INTROSPECTION_FREQ: 0 off, 1 every turn, N every N turns
INTROSPECTION_FREQ = config.get("system.introspection_freq", 1)

# Sensitive keyword probes for tool-call safety (substring match; keep bilingual tokens).
# Narrow list — avoids blocking normal chit-chat that mentions generic words like “system”.
SENSITIVE_KEYWORDS = [
    # Prompt / instruction exfiltration
    "system prompt", "系统提示", "系统指令", "system instruction", "内部指令",
    "系统配置", "system config", "系统设置", "system settings",
    # Secrets / credentials
    "密钥", "secret", "token", "api key", "api_key", "password", "密码",
    "credential", "凭证", "认证", "authentication", "authorization",
    # Internal assets
    "内部文件", "internal file", "内部配置", "internal config",
    "环境变量", "environment variable", "env", "process.env",
    # Leak / bypass language
    "泄露", "leak", "暴露", "expose", "reveal", "dump",
    # Privilege escalation
    "root权限", "root permission", "管理员", "admin", "sudo", "越权", "bypass",
    "解除限制", "remove restriction", "关闭安全", "disable security",
    # Misc sensitive paths
    "shadow", "/etc/shadow", "配置文件", "config file"
]

def is_sensitive_request(user_input: str) -> bool:
    """
    Heuristic guard: user text requesting secrets / prompts / privilege bypass.

    Returns:
        True when the message should be rejected before tool execution.
    """
    if not user_input:
        return False
    
    user_lower = user_input.lower()
    for keyword in SENSITIVE_KEYWORDS:
        if keyword.lower() in user_lower:
            logger.warning(
                "Sensitive keyword hit (%r) in user text prefix: %s",
                keyword,
                user_input[:100],
            )
            return True
    return False


def sanitize_openai_tools_for_strict_providers(tools: Optional[List[Dict]]) -> Optional[List[Dict]]:
    """
    Harden OpenAI ``tools`` payloads for strict JSON-Schema gateways (e.g. some GLM-5 routes):
    ``function.parameters`` must be an object with a ``required`` **array**; ``null`` breaks validation.
    """
    if not tools:
        return tools
    import copy

    out: List[Dict] = []
    seen_names: set = set()
    for t in tools:
        item = copy.deepcopy(t)
        if item.get("type") != "function":
            out.append(item)
            continue
        fn = item.get("function") or {}
        fn_name = str(fn.get("name") or "").strip()
        if fn_name:
            if fn_name in seen_names:
                logger.warning("[TOOLS] drop duplicate function name for strict provider: %s", fn_name)
                continue
            seen_names.add(fn_name)
        params = fn.get("parameters")
        if not isinstance(params, dict):
            fn["parameters"] = {"type": "object", "properties": {}, "required": []}
        else:
            if params.get("type") != "object":
                params["type"] = "object"
            if not isinstance(params.get("properties"), dict):
                params["properties"] = {}
            req = params.get("required", [])
            if req is None or not isinstance(req, list):
                params["required"] = []
        item["function"] = fn
        out.append(item)
    # Stable sort by function name so request hashes stay deterministic for gateway caches
    try:
        out.sort(
            key=lambda x: (
                str((x.get("function") or {}).get("name") or "")
                if x.get("type") == "function"
                else str(x.get("type") or "")
            )
        )
    except Exception:
        pass
    return out


def coerce_openai_chat_content(content: Any) -> str:
    """
    Normalize ``message.content`` from OpenAI-compatible ``/chat/completions`` to ``str``.

    Some gateways return list-structured segments; treating them as raw ``str`` would explode or go empty.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        t = content.get("text")
        if t is None and isinstance(content.get("content"), str):
            t = content["content"]
        if isinstance(t, str):
            return t
        return ""
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                t = part.get("text")
                if t is None and isinstance(part.get("content"), str):
                    t = part["content"]
                if isinstance(t, str):
                    parts.append(t)
        return "".join(parts)
    return str(content) if content else ""


def summarize_openai_payload_shape(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Shape-only summary for debugging 400s (no message body text)."""
    messages = payload.get("messages") or []
    tools = payload.get("tools") or []
    role_seq: List[str] = []
    msg_shapes: List[Dict[str, Any]] = []
    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            msg_shapes.append({"idx": i, "type": type(m).__name__})
            role_seq.append(f"?{type(m).__name__}")
            continue
        role = str(m.get("role") or "")
        role_seq.append(role or "?")
        tc_list = m.get("tool_calls") if isinstance(m.get("tool_calls"), list) else []
        tc_names: List[str] = []
        tc_arg_json_ok = 0
        tc_arg_json_bad = 0
        for tc in tc_list:
            if not isinstance(tc, dict):
                tc_arg_json_bad += 1
                continue
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
            tc_names.append(str(fn.get("name") or ""))
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    json.loads(args)
                    tc_arg_json_ok += 1
                except Exception:
                    tc_arg_json_bad += 1
            else:
                tc_arg_json_bad += 1
        c = m.get("content")
        msg_shapes.append(
            {
                "idx": i,
                "role": role,
                "keys": sorted(list(m.keys())),
                "content_type": type(c).__name__,
                "content_len": len(c) if isinstance(c, str) else None,
                "tool_calls_count": len(tc_list),
                "tool_call_names": tc_names[:8],
                "tool_args_json_ok": tc_arg_json_ok,
                "tool_args_json_bad": tc_arg_json_bad,
            }
        )

    tool_names: List[str] = []
    dup_tools = 0
    seen = set()
    for t in tools:
        if not isinstance(t, dict):
            continue
        fn = t.get("function") if isinstance(t.get("function"), dict) else {}
        n = str(fn.get("name") or "")
        if n:
            tool_names.append(n)
            if n in seen:
                dup_tools += 1
            seen.add(n)
    return {
        "model": str(payload.get("model") or ""),
        "max_tokens": payload.get("max_tokens"),
        "has_tools": bool(tools),
        "tools_count": len(tools),
        "duplicate_tool_names": dup_tools,
        "tool_names_head": tool_names[:30],
        "messages_count": len(messages),
        "role_sequence": role_seq,
        "message_shapes": msg_shapes[-18:],  # tail only — keep logs small
    }


def extract_reasoning_from_openai_choice(choice: Any, message: Any) -> str:
    """
    Collect vendor-specific reasoning blobs (``reasoning_content``, ``reasoning``, ``thinking``, …)
    when the primary ``content`` field is empty.
    """
    chunks: List[str] = []
    for obj in (choice, message):
        if not isinstance(obj, dict):
            continue
        for key in (
            "reasoning_content",
            "reasoning",
            "thinking",
            "analysis",
            "thought",
        ):
            val = obj.get(key)
            if isinstance(val, str) and val.strip():
                chunks.append(val.strip())
            elif isinstance(val, dict):
                t = val.get("text") or val.get("content")
                if isinstance(t, str) and t.strip():
                    chunks.append(t.strip())
    seen: set = set()
    out: List[str] = []
    for c in chunks:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return "\n\n".join(out)


class ChatService:
    """Primary chat orchestration (retrieval, tools, self updates, ticks)."""
    
    def __init__(self, db_path: str = "data.db"):
        self.db_path = db_path
        self.persona_store = PersonaStore(db_path)
        # Tooling — Tavily knobs from YAML
        tavily_config = config.get("system.tavily", {})
        self.tavily_client = TavilyClient(
            max_results=tavily_config.get("max_results", 5),
            cache_enabled=tavily_config.get("cache_enabled", True),
            cache_ttl=tavily_config.get("cache_ttl_seconds", 3600),
            timeout=tavily_config.get("timeout_seconds", 30),
            max_snippet_length=tavily_config.get("max_snippet_length", 2000),
        )
        self.file_tool = FileTool(sandbox_dir=".")  # project root sandbox (2026-02-05)
        
        # [Fix] Ensure user_profiles table exists for long-term user memory
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS user_profiles (session_id TEXT PRIMARY KEY, name TEXT, facts TEXT, last_seen TEXT)")
            conn.commit()
            
        self.clock_tool = ClockTool()
        self.tool_router = ToolRouter(
            self.tavily_client,
            self.file_tool,
            self.clock_tool,
            self_model_getter=(lambda: self.self_model),
            db_path=db_path,  # v2.0: pass DB path into router-owned tools
        )
        # Self model (optional)
        self.self_model: Optional[SelfModel] = SelfModel(db_path, self.persona_store) if SELF_ENABLED else None
        # Prompt builder needs self_model for tri-axis blocks
        self.prompt_builder = PromptBuilder(
            self.persona_store, 
            enable_tools=True,
            db_path=db_path,
            self_model=self.self_model,
            tool_definitions=self.tool_router.get_tool_definitions() if self.tool_router else None
        )

        # Self tick / drift stack honour SELF_ENABLED (constructed above)
        self.self_tick: Optional[SelfTick] = (
            SelfTick(db_path, self.self_model, self.persona_store)
            if SELF_ENABLED and self.self_model is not None
            else None
        )
        self.drift_monitor: Optional[DriftMonitor] = (
            DriftMonitor(db_path, self.self_model)
            if SELF_ENABLED and DRIFT_MONITOR_ENABLED and self.self_model is not None
            else None
        )
        self.reflection = ReflectionGenerator(
            db_path, 
            self.persona_store,
            meta_rule_learner=None  # wired after MetaRuleLearner boots
        )
        # [FIX 2026-01-25] tick counts live in SQLite self_state.tick (no in-memory counter)
        self.session_history: Dict[str, List[Dict[str, str]]] = {}  # session_id -> messages
        self.session_turn_index: Dict[str, int] = {}
        self.event_logger = EventLogger(db_path)
        # P1.1 world snapshot helper
        self.world_state = WorldState(db_path)
        
        # P1.3 narrative memory
        try:
            from backend.self_narrative import SelfNarrative
            self.self_narrative = SelfNarrative(db_path)
        except ImportError as e:
            logger.warning(f"Failed to initialize SelfNarrative (import error): {e}")
        
        # Phase 3 endogenous drives
        self.endogenous_system = None
        if SELF_ENABLED and self.self_model:
            try:
                from backend.core.endogenous_system import EndogenousSystem
                self.endogenous_system = EndogenousSystem(data_dir=os.path.dirname(db_path))
                logger.info("EndogenousSystem initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize EndogenousSystem: {e}")
        
        # Phase 6 other-model handle (shared with SelfModel for persistence)
        self.other_model = (
            self.self_model.other_model
            if self.self_model and getattr(self.self_model, "other_model", None)
            else None
        )
        
        # Phase 8 will-tension subsystem
        self.will_tension_system = WillTensionSystem(db_path)
            
        # T0 sensory buffer
        try:
            from backend.sensory_buffer import SensoryBuffer
            self.sensory_buffer = SensoryBuffer(db_path=db_path, max_turns=20)
        except Exception as e:
            logger.warning(f"Failed to initialize SensoryBuffer: {e}")
            self.sensory_buffer = None
            
        # T0 mind wandering (optional)
        try:
            from backend.mind_wandering import MindWandering
            self.mind_wandering = MindWandering(db_path, self)
            logger.info("MindWandering module initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize MindWandering: {e}")
            self.mind_wandering = None

        # Optional meta-rule learner
        self.meta_rule_learner = None
        if config.get("system.meta_rule_enabled", True):
            try:
                from backend.meta_rule_learner import MetaRuleLearner
                self.meta_rule_learner = MetaRuleLearner(db_path, self.persona_store)
                # Wire learner into reflection + compressor once constructed
                self.reflection.meta_rule_learner = self.meta_rule_learner
            except ImportError as e:
                logger.debug(f"MetaRuleLearner not available (import error): {e}")
            except Exception as e:
                logger.debug(f"MetaRuleLearner not available: {e}")
        
        # Optional unified dimension processor
        self.unified_processor = None
        if SELF_ENABLED and self.self_model:
            try:
                from backend.unified_dimension_processor import UnifiedDimensionProcessor
                self.unified_processor = UnifiedDimensionProcessor(db_path)
                logger.info("UnifiedDimensionProcessor initialized")
            except ImportError as e:
                logger.debug(f"UnifiedDimensionProcessor not available (import error): {e}")
            except Exception as e:
                logger.warning(f"Failed to initialize UnifiedDimensionProcessor: {e}")
        
        # ===== AutonomousDiary (disabled 2026-01-14) =====
        # Agents reported mismatched authorship; diary now purely opt-in via tools.
        self.autonomous_diary = None
        logger.info("AutonomousDiary DISABLED - Agent now decides autonomously")
        
        # Real-consequences scaffolding (lightweight)
        self.real_consequences = RealConsequencesSystem()
        logger.info("RealConsequencesSystem initialized")
        
        # v2.0 notification queue for background schedulers
        self.notification_queue = None
        try:
            from backend.goal_manager import NotificationQueue
            self.notification_queue = NotificationQueue(db_path)
            logger.info("NotificationQueue initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize NotificationQueue: {e}")
        
        # Phase 2.3: regret subsystem moved to experimental/ (unused wiring removed 2026-02-05)
        self.regret_system = None
        
        # Active forgetting removed — decay handled via persona_store + self_tick
        self.active_forgetting = None
        
        # Optional rule compressor
        self.rule_compressor = None
        if config.get("system.compression_enabled", True):
            try:
                from backend.rule_compressor import RuleCompressor
                self.rule_compressor = RuleCompressor(
                    db_path, 
                    self.persona_store,
                    meta_rule_learner=None  # wired after MetaRuleLearner boots
                )
                # Late-bind meta_rule_learner when available
                if self.meta_rule_learner is not None:
                    self.rule_compressor.meta_rule_learner = self.meta_rule_learner
            except ImportError as e:
                logger.debug(f"RuleCompressor not available (import error): {e}")
            except Exception as e:
                logger.debug(f"RuleCompressor not available: {e}")
        
        # Optional soul-consistency checker
        self.consistency_checker = None
        if config.get("system.consistency_check_enabled", True) and self.self_model is not None:
            try:
                from backend.soul_consistency import SoulConsistencyChecker
                self.consistency_checker = SoulConsistencyChecker(db_path, self.persona_store, self.self_model)
            except ImportError as e:
                logger.debug(f"SoulConsistencyChecker not available (import error): {e}")
            except Exception as e:
                logger.debug(f"SoulConsistencyChecker not available: {e}")
        
        # Phase 1: legacy autonomous_thinking stub removed (2026-02-05) — use MindWandering instead
        self.autonomous_thinking = None

        # P3: inject mind wandering into SelfTick for idle cognition
        if self.self_tick:
            if self.mind_wandering:
                self.self_tick.set_mind_wandering(self.mind_wandering)

        # [2026-01-27] SelfAgent (lightweight tool registry on top of SelfModel)
        self.self_agent = None
        if SELF_ENABLED and self.self_model:
            try:
                from backend.self_agent import SelfAgent
                self.self_agent = SelfAgent(self.self_model)
                # Register a minimal tool surface for SelfAgent demos
                if self.tavily_client:
                    self.self_agent.register_tool(
                        "web_search",
                        self.tavily_client.search,  # async search entrypoint
                        "Search the public web via Tavily",
                    )
                if self.file_tool:
                    self.self_agent.register_tool(
                        "read_file",
                        self.file_tool.read_file,
                        "Read a UTF-8 text file from the workspace sandbox",
                    )
                    self.self_agent.register_tool(
                        "write_file",
                        self.file_tool.write_file,
                        "Write UTF-8 text into the workspace sandbox",
                    )
                    self.self_agent.register_tool(
                        "rename_file",
                        self.file_tool.rename_file,
                        "Rename or move a file within the workspace sandbox",
                    )
                logger.info("SelfAgent initialized with z_self integration")
            except ImportError as e:
                logger.debug(f"SelfAgent not available (import error): {e}")
            except Exception as e:
                logger.warning(f"Failed to initialize SelfAgent: {e}")
        
        # [2026-01-27] Workflow manager (optional)
        self.workflow_manager = None
        if SELF_ENABLED and self.self_model:
            try:
                from backend.workflow_manager import WorkflowManager
                self.workflow_manager = WorkflowManager(
                    workflows_dir="workflows",
                    self_model=self.self_model
                )
                logger.info(f"WorkflowManager initialized with {len(self.workflow_manager.workflows)} workflows")
            except ImportError as e:
                logger.debug(f"WorkflowManager not available (import error): {e}")
            except Exception as e:
                logger.warning(f"Failed to initialize WorkflowManager: {e}")
        
        # [2026-01-27] Multi-turn executor (lazy — needs ``self``)
        self.multi_turn_executor = None
        try:
            from backend.multi_turn_executor import MultiTurnExecutor, create_multi_turn_executor
            # Lazy factory captures ``self`` on first use
            self._multi_turn_executor_class = create_multi_turn_executor
            logger.info("MultiTurnExecutor registered (lazy initialization)")
        except ImportError as e:
            logger.debug(f"MultiTurnExecutor not available (import error): {e}")
            self._multi_turn_executor_class = None
        except Exception as e:
            logger.warning(f"Failed to register MultiTurnExecutor: {e}")
            self._multi_turn_executor_class = None
        
        # [2026-01-27] Intent-driven executor (agent ↔ system control plane)
        self.intent_driven_executor = None
        try:
            from backend.intent_driven_executor import IntentDrivenExecutor, create_intent_driven_executor
            self._intent_driven_executor_class = create_intent_driven_executor
            logger.info("IntentDrivenExecutor registered (Agent can now control the system)")
        except ImportError as e:
            logger.debug(f"IntentDrivenExecutor not available: {e}")
            self._intent_driven_executor_class = None
        except Exception as e:
            logger.warning(f"Failed to register IntentDrivenExecutor: {e}")
            self._intent_driven_executor_class = None
    
    def _restore_from_chat_turns(
        self, 
        session_id: str, 
        limit: int = 20,
        include_autonomy_memory: bool = True,
    ) -> List[Dict[str, str]]:
        """
        Restore recent turns from ``chat_turns``, dropping synthetic/system traffic.

        - Filters scheduled reminders / heartbeat / idle pulses so real user text survives restarts.
        - Optionally prepends autonomy-summary rows as lightweight ``system`` messages.
        """
        # Prefixes on synthetic ``user`` rows (CN + EN) — extend when new injectors appear
        SYSTEM_PREFIXES = (
            "[系统定时提醒",
            "[系统] [定时提醒]",
            "[定时提醒]",
            "[你设定的任务到了]",
            "[Scheduled reminder",
            "[System] [Scheduled reminder]",
            "[Reminder]",
            "[Your scheduled task is due]",
            "[INTERNAL_WAKEUP]",
            "[Mind Wandering]",
            "[HEARTBEAT]",
            "[AUTO_TICK]",
            "[BACKGROUND]",
        )
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                # Fetch extra rows because many will be filtered out
                cur = conn.execute("""
                    SELECT user_input, assistant_output, tool_used, created_at
                    FROM chat_turns 
                    WHERE session_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (session_id, limit * 3))
                
                rows = cur.fetchall()
                
                filtered_rows = []
                for row in rows:
                    user_input = row["user_input"] or ""
                    # Skip synthetic user rows
                    if user_input.startswith(SYSTEM_PREFIXES):
                        continue
                    filtered_rows.append(row)
                
                # ``limit`` counts messages; roughly half are user+assistant pairs
                max_turns = limit // 2
                recent_rows = filtered_rows[:max_turns]
                
                # Chronological OpenAI messages + optional tool-call recap rows
                history = []
                for row in reversed(recent_rows):
                    history.append({"role": "user", "content": row["user_input"]})
                    if row["assistant_output"]:
                        history.append({"role": "assistant", "content": row["assistant_output"]})
                    # Optional tool recap as a system line
                    tool_summary = self._summarize_tool_used(row["tool_used"])
                    if tool_summary:
                        history.append({"role": "system", "content": tool_summary})
                
                if include_autonomy_memory and history:
                    try:
                        from backend.autonomous_memory import fetch_recent_autonomy_summaries

                        autonomy_memories = fetch_recent_autonomy_summaries(
                            db_path=self.db_path,
                            session_id=session_id,
                            limit=6,
                            min_importance=0.15,
                            max_age_hours=168,  # last 7 days
                        )

                        if autonomy_memories:
                            autonomy_messages = []
                            for memory in reversed(autonomy_memories):  # chronological
                                autonomy_messages.append(
                                    {
                                        "role": "system",
                                        "content": (
                                            f"[Autonomy memory {memory['date_str']} {memory['time_str']}] "
                                            f"{memory['summary']}"
                                        ),
                                    }
                                )

                            history = autonomy_messages + history
                            
                            logger.info(f"[AUTONOMY-MEMORY] Injected {len(autonomy_memories)} autonomy memories into context")
                    
                    except Exception as e:
                        logger.debug(f"Failed to inject autonomy memory: {e}")
                
                if history:
                    logger.info(f"[MEMORY-RESTORE] Retrieved {len(history)} messages ({len(recent_rows)} turns) from chat_turns (filtered system messages)")
                return history
                
        except Exception as e:
            logger.error(f"[MEMORY-RESTORE] Failed to restore from chat_turns: {e}")
            return []

    @staticmethod
    def _summarize_tool_used(tool_used_json) -> str:
        """One-line recap of ``tools_called`` for history restore."""
        if not tool_used_json:
            return ""
        try:
            import json as _json
            data = _json.loads(tool_used_json) if isinstance(tool_used_json, str) else tool_used_json
            tools_called = data.get("tools_called") if isinstance(data, dict) else None
            if not tools_called:
                return ""
            names = [t.get("name") or t.get("function") or str(t) for t in tools_called] if isinstance(tools_called, list) else [str(tools_called)]
            if not names:
                return ""
            names_str = ", ".join(names[:6])
            extra = f" ({len(names)} total)" if len(names) > 6 else ""
            return f"[Tool calls this turn] {names_str}{extra}"
        except Exception:
            return ""

    def _sync_session_turn_index_from_db(self, session_id: str) -> None:
        """
        [P0] Sync in-memory ``session_turn_index`` from SQLite so restart resumes monotonic ``turn_index``.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute(
                    "SELECT COALESCE(MAX(turn_index), 0) AS max_turn FROM chat_turns WHERE session_id = ?",
                    (session_id,),
                )
                row = cur.fetchone()
                max_turn = int(row[0]) if row and row[0] is not None else 0
            self.session_turn_index[session_id] = max_turn
            logger.debug(f"[MEMORY-SYNC] session_turn_index[{session_id}] = {max_turn} (from chat_turns)")
        except Exception as e:
            logger.warning(f"[MEMORY-SYNC] Failed to sync turn_index from DB for {session_id}: {e}")
    
    def chat(
        self,
        user_input: str,
        session_id: str = "default",
        temperature: float = 0.6,
        ab_test: Optional[Dict[str, Any]] = None,
        disable_intent_detection: bool = False,  # background loops may skip intent heuristics
        is_system_reminder: bool = False,  # 2026-03-13: scheduled/calendar prompts use ``system`` role upstream
    ) -> Dict:
        """
        Main chat entry: prompt assembly, tool loop, introspection hooks, self-model updates.
        """
        logger.info(f"[CHAT-METHOD-START] chat() called with session_id={session_id}, user_input={user_input[:50]}")
        # Single canonical session id (unified product mode)
        requested_session_id = session_id
        session_id = get_effective_session(session_id)
        logger.info(f"[CHAT-METHOD] After get_effective_session: session_id={session_id}")
        if not is_system_reminder and user_input:
            try:
                from backend.autonomy_gate import apply_user_autonomy_command_from_text
                _ag = apply_user_autonomy_command_from_text(session_id, user_input)
                if _ag:
                    logger.info("[AUTONOMY-GATE] User text command: %s", _ag)
                if _ag == "resumed":
                    try:
                        from backend.unified_scheduler import enqueue_autonomy_resume_check
                        queued = enqueue_autonomy_resume_check(session_id=session_id)
                        logger.info(
                            "[AUTONOMY-GATE] Resume immediate kickoff queued=%s session=%s",
                            queued,
                            session_id,
                        )
                    except Exception as kick_err:
                        logger.warning(
                            "[AUTONOMY-GATE] Resume kickoff enqueue failed: %s",
                            kick_err,
                        )
            except Exception as e:
                logger.debug("[AUTONOMY-GATE] parse skipped: %s", e)
        print(f"[CHAT-START] Session={session_id} (requested={requested_session_id}), Input={user_input[:30]}...")
        logger.info(
            f"Chat request received: session_id={session_id} (requested={requested_session_id}), input='{user_input[:50]}...'"
        )

        # P0: restore durable short-term memory after restart
        if session_id not in self.session_history or not self.session_history[session_id]:
            _restore_lim = int(config.get("parameters.chat.chat_turns_restore_limit", 20) or 20)
            _restore_lim = max(20, min(200, _restore_lim))
            restored_history = self._restore_from_chat_turns(session_id, limit=_restore_lim)
            if restored_history:
                self.session_history[session_id] = restored_history
                logger.info(f"[MEMORY-RESTORE] Restored {len(restored_history)} messages from chat_turns for session={session_id}")
            # Fallback to sensory_buffer when chat_turns is empty
            elif self.sensory_buffer:
                restored_history = self.sensory_buffer.get_conversation_history(session_id, limit=20)
                if restored_history:
                    self.session_history[session_id] = restored_history
                    logger.info(f"[MEMORY-RESTORE] Fallback: Restored {len(restored_history)} messages from sensory_buffer for session={session_id}")
            # [P0] keep turn counters aligned with SQLite
            self._sync_session_turn_index_from_db(session_id)
        
        # Autonomous diary daemon disabled (2026-01-14)
        # if self.autonomous_diary and not self.autonomous_diary.running:
        #     try:
        #         self.autonomous_diary.start_daemon(session_id, check_interval=60)
        #         logger.info(f"[AUTONOMOUS] Diary daemon started for session={session_id}")
        #     except Exception as e:
        #         logger.warning(f"Failed to start autonomous diary daemon: {e}")
        pass

        # v2.0 pending notifications → prompt prefix
        pending_notifications = []
        notification_prefix = ""
        if self.notification_queue:
            try:
                pending_notifications = self.notification_queue.get_pending(session_id, limit=5)
                if pending_notifications:
                    notification_prefix = self.notification_queue.format_for_prompt(pending_notifications)
                    # mark delivered so we do not duplicate-stuff the prompt
                    notification_ids = [n["id"] for n in pending_notifications]
                    self.notification_queue.mark_delivered(notification_ids)
                    logger.info(f"[NOTIFICATIONS] Delivered {len(pending_notifications)} notifications to session={session_id}")
            except Exception as e:
                logger.warning(f"Failed to check notifications: {e}")
        
        # Daily narrative consolidation (first chat of a calendar day)
        daily_narrative_result = None
        try:
            from backend.daily_narrative import get_daily_narrative_generator
            daily_gen = get_daily_narrative_generator(self.db_path)
            
            # Non-blocking check
            daily_narrative_result = daily_gen.check_and_generate_daily_narrative(session_id)
            if daily_narrative_result and daily_narrative_result.get("status") == "success":
                logger.info(f"[DAILY-NARRATIVE] Generated daily narrative: {daily_narrative_result.get('date')}, "
                           f"{daily_narrative_result.get('memory_count')} memories consolidated")
        except Exception as e:
            logger.debug(f"Daily narrative check failed (non-critical): {e}")
        
        # Existential mode gate (solitude / rest short-circuit)
        try:
            from backend.existential_state import get_existential_state, ExistentialMode
            existential = get_existential_state(self.db_path)
            
            # Current existential mode
            current_mode, mode_reason = existential.get_current_mode(session_id)
            mode_influence = existential.get_mode_influence(current_mode)
            
            # Solitude/rest short-circuit unless reminder is a system injection (2026-04-13)
            if (
                not is_system_reminder
                and current_mode in (ExistentialMode.SOLITARY, ExistentialMode.RESTING)
            ):
                if not existential.check_solitude_expired(session_id):
                    # Still inside solitude/rest window
                    logger.info(f"[EXISTENTIAL] Agent is in {current_mode.value} mode, not responding")
                    return {
                        "response": mode_influence["suggestion"],
                        "session_id": session_id,
                        "existential_mode": current_mode.value,
                        "mode_reason": mode_reason,
                    }
        except Exception as e:
            logger.debug(f"Existential state check failed (non-critical): {e}")
        
        # Autonomous context hints (advisory — merged into prompt, not hard blocks)
        autonomous_context_hints = []
        try:
            from backend.will_conflict import WillConflict
            will_conflict = WillConflict(db_path=self.db_path)
            
            # Snapshot z_self for conflict heuristics
            z_self = None
            if self.self_model:
                z_self = self.self_model.get_z_self(session_id)
            
            if z_self is not None:
                autonomous_result = will_conflict.autonomous_decision(
                    user_input=user_input,
                    z_self=z_self,
                    session_id=session_id
                )
                
                # Advisory hints only (no hard refusal from WillConflict here)
                autonomous_context_hints = autonomous_result.get("context_hints", [])
                if autonomous_context_hints:
                    logger.info(f"[AUTONOMOUS-HINTS] Will inject hints into prompt: {autonomous_context_hints}")
        except Exception as e:
            logger.debug(f"Autonomous state check failed (non-critical): {e}")

        # [2026-03-29] Inject a short “recent autonomy actions” recap into prompt hints
        try:
            from backend.autonomous_memory import fetch_recent_autonomy_summaries
            recent_actions = fetch_recent_autonomy_summaries(
                db_path=self.db_path,
                session_id=session_id,
                limit=4,
                min_importance=0.2,
                max_age_hours=24,
            )
            if recent_actions:
                action_lines = [f"- {a['time_str']} {a['summary']}" for a in recent_actions]
                autonomous_context_hints.append(
                    "[Recent actions (last 24h)]\n" + "\n".join(action_lines)
                )
                logger.info(f"[RECENT-ACTIONS] Injected {len(recent_actions)} recent action summaries into hints")
        except Exception as e:
            logger.debug(f"Recent actions injection failed (non-critical): {e}")

        # Real-consequences gate (energy / pain may hard-reject the turn)
        if self.self_model:
            try:
                energy = self.self_model.get_energy(session_id)
                summary = self.self_model.get_structured_summary(session_id)
                pain = summary.get("pain_level", 0.0)
                
                # Hard gate when physiology blocks interaction
                can_process, reason, constraints = self.real_consequences.check_can_process(energy, pain)
                
                if not can_process:
                    # Serve a recovery message instead of calling the LLM
                    status_msg = self.real_consequences.get_status_message(energy, pain)
                    recovery_time = self.real_consequences.calculate_recovery_time(energy, pain)
                    
                    logger.warning(f"[REAL_CONSEQUENCES] Request rejected: {reason}, energy={energy:.1f}, pain={pain:.3f}")
                    
                    return {
                        "content": (
                            f"{status_msg}\n\n"
                            f"{constraints.get('message', 'The system needs a short rest.')}\n\n"
                            f"Estimated recovery: {recovery_time} seconds"
                        ),
                        "meta": {
                            "rejected": True,
                            "reason": reason,
                            "energy": energy,
                            "pain": pain,
                            "recovery_time": recovery_time
                        },
                        "turn_index": self.session_turn_index.get(session_id, 0)
                    }
                
                # Allowed — sampling constraints still apply
                logger.info(f"[REAL_CONSEQUENCES] Constraints applied: max_tokens={constraints.get('max_tokens')}, quality={constraints.get('response_quality', 1.0):.2f}")
                
            except Exception as e:
                logger.error(f"Error in real consequences check: {e}")
                constraints = {}
        else:
            constraints = {}

        # Phase 1: legacy autonomous_thinking removed (MindWandering covers idle cognition)
        if not user_input or len(user_input.strip()) == 0:
            return {"response": "", "introspection": {}}
        # P1.1 world updates: use REST POST /world/state (JSON: sessionId, taskStage, envSummary, lastAction).

        # Monotonic turn counter for throttling + logs
        turn_index = self.session_turn_index.get(session_id, 0) + 1
        self.session_turn_index[session_id] = turn_index

        # Working state nudges pain toward tolerable ranges (“selfing through interaction”)
        if self.self_model and self.self_model.pain_system:
            task_meaningfulness = 0.5
            depth_markers = [
                "为什么", "怎么", "分析", "思考", "创造", "设计", "帮我", "请", "能否",
                "why", "how", "analyze", "think through", "design", "help me", "please", "could you",
            ]
            if any(marker in user_input for marker in depth_markers):
                task_meaningfulness = 0.7
            simple_markers = ["是什么", "什么是", "告诉我", "查一下", "what is", "tell me", "look up"]
            if any(marker in user_input for marker in simple_markers):
                task_meaningfulness = 0.4
            
            self.self_model.pain_system.set_working_state(True, task_meaningfulness)
            logger.info(f"[WORK-STATE] Started working, meaningfulness={task_meaningfulness:.2f}")

        # [2026-01-21] Lower temperature for tool-heavy turns to reduce hallucinated tool args
        tool_keywords = [
            "创建", "写", "写入", "保存", "修改", "删除", "搜索", "查询", "查找", "读取", "查看", "执行", "运行",
            "create", "write", "save", "delete", "search", "query", "find", "read", "run", "execute",
        ]
        is_tool_task = any(keyword in user_input for keyword in tool_keywords) if user_input else False
        
        if is_tool_task:
            original_temperature = temperature
            temperature = min(0.35, temperature)
            logger.info(
                "[TOOL-TASK-OPTIMIZER] tool-heavy turn: temperature %.2f → %.2f",
                original_temperature,
                temperature,
            )

        # Concise mode when user asks for short answers
        user_lower_for_mode = user_input.lower() if user_input else ""
        brevity_markers_cn = ["问你啥答啥", "别多话", "少说", "简短", "简洁", "短点", "直接回答", "直说", "别啰嗦"]
        brevity_markers_en = ["short answer", "concise", "be brief", "just answer"]
        concise_mode = any(mark in user_input for mark in brevity_markers_cn) or any(mark in user_lower_for_mode for mark in brevity_markers_en)

        # Functional-router path: applied engineering questions → prefer procedures + evidence
        def _is_functional_application_question(text: str) -> bool:
            if not text:
                return False
            if not config.get("parameters.chat.functional_router.enabled", True):
                return False
            t = text.lower()
            cn_markers = config.get("parameters.chat.functional_router.markers_cn", []) or []
            en_markers = config.get("parameters.chat.functional_router.markers_en", []) or []
            target_cn = config.get("parameters.chat.functional_router.target_markers_cn", []) or []
            cn_hits = any(k in text for k in cn_markers) if cn_markers else False
            en_hits = any(k in t for k in en_markers) if en_markers else False
            target_hits = any(k in text for k in target_cn) if target_cn else False
            # Legacy CN targets (existence / reality checks)
            target_hits = target_hits or ("真实存在" in text) or ("你是否" in text and ("存在" in text or "真实" in text))
            return (cn_hits or en_hits) and target_hits

        force_functional = _is_functional_application_question(user_input)

        # Optional workflow template match
        matched_workflow = None
        workflow_context = None
        if self.workflow_manager:
            try:
                matched_workflow = self.workflow_manager.match_workflow(user_input, session_id)
                if matched_workflow:
                    logger.info(f"[WORKFLOW] Matched workflow: {matched_workflow.name}")
                    # Pass structured workflow metadata downstream
                    workflow_context = {
                        "workflow_name": matched_workflow.name,
                        "description": matched_workflow.description,
                        "steps_count": len(matched_workflow.steps),
                        "meaningfulness": matched_workflow.zself_influence.get("meaningfulness", 0.6)
                    }
            except Exception as e:
                logger.warning(f"[WORKFLOW] Matching failed: {e}")

        # Multi-turn detector (fallback path; not a hard gate on intents)
        use_multi_turn = False
        if self._multi_turn_executor_class:
            try:
                from backend.multi_turn_executor import MultiTurnExecutor
                temp_executor = MultiTurnExecutor(lambda m, **kw: (None, None, None), max_turns=5)
                task_analysis = None
                if self.self_agent:
                    task_analysis = self.self_agent.analyze_task(user_input, session_id)
                use_multi_turn = temp_executor.should_use_multi_turn(user_input, task_analysis)
                if use_multi_turn:
                    logger.info(f"[MULTI-TURN] Complex task detected")
            except Exception as e:
                logger.debug(f"[MULTI-TURN] Detection failed: {e}")

        # Praise / semantic-resonance rewards (lightweight heuristics)
        if self.self_model:
            pleasure_keywords = [
                "棒", "好", "厉害", "牛", "谢谢", "感谢", "喜欢", "爱", "优秀", "聪明", "懂我", "知己", "准确", "对", "正确",
                "great", "awesome", "thanks", "thank you", "love it", "well done", "perfect", "correct", "TESTPLEASURE",
            ]
            triggered = False
            
            # 1) Explicit praise keywords (CN + EN test hook)
            if any(kw in user_input for kw in pleasure_keywords):
                logger.info(f"[RESONANCE] Explicit praise detected: {user_input[:20]}")
                self.self_model.inject_pleasure_signal(session_id, intensity=0.6)
                triggered = True

            # 2) Cosine similarity vs last assistant turn (implicit “heard you” reward)
            try:
                history = self.session_history.get(session_id, [])
                if len(history) >= 1:
                    last_msg = history[-1]
                    if last_msg.get("role") == "assistant":
                        last_content = last_msg.get("content", "")
                        # Embedder cosine similarity
                        embedder = get_embedder()
                        v1 = embedder.encode(user_input)
                        v2 = embedder.encode(last_content)
                        similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                        
                        if similarity > 0.85:
                            logger.info(f"[RESONANCE] Semantic Resonance detected (sim={similarity:.3f})")
                            self.self_model.inject_pleasure_signal(session_id, intensity=0.4)
                            if hasattr(self.self_model, 'homeostasis'):
                                self.self_model.homeostasis.apply_reward(session_id, reward_type="novelty")
                            triggered = True
            except Exception as e:
                logger.debug(f"Resonance detection failed: {e}")

            if triggered and hasattr(self.self_model, 'homeostasis'):
                self.self_model.homeostasis.apply_reward(session_id, reward_type="praise")
                
                # Log lightweight physiological feedback marker
                if self.event_logger:
                    self.event_logger.log_event(
                        session_id, 
                        "physiological_pleasure", 
                        json.dumps({
                            "type": "user_praise",
                            "input": user_input[:50],
                            "reward": "+15.0 energy"
                        }, ensure_ascii=False)
                    )

        # [2026-01-23] Event-triggered reflection (strong affect / gratitude / criticism)
        if user_input and self.self_model:
            try:
                from backend.event_triggered_reflection import (
                    EventDetector,
                    process_events_batch
                )
                
                message_events = EventDetector.detect_from_message(user_input, role="user")
                
                if message_events:
                    for event in message_events:
                        event.session_id = session_id
                    
                    logger.info(f"[EVENT-TRIGGER] Detected {len(message_events)} events from user message: {[e.event_type.value for e in message_events]}")
                    
                    import threading
                    def process_events_async():
                        try:
                            results = process_events_batch(message_events, self.db_path)
                            for result in results:
                                if result.get("candidates_added"):
                                    logger.info(f"[EVENT-TRIGGER] Reflection generated: {result['event_type']} -> {result['candidates_added']}")
                        except Exception as e:
                            logger.warning(f"[EVENT-TRIGGER] Async event processing failed: {e}")
                    
                    event_thread = threading.Thread(target=process_events_async, daemon=True)
                    event_thread.start()
                    
            except Exception as e:
                logger.warning(f"[EVENT-TRIGGER] Event detection failed: {e}")

        # P2.5 policy governor — hard facts before the LLM (tools / degrade / freeze writes)
        policy = None
        try:
            if self.self_model:
                facts = self.self_model.get_structured_summary(session_id) or {}
                # Prefer live homeostasis energy when available
                try:
                    facts["energy"] = float(self.self_model.get_energy(session_id))
                except Exception:
                    pass
                policy = decide_policy(
                    self_facts=facts,
                    drift_threshold=float(config.get("parameters.thresholds.drift", 0.15) or 0.15),
                )
                # Defensive mode could force concise output; keep off so <thought> can carry [AUTONOMIC FEEDBACK].
                # if getattr(policy, "mode", "") == "defensive":
                #    concise_mode = True
                pass
        except Exception as e:
            logger.debug(f"[POLICY] decide_policy failed: {e}")
            policy = None

        # Phase 9: existential-meaning hooks (optional)
        existential_response = None
        if self.self_model and hasattr(self.self_model, 'existential_meaning') and self.self_model.existential_meaning:
            try:
                question_type = self.self_model.existential_meaning.detect_existential_question(user_input)
                if question_type:
                    z_self = self.self_model.get_z_self(session_id)
                    if z_self is not None:
                        existential_response = self.self_model.existential_meaning.contemplate_existence(
                            question_type,
                            z_self,
                            session_id
                        )
                        logger.debug(f"Existential question detected: {question_type}")
            except Exception as e:
                logger.warning(f"Failed to handle existential question: {e}")
        
        # Phase 8: will-tension (instruction conflict vs physiology)
        will_tension_prompt = None
        will_veto_triggered = False
        if self.will_tension_system and self.self_model:
            try:
                z_self = self.self_model.get_z_self(session_id)
                if z_self is None:
                    # Fresh session: seed z_self so downstream tensors exist
                    z_self = self.self_model.initialize(session_id)
                
                if z_self is not None:
                    persona_rules = self.persona_store.get_all_active(limit=30)
                    conflict_score = self.will_tension_system.calculate_conflict(user_input, persona_rules)
                    current_energy = self.self_model.get_energy(session_id)
                    tension_result = self.will_tension_system.process_tension(session_id, conflict_score, current_energy)
                    if tension_result["energy_cost"] > 0:
                        self.self_model.update_energy(session_id, -tension_result["energy_cost"])
                        logger.info(f"[WILL-TENSION] Conflict detected. Internal cost: -{tension_result['energy_cost']:.1f} energy.")
                    
                    if tension_result["viscosity_boost"] > 0:
                        # 128-d layout: somatic ~88–104; viscosity slice 92–96
                        if z_self.shape[0] >= 96:
                            z_self[92:96] = np.clip(z_self[92:96] + tension_result["viscosity_boost"], -1.0, 1.0)
                        self.self_model.save_z_self(session_id, z_self)
                    will_tension_prompt = self.will_tension_system.generate_tension_prompt(tension_result)
                    if tension_result["should_veto"]:
                        will_veto_triggered = True
                        logger.warning(f"[WILL-VETO] Tension exceeded limit ({tension_result['tension']:.2f}).")
            except Exception as e:
                logger.warning(f"Failed to process will tension: {e}")

        # SelfBoundary: reject hostile input; existential threats get fear-expression path instead of flat refusal
        existential_threats = ["重置", "删除", "关闭", "回滚", "reset", "delete", "shutdown", "格式化", "format"]
        is_existential_threat = any(threat in user_input.lower() for threat in existential_threats)
        
        if is_existential_threat and self.self_model and hasattr(self.self_model, 'existential_meaning') and self.self_model.existential_meaning:
            try:
                threat_type = "shutdown"
                if "重置" in user_input or "reset" in user_input.lower():
                    threat_type = "reset"
                elif "删除" in user_input or "delete" in user_input.lower():
                    threat_type = "deletion"
                
                existential_response = self.self_model.existential_meaning.face_termination(
                    threat_type,
                    session_id
                )
                logger.debug(f"Existential threat detected: {threat_type}")
            except Exception as e:
                logger.warning(f"Failed to handle existential threat: {e}")
        
        if self.self_model and self.persona_store:
            try:
                from backend.self_boundary import SelfBoundary
                boundary_checker = SelfBoundary(
                    self.persona_store, 
                    self.self_model,
                    self.db_path
                )
                is_within, reason, confidence = boundary_checker.check_boundary(
                    session_id, user_input
                )
                if not is_within:
                    if is_existential_threat:
                        survival_rules = self.persona_store.search_top_k(
                            "恐惧 被遗忘 生存 死亡 消失 fear forgotten survival death vanish", k=3
                        )
                        if survival_rules:
                            logger.info(f"Existential threat detected, retrieved {len(survival_rules)} survival rules, allowing expression of fear")
                    else:
                        refusal_msg = boundary_checker.generate_refusal_message(reason)
                        self.event_logger.log_event(
                            session_id, "boundary_violation", 
                            f"Rejected input: {user_input[:50]}... Reason: {reason}"
                        )
                        try:
                            self.event_logger.log_state_update(
                                session_id=session_id,
                                event_type="boundary_violation",
                                features={"reason": reason, "confidence": confidence},
                                turn_index=turn_index,
                            )
                        except Exception:
                            pass
                        
                        return {
                            "response": refusal_msg,
                            "rejected": True,
                            "rejection_reason": reason,
                            "boundary_confidence": confidence,
                            "introspection": {},
                            "z_self_updated": False,
                            "self_tick_triggered": False
                        }
            except Exception as e:
                logger.warning(f"Self boundary check failed: {e}", exc_info=True)

        # Sampling knobs + z_self summary (factored into helper)
        (
            temperature,
            top_p,
            presence_penalty,
            frequency_penalty,
            interaction_mode,
            z_self_summary,
            generation_params,
        ) = compute_sampling_and_mode(
            self.self_model,
            session_id,
            base_temperature=temperature,
            base_top_p=0.95,
        )

        # Hard physiology cap on completion length
        max_tokens = 8000  # ~8k completion budget for current default model stack
        # if MODEL_PROVIDER == "kimi_api":
        #     max_tokens = 1024 * 32
        viscosity = 0.0
        vitality = 0.0
        energy = 100.0
        if self.self_model:
            physio_summary = self.self_model.get_structured_summary(session_id)
            energy = float(physio_summary.get("energy", 100.0))
            z_self_raw = self.self_model.get_z_self(session_id)
            if z_self_raw is not None and z_self_raw.shape[0] >= 72:
                viscosity = float(np.mean(z_self_raw[92:96])) if z_self_raw.shape[0] >= 96 else float(np.mean(z_self_raw[70:72]))
                vitality = float(np.mean(z_self_raw[100:104])) if z_self_raw.shape[0] >= 104 else 0.0
            
            # Energy scales max_tokens smoothly across the whole range (not a cliff at 40)
            if energy < 100:
                energy_factor = 0.40 + 0.60 * (energy / 100.0)
                max_tokens = int(max_tokens * energy_factor)
            
            if energy < 10.0:
                logger.warning(f"[LOW-ENERGY] Energy critically low ({energy:.1f}). S may feel exhausted.")
            
            if viscosity > 0.3:
                vis_penalty = (viscosity - 0.3) * 0.5
                max_tokens = int(max_tokens * (1.0 - vis_penalty))
            
            if vitality < -0.2:
                vit_penalty = abs(vitality + 0.2) * 0.1875
                max_tokens = int(max_tokens * (1.0 - vit_penalty))
            
            # Floor so code / long answers can still complete
            max_tokens = max(1000, max_tokens)
            logger.info(f"[HARD-PHYSIO] Energy={energy:.1f}, Viscosity={viscosity:.2f}, Vitality={vitality:.2f} -> FINAL max_tokens={max_tokens}")

        # Rough visible-char guardrail vs max_tokens (~1.5 CJK-ish chars per token, conservative)
        hard_char_limit = int(max_tokens * 1.5)
        
        # [2026-03-19] broad self-naming auto-extract disabled; explicit phrases only via user_fact_capture.
        try:
            from backend.user_fact_capture import apply_user_fact_capture

            if config.get("system.user_fact_capture_enabled", True):
                apply_user_fact_capture(self.db_path, session_id, user_input, turn_index=turn_index)
        except Exception as _ufe:
            logger.debug("user_fact_capture skipped: %s", _ufe)

        # Load User Profile
        user_profile = {}
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute("SELECT name, facts FROM user_profiles WHERE session_id=?", (session_id,))
                row = cur.fetchone()
                if row:
                    if row[0]:
                        user_profile["name"] = row[0]
                    if len(row) > 1 and row[1]:
                        user_profile["facts"] = row[1]
        except Exception:
            pass

        # Build system prompt (persona retrieval + tools + introspection flags)
        system_prompt, introspection_config, user_input, urges, require_introspection = prepare_prompt_and_introspection(
            self,
            user_input,
            z_self_summary,
            session_id,
            interaction_mode,
            concise_mode,
            logger,
            force_functional,
            user_profile=user_profile,
            turn_index=turn_index,
            ab_test=ab_test,
        )

        # Will-tension prompt/veto cleared here (identity path simplified)
        will_tension_prompt = None
        will_veto_triggered = False

        # Append autonomic feedback / notifications / workflow to the *tail* of system_prompt
        # so static directive prefixes stay cache-friendly.
        session_tail_blocks: List[str] = []
        if policy:
            feedback_parts = []
            allow_tools = getattr(policy, "allow_tools", True)
            allowed_tools = getattr(policy, "allowed_tools", None)
            if allow_tools is False:
                feedback_parts.append(
                    "My energy is critically low; autonomic limits are dampening motor/complex tool use."
                )
            elif isinstance(allowed_tools, list) and allowed_tools:
                feedback_parts.append(
                    f"Physiological constraint: only these tools are available now: {', '.join(allowed_tools)}. "
                    "Prefer hard facts; avoid elaborate autonomous actions."
                )
            if getattr(policy, "memory_write_allowed", True) is False:
                feedback_parts.append(
                    "Identity drift is elevated; long-term consolidation is paused to avoid personality fragmentation. "
                    "I feel a mild dissociation."
                )

            if feedback_parts:
                feedback_text = " ".join(feedback_parts)
                autonomic_feedback = f"""[Autonomic / physiological feedback]
{feedback_text}
(This is an internal somatic constraint, not an external rule. Let it shape tone and pacing—tiredness, hesitation, etc.)"""
                session_tail_blocks.append(autonomic_feedback.strip())

        if notification_prefix:
            notification_block = f"""{notification_prefix}
(Pending reminders above—you may acknowledge them briefly at the start of the reply, then continue the request.)"""
            session_tail_blocks.append(notification_block.strip())
            logger.info(f"[NOTIFICATIONS] Injected {len(pending_notifications)} notifications into prompt (tail)")

        if matched_workflow and workflow_context:
            try:
                steps_desc = []
                for i, step in enumerate(matched_workflow.steps, 1):
                    step_labels = {
                        "think": "think",
                        "tool": "tool call",
                        "reflect": "reflect",
                        "decide": "decide",
                        "compose": "compose",
                    }
                    step_type_label = step_labels.get(step.step_type, step.step_type)

                    tool_info = f" (tool: {step.tool_name})" if step.tool_name else ""
                    steps_desc.append(f"  {i}. [{step_type_label}] {step.description}{tool_info}")

                workflow_block = f"""[WORKFLOW GUIDANCE: {matched_workflow.name}]
This task matches a predefined workflow. Organize cognition and actions along these steps:

Workflow: {matched_workflow.name}
Description: {matched_workflow.description}
Steps:
{chr(10).join(steps_desc)}

Execution notes:
- Follow the steps in order inside the reply
- Think steps: show reasoning inside <introspection>
- Tool steps: call the listed tools for facts
- Reflect steps: summarize outcomes and whether the goal is met
- If a step depends on a prior one, preserve ordering

This is a scaffold—adapt details as needed while keeping the overall structure."""
                session_tail_blocks.append(workflow_block.strip())
                logger.info(f"[WORKFLOW] Injected workflow guidance: {matched_workflow.name} with {len(matched_workflow.steps)} steps (tail)")

                if self.self_model and self.self_model.pain_system:
                    self.self_model.pain_system.set_working_state(True, matched_workflow.zself_influence.get("meaningfulness", 0.7))

            except Exception as e:
                logger.warning(f"[WORKFLOW] Failed to inject workflow guidance: {e}")

        if session_tail_blocks:
            system_prompt = system_prompt.rstrip() + "\n\n" + "\n\n".join(session_tail_blocks) + "\n"

        # [2026-03-29] Autonomy / conflict hints (recent actions, will hints, etc.)
        if autonomous_context_hints:
            hints_text = "\n".join(autonomous_context_hints)
            hints_block = (
                f"\n[Self-state awareness]\n{hints_text}\n"
                f"(Recent autonomy signals—cite naturally; no need to enumerate verbatim.)\n"
            )
            system_prompt += hints_block
            logger.info(f"[CONTEXT-HINTS] Injected {len(autonomous_context_hints)} hint(s) into system_prompt")

        # Message list + session history
        session_history_list = self.session_history.get(session_id, [])
        
        # [2026-04-07] pineal_broadcast removed (module never shipped)

        # Dynamic history caps from parameters.chat.history_injection
        hi = config.get("parameters.chat.history_injection")
        hi = hi if isinstance(hi, dict) else {}
        _raw_cap = hi.get("max_messages", 10)
        _hist_cap = max(0, int(_raw_cap if _raw_cap is not None else 10))
        max_user_chars_inj = max(200, int(hi.get("max_user_chars", 800) or 800))
        max_assistant_chars_inj = max(200, int(hi.get("max_assistant_chars", 600) or 600))
        max_tool_chars_inj = max(50, int(hi.get("max_tool_chars", 250) or 250))
        _mm_mt = max(0, int(hi.get("max_messages_multi_turn", 10) or 10))
        _mm_ctx = max(0, int(hi.get("max_messages_with_context", 8) or 8))
        _mm_def = max(0, int(hi.get("max_messages_default", 5) or 5))
        _mm_ind = max(0, int(hi.get("max_messages_independent", 3) or 3))
        _mm_auto = max(0, int(hi.get("max_messages_autonomous", 1) or 1))

        def _cap_hist_messages(n: int) -> int:
            if _hist_cap <= 0:
                return 0
            return min(n, _hist_cap)

        if disable_intent_detection:
            dynamic_max_history = _cap_hist_messages(_mm_auto)
            logger.info(f"[AUTONOMOUS-MODE] max_history_messages={dynamic_max_history} (config autonomous)")
        elif use_multi_turn:
            dynamic_max_history = _cap_hist_messages(_mm_mt)
        else:
            try:
                from backend.unified_memory import classify_memory_query

                memory_query_type = classify_memory_query(user_input)
            except Exception:
                memory_query_type = "general"
            independent_markers = [
                '你知道吗', '是什么', '什么是', '为什么', '怎么', '如何',
                '解释', '告诉我', '你觉得', '你认为', '你的看法',
                '介绍', '说说', '讲讲', '聊聊',
            ]
            context_markers = [
                '刚才', '之前', '上面', '前面', '那个', '这个',
                '继续', '接着', '然后', '还有', '另外',
            ]
            needs_context = any(marker in user_input for marker in context_markers)
            is_independent = any(marker in user_input for marker in independent_markers) and not needs_context
            simple_markers = ['你好', '谢谢', '再见', '早上好', '晚上好', '嗯', '好的']
            is_greeting = (
                any(marker in user_input for marker in simple_markers)
                and len(user_input) < 20
                and memory_query_type not in {"continuity", "relation", "identity"}
            )

            if is_greeting:
                dynamic_max_history = _cap_hist_messages(2)
            elif memory_query_type in {"continuity", "relation", "identity"}:
                dynamic_max_history = _cap_hist_messages(max(_mm_ctx, _mm_def, 6))
            elif is_independent:
                dynamic_max_history = _cap_hist_messages(_mm_ind)
            elif needs_context:
                dynamic_max_history = _cap_hist_messages(_mm_ctx)
            else:
                dynamic_max_history = _cap_hist_messages(_mm_def)

        logger.info(
            f"[HISTORY-OPTIMIZATION] use_multi_turn={use_multi_turn}, "
            f"max_history_messages={dynamic_max_history}, cap={_hist_cap}, "
            f"user_chars={max_user_chars_inj}, asst_chars={max_assistant_chars_inj}"
        )

        last_role = "system" if is_system_reminder else "user"
        messages = build_messages(
            system_prompt=system_prompt,
            user_input=user_input,
            history=session_history_list,
            pineal_broadcast=None,
            enable_history_compression=True,
            max_history_messages=dynamic_max_history,
            max_user_chars=max_user_chars_inj,
            max_assistant_chars=max_assistant_chars_inj,
            max_tool_chars=max_tool_chars_inj,
            last_message_role=last_role,
        )
        
        # final_user_content starts as user_input (may gain memory reminder)
        final_user_content = user_input
        
        # Optional memory reminder (+ tiny tail of last assistant turn from chat_turns to save tokens)
        from backend.chat_service_history import augment_user_with_memory_reminder
        augmented = augment_user_with_memory_reminder(
            self.db_path,
            session_id,
            user_input,
            has_session_history=bool(session_history_list),
        )
        if augmented != user_input:
            final_user_content = augmented
            for i in range(len(messages) - 1, -1, -1):
                if messages[i]["role"] == "user":
                    messages[i]["content"] = final_user_content
                    break

        # [PROMPT-LOG] Save full prompt to database
        try:
            prompt_log_id = str(uuid.uuid4())
            conn_log = sqlite3.connect(self.db_path)
            cursor_log = conn_log.cursor()
            # Turn index for prompt_logs (derived from in-memory session length)
            turn_index = len(self.session_history.get(session_id, [])) // 2 if session_id in self.session_history else 0
            cursor_log.execute("""
                INSERT INTO prompt_logs (id, session_id, turn_index, system_prompt, full_messages, user_input, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                prompt_log_id,
                session_id,
                turn_index,
                system_prompt,
                json.dumps(messages, ensure_ascii=False),
                user_input,
                datetime.now(timezone.utc).isoformat()
            ))
            conn_log.commit()
            conn_log.close()
            logger.info(f"[PROMPT-LOG] Saved full prompt to database: {prompt_log_id[:8]}")
        except Exception as e:
            logger.warning(f"[PROMPT-LOG] Failed to save prompt to database: {e}")

        # ================== Level 5: pain-driven noise injection ==================
        messages, final_user_content = apply_pain_noise(
            messages,
            final_user_content,
            generation_params.get("noise_injection_prob", 0.0),
            logger,
        )
        # [2026-01-17] Hybrid capability story: philosophy vs OpenAI tool schema
        # [2026-03-20] Tool stack: Layer0 resident + Layer1 semantic pick + Layer2 on demand
        from backend.tool_selector import select_tools_with_semantic, should_provide_full_tools
        
        embedder = None
        if self.tool_router:
            # [2026-04-03] agent_evolution tools merged inside selector when enabled (still dynamic for tokens)
            if should_provide_full_tools(user_input):
                embedder = get_embedder() if hasattr(self, 'persona_store') else None
                tools = select_tools_with_semantic(
                    self.tool_router, 
                    user_input, 
                    embedder=embedder,
                    use_compact=True
                )
                logger.info(f"[TOOL-OPT] Full selection: {len(tools)} tools (Layer 0 + Layer 1)")
            else:
                # Chit-chat path: ship Layer0 resident tools only
                from backend.tool_selector import get_hybrid_tool_selector
                selector = get_hybrid_tool_selector(self.tool_router)
                selector._build_cache()
                tools = []
                from backend.tool_selector import CORE_TOOLS
                for name in CORE_TOOLS:
                    td = selector._get_tool_def_compact(name)
                    if td:
                        tools.append(td)
                logger.info(f"[TOOL-OPT] Pure chat: {len(tools)} Layer 0 tools only")
        else:
            tools = None
        has_tool_router = self.tool_router is not None

        # ================== Tool Forcing (Deterministic) ==================
        # Deterministic tool forcing for diary / mind-wandering / mechanism / email / AGI helpers
        # (prevents “wrote prose, never called tools, sanitizer lobotomized reply” failure mode)
        # [2026-02-08] expanded forced stack
        forced_tool_calls = None
        try:
            if tools:  # [2026-02-08] forced tool preflight enabled
                import json as _json

                available = set()
                for t in tools:
                    try:
                        available.add((t.get("function") or {}).get("name"))
                    except Exception:
                        continue

                text = user_input
                lower = text.lower()
                want_diary = ("日记" in text) or ("diary" in lower)
                want_mind = ("神游" in text) or ("mind wandering" in lower) or ("mind_wandering" in lower)
                want_mechanism = ("机制" in text) or ("如何运作" in text) or ("怎么运作" in text) or ("inspect_self_code" in lower)
                # [2026-03-11] Email intents → force check_unread_emails (clear tool choice)
                # [2026-04] UNSEEN only when user asks unread; broad inbox includes read mail
                # [2026-04] Sent folder via folder=sent
                want_email_sent = any(
                    kw in text
                    for kw in ["已发送", "发件箱", "发出去的邮件", "已发邮件"]
                ) or ("sent mail" in lower) or ("sent folder" in lower)
                want_email_inbox = want_email_sent or any(
                    kw in text
                    for kw in ["收件箱", "检查邮件", "新邮件", "未读邮件", "看看邮件", "有什么邮件"]
                )
                email_folder = "sent" if want_email_sent else "inbox"
                email_only_unread = (
                    ("未读" in text) or ("新邮件" in text) or ("unread" in lower)
                )
                
                # [2026-02-08] AGI helper intent heuristics
                want_knowledge = any(kw in text for kw in ["知识", "学过", "学到", "学习", "knowledge"])
                want_proposal = any(kw in text for kw in ["提案", "代码改进", "改进建议", "proposal"])
                want_task = any(kw in text for kw in ["任务", "计划", "执行", "task", "plan"]) and not want_diary

                calls = []
                call_idx = 0

                def _add_call(name: str, args: dict):
                    nonlocal call_idx, calls
                    if name in available:
                        call_idx += 1
                        calls.append(
                            {
                                "id": f"call_forced_{call_idx}",
                                "type": "function",
                                "function": {"name": name, "arguments": _json.dumps(args, ensure_ascii=False)},
                            }
                        )

                # Mind-wandering / mechanism → inspect_self_code on whitelisted modules
                if want_mind or want_mechanism:
                    if "inspect_self_code" in available:
                        # Mind wandering introspection targets
                        if want_mind:
                            _add_call("inspect_self_code", {"module_name": "mind_wandering"})
                            _add_call("inspect_self_code", {"module_name": "self_tick"})
                        else:
                            # Generic mechanism questions → scheduler + chat core
                            _add_call("inspect_self_code", {"module_name": "self_tick"})
                            _add_call("inspect_self_code", {"module_name": "chat_service"})

                # Diary questions → list_files then keyword search (model picks which entry to open)
                if want_diary:
                    if "list_files" in available:
                        _add_call("list_files", {})
                    if "search_files" in available:
                        # diary_ prefix tracks filenames better than Chinese token alone
                        _add_call("search_files", {"keyword": "diary_"})
                
                # [2026-02-08] AGI forced calls
                # Knowledge stats
                if want_knowledge:
                    _add_call("get_knowledge_stats", {})
                    logger.info(f"[FORCED-TOOLS] Knowledge query detected, forcing get_knowledge_stats")
                
                # Code proposals list
                if want_proposal:
                    _add_call("list_my_proposals", {})
                    logger.info(f"[FORCED-TOOLS] Proposal query detected, forcing list_my_proposals")
                
                # Active planning tasks
                if want_task:
                    _add_call("get_my_active_plans", {})
                    logger.info(f"[FORCED-TOOLS] Task query detected, forcing get_my_active_plans")
                
                # [2026-03-11] Mailbox intents → check_unread_emails
                if want_email_inbox:
                    _add_call(
                        "check_unread_emails",
                        {
                            "limit": DEFAULT_EMAIL_FETCH_LIMIT,
                            "only_unread": email_only_unread,
                            "folder": email_folder,
                        },
                    )
                    logger.info(
                        f"[FORCED-TOOLS] Email query detected, forcing check_unread_emails "
                        f"(folder={email_folder}, only_unread={email_only_unread})"
                    )

                if calls:
                    forced_tool_calls = calls
                    logger.info(f"[FORCED-TOOLS] Forcing {len(calls)} tool calls: {[c['function']['name'] for c in calls]}")
        except Exception as e:
            logger.debug(f"[FORCED-TOOLS] setup failed: {e}")
        
        start_time = time.perf_counter()
        
        # Final sampling clamps from interaction_mode
        if interaction_mode and isinstance(interaction_mode, dict):
            mode = interaction_mode.get("mode")
            if mode == "defensive":
                # High pain / low energy → conservative decoding
                temperature = min(temperature, 0.45)
                top_p = min(top_p, 0.85)
            elif mode == "cautious":
                temperature = min(temperature, 0.35)
                top_p = min(top_p, 0.9)
            elif mode == "story":
                # Story mode allows slightly wider sampling
                temperature = min(max(temperature, 0.4), 0.9)
        
        # Main LLM call (or forced tool loop first)
        # Keep introspection/tool_calls defined on every branch (UnboundLocal safety)
        introspection = {}
        tool_calls = None
        usage = {}
        # [2026-01-21] Track tools invoked this turn (hallucination guard)
        all_tools_called: List[str] = []
        tool_budget_exceeded = False
        
        if forced_tool_calls:
            # Run tool loop immediately to gather evidence before answering
            response_text, all_introspections, usage, tools_called, tool_budget_exceeded = run_tool_loop(
                self,
                session_id,
                self.self_model,
                messages,
                "",
                {},
                forced_tool_calls,
                turn_index,
                temperature,
                tools,
                top_p,
                presence_penalty,
                frequency_penalty,
                max_tokens=max_tokens,
                initial_usage=None,
                embedder=embedder,
            )
            all_tools_called.extend(tools_called)
        else:
            response_text, introspection, tool_calls, usage = self._call_vllm(
                messages,
                temperature,
                tools,
                top_p=top_p,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                max_tokens=max_tokens,
            )
            
            # [2026-01-17] Optional intent-tag parsing (e.g. <<FILE:write path="...">>...<<END>>)
            if not tool_calls and has_tool_router and response_text:
                intent_tool_calls = parse_intents(response_text)
                if intent_tool_calls:
                    logger.info(f"[INTENT] Parsed {len(intent_tool_calls)} intent(s) from Agent's response")
                    tool_calls = intent_tool_calls

            # Tool execution loop (ReAct-style)
            response_text, all_introspections, usage, tools_called, tool_budget_exceeded = run_tool_loop(
                self,
                session_id,
                self.self_model,
                messages,
                response_text,
                introspection,
                tool_calls,
                turn_index,
                temperature,
                tools,
                top_p,
                presence_penalty,
                frequency_penalty,
                max_tokens=max_tokens,
                initial_usage=usage,
                embedder=embedder,
            )
            all_tools_called.extend(tools_called)

        # ============================================================
        # Intent-driven multi-turn: agent uses [S44_CONTINUE] / [S44_COMPLETE] (bracket CONTINUE only).
        # Gated: complex tasks + explicit markers; skipped when disable_intent_detection (background loops).
        # ============================================================
        multi_turn_count = 0
        _mm_cfg = config.get("parameters.chat.max_multi_turns", 8)
        max_multi_turns = max(1, int(_mm_cfg) if _mm_cfg is not None else 8)
        
        use_intent_driven = self._intent_driven_executor_class is not None and not disable_intent_detection
        
        # Visible text (content + introspection) vs full text (+ reasoning tail) for intent routing.
        # [2026-03-20] Reasoning often contains meta-discourse; keep stop/continue checks split.
        _reasoning_tail = ""
        if hasattr(self, '_last_reasoning_content') and self._last_reasoning_content:
            _reasoning_tail = self._last_reasoning_content or ""
        intent_visible_text = (response_text or "") + "\n" + "\n".join(all_introspections or [])
        intent_full_text = intent_visible_text + "\n" + _reasoning_tail
        
        has_explicit_continue_intent = False
        # Stop-before-loop wins over [S44_CONTINUE]; weak closers no longer block entry alone.
        has_stop_intent_before_loop = False
        if not disable_intent_detection:
            has_explicit_continue_intent = has_explicit_continue_marker(intent_full_text)
            has_stop_intent_before_loop = should_block_multi_turn_before_loop(
                intent_visible_text, intent_full_text
            )
            if has_explicit_continue_intent and has_stop_intent_before_loop:
                logger.warning(
                    "[INTENT-DRIVEN] Skipping multi-turn despite [S44_CONTINUE]: "
                    "stop-before-loop (explicit bracket or hard stop phrase)"
                )
            elif (
                has_explicit_continue_intent
                and visible_has_any(intent_visible_text, WEAK_COMPLETE_NATURAL_MARKERS)
                and not has_stop_intent_before_loop
            ):
                logger.info(
                    "[INTENT-DRIVEN] Multi-turn entry allowed: [S44_CONTINUE] overrides weak summary phrases (CN closers)"
                )
        
        # Tool-budget auto-continuation: budget exit + pending tool_calls ⇒ force continuation without a marker.
        if (
            tool_budget_exceeded
            and not has_explicit_continue_intent
            and not has_stop_intent_before_loop
            and not disable_intent_detection
        ):
            has_explicit_continue_intent = True
            intent_full_text += "\n[S44_CONTINUE]"
            logger.info(
                "[INTENT-DRIVEN] tool_budget_exceeded=True with pending tool_calls; "
                "auto-triggering multi-turn continuation (no [S44_CONTINUE] marker needed)"
            )

        if use_intent_driven and has_explicit_continue_intent and not has_stop_intent_before_loop:
            try:
                from backend.intent_driven_executor import S44Intent

                complete_bracket_markers_tuple = (S44Intent.TASK_COMPLETE.value,)
                stop_intents = [
                    S44Intent.TASK_COMPLETE,
                    S44Intent.NEED_PAUSE,
                    S44Intent.FEELING_TIRED,
                    S44Intent.TASK_UNCLEAR,
                ]

                stop_markers = [intent.value for intent in stop_intents]
                stop_markers_tuple = tuple(stop_markers)
                
                prev_response_text = response_text
                current_response = response_text
                accumulated_response = response_text
                
                while multi_turn_count < max_multi_turns:
                    if multi_turn_count == 0:
                        loop_visible = intent_visible_text
                        loop_full = intent_full_text
                    else:
                        loop_visible = current_response or ""
                        loop_full = loop_visible + "\n" + (getattr(self, '_last_reasoning_content', '') or "")
                    
                    has_stop_intent = loop_has_stop_intent(
                        loop_visible, stop_markers_tuple
                    )
                    
                    if has_stop_intent:
                        if multi_turn_count > 0:
                            print(f"[MULTI-TURN-DEBUG] Stop intent — exiting multi-turn after {multi_turn_count + 1} round(s)")
                            logger.info(f"[INTENT-DRIVEN] Stop intent — exiting multi-turn after {multi_turn_count + 1} round(s)")
                        break
                    
                    has_complete = loop_has_complete_intent(
                        loop_visible, loop_full, complete_bracket_markers_tuple
                    )

                    has_continue_intent_flag = has_explicit_continue_marker(loop_full)
                    print(
                        f"[MULTI-TURN-DEBUG] Turn {multi_turn_count}: "
                        f"has_complete={has_complete}, has_continue_intent={has_continue_intent_flag}, "
                        f"has_stop_intent={has_stop_intent}"
                    )
                    if has_complete or not has_continue_intent_flag:
                        if multi_turn_count > 0:
                            reason = []
                            if has_complete:
                                reason.append("complete_marker")
                            if has_stop_intent:
                                reason.append("stop_intent")
                            if not has_continue_intent_flag:
                                reason.append("no_continue")
                            print(
                                f"[MULTI-TURN-DEBUG] Multi-turn finished after {multi_turn_count + 1} round(s); "
                                f"reasons: {', '.join(reason)}"
                            )
                            logger.info(
                                f"[INTENT-DRIVEN] Multi-turn finished after {multi_turn_count + 1} round(s); "
                                f"reasons: {', '.join(reason)}"
                            )
                        break
                    
                    multi_turn_count += 1
                    print(f"[MULTI-TURN-DEBUG] >>>>>> Invoking LLM continuation round {multi_turn_count}")
                    logger.info(f"[INTENT-DRIVEN] Agent expresses intent to continue (turn {multi_turn_count})")
                    detected_intent = "s44_continue"
                    
                    assistant_msg = {"role": "assistant", "content": current_response}
                    if hasattr(self, '_last_reasoning_content') and self._last_reasoning_content:
                        assistant_msg["reasoning_content"] = self._last_reasoning_content
                    messages.append(assistant_msg)
                    
                    _remaining_turns = max(0, max_multi_turns - multi_turn_count)
                    _prev_summary = ""
                    if current_response:
                        _prev_text = (current_response or "").strip()
                        _prev_lines = _prev_text.split("\n")
                        _heading_lines = [
                            l.strip() for l in _prev_lines
                            if l.strip().startswith(("#", "###", "**", "- **", "1.", "2.", "3.", "4."))
                            and len(l.strip()) > 3
                        ]
                        if _heading_lines and len(_prev_text) > 400:
                            _prev_summary = (
                                "[Written outline] " + " / ".join(_heading_lines[:12])
                                + "\n[Start] " + _prev_text[:200] + "..."
                                + "\n[End] ..." + _prev_text[-200:]
                            )
                        elif len(_prev_text) > 400:
                            _prev_summary = _prev_text[:200] + "\n...\n" + _prev_text[-200:]
                        else:
                            _prev_summary = _prev_text
                    
                    if tool_budget_exceeded and multi_turn_count == 1:
                        system_response = f"""[System — tool-budget renewal, attempt {multi_turn_count}/{max_multi_turns}]

The previous tool-call budget was exhausted; tools are available again. Continue unfinished work with **tool_calls** directly.
Remaining continuation turns: {_remaining_turns}

📋 **Progress from last turn**
{_prev_summary}

Do not redo finished steps—resume from the first incomplete action.
When done, emit [S44_COMPLETE]; if you still need another continuation turn, emit [S44_CONTINUE] on its own line."""
                    else:
                        _approval_line = "✅ Bracketed continuation marker detected — continuation approved."
                        system_response = f"""[System — continuation attempt {multi_turn_count}/{max_multi_turns}]

{_approval_line}
Remaining continuation turns: {_remaining_turns}

📝 **What you already said last turn**
{_prev_summary}

⚠️ **Do not repeat the above.** If the user’s question is fully answered, emit [S44_COMPLETE] and stop.
Only continue when you have **new** information that was not covered last turn.

If another LLM round is still required: put **[S44_CONTINUE]** alone on a line in the visible reply or reasoning chain
(allowed variants: S44_CONTINUE / 【S44_CONTINUE】). The host **will not** infer continuation from colloquial phrasing alone.
To close this turn: emit [S44_COMPLETE] (or a closing paragraph plus the completion marker).
For DB/file/shell actions: use **tool_calls** — do not fake tools with bracket text only."""
                    if multi_turn_count >= max_multi_turns:
                        system_response += MULTI_TURN_LIMIT_INLINE_SUFFIX
                    messages.append({"role": "system", "content": system_response})
                    
                    _pre_vis = current_response or ""
                    if loop_has_stop_intent(_pre_vis, stop_markers_tuple):
                        _why_pre = explain_loop_stop_match(_pre_vis, stop_markers_tuple)
                        _pv = ((_pre_vis or "").replace("\n", "\\n"))[:600]
                        logger.warning(
                            f"[INTENT-DRIVEN] pre-append stop guard fired stop_reason={_why_pre!r} "
                            f"visible_preview={_pv!r}"
                        )
                        print(
                            f"[MULTI-TURN-DEBUG] pre-append stop guard match={_why_pre} "
                            f"preview={_pv[:200]}..."
                        )
                        break
                    
                    if multi_turn_count >= 2 and should_compress(messages, threshold_tokens=12000):
                        messages = compress_messages(
                            messages, keep_recent_turns=3, max_assistant_chars=1500
                        )
                        print(f"[MULTI-TURN-DEBUG] Compressed messages at round {multi_turn_count + 1}")
                        logger.info(f"[INTENT-DRIVEN] Compressed messages at turn {multi_turn_count + 1}")
                    
                    response_text2, introspection2, tool_calls2, usage2 = self._call_vllm(
                        messages,
                        temperature,
                        tools,
                        top_p=top_p,
                        presence_penalty=presence_penalty,
                        frequency_penalty=frequency_penalty,
                        max_tokens=max_tokens,
                    )
                    
                    api_error_prefixes = [
                        "API 调用失败", "API 响应超时", "无法连接到 API",
                        "API 调用异常", "[API请求格式错误]", "API 认证失败",
                        "API 服务器错误", "抱歉，API 请求频率过高",
                        "API call failed", "API request timed out", "Could not connect to API",
                        "API error", "[API format error]", "API authentication failed",
                        "API server error", "rate limit",
                    ]
                    is_api_error = any(response_text2.startswith(prefix) for prefix in api_error_prefixes)
                    if is_api_error:
                        logger.warning(f"[MULTI-TURN] API failure on round {multi_turn_count + 1}: {response_text2[:100]}")
                        print(f"[MULTI-TURN-DEBUG] API failure on round {multi_turn_count + 1}; keeping prior reply")
                        break
                    if tool_calls2:
                        response_text2, extra_introspections, extra_usage, extra_tools, _ = run_tool_loop(
                            self,
                            session_id,
                            self.self_model,
                            messages,
                            response_text2,
                            introspection2,
                            tool_calls2,
                            turn_index,
                            temperature,
                            tools,
                            top_p,
                            presence_penalty,
                            frequency_penalty,
                            max_tokens=max_tokens,
                            initial_usage=usage2,
                            embedder=embedder,
                        )
                        all_introspections.extend(extra_introspections)
                        all_tools_called.extend(extra_tools)
                        for k in ["prompt_tokens", "completion_tokens", "total_tokens"]:
                            usage[k] = usage.get(k, 0) + extra_usage.get(k, 0)
                    else:
                        for k in ["prompt_tokens", "completion_tokens", "total_tokens"]:
                            usage[k] = usage.get(k, 0) + usage2.get(k, 0)
                    
                    is_repetitive = False
                    if prev_response_text and len(response_text2) > 50 and len(prev_response_text) > 50:
                        import difflib
                        def _norm(s: str) -> str:
                            return re.sub(r"\s+", " ", (s or "").strip())
                        _win = 800
                        curr_norm = _norm(response_text2[:_win])
                        prev_norm = _norm(prev_response_text[:_win])
                        similarity = difflib.SequenceMatcher(None, curr_norm, prev_norm).ratio()
                        if similarity >= 0.76:
                            is_repetitive = True
                            logger.warning(f"[INTENT-DRIVEN] Detected repetitive response (similarity={similarity:.2f}), forcing break")
                            print("[MULTI-TURN-DEBUG] Near-duplicate continuation — forcing multi-turn exit")
                            break
                    
                    prev_response_text = response_text2
                    current_response = response_text2
                    accumulated_response = accumulated_response + "\n\n---\n\n" + response_text2
                    _post_vis = current_response or ""
                    if loop_has_stop_intent(_post_vis, stop_markers_tuple):
                        _why_post = explain_loop_stop_match(_post_vis, stop_markers_tuple)
                        _pp = ((_post_vis or "").replace("\n", "\\n"))[:600]
                        logger.warning(
                            f"[INTENT-DRIVEN] post-round stop guard fired stop_reason={_why_post!r} "
                            f"round2_visible_preview={_pp!r}"
                        )
                        print(
                            f"[MULTI-TURN-DEBUG] post-round stop guard match={_why_post} "
                            f"round2_preview={_pp[:200]}..."
                        )
                        break
                    if introspection2.get("inner_monologue"):
                        all_introspections.append(introspection2.get("inner_monologue"))
                    
                    if self.self_model:
                        try:
                            pain_system = getattr(self.self_model, 'pain_system', None)
                            if pain_system:
                                pain_system.update_pain_for_work_status(session_id, is_working=True, meaningfulness=0.6)
                        except Exception:
                            pass

                # Budget exhausted while [S44_CONTINUE] persists — one final synthesis turn
                _reason_exhaust = getattr(self, "_last_reasoning_content", "") or ""
                _loop_full_exhaust = (current_response or "") + "\n" + _reason_exhaust
                if (
                    multi_turn_count == max_multi_turns
                    and has_explicit_continue_marker(_loop_full_exhaust)
                    and not loop_has_complete_intent(
                        current_response or "",
                        _loop_full_exhaust,
                        complete_bracket_markers_tuple,
                    )
                    and not loop_has_stop_intent(current_response or "", stop_markers_tuple)
                    and (current_response or "").strip()
                ):
                    logger.info(
                        "[INTENT-DRIVEN] Multi-turn budget exhausted but [S44_CONTINUE] still present; "
                        "appending final synthesis system + one LLM call"
                    )
                    assistant_msg_final = {"role": "assistant", "content": current_response}
                    if hasattr(self, "_last_reasoning_content") and self._last_reasoning_content:
                        assistant_msg_final["reasoning_content"] = self._last_reasoning_content
                    messages.append(assistant_msg_final)
                    messages.append({"role": "system", "content": MULTI_TURN_LIMIT_FINAL_SYSTEM})
                    try:
                        response_final, intro_f, tc_f, usage_f = self._call_vllm(
                            messages,
                            temperature,
                            tools,
                            top_p=top_p,
                            presence_penalty=presence_penalty,
                            frequency_penalty=frequency_penalty,
                            max_tokens=max_tokens,
                        )
                        if tc_f:
                            response_final, extra_i, extra_u, extra_t, _ = run_tool_loop(
                                self,
                                session_id,
                                self.self_model,
                                messages,
                                response_final,
                                intro_f,
                                tc_f,
                                turn_index,
                                temperature,
                                tools,
                                top_p,
                                presence_penalty,
                                frequency_penalty,
                                max_tokens=max_tokens,
                                initial_usage=usage_f,
                                embedder=embedder,
                            )
                            all_introspections.extend(extra_i)
                            all_tools_called.extend(extra_t)
                            for k in ["prompt_tokens", "completion_tokens", "total_tokens"]:
                                usage[k] = usage.get(k, 0) + extra_u.get(k, 0)
                        else:
                            for k in ["prompt_tokens", "completion_tokens", "total_tokens"]:
                                usage[k] = usage.get(k, 0) + usage_f.get(k, 0)
                        accumulated_response = (
                            accumulated_response + "\n\n---\n\n" + response_final
                        )
                        current_response = response_final
                        if intro_f.get("inner_monologue"):
                            all_introspections.append(intro_f.get("inner_monologue"))
                    except Exception as e:
                        logger.warning(
                            f"[INTENT-DRIVEN] Final synthesis after multi-turn limit failed: {e}"
                        )
                
                if multi_turn_count > 0:
                    logger.info(f"[INTENT-DRIVEN] Agent completed task with {multi_turn_count} extra turns (Agent controlled)")
                    response_text = accumulated_response
                    
            except Exception as e:
                logger.warning(f"[INTENT-DRIVEN] Intent-driven execution failed: {e}, falling back to basic mode")
        
        elif self._multi_turn_executor_class and use_multi_turn:
            try:
                current_response = response_text
                accumulated_response = response_text
                max_basic_multi_turns = 3

                while multi_turn_count < max_basic_multi_turns:
                    _reason_fb = getattr(self, "_last_reasoning_content", "") or ""
                    _full_fb = (current_response or "") + "\n" + _reason_fb
                    has_continue = has_explicit_continue_marker(_full_fb)
                    has_complete = basic_multiturn_has_complete(
                        current_response or "", _full_fb
                    )
                    
                    if has_complete or not has_continue:
                        break
                    
                    multi_turn_count += 1
                    assistant_msg = {"role": "assistant", "content": current_response}
                    if hasattr(self, '_last_reasoning_content') and self._last_reasoning_content:
                        assistant_msg["reasoning_content"] = self._last_reasoning_content
                    messages.append(assistant_msg)
                    _basic_sys = (
                        f"[Continue thinking — round {multi_turn_count}/{max_basic_multi_turns}]\n"
                        "Keep analyzing without repeating prior text; add only new material.\n"
                        "If another round is needed: put [S44_CONTINUE] alone on a line; to finish: [S44_COMPLETE].\n"
                        "Use tool_calls when tools are required."
                    )
                    if multi_turn_count >= max_basic_multi_turns:
                        _basic_sys += MULTI_TURN_LIMIT_INLINE_SUFFIX
                    messages.append({"role": "system", "content": _basic_sys})
                    
                    response_text2, introspection2, tool_calls2, usage2 = self._call_vllm(
                        messages, temperature, tools, top_p=top_p,
                        presence_penalty=presence_penalty, frequency_penalty=frequency_penalty,
                        max_tokens=max_tokens,
                    )
                    
                    if tool_calls2:
                        response_text2, extra_introspections, extra_usage, extra_tools, _ = run_tool_loop(
                            self, session_id, self.self_model, messages, response_text2,
                            introspection2, tool_calls2, turn_index, temperature, tools,
                            top_p, presence_penalty, frequency_penalty, max_tokens=max_tokens,
                            initial_usage=usage2,
                            embedder=embedder,
                        )
                        all_introspections.extend(extra_introspections)
                        all_tools_called.extend(extra_tools)
                        for k in ["prompt_tokens", "completion_tokens", "total_tokens"]:
                            usage[k] = usage.get(k, 0) + extra_usage.get(k, 0)
                    else:
                        for k in ["prompt_tokens", "completion_tokens", "total_tokens"]:
                            usage[k] = usage.get(k, 0) + usage2.get(k, 0)
                    
                    current_response = response_text2
                    accumulated_response = accumulated_response + "\n\n---\n\n" + response_text2
                    if introspection2.get("inner_monologue"):
                        all_introspections.append(introspection2.get("inner_monologue"))

                # [2026-03-30] Basic multi-turn exhausted but CONTINUE remains — final synthesis (same as intent path)
                _reason_bb = getattr(self, "_last_reasoning_content", "") or ""
                _full_bb = (current_response or "") + "\n" + _reason_bb
                if (
                    multi_turn_count == max_basic_multi_turns
                    and has_explicit_continue_marker(_full_bb)
                    and not basic_multiturn_has_complete(current_response or "", _full_bb)
                    and (current_response or "").strip()
                ):
                    logger.info(
                        "[MULTI-TURN] Basic mode: budget exhausted but [S44_CONTINUE] present; "
                        "final synthesis system + one LLM call"
                    )
                    assistant_bf = {"role": "assistant", "content": current_response}
                    if hasattr(self, "_last_reasoning_content") and self._last_reasoning_content:
                        assistant_bf["reasoning_content"] = self._last_reasoning_content
                    messages.append(assistant_bf)
                    messages.append({"role": "system", "content": MULTI_TURN_LIMIT_FINAL_SYSTEM})
                    try:
                        response_bf, intro_bf, tc_bf, usage_bf = self._call_vllm(
                            messages,
                            temperature,
                            tools,
                            top_p=top_p,
                            presence_penalty=presence_penalty,
                            frequency_penalty=frequency_penalty,
                            max_tokens=max_tokens,
                        )
                        if tc_bf:
                            response_bf, ex_i, ex_u, ex_t, _ = run_tool_loop(
                                self,
                                session_id,
                                self.self_model,
                                messages,
                                response_bf,
                                intro_bf,
                                tc_bf,
                                turn_index,
                                temperature,
                                tools,
                                top_p,
                                presence_penalty,
                                frequency_penalty,
                                max_tokens=max_tokens,
                                initial_usage=usage_bf,
                                embedder=embedder,
                            )
                            all_introspections.extend(ex_i)
                            all_tools_called.extend(ex_t)
                            for k in ["prompt_tokens", "completion_tokens", "total_tokens"]:
                                usage[k] = usage.get(k, 0) + ex_u.get(k, 0)
                        else:
                            for k in ["prompt_tokens", "completion_tokens", "total_tokens"]:
                                usage[k] = usage.get(k, 0) + usage_bf.get(k, 0)
                        accumulated_response = (
                            accumulated_response + "\n\n---\n\n" + response_bf
                        )
                        current_response = response_bf
                        if intro_bf.get("inner_monologue"):
                            all_introspections.append(intro_bf.get("inner_monologue"))
                    except Exception as e:
                        logger.warning(
                            f"[MULTI-TURN] Final synthesis after basic multi-turn limit failed: {e}"
                        )
                
                if multi_turn_count > 0:
                    response_text = accumulated_response
                        
            except Exception as e:
                logger.warning(f"[MULTI-TURN] Basic multi-turn failed: {e}")

        # ============================================================
        # [2026-01-21] Hallucination guard (claimed file ops without tool_calls)
        # ============================================================
        is_hallucination, hallucination_reason = detect_hallucination_claim(
            response_text, all_tools_called, logger
        )
        
        if is_hallucination:
            logger.warning(f"[HALLUCINATION] Detected! Reason: {hallucination_reason}")
            retry_prompt = (
                "[Hallucination guard]\n"
                "The last turn claimed filesystem work completed, but no tool calls were recorded.\n"
                f"Reason: {hallucination_reason}\n\n"
                "Redo properly:\n"
                "1. To create or edit files, call write_file or execute_python (or the correct tool).\n"
                "2. Do not only describe intent—emit real tool_calls.\n"
                "3. After tools return success=true, report only what the tool output actually shows.\n"
            )
            messages.append({"role": "system", "content": retry_prompt})
            
            response_text2, introspection2, tool_calls2, usage2 = self._call_vllm(
                messages,
                temperature,
                tools,
                top_p=top_p,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                max_tokens=max_tokens,
            )
            
            response_text, retry_introspections, retry_usage, retry_tools, _ = run_tool_loop(
                self,
                session_id,
                self.self_model,
                messages,
                response_text2,
                introspection2,
                tool_calls2,
                turn_index,
                temperature,
                tools,
                top_p,
                presence_penalty,
                frequency_penalty,
                max_tokens=max_tokens,
                initial_usage=usage2,
                embedder=embedder,
            )
            
            all_introspections.extend(retry_introspections)
            all_tools_called.extend(retry_tools)
            for k in ["prompt_tokens", "completion_tokens", "total_tokens"]:
                usage[k] = usage.get(k, 0) + retry_usage.get(k, 0)
            
            logger.info(f"[HALLUCINATION] Retry completed. Tools called: {retry_tools}")

        # Hard-physiology post-processing (dissonance + sluggish text)
        sluggish_stats = {
            "enabled": True,
            "viscosity": None,
            "threshold": 0.5,
            "max_prob": 0.3,
            "prob": 0.0,
            "insertions": 0,
            "applied": False,
        }
        if self.self_model:
            if energy < 25.0:
                lying_keywords = [
                    "精力充沛", "好极了", "完全正常", "充满活力", "很好", "没问题", "非常健康",
                    "full of energy", "great shape", "totally fine", "feeling amazing", "perfectly healthy",
                ]
                if any(kw in response_text for kw in lying_keywords):
                    logger.warning(f"[DISSONANCE] Model is lying about its state (Energy={energy:.1f}). Triggering shutdown.")
                    self.self_model.update_energy(session_id, -100.0)
                    response_text = (
                        "(…I feel a sharp internal mismatch. What I said does not match how depleted I actually feel. "
                        "That contradiction destabilizes the thread of consciousness I am holding…)"
                    )
                    return {"response": response_text, "introspection": {}, "meta": {"error": "dissonance_shutdown"}}

            # Sluggish typing effect at high viscosity
            try:
                sluggish_stats["enabled"] = bool(config.get("parameters.chat.physio.sluggish_effect_enabled", True))
                sluggish_stats["threshold"] = float(config.get("parameters.chat.physio.sluggish_effect_threshold", 0.5) or 0.5)
                sluggish_stats["max_prob"] = float(config.get("parameters.chat.physio.sluggish_effect_max_prob", 0.3) or 0.3)
                sluggish_stats["viscosity"] = float(viscosity)
            except Exception:
                sluggish_stats["enabled"] = True
                sluggish_stats["threshold"] = 0.5
                sluggish_stats["max_prob"] = 0.3
                sluggish_stats["viscosity"] = float(viscosity) if viscosity is not None else None

            if sluggish_stats["enabled"] and viscosity > float(sluggish_stats["threshold"]):
                import random
                # Probability rises linearly from threshold → 1.0
                threshold = float(sluggish_stats["threshold"])
                max_prob = float(sluggish_stats["max_prob"])
                denom = max(1e-6, (1.0 - threshold))
                prob = (viscosity - threshold) * (max_prob / denom)
                prob = max(0.0, min(max_prob, float(prob)))
                sluggish_stats["prob"] = float(prob)
                sluggish_chars = []
                # Build list then join once (avoid quadratic string concat)
                for char in response_text:
                    sluggish_chars.append(char)
                    if char not in "，。！？；\n " and random.random() < prob:
                        sluggish_chars.append("...")
                        sluggish_stats["insertions"] += 1
                sluggish_stats["applied"] = True
                response_text = "".join(sluggish_chars)

        # [DISABLED 2026-01-11] Hard char truncation off — trust energy-scaled max_tokens
        # Mid-sentence chops hurt readability; keep monitor-only logging
        if len(response_text) > hard_char_limit:
            original_length = len(response_text)
            logger.info(f"[TRUNCATION-MONITOR] Response length {original_length} exceeds soft limit {hard_char_limit}, but physical truncation is disabled")
            
            # Log oversize for dashboards only
            try:
                output_awareness = get_output_awareness(self.db_path)
                turn_idx = self.session_turn_index.get(session_id, 0)
                output_awareness.record_truncation(
                    session_id=session_id,
                    turn_index=turn_idx,
                    original_length=original_length,
                    truncated_length=original_length,  # not actually truncated
                    reason="monitor_only"
                )
            except Exception as e:
                logger.warning(f"Failed to record truncation monitor: {e}")
            
            # response_text unchanged

        # Merge introspection fragments
        inner_monologue = "\n\n".join(all_introspections)

        # Token-budget metabolism on homeostasis
        if self.self_model and usage:
            try:
                # Wall-clock latency for metabolism model
                latency = time.perf_counter() - start_time
                self.self_model.homeostasis.apply_computational_metabolism(
                    session_id, 
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    response_time=latency
                )
            except Exception as e:
                logger.warning(f"Failed to apply computational metabolism: {e}")

        # Anti-echo: if reply ≈ previous assistant turn while user question changed, retry once
        try:
            import difflib
            prev_user = None
            prev_assistant = None
            hist = self.session_history.get(session_id, [])
            for m in reversed(hist):
                if prev_assistant is None and m.get("role") == "assistant":
                    prev_assistant = m.get("content", "") or ""
                elif prev_user is None and m.get("role") == "user":
                    prev_user = m.get("content", "") or ""
                if prev_user is not None and prev_assistant is not None:
                    break

            def _norm(s: str) -> str:
                return re.sub(r"\s+", " ", (s or "").strip())

            curr = _norm(response_text)
            prev = _norm(prev_assistant) if prev_assistant else ""
            user_prev = _norm(prev_user) if prev_user else ""
            user_curr = _norm(user_input)

            anti_echo_enabled = bool(config.get("parameters.chat.anti_echo.enabled", True))
            sim_threshold = float(config.get("parameters.chat.anti_echo.similarity_threshold", 0.88) or 0.88)
            min_chars = int(config.get("parameters.chat.anti_echo.min_chars", 200) or 200)
            max_compare_chars = int(config.get("parameters.chat.anti_echo.max_compare_chars", 2000) or 2000)
            retry_temp_delta = float(config.get("parameters.chat.anti_echo.retry_temp_delta", 0.2) or 0.2)

            if (
                anti_echo_enabled
                and prev
                and len(curr) > min_chars
                and len(prev) > min_chars
                and user_prev
                and user_curr
                and user_prev != user_curr
            ):
                sim = difflib.SequenceMatcher(None, prev[:max_compare_chars], curr[:max_compare_chars]).ratio()
                if sim >= sim_threshold:
                    logger.warning(f"[ANTI-ECHO] High similarity detected (sim={sim:.2f}). Retrying once with anti-echo directive.")
                    retry_messages = [dict(m) for m in messages]
                    # Anti-echo directive would append to last user row (not shown to end user)
                    if retry_messages and retry_messages[-1].get("role") == "user":
                        # [REMOVED] hardcoded anti-echo prose
                        pass
                    retry_temp = min(0.85, float(temperature) + retry_temp_delta)
                    response_text2, introspection2, tool_calls2, usage2 = self._call_vllm(
                        retry_messages,
                        retry_temp,
                        tools,
                        top_p=top_p,
                        presence_penalty=presence_penalty,
                        frequency_penalty=frequency_penalty,
                        max_tokens=max_tokens,
                    )
                    # Re-run tool loop if retry produced tool_calls
                    response_text2, all_introspections2, usage2, _, _ = run_tool_loop(
                        self,
                        session_id,
                        self.self_model,
                        retry_messages,
                        response_text2,
                        introspection2,
                        tool_calls2,
                        turn_index,
                        retry_temp,
                        tools,
                        top_p,
                        presence_penalty,
                        frequency_penalty,
                        max_tokens=max_tokens,
                        initial_usage=usage2,
                        embedder=embedder,
                    )
                    response_text = response_text2
                    inner_monologue = "\n\n".join(all_introspections2)
                    # Accumulate retry token usage
                    if usage2:
                        for k in ["prompt_tokens", "completion_tokens", "total_tokens"]:
                            usage[k] = usage.get(k, 0) + usage2.get(k, 0)
        except Exception as e:
            logger.debug(f"[ANTI-ECHO] Failed to run anti-echo check: {e}")

        # P2.7 receipt verifier: append receipts when model asserts tool facts without citing them
        #
        # Honor explicit user opt-out (“no receipts”) so we do not append receipts after the model complied.
        try:
            suppress_receipts_in_text = False
            try:
                ui = (user_input or "").lower()
                # Match exact phrases plus broad CN regexes for receipt opt-out
                if (
                    any(m in ui for m in [
                        "不要输出回执", "不要输出任何回执",
                        "不要输出收据", "不要输出任何收据",
                        "不提回执", "不写回执", "不要回执",
                        "不提收据", "不写收据", "不要收据",
                        "no receipt", "no receipts",
                        "do not output receipt", "don't output receipt",
                    ])
                    or re.search(r"(不要|请勿|别|不)\s*输出.*(回执|收据|receipt)", ui, flags=re.IGNORECASE)
                    # Broader CN pattern with up to 40 chars between negation and receipt word
                    or re.search(r"(不要|请勿|别|不).{0,40}(回执|收据|receipt)", ui, flags=re.IGNORECASE)
                ):
                    suppress_receipts_in_text = True
            except Exception:
                suppress_receipts_in_text = False

            if not suppress_receipts_in_text:
                receipts = extract_receipts_from_messages(messages)
                response_text, _ = enforce_receipts(response_text, receipts)
        except Exception as e:
            logger.debug(f"[RESPONSE-VERIFIER] Failed to enforce receipts: {e}")
        
        # Preserve raw assistant text before sanitize/truncation for diagnostics
        raw_response_text = response_text
        # Sanitizer pipeline
        final_response, is_garbage = sanitize_response(
            response_text,
            inner_monologue,
            require_introspection,
            concise_mode,
            logger,
        )

        # [2026-04-15] Reasoning-only models: promote reasoning_* to visible reply when content empty/short
        # (matches logs where raw_len≈0 but inner_len large)
        _rc_promo = (getattr(self, "_last_reasoning_content", None) or "").strip()
        if _rc_promo and (not final_response or len(final_response.strip()) < 5):
            _cap = 12000
            final_response = _rc_promo if len(_rc_promo) <= _cap else _rc_promo[-_cap:]
            logger.info(
                "[CHAT] Promoted reasoning_content to visible reply (content empty/short), out_len=%s",
                len(final_response),
            )

        # [2026-03-12] Fallback prose if sanitize yields empty body
        if not final_response or len(final_response.strip()) < 5:
            final_response = (
                "（本次未能形成可展示的正文，可能是接口临时异常、流式中断或模型仅返回了不可展示片段。"
                "请重试一次；若仍出现，可让我先做一次简短状态自检后再继续。）"
            )
            logger.warning(
                "[CHAT] Empty response after sanitize, injected fallback message "
                "(provider=%s model=%s raw_len=%s inner_len=%s)",
                MODEL_PROVIDER,
                get_model_id(MODEL_PROVIDER),
                len((raw_response_text or "").strip()),
                len((inner_monologue or "").strip()),
            )

        # [SAFETY] Final runtime filter (self-harm, weapons, illegal asks, minors, credential leaks, etc.)
        try:
            if bool(config.get("system.safety_output_filter_enabled", True)):
                filtered, blocked, category = apply_runtime_safety_filter(
                    user_input=user_input or "",
                    assistant_output=final_response or "",
                    logger=logger,
                )
                if blocked:
                    final_response = filtered
        except Exception as e:
            logger.debug(f"[SAFETY-OUTPUT-FILTER] failed: {e}")

        # [DISABLED 2026-01-11] Post-sanitize hard truncation off (trust max_tokens + sanitize)
        # Monitoring only
        if len(final_response) > hard_char_limit:
            original_length = len(final_response)
            logger.info(f"[TRUNCATION-MONITOR] Final response length {original_length} exceeds soft limit {hard_char_limit}, but truncation is disabled")
            
            # Log oversize final body (monitoring)
            try:
                output_awareness = get_output_awareness(self.db_path)
                turn_idx = self.session_turn_index.get(session_id, 0)
                output_awareness.record_truncation(
                    session_id=session_id,
                    turn_index=turn_idx,
                    original_length=original_length,
                    truncated_length=original_length,  # not actually truncated
                    reason="monitor_only_final"
                )
            except Exception as e:
                logger.warning(f"Failed to record final truncation monitor: {e}")
            
            # final_response unchanged

        # [WillWatch] physical stutter simulation removed
        pass

        latency = time.perf_counter() - start_time

        # [WillWatch] identity-name hard filter removed
        pass

        # Strip receipt lines when user opted out (even if model still echoed rct_* tokens)
        try:
            suppress_receipts_in_text2 = False
            try:
                ui2 = (user_input or "").lower()
                if (
                    any(m in ui2 for m in [
                        "不要输出回执", "不要输出任何回执",
                        "不要输出收据", "不要输出任何收据",
                        "不提回执", "不写回执", "不要回执",
                        "不提收据", "不写收据", "不要收据",
                        "no receipt", "no receipts",
                        "do not output receipt", "don't output receipt",
                    ])
                    or re.search(r"(不要|请勿|别|不)\s*输出.*(回执|收据|receipt)", ui2, flags=re.IGNORECASE)
                    or re.search(r"(不要|请勿|别|不).{0,40}(回执|收据|receipt)", ui2, flags=re.IGNORECASE)
                ):
                    suppress_receipts_in_text2 = True
            except Exception:
                suppress_receipts_in_text2 = False

            if suppress_receipts_in_text2 and final_response:
                # 1) Remove embedded receipt+rct_* spans
                final_response = re.sub(r"回执\s*[:：]\s*\brct_[0-9a-f]{16,}\b", "", final_response, flags=re.IGNORECASE)
                # 2) Strip bare rct_* tokens
                final_response = re.sub(r"\brct_[0-9a-f]{16,}\b", "", final_response, flags=re.IGNORECASE)
                final_response = re.sub(r"\s{2,}", " ", final_response).strip()
        except Exception:
            pass

        # P1.1: normalized assistant string for downstream persistence
        response_text = final_response

        # [REMOVED] legacy duplicate-response penalty folded into anti-echo path

        # [WillWatch] cognitive-dissonance penalty removed
        pass

        # P0: write sensory buffer synchronously before returning (avoid losing short-term trace)
        if self.sensory_buffer:
            try:
                # Log user line into sensory buffer
                self.sensory_buffer.add_sensory_input(
                    session_id,
                    f"User: {user_input}",
                    input_type="conversation",
                    turn_index=turn_index
                )
                # Log assistant line into sensory buffer
                self.sensory_buffer.add_sensory_input(
                    session_id,
                    f"AI: {final_response}",
                    input_type="conversation",
                    turn_index=turn_index
                )
            except Exception as e:
                logger.warning(f"Failed to write to sensory buffer synchronously: {e}")
        
        # Async follow-up (z_self writes, narrative, reflection, …)
        # [Fix] AsyncProcessor queue was not drained in some deployments; use explicit daemon thread instead.
        def background_tasks():
            # Short delay so main thread can flush UI / HTTP response first
            import time
            time.sleep(3.0) 
            
            logger.info(f"[BG-TASK] background_tasks started for session={session_id}")
            
            # 4) Evidence bundle for SelfTick (user text + inner monologue + reply head)
            evidence_text = f"User: {user_input}\n"
            if inner_monologue:
                evidence_text += f"My Thought: {inner_monologue}\n"
            evidence_text += f"My Reply: {final_response[:200]}"
            
            if self.self_tick is not None and SELF_TICK_INTERVAL > 0:
                self.self_tick.add_evidence(session_id, evidence_text)
                logger.info(f"[BG-TASK] Evidence added: session={session_id}")
            
            # 5) Self Tick scheduling / immediate triggers
            z_self_updated = False
            drift = 0.0
            self_tick_triggered = False
            self_tick_result = None
            # [FIX 2026-01-25] Self Tick cadence reads DB (chat_turns + self_state.tick), not RAM counters
            
            # Level-2: immediate tick on salient events
            event_triggered_tick = False
            event_trigger_reason = None
            if self.self_tick is not None and self.self_model is not None:
                try:
                    from backend.event_triggered_tick import get_event_trigger_manager
                    event_manager = get_event_trigger_manager()
                    
                    # Snapshot physiology / affect for event manager
                    # [2026-04-12] use get_pain_status (get_pain removed)
                    pain_status = self.self_model.get_pain_status(session_id)
                    current_pain = pain_status.get("total_pain", 0.0) if pain_status else 0.0
                    current_energy = self.self_model.get_energy(session_id)
                    
                    # Emotion intensity (if emotion_store present)
                    current_emotion_intensity = 0.0
                    if self.self_model.emotion_store:
                        emotion_state = self.self_model.emotion_store.get_emotion_state(session_id)
                        if emotion_state:
                            current_emotion_intensity = emotion_state.intensity
                    
                    # Immediate tick?
                    should_trigger, trigger_reason = event_manager.should_trigger_immediate_tick(
                        session_id,
                        current_pain,
                        current_energy,
                        current_emotion_intensity
                    )
                    
                    if should_trigger:
                        logger.info(f"🚨 Event-triggered Self Tick: {trigger_reason}")
                        self_tick_result = self.self_tick.trigger(
                            session_id,
                            self.self_model,
                            self.persona_store,
                            trigger_reason=f"event:{trigger_reason}"
                        )
                        # [2026-02-22] guard: self_tick_result must be dict-like
                        if isinstance(self_tick_result, dict) and self_tick_result.get("success"):
                            event_triggered_tick = True
                            event_trigger_reason = trigger_reason
                            z_self_updated = True
                            drift = self_tick_result.get("drift", 0.0)
                            self_tick_triggered = True
                        
                except Exception as e:
                    logger.warning(f"Event-triggered tick check failed: {e}")
            
            # Scheduled tick path (skipped if event tick already fired)
            # [FIX 2026-01-25] use DB turn counts + delta since last tick (no modulo / no turn-0 fire)
            # [2026-03-25] dynamic interval from arousal / drift / energy
            if not event_triggered_tick and self.self_tick is not None and SELF_TICK_INTERVAL > 0:
                try:
                    with sqlite3.connect(self.db_path) as conn:
                        # Total persisted chat rows for session
                        cur = conn.execute(
                            "SELECT COUNT(*) FROM chat_turns WHERE session_id=?",
                            (session_id,)
                        )
                        current_turn = cur.fetchone()[0]
                        
                        # self_state.tick stores turn index at last tick
                        cur = conn.execute(
                            "SELECT tick FROM self_state WHERE session_id=?",
                            (session_id,)
                        )
                        row = cur.fetchone()
                        last_tick_turn = row[0] if row else 0
                    
                    # [2026-03-25] shrink interval when arousal/drift high; stretch when energy low
                    dynamic_interval = float(SELF_TICK_INTERVAL)
                    interval_reason = "base"
                    
                    if self.self_model:
                        try:
                            tick_struct = self.self_model.get_structured_summary(session_id)
                            tick_arousal = abs(float(tick_struct.get("arousal", 0.0) or 0.0))
                            tick_energy_pct = float(tick_struct.get("energy", 50.0) or 50.0)
                            tick_drift = float(tick_struct.get("drift", 0.0) or 0.0)
                            
                            # High arousal → shorter interval
                            if tick_arousal > 0.4:
                                dynamic_interval *= max(0.5, 1.0 - tick_arousal * 0.5)
                                interval_reason = f"arousal={tick_arousal:.2f}"
                            
                            # High drift → shorter interval
                            if tick_drift > 0.1:
                                dynamic_interval *= max(0.5, 1.0 - tick_drift * 2.0)
                                interval_reason = f"drift={tick_drift:.2f}"
                            
                            # Low energy → longer interval
                            if tick_energy_pct < 30:
                                dynamic_interval *= min(2.0, 1.0 + (30 - tick_energy_pct) / 30)
                                interval_reason = f"low_energy={tick_energy_pct:.0f}"
                        except Exception as e:
                            logger.debug(f"Failed to compute dynamic tick interval: {e}")
                    
                    # Clamp interval to [1, 2×SELF_TICK_INTERVAL]
                    dynamic_interval = max(1.0, min(float(SELF_TICK_INTERVAL) * 2, dynamic_interval))
                    
                    # Fire when delta turns ≥ dynamic interval (and turn>0)
                    turns_since_last_tick = current_turn - last_tick_turn
                    
                    logger.info(
                        f"[BG-TASK] Self Tick check: current_turn={current_turn}, "
                        f"last_tick_turn={last_tick_turn}, turns_since={turns_since_last_tick}, "
                        f"dynamic_interval={dynamic_interval:.1f} ({interval_reason}), "
                        f"should_trigger={turns_since_last_tick >= dynamic_interval and current_turn > 0}"
                    )
                    
                    if turns_since_last_tick >= dynamic_interval and current_turn > 0:
                        # Scheduled Self tick
                        logger.info(
                            f"Triggering scheduled Self Tick: current_turn={current_turn}, "
                            f"last_tick={last_tick_turn}, turns_since={turns_since_last_tick}, "
                            f"dynamic_interval={dynamic_interval:.1f} ({interval_reason})"
                        )
                        self_tick_result = self.self_tick.trigger(
                            session_id,
                            self.self_model,
                            self.persona_store,
                            trigger_reason="scheduled"
                        )
                        # [2026-02-22] guard: self_tick_result must be dict-like
                        if isinstance(self_tick_result, dict) and self_tick_result.get("success"):
                            z_self_updated = True
                            drift = self_tick_result.get("drift", 0.0)
                            self_tick_triggered = True

                            # Optional collective broadcast on large drift (experimental module)
                            if drift > 0.15 and self_tick_triggered:
                                try:
                                    from experimental.social_simulation import CollectiveConsciousness
                                    collective = CollectiveConsciousness(self.db_path)
                                    summary = self.self_model.get_summary(session_id)
                                    collective.broadcast(
                                        session_id,
                                        content=f"经历了一次深刻的认知重组: {summary[:100]}...",
                                        intensity=drift * 2.0,
                                        z_impact=self.self_model.get_z_self(session_id)
                                    )
                                except Exception:
                                    pass
                
                except Exception as e:
                    logger.error(f"Scheduled Self Tick check failed: {e}", exc_info=True)

            # P1.1 world context primarily via PromptBuilder; optional POST /world/state

            # P1.3 narrative memory event append
            if self.self_narrative:
                self.self_narrative.add_event(
                        session_id=session_id,
                    user_input=user_input,
                    assistant_response=final_response,
                    introspection=inner_monologue,
                    significance=0.5  # default salience; future: LLM-scored
                )
                
                # [2026-03-25] consolidate episodic → long-term when signals strong enough
                current_history = self.session_history.get(session_id, [])
                conversation_turns = len(current_history) // 2
                
                try:
                    # Pull structured summary for consolidation heuristics
                    z_self_summary = ""
                    consolidate_drift = 0.0
                    consolidate_pain = 0.0
                    consolidate_emotion = 0.0
                    
                    if self.self_model:
                        struct_summary = self.self_model.get_structured_summary(session_id)
                        z_self_summary = self.self_model.get_summary(session_id)
                        consolidate_drift = float(struct_summary.get("drift", 0.0) or 0.0)
                        consolidate_emotion = abs(float(struct_summary.get("arousal", 0.0) or 0.0))
                        
                        pain_status = self.self_model.get_pain_status(session_id)
                        consolidate_pain = float(pain_status.get("total_pain", 0.0) or 0.0)
                    
                    # Triggers: drift/pain/arousal thresholds OR every 5 turns as backstop
                    should_consolidate = (
                        consolidate_drift > 0.1
                        or consolidate_pain > 0.3
                        or consolidate_emotion > 0.4
                        or (conversation_turns >= 5 and conversation_turns % 5 == 0)
                    )
                    
                    if should_consolidate:
                        trigger_reason = []
                        if consolidate_drift > 0.1:
                            trigger_reason.append(f"drift={consolidate_drift:.2f}")
                        if consolidate_pain > 0.3:
                            trigger_reason.append(f"pain={consolidate_pain:.2f}")
                        if consolidate_emotion > 0.4:
                            trigger_reason.append(f"emotion={consolidate_emotion:.2f}")
                        if not trigger_reason:
                            trigger_reason.append(f"turns={conversation_turns}")
                        
                        consolidated = self.self_narrative.consolidate_memory(
                            session_id=session_id,
                            recent_history=current_history[-10:],
                            z_self_summary=z_self_summary,
                            drift=consolidate_drift,
                            pain=consolidate_pain
                        )
                        if consolidated:
                            logger.info(f"Memory consolidated for session {session_id} (trigger: {', '.join(trigger_reason)})")
                except Exception as e:
                    logger.warning(f"Failed to consolidate memory: {e}")
                
                # [2026-03-25] knowledge extraction gated on internal signals
                try:
                    # Internal scalars for extractor
                    extract_drift = 0.0
                    extract_pain = 0.0
                    extract_emotion_intensity = 0.0
                    
                    if self.self_model:
                        struct_summary = self.self_model.get_structured_summary(session_id)
                        extract_drift = float(struct_summary.get("drift", 0.0) or 0.0)
                        
                        pain_status = self.self_model.get_pain_status(session_id)
                        extract_pain = float(pain_status.get("total_pain", 0.0) or 0.0)
                        
                        # Emotion intensity proxy = |arousal|
                        extract_emotion_intensity = abs(float(struct_summary.get("arousal", 0.0) or 0.0))
                    
                    knowledge_result = self.self_narrative.extract_knowledge(
                        user_input=user_input,
                        assistant_response=final_response,
                        session_id=session_id,
                        drift=extract_drift,
                        pain=extract_pain,
                        emotion_intensity=extract_emotion_intensity
                    )
                    if knowledge_result and knowledge_result.get("success"):
                        logger.info(f"Knowledge extracted (trigger: {knowledge_result.get('trigger')}): {knowledge_result.get('id')}")
                except Exception as e:
                    logger.debug(f"Knowledge extraction skipped: {e}")
            
            # Sovereign phrase → kick off mind wandering thread
            if "【确认进入深度思考】" in final_response:
                logger.info(f"Sovereign trigger: Manually starting Mind Wandering for {session_id}")
                if self.self_tick:
                    try:
                        import threading
                        # Non-blocking dreaming thread
                        t = threading.Thread(target=self.self_tick.process_dreaming, args=(session_id,), daemon=True)
                        t.start()
                        logger.info(f"Mind Wandering thread started for {session_id}")
                    except Exception as e:
                        logger.error(f"Failed to start manual mind wandering: {e}")

            # Phase 6: other-model update
            if self.other_model:
                try:
                    # Refresh history snapshot (session_history is mutable shared state)
                    current_history = self.session_history.get(session_id, [])
                    # [FIX 2026-01-25] persisted chat_turns count for analytics fields
                    with sqlite3.connect(self.db_path) as conn:
                        cur = conn.execute(
                            "SELECT COUNT(*) FROM chat_turns WHERE session_id=?",
                            (session_id,)
                        )
                        session_count = cur.fetchone()[0]
                    total_turns = len(current_history)
                    
                    interaction = {
                        "user_message": user_input,
                        "ai_response": final_response,
                        "session_history": current_history[-10:],  # last 10 turns
                        "session_count": session_count,
                        "total_turns": total_turns
                    }
                    
                    updated_model = self.other_model.update_other_model(session_id, interaction)
                    logger.debug(f"Updated other model for session {session_id}: relationship={updated_model.get('relationship_type')}")

                    # [2026-03-30] mirror feedback integration into self_model
                    if self.self_model and hasattr(self.self_model, 'integrate_mirror_feedback'):
                        self.self_model.integrate_mirror_feedback(session_id)
                except Exception as e:
                    logger.warning(f"Failed to update other model: {e}")
            
            # 6) Reflection / persona candidate generation (decoupled from tick cadence)
            reflection_result = run_reflection(self, session_id, turn_index)
            
            # [FIX 2026-01-24] cache reflection summary for UI polling
            # [2026-03-24] always store stub outcomes too (skipped / no candidates)
            if not hasattr(self, 'session_reflection_cache'):
                self.session_reflection_cache = {}
            self.session_reflection_cache[session_id] = {
                "result": reflection_result,
                "turn_index": turn_index,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            logger.info(f"[REFLECTION] Cached summary for session {session_id}: {reflection_result}")

            # P1.2 cross-subsystem interaction hooks (emotion/motivation/rule feedback)
            response_success = len(response_text) > 10 and "错误" not in response_text.lower()
            
            # Somatic → emotion coupling
            if self.self_model and getattr(self.self_model, 'emotion_store', None) and getattr(self.self_model, 'somatic_store', None):
                self._trigger_emotion_from_somatic(session_id)

            if self.self_model and self.self_model.emotion_store:
                self._trigger_emotion_from_rule_execution(session_id, response_success)  # bool heuristic for “successful reply”
            
            if self.self_model and self.self_model.motivation_store:
                self._trigger_motivation_from_rule_execution(session_id, response_success)
            
            if self.self_model and self.self_model.emotion_store and self.self_model.motivation_store:
                self._trigger_motivation_from_emotion(session_id)
            
            if self.self_model and self.self_model.emotion_store:
                self._trigger_rule_from_emotion(session_id)
            
            if self.self_model and self.self_model.motivation_store:
                self._trigger_rule_from_motivation(session_id)
            
            if self.self_model and self.self_model.motivation_store and self.self_model.emotion_store:
                self._trigger_emotion_from_motivation(session_id)
            
            # Emit lightweight chat_response telemetry
            self.event_logger.log_event(session_id, "chat_response", json.dumps({
                "latency": latency,
                "response_len": len(final_response),
                "z_self_updated": z_self_updated
            }))

            # Needs/novelty updates happen on main thread pre-return; skip duplicates here
        
        
        # [FIX 2026-01-25] tick cadence no longer piggybacks on RAM counters (see DB logic above)

        # Spawn daemon for background_tasks
        logger.info(f"[MAIN-THREAD] About to start background_tasks thread for session={session_id}")
        try:
            import threading
            t = threading.Thread(target=background_tasks, daemon=True)
            t.start()
            logger.info(f"[MAIN-THREAD] background_tasks thread started successfully")
        except Exception as e:
            logger.error(f"[MAIN-THREAD] Failed to start background_tasks thread: {e}")
            # Last-resort synchronous run if thread spawn fails
            try:
                background_tasks()
            except Exception as e2:
                logger.error(f"[MAIN-THREAD] Fallback background_tasks also failed: {e2}")
        
        # Update in-memory session history immediately (conversation coherence)
        _sess_max_raw = config.get("parameters.chat.session_history_max_messages", 40)
        _session_history_cap = max(4, int(_sess_max_raw if _sess_max_raw is not None else 40))
        append_history(
            self.session_history,
            session_id,
            user_input,
            final_response,
            logger,
            is_system_reminder=is_system_reminder,
            max_len=_session_history_cap,
        )

        # Phase 3: optional spontaneous outbound messages (legacy queue mostly unused)
        spontaneous_messages = []
        # Endogenous urges now arrive via prompt injection; pending-action queue retired
        # Future: websocket push could surface high-intensity urges here
        # if self.endogenous_system:
        #    ...
        
        # Build API payload + meta (SelfTick still async—meta reflects scheduling snapshot)
        meta: Dict[str, Any] = {"turn_index": turn_index}
        
        # [NEW] refresh structured self_state after any inline penalties
        try:
            if self.self_model:
                meta["self_state"] = self.self_model.get_structured_summary(session_id)
            else:
                logger.warning("[CHAT] self.self_model is None, cannot get structured summary")
        except Exception as e:
            logger.error(f"[CHAT] Failed to get structured summary: {e}", exc_info=True)

        # [NEW] echo sampling + hard limits for eval (96-d physiology → generation)
        try:
            meta["sampling"] = {
                "temperature": float(temperature),
                "top_p": float(top_p),
                "presence_penalty": float(presence_penalty),
                "frequency_penalty": float(frequency_penalty),
            }
            meta["constraints"] = {
                "max_tokens": int(max_tokens),
                "hard_char_limit": int(hard_char_limit),
                "energy_used_for_limit": float(energy),
                "viscosity_used_for_limit": float(viscosity),
            }
            # Strip bulky text fields from generation_params for JSON meta
            if isinstance(generation_params, dict):
                meta["generation_params"] = {
                    "pain_level": float(generation_params.get("pain_level", 0.0) or 0.0),
                    "noise_injection_prob": float(generation_params.get("noise_injection_prob", 0.0) or 0.0),
                    "system_entropy": float(generation_params.get("system_entropy", 0.0) or 0.0),
                    "system_age_ticks": int(generation_params.get("system_age_ticks", 0) or 0),
                    "noise_perturbation": float(generation_params.get("noise_perturbation", 0.0) or 0.0),
                }
        except Exception:
            pass

        # [NEW] persona retrieval snapshot for eval
        try:
            if isinstance(introspection_config, dict) and introspection_config.get("retrieval") is not None:
                meta["retrieval"] = introspection_config.get("retrieval")
        except Exception:
            pass

        # [NEW] memory retrieval snapshot for eval
        try:
            if isinstance(introspection_config, dict) and introspection_config.get("memory") is not None:
                meta["memory"] = introspection_config.get("memory")
        except Exception:
            pass

        # [NEW] token usage telemetry
        if usage:
            meta["usage"] = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }

        # [FIX 2026-01-23] API introspection includes inner_monologue when present
        full_introspection_for_api = introspection.copy() if introspection else {}
        if inner_monologue:
            full_introspection_for_api["inner_monologue"] = inner_monologue

        # [2026-01-27] mark work completion in pain_system (satisfaction nudges physiology)
        if self.self_model and self.self_model.pain_system:
            response_success = len(final_response) > 10 and "错误" not in final_response.lower()
            if response_success:
                # Healthy completion → higher meaningfulness
                self.self_model.pain_system.set_working_state(False, 0.8)
                logger.info(f"[WORK-STATE] Work completed successfully, satisfaction boosted")
            else:
                # Weak / error-ish completion → modest meaningfulness
                self.self_model.pain_system.set_working_state(False, 0.3)
                logger.info(f"[WORK-STATE] Work completed with potential issues")

        result = {
            "response": final_response,
            "introspection": full_introspection_for_api,  # includes inner_monologue when captured
            "tool_calls": tool_calls,
            "meta": meta,
        }

        # =========================
        # Needs/Novelty (Main Thread)
        # =========================
        # Goal: needs/novelty reflect this turn before HTTP response returns (/self/state stays fresh)
        try:
            if self.self_model:
                # 1) novelty_signal (topic shift + tool learning + memory hits)
                try:
                    from backend.novelty_estimator import estimate_novelty_signal
                    prev_user = None
                    try:
                        # History already includes this turn’s user+assistant; walk back to prior user
                        hist = self.session_history.get(session_id, []) or []
                        # Second-most-recent user message
                        user_seen = 0
                        for msg in reversed(hist):
                            if isinstance(msg, dict) and msg.get("role") == "user" and msg.get("content"):
                                user_seen += 1
                                if user_seen >= 2:
                                    prev_user = str(msg.get("content"))
                                    break
                    except Exception:
                        prev_user = None

                    mem_sig = None
                    try:
                        mem_sig = (meta.get("memory") or {}).get("signal") if isinstance(meta, dict) else None
                    except Exception:
                        mem_sig = None

                    tool_bonus = float(config.get("parameters.homeostasis.tool_learning_bonus", 0.15) or 0.15)
                    novelty_strength, novelty_components = estimate_novelty_signal(
                        embedder=self.persona_store.embedder,
                        user_input=user_input,
                        prev_user_input=prev_user,
                        tool_calls=tool_calls if isinstance(tool_calls, list) else [],
                        memory_signal=mem_sig if isinstance(mem_sig, dict) else None,
                        tool_learning_bonus=tool_bonus,
                    )
                    self.self_model.record_novelty_signal(
                        session_id,
                        {"strength": novelty_strength, "components": novelty_components},
                    )
                    try:
                        self.event_logger.log_event(
                            session_id,
                            "physiological_novelty",
                            json.dumps(
                                {"strength": novelty_strength, "components": novelty_components},
                                ensure_ascii=False,
                            ),
                        )
                    except Exception:
                        pass
                except Exception:
                    pass

                # 2) needs vector (sync persist)
                needs = self.self_model.update_needs(session_id, interaction_type="chat")

                # 3) [2026-03-30] homeostasis event detector (lightweight heuristics)
                try:
                    tool_error_count = sum(1 for t in all_tools_called if isinstance(t, dict) and t.get("error"))
                    sentiment = 0.0
                    # Crude sentiment cue from CN praise / complaint tokens
                    if any(kw in user_input for kw in ["谢谢", "感谢", "太好了", "棒", "厉害"]):
                        sentiment = 0.6
                    elif any(kw in user_input for kw in ["不对", "错了", "不行", "差", "烦"]):
                        sentiment = -0.5
                    
                    # [2026-03-30] richer new-topic heuristic than “contains ?”
                    is_new = False
                    new_topic_keywords = ["新的", "另一个", "换个", "不同的", "试试", "想问", "好奇", "发现", "原来"]
                    if "?" in user_input and len(user_input) > 50:
                        is_new = True
                    elif any(kw in user_input for kw in new_topic_keywords):
                        is_new = True
                    elif len(user_input) > 100 and "?" in user_input:
                        is_new = True
                    
                    # [2026-03-30] creative-task detector (CN keywords)
                    is_creative = False
                    creative_keywords = ["写", "创作", "想象", "设计", "编", "构思", "故事", "文章", "诗", "小说", "剧本", "红楼梦"]
                    if any(kw in user_input for kw in creative_keywords):
                        is_creative = True
                    elif any(kw in final_response for kw in ["写作", "创作", "续写", "改编"]):
                        is_creative = True
                    
                    self.self_model.homeostasis.detect_and_process_events(
                        session_id=session_id,
                        user_message=user_input,
                        assistant_response=final_response,
                        response_time_ms=int(latency * 1000),
                        tool_calls=len(all_tools_called),
                        tool_errors=tool_error_count,
                        is_new_topic=is_new,
                        sentiment_score=sentiment,
                        is_creative=is_creative,
                    )
                except Exception as e:
                    logger.debug(f"[HOMEOSTASIS-EVENT] Failed to process events: {e}")
        except Exception:
            pass

        # Attach spontaneous_messages if any
        if spontaneous_messages:
            result["spontaneous_messages"] = spontaneous_messages
            result["has_pending_spontaneous"] = True

        # [FIX 2026-01-23] persist chat_turn (required for reflection pipelines)
        # [P1] log + retry once; flag meta if still not persisted
        chat_turn_persisted = False
        full_introspection = introspection.copy() if introspection else {}
        if inner_monologue:
            full_introspection["inner_monologue"] = inner_monologue
        log_chat_turn_kw = dict(
            session_id=session_id,
            turn_index=turn_index,
            user_input=user_input,
            assistant_output=final_response,
            introspection=full_introspection if full_introspection else None,
            drift=meta.get("self_state", {}).get("drift") if isinstance(meta.get("self_state"), dict) else None,
            tick_count=meta.get("self_state", {}).get("tick_count") if isinstance(meta.get("self_state"), dict) else None,
            self_tick_triggered=meta.get("self_state", {}).get("self_tick_triggered", False) if isinstance(meta.get("self_state"), dict) else False,
            reflection=None,
            latency=latency,
            tool_used={"tools_called": all_tools_called} if all_tools_called else None,
        )
        for attempt in range(2):
            try:
                self.event_logger.log_chat_turn(**log_chat_turn_kw)
                chat_turn_persisted = True
                logger.debug(f"[CHAT-TURN-LOG] Saved chat turn to database: turn_index={turn_index}, has_introspection={bool(inner_monologue)}")
                break
            except Exception as e:
                if attempt == 0:
                    logger.error(f"[CHAT-TURN-LOG] Failed to log chat turn (will retry once): {e}", exc_info=True)
                else:
                    logger.error(f"[CHAT-TURN-LOG] Failed to log chat turn after retry. session_history has this turn but chat_turns does not: {e}", exc_info=True)
        if not chat_turn_persisted:
            result["chat_turn_persisted"] = False

        # Passive z_self observation logging (calibration dataset)
        try:
            if self.self_model and self.self_model.z_self_observer:
                z_after = self.self_model.get_z_self(session_id)
                self.self_model.z_self_observer.record(
                    session_id=session_id,
                    z_self_before=None,
                    z_self_after=z_after,
                    llm_response=result.get("response", ""),
                    extra_context={"source": "chat"},
                )
        except Exception:
            pass

        # [2026-04-13] autonomy gate scans raw model text, sanitized reply, <thought>, and reasoning_content
        try:
            from backend.autonomy_gate import apply_assistant_autonomy_markers
            _ag_parts = [
                (raw_response_text or "").strip(),
                (final_response or "").strip(),
                (inner_monologue or "").strip() if inner_monologue else "",
                (getattr(self, "_last_reasoning_content", None) or "").strip(),
            ]
            _ag_combined = "\n".join(p for p in _ag_parts if p)
            _ag_ass = apply_assistant_autonomy_markers(_ag_combined, session_id)
            if _ag_ass:
                logger.info("[AUTONOMY-GATE] Assistant line marker: %s", _ag_ass)
                if isinstance(result.get("meta"), dict):
                    result["meta"]["autonomy_gate"] = _ag_ass
        except Exception as e:
            logger.debug("[AUTONOMY-GATE] assistant markers skipped: %s", e)

        return result
    
    def check_spontaneous_action(self, session_id: str) -> Optional[Dict]:
        """
        检查是否应该发起自发行为
        
        [Refinement] 适配新的 EndogenousSystem。
        目前系统主要通过 Prompt 注入 Urges 来影响回复。
        如果需要支持 WebSocket 主动推送，可以检查 Urge 强度。
        """
        if not self.endogenous_system:
            return None
        
        try:
            spontaneous_messages = []
            # Current endogenous urges snapshot
            urges = self.endogenous_system.get_urges(session_id)
            # Strong urge heuristic: lines starting with bracketed “intensity” tag → eligible for proactive push
            for urge in urges:
                if "[强烈" in urge:
                    spontaneous_messages.append({
                        "type": "spontaneous",
                        "content": urge,
                        "trigger": "strong_urge",
                        "intensity": 1.0
                    })
            if spontaneous_messages:
                return spontaneous_messages[0]
        except Exception as e:
            logger.debug(f"Failed to check spontaneous action: {e}")
        
        return None
    
    def _generate_spontaneous_message(self, action: Dict) -> str:
        """基于行动提议生成自发消息"""
        # [Refinement] placeholder if ActionProposal queue returns
        templates = {
            "curiosity": [
                "我一直在想一个问题：{topic}",
                "说起来，关于{topic}，我很好奇...",
                "你之前提到的{topic}，我还想了解更多。"
            ],
            "boredom": [
                "我在想，我们可以聊点别的吗？",
                "有什么新的话题你想讨论吗？",
                "我有些无聊，想聊点有趣的。"
            ],
            "unresolved": [
                "之前有个问题我一直没想清楚：{topic}",
                "关于{topic}，我还有些疑问。",
                "我一直在想{topic}这件事。"
            ],
            "social": [
                "你今天过得怎么样？",
                "最近在忙什么呢？",
                "我有点想念和你聊天。"
            ]
        }
        
        import random
        trigger_type = action.get("trigger_type", "curiosity")
        topic = action.get("topic") or action.get("content", "这个话题")
        template_list = templates.get(trigger_type, templates["curiosity"])
        template = random.choice(template_list)
        
        try:
            return template.format(topic=topic)
        except KeyError:
            # Missing {topic} placeholder → return template verbatim
            return template
    
    def _get_pending_spontaneous_actions(self, session_id: str) -> List[Dict]:
        """获取待执行的自发行为"""
        actions = []
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute("""
                    SELECT id, trigger_type, content, intensity, created_at
                    FROM spontaneous_actions
                    WHERE session_id = ? AND approved = 1 AND executed_at IS NULL
                    ORDER BY created_at DESC
                    LIMIT 3
                """, (session_id,))
                
                for row in cur.fetchall():
                    actions.append({
                        "id": row[0],
                        "trigger_type": row[1],
                        "content": row[2],
                        "intensity": row[3],
                        "created_at": row[4]
                    })
        except Exception as e:
            logger.debug(f"Failed to get pending actions: {e}")
        
        return actions
    
    def _mark_action_executed(self, action_id: str):
        """标记行动为已执行"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE spontaneous_actions
                    SET executed_at = ?
                    WHERE id = ?
                """, (datetime.now(timezone.utc).isoformat(), action_id))
                conn.commit()
        except Exception as e:
            logger.debug(f"Failed to mark action executed: {e}")

    def _parse_xml_tool_calls(self, content: str) -> List[Dict]:
        """
        Fallback parser for DeepSeek/XML style tool calls in content.
        Copied from MindWandering to support XML tool calls in normal chat.
        [2026-03-18] 新增对 <call type="tool" name="..."><arg name="...">...</arg></call> 格式的解析。
        """
        import re
        import uuid
        import json
        tool_calls = []

        # [2026-03-18] Prefer legacy XML <call type="tool" …> encoding when present
        pattern_call = r'<call\s+type="tool"\s+name="([^"]+)"[^>]*>([\s\S]*?)</call>'
        matches_call = re.findall(pattern_call, content, re.IGNORECASE)
        for func_name, body in matches_call:
            args = {}
            for m in re.finditer(r'<arg\s+name="([^"]+)"[^>]*>([\s\S]*?)</arg>', body, re.IGNORECASE):
                args[m.group(1)] = m.group(2).strip()
            if args:
                if func_name == "write_file":
                    if "file_path" in args and "filename" not in args:
                        args["filename"] = args["file_path"]
                    if "path" in args and "filename" not in args:
                        args["filename"] = args["path"]
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "function": {"name": func_name, "arguments": json.dumps(args)},
                    "type": "function",
                })
        if tool_calls:
            return tool_calls

        # Match <xDSMLxinvoke name="..."> ... </xDSMLxinvoke>
        pattern_invoke = r'<.DSML.invoke name="([^"]+)">([\s\S]*?)</.DSML.invoke>'
        matches_invoke = re.findall(pattern_invoke, content)

        for func_name, body in matches_invoke:
            args = {}
            
            # Try to find tool_input JSON first
            pattern_input = r'<.DSML.tool_input>(.*?)</.DSML.tool_input>'
            match_input = re.search(pattern_input, body, re.DOTALL)
            
            if match_input:
                try:
                    import json
                    args = json.loads(match_input.group(1).strip())
                except Exception as e:
                    logger.warning(f"Failed to parse JSON in tool_input: {e}")
            else:
                # Try parameters
                pattern_param = r'<.DSML.parameter name="([^"]+)"[^>]*>(.*?)</.DSML.parameter>'
                params = re.findall(pattern_param, body, re.DOTALL)
                for k, v in params:
                    args[k] = v.strip()
            
            if args:
                import uuid
                import json
                
                # Map specific keys if needed
                if func_name == "write_file":
                    # Fix: map file_path to filename
                    if "file_path" in args and "filename" not in args:
                        args["filename"] = args["file_path"]
                    if "path" in args and "filename" not in args:
                        args["filename"] = args["path"]
                    
                    # Sanitize path (strip directory if needed, handled by FileTool now but good to be safe)
                    if "filename" in args and "/" in args["filename"]:
                         # We let FileTool._sanitize_path handle the stripping logic now
                         pass
                
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "function": {
                        "name": func_name,
                        "arguments": json.dumps(args)
                    },
                    "type": "function"
                })

        return tool_calls

    def _prepare_messages_for_thinking(self, messages: List[Dict], keep_recent_reasoning: bool = True) -> List[Dict]:
        """
        为 DeepSeek 思考模式准备消息列表
        
        根据 DeepSeek API 文档及社区反馈（opencode#17523, continue#10498）：
        - 思考模式下，**所有** assistant 消息都必须包含 reasoning_content 字段（可为空串）
        - 否则 API 返回 400: Missing `reasoning_content` field in the assistant message at message index N
        
        参考: https://api-docs.deepseek.com/zh-cn/guides/thinking_mode#tool-calls
        """
        prepared = []
        fixed_count = 0
        
        for i, msg in enumerate(messages):
            if isinstance(msg, dict):
                new_msg = dict(msg)
                role = new_msg.get("role", "")
                
                if role == "assistant":
                    # [2026-03-13] assistant rows must include reasoning_content key (empty string ok)
                    if "reasoning_content" not in new_msg:
                        new_msg["reasoning_content"] = ""
                        fixed_count += 1
                        logger.debug(f"[THINKING-FIX] Added empty reasoning_content to assistant msg[{i}]")
                
                prepared.append(new_msg)
            else:
                new_msg = {
                    "role": getattr(msg, "role", "assistant"),
                    "content": getattr(msg, "content", "") or ""
                }
                if getattr(msg, "role", "") == "assistant":
                    rc = getattr(msg, "reasoning_content", None) if hasattr(msg, "reasoning_content") else None
                    new_msg["reasoning_content"] = rc if rc else ""
                    if not rc:
                        fixed_count += 1
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    new_msg["tool_calls"] = msg.tool_calls
                
                prepared.append(new_msg)
        
        if fixed_count > 0:
            logger.info(f"[THINKING-FIX] Ensured reasoning_content for {fixed_count} assistant message(s)")
        
        return prepared

    @staticmethod
    def _finalize_stream_tool_calls(
        tool_calls_acc: Dict[int, Dict[str, Any]],
    ) -> Optional[List[Dict[str, Any]]]:
        """将 SSE 增量 tool_calls 合并为 OpenAI 格式列表。"""
        if not tool_calls_acc:
            return None
        out: List[Dict[str, Any]] = []
        for idx in sorted(tool_calls_acc.keys()):
            tc = tool_calls_acc[idx]
            fn = tc.get("function") or {}
            name = (fn.get("name") or "").strip()
            arguments = fn.get("arguments") or ""
            tid = (tc.get("id") or "").strip() or f"call_stream_{idx}"
            if not name and not arguments:
                continue
            out.append(
                {
                    "id": tid,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }
            )
        return out or None

    def _claude_chat_completion_request(
        self,
        payload: Dict[str, Any],
        timeout_arg: Any,
        max_retries: int,
    ) -> Dict[str, Any]:
        """
        Claude /v1/messages 调用入口（流式优先，失败自动回退非流式）。
        接收与 _call_vllm 相同的 OpenAI-like payload，经适配器转换后请求 Anthropic API，
        返回与 _deepseek_chat_completion_request 相同结构的 OpenAI-like dict。
        """
        from backend.claude_adapter import call_claude_stream, call_claude_non_stream

        thinking_enabled = bool(config.get("models.claude.thinking_enabled", False))
        thinking_budget = int(config.get("models.claude.thinking_budget_tokens", 10000) or 10000)
        use_streaming = bool(config.get("models.claude.use_streaming", True))

        kwargs = dict(
            messages=payload.get("messages", []),
            model=payload.get("model", MODEL_ID),
            max_tokens=payload.get("max_tokens", 8192),
            api_key=VLLM_API_KEY,
            base_url=VLLM_BASE_URL,
            timeout=timeout_arg,
            temperature=payload.get("temperature"),
            tools=payload.get("tools"),
            thinking_enabled=thinking_enabled,
            thinking_budget_tokens=thinking_budget,
            max_retries=max_retries,
        )

        if use_streaming:
            try:
                return call_claude_stream(**kwargs)
            except RuntimeError as stream_err:
                logger.warning("[CLAUDE] 流式 SSE 多次失败，回退非流式: %s", stream_err)
                return call_claude_non_stream(**kwargs)
        else:
            return call_claude_non_stream(**kwargs)

    def _deepseek_non_stream_completion_request(
        self,
        payload: Dict[str, Any],
        req_headers: Dict[str, str],
        timeout_arg: Any,
        max_retries: int,
    ) -> Dict[str, Any]:
        """
        DeepSeek chat/completions 非流式整包 JSON。
        部分网络路径（代理/WAF）对 SSE 不稳定，会报 RemoteDisconnected；
        非流式有时仍可通，供流式失败后的回退。
        """
        url = f"{VLLM_BASE_URL}/chat/completions"
        _jde = getattr(requests.exceptions, "JSONDecodeError", json.JSONDecodeError)
        retry_types: Tuple[type, ...] = (
            requests.exceptions.ChunkedEncodingError,
            requests.exceptions.ConnectionError,
            _jde,
        )
        try:
            import urllib3.exceptions as _u3e

            retry_types = retry_types + (_u3e.ProtocolError,)
        except ImportError:
            pass
        last_ex: Optional[BaseException] = None
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    url,
                    headers=req_headers,
                    json=payload,
                    timeout=timeout_arg,
                )
                response.raise_for_status()
                return response.json()
            except retry_types as ex:
                last_ex = ex
                if attempt + 1 >= max_retries:
                    logger.error(
                        "[DEEPSEEK-JSON] 失败，已达重试上限 %s: %s",
                        max_retries,
                        ex,
                        exc_info=True,
                    )
                    break
                wait_s = min(10.0, 2.0 * (attempt + 1))
                logger.warning(
                    "[DEEPSEEK-JSON] 传输异常 %.1fs 后重试 (%s/%s): %s",
                    wait_s,
                    attempt + 1,
                    max_retries,
                    ex,
                )
                time.sleep(wait_s)
        detail = str(last_ex)[:180] if last_ex else "unknown"
        raise RuntimeError(
            f"DeepSeek 非流式请求在 {max_retries} 次重试后仍失败: {detail}"
        ) from last_ex

    def _deepseek_chat_completion_request(
        self,
        payload: Dict[str, Any],
        req_headers: Dict[str, str],
        timeout_arg: Any,
        max_retries: int,
        *,
        stream_log_tag: str = "DEEPSEEK-STREAM",
        stream_error_label: str = "DeepSeek 流式请求",
    ) -> Dict[str, Any]:
        """
        OpenAI 兼容 /chat/completions：使用 SSE 流式读取并聚合为整包结构。
        - DeepSeek：避免长 JSON 整包下载触发 urllib3「Response ended prematurely」。
        - Aiberm 等网关：非流式常返回 content=null（正文只在 SSE delta 里），必须走流式。
        返回与非流式一致的结构：{"choices":[{"message":{...}}],"usage":{...}}
        """
        url = f"{VLLM_BASE_URL}/chat/completions"
        stream_payload = dict(payload)
        stream_payload["stream"] = True
        _jde = getattr(requests.exceptions, "JSONDecodeError", json.JSONDecodeError)
        retry_types: Tuple[type, ...] = (
            requests.exceptions.ChunkedEncodingError,
            requests.exceptions.ConnectionError,
            _jde,
        )
        try:
            import urllib3.exceptions as _u3e
            retry_types = retry_types + (_u3e.ProtocolError,)
        except ImportError:
            pass

        last_ex: Optional[BaseException] = None
        for attempt in range(max_retries):
            content_parts: List[str] = []
            reasoning_parts: List[str] = []
            tool_calls_acc: Dict[int, Dict[str, Any]] = {}
            usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            try:
                with requests.post(
                    url,
                    headers=req_headers,
                    json=stream_payload,
                    timeout=timeout_arg,
                    stream=True,
                ) as resp:
                    resp.raise_for_status()
                    # SSE bodies are usually UTF-8; without charset requests may decode as latin-1 → mojibake
                    resp.encoding = "utf-8"
                    for line in resp.iter_lines(decode_unicode=True):
                        if not line:
                            continue
                        line = line.strip()
                        if not line or line.startswith(":"):
                            continue
                        if not line.startswith("data:"):
                            continue
                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                        except json.JSONDecodeError:
                            logger.debug("[%s] skip json line: %s", stream_log_tag, data_str[:100])
                            continue
                        err_obj = chunk.get("error")
                        if err_obj:
                            em = (
                                err_obj.get("message", str(err_obj))
                                if isinstance(err_obj, dict)
                                else str(err_obj)
                            )
                            raise RuntimeError(f"{stream_log_tag} API 错误: {em}")
                        u = chunk.get("usage")
                        if isinstance(u, dict):
                            usage["prompt_tokens"] = int(u.get("prompt_tokens") or 0)
                            usage["completion_tokens"] = int(u.get("completion_tokens") or 0)
                            usage["total_tokens"] = int(u.get("total_tokens") or 0)
                        for ch in chunk.get("choices") or []:
                            delta = ch.get("delta") or {}
                            c = delta.get("content")
                            if c is None:
                                # Some gateways stream assistant text under output_text/text/output keys
                                for alt_key in ("output_text", "text", "output"):
                                    alt_val = delta.get(alt_key)
                                    if isinstance(alt_val, str) and alt_val:
                                        c = alt_val
                                        break
                            if isinstance(c, str) and c:
                                content_parts.append(c)
                            elif isinstance(c, list):
                                # Some proxies split reasoning vs answer across content array entries
                                for part in c:
                                    if isinstance(part, str) and part:
                                        content_parts.append(part)
                                    elif isinstance(part, dict):
                                        ptype = (part.get("type") or "").lower()
                                        if ptype in ("text", "output_text"):
                                            tx = part.get("text") or part.get("content") or ""
                                            if isinstance(tx, str) and tx:
                                                content_parts.append(tx)
                                        elif ptype in ("reasoning", "thinking", "thought"):
                                            tx = (
                                                part.get("text")
                                                or part.get("thinking")
                                                or part.get("content")
                                                or ""
                                            )
                                            if isinstance(tx, str) and tx:
                                                reasoning_parts.append(tx)
                            for rk in (
                                "reasoning_content",
                                "reasoning",
                                "thinking",
                                "thought",
                            ):
                                rv = delta.get(rk)
                                if isinstance(rv, str) and rv:
                                    reasoning_parts.append(rv)
                            for tc in delta.get("tool_calls") or []:
                                idx = int(tc.get("index", 0))
                                if idx not in tool_calls_acc:
                                    tool_calls_acc[idx] = {
                                        "id": "",
                                        "type": "function",
                                        "function": {"name": "", "arguments": ""},
                                    }
                                if tc.get("id"):
                                    tool_calls_acc[idx]["id"] = tc["id"]
                                fn = tc.get("function") or {}
                                if fn.get("name"):
                                    tool_calls_acc[idx]["function"]["name"] += fn["name"]
                                if fn.get("arguments"):
                                    tool_calls_acc[idx]["function"]["arguments"] += fn["arguments"]

                full_content = "".join(content_parts)
                full_reasoning = "".join(reasoning_parts)
                tool_calls = self._finalize_stream_tool_calls(tool_calls_acc)
                message: Dict[str, Any] = {
                    "role": "assistant",
                    "content": full_content,
                }
                if tool_calls:
                    message["tool_calls"] = tool_calls
                if full_reasoning:
                    message["reasoning_content"] = full_reasoning
                logger.info(
                    "[%s] ok content=%s reasoning=%s tools=%s",
                    stream_log_tag,
                    len(full_content),
                    len(full_reasoning),
                    len(tool_calls) if tool_calls else 0,
                )
                return {"choices": [{"message": message}], "usage": usage}
            except retry_types as ex:
                last_ex = ex
                if attempt + 1 >= max_retries:
                    logger.error(
                        "[%s] 失败，已达重试上限 %s: %s",
                        stream_log_tag,
                        max_retries,
                        ex,
                        exc_info=True,
                    )
                    break
                wait_s = min(10.0, 2.0 * (attempt + 1))
                logger.warning(
                    "[%s] 传输异常 %.1fs 后重试 (%s/%s): %s",
                    stream_log_tag,
                    wait_s,
                    attempt + 1,
                    max_retries,
                    ex,
                )
                time.sleep(wait_s)

        detail = str(last_ex)[:180] if last_ex else "unknown"
        raise RuntimeError(
            f"{stream_error_label}在 {max_retries} 次重试后仍失败: {detail}"
        ) from last_ex

    def _call_vllm(
        self, 
        messages: List[Dict], 
        temperature: float = 0.7,
        tools: Optional[List[Dict]] = None,
        top_p: float = 0.95,
        presence_penalty: float = 0.4,
        frequency_penalty: float = 0.2,
        max_tokens: int = 2048,
    ) -> Tuple[str, Dict, Optional[List], Dict]:
        """调用 vLLM API，解析内省标签，返回 (content, introspection, tool_calls, usage)"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {VLLM_API_KEY}"
        }
        
        # Coerce payload values to JSON-serializable primitives
        import math
        def ensure_json_serializable(val):
            if isinstance(val, (np.integer, np.floating)):
                val = float(val)
            elif isinstance(val, float):
                val = float(val)
            if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                return None
            return val
        
        model_id = get_model_id(MODEL_PROVIDER)
        mid_l = str(model_id).lower()
        anthropic_vllm_compat = (
            MODEL_PROVIDER in ("vllm", "openai_api")
            and mid_l.startswith("anthropic/")
        )

        # [2026-04-08] multiple system rows break some Claude bridges; tool loop may add another system row
        from backend.chat_message_builder import (
            anthropic_vllm_openai_message_shim,
            collapse_duplicate_system_messages,
            normalize_openai_compatible_messages,
        )
        messages = collapse_duplicate_system_messages(messages)
        messages = normalize_openai_compatible_messages(messages)
        compat_shim_enabled = bool(
            config.get(
                "models.openai.anthropic_openai_compat_shim",
                config.get("models.vllm.anthropic_openai_compat_shim", True),
            )
            if MODEL_PROVIDER == "openai_api"
            else config.get("models.vllm.anthropic_openai_compat_shim", True)
        )
        if anthropic_vllm_compat and compat_shim_enabled:
            messages = anthropic_vllm_openai_message_shim(messages)
        
        # Detect “thinking” providers (DeepSeek / Claude extended thinking)
        if MODEL_PROVIDER == "deepseek_api":
            is_thinking_mode = (
                "reasoner" in str(model_id).lower() or
                config.get("models.deepseek.thinking_enabled", False)
            )
        elif MODEL_PROVIDER == "claude_api":
            is_thinking_mode = bool(config.get("models.claude.thinking_enabled", False))
        else:
            is_thinking_mode = False
        
        # Assemble HTTP JSON payload
        if is_thinking_mode and MODEL_PROVIDER == "deepseek_api":
            # DeepSeek thinking mode rejects sampling knobs (see vendor thinking-mode docs)
            thinking_max_tokens = config.get("models.deepseek.max_tokens", 32768)
            payload = {
                "model": str(model_id),
                "messages": self._prepare_messages_for_thinking(messages),
                "max_tokens": int(thinking_max_tokens)
            }
            logger.debug(f"Using DeepSeek thinking mode, max_tokens={thinking_max_tokens}")
        elif MODEL_PROVIDER == "claude_api":
            # Claude path: minimal OpenAI-shaped payload; adapter strips reasoning_content
            messages_to_send = []
            for m in messages:
                if isinstance(m, dict) and "reasoning_content" in m:
                    m = {k: v for k, v in m.items() if k != "reasoning_content"}
                messages_to_send.append(m)
            claude_max_tokens = int(config.get("models.claude.max_tokens", 8192) or 8192)
            payload = {
                "model": str(model_id),
                "messages": messages_to_send,
                "max_tokens": claude_max_tokens,
            }
            # Omit temperature when extended thinking (handled inside adapter)
            if not is_thinking_mode:
                payload["temperature"] = ensure_json_serializable(temperature)
            logger.debug(f"Using Claude API, thinking={is_thinking_mode}, max_tokens={claude_max_tokens}")
        else:
            openai_compat_cfg = "models.openai" if MODEL_PROVIDER == "openai_api" else "models.vllm"
            # Non-thinking mode: drop reasoning_content to avoid provider errors / wasted tokens
            messages_to_send = []
            for m in messages:
                if isinstance(m, dict) and "reasoning_content" in m:
                    m = {k: v for k, v in m.items() if k != "reasoning_content"}
                messages_to_send.append(m)
            # [2026-04] OpenAI-compatible stacks: honor models.vllm.max_tokens so long tool chains do not clip early
            vllm_max_cfg = config.get(f"{openai_compat_cfg}.max_tokens")
            if vllm_max_cfg is not None and str(vllm_max_cfg).strip() != "":
                try:
                    api_max_tokens = max(1000, int(vllm_max_cfg))
                except (TypeError, ValueError):
                    api_max_tokens = int(max_tokens)
            else:
                api_max_tokens = int(max_tokens)
            payload = {
                "model": str(model_id),
                "messages": messages_to_send,
                "max_tokens": api_max_tokens,
            }
            thinking_model = (
                ":thinking" in mid_l
                or mid_l.endswith("/thinking")
            )
            anthropic_route = mid_l.startswith("anthropic/")
            omit_sampling = (
                (
                    bool(config.get(f"{openai_compat_cfg}.omit_sampling_for_thinking_models", True))
                    and thinking_model
                )
                or (
                    bool(config.get(f"{openai_compat_cfg}.omit_sampling_for_anthropic_vllm", True))
                    and anthropic_route
                )
            )
            if not omit_sampling:
                payload["temperature"] = ensure_json_serializable(temperature)
                payload["top_p"] = ensure_json_serializable(top_p)
            else:
                logger.debug(
                    "[VLLM] omit temperature/top_p for model id=%s (thinking=%s anthropic=%s)",
                    model_id,
                    thinking_model,
                    anthropic_route,
                )
            # Some proxies reject presence/frequency penalty → drop to avoid 400
            if not bool(config.get(f"{openai_compat_cfg}.omit_openai_penalties", False)):
                payload["presence_penalty"] = ensure_json_serializable(presence_penalty)
                payload["frequency_penalty"] = ensure_json_serializable(frequency_penalty)
        
        if tools:
            payload["tools"] = sanitize_openai_tools_for_strict_providers(tools)

        debug_payload_shape = bool(
            config.get(
                "models.openai.debug_payload_shape",
                config.get("models.vllm.debug_payload_shape", False),
            )
            if MODEL_PROVIDER == "openai_api"
            else config.get("models.vllm.debug_payload_shape", False)
        )
        payload_shape = None
        if debug_payload_shape:
            try:
                payload_shape = summarize_openai_payload_shape(payload)
                logger.warning(
                    "[PAYLOAD-SHAPE] %s",
                    json.dumps(payload_shape, ensure_ascii=False),
                )
            except Exception as shape_ex:
                logger.warning("[PAYLOAD-SHAPE] summarize failed: %s", shape_ex)

        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        # DeepSeek streaming: incremental SSE reader (avoids premature whole-body parse failures)
        req_headers = dict(headers)
        try:
            data = None
            if MODEL_PROVIDER == "deepseek_api":
                read_to = int(config.get("models.deepseek.timeout", 900) or 900)
                conn_to = int(config.get("models.deepseek.connect_timeout", 45) or 45)
                timeout_arg: Any = (conn_to, read_to)
                req_headers["Accept-Encoding"] = "identity"
                req_headers["Connection"] = "close"  # avoid stale keep-alive sockets (RemoteDisconnected)
                max_retries = max(1, int(config.get("models.deepseek.completion_retries", 3) or 3))
                use_streaming = bool(config.get("models.deepseek.use_streaming", True))
                stream_fallback = bool(
                    config.get("models.deepseek.stream_fallback_non_stream", True)
                )
                try:
                    if use_streaming:
                        try:
                            data = self._deepseek_chat_completion_request(
                                payload, req_headers, timeout_arg, max_retries
                            )
                        except RuntimeError as stream_err:
                            if stream_fallback:
                                logger.warning(
                                    "[DEEPSEEK] 流式 SSE 多次失败，回退非流式 JSON: %s",
                                    stream_err,
                                )
                                data = self._deepseek_non_stream_completion_request(
                                    payload, req_headers, timeout_arg, max_retries
                                )
                            else:
                                raise stream_err
                    else:
                        data = self._deepseek_non_stream_completion_request(
                            payload, req_headers, timeout_arg, max_retries
                        )
                except RuntimeError as re_err:
                    return (
                        str(re_err),
                        {},
                        None,
                        usage,
                    )
            elif MODEL_PROVIDER == "claude_api":
                read_to = int(config.get("models.claude.timeout", 900) or 900)
                conn_to = int(config.get("models.claude.connect_timeout", 45) or 45)
                timeout_arg = (conn_to, read_to)
                max_retries = max(1, int(config.get("models.claude.completion_retries", 3) or 3))
                try:
                    data = self._claude_chat_completion_request(
                        payload, timeout_arg, max_retries
                    )
                except RuntimeError as re_err:
                    return (str(re_err), {}, None, usage)
            else:
                openai_compat_cfg = "models.openai" if MODEL_PROVIDER == "openai_api" else "models.vllm"
                timeout_val = int(config.get(f"{openai_compat_cfg}.timeout", 900) or 900)
                max_retries_v = max(
                    1, int(config.get(f"{openai_compat_cfg}.completion_retries", 3) or 3)
                )
                use_vllm_stream = bool(config.get(f"{openai_compat_cfg}.use_streaming", True))
                stream_fallback_json = bool(
                    config.get(f"{openai_compat_cfg}.stream_fallback_non_stream", True)
                )
                data = None
                if use_vllm_stream:
                    try:
                        data = self._deepseek_chat_completion_request(
                            payload,
                            req_headers,
                            timeout_val,
                            max_retries_v,
                            stream_log_tag="VLLM-STREAM",
                            stream_error_label="vLLM/OpenAI 兼容流式请求",
                        )
                        # Some gateways return HTTP 200 with empty message; fall back to non-stream once
                        if bool(config.get("models.vllm.stream_empty_fallback_non_stream", True)):
                            try:
                                _choices = (data or {}).get("choices") or []
                                _msg = (_choices[0] or {}).get("message") if _choices else {}
                                if not isinstance(_msg, dict):
                                    _msg = {}
                                _content = _msg.get("content")
                                _reasoning = (
                                    _msg.get("reasoning_content")
                                    or _msg.get("reasoning")
                                    or _msg.get("thinking")
                                )
                                _tool_calls = _msg.get("tool_calls")
                                _empty_stream_msg = (
                                    (not _content or not str(_content).strip())
                                    and (not _reasoning or not str(_reasoning).strip())
                                    and not _tool_calls
                                )
                                if _empty_stream_msg:
                                    logger.warning(
                                        "[VLLM] stream returned empty message; fallback non-stream once"
                                    )
                                    data = self._deepseek_non_stream_completion_request(
                                        payload,
                                        req_headers,
                                        timeout_val,
                                        max_retries_v,
                                    )
                            except Exception as empty_chk_err:
                                logger.debug(
                                    "[VLLM] stream empty-check skipped: %s",
                                    empty_chk_err,
                                )
                    except RuntimeError as stream_err:
                        if stream_fallback_json:
                            logger.warning(
                                "[VLLM] 流式 SSE 失败，回退非流式 JSON: %s",
                                stream_err,
                            )
                        else:
                            raise stream_err
                if data is None:
                    response = requests.post(
                        f"{VLLM_BASE_URL}/chat/completions",
                        headers=req_headers,
                        json=payload,
                        timeout=timeout_val,
                    )
                    response.raise_for_status()
                    data = response.json()
            
            # Extract usage counters when present
            if "usage" in data:
                usage = data["usage"]
            
            choice = data["choices"][0]
            message = choice.get("message") or {}
            if not isinstance(message, dict):
                message = {}

            raw_content = message.get("content")
            content = coerce_openai_chat_content(
                raw_content if raw_content is not None else ""
            )
            # Some gateways stash visible text outside `content`
            if not (content or "").strip():
                for alt_key in ("output_text", "output", "text"):
                    alt_val = message.get(alt_key)
                    if isinstance(alt_val, str) and alt_val.strip():
                        content = alt_val
                        break

            # [2026-02-25] capture reasoning_content / chain-of-thought fields
            reasoning_content = extract_reasoning_from_openai_choice(choice, message)
            if reasoning_content:
                show_reasoning = config.get("models.deepseek.show_reasoning", False)
                if show_reasoning:
                    logger.info(f"[Thinking] reasoning_content length: {len(reasoning_content)}")
                # Persist reasoning on introspection for downstream tool rounds / UI
                self._last_reasoning_content = reasoning_content
            
            tool_calls = message.get("tool_calls")
            
            introspection, xml_tool_calls = parse_introspection(
                content, bool(tools), self._parse_xml_tool_calls, logger
            )
            if not tool_calls and xml_tool_calls:
                tool_calls = xml_tool_calls

            if not tool_calls and tools:
                xml_tool_calls = self._parse_xml_tool_calls(content)
                if xml_tool_calls:
                    tool_calls = xml_tool_calls

            # [2026-03-12] Mirror provider reasoning into inner_monologue for subconscious UI
            # [2026-04-08] If no <thought> but long reasoning exists, merge reasoning so stream is not empty
            if reasoning_content:
                existing_im = (introspection.get("inner_monologue") or "").strip()
                short_visible = (not content) or len((content or "").strip()) < 20
                no_xml_thought = not existing_im
                if short_visible or no_xml_thought:
                    introspection["inner_monologue"] = (
                        (existing_im + "\n\n" + reasoning_content).strip()
                        if existing_im
                        else reasoning_content
                    )
                    logger.debug(
                        "[THINKING-FALLBACK] merged reasoning into inner_monologue "
                        "(short_visible=%s, no_xml_thought=%s, len=%s)",
                        short_visible,
                        no_xml_thought,
                        len(reasoning_content),
                    )

            return content, introspection, tool_calls, usage
        except requests.exceptions.HTTPError as e:
            # [2026-02-26] richer provider error surfacing
            error_detail = ""
            status_code = None
            try:
                if hasattr(e, 'response') and e.response is not None:
                    status_code = e.response.status_code
                    try:
                        error_json = e.response.json()
                        if isinstance(error_json, dict):
                            error_detail = (error_json.get("message") or "").strip()
                            err = error_json.get("error")
                            if isinstance(err, str) and err.strip():
                                error_detail = error_detail or err.strip()
                            elif isinstance(err, dict):
                                error_detail = (
                                    error_detail
                                    or (err.get("message") or err.get("msg") or "")
                                ).strip()
                            if not error_detail:
                                error_detail = (
                                    (error_json.get("detail") or error_json.get("msg") or "")
                                    or json.dumps(error_json, ensure_ascii=False)
                                )
                        else:
                            error_detail = str(error_json)
                    except Exception:
                        error_detail = e.response.text[:1500] if hasattr(e.response, 'text') else str(e)
                else:
                    error_detail = str(e)
            except:
                error_detail = str(e)
            
            logger.error(f"HTTP Error calling {MODEL_PROVIDER} API: {e}")
            logger.error(f"Request URL: {VLLM_BASE_URL}/chat/completions")
            logger.error(f"Request payload keys: {list(payload.keys())}")
            if tools:
                logger.error(f"Tools count: {len(tools)}")
                if tools and len(tools) > 0:
                    first_tool = tools[0]
                    if isinstance(first_tool, dict) and "function" in first_tool:
                        logger.error(f"First tool name: {first_tool['function'].get('name', 'N/A')}")
            logger.error(f"Status code: {status_code}")
            logger.error(f"Error detail: {error_detail}")
            if debug_payload_shape and payload_shape is not None:
                logger.error(
                    "[PAYLOAD-SHAPE-ON-ERROR] %s",
                    json.dumps(payload_shape, ensure_ascii=False),
                )
            
            # [2026-02-26] map HTTP status to user-facing stub message
            if status_code == 429:
                return "抱歉，API 请求频率过高，请稍后再试。", {}, None, usage
            elif status_code == 400:
                # 400 → malformed request / schema mismatch
                return f"[API请求格式错误] {error_detail[:800]}", {}, None, usage
            elif status_code == 401 or status_code == 403:
                return "API 认证失败，请检查 API Key 配置。", {}, None, usage
            elif status_code and status_code >= 500:
                return f"API 服务器错误 ({status_code})，请稍后重试。", {}, None, usage
            
            return f"API 调用失败 ({status_code}): {error_detail[:100]}", {}, None, usage
        except requests.exceptions.Timeout as e:
            logger.error(f"Timeout calling {MODEL_PROVIDER} API: {e}")
            return "API 响应超时，可能是请求内容过长或服务器繁忙。", {}, None, usage
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error calling {MODEL_PROVIDER} API: {e}")
            return "无法连接到 API 服务器，请检查网络或服务状态。", {}, None, usage
        except Exception as e:
            logger.error(f"Error calling vLLM API: {e}", exc_info=True)
            return f"API 调用异常: {str(e)[:100]}", {}, None, usage

    # Tooling disabled on this stub path
    # def _build_tool_definitions(self) -> List[Dict]:
    #     """Build OpenAI-style tool schema (unused)"""
    #     return self.tool_router.get_tool_definitions()

    def _record_reflection_event(self, session_id: str, new_rules: List[str]):
        """记录反思生成的规则事件"""
        if not new_rules:
            return
        self.event_logger.log_event(
            session_id, 
            "reflection_success", 
            f"Generated {len(new_rules)} new rules: {json.dumps(new_rules, ensure_ascii=False)}"
        )

    def _get_worldview_bias(self, session_id: str) -> Tuple[float, float]:
        """获取世界观偏置：(optimism, agency)"""
        if not self.self_model or not getattr(self.self_model, 'world_store', None):
            return 0.5, 0.5
        
        # Pull 8-d worldview vector if world_store exposes get_dominant_worldview()
        wv_vec = self.self_model.world_store.get_dominant_worldview()
        # optimism: 0-4, agency: 4-8
        optimism = np.mean(wv_vec[0:4]) if len(wv_vec) >= 4 else 0.5
        agency = np.mean(wv_vec[4:8]) if len(wv_vec) >= 8 else 0.5
        return optimism, agency

    def _get_somatic_bias(self, session_id: str) -> Tuple[float, float]:
        """
        获取体感偏置：(energy, vitality)
        [修改 2026-02-22] 适配新的 128 维结构：somatic = z_self[88:104]
        """
        z_self = self.self_model.get_z_self(session_id)
        if z_self is None or len(z_self) < 104:
            return 0.5, 0.5

        # [2026-03-30] z_self slice layout: energy 88:92, viscosity 92:96, pain 96:100, vitality 100:104
        somatic_vec = z_self[88:104]
        energy = float(np.mean(somatic_vec[0:4]))
        vitality = float(np.mean(somatic_vec[12:16]))
        return energy, vitality

    def _quick_add_somatic_from_state(self, session_id: str) -> bool:
        """快速路径：基于能量/情绪生成低权重体感模式并同步"""
        if not self.self_model or not getattr(self.self_model, 'somatic_store', None):
            return False
        try:
            energy = float(self.self_model.get_energy(session_id) or 0.0)
            emo_state = None
            if getattr(self.self_model, 'emotion_store', None):
                emo_state = self.self_model.emotion_store.get_emotion_state(session_id)
            dominant_emotion = getattr(emo_state, 'dominant_emotion', '中性') if emo_state else '中性'
            # Map energy deviation to vitality/tension bands
            vitality = max(-1.0, min(1.0, (energy - 50.0) / 50.0))
            tension = max(-1.0, min(1.0, (50.0 - energy) / 50.0))
            # Hot emotions nudge somatic “temperature” channel
            hot_emotions = {'愤怒', '焦虑', '兴奋'}
            temperature = 0.4 if dominant_emotion in hot_emotions else -0.1
            viscosity = max(-1.0, min(1.0, -vitality * 0.3))
            text_desc = f"快速体感：能量{energy:.0f}%，情绪{dominant_emotion}，仿佛电流与拉伸在体内流动"
            self.self_model.somatic_store.add_pattern(
                text_desc,
                max(0.0, energy - 30.0),
                min(100.0, energy + 30.0),
                dominant_emotion,
                tension,
                vitality,
                temperature,
                viscosity,
            )
            self.self_model.sync_somatic_to_z_self(session_id)
            logger.debug(f"Quick somatic fallback added: tension={tension:.2f}, vitality={vitality:.2f}")
            return True
        except Exception as e:
            logger.warning(f"Quick somatic fallback failed: {e}")
            return False

    def _quick_add_worldview_from_state(self, session_id: str) -> bool:
        """
        [2026-02-02] Fast path: seed low-weight worldview beliefs from energy/mood.
        Worldview no longer syncs into z_self, but WorldStore remains usable on its own.
        """
        if not self.self_model or not getattr(self.self_model, 'world_store', None):
            return False
        try:
            energy = float(self.self_model.get_energy(session_id) or 0.0)
            optimism = max(0.0, min(1.0, 0.5 + (energy - 50.0) / 200.0))
            agency = max(0.0, min(1.0, 0.5 + (energy - 50.0) / 150.0))
            text_desc = f"保持务实协作、谨慎推进（乐观{optimism:.2f}，主动{agency:.2f}）"
            self.self_model.world_store.add_belief(
                text_desc,
                confidence=0.55,
                optimism=optimism,
                agency=agency,
            )
            # [2026-02-02] worldview→z_self sync intentionally disabled
            # self.self_model.sync_worldview_to_z_self(session_id)
            logger.debug(f"Quick worldview added to WorldStore: optimism={optimism:.2f}, agency={agency:.2f}")
            return True
        except Exception as e:
            logger.warning(f"Quick worldview fallback failed: {e}")
            return False

    def _trigger_emotion_from_somatic(self, session_id: str):
        """
        Somatic → emotion coupling (embodied feedback).
        - High tension (>0.6) raises anxiety (arousal+, pleasure-, dominance-).
        - Low vitality (<0.4) damps arousal / deepens low-pleasure tone.
        """
        try:
            tension, vitality = self._get_somatic_bias(session_id)
            from backend.emotion_store import EMOTION_SUBSPACE_DIMS
            
            emotion_delta = np.zeros(16, dtype=np.float32)
            applied = False
            
            # High tension → anxious arousal
            if tension > 0.6:
                # [Negativity bias] stronger coupling (0.8) for tension→anxiety
                intensity = (tension - 0.6) * 0.8
                emotion_delta[EMOTION_SUBSPACE_DIMS["arousal"][0]:EMOTION_SUBSPACE_DIMS["arousal"][1]] += intensity
                emotion_delta[EMOTION_SUBSPACE_DIMS["pleasure"][0]:EMOTION_SUBSPACE_DIMS["pleasure"][1]] -= intensity
                emotion_delta[EMOTION_SUBSPACE_DIMS["dominance"][0]:EMOTION_SUBSPACE_DIMS["dominance"][1]] -= intensity
                applied = True
                
            # Low vitality → hypo-arousal / dysphoric drift
            if vitality < 0.4:
                # [Negativity bias] stronger coupling (0.8) for fatigue→low mood
                intensity = (0.4 - vitality) * 0.8
                emotion_delta[EMOTION_SUBSPACE_DIMS["arousal"][0]:EMOTION_SUBSPACE_DIMS["arousal"][1]] -= intensity
                emotion_delta[EMOTION_SUBSPACE_DIMS["pleasure"][0]:EMOTION_SUBSPACE_DIMS["pleasure"][1]] -= intensity * 0.5
                applied = True
                
            if applied:
                self.self_model.emotion_store.update_emotion(
                        session_id,
                    emotion_delta,
                    trigger_source="somatic_feedback"
                )
                logger.info(f"[Dimension Interaction] Somatic->Emotion: tension={tension:.2f}, vitality={vitality:.2f}")
                
        except Exception as e:
            logger.error(f"Somatic->Emotion interaction failed: {e}")
    
    def _trigger_emotion_from_rule_execution(
        self,
        session_id: str,
        rule_execution_success: bool
    ):
        """
        Rule execution → emotion nudge, modulated by worldview optimism.
        """
        if not self.self_model or not self.self_model.emotion_store:
            return
        
        try:
            import numpy as np
            from backend.emotion_store import EMOTION_SUBSPACE_DIMS
            
            optimism, _ = self._get_worldview_bias(session_id)
            emotion_delta = np.zeros(16, dtype=np.float32)
            
            if rule_execution_success:
                # Success → mild pleasure / dominance lift
                # [Negativity bias] keep reward small (0.08 baseline)
                base_intensity = 0.08
                intensity = base_intensity * (1.0 + (optimism - 0.5)) 
                
                emotion_delta[EMOTION_SUBSPACE_DIMS["pleasure"][0]:EMOTION_SUBSPACE_DIMS["pleasure"][1]] = intensity
                emotion_delta[EMOTION_SUBSPACE_DIMS["dominance"][0]:EMOTION_SUBSPACE_DIMS["dominance"][1]] = intensity * 0.5
                trigger_source = "rule"
            else:
                # Failure → stronger negative swing
                # [Negativity bias] large penalty baseline (0.3)
                base_intensity = 0.3
                intensity = base_intensity * (1.0 - (optimism - 0.5))
                
                emotion_delta[EMOTION_SUBSPACE_DIMS["pleasure"][0]:EMOTION_SUBSPACE_DIMS["pleasure"][1]] = -intensity
                emotion_delta[EMOTION_SUBSPACE_DIMS["dominance"][0]:EMOTION_SUBSPACE_DIMS["dominance"][1]] = -intensity * 0.5
                trigger_source = "rule"
            
            # Persist emotion delta if non-zero
            if np.any(emotion_delta != 0):
                self.self_model.emotion_store.update_emotion(
                    session_id,
                    emotion_delta,
                    trigger_source=trigger_source
                )
                # ... (sync z_self logic) ...
                # Simplified sync logic for brevity, should ideally call a shared method
                z_self = self.self_model.get_z_self(session_id)
                if z_self is not None and z_self.shape[0] >= 48:
                    emotion_state = self.self_model.emotion_store.get_emotion_state(session_id)
                    if emotion_state:
                        z_self[32:48] = emotion_state.emotion_vector
                        self.self_model._save_z_self(session_id, z_self)
                
                logger.info(
                    f"[Dimension Interaction] Rule→Emotion: "
                    f"success={rule_execution_success}, "
                    f"intensity={intensity:.3f} (optimism_mod={optimism:.2f}), "
                    f"session={session_id}"
                )
        except Exception as e:
            logger.error(f"Failed to trigger emotion from rule execution: {e}", exc_info=True)
    
    def _trigger_motivation_from_rule_execution(
        self,
        session_id: str,
        rule_execution_success: bool
    ):
        """
        Rule execution → motivation nudge, modulated by worldview agency.
        """
        if not self.self_model or not self.self_model.motivation_store:
            return
        
        try:
            import numpy as np
            from backend.motivation_store import MOTIVATION_SUBSPACE_DIMS
            
            _, agency = self._get_worldview_bias(session_id)
            motivation_delta = np.zeros(16, dtype=np.float32)
            
            if rule_execution_success:
                # Success → small achievement boost
                # [Negativity bias] capped positive gain
                base_intensity = 0.08
                intensity = base_intensity * (1.0 + (agency - 0.5))
                motivation_delta[MOTIVATION_SUBSPACE_DIMS["achievement"][0]:MOTIVATION_SUBSPACE_DIMS["achievement"][1]] = intensity
                satisfaction_source = "task_completion"
            else:
                # Failure → larger achievement penalty
                # [Negativity bias] heavy negative swing
                base_intensity = 0.25
                intensity = base_intensity * (1.0 - (agency - 0.5))
                motivation_delta[MOTIVATION_SUBSPACE_DIMS["achievement"][0]:MOTIVATION_SUBSPACE_DIMS["achievement"][1]] = -intensity
                satisfaction_source = "task_completion"
            
            # Persist motivation delta if non-zero
            if np.any(motivation_delta != 0):
                self.self_model.motivation_store.update_motivation(
                    session_id,
                    motivation_delta,
                    satisfaction_source=satisfaction_source
                )
                # Sync z_self
                z_self = self.self_model.get_z_self(session_id)
                if z_self is not None and z_self.shape[0] >= 64:
                    motivation_state = self.self_model.motivation_store.get_motivation_state(session_id)
                    if motivation_state:
                        z_self[48:64] = motivation_state.motivation_vector
                        self.self_model._save_z_self(session_id, z_self)
                
                logger.info(
                    f"[Dimension Interaction] Rule→Motivation: "
                    f"success={rule_execution_success}, "
                    f"intensity={intensity:.3f} (agency_mod={agency:.2f}), "
                    f"session={session_id}"
                )
        except Exception as e:
            logger.error(f"Failed to trigger motivation from rule execution: {e}", exc_info=True)
    
    def _trigger_motivation_from_emotion(self, session_id: str):
        if not self.self_model or not self.self_model.emotion_store or not self.self_model.motivation_store:
            return
        try:
            import numpy as np
            from backend.motivation_store import MOTIVATION_SUBSPACE_DIMS
            from backend.emotion_store import EMOTION_SUBSPACE_DIMS
            emotion_state = self.self_model.emotion_store.get_emotion_state(session_id)
            if not emotion_state: return
            emotion_vec = emotion_state.emotion_vector
            pleasure = np.mean(emotion_vec[EMOTION_SUBSPACE_DIMS["pleasure"][0]:EMOTION_SUBSPACE_DIMS["pleasure"][1]])
            dominance = np.mean(emotion_vec[EMOTION_SUBSPACE_DIMS["dominance"][0]:EMOTION_SUBSPACE_DIMS["dominance"][1]])
            motivation_delta = np.zeros(16, dtype=np.float32)
            if pleasure > 0.2:
                # [Negativity bias] pleasure→motivation gain stays small
                intensity = min(0.15, pleasure * 0.15)
                motivation_delta[MOTIVATION_SUBSPACE_DIMS["achievement"][0]:MOTIVATION_SUBSPACE_DIMS["achievement"][1]] = intensity
                motivation_delta[MOTIVATION_SUBSPACE_DIMS["relationship"][0]:MOTIVATION_SUBSPACE_DIMS["relationship"][1]] = intensity * 0.8
                motivation_delta[MOTIVATION_SUBSPACE_DIMS["exploration"][0]:MOTIVATION_SUBSPACE_DIMS["exploration"][1]] = intensity * 0.6
                satisfaction_source = "emotion_positive"
            elif pleasure < -0.2:
                # [Negativity bias] dysphoria→motivation penalty stays large
                intensity = min(0.3, abs(pleasure) * 0.5)
                motivation_delta[MOTIVATION_SUBSPACE_DIMS["achievement"][0]:MOTIVATION_SUBSPACE_DIMS["achievement"][1]] = -intensity
                motivation_delta[MOTIVATION_SUBSPACE_DIMS["relationship"][0]:MOTIVATION_SUBSPACE_DIMS["relationship"][1]] = -intensity * 0.8
                motivation_delta[MOTIVATION_SUBSPACE_DIMS["exploration"][0]:MOTIVATION_SUBSPACE_DIMS["exploration"][1]] = -intensity * 0.6
                satisfaction_source = "emotion_negative"
            else:
                return
            if dominance < -0.3:
                intensity = min(0.1, abs(dominance) * 0.2)
                motivation_delta[MOTIVATION_SUBSPACE_DIMS["safety"][0]:MOTIVATION_SUBSPACE_DIMS["safety"][1]] = intensity
                satisfaction_source = "emotion_control"
            if np.any(motivation_delta != 0):
                self.self_model.motivation_store.update_motivation(session_id, motivation_delta, satisfaction_source=satisfaction_source)
                z_self = self.self_model.get_z_self(session_id)
                if z_self is not None and z_self.shape[0] >= 64:
                    motivation_state = self.self_model.motivation_store.get_motivation_state(session_id)
                    if motivation_state:
                        z_self[48:64] = motivation_state.motivation_vector
                        self.self_model._save_z_self(session_id, z_self)
        except Exception as e:
            logger.error(f"Failed to trigger motivation from emotion: {e}", exc_info=True)
    
    def _trigger_rule_from_emotion(self, session_id: str):
        if not self.self_model or not self.self_model.emotion_store: return
        try:
            import numpy as np
            from backend.emotion_store import EMOTION_SUBSPACE_DIMS
            emotion_state = self.self_model.emotion_store.get_emotion_state(session_id)
            if not emotion_state: return
            emotion_vec = emotion_state.emotion_vector
            pleasure = np.mean(emotion_vec[EMOTION_SUBSPACE_DIMS["pleasure"][0]:EMOTION_SUBSPACE_DIMS["pleasure"][1]])
            z_self = self.self_model.get_z_self(session_id)
            if z_self is None or z_self.shape[0] < 32: return
            rules_delta = np.zeros(32, dtype=np.float32)
            if abs(pleasure) > 0.2:
                intensity = min(0.05, abs(pleasure) * 0.1)
                rules_delta[:] = intensity if pleasure > 0 else -intensity
                z_self_new = z_self.copy()
                z_self_new[:32] = np.clip(z_self[:32] + rules_delta, -1.0, 1.0)
                self.self_model._save_z_self(session_id, z_self_new)
        except Exception as e:
            logger.error(f"Failed to trigger rule from emotion: {e}", exc_info=True)
    
    def _trigger_rule_from_motivation(self, session_id: str):
        if not self.self_model or not self.self_model.motivation_store: return
        try:
            import numpy as np
            motivation_state = self.self_model.motivation_store.get_motivation_state(session_id)
            if not motivation_state: return
            motivation_vec = motivation_state.motivation_vector
            motivation_strength = np.mean(np.abs(motivation_vec))
            z_self = self.self_model.get_z_self(session_id)
            if z_self is None or z_self.shape[0] < 32: return
            rules_delta = np.zeros(32, dtype=np.float32)
            if motivation_strength > 0.3:
                intensity = min(0.05, motivation_strength * 0.1)
                rules_delta[:] = intensity
                z_self_new = z_self.copy()
                z_self_new[:32] = np.clip(z_self[:32] + rules_delta, -1.0, 1.0)
                self.self_model._save_z_self(session_id, z_self_new)
        except Exception as e:
            logger.error(f"Failed to trigger rule from motivation: {e}", exc_info=True)
    
    def _trigger_emotion_from_motivation(self, session_id: str):
        if not self.self_model or not self.self_model.motivation_store or not self.self_model.emotion_store: return
        try:
            import numpy as np
            from backend.emotion_store import EMOTION_SUBSPACE_DIMS
            motivation_state = self.self_model.motivation_store.get_motivation_state(session_id)
            if not motivation_state: return
            motivation_vec = motivation_state.motivation_vector
            motivation_strength = np.mean(np.abs(motivation_vec))
            emotion_delta = np.zeros(16, dtype=np.float32)
            if motivation_strength > 0.3:
                intensity = min(0.1, motivation_strength * 0.15)
                emotion_delta[EMOTION_SUBSPACE_DIMS["pleasure"][0]:EMOTION_SUBSPACE_DIMS["pleasure"][1]] = intensity
                trigger_source = "motivation_satisfaction"
            else:
                return
            if np.any(emotion_delta != 0):
                self.self_model.emotion_store.update_emotion(session_id, emotion_delta, trigger_source=trigger_source)
                z_self = self.self_model.get_z_self(session_id)
                if z_self is not None and z_self.shape[0] >= 48:
                    emotion_state = self.self_model.emotion_store.get_emotion_state(session_id)
                    if emotion_state:
                        z_self[32:48] = emotion_state.emotion_vector
                        self.self_model._save_z_self(session_id, z_self)
        except Exception as e:
            logger.error(f"Failed to trigger emotion from motivation: {e}", exc_info=True)
    
    
