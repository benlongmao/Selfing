import os
import yaml
from typing import Any, Dict, Optional

# Load ``.env`` into the process environment (project root).
from dotenv import load_dotenv
load_dotenv()

class Config:
    _instance = None
    _config_data = {}

    # Env var name → dotted config path (keeps ``.env`` flat keys while YAML stays structured).
    ENV_MAPPING = {
        # System
        "AGENT_NAME": "system.agent_name",
        "AGENT_IDENTITY": "system.agent_identity",
        "MODEL_PROVIDER": "system.model_provider",
        "SELF_TICK_INTERVAL": "system.self_tick_interval",
        "INTROSPECTION_FREQ": "system.introspection_freq",
        "SELF_ENABLED": "system.self_enabled",
        "INTROSPECTION_ENABLED": "system.introspection_enabled",
        "DREAMING_ENABLED": "system.dreaming_enabled",
        
        # Models (Default/vLLM)
        "VLLM_BASE_URL": "models.vllm.base_url",
        "MODEL_ID": "models.vllm.model_id",
        "VLLM_MODEL": "models.vllm.model_id",  # legacy alias for MODEL_ID
        "VLLM_API_KEY": "models.vllm.api_key",
        "VLLM_TIMEOUT": "models.vllm.timeout",
        "VLLM_MAX_TOKENS": "models.vllm.max_tokens",
        
        # Models (DeepSeek)
        "DEEPSEEK_BASE_URL": "models.deepseek.base_url",
        "DEEPSEEK_API_KEY": "models.deepseek.api_key",
        "DEEPSEEK_MODEL": "models.deepseek.model_id",
        "DEEPSEEK_TIMEOUT": "models.deepseek.timeout",
        
        # Models (Claude / Anthropic)
        "CLAUDE_API_KEY": "models.claude.api_key",
        "CLAUDE_BASE_URL": "models.claude.base_url",
        "CLAUDE_MODEL": "models.claude.model_id",
        "CLAUDE_LITE_MODEL": "models.claude.model_id_lite",
        "CLAUDE_TIMEOUT": "models.claude.timeout",

        # Models (OpenAI direct)
        "OPENAI_API_KEY": "models.openai.api_key",
        "OPENAI_BASE_URL": "models.openai.base_url",
        "OPENAI_MODEL": "models.openai.model_id",
        "OPENAI_TIMEOUT": "models.openai.timeout",
        
        # Models (Kimi) — commented out; kept for a future provider switch.
        # "KIMI_BASE_URL": "models.kimi.base_url",
        # "KIMI_API_KEY": "models.kimi.api_key",
        # "MOONSHOT_API_KEY": "models.kimi.api_key",  # official Moonshot env name
        # "KIMI_MODEL": "models.kimi.model_id",
        # "KIMI_TIMEOUT": "models.kimi.timeout",
        # "KIMI_DISABLE_THINKING": "models.kimi.disable_thinking",
        
        # Parameters
        "SELF_UPDATE_ALPHA": "parameters.learning.update_alpha",
        "SELF_LATENT_DIM": "parameters.model.latent_dim",
        "SELF_DRIFT_THRESHOLD": "parameters.thresholds.drift",
        "SELF_PROJ_PATH": "system.self_proj_path",
        "DECAY_RATE_CONNECTION": "parameters.homeostasis.decay_rate_connection",
        "DECAY_RATE_NOVELTY": "parameters.homeostasis.decay_rate_novelty",
        
        # Chat agent loops (see also ``parameters.chat.max_*`` in YAML)
        "CHAT_MAX_MULTI_TURNS": "parameters.chat.max_multi_turns",
        "CHAT_MAX_TOOL_TURNS": "parameters.chat.max_tool_turns",
        "CHAT_FRONTEND_FETCH_TIMEOUT_MS": "parameters.chat.frontend_fetch_timeout_ms",

        # Agent evolution (repo-wide tools; see config/settings.yaml agent_evolution)
        "S_AGENT_EVOLUTION_ENABLED": "agent_evolution.enabled",

        # Feature Flags
        "DRIFT_MONITOR_ENABLED": "system.drift_monitor_enabled",
        "META_RULE_ENABLED": "system.meta_rule_enabled",
        "COMPRESSION_ENABLED": "system.compression_enabled",
        "CONSISTENCY_CHECK_ENABLED": "system.consistency_check_enabled",
        "SELF_TICK_EVIDENCE_WINDOW": "parameters.self_tick_evidence_window",
        "PROMOTION_ENABLED": "system.promotion_enabled",
        "PROMOTION_CRON_MINUTES": "system.promotion_cron_minutes",
        "SPONTANEOUS_ACTION_ENABLED": "system.spontaneous_action_enabled",
        "SPONTANEOUS_CHECK_INTERVAL_S": "system.spontaneous_check_interval",
        
        # Heartbeat (HEARTBEAT.md)
        "HEARTBEAT_ENABLED": "system.heartbeat_enabled",
        "HEARTBEAT_INTERVAL": "system.heartbeat_interval",
        
        # Reflection Parameters
        "REFLECTION_ENABLED": "system.reflection_enabled",
        "REFLECTION_MIN_EVIDENCE": "parameters.thresholds.reflection_min_evidence",  # default overridden to 1 in code
        "REFLECTION_MIN_SIM": "parameters.thresholds.reflection_min_sim",
        "REFLECTION_JUDGE_ENABLED": "system.reflection_judge_enabled",
        "REFLECTION_MIN_ALIGNMENT": "parameters.thresholds.reflection_min_alignment",
        "REFLECTION_MIN_SAFETY": "parameters.thresholds.reflection_min_safety",
        "REFLECTION_MIN_INTERVAL_TURNS": "parameters.thresholds.reflection_min_interval_turns",
        "REFLECTION_MAX_RULES": "parameters.thresholds.reflection_max_rules",
        "REFLECTION_BREAKTHROUGH_SIM_THRESHOLD": "parameters.thresholds.reflection_breakthrough_sim_threshold",
        "REFLECTION_BREAKTHROUGH_RATIO": "parameters.thresholds.reflection_breakthrough_ratio",
        "REFLECTION_IRRATIONAL_SIM_THRESHOLD": "parameters.thresholds.reflection_irrational_sim_threshold",
        
        # Safety (Runtime Output Filter)
        "SAFETY_OUTPUT_FILTER_ENABLED": "system.safety_output_filter_enabled",

        # Mind Wandering (Quota / Rate Limit)
        "MIND_WANDERING_CRYSTALLIZE_ENABLED": "system.mind_wandering_crystallize_enabled",
        "MIND_WANDERING_MAX_PER_DAY": "system.mind_wandering_max_per_day",
        "MIND_WANDERING_MIN_INTERVAL_MINUTES": "system.mind_wandering_min_interval_minutes",
        
        # Pain System
        "PAIN_DISCOMFORT": "parameters.thresholds.pain_discomfort",
        "PAIN_SUFFERING": "parameters.thresholds.pain_suffering",
        "PAIN_AGONY": "parameters.thresholds.pain_agony",
        
        # Multi-level drift thresholds (UI / monitor bands)
        "DRIFT_THRESHOLD_NORMAL": "parameters.thresholds.drift_normal",  # e.g. 0.05 — normal
        "DRIFT_THRESHOLD_WARNING": "parameters.thresholds.drift_warning",  # e.g. 0.10 — warning
        "DRIFT_THRESHOLD_ATTENTION": "parameters.thresholds.drift_attention",  # e.g. 0.15 — attention
        "DRIFT_THRESHOLD_ALERT": "parameters.thresholds.drift_alert",  # e.g. 0.25 — alert
        "DRIFT_CUMULATIVE_WINDOW": "parameters.thresholds.drift_cumulative_window",  # cumulative window size
        "DRIFT_CUMULATIVE_THRESHOLD": "parameters.thresholds.drift_cumulative_threshold",  # cumulative threshold
        
        # Event-triggered Self Tick (some keys are legacy aliases → same YAML path)
        "EVENT_TRIGGER_EMOTION_THRESHOLD": "parameters.event_trigger.emotion_intensity",
        "EVENT_TRIGGER_EMOTION_INTENSITY": "parameters.event_trigger.emotion_intensity",  # [FIX 2026-01-25] clearer name
        "EVENT_TRIGGER_EMOTION_VOLATILITY": "parameters.event_trigger.emotion_volatility",  # [2026-01-25] volatility
        "EVENT_TRIGGER_PAIN_THRESHOLD": "parameters.event_trigger.pain_change",
        "EVENT_TRIGGER_PAIN_CHANGE": "parameters.event_trigger.pain_change",  # [FIX 2026-01-25] clearer name
        "EVENT_TRIGGER_PAIN_ABSOLUTE": "parameters.event_trigger.pain_absolute",  # [2026-01-25] absolute pain
        "EVENT_TRIGGER_ENERGY_DROP_THRESHOLD": "parameters.event_trigger.energy_drop",
        "EVENT_TRIGGER_ENERGY_DROP": "parameters.event_trigger.energy_drop",  # [FIX 2026-01-25] clearer name
        "EVENT_TRIGGER_ENERGY_CRITICAL": "parameters.event_trigger.energy_critical",  # [2026-01-25] critical energy
        
        # Database cleanup (2026-01-31)
        "DB_CLEANUP_ENABLED": "db_cleanup.enabled",
        "DB_CLEANUP_RUN_ON_STARTUP": "db_cleanup.run_on_startup",
        "DB_CLEANUP_CHAT_TURNS_KEEP_DAYS": "db_cleanup.chat_turns.keep_days",
        "DB_CLEANUP_PROMPT_LOGS_KEEP_DAYS": "db_cleanup.prompt_logs.keep_days",
    }

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self):
        self._load_yaml()
        self._override_from_env()
        self._apply_provider_logic()

    def _load_yaml(self):
        """Load ``settings.yaml`` into ``_config_data``."""
        # Search a few common locations (repo root / cwd).
        possible_paths = [
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "settings.yaml"),
            "config/settings.yaml",
            "settings.yaml"
        ]
        
        config_path = None
        for p in possible_paths:
            if os.path.exists(p):
                config_path = p
                break
                
        if config_path:
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    self._config_data = yaml.safe_load(f) or {}
            except Exception as e:
                print(f"Warning: Failed to load config file {config_path}: {e}")
                self._config_data = {}
        else:
            print("Warning: No settings.yaml found, using defaults/env only.")
            self._config_data = {}

    def _override_from_env(self):
        """Overlay values from the process environment using ``ENV_MAPPING``."""
        for env_key, config_path in self.ENV_MAPPING.items():
            env_val = os.environ.get(env_key)
            if env_val is not None:
                # Best-effort cast for bool / int / float / str
                val = self._auto_cast(env_val)
                self._set_by_path(config_path, val)

    def _apply_provider_logic(self):
        """
        Mirror the active provider into the slots the app reads most often.

        For ``deepseek_api``, copy ``models.deepseek`` into ``models.vllm`` so existing
        call sites that read the vLLM slot keep working. Claude/OpenAI use their own env keys.
        """
        provider = self.get("system.model_provider")
        
        if provider == "deepseek_api":
            deepseek_conf = self.get("models.deepseek")
            if deepseek_conf:
                self._set_by_path("models.vllm.base_url", deepseek_conf.get("base_url"))
                self._set_by_path("models.vllm.api_key", deepseek_conf.get("api_key"))
                self._set_by_path("models.vllm.model_id", deepseek_conf.get("model_id"))
                self._set_by_path("models.vllm.timeout", deepseek_conf.get("timeout"))
                # Legacy code may read ``VLLM_API_KEY`` directly from the environment.
                if deepseek_conf.get("api_key"):
                    os.environ["VLLM_API_KEY"] = deepseek_conf.get("api_key")
                print(f"Config: Switched to DeepSeek API ({deepseek_conf.get('model_id')})")
        
        elif provider == "claude_api":
            claude_conf = self.get("models.claude")
            if claude_conf and claude_conf.get("api_key"):
                os.environ["CLAUDE_API_KEY"] = claude_conf.get("api_key")
            print(f"Config: Using Claude API ({(claude_conf or {}).get('model_id', 'unknown')})")

        elif provider == "openai_api":
            openai_conf = self.get("models.openai")
            if openai_conf and openai_conf.get("api_key"):
                os.environ["OPENAI_API_KEY"] = openai_conf.get("api_key")
            print(f"Config: Using OpenAI API ({(openai_conf or {}).get('model_id', 'unknown')})")
        
        # Kimi / Moonshot (commented): would mirror ``models.kimi`` into ``models.vllm`` like DeepSeek.
        # if provider == "kimi_api":
        #     ...

    def _auto_cast(self, val: str) -> Any:
        """Coerce a string env value to bool / number when obvious."""
        if val.lower() == "true": return True
        if val.lower() == "false": return False
        try:
            if "." in val:
                return float(val)
            return int(val)
        except ValueError:
            return val

    def _set_by_path(self, path: str, value: Any):
        """Set ``value`` at a dotted path such as ``models.vllm.base_url``."""
        keys = path.split('.')
        current = self._config_data
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
            if not isinstance(current, dict):
                # Path collides with a non-dict leaf — skip overwrite
                return 
        current[keys[-1]] = value

    def get(self, path: str, default: Any = None) -> Any:
        """
        Read a dotted config path, e.g. ``config.get('system.self_tick_interval', 4)``.
        """
        keys = path.split('.')
        value = self._config_data
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

# Global instance
config = Config()
