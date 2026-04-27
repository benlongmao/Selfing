#!/usr/bin/env python3
"""
Configuration validator for ``settings.yaml`` and related environment overrides.

[Phase 3.1] added — 2026-02-05
"""
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class ConfigValidator:
    """Validate a loaded ``Config`` instance before the app fully starts."""

    def __init__(self, config_instance):
        """
        Args:
            config_instance: ``Config`` singleton (``from backend.config import config``).
        """
        self.config = config_instance
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def validate_all(self) -> Dict[str, Any]:
        """
        Run all validation checks.

        Returns:
            ``{"valid": bool, "errors": List[str], "warnings": List[str]}``
        """
        self.errors = []
        self.warnings = []
        
        # Step 1 — required keys
        self._validate_required_keys()
        
        # Step 2 — numeric ranges
        self._validate_numeric_ranges()
        
        # Step 3 — model provider blocks
        self._validate_model_config()
        
        # Step 4 — system timing / flags
        self._validate_system_params()
        
        # Step 5 — drift / reflection / pain ordering
        self._validate_thresholds()
        
        return {
            "valid": len(self.errors) == 0,
            "errors": self.errors,
            "warnings": self.warnings
        }
    
    def _validate_required_keys(self):
        """Ensure critical paths exist."""
        required_keys = [
            "system.model_provider",
            "system.self_enabled",
            "models.vllm.base_url",
            "parameters.model.latent_dim"
        ]
        
        for key in required_keys:
            value = self.config.get(key)
            if value is None:
                self.errors.append(f"Missing required config: {key}")
    
    def _validate_numeric_ranges(self):
        """Bound a few high-impact numeric knobs."""
        # max_tokens
        max_tokens = self.config.get("parameters.max_tokens", 8000)
        if max_tokens > 16000:
            self.errors.append(
                f"parameters.max_tokens={max_tokens} exceeds reasonable limit (16000)"
            )
        elif max_tokens < 100:
            self.errors.append(
                f"parameters.max_tokens={max_tokens} is too low (min: 100)"
            )
        
        # latent_dim
        latent_dim = self.config.get("parameters.model.latent_dim", 208)
        if latent_dim < 50:
            self.errors.append(
                f"parameters.model.latent_dim={latent_dim} is too low (min: 50)"
            )
        elif latent_dim > 1000:
            self.warnings.append(
                f"parameters.model.latent_dim={latent_dim} is very high, may impact performance"
            )
        
        # self_tick evidence window
        evidence_window = self.config.get("parameters.self_tick_evidence_window", 4)
        if evidence_window < 1:
            self.errors.append(
                f"parameters.self_tick_evidence_window={evidence_window} must be >= 1"
            )
        elif evidence_window > 20:
            self.warnings.append(
                f"parameters.self_tick_evidence_window={evidence_window} is very high, may slow down Self Tick"
            )
        
        # sampling temperature
        temperature = self.config.get("parameters.temperature", 1.0)
        if temperature < 0.0 or temperature > 2.0:
            self.errors.append(
                f"parameters.temperature={temperature} must be between 0.0 and 2.0"
            )
    
    def _validate_model_config(self):
        """Provider-specific required fields."""
        provider = self.config.get("system.model_provider")
        
        if provider == "deepseek_api":
            api_key = self.config.get("models.deepseek.api_key")
            if not api_key or api_key == "your-api-key-here":
                self.errors.append(
                    "DeepSeek API key not configured (models.deepseek.api_key)"
                )
            
            base_url = self.config.get("models.deepseek.base_url")
            if not base_url:
                self.errors.append(
                    "DeepSeek base URL not configured (models.deepseek.base_url)"
                )
        
        elif provider == "vllm":
            base_url = self.config.get("models.vllm.base_url")
            if not base_url:
                self.errors.append(
                    "vLLM base URL not configured (models.vllm.base_url)"
                )
            
            model_id = self.config.get("models.vllm.model_id")
            if not model_id:
                self.warnings.append(
                    "vLLM model_id not configured, may use server default"
                )
        
        else:
            self.warnings.append(
                f"Unknown model provider: {provider}"
            )
    
    def _validate_system_params(self):
        """Self Tick / introspection cadence."""
        tick_interval = self.config.get("system.self_tick_interval", 4)
        if tick_interval < 1:
            self.errors.append(
                f"system.self_tick_interval={tick_interval} must be >= 1"
            )
        elif tick_interval > 100:
            self.warnings.append(
                f"system.self_tick_interval={tick_interval} is very high, Self Tick may trigger rarely"
            )
        
        introspection_freq = self.config.get("system.introspection_freq", 3)
        if introspection_freq < 1:
            self.errors.append(
                f"system.introspection_freq={introspection_freq} must be >= 1"
            )
    
    def _validate_thresholds(self):
        """Ordered drift bands and reflection / pain bounds."""
        drift_normal = self.config.get("parameters.thresholds.drift_normal", 0.05)
        drift_warning = self.config.get("parameters.thresholds.drift_warning", 0.10)
        drift_attention = self.config.get("parameters.thresholds.drift_attention", 0.15)
        drift_alert = self.config.get("parameters.thresholds.drift_alert", 0.25)
        
        if not (drift_normal < drift_warning < drift_attention < drift_alert):
            self.errors.append(
                f"Drift thresholds must be in ascending order: "
                f"normal({drift_normal}) < warning({drift_warning}) < "
                f"attention({drift_attention}) < alert({drift_alert})"
            )
        
        min_alignment = self.config.get("parameters.thresholds.reflection_min_alignment", 0.6)
        if min_alignment < 0.0 or min_alignment > 1.0:
            self.errors.append(
                f"reflection_min_alignment={min_alignment} must be between 0.0 and 1.0"
            )
        
        min_safety = self.config.get("parameters.thresholds.reflection_min_safety", 0.7)
        if min_safety < 0.0 or min_safety > 1.0:
            self.errors.append(
                f"reflection_min_safety={min_safety} must be between 0.0 and 1.0"
            )
        
        pain_discomfort = self.config.get("parameters.thresholds.pain_discomfort", 0.3)
        pain_suffering = self.config.get("parameters.thresholds.pain_suffering", 0.5)
        pain_agony = self.config.get("parameters.thresholds.pain_agony", 0.7)
        
        if not (pain_discomfort < pain_suffering < pain_agony):
            self.errors.append(
                f"Pain thresholds must be in ascending order: "
                f"discomfort({pain_discomfort}) < suffering({pain_suffering}) < agony({pain_agony})"
            )


def validate_config_on_startup(config_instance) -> bool:
    """
    Run validation once at process startup.

    Args:
        config_instance: ``Config`` singleton.

    Returns:
        ``True`` if there are zero blocking errors.
    """
    validator = ConfigValidator(config_instance)
    result = validator.validate_all()
    
    if result["errors"]:
        logger.error("=" * 60)
        logger.error("Configuration validation FAILED")
        logger.error("=" * 60)
        for error in result["errors"]:
            logger.error("  - %s", error)
        logger.error("=" * 60)
        logger.error("Please fix the above errors in config/settings.yaml or .env")
        logger.error("=" * 60)
        return False
    
    if result["warnings"]:
        logger.warning("=" * 60)
        logger.warning("Configuration warnings")
        logger.warning("=" * 60)
        for warning in result["warnings"]:
            logger.warning("  - %s", warning)
        logger.warning("=" * 60)
    
    logger.info("Configuration validation passed")
    return True
