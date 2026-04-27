#!/usr/bin/env python3
"""
Thin LLM client for autonomous subsystems (reads ``backend.config``).
"""
import os
import requests
import logging

logger = logging.getLogger(__name__)


class SimpleLLMClient:
    """
    Minimal chat-completions wrapper.

    Honors ``system.model_provider`` for ``deepseek_api``, ``claude_api``, or local ``vllm``.
    """

    def __init__(self):
        try:
            from backend.config import config

            provider = config.get("system.model_provider", "vllm")
            self._provider = provider

            if provider == "deepseek_api":
                deepseek_conf = config.get("models.deepseek", {})
                self.base_url = deepseek_conf.get("base_url", "https://api.deepseek.com/v1")
                self.api_key = deepseek_conf.get("api_key", "")
                self.model = deepseek_conf.get("model_id", "deepseek-chat")
                self.timeout = int(deepseek_conf.get("timeout", 300) or 300)
                logger.info(f"SimpleLLMClient: Using DeepSeek API ({self.model})")
            elif provider == "claude_api":
                claude_conf = config.get("models.claude", {})
                self.base_url = claude_conf.get("base_url", "https://api.anthropic.com")
                self.api_key = claude_conf.get("api_key", "")
                self.model = (
                    claude_conf.get("model_id_lite")
                    or claude_conf.get("model_id", "claude-haiku-4-5")
                )
                self.timeout = int(claude_conf.get("timeout", 300) or 300)
                self._thinking_enabled = bool(claude_conf.get("thinking_enabled", False))
                self._thinking_budget = int(claude_conf.get("thinking_budget_tokens", 10000) or 10000)
                logger.info(f"SimpleLLMClient: Using Claude API ({self.model})")
            else:
                vllm_conf = config.get("models.vllm", {})
                self.base_url = vllm_conf.get("base_url", "http://localhost:8000/v1")
                self.api_key = vllm_conf.get("api_key", "")
                self.model = vllm_conf.get("model_id", "deepseek-ai/DeepSeek-V3")
                self.timeout = int(vllm_conf.get("timeout", 300) or 300)
                logger.info(f"SimpleLLMClient: Using vLLM ({self.model})")

        except Exception as e:
            logger.warning(f"SimpleLLMClient: Config load failed ({e}), falling back to env vars")
            self._provider = "vllm"
            self.base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
            self.api_key = os.environ.get("VLLM_API_KEY", "")
            self.model = os.environ.get("MODEL_ID") or os.environ.get("VLLM_MODEL") or "deepseek-ai/DeepSeek-V3"
            self.timeout = int(os.environ.get("VLLM_TIMEOUT", "600"))

    def call(self, prompt: str, temperature: float = 0.7, max_tokens: int = 500) -> dict:
        """
        Returns:
            ``{"content": str, "reasoning_content": str, "success": bool}``
        """
        try:
            if getattr(self, "_provider", "") == "claude_api":
                from backend.claude_adapter import call_claude_non_stream
                thinking_enabled = getattr(self, "_thinking_enabled", False)
                thinking_budget = getattr(self, "_thinking_budget", 10000)
                result = call_claude_non_stream(
                    messages=[{"role": "user", "content": prompt}],
                    model=self.model,
                    max_tokens=max_tokens,
                    api_key=self.api_key,
                    base_url=self.base_url,
                    timeout=getattr(self, "timeout", 300),
                    temperature=None if thinking_enabled else temperature,
                    thinking_enabled=thinking_enabled,
                    thinking_budget_tokens=thinking_budget,
                )
                msg = (result.get("choices") or [{}])[0].get("message") or {}
                return {
                    "content": msg.get("content", ""),
                    "reasoning_content": msg.get("reasoning_content", ""),
                    "success": True,
                }

            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            is_thinking_mode = "reasoner" in self.model.lower()

            if is_thinking_mode:
                payload = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens
                }
            else:
                payload = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }

            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=getattr(self, 'timeout', 300)
            )

            if response.status_code == 200:
                data = response.json()
                message = data["choices"][0]["message"]
                content = message.get("content", "") or ""
                reasoning_content = message.get("reasoning_content", "")
                return {
                    "content": content,
                    "reasoning_content": reasoning_content,
                    "success": True
                }
            else:
                logger.error(f"LLM call failed: {response.status_code}")
                return {"content": "", "reasoning_content": "", "success": False}

        except Exception as e:
            logger.error(f"LLM call error: {e}")
            return {"content": "", "reasoning_content": "", "success": False}
