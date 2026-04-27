#!/usr/bin/env python3
"""
Central non-streaming LLM entry point.

Backend features (reflection, judge, rule_compressor, self_narrative,
meta_rule_learner, …) must call through this module instead of ad-hoc ``requests.post``.

Guarantees:
- ``Connection: close`` headers to avoid stale keep-alive ``RemoteDisconnected`` drops.
- Configurable retries (default 3) with bounded linear backoff.
- Shared timeouts, headers, and structured error logging.
- Provider routing via ``system.model_provider`` (``deepseek_api``, ``claude_api``, or ``vllm``).
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

from backend.config import config

logger = logging.getLogger(__name__)

def _get_provider_config(use_lite: bool = True) -> Tuple[str, str, str, int]:
    """Return ``(base_url, api_key, model_id, timeout_seconds)`` for the active provider."""
    provider = config.get("system.model_provider", "vllm")
    if provider == "deepseek_api":
        base_url = config.get("models.deepseek.base_url", "https://api.deepseek.com/v1")
        api_key = config.get("models.deepseek.api_key", "")
        if use_lite:
            model_id = (
                config.get("models.deepseek.model_id_lite")
                or config.get("models.deepseek.model_id", "deepseek-chat")
            )
        else:
            model_id = config.get("models.deepseek.model_id", "deepseek-chat")
        timeout = int(config.get("models.deepseek.timeout", 900) or 900)
    elif provider == "claude_api":
        base_url = config.get("models.claude.base_url", "https://api.anthropic.com")
        api_key = config.get("models.claude.api_key", "")
        if use_lite:
            model_id = (
                config.get("models.claude.model_id_lite")
                or config.get("models.claude.model_id", "claude-haiku-4-5")
            )
        else:
            model_id = config.get("models.claude.model_id", "claude-opus-4-5")
        timeout = int(config.get("models.claude.timeout", 600) or 600)
    else:
        base_url = config.get("models.vllm.base_url", "http://localhost:8000/v1")
        api_key = os.environ.get("VLLM_API_KEY", "")
        model_id = (
            os.environ.get("MODEL_ID")
            or os.environ.get("VLLM_MODEL")
            or config.get("models.vllm.model_id", "")
        )
        timeout = int(config.get("models.vllm.timeout", 900) or 900)
    return base_url, api_key, model_id, timeout


def _build_headers(api_key: str) -> Dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "Accept-Encoding": "identity",
        "Connection": "close",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def llm_completion(
    messages: List[Dict[str, str]],
    *,
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 500,
    timeout: Optional[int] = None,
    max_retries: int = 3,
    use_lite: bool = True,
) -> Dict[str, Any]:
    """
    Blocking chat completion for helper pipelines (reflection, judge, compressors, …).

    Returns:
        ``{"content": str, "reasoning_content": str, "usage": dict, "success": bool}``
        and ``error`` (str) when ``success`` is false.
    """
    base_url, api_key, default_model, default_timeout = _get_provider_config(use_lite)
    effective_timeout = timeout or default_timeout

    # Anthropic Messages API (non-OpenAI wire format) via ``claude_adapter``.
    provider = config.get("system.model_provider", "vllm")
    if provider == "claude_api":
        try:
            from backend.claude_adapter import call_claude_non_stream
            thinking_enabled = bool(config.get("models.claude.thinking_enabled", False))
            thinking_budget = int(config.get("models.claude.thinking_budget_tokens", 10000) or 10000)
            result = call_claude_non_stream(
                messages=messages,
                model=model or default_model,
                max_tokens=max_tokens,
                api_key=api_key,
                base_url=base_url,
                timeout=effective_timeout,
                temperature=None if thinking_enabled else temperature,
                thinking_enabled=thinking_enabled,
                thinking_budget_tokens=thinking_budget,
                max_retries=max_retries,
            )
            msg = (result.get("choices") or [{}])[0].get("message") or {}
            return {
                "content": (msg.get("content") or "").strip(),
                "reasoning_content": msg.get("reasoning_content", ""),
                "usage": result.get("usage", {}),
                "success": True,
            }
        except Exception as ex:
            logger.error("[llm_api] Claude request failed: %s", ex, exc_info=True)
            return {
                "content": "",
                "reasoning_content": "",
                "usage": {},
                "success": False,
                "error": str(ex),
            }

    # OpenAI-compatible ``/chat/completions`` (DeepSeek hosted or local vLLM).
    url = f"{base_url}/chat/completions"
    headers = _build_headers(api_key)

    is_thinking = "reasoner" in (model or default_model).lower()
    payload: Dict[str, Any] = {
        "model": model or default_model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if not is_thinking:
        payload["temperature"] = temperature

    retry_types = (
        requests.ConnectionError,
        requests.Timeout,
        ConnectionResetError,
    )
    try:
        import urllib3.exceptions as _u3e
        retry_types = retry_types + (_u3e.ProtocolError,)
    except ImportError:
        pass

    last_ex: Optional[BaseException] = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(
                url, headers=headers, json=payload, timeout=effective_timeout
            )
            resp.raise_for_status()
            data = resp.json()
            msg = data["choices"][0].get("message") or {}
            return {
                "content": (msg.get("content") or "").strip(),
                "reasoning_content": msg.get("reasoning_content", ""),
                "usage": data.get("usage", {}),
                "success": True,
            }
        except retry_types as ex:
            last_ex = ex
            if attempt + 1 >= max_retries:
                logger.error(
                    "[llm_api] Still failing after %d retries: %s", max_retries, ex
                )
                break
            wait = min(10.0, 2.0 * (attempt + 1))
            logger.warning(
                "[llm_api] Transient error; retry in %.1fs (%d/%d): %s",
                wait, attempt + 1, max_retries, ex,
            )
            time.sleep(wait)
        except requests.HTTPError as ex:
            logger.error("[llm_api] HTTP error: %s", ex)
            last_ex = ex
            break
        except Exception as ex:
            logger.error("[llm_api] Unexpected error: %s", ex, exc_info=True)
            last_ex = ex
            break

    return {
        "content": "",
        "reasoning_content": "",
        "usage": {},
        "success": False,
        "error": str(last_ex) if last_ex else "unknown",
    }
