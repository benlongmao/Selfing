#!/usr/bin/env python3
"""
Claude (Anthropic) API adapter.

Bidirectional conversion between the project’s OpenAI-compatible wire format and
Anthropic’s native ``/v1/messages`` format:

  - Request: ``messages`` + ``tools`` → Anthropic JSON
  - Response: Anthropic JSON → OpenAI-like ``dict`` (including ``reasoning_content``)
  - Streaming: Anthropic SSE → merged OpenAI-like ``dict`` (same shape as non-streaming)

Design: this module does not alter DeepSeek / vLLM call paths; it is only used when
``MODEL_PROVIDER == "claude_api"``.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_BASE_URL = "https://api.anthropic.com"


# --- Request headers ---

def _build_headers(api_key: str) -> Dict[str, str]:
    return {
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
        "Accept-Encoding": "identity",
        "Connection": "close",
    }


# --- Request: OpenAI → Anthropic ---

def convert_messages_to_claude(
    messages: List[Dict[str, Any]],
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Convert OpenAI-style ``messages`` into Anthropic format.

    Rules:
    - ``role=system`` → merged into top-level ``system`` string
    - ``role=tool`` → ``tool_result`` blocks inside a ``user`` message (merged with prior ``user``)
    - ``role=assistant`` with ``tool_calls`` → array of ``content`` blocks
    - ``reasoning_content`` is dropped (Anthropic does not accept historical thinking in history)

    Returns:
        ``(system_prompt, claude_messages)``
    """
    system_parts: List[str] = []
    claude_messages: List[Dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content") or ""

        if role == "system":
            if content:
                system_parts.append(content)
            continue

        if role == "tool":
            # OpenAI tool result → Claude ``tool_result`` block
            tool_result_block = {
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id", ""),
                "content": content,
            }
            # If the last message is already ``user`` with a list body, append; else start a new ``user``
            if claude_messages and claude_messages[-1]["role"] == "user":
                last = claude_messages[-1]
                if isinstance(last["content"], list):
                    last["content"].append(tool_result_block)
                else:
                    # Turn plain-text ``user`` content into a block array, then append
                    last["content"] = [
                        {"type": "text", "text": last["content"]},
                        tool_result_block,
                    ]
            else:
                claude_messages.append({"role": "user", "content": [tool_result_block]})
            continue

        if role == "assistant":
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                # Assistant with tool calls → array of ``content`` blocks
                content_blocks: List[Dict[str, Any]] = []
                if content:
                    content_blocks.append({"type": "text", "text": content})
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    raw_args = fn.get("arguments", "{}")
                    try:
                        input_dict = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    except json.JSONDecodeError:
                        input_dict = {"raw": raw_args}
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tc.get("id", ""),
                        "name": fn.get("name", ""),
                        "input": input_dict,
                    })
                claude_messages.append({"role": "assistant", "content": content_blocks})
            else:
                # Plain assistant message; drop ``reasoning_content`` from history
                claude_messages.append({"role": "assistant", "content": content})
            continue

        if role == "user":
            claude_messages.append({"role": "user", "content": content})
            continue

        # Unknown roles are skipped to avoid Anthropic 400s
        logger.debug("[CLAUDE] Skipping message with unknown role=%s", role)

    return "\n\n".join(system_parts), claude_messages


def convert_tools_to_claude(
    tools: Optional[List[Dict[str, Any]]],
) -> Optional[List[Dict[str, Any]]]:
    """Convert OpenAI ``tools`` definitions to Anthropic tool format."""
    if not tools:
        return None
    result = []
    for t in tools:
        if t.get("type") != "function":
            continue
        fn = t.get("function", {})
        result.append({
            "name": fn.get("name", ""),
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
        })
    return result or None


# --- Response: Anthropic → OpenAI-like ---

def convert_claude_response_to_openai(
    response: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Convert an Anthropic ``/v1/messages`` response into the OpenAI-like shape used elsewhere.

    - ``thinking`` blocks → ``message["reasoning_content"]`` (same field name as DeepSeek)
    - ``tool_use`` blocks → ``message["tool_calls"]`` (``arguments`` as JSON string)
    - ``usage.input_tokens`` / ``output_tokens`` → ``prompt_tokens`` / ``completion_tokens``
    """
    content_blocks = response.get("content") or []
    text_parts: List[str] = []
    thinking_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []

    for block in content_blocks:
        btype = block.get("type", "")
        if btype == "text":
            text = block.get("text", "")
            if text:
                text_parts.append(text)
        elif btype == "thinking":
            thinking = block.get("thinking", "")
            if thinking:
                thinking_parts.append(thinking)
        elif btype == "tool_use":
            tool_calls.append({
                "id": block.get("id", ""),
                "type": "function",
                "function": {
                    "name": block.get("name", ""),
                    "arguments": json.dumps(block.get("input") or {}, ensure_ascii=False),
                },
            })

    message: Dict[str, Any] = {
        "role": "assistant",
        "content": "".join(text_parts),
    }
    if thinking_parts:
        message["reasoning_content"] = "".join(thinking_parts)
    if tool_calls:
        message["tool_calls"] = tool_calls

    usage_raw = response.get("usage") or {}
    in_tok = int(usage_raw.get("input_tokens", 0))
    out_tok = int(usage_raw.get("output_tokens", 0))
    usage = {
        "prompt_tokens": in_tok,
        "completion_tokens": out_tok,
        "total_tokens": in_tok + out_tok,
    }

    return {"choices": [{"message": message}], "usage": usage}


# --- Non-streaming call ---

def call_claude_non_stream(
    messages: List[Dict[str, Any]],
    model: str,
    max_tokens: int,
    api_key: str,
    base_url: str,
    timeout: Any,
    temperature: Optional[float] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    thinking_enabled: bool = False,
    thinking_budget_tokens: int = 10000,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """
    POST Anthropic ``/v1/messages`` (non-streaming); returns an OpenAI-like ``dict``.

    Returns:
        ``{"choices": [{"message": {...}}], "usage": {...}}``
    Raises:
        RuntimeError: retries exhausted
        requests.HTTPError: HTTP 4xx/5xx (re-raised without retry)
    """
    system_prompt, claude_messages = convert_messages_to_claude(messages)
    claude_tools = convert_tools_to_claude(tools)

    payload: Dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": claude_messages,
    }
    if system_prompt:
        payload["system"] = system_prompt
    if claude_tools:
        payload["tools"] = claude_tools
    if thinking_enabled:
        # Extended Thinking: do not send ``temperature`` in the same request
        payload["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget_tokens}
    elif temperature is not None:
        payload["temperature"] = temperature

    url = f"{base_url.rstrip('/')}/v1/messages"
    headers = _build_headers(api_key)

    _jde = getattr(requests.exceptions, "JSONDecodeError", json.JSONDecodeError)
    retry_types: tuple = (requests.ConnectionError, requests.Timeout, ConnectionResetError)
    try:
        import urllib3.exceptions as _u3e
        retry_types = retry_types + (_u3e.ProtocolError,)
    except ImportError:
        pass

    last_ex: Optional[BaseException] = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            resp.raise_for_status()
            return convert_claude_response_to_openai(resp.json())
        except retry_types as ex:
            last_ex = ex
            if attempt + 1 >= max_retries:
                logger.error("[CLAUDE] Still failing after %d retries: %s", max_retries, ex)
                break
            wait = min(10.0, 2.0 * (attempt + 1))
            logger.warning(
                "[CLAUDE] Request error; retrying in %.1fs (%d/%d): %s",
                wait, attempt + 1, max_retries, ex,
            )
            time.sleep(wait)
        except requests.HTTPError as ex:
            logger.error("[CLAUDE] HTTP error: %s", ex)
            raise
        except Exception as ex:
            logger.error("[CLAUDE] Unexpected error: %s", ex, exc_info=True)
            raise

    raise RuntimeError(f"Claude request failed after {max_retries} retries: {last_ex}") from last_ex


# --- Streaming call ---

def call_claude_stream(
    messages: List[Dict[str, Any]],
    model: str,
    max_tokens: int,
    api_key: str,
    base_url: str,
    timeout: Any,
    temperature: Optional[float] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    thinking_enabled: bool = False,
    thinking_budget_tokens: int = 10000,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """
    POST Anthropic ``/v1/messages`` with SSE streaming; merge events into the same OpenAI-like dict.

    Anthropic stream event types (simplified):
      ``message_start`` → initial ``usage.input_tokens``
      ``content_block_start`` → record block type (``text`` / ``thinking`` / ``tool_use``)
      ``content_block_delta`` → ``text_delta`` / ``thinking_delta`` / ``input_json_delta``
      ``message_delta`` → final ``usage.output_tokens``
      ``message_stop`` → end of message

    Returns:
        Same shape as ``call_claude_non_stream``.
    Raises:
        RuntimeError: retries exhausted
    """
    system_prompt, claude_messages = convert_messages_to_claude(messages)
    claude_tools = convert_tools_to_claude(tools)

    payload: Dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": claude_messages,
        "stream": True,
    }
    if system_prompt:
        payload["system"] = system_prompt
    if claude_tools:
        payload["tools"] = claude_tools
    if thinking_enabled:
        payload["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget_tokens}
    elif temperature is not None:
        payload["temperature"] = temperature

    url = f"{base_url.rstrip('/')}/v1/messages"
    headers = _build_headers(api_key)

    _jde = getattr(requests.exceptions, "JSONDecodeError", json.JSONDecodeError)
    retry_types: tuple = (
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
        text_parts: List[str] = []
        thinking_parts: List[str] = []
        # Per block index: accumulated tool JSON / block type
        tool_calls_acc: Dict[int, Dict[str, Any]] = {}
        block_types: Dict[int, str] = {}
        usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        try:
            with requests.post(
                url, headers=headers, json=payload, timeout=timeout, stream=True
            ) as resp:
                resp.raise_for_status()
                resp.encoding = "utf-8"
                for line in resp.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    line = line.strip()
                    # ``event:`` lines only name the type; payload is on the following ``data:`` line
                    if line.startswith("event:"):
                        continue
                    if not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if not data_str:
                        continue
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        logger.debug("[CLAUDE-STREAM] skip malformed json: %s", data_str[:80])
                        continue

                    ctype = chunk.get("type", "")

                    if ctype == "message_start":
                        u = chunk.get("message", {}).get("usage") or {}
                        usage["prompt_tokens"] = int(u.get("input_tokens", 0))

                    elif ctype == "content_block_start":
                        idx = int(chunk.get("index", 0))
                        cb = chunk.get("content_block") or {}
                        btype = cb.get("type", "")
                        block_types[idx] = btype
                        if btype == "tool_use":
                            tool_calls_acc[idx] = {
                                "id": cb.get("id", ""),
                                "name": cb.get("name", ""),
                                "input_json": "",
                            }

                    elif ctype == "content_block_delta":
                        idx = int(chunk.get("index", 0))
                        delta = chunk.get("delta") or {}
                        dtype = delta.get("type", "")
                        if dtype == "text_delta":
                            text_parts.append(delta.get("text", ""))
                        elif dtype == "thinking_delta":
                            thinking_parts.append(delta.get("thinking", ""))
                        elif dtype == "input_json_delta" and idx in tool_calls_acc:
                            tool_calls_acc[idx]["input_json"] += delta.get("partial_json", "")

                    elif ctype == "message_delta":
                        u = chunk.get("usage") or {}
                        out_tok = int(u.get("output_tokens", 0))
                        usage["completion_tokens"] = out_tok
                        usage["total_tokens"] = usage["prompt_tokens"] + out_tok

                    elif ctype == "error":
                        err = chunk.get("error") or {}
                        raise RuntimeError(f"Claude API error: {err.get('message', str(err))}")

            # Assemble ``tool_calls`` list
            tool_calls_list: List[Dict[str, Any]] = []
            for idx in sorted(tool_calls_acc.keys()):
                tc = tool_calls_acc[idx]
                try:
                    input_dict = json.loads(tc["input_json"]) if tc["input_json"] else {}
                except json.JSONDecodeError:
                    input_dict = {"raw": tc["input_json"]}
                tool_calls_list.append({
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(input_dict, ensure_ascii=False),
                    },
                })

            message: Dict[str, Any] = {
                "role": "assistant",
                "content": "".join(text_parts),
            }
            if thinking_parts:
                message["reasoning_content"] = "".join(thinking_parts)
            if tool_calls_list:
                message["tool_calls"] = tool_calls_list

            logger.info(
                "[CLAUDE-STREAM] ok content=%d thinking=%d tools=%d",
                len("".join(text_parts)),
                len("".join(thinking_parts)),
                len(tool_calls_list),
            )
            return {"choices": [{"message": message}], "usage": usage}

        except retry_types as ex:
            last_ex = ex
            if attempt + 1 >= max_retries:
                logger.error(
                    "[CLAUDE-STREAM] Failed after %d retries: %s", max_retries, ex, exc_info=True
                )
                break
            wait = min(10.0, 2.0 * (attempt + 1))
            logger.warning(
                "[CLAUDE-STREAM] Transport error; retrying in %.1fs (%d/%d): %s",
                wait, attempt + 1, max_retries, ex,
            )
            time.sleep(wait)

    raise RuntimeError(f"Claude streaming request failed after {max_retries} retries: {last_ex}") from last_ex
