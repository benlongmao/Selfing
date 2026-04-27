"""
Message list compressor to curb multi-turn token growth.

Design (2026-02-01):
1. Keep essential context (system prompt + freshest turns).
2. Summarize / truncate older segments.
3. Cut prompt mass (rough target: high five-figure tokens instead of six).

Policy:
- First ``system`` message stays intact.
- Last ``N`` conversational turns stay intact.
- Older ``assistant`` bodies shrink (see ``max_assistant_chars``).
- Older ``tool`` payloads become compact JSON summaries (success/fail + hints).
"""

import json
import logging
from typing import List, Dict, Optional

from backend.chat_message_builder import compress_text_for_llm_history

logger = logging.getLogger(__name__)


def compress_messages(
    messages: List[Dict],
    keep_recent_turns: int = 3,
    max_early_content_chars: int = 200,
    max_tool_result_chars: int = 300,
    max_assistant_chars: Optional[int] = None,
) -> List[Dict]:
    """
    Shrink ``messages`` while preserving the newest turns verbatim.

    Args:
        messages: OpenAI-style chat rows.
        keep_recent_turns: How many recent turns to keep whole (turn ≈ user+assistant or assistant+tool).
        max_early_content_chars: Cap for early ``user`` / non-primary ``system`` text.
        max_tool_result_chars: Cap for summarized ``tool`` JSON blobs.
        max_assistant_chars: Optional larger cap for early ``assistant`` rows (defaults to
            ``max_early_content_chars``). Raising this (e.g. 500) reduces “re-summary loops” because the
            model can still see what it wrote last round.

    Returns:
        A new message list safe to send to the LLM.
    """
    asst_chars = max_assistant_chars if max_assistant_chars is not None else max_early_content_chars
    if not messages:
        return messages
    
    keep_recent_count = keep_recent_turns * 3

    if len(messages) <= keep_recent_count + 1:  # +1 for leading system
        return messages
    
    compressed = []
    
    if messages[0].get("role") == "system":
        compressed.append(messages[0])
        remaining = messages[1:]
    else:
        remaining = messages
    
    if len(remaining) > keep_recent_count:
        early_messages = remaining[:-keep_recent_count]
        recent_messages = remaining[-keep_recent_count:]
    else:
        early_messages = []
        recent_messages = remaining
    
    if early_messages:
        compressed.append({
            "role": "system",
            "content": (
                f"[History digest: compressed view of the first {len(early_messages)} older messages]"
            ),
        })
        
        for msg in early_messages:
            compressed_msg = _compress_single_message(
                msg, 
                max_early_content_chars, 
                max_tool_result_chars,
                max_assistant_chars=asst_chars,
            )
            if compressed_msg:
                compressed.append(compressed_msg)
    
    compressed.extend(recent_messages)

    original_chars = sum(len(str(m.get("content", ""))) for m in messages)
    compressed_chars = sum(len(str(m.get("content", ""))) for m in compressed)
    reduction = (1 - compressed_chars / original_chars) * 100 if original_chars > 0 else 0
    
    logger.info(
        f"[MSG-COMPRESS] {len(messages)} → {len(compressed)} messages, "
        f"chars {original_chars} → {compressed_chars} ({reduction:.1f}% reduction)"
    )
    
    return compressed


def _compress_single_message(
    msg: Dict,
    max_content_chars: int,
    max_tool_chars: int,
    max_assistant_chars: Optional[int] = None,
) -> Optional[Dict]:
    """Return a shrunk copy of one chat row (or ``None`` if unchanged)."""
    role = msg.get("role", "")
    content = msg.get("content", "")
    asst_limit = max_assistant_chars if max_assistant_chars is not None else max_content_chars
    
    if role == "system":
        if len(content) > max_content_chars:
            return {
                "role": "system",
                "content": content[:max_content_chars] + "...[truncated]",
            }
        return msg
    
    elif role == "assistant":
        compressed_content = _truncate_content(content, asst_limit)

        result = {"role": "assistant", "content": compressed_content}
        if "tool_calls" in msg:
            simplified_calls = []
            for tc in msg.get("tool_calls", []):
                simplified_calls.append({
                    "id": tc.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": tc.get("function", {}).get("name", "unknown"),
                        "arguments": "{}",  # drop bulky args to save tokens
                    }
                })
            result["tool_calls"] = simplified_calls
            # DeepSeek-style reasoners require ``reasoning_content`` when ``tool_calls`` exist.
            result["reasoning_content"] = msg.get("reasoning_content", "")
        return result
    
    elif role == "tool":
        return _compress_tool_message(msg, max_tool_chars)

    elif role == "user":
        return {
            "role": "user",
            "content": _truncate_content(content, max_content_chars)
        }
    
    return msg


def _truncate_content(content: str, max_chars: int) -> str:
    """Extractive compression to ``max_chars`` (same helper as ``chat_message_builder``)."""
    return compress_text_for_llm_history(content or "", max_chars)


def _compress_tool_message(msg: Dict, max_chars: int) -> Dict:
    """Shrink structured tool JSON down to a tiny summary blob."""
    content = msg.get("content", "")
    tool_call_id = msg.get("tool_call_id", "")
    
    try:
        data = json.loads(content)

        tool_name = data.get("tool_name", "unknown")
        ok = data.get("ok", False)
        result = data.get("result", {})
        
        if ok:
            summary_parts = [f"✓ {tool_name} ok"]

            if "files" in result:
                files = result.get("files", [])
                summary_parts.append(f"({len(files)} file(s))")
            elif "content" in result:
                content_preview = str(result.get("content", ""))[:100]
                summary_parts.append(f"content: {content_preview}...")
            elif "success" in result:
                summary_parts.append(result.get("message", ""))

            summary = " ".join(summary_parts)
        else:
            error = result.get("error", str(result)[:100])
            summary = f"✗ {tool_name} failed: {error}"

        if len(summary) > max_chars:
            summary = summary[:max_chars] + "..."
        
        compressed_data = {
            "tool_name": tool_name,
            "ok": ok,
            "summary": summary
        }
        
        return {
            "role": "tool",
            "content": json.dumps(compressed_data, ensure_ascii=False),
            "tool_call_id": tool_call_id
        }
        
    except (json.JSONDecodeError, TypeError):
        return {
            "role": "tool",
            "content": content[:max_chars] + "..." if len(content) > max_chars else content,
            "tool_call_id": tool_call_id
        }


def estimate_tokens(messages: List[Dict]) -> int:
    """Very rough token estimate (~1 token per two characters incl. tool_calls JSON)."""
    total_chars = 0
    for msg in messages:
        content = str(msg.get("content", ""))
        total_chars += len(content)
        
        if "tool_calls" in msg:
            total_chars += len(json.dumps(msg.get("tool_calls", []), ensure_ascii=False))
    
    return total_chars // 2


def should_compress(messages: List[Dict], threshold_tokens: int = 10000) -> bool:
    """Return True when the rough token estimate exceeds ``threshold_tokens``."""
    estimated = estimate_tokens(messages)
    return estimated > threshold_tokens
