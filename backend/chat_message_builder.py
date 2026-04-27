#!/usr/bin/env python3
"""
Chat message builder (optimized).

Assembles the full message list for the LLM, including system prompt and history.

[2026-01-16] Smart history compression.
"""
import logging
from typing import Any, List, Dict, Optional

from backend.unified_memory import estimate_history_need_from_query

logger = logging.getLogger(__name__)


def collapse_duplicate_system_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge every ``role=system`` chunk into a single system message (joined with ``\\n\\n---\\n``).

    Anthropic / Claude-via-Aiberm often allow only **one** system block per request; extras yield 400.
    Typical duplicates: main prompt + ``[Earlier conversation summary]`` + scheduler nudges; tool loops may add budget notices.
    """
    if not messages:
        return messages
    parts: List[str] = []
    rest: List[Dict[str, Any]] = []
    n_system = 0
    for m in messages:
        if not isinstance(m, dict):
            rest.append(m)
            continue
        if m.get("role") == "system":
            n_system += 1
            c = m.get("content")
            if isinstance(c, str) and c.strip():
                parts.append(c.strip())
            elif c is not None and not isinstance(c, str):
                s = str(c).strip()
                if s:
                    parts.append(s)
        else:
            rest.append(m)
    if n_system <= 1:
        return messages
    if not parts:
        logger.warning("[MESSAGES] Multiple system roles but no non-empty content; leaving messages unchanged")
        return messages
    logger.info(
        "[MESSAGES] Collapsing %s system message(s) into one (provider compat)",
        n_system,
    )
    merged = "\n\n---\n".join(parts)
    return [{"role": "system", "content": merged}] + rest


def normalize_openai_compatible_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize payloads for OpenAI-style gateways: assistant ``content`` must be a string when
    ``tool_calls`` exist; tool ``content`` must be stringifiable JSON text.
    """
    def _normalize_tool_calls(tcs: Any) -> List[Dict[str, Any]]:
        import json
        if not isinstance(tcs, list):
            return []
        normalized: List[Dict[str, Any]] = []
        for i, tc in enumerate(tcs):
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
            args = fn.get("arguments", "")
            if isinstance(args, (dict, list)):
                args_str = json.dumps(args, ensure_ascii=False)
            elif args is None:
                args_str = "{}"
            else:
                args_str = str(args)
            try:
                json.loads(args_str)
            except Exception:
                args_str = "{}"
            normalized.append(
                {
                    "id": str(tc.get("id") or f"call_norm_{i}"),
                    "type": str(tc.get("type") or "function"),
                    "function": {
                        "name": str(fn.get("name") or ""),
                        "arguments": args_str,
                    },
                }
            )
        return normalized

    out: List[Dict[str, Any]] = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        mm = dict(m)
        role = mm.get("role")
        if role == "assistant":
            if mm.get("tool_calls"):
                mm["tool_calls"] = _normalize_tool_calls(mm.get("tool_calls"))
            if mm.get("tool_calls") and mm.get("content") is None:
                mm["content"] = ""
            elif mm.get("content") is None:
                mm["content"] = ""
        elif role == "tool":
            if mm.get("content") is None:
                mm["content"] = ""
        out.append(mm)
    return out


def anthropic_vllm_openai_message_shim(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Mitigate common 400s when OpenAI-format chats are bridged to Claude (Aiberm / vLLM shims):

    - Drop empty assistant ``content`` when ``tool_calls`` are present (some gateways reject ``""``).
    - If the final message is ``role=tool``, append a tiny synthetic ``user`` turn (protocol text).

    Enable only on Anthropic-style models routed through the OpenAI-compatible path.
    """
    if not messages:
        return messages
    out: List[Dict[str, Any]] = []
    stripped_empty = False
    for m in messages:
        if not isinstance(m, dict):
            out.append(m)
            continue
        mm = dict(m)
        role = mm.get("role")
        if role == "assistant" and mm.get("tool_calls"):
            c = mm.get("content")
            if c is None or (isinstance(c, str) and not c.strip()):
                mm.pop("content", None)
                stripped_empty = True
        out.append(mm)
    if out and out[-1].get("role") == "tool":
        out.append(
            {
                "role": "user",
                "content": (
                    "[System · tool continuation] Above are tool results. Continue reasoning from them, "
                    "issue the next batch of tool_calls, or give the final reply. "
                    "(Protocol placeholder—not end-user small talk.)"
                ),
            }
        )
        logger.debug(
            "[ANTHROPIC-VLLM-SHIM] trailing tool → synthetic user (gateway compat)"
        )
    elif stripped_empty:
        logger.debug(
            "[ANTHROPIC-VLLM-SHIM] omitted empty assistant content where tool_calls present"
        )
    return out


def build_messages(
    system_prompt: str,
    user_input: str,
    history: Optional[List[Dict[str, str]]] = None,
    pineal_broadcast: Optional[str] = None,
    enable_history_compression: bool = True,
    max_history_messages: int = 10,
    max_user_chars: int = 800,
    max_assistant_chars: int = 600,
    max_tool_chars: int = 250,
    last_message_role: str = "user",
) -> List[Dict[str, str]]:
    """
    Build the full chat payload: ``system`` + optional compressed ``history`` + final user turn.

    ``pineal_broadcast`` is accepted for backwards compatibility but ignored.

    Schedulers may pass ``last_message_role="system"``; that is re-encoded as ``role=user`` with a
    ``[System scheduled input]`` prefix so the chat ends with a user message (OpenAI/Claude compat).
    """
    messages = []
    
    messages.append({
        "role": "system",
        "content": system_prompt
    })
    
    if history:
        if enable_history_compression:
            compressed_history = _compress_history(
                history, 
                max_history_messages,
                max_user_chars=max_user_chars,
                max_assistant_chars=max_assistant_chars,
                max_tool_chars=max_tool_chars,
                current_input=user_input,
            )
            messages.extend(compressed_history)
            original_count = len(history)
            compressed_count = len([m for m in compressed_history if m.get("role") != "system"])
            logger.info(
                "[MESSAGES] History: %s -> %s messages (after compression/truncation)",
                original_count,
                compressed_count,
            )
        else:
            processed_history = _process_messages_hybrid(
                history,
                max_user_chars=max_user_chars,
                max_assistant_chars=max_assistant_chars,
                max_tool_chars=max_tool_chars
            )
            messages.extend(processed_history)
            logger.debug("[MESSAGES] History processed: %s messages", len(history))
    
    if last_message_role == "system":
        if (user_input or "").strip():
            messages.append(
                {
                    "role": "user",
                    "content": "[System scheduled input]\n" + user_input.strip(),
                }
            )
            logger.info(
                "[build_messages] system reminder encoded as user + [System scheduled input] prefix (API compat)"
            )
        else:
            logger.warning("[build_messages] empty system reminder, no message appended")
    else:
        messages.append(
            {
                "role": last_message_role,
                "content": user_input,
            }
        )

    return messages


# Heuristic “important line” tokens for smart compression (bilingual; keep CJK triggers).
KEY_PATTERNS = [
    "?",
    "？",
    "1.",
    "2.",
    "3.",
    "4.",
    "5.",
    "•",
    "-",
    "：",
    ":",
    "想",
    "问",
    "能不能",
    "如何",
    "怎么",
    "是什么",
    "为什么",
    "总结",
    "结论",
    "最终",
    "答案",
    "解决",
    "方案",
    "建议",
    "能否",
    "可否",
    "是否",
    "definition",
    "wondering",
    "summarize",
    "summarise",
    "recap",
    "outline",
    "explain",
    "please",
    "help",
    "what",
    "why",
    "how",
    "can you",
    "could you",
    "would you",
    "summary",
    "conclusion",
    "answer",
    "solution",
]

def _smart_compress(text: str, max_chars: int = 300) -> str:
    """
    Keep question/list/conclusion-looking lines; trim narrative filler to ``max_chars``.
    """
    if not text or len(text) <= max_chars:
        return text
    
    lines = text.strip().split('\n')
    
    key_lines = []
    bg_lines = []
    
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue
        
        is_key = any(p in line_stripped for p in KEY_PATTERNS)
        if line_stripped.startswith('```') or line_stripped.startswith('def ') or line_stripped.startswith('class '):
            is_key = True
        
        if is_key:
            key_lines.append(line_stripped)
        else:
            bg_lines.append(line_stripped)
    
    result = []
    char_count = 0
    reserve_for_note = 40
    
    for line in key_lines:
        if char_count + len(line) > max_chars - reserve_for_note:
            remaining = max_chars - reserve_for_note - char_count - 10
            if remaining > 50:
                result.append(line[:remaining] + "...")
            break
        result.append(line)
        char_count += len(line) + 1
    
    if bg_lines and char_count < max_chars - reserve_for_note - 50:
        first_bg = bg_lines[0][:80]
        result.insert(0, first_bg + ("..." if len(bg_lines[0]) > 80 else ""))
    
    omitted_count = len(bg_lines) + len(key_lines) - len(result)
    if omitted_count > 0:
        result.append("[Additional background lines omitted for brevity]")
    
    return '\n'.join(result)


def _line_looks_structured(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if any(p in s for p in KEY_PATTERNS):
        return True
    if s.startswith(("```", "- ", "•", "* ", "1.", "2.", "3.", "4.", "5.")):
        return True
    if s.startswith(("def ", "class ")):
        return True
    return False


def _medium_long_threshold(max_single: int) -> int:
    """Length at which we switch from line-smart compress to head/mid/tail digest."""
    return max(900, int(max_single) * 2)


def compress_text_for_llm_history(text: str, max_chars: int) -> str:
    """
    Fit ``text`` into ``max_chars`` using head + structured mid picks + tail (no ``[omitted N chars]`` placeholders).
    """
    if not text or len(text) <= max_chars:
        return text
    notice = ""
    mc = max_chars
    if mc < 100:
        return text[: max(0, mc - 1)] + "…"

    head_cap = min(360, max(120, mc // 3))
    tail_cap = min(260, max(80, mc // 4))
    mid_cap = max(80, mc - head_cap - tail_cap - 40)

    head = text[:head_cap]
    for sep in ("\n\n", "\n", "。", "！", "？", ".", "!", ";", "；"):
        i = head.rfind(sep)
        if i >= max(20, head_cap // 2):
            head = text[: i + len(sep)]
            break

    tail = text[-tail_cap:] if tail_cap else ""
    if tail:
        for sep in ("\n\n", "\n", "。", "！", "？"):
            j = tail.find(sep)
            if 0 < j < len(tail) // 3:
                tail = tail[j + len(sep) :].lstrip()
                break

    picks: List[str] = []
    used = 0
    for ln in text.split("\n"):
        ln = ln.strip()
        if not ln or not _line_looks_structured(ln):
            continue
        chunk = ln[: min(140, len(ln))]
        if used + len(chunk) + 3 > mid_cap:
            break
        picks.append(chunk)
        used += len(chunk) + 3

    if not picks:
        mid_start = max(0, len(text) // 3)
        raw = text[mid_start : mid_start + min(160, mid_cap)].replace("\n", " ").strip()
        if raw:
            picks.append(f"…{raw}…")

    mid = "\n".join(f"· {p}" for p in picks[:12])
    parts = ["---", head.strip()]
    if mid:
        parts += ["...", mid]
    if tail.strip():
        parts += ["...", tail.strip()]
    out = "\n".join(parts)
    if len(out) > max_chars:
        out = out[: max_chars - 1].rstrip() + "…"
    return out


def _truncate_middle(text: str, head: int = 320, tail: int = 160) -> str:
    """Legacy wrapper around :func:`compress_text_for_llm_history`."""
    if not text:
        return text
    budget = max(200, head + tail + 80)
    if len(text) <= budget:
        return text
    return compress_text_for_llm_history(text, budget)


GREETING_KEYWORDS = {
    "你好",
    "嗨",
    "hi",
    "hello",
    "hey",
    "早",
    "晚安",
    "再见",
    "拜拜",
    "good morning",
    "good evening",
    "good night",
    "bye",
    "good afternoon",
    "how are you",
    "what's up",
    "greetings",
    "nice to meet you",
    "morning",
    "evening",
    "afternoon",
    "yo ",
}
TECHNICAL_KEYWORDS = {
    "代码",
    "函数",
    "报错",
    "error",
    "bug",
    "实现",
    "调试",
    "debug",
    "优化",
    "分析",
    "架构",
    "数据库",
    "api",
    "接口",
    "implement",
    "refactor",
    "stack trace",
    "traceback",
    "compile",
    "typescript",
    "python",
    "function",
    "class ",
    "deploy",
    "kubernetes",
    "docker",
    "latency",
    "memory leak",
    "stack overflow",
    "segmentation fault",
    "unit test",
    "ci/cd",
    "sql",
    "frontend",
    "backend",
    "exception",
    "stacktrace",
}

def _estimate_history_need(user_input: str) -> int:
    """
    Heuristic number of **turns** worth of history (×2 == message count heuristic upstream).

    Short greetings → fewer turns; technical keywords → more.
    """
    if not user_input:
        return 4

    estimated = estimate_history_need_from_query(user_input)
    if estimated:
        return estimated

    input_lower = user_input.lower()

    if any(kw in input_lower for kw in GREETING_KEYWORDS) and len(user_input) < 20:
        return 2

    if any(kw in input_lower for kw in TECHNICAL_KEYWORDS):
        return 5

    return 4


def _assistant_long_threshold(max_assistant_chars: int) -> int:
    """Assistant length where we prefer digest compression over line-smart trim."""
    return max(900, int(max_assistant_chars) * 2)


def _user_long_threshold(max_user_chars: int) -> int:
    return max(900, int(max_user_chars) * 2)


def _compress_history(
    history: List[Dict[str, str]],
    max_messages: int = 5,
    max_user_chars: int = 800,
    max_assistant_chars: int = 600,
    max_tool_chars: int = 250,
    current_input: str = "",
) -> List[Dict[str, str]]:
    """
    Trim ``history`` with dynamic depth, relevance picks for older turns, and hybrid per-message compression.
    """
    if max_messages == 0:
        return []
    
    dynamic_max = _estimate_history_need(current_input)
    effective_max = min(max_messages, dynamic_max * 2)
    
    if len(history) <= effective_max:
        return _process_messages_hybrid(history, max_user_chars, max_assistant_chars, max_tool_chars)
    
    recent_keep = min(dynamic_max * 2, effective_max)
    relevance_slots = max(0, effective_max - recent_keep)
    
    recent_messages = history[-recent_keep:]
    old_messages = history[:-recent_keep]
    
    selected_old = []
    old_for_summary = old_messages
    
    if relevance_slots > 0 and current_input and old_messages:
        input_keywords = set(w for w in current_input.lower().split() if len(w) > 1)
        
        scored = []
        for i, msg in enumerate(old_messages):
            content = (msg.get("content") or "").lower()
            score = sum(1 for kw in input_keywords if kw in content)
            if score > 0:
                scored.append((i, score, msg))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        selected_indices = set()
        for i, score, msg in scored[:relevance_slots]:
            selected_old.append(msg)
            selected_indices.add(i)
        
        old_for_summary = [m for i, m in enumerate(old_messages) if i not in selected_indices]
    
    summary = _generate_history_summary(old_for_summary) if old_for_summary else ""
    
    compressed = []
    
    if summary:
        compressed.append({
            "role": "system",
            "content": f"[Earlier conversation summary]\n{summary}"
        })
    
    if selected_old:
        compressed.extend(_process_messages_hybrid(selected_old, max_user_chars, max_assistant_chars, max_tool_chars))
    
    processed_recent = _process_messages_hybrid(recent_messages, max_user_chars, max_assistant_chars, max_tool_chars)
    compressed.extend(processed_recent)
    
    return compressed


def _process_messages_hybrid(
    messages: List[Dict[str, str]],
    max_user_chars: int = 800,
    max_assistant_chars: int = 600,
    max_tool_chars: int = 250
) -> List[Dict[str, str]]:
    """
    Per-role truncation: keep raw under caps, ``_smart_compress`` in the medium band, digest beyond.
    """
    ast_long = _assistant_long_threshold(max_assistant_chars)
    usr_long = _user_long_threshold(max_user_chars)
    result = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "") or ""
        content_len = len(content)
        
        if role == "assistant":
            if content_len <= max_assistant_chars:
                pass
            elif content_len <= ast_long:
                content = _smart_compress(content, max_assistant_chars)
            else:
                content = compress_text_for_llm_history(content, max_assistant_chars)
                
        elif role == "user":
            if content_len <= max_user_chars:
                pass
            elif content_len <= usr_long:
                content = _smart_compress(content, max_user_chars)
            else:
                content = compress_text_for_llm_history(content, max_user_chars)
                
        elif role == "tool":
            if content_len > max_tool_chars:
                content = compress_text_for_llm_history(content, max_tool_chars)

        out_msg: Dict[str, Any] = {"role": role, "content": content}
        if role == "assistant":
            if msg.get("tool_calls"):
                out_msg["tool_calls"] = msg["tool_calls"]
            rc = msg.get("reasoning_content")
            if isinstance(rc, str) and rc.strip():
                out_msg["reasoning_content"] = rc
        elif role == "tool":
            tcid = msg.get("tool_call_id")
            if tcid:
                out_msg["tool_call_id"] = tcid
            tname = msg.get("name")
            if tname:
                out_msg["name"] = tname
        result.append(out_msg)

    return result


def _truncate_long_messages(
    messages: List[Dict[str, str]],
    max_user_chars: int = 800,
    max_assistant_chars: int = 600,
    max_tool_chars: int = 250
) -> List[Dict[str, str]]:
    """
    Deprecated alias for :func:`_process_messages_hybrid`.
    """
    return _process_messages_hybrid(
        messages, max_user_chars, max_assistant_chars, max_tool_chars
    )


_CONCLUSION_MARKERS = [
    "总结",
    "结论",
    "决定",
    "最终",
    "确认",
    "同意",
    "好的",
    "就这样",
    "方案",
    "选择了",
    "答案是",
    "结果是",
    "因此",
    "所以",
    "综上",
    "总之",
    "最后",
    "in summary",
    "to recap",
    "the answer is",
    "we decided",
    "in conclusion",
    "therefore",
    "final answer",
    "to summarize",
    "in short",
    "decided",
    "agreed",
    "overall",
    "all in all",
    "wrapping up",
]
_EMOTION_MARKERS = [
    "开心",
    "难过",
    "担心",
    "害怕",
    "期待",
    "感动",
    "生气",
    "失望",
    "有趣",
    "感谢",
    "抱歉",
    "遗憾",
    "兴奋",
    "紧张",
    "放松",
    "自豪",
    "孤独",
    "好奇",
    "焦虑",
    "happy",
    "sad",
    "worried",
    "afraid",
    "thanks",
    "sorry",
    "excited",
    "angry",
    "disappointed",
    "grateful",
    "anxious",
    "frustrated",
    "relieved",
]


def _generate_history_summary(messages: List[Dict[str, str]]) -> str:
    """Lightweight English summary: topic snippets + conclusion hits + affect cues."""
    if not messages:
        return ""

    user_count = sum(1 for m in messages if m.get("role") == "user")

    topics = []
    conclusions = []
    emotions = []

    for i, msg in enumerate(messages):
        role = msg.get("role", "")
        content = (msg.get("content") or "").replace("\n", " ").strip()
        if not content:
            continue

        if role == "user":
            preview = content[:40].strip()
            if preview:
                topics.append(f"  • {preview}{'...' if len(content) > 40 else ''}")

        elif role == "assistant" and len(conclusions) < 3:
            for marker in _CONCLUSION_MARKERS:
                if marker in content:
                    idx = content.index(marker)
                    start = max(0, idx - 10)
                    snippet = content[start:start + 60].strip()
                    conclusions.append(f"  → {snippet}{'...' if len(content) > start + 60 else ''}")
                    break

            if not emotions:
                for marker in _EMOTION_MARKERS:
                    if marker in content:
                        idx = content.index(marker)
                        start = max(0, idx - 8)
                        snippet = content[start:start + 40].strip()
                        emotions.append(f"  ∿ {snippet}")
                        break

    if len(topics) > 4:
        topics = topics[:2] + [f"  ... ({len(topics)} topic snippets total)"] + topics[-1:]

    lines = [f"Earlier segment: {user_count} user turn(s)"]
    lines.extend(topics[:5])
    if conclusions:
        lines.append("Key conclusions:")
        lines.extend(conclusions[:3])
    if emotions:
        lines.append("Affective cues:")
        lines.extend(emotions[:2])

    return "\n".join(lines)


def build_messages_simple(
    system_prompt: str,
    user_input: str,
    history: Optional[List[Dict[str, str]]] = None,
    pineal_broadcast: Optional[str] = None
) -> List[Dict[str, str]]:
    """``build_messages`` with ``enable_history_compression=False`` (compat helper)."""
    return build_messages(
        system_prompt=system_prompt,
        user_input=user_input,
        history=history,
        pineal_broadcast=pineal_broadcast,
        enable_history_compression=False
    )


if __name__ == "__main__":
    print("chat_message_builder (optimized) loaded")
    
    test_history = [
        {"role": "user", "content": f"question {i}"} for i in range(1, 16)
    ] + [
        {"role": "assistant", "content": f"answer {i}"} for i in range(1, 16)
    ]
    
    compressed = build_messages(
        system_prompt="You are a helpful assistant.",
        user_input="current question",
        history=test_history,
        enable_history_compression=True,
        max_history_messages=7
    )
    
    print(f"raw history messages: {len(test_history)}")
    print(f"after build_messages (excl. system+last user): {len(compressed) - 2}")
