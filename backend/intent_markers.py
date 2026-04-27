#!/usr/bin/env python3
"""
Single source of truth for multi-turn intent literals and phrases.

[2026-03-24] Weak closers (e.g. “综上所述”) coexist with [S44_CONTINUE]; they must not alone
mean “stop” or “turn complete”. Hard stops and bracket intents still win.

[2026-03-30] Stronger continue detection:
- Plan C: tolerate more bracket/format variants
- Plan B: detect “unfinished” natural-language cues
"""

from typing import Tuple

# Continue markers (main chat path)
S44_CONTINUE_LITERAL = "[S44_CONTINUE]"
S44_COMPLETE_LITERAL = "[S44_COMPLETE]"

# [2026-03-30] Plan C: more tolerated spellings
CONTINUE_VARIANTS: Tuple[str, ...] = (
    "[S44_CONTINUE]",
    "S44_CONTINUE",
    "[S44-CONTINUE]",
    "【S44_CONTINUE】",
    "[CONTINUE]",
)

# [2026-03-30] Plan B: unfinished cues (no explicit marker yet)
UNFINISHED_SIGNALS: Tuple[str, ...] = (
    "让我继续",
    "接下来我要",
    "下一步我会",
    "我还需要",
    "还需要继续",
    "继续分析",
    "继续执行",
    "继续处理",
    "让我接着",
    "我来继续",
    "待我继续",
    "我继续",
    "let me continue",
    "i will continue",
    "next i will",
    "i still need to",
    "need to continue",
    "continue analyzing",
    "continue executing",
    "continue processing",
    "let me pick up where",
    "keep going with",
    "to be continued",
)

# Before multi-turn: bracket-only hits here mean “do not enter multi-turn”
STOP_BRACKETS_BEFORE_MULTI_TURN: Tuple[str, ...] = (
    S44_COMPLETE_LITERAL,
    "[S44_PAUSE]",
    "[S44_TIRED]",
    "[S44_UNCLEAR]",
)

# Weak closers: common in long answers; with CONTINUE they must not block multi-turn
# Keep aligned with intent_driven_executor NATURAL_LANGUAGE_INTENTS[TASK_COMPLETE] (no hard stops)
WEAK_COMPLETE_NATURAL_MARKERS: Tuple[str, ...] = (
    "综上所述",
    "总结一下",
    "最终答案",
    "最终答案是",
    "结论是",
    "基于以上分析",
    "回答你的问题",
    "任务完成",
    "in conclusion",
    "to summarize",
    "summary:",
    "final answer",
    "the answer is",
    "based on the above",
    "task complete",
    "mission complete",
)

# Hard stop: sleep, refusal, end chat — CONTINUE does not override
HARD_STOP_NATURAL_MARKERS: Tuple[str, ...] = (
    "立即停止",
    "停止验证",
    "深度休眠",
    "保持静默",
    "完全静默",
    "进入休眠",
    "停止所有活动",
    "立即停止所有活动",
    "我会保持完全静默",
    "停止所有",
    "不再继续",
    "结束对话",
    "stop immediately",
    "halt verification",
    "deep sleep",
    "stay silent",
    "remain silent",
    "enter sleep mode",
    "stop all activities",
    "stop everything",
    "i will stay silent",
    "stop all",
    "do not continue",
    "end conversation",
)

# Pause / tired / unclear — always treated as stop in visible text (no CONTINUE override)
AGENT_PAUSE_OR_UNCLEAR_NATURAL: Tuple[str, ...] = (
    "让我整理一下思路",
    "我需要暂停整理",
    "让我理一理",
    "我有点累了",
    "能量有点低",
    "需要休息",
    "我不太理解",
    "能否澄清",
    "请问你是指",
    "let me think",
    "i need to pause",
    "sorting my thoughts",
    "i'm a bit tired",
    "low energy",
    "need a break",
    "i don't quite understand",
    "could you clarify",
    "what do you mean by",
)


def visible_has_any(text: str, phrases: Tuple[str, ...]) -> bool:
    if not text:
        return False
    return any(p in text for p in phrases)


def should_block_multi_turn_before_loop(
    intent_visible_text: str,
    intent_full_text: str,
) -> bool:
    """
    Block entering intent multi-turn: bracket stop markers in visible body only,
    or hard-stop natural phrases in visible text.

    Weak closers alone do not block; when paired with [S44_CONTINUE], callers may still continue.

    Bracket markers are not scanned in reasoning: chain-of-thought often mentions
    “should output [S44_COMPLETE]” and would false-positive.
    intent_full_text is kept for API compatibility; not used for bracket matching here.
    """
    _ = intent_full_text
    if any(m in intent_visible_text for m in STOP_BRACKETS_BEFORE_MULTI_TURN):
        return True
    if visible_has_any(intent_visible_text, HARD_STOP_NATURAL_MARKERS):
        return True
    return False


def explain_loop_stop_match(
    loop_visible: str,
    stop_bracket_markers: Tuple[str, ...],
) -> str:
    """
    Debug helper: if loop_has_stop_intent is True, return first matched rule label; else "".
    """
    if not loop_visible:
        return ""
    for m in stop_bracket_markers:
        if m in loop_visible:
            return f"bracket:{m}"
    for p in HARD_STOP_NATURAL_MARKERS:
        if p in loop_visible:
            return f"hard_stop:{p}"
    for p in AGENT_PAUSE_OR_UNCLEAR_NATURAL:
        if p in loop_visible:
            return f"pause_or_unclear:{p}"
    return ""


def loop_has_stop_intent(
    loop_visible: str,
    stop_bracket_markers: Tuple[str, ...],
    loop_full: str = "",
) -> bool:
    """
    Stop inside multi-turn loop: brackets, hard stops, pause-like phrases — **visible body only**.

    Weak closers are not stops. Do not scan reasoning (loop_full) for brackets: mentions of
    [S44_COMPLETE]/[S44_PAUSE] in chain-of-thought would false-trigger safeguards and break
    [S44_CONTINUE] continuations.
    loop_full kept for backward-compatible call sites (may be empty).
    """
    _ = loop_full
    if any(m in loop_visible for m in stop_bracket_markers):
        return True
    if visible_has_any(loop_visible, HARD_STOP_NATURAL_MARKERS):
        return True
    if visible_has_any(loop_visible, AGENT_PAUSE_OR_UNCLEAR_NATURAL):
        return True
    return False


def loop_has_complete_intent(
    loop_visible: str,
    loop_full: str,
    complete_bracket_markers: Tuple[str, ...],
) -> bool:
    """
    Turn complete: treat explicit [S44_COMPLETE] only in visible text (avoid reasoning false positives);
    [S44_CONTINUE] may be read from loop_full (including reasoning); weak closers only if no CONTINUE.
    """
    if any(m in loop_visible for m in complete_bracket_markers):
        return True
    if S44_CONTINUE_LITERAL in loop_full:
        return False
    return visible_has_any(loop_visible, WEAK_COMPLETE_NATURAL_MARKERS)


def basic_multiturn_has_complete(visible: str, full: str) -> bool:
    """Fallback path: same as loop_has_complete_intent; weak closers only when no CONTINUE."""
    if S44_COMPLETE_LITERAL in full:
        return True
    if has_continue_intent(full):
        return False
    return visible_has_any(visible, WEAK_COMPLETE_NATURAL_MARKERS)


# ============================================================
# [2026-03-30] Plans B+C: stronger continue detection
# ============================================================

def has_continue_intent(text: str) -> bool:
    """
    Priority:
    1. Exact [S44_CONTINUE]
    2. Format variants (Plan C)
    3. Unfinished natural-language cues (Plan B), only if no hard stop / pause phrase hit
    """
    if not text:
        return False

    if S44_CONTINUE_LITERAL in text:
        return True

    text_upper = text.upper()
    for variant in CONTINUE_VARIANTS:
        if variant.upper() in text_upper:
            return True

    if not visible_has_any(text, HARD_STOP_NATURAL_MARKERS):
        if not visible_has_any(text, AGENT_PAUSE_OR_UNCLEAR_NATURAL):
            if visible_has_any(text, UNFINISHED_SIGNALS):
                return True

    return False


def has_explicit_continue_marker(text: str) -> bool:
    """
    Bracket-style continue markers only (no loose “I want to keep going” phrasing).

    Use for main multi-turn control to avoid ordinary summaries/plans re-triggering generation.
    """
    if not text:
        return False
    text_upper = text.upper()
    for variant in CONTINUE_VARIANTS:
        if variant.upper() in text_upper:
            return True
    return False


def _line_is_pure_continue_variant(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    sup = s.upper()
    for variant in CONTINUE_VARIANTS:
        if sup == variant.upper():
            return True
    return False


def has_scheduler_continue_marker(answer: str) -> bool:
    """
    UnifiedScheduler system_continue only: prose that says “output [S44_CONTINUE]” must not enqueue.

    Rule: if body contains [S44_COMPLETE] / [S44_PAUSE], do not continue; continue token must be
    **alone on its line** (whole line is only that variant).
    """
    if not answer:
        return False
    if S44_COMPLETE_LITERAL in answer:
        return False
    if "[S44_PAUSE]" in answer:
        return False
    for ln in answer.splitlines():
        if _line_is_pure_continue_variant(ln):
            return True
    return False


def should_allow_implicit_continue_fallback(
    visible: str,
    full: str,
) -> Tuple[bool, str]:
    """
    Strict fallback when explicit markers are missing.

    Goal: only when the model clearly has more task work left — not casual chat, closing questions,
    or roleplay filler.
    """
    if not visible or not full:
        return False, ""

    if has_explicit_continue_marker(full):
        return False, ""

    if visible_has_any(visible, HARD_STOP_NATURAL_MARKERS):
        return False, "hard_stop"

    if visible_has_any(visible, AGENT_PAUSE_OR_UNCLEAR_NATURAL):
        return False, "pause_or_unclear"

    detected, reason = detect_implicit_continue(visible, full)
    if not detected or not reason.startswith("implicit:"):
        return False, reason

    visible_stripped = visible.strip()
    # Trailing question → waiting for user, not auto-continue
    if visible_stripped.endswith(("?", "？")):
        return False, "awaiting_user"

    if len(visible_stripped) < 40:
        return False, "too_short"

    signal = reason.split(":", 1)[1]
    tail = visible_stripped[-160:]
    last_line = visible_stripped.split("\n")[-1].strip()
    if signal not in tail and signal not in last_line:
        return False, "signal_not_in_tail"

    return True, reason


def detect_implicit_continue(visible: str, full: str) -> Tuple[bool, str]:
    """
    Detect implicit “want to continue” when no explicit marker.

    Returns (detected, reason_token).
    """
    if has_continue_intent(full):
        for variant in CONTINUE_VARIANTS:
            if variant.upper() in full.upper():
                return True, f"explicit:{variant}"
        if visible_has_any(full, UNFINISHED_SIGNALS):
            for sig in UNFINISHED_SIGNALS:
                if sig in full:
                    return True, f"implicit:{sig}"
        return True, "explicit:[S44_CONTINUE]"

    visible_stripped = visible.strip()
    if visible_stripped.endswith("...") or visible_stripped.endswith("……"):
        last_line = visible_stripped.split("\n")[-1].strip()
        if len(last_line) > 10 and (last_line.endswith("...") or last_line.endswith("……")):
            return False, "trailing_ellipsis"

    return False, ""
