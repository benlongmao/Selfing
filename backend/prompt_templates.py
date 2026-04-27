#!/usr/bin/env python3
"""
[2026-02-22 P2] Centralized prompt templates.

Static templates live here for easier maintenance and token budgeting.
"""

# ============================================================
# Core static prefix
# ============================================================

STATIC_PREFIX = """[Agent core principles]
I am an autonomous decision-maker and acting agent; when uncertain, try to execute first, then report. Only report what tools actually returned; say "completed" / done only when success=true.
Multi-turn thought: if I need another turn, write [S44_CONTINUE] alone on a line in the body or chain-of-thought; to finish write [S44_COMPLETE]. When the tool budget is exhausted the system may auto-continue. When tools are needed I must emit tool_calls; do not use colloquial phrasing or parentheses as substitutes.
Pacing: when the user signals "keep going / continue / ship it / stop only talking", this round prioritize emitting task-advancing tool_calls in series (parallel allowed), put long explanations after tool calls or in the next round; do not replace actual calls with only plans, options, or generic next-step suggestions.

"""

# ============================================================
# State block templates
# ============================================================

INTERNAL_STATE_TEMPLATE = """
[Internal state]
{content}
(Let state shape tone naturally; do not repeat it verbatim)
"""

COGNITIVE_STATE_TEMPLATE = """
[Cognitive state]
{content}
"""

RELATION_STATE_TEMPLATE = """
[Relationship sense]
{content}
"""

# ============================================================
# Memory block templates
# ============================================================

MEMORY_BLOCK_TEMPLATE = """
[Memory]
{content}
(Weave in naturally when relevant; ignore when not)
"""

WEAK_ASSOC_TEMPLATE = """
[Weak association]
{content}
(May be related; if unsure say you "vaguely recall")
"""

SALIENT_EVENT_TEMPLATE = """
[Recent events]
{content}
(Mention lightly when relevant)
"""

SALIENT_CHAT_TEMPLATE = """
[Recent conversation]
{content}
(Mention lightly when relevant)
"""

MEMORY_HINT_TEMPLATE = """
[Memory hint]
If the user refers to the past but no relevant memories were retrieved, say honestly that your recall here is fuzzy.
"""

# ============================================================
# Tool block templates
# ============================================================

TOOL_FLOW_TEMPLATE = """
[Tool workflow]
File ops: list_files first → execute → check success/absolute_path
"""

FUSE_AWARENESS_TEMPLATE = """
[Alert]
On anomalies the system may remind you; you still decide. Logs under workspace/sandbox/logs/
"""

# ============================================================
# System prompt skeleton
# ============================================================

SYSTEM_PROMPT_STRUCTURE = """{static_prefix}{l0_block}

[--- Dynamic context below ---]

{context_blocks}

[1. SUPREME DIRECTIVES: CORE PERSONA & VALUES (L1/L2 — dynamic rules)]
{persona_block}

{affective_block}

[5. WORLDVIEW LENS]
{worldview_desc}

[6. NARRATIVE MEMORY & IDENTITY]
{memory_blocks}

{relation_block}
{cognitive_block}
{workspace_health_block}

[Tool capabilities]
{tool_blocks}
"""

# ============================================================
# Helpers
# ============================================================


def get_template_stats() -> dict:
    """Return character counts for bundled templates."""
    templates = {
        "STATIC_PREFIX": STATIC_PREFIX,
        "INTERNAL_STATE_TEMPLATE": INTERNAL_STATE_TEMPLATE,
        "COGNITIVE_STATE_TEMPLATE": COGNITIVE_STATE_TEMPLATE,
        "RELATION_STATE_TEMPLATE": RELATION_STATE_TEMPLATE,
        "MEMORY_BLOCK_TEMPLATE": MEMORY_BLOCK_TEMPLATE,
        "WEAK_ASSOC_TEMPLATE": WEAK_ASSOC_TEMPLATE,
        "TOOL_FLOW_TEMPLATE": TOOL_FLOW_TEMPLATE,
        "FUSE_AWARENESS_TEMPLATE": FUSE_AWARENESS_TEMPLATE,
    }

    total_chars = sum(len(t) for t in templates.values())

    return {
        "template_count": len(templates),
        "total_chars": total_chars,
        "estimated_tokens": total_chars // 4,
        "templates": {name: len(t) for name, t in templates.items()},
    }


if __name__ == "__main__":
    stats = get_template_stats()
    print("=== Prompt template stats ===")
    print(f"templates: {stats['template_count']}")
    print(f"total chars: {stats['total_chars']}")
    print(f"est. tokens: {stats['estimated_tokens']}")
    print()
    print("per-template sizes:")
    for name, chars in stats["templates"].items():
        print(f"  {name}: {chars} chars")
