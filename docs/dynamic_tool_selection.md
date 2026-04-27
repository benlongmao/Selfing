# Dynamic tool selection

## Goals

- **Subset per turn:** send only tool definitions **related to the current user text**, not the full tool table—save tokens and latency.
- **Compact descriptions:** `COMPACT_DESCRIPTIONS` overrides long OpenAI `description` strings.

---

## End-to-end flow

```
User text
    ↓
should_provide_tools_for_input(user_input)  → if “small talk / no tools” → tools=None
    ↓ else
select_tools_with_semantic(tool_router, user_input, embedder, use_compact=True)
    ↓
HybridToolSelector.select_tools()
    ├── 1. Keyword match: TOOL_GROUPS keywords hit → tool groups
    ├── 2. Semantic match (only if keywords miss): embed user text, cosine vs TOOL_GROUPS_SEMANTIC per group; threshold ~0.42
    ├── 3. Union tool names per group; if none, add defaults (read_file, write_file, list_files, list_my_tools)
    ├── 4. Cap: MAX_TOOLS_PER_TURN (e.g. 12); keep ESSENTIAL_TOOLS, then HIGH_PRIORITY_GROUPS order
    └── 5. Fetch full defs from tool_router.get_tool_definitions(), subset by name, swap description with COMPACT_DESCRIPTIONS
    ↓
tools = [ OpenAI-style function defs ]
    ↓
Pass as API `tools`; model may only choose from this set
    ↓
tool_calls → chat_tool_runner → tool_router.route(...)
(optional) verify function_name ∈ allowed_tool_names for this turn
```

---

## Core components

| Piece | File | Role |
|-------|------|------|
| **Gate: send tools?** | `tool_selector.should_provide_full_tools` | `NO_TOOL_MARKERS` / `NEED_TOOL_MARKERS` + semantic gate for borderline short chat |
| **Keyword groups** | `tool_selector.TOOL_GROUPS` | Keywords → group → list of tool names (~28 groups) |
| **Semantic groups** | `tool_semantic_descriptions.TOOL_GROUPS_SEMANTIC` | `description` + `examples` → embeddings vs user text |
| **Keyword selector** | `ToolSelector` | Keyword routing, caps, priorities |
| **Semantic selector** | `SemanticToolSelector` | Cosine similarity vs group synopsis |
| **Hybrid** | `HybridToolSelector` | Keywords first, then semantic; same trim rules |
| **Definitions** | `tool_router.get_tool_definitions()` | Full catalog; selector returns a subset |
| **Execution** | `chat_tool_runner` + `tool_router.route` | Run tool_calls; optional allowlist enforcement |

---

## Call sites

- **`chat_service`:** (1) `should_provide_tools_for_input`; if false, `tools=None`. (2) Else `select_tools_with_semantic(..., use_compact=True)`. (3) Pass `tools` to the API; optional **tool forcing** adds a few tools by keyword.
- **`chat_tool_runner`:** builds `allowed_tool_names` from the turn’s `tools`; optional reject if `function_name` not allowed.

---

## Design summary

1. **Layers:** TOOL_GROUPS / TOOL_GROUPS_SEMANTIC → pick groups → union tools → subset defs + compact text.
2. **Dual path:** keywords (fast) + semantic (recall); semantic only when keywords miss.
3. **Caps & priority:** max tools per turn; essentials first, then priority groups.
4. **Small-talk gate:** centralized so trivial lines do not carry the whole tool table.

---

## ReAct tool loop (multi-step inside one user message)

For a heavy task the agent may loop **think → tools → think → tools** until a final answer or a cap.

- **`max_tool_turns`:** from `parameters.chat.max_tool_turns` (default **8**) or `CHAT_MAX_TOOL_TURNS`; each round can include multiple `tool_calls`, then append results and call the model again.
- **Tool list frozen:** the same `tools` list from **turn 1** is reused inside the loop (unless `tools.reselect_in_tool_loop` merges new names—see `config/settings.yaml`).
- **At limit:** a system line `[TOOL LOOP LIMIT REACHED]` forces closure; model may emit `[S44_CONTINUE]` to resume tools next user turn.
- **Intent-driven multi-turn:** separate cap `max_multi_turns`; continuation is **`[S44_CONTINUE]` literal**, not colloquial “continue”.

---

## Re-select tools inside the loop (optional)

If `tools.reselect_in_tool_loop: true`, from round 2 onward the stack may re-run semantic selection on a digest of (last user line + assistant snippet + tool results), **union** new tool names into the allowlist (monotone: add only). Trade-off: extra embedding + selection cost per tool round.
