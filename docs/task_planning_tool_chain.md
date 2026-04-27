# Task-planning tools: wiring checklist

## Are tools registered?

| Step | Status | Notes |
|------|--------|-------|
| `tool_router.get_tool_definitions()` | Yes | When `task_manager` / `TASK_PLANNING_AVAILABLE`, registers `create_task`, `list_tasks`, `decompose_task`, `plan_today`, etc. |
| `tool_router.route()` | Yes | Branches dispatch to task manager / planner |
| `chat_tool_runner` | Yes | Calls `tool_router.route(function_name, args, session_id)` |

---

## Dynamic tool selection

| Step | Status | Notes |
|------|--------|-------|
| `TOOL_GROUPS["task_planning"]` | Yes | Includes the task tools |
| Keywords | Yes | CN/EN cues for todos, plans, decompose, today’s plan, etc. |
| `select_tools_with_semantic` | Yes | When the group matches, those tools enter the turn |
| `COMPACT_DESCRIPTIONS` | Yes | Short OpenAI descriptions for the four core helpers |

---

## Does the agent “know” its tools?

1. **API `tools` payload** — each turn sends selected tool schemas (`name`, `description`, `parameters`).
2. **`list_my_tools`** — introspection over `TOOL_GROUPS`; must itself be selected (often via `self_introspection` group when the user asks capabilities).

---

## Does the agent “know” how to call them?

| Aspect | Status |
|--------|--------|
| JSON `parameters` schema | Full schema is sent |
| Compact descriptions | Short; edge cases may need the long description in `tool_router` when `use_compact=False` |
| Prompt “capabilities” block | May lag new tools—keep `prompt_builder` tool blocks in sync when adding groups |

---

## Risks called out historically

1. **Compact text too short** — mitigated by expanding `COMPACT_DESCRIPTIONS` and the tools block (2026-03-12 pass).
2. **Tools block omitting names** — same pass listed concrete tool names in the capability reminder.

---

## Follow-ups implemented (2026-03-12)

1. Expanded `COMPACT_DESCRIPTIONS` for task tools (required fields + typical use).
2. Updated tools/capability reminder text to name `create_task`, `list_tasks`, `decompose_task`, `plan_today`.
