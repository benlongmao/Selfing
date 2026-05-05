from typing import List, Optional
import os


def build_tools_block(enable_tools: bool, tool_definitions: Optional[List[dict]], user_input: str = "", use_optimizer: bool = True) -> str:
    """
    Short English guidance for tool use; concrete schemas come from the API ``tools`` list.
    """
    if not enable_tools:
        return ""

    try:
        from backend.tool_selector import agent_evolution_enabled

        evolution_on = agent_evolution_enabled()
    except Exception:
        evolution_on = False

    ev_line = ""
    if evolution_on:
        ev_line = (
            "Repo evolution is ON: you may use evolution_*, evolution_git_*, execute_bash_project on the project tree; "
            "execute_bash_project can run pytest, ``python -m compileall`` (e.g. ``-q backend``), npm test|run|ci per tool descriptions. "
            "Personal sandbox work still favors read_file/write_file. Do not edit .env or write .git/ directly.\n"
        )

    return f"""[My tools]
This round's **available tools** are in the API ``tools`` field (names and parameters are the spec). **You choose which to call and when.** If the task needs disk reads, edits, or live state, default to calling tools first for facts, then answer; you may issue multiple ``tool_calls`` in parallel.
To list what is attached this turn: ``list_my_tools``. To request a missing bundle: ``request_tool_group('name')``.
{ev_line}Memory: for “what did we say on \<date\> / yesterday” use ``get_chat_turns_day_summary``; topic drill-down still uses ``recall_memory``. Search, calendar, PDF, etc. follow each tool description.
Note: each turn has a tool-call budget; the system will warn you as you approach the cap."""
