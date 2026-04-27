#!/usr/bin/env python3
"""
Intent tag parser — turns bracketed agent intents into OpenAI-style tool calls.

Design:
- The agent is treated as a subject that already knows its capabilities.
- Full tool schemas need not be re-sent every turn (token savings).
- The model emits compact ``<<INTENT:...>>`` tags; this module maps them to ``tool_calls``.
"""

import re
import json
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# Intent regex table (stable wire format)
INTENT_PATTERNS = {
    # Files
    "file_write": re.compile(r'<<FILE:write\s+path="([^"]+)"\s+content="([^"]*)"?>>', re.DOTALL),
    "file_write_block": re.compile(r'<<FILE:write\s+path="([^"]+)">>(.+?)<<END>>', re.DOTALL),
    "file_read": re.compile(r'<<FILE:read\s+path="([^"]+)">>'),
    "file_list": re.compile(r'<<FILE:list(?:\s+path="([^"]*)")?\s*>>'),
    "file_search": re.compile(r'<<FILE:search\s+keyword="([^"]+)">>'),
    "file_delete": re.compile(r'<<FILE:delete\s+path="([^"]+)">>'),

    # Web search
    "search": re.compile(r'<<SEARCH\s+query="([^"]+)">>'),

    # Email
    "email_send": re.compile(r'<<EMAIL\s+to="([^"]+)"\s+subject="([^"]+)"\s+body="([^"]*)"?>>', re.DOTALL),
    "email_send_block": re.compile(r'<<EMAIL\s+to="([^"]+)"\s+subject="([^"]+)">>(.+?)<<END>>', re.DOTALL),
    "email_check": re.compile(r'<<EMAIL:check(?:\s+limit="?(\d+)"?)?\s*>>'),

    # Code execution
    "code_python": re.compile(r'<<CODE\s+lang="python">>(.+?)<<END>>', re.DOTALL),

    # Calendar
    "calendar_add": re.compile(r'<<CALENDAR:add\s+title="([^"]+)"(?:\s+time="([^"]+)")?\s*>>'),
    "calendar_list": re.compile(r'<<CALENDAR:list(?:\s+days="?(\d+)"?)?\s*>>'),
    "calendar_delete": re.compile(r'<<CALENDAR:delete\s+id="([^"]+)">>'),

    # Time
    "time": re.compile(r'<<TIME>>'),

    # Self workspace
    "self_facts": re.compile(r'<<SELF:facts>>'),
    "self_code": re.compile(r'<<SELF:code\s+path="([^"]+)">>'),
    "self_list": re.compile(r'<<SELF:list\s+path="([^"]+)">>'),
    "self_search": re.compile(r'<<SELF:search\s+keyword="([^"]+)"(?:\s+path="([^"]*)")?\s*>>'),

    # Goals
    "goal_add": re.compile(r'<<GOAL:add\s+title="([^"]+)"(?:\s+description="([^"]*)")?\s*>>'),
    "goal_list": re.compile(r'<<GOAL:list>>'),
    "goal_update": re.compile(r'<<GOAL:update\s+id="([^"]+)"\s+status="([^"]+)">>'),
    "goal_detail": re.compile(r'<<GOAL:detail\s+id="([^"]+)">>'),

    # Research controls
    "research_pause": re.compile(r'<<RESEARCH:pause(?:\s+reason="([^"]*)")?\s*>>'),
    "research_resume": re.compile(r'<<RESEARCH:resume>>'),
    "research_status": re.compile(r'<<RESEARCH:status>>'),

    # Mind wandering
    "mind_wandering": re.compile(r'<<MIND_WANDERING(?:\s+reason="([^"]*)")?\s*>>'),
}

INTENT_TO_TOOL = {
    "file_write": "write_file",
    "file_write_block": "write_file",
    "file_read": "read_file",
    "file_list": "list_files",
    "file_search": "search_files",
    "file_delete": "delete_file",
    "file_rename": "rename_file",
    "search": "tavily_search",
    "email_send": "send_email",
    "email_send_block": "send_email",
    "email_check": "check_unread_emails",
    "code_python": "execute_python",
    "calendar_add": "add_calendar_event",
    "calendar_list": "list_calendar_events",
    "calendar_delete": "delete_calendar_event",
    "time": "get_current_time",
    "self_facts": "get_self_facts",
    "self_code": "read_self_code",
    "self_list": "list_self_files",
    "self_search": "search_self_code",
    "goal_add": "goal_add",
    "goal_list": "goal_list",
    "goal_update": "goal_update_status",
    "goal_detail": "goal_detail",
    "research_pause": "research_pause",
    "research_resume": "research_resume",
    "research_status": "research_status",
    "mind_wandering": "request_mind_wandering",
}


def _natural_language_mind_wandering(content: str) -> bool:
    """True when the model asks to mind-wander in Chinese or common English phrasing."""
    if "神游" in content:
        return True
    low = content.lower()
    return any(
        needle in low
        for needle in (
            "mind wander",
            "mind-wandering",
            "mindwandering",
            "let my mind wander",
        )
    )


def parse_intents(content: str) -> List[Dict[str, Any]]:
    """
    Parse bracket intents from ``content`` into pseudo tool_calls (OpenAI-compatible shape).

    Args:
        content: assistant-visible text that may embed intent tags.

    Returns:
        List of dicts shaped like ``{"id", "type": "function", "function": {"name", "arguments"}}``.
    """
    tool_calls: List[Dict[str, Any]] = []
    call_id = 0

    for intent_name, pattern in INTENT_PATTERNS.items():
        matches = pattern.finditer(content)

        for match in matches:
            tool_name = INTENT_TO_TOOL.get(intent_name)
            if not tool_name:
                continue

            args = _extract_args(intent_name, match)

            tool_calls.append({
                "id": f"intent_{call_id}",
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(args, ensure_ascii=False)
                }
            })
            call_id += 1

            logger.info(f"[INTENT] Parsed: {intent_name} -> {tool_name}({args})")

    # NL fallback: CN mind-wandering phrasing or English "mind wander*" without explicit tags.
    if not tool_calls and _natural_language_mind_wandering(content):
        nl_mind_patterns = [
            r'(?:让我|我来|我想|要|先)\s*(?:调用|进入|进行)?\s*神游\s*(?:来|去|一下)?\s*([^。！？\n]{0,80})',
            r'神游\s*(?:来|去|一下)\s*([^。！？\n]{0,80})',
            r'(?:调用|进入)\s*神游',
            r'(?:let me|i want to|i will)\s+(?:do\s+)?(?:some\s+)?mind[- ]?wandering'
            r'(?:\s+about|\s+on|\s+to)?\s*([^.\n!?]{0,80})',
            r'mind[- ]?wander(?:ing)?\s*(?:about|on|to)?\s*([^.\n!?]{0,80})',
        ]
        for pat in nl_mind_patterns:
            m = re.search(pat, content, flags=re.IGNORECASE)
            if m:
                try:
                    reason = (m.group(1) or "").strip() or "Explore inner state"
                except (IndexError, AttributeError):
                    reason = "Explore inner state"
                if len(reason) > 100:
                    reason = reason[:97] + "..."
                tool_calls.append({
                    "id": f"intent_{call_id}",
                    "type": "function",
                    "function": {
                        "name": "request_mind_wandering",
                        "arguments": json.dumps(
                            {"reason": reason or "Autonomous reflection"},
                            ensure_ascii=False,
                        )
                    }
                })
                call_id += 1
                logger.info(
                    f"[INTENT] Natural-language mind wandering -> request_mind_wandering(reason={reason[:50]}...)"
                )
                break

    return tool_calls


def _extract_args(intent_name: str, match: re.Match) -> Dict[str, Any]:
    """Map regex capture groups to tool argument dicts."""
    groups = match.groups()

    if intent_name == "file_write":
        return {"filename": groups[0], "content": groups[1] if len(groups) > 1 else ""}
    elif intent_name == "file_write_block":
        return {"filename": groups[0], "content": groups[1].strip() if len(groups) > 1 else ""}
    elif intent_name == "file_read":
        return {"filename": groups[0]}
    elif intent_name == "file_list":
        return {"path": groups[0] if groups[0] else ""}
    elif intent_name == "file_search":
        return {"keyword": groups[0]}
    elif intent_name == "file_delete":
        return {"filename": groups[0]}

    elif intent_name == "search":
        return {"query": groups[0]}

    elif intent_name == "email_send":
        return {"to_address": groups[0], "subject": groups[1], "content": groups[2] if len(groups) > 2 else ""}
    elif intent_name == "email_send_block":
        return {"to_address": groups[0], "subject": groups[1], "content": groups[2].strip() if len(groups) > 2 else ""}
    elif intent_name == "email_check":
        return {"limit": int(groups[0]) if groups[0] else 5}

    elif intent_name == "code_python":
        return {"code": groups[0].strip()}

    elif intent_name == "calendar_add":
        result = {"title": groups[0]}
        if len(groups) > 1 and groups[1]:
            result["start_time"] = groups[1]
        return result
    elif intent_name == "calendar_list":
        return {"days_ahead": int(groups[0]) if groups[0] else 7}
    elif intent_name == "calendar_delete":
        return {"event_id": groups[0]}

    elif intent_name == "time":
        return {}

    elif intent_name == "self_facts":
        return {}
    elif intent_name == "self_code":
        return {"path": groups[0]}
    elif intent_name == "self_list":
        return {"path": groups[0]}
    elif intent_name == "self_search":
        result = {"keyword": groups[0]}
        if len(groups) > 1 and groups[1]:
            result["path"] = groups[1]
        return result

    elif intent_name == "goal_add":
        result = {"title": groups[0]}
        if len(groups) > 1 and groups[1]:
            result["description"] = groups[1]
        return result
    elif intent_name == "goal_list":
        return {}
    elif intent_name == "goal_update":
        return {"goal_id": groups[0], "status": groups[1]}
    elif intent_name == "goal_detail":
        return {"goal_id": groups[0]}

    elif intent_name == "research_pause":
        return {"reason": groups[0] if groups[0] else "User-requested pause"}
    elif intent_name == "research_resume":
        return {}
    elif intent_name == "research_status":
        return {}

    elif intent_name == "mind_wandering":
        return {"reason": groups[0] if groups[0] else "Autonomous reflection"}

    return {}


def remove_intent_tags(content: str) -> str:
    """Strip intent tags (and stray ``<<END>>``) from assistant-visible text."""
    cleaned = content

    for pattern in INTENT_PATTERNS.values():
        cleaned = pattern.sub("", cleaned)

    cleaned = re.sub(r'<<END>>', '', cleaned)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)

    return cleaned.strip()


def has_intents(content: str) -> bool:
    """Return True if any intent regex matches."""
    for pattern in INTENT_PATTERNS.values():
        if pattern.search(content):
            return True
    return False


if __name__ == "__main__":
    test_content = '''
    I want to write a short diary entry.

    <<FILE:write path="workspace/sandbox/diary_test.md">>
    # Today's note

    This is the diary body.
    <<END>>

    Then I want to search for something.
    <<SEARCH query="latest AI research">>

    Also show the current time.
    <<TIME>>
    '''

    intents = parse_intents(test_content)
    print("Parsed intents:")
    for intent in intents:
        print(f"  - {intent}")

    print("\nCleaned text:")
    print(remove_intent_tags(test_content))
