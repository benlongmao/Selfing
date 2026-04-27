#!/usr/bin/env python3
"""
[2026-02-05] Tool result standardizer

Core ideas:
1. Normalize every tool return to a structured shape with an explicit ``status`` field.
2. Reduce model hallucination via mandatory verification reminders and status flags.
3. Structured status + check hints cut false “success” reports.

Standard shape:
{
    "content": [{"type": "text", "text": "..."}],  # OpenAI-compatible
    "details": {
        "status": "completed" | "failed" | "error" | "timeout" | "approval-pending",
        "tool_name": str,
        "metadata": {...},  # tool-specific metadata
    },
    "verification_reminder": str  # nudge the agent to verify before answering the user
}

Usage:
    from backend.tool_result_standardizer import standardize_tool_result

    raw_result = {"success": True, "data": "..."}
    standardized = standardize_tool_result(
        tool_name="write_file",
        result=raw_result,
        session_id=session_id
    )
"""

import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


def _extract_status(result: Dict[str, Any]) -> str:
    """
    Extract or infer status from a tool return.

    Precedence:
    1. result["details"]["status"]
    2. result["status"]
    3. Infer from success / error fields
    """
    # 1. details.status
    if isinstance(result.get("details"), dict):
        status = result["details"].get("status")
        if isinstance(status, str):
            return _normalize_status(status)

    # 2. Root-level status
    if "status" in result and isinstance(result["status"], str):
        return _normalize_status(result["status"])

    # 3. success / error inference
    if "error" in result:
        return "error"

    if result.get("success") is True:
        return "completed"

    if result.get("success") is False:
        return "failed"

    # 4. Default: unknown (needs human/agent inspection)
    return "unknown"


def _normalize_status(status: str) -> str:
    """
    Normalize free-form status strings to canonical values.

    Canonical values:
    - completed: finished successfully
    - failed: execution failed with a clear failure
    - error: system / transport error
    - timeout: timed out
    - approval-pending: waiting for approval
    - unknown: unclear outcome
    """
    status_lower = status.lower().strip()

    # Success-like
    if status_lower in ["completed", "success", "ok", "done", "✅"]:
        return "completed"

    # Failure-like
    if status_lower in ["failed", "fail", "failure"]:
        return "failed"

    # Error-like
    if status_lower in ["error", "❌"]:
        return "error"

    # Timeout
    if status_lower in ["timeout", "timed_out", "time_out"]:
        return "timeout"

    # Approval pending
    if status_lower in ["approval-pending", "pending", "waiting"]:
        return "approval-pending"

    # Passthrough (already normalized-ish)
    return status_lower


def _format_content(result: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Format a tool return as OpenAI ``content`` items.

    Returns:
        [{"type": "text", "text": "..."}]
    """
    if "content" in result and isinstance(result["content"], list):
        return result["content"]

    text_parts = []

    if "message" in result:
        text_parts.append(str(result["message"]))

    if "content" in result and isinstance(result["content"], str):
        text_parts.append(result["content"])

    if "data" in result:
        data = result["data"]
        if isinstance(data, str):
            text_parts.append(f"Data: {data[:500]}..." if len(data) > 500 else f"Data: {data}")
        elif isinstance(data, (list, dict)):
            import json
            try:
                data_str = json.dumps(data, ensure_ascii=False, indent=2)
                text_parts.append(f"Data:\n{data_str[:500]}..." if len(data_str) > 500 else f"Data:\n{data_str}")
            except:
                text_parts.append(f"Data: {str(data)[:200]}...")

    if "error" in result:
        text_parts.append(f"❌ Error: {result['error']}")

    if "warning" in result:
        text_parts.append(f"⚠️ Warning: {result['warning']}")

    text = "\n\n".join(text_parts) if text_parts else str(result)

    return [{"type": "text", "text": text}]


def _extract_metadata(result: Dict[str, Any], tool_name: str) -> Dict[str, Any]:
    """
    Extract tool-specific metadata for debugging and downstream use.
    """
    metadata = {}

    preserve_keys = [
        "execution_time", "execution_id", "durationMs",
        "file_path", "filename", "size", "lines", "bytes",
        "exitCode", "exit_code", "returncode",
        "stdout", "stderr", "return_value",
        "verified", "verification_method",
        "pid", "session_id",
        "url", "status_code",
        "count", "total_count", "truncated",
        "timestamp", "created_at", "updated_at"
    ]

    for key in preserve_keys:
        if key in result:
            metadata[key] = result[key]

    if isinstance(result.get("details"), dict):
        details = result["details"].copy()
        details.pop("status", None)  # avoid duplicating status
        if details:
            metadata.update(details)

    metadata["tool_name"] = tool_name

    return metadata


def _generate_verification_reminder(
    tool_name: str,
    status: str,
    result: Dict[str, Any]
) -> str:
    """
    Build a short reminder for the agent based on tool type and status.

    This is the main anti-hallucination nudge: force the model to reconcile
    its answer with the actual payload.
    """
    if status == "completed":
        if tool_name == "write_file":
            system_verified = result.get("system_verification", {})
            file_exists = system_verified.get("file_exists", False)

            if file_exists:
                return (
                    "✅ File was created and the system verified it exists on disk.\n"
                    "⚠️ Before telling the user, confirm ``content_preview`` matches what you intend to report."
                )
            else:
                return (
                    "❌ Warning: the tool claimed success but verification failed — the file may not exist.\n"
                    "⚠️ Tell the user honestly that creation failed; do not pretend it succeeded."
                )

        elif tool_name == "execute_python":
            stdout = result.get("stdout", "")
            stderr = result.get("stderr", "")

            if stderr:
                return (
                    "⚠️ Execution finished but stderr is non-empty.\n"
                    "Read stderr carefully; if there is a real exception, say so clearly to the user."
                )
            elif not stdout or not stdout.strip():
                return (
                    "⚠️ Execution finished with empty stdout.\n"
                    "That may mean the code did nothing useful — double-check logic or tell the user there was no output."
                )
            else:
                return (
                    "✅ Python ran successfully; stdout above is the real output.\n"
                    "⚠️ Read stdout before reporting; only describe what it actually shows."
                )

        elif tool_name == "list_files":
            files = result.get("files", [])
            return (
                f"✅ Listed {len(files)} file(s).\n"
                f"⚠️ This is the real listing — do not invent paths that are not shown here."
            )

        elif tool_name == "read_file":
            content = result.get("content", "")
            filename = result.get("filename", "") or result.get("file_path", "") or ""
            reminder = (
                f"✅ File read; payload length is {len(content)} character(s).\n"
                f"⚠️ When quoting, use only what was read here — not memory or guesses."
            )
            _analysis_hints = (
                "research", "report", "analysis", "assessment", "conclusion", "comparison",
                "对比", "评估", "结论", "报告",
            )
            if any(h in filename.lower() for h in _analysis_hints):
                reminder += (
                    "\n⚠️ This file may contain prior analysis or reasoning, not ground truth."
                    " If you cite it, label it as your earlier analysis; re-derive from raw data when in doubt."
                )
            return reminder

        elif tool_name == "rename_file":
            if result.get("success"):
                return (
                    "✅ File renamed or moved.\n"
                    "⚠️ When reporting to the user, use the returned ``new_path``; call ``read_file(new_path)`` if you need contents."
                )
            return (
                f"✅ ``rename_file`` returned.\n"
                f"⚠️ Report exactly what ``success`` / ``error`` says — do not smooth over failures."
            )

        elif tool_name in ("recall_memory", "get_recent_context"):
            return (
                f"✅ Memory retrieval finished.\n"
                f"⚠️ Action logs (what was done) can be quoted directly; labeled analysis can be cited as prior reasoning without re-proving every time."
            )

        else:
            return (
                f"✅ Tool ``{tool_name}`` completed successfully.\n"
                f"⚠️ Base your reply only on the payload above — do not invent extra fields or outcomes."
            )

    elif status in ["failed", "error"]:
        error_msg = result.get("error", "Unknown error")
        return (
            f"❌ Tool ``{tool_name}`` failed.\n"
            f"Reason: {error_msg}\n"
            f"⚠️ Tell the user plainly; do not claim success or hide the error.\n"
            f"Suggestion: explain the cause and what they can try next."
        )

    elif status == "timeout":
        return (
            f"⏱️ Tool ``{tool_name}`` timed out.\n"
            f"⚠️ Let the user know; they may need different parameters or a retry."
        )

    elif status == "approval-pending":
        return (
            f"⏳ Tool ``{tool_name}`` is waiting for user approval.\n"
            f"⚠️ Explain that approval is required, surface any approval id, and pause until they confirm."
        )

    else:
        return (
            f"⚠️ Tool ``{tool_name}`` finished but status is unclear (status={status}).\n"
            f"Inspect the payload carefully before you tell the user it succeeded or failed."
        )


def _coerce_plain_string_result(s: str) -> Dict[str, Any]:
    """
    Wrap legacy string-only tool returns (e.g. calendar text) into a dict.

    Recognizes common CN/EN success and failure cues so both locales keep working.
    """
    sl = s.lower()
    cn_ok = ("已添加" in s and "❌" not in s) or ("成功" in s and "失败" not in s)
    en_ok = (
        ("added" in sl or "created" in sl or "scheduled" in sl)
        and "failed" not in sl
        and "error" not in sl
        and "❌" not in s
    ) or (
        "successfully" in sl
        and "failed" not in sl
        and "unsuccessful" not in sl
    )
    if "✅" in s or cn_ok or en_ok:
        return {"success": True, "content": s}
    cn_fail = "失败" in s or "错误" in s
    en_fail = "failed" in sl or "error:" in sl or "exception" in sl
    if "❌" in s or cn_fail or en_fail:
        return {"error": s, "success": False}
    # Unknown shape: treat as success (most tools return benign plain text on success)
    return {"success": True, "content": s}


def standardize_tool_result(
    tool_name: str,
    result: Dict[str, Any],
    session_id: Optional[str] = None,
    add_verification_reminder: bool = True
) -> Dict[str, Any]:
    """
    Normalize a raw tool return into the standard envelope.

    Args:
        tool_name: Logical tool name.
        result: Raw return (should be dict; plain str is wrapped automatically).
        session_id: Optional session id for logs.
        add_verification_reminder: When True, attach ``verification_reminder``.

    Returns:
        Standardized dict with ``content``, ``details``, and optional reminder.
    """
    # [fix] Non-dict returns (e.g. calendar strings like "✅ Added event …") must be wrapped
    if not isinstance(result, dict):
        s = str(result) if result is not None else ""
        result = _coerce_plain_string_result(s)

    status = _extract_status(result)
    content = _format_content(result)
    metadata = _extract_metadata(result, tool_name)

    verification_reminder = ""
    if add_verification_reminder:
        verification_reminder = _generate_verification_reminder(tool_name, status, result)

    standardized = {
        "content": content,
        "details": {
            "status": status,
            **metadata
        }
    }

    if verification_reminder:
        standardized["verification_reminder"] = verification_reminder

    if "success" in result:
        standardized["success"] = result["success"]

    if "ok" in result:
        standardized["ok"] = result["ok"]

    logger.debug(
        f"Standardized tool result: tool={tool_name}, status={status}, "
        f"session={session_id or 'unknown'}"
    )

    return standardized


def is_tool_result_error(result: Dict[str, Any]) -> bool:
    """
    Return True if the payload clearly represents a tool error.

    ``chat_tool_runner`` can use this for lightweight failure detection.
    """
    if not isinstance(result, dict):
        return False

    if isinstance(result.get("details"), dict):
        status = result["details"].get("status", "")
        if status in ["error", "failed", "timeout"]:
            return True

    if "error" in result:
        return True

    if result.get("success") is False:
        return True

    if result.get("ok") is False:
        return True

    return False


__all__ = [
    "standardize_tool_result",
    "is_tool_result_error",
]
