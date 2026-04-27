"""
Agent external memory bridge: writes JSON under workspace/sandbox/agent_memory/
(state + runs) and refreshes STATUS_SNAPSHOT.md. Intended for S-44 style agents;
operators need not run a separate CLI.
"""
from __future__ import annotations

import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from backend.project_paths import PROJECT_ROOT, SANDBOX_ROOT

logger = logging.getLogger(__name__)

AGENT_MEMORY_DIR = Path(SANDBOX_ROOT) / "agent_memory"
STATE_PATH = AGENT_MEMORY_DIR / "state.json"
RUNS_DIR = AGENT_MEMORY_DIR / "runs"
SNAPSHOT_PATH = AGENT_MEMORY_DIR / "STATUS_SNAPSHOT.md"

MARK_BEGIN = "<!-- AGENT_MEMORY:BEGIN -->"
MARK_END = "<!-- AGENT_MEMORY:END -->"


def _detect_git_sha() -> Optional[str]:
    try:
        repo = Path(PROJECT_ROOT)
        for _ in range(4):
            if (repo / ".git").exists():
                out = subprocess.check_output(
                    ["git", "-C", str(repo), "rev-parse", "HEAD"],
                    stderr=subprocess.DEVNULL,
                    text=True,
                ).strip()
                return out[:12] if out else None
            if repo.parent == repo:
                break
            repo = repo.parent
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        pass
    return None


def _coerce_payload(payload: Any) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Models often pass JSON as a string; normalize to dict when possible."""
    if isinstance(payload, dict):
        return payload, None
    if isinstance(payload, str):
        s = payload.strip()
        if s.startswith("{") or s.startswith("["):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, dict):
                    return parsed, None
                if isinstance(parsed, list):
                    return {"items": parsed}, None
            except json.JSONDecodeError:
                pass
        return {"note": s}, None
    if isinstance(payload, list):
        return {"items": payload}, None
    if payload is None:
        return {}, None
    return (
        None,
        "payload must be a JSON object, a JSON string, or a short plain string (stored under note)",
    )


def _ensure_dirs() -> None:
    AGENT_MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    if not STATE_PATH.is_file():
        default = {
            "schema_version": 1,
            "agent_id": None,
            "phase": "unspecified",
            "summary": "",
            "blockers": [],
            "last_run": None,
            "extra": {},
        }
        STATE_PATH.write_text(
            json.dumps(default, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )


def record_agent_run(
    kind: str,
    payload: Union[Dict[str, Any], str, List[Any], None],
    phase: Optional[str] = None,
) -> Dict[str, Any]:
    """Append runs/<ts>_<kind>.json and bump state.json last_run."""
    _ensure_dirs()
    payload_dict, perr = _coerce_payload(payload)
    if payload_dict is None:
        return {"success": False, "error": perr or "invalid payload"}

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_kind = "".join(c if c.isalnum() or c in "-_" else "_" for c in kind)[:80]
    run_name = f"{ts}_{safe_kind}.json"
    run_path = RUNS_DIR / run_name

    envelope = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "kind": kind,
        "payload": payload_dict,
        "git_sha": _detect_git_sha(),
    }
    run_path.write_text(
        json.dumps(envelope, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    state["last_run"] = {
        "path": f"runs/{run_name}",
        "kind": kind,
        "at": envelope["created_at"],
    }
    if phase is not None:
        state["phase"] = phase
    STATE_PATH.write_text(
        json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    rel = run_path.relative_to(AGENT_MEMORY_DIR)
    logger.info("[agent_memory] recorded %s", rel)
    return {
        "success": True,
        "run_file": str(rel).replace("\\", "/"),
        "message": f"Wrote agent_memory/{rel} and updated state.last_run",
    }


def _latest_run() -> tuple[Optional[Path], Optional[Dict[str, Any]]]:
    if not RUNS_DIR.is_dir():
        return None, None
    files = sorted(
        [p for p in RUNS_DIR.iterdir() if p.suffix == ".json" and p.is_file()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not files:
        return None, None
    p = files[0]
    try:
        return p, json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return p, None


def _render_snapshot(state: dict, run_path: Optional[Path], run_env: Optional[dict]) -> str:
    lines = [
        "# Agent memory snapshot (auto-generated)",
        "",
        "Do not hand-edit numeric claims: treat `workspace/sandbox/agent_memory/state.json` "
        "and `runs/*.json` as source of truth.",
        "",
        "## state.json",
        "",
        "```json",
        json.dumps(state, indent=2, ensure_ascii=False),
        "```",
        "",
    ]
    if run_path and run_env:
        rel = run_path.relative_to(AGENT_MEMORY_DIR)
        lines += [
            "## Latest run",
            "",
            f"- file: `{rel}`",
            f"- kind: `{run_env.get('kind', '')}`",
            f"- time: `{run_env.get('created_at', '')}`",
        ]
        sha = run_env.get("git_sha")
        if sha:
            lines.append(f"- git: `{sha}`")
        lines += [
            "",
            "```json",
            json.dumps(run_env.get("payload") or {}, indent=2, ensure_ascii=False),
            "```",
            "",
        ]
    else:
        lines += ["## Latest run", "", "_No runs/*.json yet_", ""]
    lines += [
        "---",
        "_Generated by tool `agent_memory_sync`_",
        "",
    ]
    return "\n".join(lines)


def _resolve_inject_target(rel: Optional[str]) -> Optional[Path]:
    """Resolve HEARTBEAT.md / sandbox-relative / repo-relative paths."""
    if not rel or not str(rel).strip():
        return None
    r = str(rel).strip().lstrip("/").replace("\\", "/")
    root = Path(PROJECT_ROOT)
    sandbox = Path(SANDBOX_ROOT)
    candidates = [
        root / r,
        sandbox / r,
        root / "workspace" / "sandbox" / r,
    ]
    for c in candidates:
        if c.is_file():
            return c.resolve()
    if r in ("HEARTBEAT.md", "sandbox/HEARTBEAT.md", "workspace/sandbox/HEARTBEAT.md"):
        return (sandbox / "HEARTBEAT.md").resolve()
    return (root / r).resolve()


def sync_agent_memory_snapshot(inject_markdown_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Refresh STATUS_SNAPSHOT.md; optionally inject the body between AGENT_MEMORY markers
    in another markdown file (e.g. HEARTBEAT).
    """
    _ensure_dirs()
    state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    run_path, run_env = _latest_run()
    body = _render_snapshot(state, run_path, run_env)
    SNAPSHOT_PATH.write_text(body, encoding="utf-8")
    out: Dict[str, Any] = {
        "success": True,
        "snapshot": str(SNAPSHOT_PATH.relative_to(Path(PROJECT_ROOT))).replace("\\", "/"),
        "message": f"Updated {SNAPSHOT_PATH.relative_to(Path(PROJECT_ROOT))}",
    }

    if inject_markdown_path:
        target = _resolve_inject_target(inject_markdown_path)
        if target is None:
            return {**out, "success": False, "error": "inject_markdown_path is invalid"}

        target.parent.mkdir(parents=True, exist_ok=True)
        rel_report = None
        try:
            rel_report = str(target.relative_to(Path(PROJECT_ROOT))).replace("\\", "/")
        except ValueError:
            rel_report = str(target)

        if not target.is_file():
            seed = f"""# Heartbeat Tasks

## Active Tasks

## Completed

## Agent memory snapshot (auto-maintained)

{MARK_BEGIN}

{body.strip()}

{MARK_END}
"""
            target.write_text(seed, encoding="utf-8")
            out["injected"] = rel_report
            out["message"] = out.get("message", "") + f"; created and injected {rel_report}"
            return out

        text = target.read_text(encoding="utf-8")
        if MARK_BEGIN in text and MARK_END in text:
            before, rest = text.split(MARK_BEGIN, 1)
            mid, after = rest.split(MARK_END, 1)
            _ = mid
            new_text = before + MARK_BEGIN + "\n\n" + body.strip() + "\n\n" + MARK_END + after
            target.write_text(new_text, encoding="utf-8")
        else:
            new_text = (
                text.rstrip()
                + "\n\n## Agent memory snapshot (appended)\n\n"
                + MARK_BEGIN
                + "\n\n"
                + body.strip()
                + "\n\n"
                + MARK_END
                + "\n"
            )
            target.write_text(new_text, encoding="utf-8")
        out["injected"] = rel_report

    return out


def get_tool_definitions() -> list:
    return [
        {
            "type": "function",
            "function": {
                "name": "agent_memory_record",
                "description": (
                    "Write an auditable JSON run under workspace/sandbox/agent_memory/runs/ "
                    "and update agent_memory/state.json last_run. Call at phase boundaries "
                    "(stats done, hypothesis updated, milestone reached). "
                    "Then optionally call agent_memory_sync for a human-readable snapshot."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "kind": {
                            "type": "string",
                            "description": "Short label, e.g. overlap_stats, bugfix_compute_alpha",
                        },
                        "payload": {
                            "type": "object",
                            "description": "JSON object; strings are parsed as JSON when possible; "
                            "plain text becomes {\"note\": \"...\"}.",
                        },
                        "phase": {
                            "type": "string",
                            "description": "Optional: also writes state.phase (e.g. active/blocked/idle)",
                        },
                    },
                    "required": ["kind", "payload"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "agent_memory_sync",
                "description": (
                    "Refresh workspace/sandbox/agent_memory/STATUS_SNAPSHOT.md. "
                    "Optional inject_markdown_path writes between markers in HEARTBEAT etc.; "
                    "accepts workspace/sandbox/HEARTBEAT.md or HEARTBEAT.md. "
                    "Creates file/markers when missing."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "inject_markdown_path": {
                            "type": "string",
                            "description": "Optional. Recommended: workspace/sandbox/HEARTBEAT.md or HEARTBEAT.md",
                        },
                    },
                    "required": [],
                },
            },
        },
    ]
