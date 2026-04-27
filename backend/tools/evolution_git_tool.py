#!/usr/bin/env python3
"""Evolution-mode Git: run a restricted git CLI at repo root (no push / no remote changes)."""
from __future__ import annotations

import logging
import os
import subprocess
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MAX_OUT = 80000


def _run_git(args: List[str], timeout: int = 60) -> Dict[str, Any]:
    try:
        p = subprocess.run(
            ["git", "-C", REPO_ROOT, *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "LC_ALL": "C.UTF-8"},
        )
        out = (p.stdout or "") + (p.stderr or "")
        if len(out) > MAX_OUT:
            out = out[:MAX_OUT] + "\n... (truncated)"
        return {"success": p.returncode == 0, "returncode": p.returncode, "output": out}
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "git command timed out"}
    except Exception as e:
        return {"success": False, "error": str(e)}


class EvolutionGitTool:
    def __init__(self, allow_commit: bool = True, allow_push: bool = False) -> None:
        self.allow_commit = allow_commit
        self.allow_push = allow_push
        self.repo_root = REPO_ROOT

    def evolution_git_status(self) -> Dict[str, Any]:
        return _run_git(["status", "--porcelain", "-b"])

    def evolution_git_diff(self, staged: bool = False) -> Dict[str, Any]:
        cmd = ["diff"]
        if staged:
            cmd.append("--staged")
        return _run_git(cmd)

    def evolution_git_log(self, n: int = 30) -> Dict[str, Any]:
        n = max(1, min(int(n), 100))
        return _run_git(["log", f"-n{n}", "--oneline", "--decorate"])

    def evolution_git_add(self, paths: Optional[List[str]] = None) -> Dict[str, Any]:
        if not paths:
            return _run_git(["add", "-u"])
        safe: List[str] = []
        for p in paths:
            ps = str(p).strip().replace("\\", "/")
            if ".." in ps or ps.startswith("/"):
                return {"success": False, "error": f"Invalid path: {p}"}
            safe.append(ps)
        return _run_git(["add", "--", *safe])

    def evolution_git_commit(self, message: str) -> Dict[str, Any]:
        if not self.allow_commit:
            return {
                "success": False,
                "error": "Commits disabled (agent_evolution.allow_git_commit=false)",
            }
        msg = (message or "").strip()
        if len(msg) < 3:
            return {"success": False, "error": "Commit message too short"}
        return _run_git(["commit", "-m", msg], timeout=120)

    def evolution_git_checkout_file(self, path: str, revision: str = "HEAD") -> Dict[str, Any]:
        ps = str(path).strip().replace("\\", "/")
        if ".." in ps or ps.startswith("/") or not ps:
            return {"success": False, "error": "Invalid path"}
        rev = str(revision or "HEAD").strip()
        if any(c in rev for c in ";|&$`"):
            return {"success": False, "error": "Invalid revision"}
        return _run_git(["checkout", rev, "--", ps], timeout=60)

    def evolution_git_tag(self, name: str, message: Optional[str] = None) -> Dict[str, Any]:
        if not self.allow_commit:
            return {"success": False, "error": "Tagging requires commits to be enabled"}
        n = (name or "").strip()
        if not n or ".." in n or "/" in n:
            return {"success": False, "error": "Invalid tag name"}
        if message:
            return _run_git(["tag", "-a", n, "-m", message])
        return _run_git(["tag", n])

    def route_tool_call(self, func_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if func_name == "evolution_git_status":
            return self.evolution_git_status()
        if func_name == "evolution_git_diff":
            return self.evolution_git_diff(bool(args.get("staged", False)))
        if func_name == "evolution_git_log":
            return self.evolution_git_log(int(args.get("n") or 30))
        if func_name == "evolution_git_add":
            paths = args.get("paths")
            if isinstance(paths, list):
                return self.evolution_git_add(paths)
            return self.evolution_git_add(None)
        if func_name == "evolution_git_commit":
            return self.evolution_git_commit(args.get("message", ""))
        if func_name == "evolution_git_checkout_file":
            return self.evolution_git_checkout_file(
                args.get("path", ""), args.get("revision") or "HEAD"
            )
        if func_name == "evolution_git_tag":
            return self.evolution_git_tag(args.get("name", ""), args.get("message"))
        return {"success": False, "error": f"unknown git tool: {func_name}"}

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "evolution_git_status",
                    "description": "Git status (porcelain). Use after local edits for self-check.",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "evolution_git_diff",
                    "description": "Show diff; set staged=true for the index.",
                    "parameters": {
                        "type": "object",
                        "properties": {"staged": {"type": "boolean"}},
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "evolution_git_log",
                    "description": "Recent commits (oneline), newest first.",
                    "parameters": {
                        "type": "object",
                        "properties": {"n": {"type": "integer"}},
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "evolution_git_add",
                    "description": "Stage paths; when paths is empty, runs `git add -u`.",
                    "parameters": {
                        "type": "object",
                        "properties": {"paths": {"type": "array", "items": {"type": "string"}}},
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "evolution_git_commit",
                    "description": "Commit staged changes (requires allow_git_commit).",
                    "parameters": {
                        "type": "object",
                        "properties": {"message": {"type": "string"}},
                        "required": ["message"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "evolution_git_checkout_file",
                    "description": "Restore a single file from revision (default HEAD).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "revision": {"type": "string"},
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "evolution_git_tag",
                    "description": "Create a tag; when message is set, creates an annotated tag.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "message": {"type": "string"},
                        },
                        "required": ["name"],
                    },
                },
            },
        ]
