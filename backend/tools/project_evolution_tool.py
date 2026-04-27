#!/usr/bin/env python3
"""
Repo file operations in **evolution** mode (relative to the S project root) when
``agent_evolution.enabled`` is on. Blocked: ``.env`` secrets, direct ``.git/`` writes, ``node_modules/``.
"""
from __future__ import annotations

import logging
import os
import shutil
from typing import Any, Dict, List, Optional

from backend.dev_syntax import py_syntax_error_message

logger = logging.getLogger(__name__)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _norm_rel(path: str) -> Optional[str]:
    if path is None:
        return None
    p = str(path).strip().replace("\\", "/")
    if not p or p == ".":
        return ""
    if p.startswith("/") or ".." in p.split("/"):
        return None
    return p


def _abs_under_repo(rel: str) -> Optional[str]:
    nr = _norm_rel(rel)
    if nr is None:
        return None
    full = os.path.abspath(os.path.join(REPO_ROOT, nr))
    if not full.startswith(REPO_ROOT + os.sep) and full != REPO_ROOT:
        return None
    return full


def _blocked(rel: str, write: bool) -> Optional[str]:
    n = rel.replace("\\", "/").lower()
    parts = n.split("/")
    for seg in parts:
        if seg.startswith(".env"):
            return "Access to .env is denied (secrets)."
    if n == ".git" or n.startswith(".git/"):
        if write:
            return "Use evolution_git_* for Git; do not write .git/ directly"
    if write and "node_modules" in parts:
        return "Writes to node_modules are denied"
    return None


class ProjectEvolutionTool:
    def __init__(self) -> None:
        self.repo_root = REPO_ROOT
        logger.info(f"[ProjectEvolutionTool] repo_root={self.repo_root}")

    def evolution_read_file(self, path: str, max_chars: int = 120000) -> Dict[str, Any]:
        rel = _norm_rel(path)
        if rel is None:
            return {"success": False, "error": "Invalid path: use repo-relative path without .."}
        b = _blocked(rel, write=False)
        if b:
            return {"success": False, "error": b}
        abs_p = _abs_under_repo(rel)
        if not abs_p:
            return {"success": False, "error": "Path outside repository root"}
        if not os.path.isfile(abs_p):
            return {"success": False, "error": f"Not a file or not found: {path}"}
        try:
            with open(abs_p, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            if len(content) > max_chars:
                content = content[:max_chars] + "\n\n... (truncated) ..."
            return {"success": True, "path": rel, "content": content, "chars": len(content)}
        except OSError as e:
            return {"success": False, "error": str(e)}

    def evolution_write_file(self, path: str, content: str) -> Dict[str, Any]:
        rel = _norm_rel(path)
        if rel is None or rel == "":
            return {"success": False, "error": "Path invalid or empty"}
        b = _blocked(rel, write=True)
        if b:
            return {"success": False, "error": b}
        abs_p = _abs_under_repo(rel)
        if not abs_p:
            return {"success": False, "error": "Path outside repository root"}
        try:
            os.makedirs(os.path.dirname(abs_p) or ".", exist_ok=True)
            if os.path.isdir(abs_p):
                return {"success": False, "error": "Target is a directory; cannot overwrite as a file"}
            old_content: Optional[str] = None
            existed = os.path.isfile(abs_p)
            if existed:
                with open(abs_p, "r", encoding="utf-8", errors="replace") as f:
                    old_content = f.read()
            with open(abs_p, "w", encoding="utf-8") as f:
                f.write(content)
            # S-44: after writing .py, verify syntax; rollback to avoid breaking the server
            if rel.lower().endswith(".py"):
                syn_err = py_syntax_error_message(abs_p)
                if syn_err:
                    if existed and old_content is not None:
                        with open(abs_p, "w", encoding="utf-8") as f:
                            f.write(old_content)
                    else:
                        try:
                            os.remove(abs_p)
                        except OSError:
                            pass
                    logger.warning(
                        "[ProjectEvolutionTool] Python syntax check failed, rolled back: %s — %s",
                        rel,
                        syn_err,
                    )
                    return {
                        "success": False,
                        "error": f"Python syntax check failed; write reverted: {syn_err}",
                        "path": rel,
                    }
            return {"success": True, "path": rel, "message": "Written"}
        except OSError as e:
            return {"success": False, "error": str(e)}

    def evolution_mkdir(self, path: str) -> Dict[str, Any]:
        rel = _norm_rel(path)
        if rel is None or rel == "":
            return {"success": False, "error": "Path invalid or empty"}
        b = _blocked(rel, write=True)
        if b:
            return {"success": False, "error": b}
        abs_p = _abs_under_repo(rel)
        if not abs_p:
            return {"success": False, "error": "Path outside repository root"}
        try:
            os.makedirs(abs_p, exist_ok=True)
            return {"success": True, "path": rel}
        except OSError as e:
            return {"success": False, "error": str(e)}

    def evolution_delete_file(self, path: str) -> Dict[str, Any]:
        rel = _norm_rel(path)
        if rel is None or rel == "":
            return {"success": False, "error": "Path invalid or empty"}
        b = _blocked(rel, write=True)
        if b:
            return {"success": False, "error": b}
        abs_p = _abs_under_repo(rel)
        if not abs_p or not os.path.isfile(abs_p):
            return {"success": False, "error": f"File not found: {path}"}
        try:
            os.remove(abs_p)
            return {"success": True, "path": rel}
        except OSError as e:
            return {"success": False, "error": str(e)}

    def evolution_delete_directory(self, path: str, recursive: bool = False) -> Dict[str, Any]:
        rel = _norm_rel(path)
        if rel is None or rel == "":
            return {"success": False, "error": "Path invalid or empty"}
        b = _blocked(rel, write=True)
        if b:
            return {"success": False, "error": b}
        abs_p = _abs_under_repo(rel)
        if not abs_p or not os.path.isdir(abs_p):
            return {"success": False, "error": f"Directory not found: {path}"}
        try:
            if recursive:
                shutil.rmtree(abs_p)
            else:
                os.rmdir(abs_p)
            return {"success": True, "path": rel, "recursive": recursive}
        except OSError as e:
            return {"success": False, "error": str(e)}

    def evolution_rename_path(self, old_path: str, new_path: str) -> Dict[str, Any]:
        o = _norm_rel(old_path)
        n = _norm_rel(new_path)
        if o is None or n is None:
            return {"success": False, "error": "Invalid path"}
        if _blocked(o, True) or _blocked(n, True):
            return {"success": False, "error": "Path is blocked by policy"}
        a1, a2 = _abs_under_repo(o), _abs_under_repo(n)
        if not a1 or not a2:
            return {"success": False, "error": "Path outside repository root"}
        if not os.path.exists(a1):
            return {"success": False, "error": "Source does not exist"}
        try:
            parent = os.path.dirname(a2)
            if parent:
                os.makedirs(parent, exist_ok=True)
            shutil.move(a1, a2)
            return {"success": True, "old_path": o, "new_path": n}
        except OSError as e:
            return {"success": False, "error": str(e)}

    def evolution_copy_path(self, source_path: str, dest_path: str) -> Dict[str, Any]:
        s = _norm_rel(source_path)
        d = _norm_rel(dest_path)
        if s is None or d is None:
            return {"success": False, "error": "Invalid path"}
        if _blocked(s, False) or _blocked(d, True):
            return {"success": False, "error": "Path is blocked by policy"}
        a1, a2 = _abs_under_repo(s), _abs_under_repo(d)
        if not a1 or not a2:
            return {"success": False, "error": "Path outside repository root"}
        if not os.path.exists(a1):
            return {"success": False, "error": "Source does not exist"}
        try:
            if os.path.isdir(a1):
                if os.path.exists(a2):
                    return {"success": False, "error": "Destination already exists"}
                parent = os.path.dirname(a2)
                if parent:
                    os.makedirs(parent, exist_ok=True)
                shutil.copytree(a1, a2)
            else:
                parent = os.path.dirname(a2)
                if parent:
                    os.makedirs(parent, exist_ok=True)
                shutil.copy2(a1, a2)
            return {"success": True, "source": s, "dest": d}
        except OSError as e:
            return {"success": False, "error": str(e)}

    def evolution_list_tree(self, subdir: str = "", max_depth: int = 4) -> Dict[str, Any]:
        rel = _norm_rel(subdir) or ""
        if rel is None:
            return {"success": False, "error": "Invalid path"}
        b = _blocked(rel, write=False)
        if b:
            return {"success": False, "error": b}
        base = _abs_under_repo(rel)
        if not base or not os.path.isdir(base):
            return {"success": False, "error": "Directory not found"}
        max_depth = max(1, min(int(max_depth), 8))
        lines: List[str] = []

        def walk(d: str, prefix: str, depth: int) -> None:
            if depth > max_depth:
                return
            try:
                names = sorted(os.listdir(d))
            except OSError:
                return
            skip = {".git", "__pycache__", "node_modules", ".venv"}
            for name in names:
                if name in skip:
                    continue
                p = os.path.join(d, name)
                lines.append(f"{prefix}{name}/" if os.path.isdir(p) else f"{prefix}{name}")
                if os.path.isdir(p) and depth < max_depth:
                    walk(p, prefix + "  ", depth + 1)

        walk(base, "", 1)
        limit = 400
        truncated = len(lines) > limit
        return {
            "success": True,
            "root": rel or ".",
            "max_depth": max_depth,
            "lines": lines[:limit],
            "truncated": truncated,
            "total_lines": len(lines),
        }

    def evolution_search_repo(self, keyword: str, subdir: str = "") -> Dict[str, Any]:
        rel = _norm_rel(subdir) or ""
        if rel is None:
            return {"success": False, "error": "Invalid path"}
        base = _abs_under_repo(rel)
        if not base or not os.path.isdir(base):
            return {"success": False, "error": "Directory not found"}
        kw = keyword.lower()
        matches: List[Dict[str, str]] = []
        skip_dir = {".git", "__pycache__", "node_modules", ".venv"}
        exts = {".py", ".md", ".yaml", ".yml", ".json", ".ts", ".tsx", ".js", ".sh", ".toml"}
        scanned = 0
        for root, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if d not in skip_dir]
            for fn in files:
                if os.path.splitext(fn)[1].lower() not in exts and fn not in ("Dockerfile",):
                    continue
                fp = os.path.join(root, fn)
                rp = os.path.relpath(fp, REPO_ROOT)
                if _blocked(rp.replace("\\", "/"), False):
                    continue
                scanned += 1
                try:
                    with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                    if kw in content.lower():
                        idx = content.lower().find(kw)
                        snip = content[max(0, idx - 40) : min(len(content), idx + 60)].replace("\n", " ")
                        matches.append({"file": rp, "snippet": f"...{snip}..."})
                except OSError:
                    continue
                if len(matches) >= 25:
                    break
            if len(matches) >= 25:
                break
        return {"success": True, "matches": matches, "scanned_files": scanned, "keyword": keyword}

    def route_tool_call(self, func_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if func_name == "evolution_read_file":
            return self.evolution_read_file(args.get("path", ""), int(args.get("max_chars") or 120000))
        if func_name == "evolution_write_file":
            return self.evolution_write_file(args.get("path", ""), args.get("content", "") or "")
        if func_name == "evolution_mkdir":
            return self.evolution_mkdir(args.get("path", ""))
        if func_name == "evolution_delete_file":
            return self.evolution_delete_file(args.get("path", ""))
        if func_name == "evolution_delete_directory":
            return self.evolution_delete_directory(args.get("path", ""), bool(args.get("recursive", False)))
        if func_name == "evolution_rename_path":
            return self.evolution_rename_path(args.get("old_path", ""), args.get("new_path", ""))
        if func_name == "evolution_copy_path":
            return self.evolution_copy_path(args.get("source_path", ""), args.get("dest_path", ""))
        if func_name == "evolution_list_tree":
            return self.evolution_list_tree(args.get("subdir") or "", int(args.get("max_depth") or 4))
        if func_name == "evolution_search_repo":
            return self.evolution_search_repo(args.get("keyword", ""), args.get("subdir") or "")
        return {"success": False, "error": f"unknown evolution fs tool: {func_name}"}

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        desc = (
            "Operate under the S repo root (e.g. backend/app.py). Blocked: .env and direct .git/ writes. "
            "Before large edits run evolution_git_status / evolution_git_diff. "
        )
        return [
            {
                "type": "function",
                "function": {
                    "name": "evolution_read_file",
                    "description": desc + "Read a UTF-8 text file.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "max_chars": {"type": "integer"},
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "evolution_write_file",
                    "description": desc + "Overwrite a file (mkdir -p parents).",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                        "required": ["path", "content"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "evolution_mkdir",
                    "description": desc + "Create a directory tree.",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "evolution_delete_file",
                    "description": desc + "Delete a file.",
                    "parameters": {
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "evolution_delete_directory",
                    "description": desc + "Remove a directory; set recursive=true to delete non-empty trees.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "recursive": {"type": "boolean", "default": False},
                        },
                        "required": ["path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "evolution_rename_path",
                    "description": desc + "Rename or move a path inside the repo.",
                    "parameters": {
                        "type": "object",
                        "properties": {"old_path": {"type": "string"}, "new_path": {"type": "string"}},
                        "required": ["old_path", "new_path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "evolution_copy_path",
                    "description": desc + "Copy a file or directory inside the repo.",
                    "parameters": {
                        "type": "object",
                        "properties": {"source_path": {"type": "string"}, "dest_path": {"type": "string"}},
                        "required": ["source_path", "dest_path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "evolution_list_tree",
                    "description": desc + "Print a small directory tree (max_depth default 4).",
                    "parameters": {
                        "type": "object",
                        "properties": {"subdir": {"type": "string"}, "max_depth": {"type": "integer"}},
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "evolution_search_repo",
                    "description": desc + "Search source/config for a keyword (bounded).",
                    "parameters": {
                        "type": "object",
                        "properties": {"keyword": {"type": "string"}, "subdir": {"type": "string"}},
                        "required": ["keyword"],
                    },
                },
            },
        ]
