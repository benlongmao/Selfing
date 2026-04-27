#!/usr/bin/env python3
"""
Unified workspace path resolution for s-main.

Resolves user-supplied paths against a fixed sandbox root so file tools and
memory do not depend on ``os.getcwd()``.

Design:
1. Stable workspace root (not cwd-relative).
2. Single entry points for resolve / normalize / safety checks.
3. Supports ``~`` expansion and relative segments under the workspace.
4. Normalized relative paths use ``/`` for cross-platform display/storage.
"""
import os
import logging
from typing import Tuple, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# --- Fixed workspace root ---


def get_project_root() -> str:
    """
    Repository root directory.

    Order:
    1. ``S_PROJECT_ROOT`` when set.
    2. Infer from this file: ``backend/utils/path_utils.py`` -> repo root (two levels up).
    """
    env_root = os.environ.get("S_PROJECT_ROOT")
    if env_root:
        return os.path.abspath(env_root)
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def get_workspace_root() -> str:
    """
    Default agent workspace (sandbox) root.

    Default: ``{project_root}/workspace/sandbox``.
    """
    project_root = get_project_root()
    workspace_root = os.path.join(project_root, "workspace", "sandbox")
    return os.path.abspath(workspace_root)


# --- Resolve / normalize ---


def resolve_workspace_path(
    input_path: str,
    workspace_root: Optional[str] = None,
    allow_absolute: bool = False
) -> Tuple[str, str]:
    """
    Map arbitrary user input to ``(absolute_path, relative_to_workspace)`` inside the sandbox.

    Args:
        input_path: Relative path, absolute path, or ``~``-prefixed path.
        workspace_root: Override sandbox root; defaults to ``get_workspace_root()``.
        allow_absolute: When False, reject absolute paths that fall outside ``workspace_root``.

    Returns:
        ``(absolute_path, relative_path)`` where ``relative_path`` uses ``/``.

    Raises:
        ValueError: Resolved path escapes ``workspace_root``.
    """
    if workspace_root is None:
        workspace_root = get_workspace_root()

    workspace_root = os.path.abspath(workspace_root)

    # Empty -> workspace root
    if not input_path or input_path.strip() == "":
        return workspace_root, "."

    input_path = input_path.strip()

    # Expand home directory
    if input_path.startswith("~"):
        input_path = os.path.expanduser(input_path)

    # Strip redundant workspace/sandbox prefixes (avoid double sandbox)
    normalized = input_path.replace("\\", "/")
    prefixes_to_remove = [
        "workspace/sandbox/",
        "workspace/",
        "sandbox/",
    ]
    for prefix in prefixes_to_remove:
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]
            break

    if os.path.isabs(normalized):
        abs_path = os.path.normpath(normalized)
        if not allow_absolute:
            if not abs_path.startswith(workspace_root):
                raise ValueError(
                    f"Absolute path outside workspace: {input_path} "
                    f"(workspace: {workspace_root})"
                )
    else:
        abs_path = os.path.normpath(os.path.join(workspace_root, normalized))

    abs_path = os.path.abspath(abs_path)
    if not abs_path.startswith(workspace_root):
        raise ValueError(
            f"Path escapes workspace root: {input_path} "
            f"(resolved: {abs_path}, workspace: {workspace_root})"
        )

    try:
        rel_path = os.path.relpath(abs_path, workspace_root)
        if rel_path == ".":
            rel_path = ""
        rel_path = rel_path.replace("\\", "/")
    except ValueError:
        rel_path = os.path.basename(abs_path)

    return abs_path, rel_path


def normalize_path_for_storage(rel_path: str) -> str:
    """
    Normalize a relative path for stable storage keys.

    Uses ``/`` and strips leading ``./``.
    """
    if not rel_path:
        return ""

    normalized = rel_path.replace("\\", "/")
    normalized = normalized.lstrip("./")

    return normalized


def is_path_safe(file_path: str, workspace_root: Optional[str] = None) -> bool:
    """
    Return True when ``file_path`` resolves inside the workspace sandbox.

    Args:
        file_path: Candidate path.
        workspace_root: Optional override; see ``resolve_workspace_path``.

    Returns:
        True if resolution succeeds without ``ValueError``.
    """
    try:
        resolve_workspace_path(file_path, workspace_root)
        return True
    except (ValueError, Exception) as e:
        logger.debug(f"Path not safe: {file_path}, error: {e}")
        return False


# --- Display helpers ---


def shorten_home_path(file_path: str) -> str:
    """Replace the expanded home directory prefix with ``~`` when applicable."""
    home = os.path.expanduser("~")
    if file_path.startswith(home):
        return "~" + file_path[len(home):]
    return file_path


def display_path(file_path: str) -> str:
    """User-facing short path (home shortened)."""
    return shorten_home_path(file_path)
