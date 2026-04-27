"""
Project path layout (single source of truth).

Centralize filesystem paths so modules do not hard-code local locations.
Import from here instead of embedding paths.

Usage:
    from backend.project_paths import PROJECT_ROOT, WORKSPACE_ROOT, DATA_DB_PATH
"""

import os

# Repo root: derived from this file (backend/project_paths.py -> parent directory)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Common roots
BACKEND_ROOT = os.path.join(PROJECT_ROOT, "backend")
WORKSPACE_ROOT = os.path.join(PROJECT_ROOT, "workspace")
SANDBOX_ROOT = os.path.join(WORKSPACE_ROOT, "sandbox")
DATA_DB_PATH = os.path.join(PROJECT_ROOT, "data.db")
MODELS_ROOT = os.path.join(PROJECT_ROOT, "models")

# Helpers
def get_project_path(*parts: str) -> str:
    """Path under the repository root.

    Example:
        get_project_path("backend", "tools") -> "/path/to/s/backend/tools"
    """
    return os.path.join(PROJECT_ROOT, *parts)


def get_workspace_path(*parts: str) -> str:
    """Path under the workspace directory.

    Example:
        get_workspace_path("sandbox", "diaries") -> "/path/to/s/workspace/sandbox/diaries"
    """
    return os.path.join(WORKSPACE_ROOT, *parts)


def get_sandbox_path(*parts: str) -> str:
    """Path under the sandbox directory.

    Example:
        get_sandbox_path("diaries") -> "/path/to/s/workspace/sandbox/diaries"
    """
    return os.path.join(SANDBOX_ROOT, *parts)
