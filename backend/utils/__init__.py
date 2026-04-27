"""
Shared backend utilities (path resolution, etc.).
"""
from .path_utils import (
    get_project_root,
    get_workspace_root,
    resolve_workspace_path,
    normalize_path_for_storage,
    is_path_safe,
    shorten_home_path,
    display_path
)

__all__ = [
    "get_project_root",
    "get_workspace_root",
    "resolve_workspace_path",
    "normalize_path_for_storage",
    "is_path_safe",
    "shorten_home_path",
    "display_path",
]
