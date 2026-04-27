import os
import logging
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)

# Repo root inferred from this file location
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Safety boundary:
# - inspect_self_code may only read under workspace/
# - backend/, config/, eval/, etc. are intentionally out of scope
WORKSPACE_DIR = os.path.join(BASE_DIR, "workspace")

# Legacy: ToolRouter schemas once enumerated CONSCIOUSNESS_MAP keys.
# Backend source introspection is disabled; keep the map empty so paths are not exposed.
CONSCIOUSNESS_MAP: Dict[str, str] = {}


def _safe_join_under_workspace(rel_path: str) -> Optional[str]:
    """
    Join a user-supplied relative path under workspace/ with traversal checks.
    Returns an absolute path, or None when unsafe.
    """
    if rel_path is None:
        return None
    rel_path = str(rel_path).strip()
    if not rel_path:
        return None

    # Reject absolute paths and .. traversal
    candidate = os.path.abspath(os.path.join(WORKSPACE_DIR, rel_path))
    workspace_abs = os.path.abspath(WORKSPACE_DIR)
    if not candidate.startswith(workspace_abs + os.sep) and candidate != workspace_abs:
        return None
    return candidate


def _read_text_file(abs_path: str, max_chars: int = 20000) -> str:
    """Read UTF-8 text with a hard size cap."""
    with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    if len(content) > max_chars:
        return content[:max_chars] + "\n\n... (truncated) ..."
    return content

def inspect_self_code(module_name: Optional[str] = None, path: Optional[str] = None) -> Dict:
    """
    Workspace-only inspection.

    Args:
        path: Path relative to workspace/ (e.g. "sandbox/notes.md").
        module_name: Legacy argument; backend source reads are no longer supported.

    Returns:
        Dict with content/description or an error field.
    """

    # Legacy: module_name alone used to read backend sources — always deny now
    if path is None and module_name is not None:
        return {
            "error": "Access denied: module source inspection has been disabled. Provide a workspace-relative 'path' instead.",
            "allowed_root": "workspace/",
        }

    abs_path = _safe_join_under_workspace(path or "")
    if not abs_path:
        return {
            "error": "Access denied: path must be under workspace/ and must not use absolute paths or '..'.",
            "allowed_root": "workspace/",
        }

    if os.path.isdir(abs_path):
        # Directory listing (workspace subtree only)
        try:
            entries: List[str] = []
            for name in os.listdir(abs_path):
                entries.append(name)
            entries.sort()
            return {
                "path": os.path.relpath(abs_path, WORKSPACE_DIR),
                "description": "Directory listing under workspace/ (read-only).",
                "entries": entries[:200],
                "truncated": len(entries) > 200,
            }
        except Exception as e:
            return {"error": f"Failed to list directory: {e}"}

    if not os.path.exists(abs_path):
        return {
            "error": f"File not found under workspace/: {os.path.relpath(abs_path, WORKSPACE_DIR)}",
            "allowed_root": "workspace/",
        }

    try:
        content = _read_text_file(abs_path)
        return {
            "path": os.path.relpath(abs_path, WORKSPACE_DIR),
            "description": "Workspace file content (read-only, truncated).",
            "content": content,
        }
    except Exception as e:
        logger.error(f"Error reading workspace file {abs_path}: {e}")
        return {"error": f"Failed to read workspace file: {str(e)}"}
