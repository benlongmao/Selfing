"""
Dev-time syntax gate: run py_compile only (no imports — avoids heavy deps like transformers).
Used after writes from evolution_write_file and similar hooks.
"""
from __future__ import annotations

import py_compile
from typing import Optional


def py_syntax_error_message(path: str) -> Optional[str]:
    """Return compiler error text if ``path`` is not valid Python; otherwise ``None``."""
    try:
        py_compile.compile(path, doraise=True)
    except py_compile.PyCompileError as e:
        return str(e)
    except OSError as e:
        return str(e)
    return None
