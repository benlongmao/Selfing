#!/usr/bin/env bash
# Pre-flight: import backend.app on the same path as uvicorn (catches ImportError that compileall misses).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
if [[ -x "${ROOT}/.venv/bin/python" ]]; then
  PY="${ROOT}/.venv/bin/python"
else
  PY="python3"
fi

"${PY}" <<'PY'
import os
import sys

os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

try:
    import backend.app  # noqa: F401 — same as uvicorn backend.app:app
except SystemExit as e:
    if e.code not in (0, None):
        raise
except Exception as e:
    print(f"IMPORT FAIL: {e!r}", file=sys.stderr)
    sys.exit(1)

print("OK: backend app import (same graph as uvicorn)")
PY
