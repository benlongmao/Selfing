#!/usr/bin/env bash
# Byte-compile all backend/*.py (no import execution; fast)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ -x "${ROOT}/.venv/bin/python" ]]; then
  PY="${ROOT}/.venv/bin/python"
else
  PY="python3"
fi
"${PY}" -m compileall -q "${ROOT}/backend"
echo "OK: backend py_compile (compileall)"
