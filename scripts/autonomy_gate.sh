#!/usr/bin/env bash
# Autonomy gate helper: inspect, pause, or resume background autonomy without editing JSON by hand.
# Usage from the repository root:
#   bash scripts/autonomy_gate.sh status
#   bash scripts/autonomy_gate.sh resume
#   bash scripts/autonomy_gate.sh pause
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
PY="${ROOT}/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  PY="python3"
fi
cmd="${1:-status}"
case "$cmd" in
  status|st|"")
    "$PY" << 'PY'
from backend.autonomy_gate import (
    _state_path,
    load_state,
    is_autonomous_execution_paused,
    gate_enabled,
)
import json

path = _state_path()
state = load_state()
print("State file:", path)
print("gate_enabled:", gate_enabled())
print("Background autonomy paused:", is_autonomous_execution_paused("selfing-session"))
print("Last updated by:", state.get("updated_by") or "-")
print("Reason:", state.get("reason") or "-")
print("Raw state:", json.dumps(state, ensure_ascii=False, indent=2))
PY
    ;;
  resume|on|start|open)
    "$PY" << 'PY'
from backend.autonomy_gate import set_autonomous_pause_from_cli, load_state, is_autonomous_execution_paused
import json

set_autonomous_pause_from_cli(False, "")
print("Resumed background autonomy (paused=false).")
print("Current:", json.dumps(load_state(), ensure_ascii=False))
print("Still paused?", is_autonomous_execution_paused("selfing-session"))
PY
    ;;
  pause|off|stop)
    "$PY" << 'PY'
from backend.autonomy_gate import set_autonomous_pause_from_cli, load_state, is_autonomous_execution_paused
import json

set_autonomous_pause_from_cli(True, "cli: scripts/autonomy_gate.sh pause")
print("Paused background autonomy (paused=true).")
print("Current:", json.dumps(load_state(), ensure_ascii=False))
print("Paused?", is_autonomous_execution_paused("selfing-session"))
PY
    ;;
  *)
    echo "Usage: bash scripts/autonomy_gate.sh {status|resume|pause}" >&2
    echo "  status - print state path, gate status, and JSON content" >&2
    echo "  resume - same effect as the chat command to resume/start autonomy" >&2
    echo "  pause  - pause background autonomous scheduling" >&2
    exit 1
    ;;
esac
