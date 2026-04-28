#!/usr/bin/env bash
set -euo pipefail
PORT="${JUDGE_ADAPTER_PORT:-9101}"
mkdir -p logs
if ! command -v uvicorn >/dev/null 2>&1; then
  echo "uvicorn not found. Activate venv and install requirements." >&2
  exit 2
fi
nohup uvicorn scripts.judge_adapter:app --host 0.0.0.0 --port "${PORT}" > "logs/judge_adapter.out" 2>&1 &
echo $! > logs/judge_adapter.pid
echo "Judge Adapter started on :${PORT}"


