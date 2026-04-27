#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-9000}"
mkdir -p logs
if ! command -v uvicorn >/dev/null 2>&1; then
  echo "uvicorn not found. Activate venv and install requirements." >&2
  exit 2
fi
nohup uvicorn backend.app:app --host 0.0.0.0 --port "${PORT}" > "logs/backend.out" 2>&1 &
echo $! > logs/backend.pid
echo "Backend started on :${PORT}"


