#!/usr/bin/env bash
set -euo pipefail

VLLM_PORT="${VLLM_PORT:-8000}"
REPORT_DIR="reports"
LOG_DIR="logs"
mkdir -p "${REPORT_DIR}" "${LOG_DIR}"

ok_vllm=0
ok_db=0
msg_vllm=""
msg_db=""

# Check vLLM endpoint
if curl -sf "http://localhost:${VLLM_PORT}/v1/models" >/dev/null; then
  ok_vllm=1
  msg_vllm="vLLM OK"
else
  ok_vllm=0
  msg_vllm="vLLM NOT READY on port ${VLLM_PORT}"
fi

# Check DB tables
if command -v sqlite3 >/dev/null 2>&1; then
  if sqlite3 data.db ".tables" | grep -q "self_state"; then
    ok_db=1
    msg_db="DB OK"
  else
    ok_db=0
    msg_db="DB MISSING TABLES"
  fi
else
  ok_db=0
  msg_db="sqlite3 not found"
fi

status=$(( ok_vllm & ok_db ))

jq -n \
  --arg vllm "${msg_vllm}" \
  --arg db "${msg_db}" \
  --arg port "${VLLM_PORT}" \
  --arg ts "$(date -Is)" \
  --arg model "${MODEL_ID:-unknown}" \
  --arg dtype "${DTYPE:-float16}" \
  --arg commit "$(git rev-parse --short HEAD 2>/dev/null || echo 'na')" \
  --arg status "$([ $status -eq 1 ] && echo ok || echo fail)" \
  '{
    ts: $ts,
    status: $status,
    vllm: $vllm,
    db: $db,
    env: { port: $port, model: $model, dtype: $dtype, commit: $commit }
  }' > "${REPORT_DIR}/acceptance_report.json"

echo "Acceptance report -> ${REPORT_DIR}/acceptance_report.json"

if [ $status -ne 1 ]; then
  echo "Acceptance failed."
  exit 1
fi

exit 0


