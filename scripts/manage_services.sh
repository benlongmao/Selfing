#!/bin/bash
#
# One-shot service manager for this project.
# Commands: start / stop / status / restart / open
# On start: may launch local vLLM, backend, heartbeat, and try to open the web UI.
#
# Design notes:
# 1. Prefer variables from .env; environment can still override.
# 2. .env is loaded early so later logic sees the right values.
# 3. Ports and DB path can come from .env.
# 4. Model path detection prefers explicit VLLM_MODEL / MODEL_ID.

set -e
# Relaxed strictness on purpose:
# -e: exit on first failing command
# We omit -u and pipefail so partial failures (e.g. grep with no match) do not abort the whole script.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
PID_DIR="${ROOT_DIR}/run"
VENV_ACTIVATE="${ROOT_DIR}/.venv/bin/activate"
VLLM_PID_FILE="${PID_DIR}/vllm.pid"
BACKEND_PID_FILE="${PID_DIR}/backend.pid"
HEARTBEAT_PID_FILE="${PID_DIR}/heartbeat.pid"

# ============================================
# Load .env first (so exports are visible below)
# ============================================
if [[ -f "${ROOT_DIR}/.env" ]]; then
  # shellcheck disable=SC2046
  set -o allexport
  # shellcheck source=/dev/null
  source "${ROOT_DIR}/.env" || {
    echo "⚠️  Failed to load .env; continuing with defaults"
  }
  set +o allexport
  echo "✅ Loaded .env"
fi

# ============================================
# Optional YAML defaults (only if env not already set)
# ============================================
if command -v python3 >/dev/null 2>&1; then
  if [[ -f "${ROOT_DIR}/config/settings.yaml" ]]; then
    # Parse YAML with Python and emit export lines
    _S_ROOT="${ROOT_DIR}" eval "$(python3 << 'PYTHON_EOF'
import yaml
import os
import sys

try:
    with open(os.path.join(os.environ.get('_S_ROOT', '.'), 'config/settings.yaml'), 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f) or {}

    # system.model_provider
    model_provider = config.get('system', {}).get('model_provider', '')
    if model_provider and not os.environ.get('MODEL_PROVIDER'):
        print(f'export MODEL_PROVIDER="{model_provider}"')

    # models.deepseek
    deepseek_config = config.get('models', {}).get('deepseek', {})
    if deepseek_config:
        if not os.environ.get('DEEPSEEK_BASE_URL') and deepseek_config.get('base_url'):
            print(f'export DEEPSEEK_BASE_URL="{deepseek_config["base_url"]}"')
        if not os.environ.get('DEEPSEEK_API_KEY') and deepseek_config.get('api_key'):
            print(f'export DEEPSEEK_API_KEY="{deepseek_config["api_key"]}"')
        if not os.environ.get('DEEPSEEK_MODEL') and deepseek_config.get('model_id'):
            print(f'export DEEPSEEK_MODEL="{deepseek_config["model_id"]}"')

    # [Commented] Kimi block kept for future provider switch
    # kimi_config = config.get('models', {}).get('kimi', {})
    # ...

    # models.vllm (fallback defaults)
    vllm_config = config.get('models', {}).get('vllm', {})
    if vllm_config:
        if not os.environ.get('VLLM_BASE_URL') and vllm_config.get('base_url'):
            print(f'export VLLM_BASE_URL="{vllm_config["base_url"]}"')
        if not os.environ.get('VLLM_API_KEY') and vllm_config.get('api_key'):
            print(f'export VLLM_API_KEY="{vllm_config["api_key"]}"')
        if not os.environ.get('MODEL_ID') and vllm_config.get('model_id'):
            model_id = vllm_config.get('model_id')
            if model_id:  # only export non-empty ids
                print(f'export MODEL_ID="{model_id}"')

    # models.claude
    claude_config = config.get('models', {}).get('claude', {})
    if claude_config:
        if not os.environ.get('CLAUDE_BASE_URL') and claude_config.get('base_url'):
            print(f'export CLAUDE_BASE_URL="{claude_config["base_url"]}"')
        if not os.environ.get('CLAUDE_API_KEY') and claude_config.get('api_key'):
            print(f'export CLAUDE_API_KEY="{claude_config["api_key"]}"')
        if not os.environ.get('CLAUDE_MODEL') and claude_config.get('model_id'):
            print(f'export CLAUDE_MODEL="{claude_config["model_id"]}"')
except Exception as e:
    # Silent: YAML is optional
    pass
PYTHON_EOF
    )"
  fi
fi

# ============================================
# Environment (from .env / shell, with defaults)
# ============================================

# Ports
VLLM_PORT="${VLLM_PORT:-8000}"
BACKEND_PORT="${BACKEND_PORT:-8080}"

# Default local vLLM client settings
VLLM_API_KEY="${VLLM_API_KEY:-token-abc123}"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:${VLLM_PORT}/v1}"
# Fallback when not set in settings.yaml / .env; chat timeouts follow app config
VLLM_TIMEOUT="${VLLM_TIMEOUT:-600}"

# Provider selection
MODEL_PROVIDER="$(echo "${MODEL_PROVIDER:-local_vllm}" | tr '[:upper:]' '[:lower:]')"
DEEPSEEK_BASE_URL="${DEEPSEEK_BASE_URL:-https://api.deepseek.com/v1}"
DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-deepseek-chat}"
DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-}"
# [Commented] Kimi defaults kept for future use
# KIMI_BASE_URL="${KIMI_BASE_URL:-https://api.moonshot.cn/v1}"
# KIMI_MODEL="${KIMI_MODEL:-kimi-k2.5}"
# KIMI_API_KEY="${MOONSHOT_API_KEY:-${KIMI_API_KEY:-}}"
IS_LOCAL_PROVIDER="true"
PROVIDER_NAME="Local vLLM"

# Database
DB_PATH="${DB_PATH:-${ROOT_DIR}/data.db}"
# ========================================================
# [Eval] Optional DB_PATH_OVERRIDE for isolated runs / benchmarks
# Note: sourcing .env may reset DB_PATH from the shell; this is the final override.
# ========================================================
if [[ -n "${DB_PATH_OVERRIDE:-}" ]]; then
  DB_PATH="${DB_PATH_OVERRIDE}"
fi
# Resolve relative DB_PATH against repo root
if [[ "${DB_PATH}" != /* ]]; then
  DB_PATH="${ROOT_DIR}/${DB_PATH}"
fi

# Model ids (no hard-coded inference default; must be configured for real runs)
if [[ -z "${MODEL_ID:-}" ]]; then
  echo "⚠️  MODEL_ID is empty; set it in .env"
  MODEL_ID=""
fi
if [[ -z "${VLLM_MODEL:-}" ]]; then
  VLLM_MODEL="${MODEL_ID}"
fi

case "${MODEL_PROVIDER}" in
  deepseek_api)
    IS_LOCAL_PROVIDER="false"
    PROVIDER_NAME="DeepSeek API"
    VLLM_BASE_URL="${DEEPSEEK_BASE_URL}"
    if [[ -n "${DEEPSEEK_API_KEY}" ]]; then
      VLLM_API_KEY="${DEEPSEEK_API_KEY}"
    fi
    MODEL_ID="${DEEPSEEK_MODEL}"
    VLLM_MODEL="${MODEL_ID}"
    if [[ -z "${VLLM_API_KEY}" ]]; then
      echo "⚠️  DEEPSEEK_API_KEY missing; set it in .env"
    fi
    ;;
  claude_api)
    IS_LOCAL_PROVIDER="false"
    PROVIDER_NAME="Claude API"
    CLAUDE_BASE_URL="${CLAUDE_BASE_URL:-https://api.anthropic.com}"
    CLAUDE_API_KEY="${CLAUDE_API_KEY:-}"
    CLAUDE_MODEL="${CLAUDE_MODEL:-claude-opus-4-5}"
    # Do not overwrite VLLM_* for Claude (separate adapter); only warn if key missing
    if [[ -z "${CLAUDE_API_KEY}" ]]; then
      echo "⚠️  CLAUDE_API_KEY missing; set it in .env"
    fi
    ;;
  # [Commented] kimi_api case for future switch
  # kimi_api)
  #   ...
  #   ;;
  *)
    PROVIDER_NAME="Local vLLM"
    ;;
esac

# Tools
TAVILY_API_KEY="${TAVILY_API_KEY:-}"

# Web UI
UI_URL="http://localhost:${BACKEND_PORT}/ui/"

# ============================================
# Ensure runtime dirs
# ============================================
mkdir -p "${LOG_DIR}" "${PID_DIR}"

# ============================================
# Export for child processes
# ============================================
export TAVILY_API_KEY
export DB_PATH
export VLLM_BASE_URL
export VLLM_API_KEY
export MODEL_ID
export VLLM_MODEL

# ============================================
# Helpers
# ============================================

ensure_venv() {
  if [[ ! -f "${VENV_ACTIVATE}" ]]; then
    echo "❌ Virtualenv not found: ${VENV_ACTIVATE}"
    echo "Run: bash ${ROOT_DIR}/install_s_project.sh"
    exit 1
  fi
}

activate_venv() {
  # shellcheck source=/dev/null
  if [[ -f "${VENV_ACTIVATE}" ]]; then
    source "${VENV_ACTIVATE}"
    return 0
  else
    echo "❌ Missing venv activate script: ${VENV_ACTIVATE}"
    return 1
  fi
}

is_process_alive() {
  local pid_file=$1
  [[ -f "${pid_file}" ]] || return 1
  local pid shell_pid
  pid="$(tr -d '[:space:]' < "${pid_file}")"
  [[ -n "${pid}" ]] || return 1
  if kill -0 "${pid}" >/dev/null 2>&1; then
    shell_pid="${pid}"
    echo "${shell_pid}"
    return 0
  else
    return 1
  fi
}

detect_model_path() {
  # Prefer VLLM_MODEL from the environment
  if [[ -n "${VLLM_MODEL}" ]]; then
    # Local path under repo root
    if [[ -e "${ROOT_DIR}/${VLLM_MODEL}/config.json" ]]; then
      echo "${ROOT_DIR}/${VLLM_MODEL}"
      return 0
    elif [[ -e "${VLLM_MODEL}/config.json" ]]; then
      echo "${VLLM_MODEL}"
      return 0
    fi
    # Otherwise treat as remote model id string
    echo "${VLLM_MODEL}"
    return 0
  fi

  # Nothing configured
  echo ""
  return 1
}

wait_for_service() {
  local url=$1
  local header=$2
  local name=$3
  local timeout=${4:-90}

  echo "⏳ Waiting for ${name} at ${url} ..."
  for ((i=0; i<timeout; i++)); do
    if [[ -n "${header}" ]]; then
      if curl -sSf -H "${header}" "${url}" >/dev/null 2>&1; then
        echo "✅ ${name} is ready"
        return 0
      fi
    else
      if curl -sSf "${url}" >/dev/null 2>&1; then
        echo "✅ ${name} is ready"
        return 0
      fi
    fi
    sleep 2
  done
  echo "⚠️  Timed out waiting for ${name}; check logs."
  return 1
}

if ! command -v curl >/dev/null 2>&1; then
  echo "⚠️  curl not found; health checks may fail"
fi

# ============================================
# Service starters
# ============================================

start_vllm() {
  if [[ "${IS_LOCAL_PROVIDER}" != "true" ]]; then
    echo "ℹ️  Provider is ${PROVIDER_NAME}; local vLLM is not used."
    return 0
  fi

  if is_process_alive "${VLLM_PID_FILE}" >/dev/null; then
    echo "ℹ️  vLLM already running (PID $(cat "${VLLM_PID_FILE}"))"
    return 0
  fi

  if [[ -z "${VLLM_MODEL:-}" ]] && [[ -z "${MODEL_ID:-}" ]]; then
    echo "❌ VLLM_MODEL / MODEL_ID not set; cannot start local vLLM"
    echo "   Set MODEL_PROVIDER=deepseek_api for API mode, or set VLLM_MODEL in .env"
    return 1
  fi

  ensure_venv
  activate_venv || {
    echo "❌ Could not activate virtualenv"
    return 1
  }

  local model_path
  if ! model_path="$(detect_model_path)"; then
    echo "❌ Could not resolve model path; check VLLM_MODEL"
    return 1
  fi

  if [[ -z "${model_path}" ]]; then
    echo "❌ Empty model path; check VLLM_MODEL"
    return 1
  fi

  echo "▶️  Starting vLLM"
  echo "   Model path: ${model_path}"
  echo "   Port: ${VLLM_PORT}"
  echo "   API key prefix: ${VLLM_API_KEY:0:10}..."

  export VLLM_WORKER_MULTIPROC_METHOD=spawn
  export TMPDIR="${ROOT_DIR}/.cache/vllm"
  mkdir -p "${TMPDIR}"

  local vllm_cmd=(
    python -m vllm.entrypoints.openai.api_server
    --model "${model_path}"
    --trust-remote-code
    --port "${VLLM_PORT}"
    --api-key "${VLLM_API_KEY}"
  )

  if [[ -n "${VLLM_DTYPE:-}" ]]; then
    vllm_cmd+=(--dtype "${VLLM_DTYPE}")
  fi
  if [[ -n "${VLLM_GPU_MEMORY_UTILIZATION:-}" ]]; then
    vllm_cmd+=(--gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}")
  fi
  if [[ -n "${VLLM_MAX_MODEL_LEN:-}" ]]; then
    vllm_cmd+=(--max-model-len "${VLLM_MAX_MODEL_LEN}")
  fi

  mkdir -p "${LOG_DIR}"

  nohup "${vllm_cmd[@]}" > "${LOG_DIR}/vllm.log" 2>&1 &
  local pid=$!

  if ! kill -0 "${pid}" >/dev/null 2>&1; then
    echo "❌ vLLM failed to start (process exited immediately)"
    echo "   See ${LOG_DIR}/vllm.log"
    if [[ -f "${LOG_DIR}/vllm.log" ]]; then
      tail -n 20 "${LOG_DIR}/vllm.log" || true
    fi
    return 1
  fi

  echo "${pid}" > "${VLLM_PID_FILE}"
  echo "   PID: ${pid}"

  sleep 2

  if ! kill -0 "${pid}" >/dev/null 2>&1; then
    echo "❌ vLLM exited right after start; see ${LOG_DIR}/vllm.log"
    tail -n 20 "${LOG_DIR}/vllm.log" || true
    rm -f "${VLLM_PID_FILE}"
    return 1
  fi

  if ! wait_for_service "${VLLM_BASE_URL}/models" "Authorization: Bearer ${VLLM_API_KEY}" "vLLM" 120; then
    echo "❌ vLLM did not become ready in time; see ${LOG_DIR}/vllm.log"
    tail -n 40 "${LOG_DIR}/vllm.log" || true
    return 1
  fi
}

detect_model_id() {
  if [[ "${IS_LOCAL_PROVIDER}" != "true" ]]; then
    echo "${MODEL_ID}"
    return 0
  fi

  if command -v curl >/dev/null 2>&1 && command -v python3 >/dev/null 2>&1; then
    local detected_id
    detected_id=$(curl -s -H "Authorization: Bearer ${VLLM_API_KEY}" "${VLLM_BASE_URL}/models" 2>/dev/null \
      | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['data'][0]['id'] if data.get('data') else '${MODEL_ID}')" 2>/dev/null) || true
    if [[ -n "${detected_id}" ]]; then
      echo "${detected_id}"
      return 0
    fi
  fi
  echo "${MODEL_ID}"
}

start_heartbeat() {
  if is_process_alive "${HEARTBEAT_PID_FILE}" >/dev/null; then
    echo "ℹ️  Heartbeat already running (PID $(cat "${HEARTBEAT_PID_FILE}"))"
    return 0
  fi

  echo "▶️  Starting heartbeat worker"

  local python_cmd
  if [[ -f "${ROOT_DIR}/.venv/bin/python" ]]; then
    python_cmd="${ROOT_DIR}/.venv/bin/python"
  else
    python_cmd="python3"
  fi

  nohup "${python_cmd}" "${ROOT_DIR}/scripts/heartbeat.py" \
    > "${LOG_DIR}/heartbeat.log" 2>&1 &

  local pid=$!
  echo "${pid}" > "${HEARTBEAT_PID_FILE}"
  echo "   PID: ${pid}"
  echo "   ✅ Heartbeat started (interval: 300s)"
}

start_backend() {
  if is_process_alive "${BACKEND_PID_FILE}" >/dev/null; then
    echo "ℹ️  Backend already running (PID $(cat "${BACKEND_PID_FILE}"))"
    return 0
  fi

  ensure_venv

  cd "${ROOT_DIR}"

  export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

  if [[ "${IS_LOCAL_PROVIDER}" == "true" ]]; then
    if ! wait_for_service "${VLLM_BASE_URL}/models" "Authorization: Bearer ${VLLM_API_KEY}" "vLLM" 1; then
      echo "⚠️  vLLM not detected yet; backend may fail until inference is up."
    fi
  else
    wait_for_service "${VLLM_BASE_URL}/models" "Authorization: Bearer ${VLLM_API_KEY}" "${PROVIDER_NAME}" 10 || true
  fi

  local model_id
  model_id="$(detect_model_id)"
  echo "▶️  Starting backend"
  echo "   Database: ${DB_PATH}"
  echo "   Model id: ${model_id}"
  echo "   Port: ${BACKEND_PORT}"
  echo "   Inference base URL: ${VLLM_BASE_URL}"

  RELOAD_ARGS=()
  if [[ "${ENABLE_RELOAD:-false}" == "true" ]]; then
    RELOAD_ARGS=(--reload)
    echo "⚠️  Dev mode: --reload enabled (.cache churn may restart often)"
  else
    echo "ℹ️  Prod mode: auto-reload disabled (avoids vLLM temp-file reload loops)"
  fi

  mkdir -p "${LOG_DIR}"

  local python_cmd
  if [[ -f "${ROOT_DIR}/.venv/bin/python" ]]; then
    python_cmd="${ROOT_DIR}/.venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    python_cmd="python3"
  else
    echo "❌ Python interpreter not found"
    return 1
  fi

  if ! "${python_cmd}" -m uvicorn --help >/dev/null 2>&1; then
    echo "❌ uvicorn not available; install deps in the venv"
    echo "   Try: source ${VENV_ACTIVATE} && pip install uvicorn"
    return 1
  fi

  export DB_PATH
  export VLLM_BASE_URL
  export VLLM_API_KEY
  export MODEL_ID="${model_id}"
  export VLLM_MODEL="${model_id}"
  if [[ -n "${TAVILY_API_KEY:-}" ]]; then
    export TAVILY_API_KEY
  fi

  echo "   Starting uvicorn (${python_cmd})..."
  echo "   Env: DB_PATH=${DB_PATH}, VLLM_BASE_URL=${VLLM_BASE_URL}, MODEL_ID=${model_id}"

  local env_vars=(
    "DB_PATH=${DB_PATH}"
    "VLLM_BASE_URL=${VLLM_BASE_URL}"
    "VLLM_API_KEY=${VLLM_API_KEY}"
    "MODEL_ID=${model_id}"
    "VLLM_MODEL=${model_id}"
    "PYTHONPATH=${PYTHONPATH}"
  )
  if [[ -n "${TAVILY_API_KEY:-}" ]]; then
    env_vars+=("TAVILY_API_KEY=${TAVILY_API_KEY}")
  fi
  # ========================================================
  # Forward mail-related env explicitly
  # ========================================================
  if [[ -n "${EMAIL_USER:-}" ]]; then
    env_vars+=("EMAIL_USER=${EMAIL_USER}")
  fi
  if [[ -n "${EMAIL_PASSWORD:-}" ]]; then
    env_vars+=("EMAIL_PASSWORD=${EMAIL_PASSWORD}")
  fi
  # Legacy names
  if [[ -n "${RENSI_EMAIL_USER:-}" ]]; then
    env_vars+=("EMAIL_USER=${RENSI_EMAIL_USER}")
  fi
  if [[ -n "${RENSI_EMAIL_PASSWORD:-}" ]]; then
    env_vars+=("EMAIL_PASSWORD=${RENSI_EMAIL_PASSWORD}")
  fi
  if [[ -n "${SMTP_SERVER:-}" ]]; then
    env_vars+=("SMTP_SERVER=${SMTP_SERVER}")
  fi
  if [[ -n "${SMTP_PORT:-}" ]]; then
    env_vars+=("SMTP_PORT=${SMTP_PORT}")
  fi
  if [[ -n "${IMAP_SERVER:-}" ]]; then
    env_vars+=("IMAP_SERVER=${IMAP_SERVER}")
  fi

  nohup env "${env_vars[@]}" \
           "${python_cmd}" -m uvicorn backend.app:app \
    --host 0.0.0.0 \
    --port "${BACKEND_PORT}" \
    "${RELOAD_ARGS[@]}" \
    > "${LOG_DIR}/backend.log" 2>&1 &

  local pid=$!
  echo "${pid}" > "${BACKEND_PID_FILE}"
  echo "   PID: ${pid}"

  sleep 3

  if ! kill -0 "${pid}" >/dev/null 2>&1; then
    echo "❌ Backend exited immediately after start"
    echo "   See ${LOG_DIR}/backend.log"
    echo ""
    echo "   Last log lines:"
    if [[ -f "${LOG_DIR}/backend.log" ]]; then
      tail -n 30 "${LOG_DIR}/backend.log" || true
    else
      echo "   (log file missing)"
    fi
    rm -f "${BACKEND_PID_FILE}"
    return 1
  fi

  echo "   ✅ Backend process is running"

  if ! wait_for_service "http://localhost:${BACKEND_PORT}/health" "" "Backend" 60; then
    echo "❌ Backend did not become healthy in time; see ${LOG_DIR}/backend.log"
    tail -n 40 "${LOG_DIR}/backend.log" || true
    return 1
  fi
}

open_ui() {
  echo "🌐 Opening UI: ${UI_URL}"
  if command -v wslview >/dev/null 2>&1; then
    wslview "${UI_URL}" >/dev/null 2>&1 || true
  elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "${UI_URL}" >/dev/null 2>&1 || true
  else
    echo "Open manually in a browser: ${UI_URL}"
  fi
}

# ============================================
# Stop helpers
# ============================================

stop_process() {
  local pid_file=$1
  local name=$2

  if ! is_process_alive "${pid_file}" >/dev/null; then
    echo "ℹ️  ${name} is not running"
    [[ -f "${pid_file}" ]] && rm -f "${pid_file}"
    return 0
  fi

  local pid
  pid="$(cat "${pid_file}")"
  echo "⏹  Stopping ${name} (PID ${pid})"

  kill -TERM "${pid}" >/dev/null 2>&1 || true

  if [[ "${name}" == *"vLLM"* ]]; then
    echo "   Cleaning vLLM child processes..."
    pkill -P "${pid}" >/dev/null 2>&1 || true
    pkill -f "VLLM::EngineCore" >/dev/null 2>&1 || true
    pkill -f "vllm.entrypoints.openai.api_server" >/dev/null 2>&1 || true
  fi

  for _ in {1..20}; do
    if kill -0 "${pid}" >/dev/null 2>&1; then
      sleep 1
    else
      break
    fi
  done

  if kill -0 "${pid}" >/dev/null 2>&1; then
    echo "⚠️  ${name} did not exit cleanly; sending SIGKILL."
    kill -9 "${pid}" >/dev/null 2>&1 || true
    if [[ "${name}" == *"vLLM"* ]]; then
      pkill -9 -P "${pid}" >/dev/null 2>&1 || true
      pkill -9 -f "VLLM::EngineCore" >/dev/null 2>&1 || true
      pkill -9 -f "vllm.entrypoints.openai.api_server" >/dev/null 2>&1 || true
    fi
  fi
  rm -f "${pid_file}"
}

status_process() {
  local pid_file=$1
  local name=$2
  if is_process_alive "${pid_file}" >/dev/null; then
    local pid
    pid="$(cat "${pid_file}")"
    echo "✅ ${name} running (PID ${pid})"
  else
    echo "❌ ${name} not running"
  fi
}

# ============================================
# Commands
# ============================================

cmd_start() {
  echo "🚀 Starting services..."
  echo "   Config: ${ROOT_DIR}/.env"
  echo "   Provider: ${PROVIDER_NAME}"
  echo "   Database: ${DB_PATH}"
  if [[ "${IS_LOCAL_PROVIDER}" == "true" ]]; then
    echo "   vLLM port: ${VLLM_PORT}"
  else
    echo "   vLLM port: (remote provider; no local vLLM port)"
  fi
  echo "   Backend port: ${BACKEND_PORT}"
  echo ""
  if [[ "${IS_LOCAL_PROVIDER}" == "true" ]]; then
    start_vllm
  else
    echo "ℹ️  Mode ${PROVIDER_NAME}: skipping local vLLM"
  fi
  start_backend
  if [[ "${SKIP_HEARTBEAT:-false}" == "true" ]]; then
    echo "ℹ️  SKIP_HEARTBEAT=true: heartbeat not started (eval / isolation)"
  else
    start_heartbeat
  fi
  open_ui
  echo ""
  echo "🎯 Done. Logs under ${LOG_DIR}/"
}

cmd_stop() {
  stop_process "${BACKEND_PID_FILE}" "Backend service"
  stop_process "${HEARTBEAT_PID_FILE}" "Heartbeat service"
  if [[ "${IS_LOCAL_PROVIDER}" == "true" ]]; then
    stop_process "${VLLM_PID_FILE}" "vLLM service"
  fi

  echo "🧹 Cleaning stray processes..."
  pkill -f "uvicorn backend.app:app" >/dev/null 2>&1 || true
  pkill -f "scripts/heartbeat.py" >/dev/null 2>&1 || true
  pkill -f "vllm.entrypoints.openai.api_server" >/dev/null 2>&1 || true
  pkill -f "VLLM::EngineCore" >/dev/null 2>&1 || true

  sleep 2

  if command -v nvidia-smi >/dev/null 2>&1; then
    local gpu_mem
    gpu_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "0")
    if [[ -n "${gpu_mem}" ]] && [[ "${gpu_mem}" -gt 1000 ]]; then
      echo "⚠️  GPU memory still high (${gpu_mem} MB); wait a few seconds for the driver to free VRAM."
    fi
  fi

  echo "🛑 All services stopped."
}

cmd_status() {
  echo "📊 Service status"
  if [[ "${IS_LOCAL_PROVIDER}" == "true" ]]; then
    status_process "${VLLM_PID_FILE}" "vLLM service"
  else
    echo "ℹ️  Mode ${PROVIDER_NAME}: no local vLLM"
  fi
  status_process "${BACKEND_PID_FILE}" "Backend service"
  status_process "${HEARTBEAT_PID_FILE}" "Heartbeat service"
  echo ""
  echo "📄 Logs"
  echo "  vLLM      -> ${LOG_DIR}/vllm.log"
  echo "  backend   -> ${LOG_DIR}/backend.log"
  echo "  heartbeat -> ${LOG_DIR}/heartbeat.log"
  echo ""
  echo "⚙️  Config"
  echo "  Database: ${DB_PATH}"
  echo "  Provider: ${PROVIDER_NAME}"
  echo "  Model:    ${MODEL_ID}"
  echo "  Base URL: ${VLLM_BASE_URL}"
}

cmd_restart() {
  cmd_stop
  sleep 2
  cmd_start
}

usage() {
  cat <<EOF
Usage: $(basename "$0") [start|stop|status|restart|open]

Commands:
  start    Start vLLM (if local), backend, heartbeat; try to open the UI
  stop     Stop everything managed here
  status   Show PIDs and log paths
  restart  stop then start
  open     Open the UI only (${UI_URL})

Environment:
  Read from ${ROOT_DIR}/.env (and your shell). Common keys:
    - DB_PATH
    - VLLM_PORT (default 8000)
    - BACKEND_PORT (default 8080)
    - VLLM_BASE_URL
    - VLLM_API_KEY
    - MODEL_ID

Examples:
  $0 start
  $0 status
EOF
}

main() {
  local cmd=${1:-}
  case "${cmd}" in
    start)   cmd_start ;;
    stop)    cmd_stop ;;
    status)  cmd_status ;;
    restart) cmd_restart ;;
    open)    open_ui ;;
    *)       usage; exit 1 ;;
  esac
}

main "$@"
