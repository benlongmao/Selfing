#!/bin/bash
#
# One-click service manager: start / stop / status / restart.
# May launch local vLLM, backend, and open the web UI.
#
# Notes: prefer .env; env overrides; ports/DB from .env; improved model path checks.

set -e
# Relaxed error handling on purpose
# -e: exit on first error
# -u / pipefail omitted for broader shell compatibility

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
PID_DIR="${ROOT_DIR}/run"
VENV_ACTIVATE="${ROOT_DIR}/.venv/bin/activate"
VLLM_PID_FILE="${PID_DIR}/vllm.pid"
BACKEND_PID_FILE="${PID_DIR}/backend.pid"
HEARTBEAT_PID_FILE="${PID_DIR}/heartbeat.pid"

# ============================================
# Load .env first
# ============================================
if [[ -f "${ROOT_DIR}/.env" ]]; then
  # shellcheck disable=SC2046
  set -o allexport
  # shellcheck source=/dev/null
  source "${ROOT_DIR}/.env" || {
    echo "⚠️  Warning: could not load .env; continuing with defaults"
  }
  set +o allexport
  echo "✅ Loaded .env"
fi

# ============================================
# Optional: read YAML when env not set
# ============================================
if command -v python3 >/dev/null 2>&1; then
  if [[ -f "${ROOT_DIR}/config/settings.yaml" ]]; then
    # Use Python to parse YAML and export missing env vars
    _S_ROOT="${ROOT_DIR}" eval "$(python3 << 'PYTHON_EOF'
import yaml
import os
import sys

try:
    with open(os.path.join(os.environ.get('_S_ROOT', '.'), 'config/settings.yaml'), 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f) or {}
    
    # model_provider
    model_provider = config.get('system', {}).get('model_provider', '')
    if model_provider and not os.environ.get('MODEL_PROVIDER'):
        print(f'export MODEL_PROVIDER="{model_provider}"')

    # heartbeat (for manage_services vs legacy scripts/heartbeat.py)
    system_config = config.get('system', {})
    heartbeat_enabled = system_config.get('heartbeat_enabled')
    if heartbeat_enabled is not None and not os.environ.get('HEARTBEAT_ENABLED'):
        print(f'export HEARTBEAT_ENABLED="{str(heartbeat_enabled).lower()}"')
    heartbeat_interval = system_config.get('heartbeat_interval')
    if heartbeat_interval is not None and not os.environ.get('HEARTBEAT_INTERVAL'):
        print(f'export HEARTBEAT_INTERVAL="{heartbeat_interval}"')
    
    # deepseek block
    deepseek_config = config.get('models', {}).get('deepseek', {})
    if deepseek_config:
        if not os.environ.get('DEEPSEEK_BASE_URL') and deepseek_config.get('base_url'):
            print(f'export DEEPSEEK_BASE_URL="{deepseek_config["base_url"]}"')
        if not os.environ.get('DEEPSEEK_API_KEY') and deepseek_config.get('api_key'):
            print(f'export DEEPSEEK_API_KEY="{deepseek_config["api_key"]}"')
        if not os.environ.get('DEEPSEEK_MODEL') and deepseek_config.get('model_id'):
            print(f'export DEEPSEEK_MODEL="{deepseek_config["model_id"]}"')
    
    # [disabled] kimi block — kept for future provider switch
    # kimi_config = config.get('models', {}).get('kimi', {})
    # if kimi_config:
    #     if not os.environ.get('KIMI_BASE_URL') and kimi_config.get('base_url'):
    #         print(f'export KIMI_BASE_URL="{kimi_config["base_url"]}"')
    #     if not os.environ.get('MOONSHOT_API_KEY') and not os.environ.get('KIMI_API_KEY') and kimi_config.get('api_key'):
    #         print(f'export MOONSHOT_API_KEY="{kimi_config["api_key"]}"')
    #         print(f'export KIMI_API_KEY="{kimi_config["api_key"]}"')
    #     if not os.environ.get('KIMI_MODEL') and kimi_config.get('model_id'):
    #         print(f'export KIMI_MODEL="{kimi_config["model_id"]}"')
    
    # vllm block (fallback when env unset)
    vllm_config = config.get('models', {}).get('vllm', {})
    if vllm_config:
        if not os.environ.get('VLLM_BASE_URL') and vllm_config.get('base_url'):
            print(f'export VLLM_BASE_URL="{vllm_config["base_url"]}"')
        if not os.environ.get('VLLM_API_KEY') and vllm_config.get('api_key'):
            print(f'export VLLM_API_KEY="{vllm_config["api_key"]}"')
        if not os.environ.get('MODEL_ID') and vllm_config.get('model_id'):
            model_id = vllm_config.get('model_id')
            if model_id:  # only export non-empty
                print(f'export MODEL_ID="{model_id}"')
    
    # claude block
    claude_config = config.get('models', {}).get('claude', {})
    if claude_config:
        if not os.environ.get('CLAUDE_BASE_URL') and claude_config.get('base_url'):
            print(f'export CLAUDE_BASE_URL="{claude_config["base_url"]}"')
        if not os.environ.get('CLAUDE_API_KEY') and claude_config.get('api_key'):
            print(f'export CLAUDE_API_KEY="{claude_config["api_key"]}"')
        if not os.environ.get('CLAUDE_MODEL') and claude_config.get('model_id'):
            print(f'export CLAUDE_MODEL="{claude_config["model_id"]}"')

    # openai block
    openai_config = config.get('models', {}).get('openai', {})
    if openai_config:
        if not os.environ.get('OPENAI_BASE_URL') and openai_config.get('base_url'):
            print(f'export OPENAI_BASE_URL="{openai_config["base_url"]}"')
        if not os.environ.get('OPENAI_API_KEY') and openai_config.get('api_key'):
            print(f'export OPENAI_API_KEY="{openai_config["api_key"]}"')
        if not os.environ.get('OPENAI_MODEL') and openai_config.get('model_id'):
            print(f'export OPENAI_MODEL="{openai_config["model_id"]}"')
except Exception as e:
    # Fail silently; script continues with .env / defaults
    pass
PYTHON_EOF
    )"
  fi
fi

# ============================================
# Environment (from .env with defaults)
# ============================================

# Ports
VLLM_PORT="${VLLM_PORT:-8000}"
BACKEND_PORT="${BACKEND_PORT:-8080}"

# Default vLLM client: do not assume token-abc123 for remote gateways (stable 401). Local vLLM gets a placeholder below.
VLLM_API_KEY="${VLLM_API_KEY:-}"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:${VLLM_PORT}/v1}"
# Fallback only if unset in settings.yaml / .env; chat timeouts follow app config
VLLM_TIMEOUT="${VLLM_TIMEOUT:-600}"

# Provider selection
MODEL_PROVIDER="$(echo "${MODEL_PROVIDER:-local_vllm}" | tr '[:upper:]' '[:lower:]')"
DEEPSEEK_BASE_URL="${DEEPSEEK_BASE_URL:-https://api.deepseek.com/v1}"
DEEPSEEK_MODEL="${DEEPSEEK_MODEL:-deepseek-chat}"
DEEPSEEK_API_KEY="${DEEPSEEK_API_KEY:-}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"
OPENAI_MODEL="${OPENAI_MODEL:-gpt-4o-mini}"
OPENAI_API_KEY="${OPENAI_API_KEY:-}"
HEARTBEAT_ENABLED="$(echo "${HEARTBEAT_ENABLED:-true}" | tr '[:upper:]' '[:lower:]')"
HEARTBEAT_INTERVAL="${HEARTBEAT_INTERVAL:-1800}"
LEGACY_HEARTBEAT_SCRIPT="$(echo "${LEGACY_HEARTBEAT_SCRIPT:-false}" | tr '[:upper:]' '[:lower:]')"
# [disabled] Kimi — kept for future switch
# KIMI_BASE_URL="${KIMI_BASE_URL:-https://api.moonshot.cn/v1}"
# KIMI_MODEL="${KIMI_MODEL:-kimi-k2.5}"
# KIMI_API_KEY="${MOONSHOT_API_KEY:-${KIMI_API_KEY:-}}"
IS_LOCAL_PROVIDER="true"
PROVIDER_NAME="Local vLLM"

# Database
DB_PATH="${DB_PATH:-${ROOT_DIR}/data.db}"
# ========================================================
# [Eval] Allow overriding DB path for benchmarks / isolated runs (avoid clobbering main data.db)
# Note: .env often sets DB_PATH=data.db; sourcing .env overwrites the shell. Use DB_PATH_OVERRIDE
# as the final override when you need a different DB regardless of .env.
# ========================================================
if [[ -n "${DB_PATH_OVERRIDE:-}" ]]; then
  DB_PATH="${DB_PATH_OVERRIDE}"
fi
# Resolve relative DB_PATH against repo root
if [[ "${DB_PATH}" != /* ]]; then
  DB_PATH="${ROOT_DIR}/${DB_PATH}"
fi

# Model id — should be set in .env for local vLLM
if [[ -z "${MODEL_ID:-}" ]]; then
  echo "⚠️  Warning: MODEL_ID is unset; configure it in .env"
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
      echo "⚠️  DEEPSEEK_API_KEY not set; add it to .env."
    fi
    ;;
  claude_api)
    IS_LOCAL_PROVIDER="false"
    PROVIDER_NAME="Claude API"
    CLAUDE_BASE_URL="${CLAUDE_BASE_URL:-https://api.anthropic.com}"
    CLAUDE_API_KEY="${CLAUDE_API_KEY:-}"
    CLAUDE_MODEL="${CLAUDE_MODEL:-claude-opus-4-5}"
    # Do not overwrite VLLM_* for routing; Claude uses a separate adapter — key presence check only
    if [[ -z "${CLAUDE_API_KEY}" ]]; then
      echo "⚠️  CLAUDE_API_KEY not set; add it to .env."
    fi
    ;;
  openai_api)
    IS_LOCAL_PROVIDER="false"
    PROVIDER_NAME="OpenAI API"
    OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"
    OPENAI_API_KEY="${OPENAI_API_KEY:-}"
    OPENAI_MODEL="${OPENAI_MODEL:-gpt-4o-mini}"
    # Reuse the same probe/display vars as other HTTP providers
    VLLM_BASE_URL="${OPENAI_BASE_URL}"
    if [[ -n "${OPENAI_API_KEY}" ]]; then
      VLLM_API_KEY="${OPENAI_API_KEY}"
    fi
    MODEL_ID="${OPENAI_MODEL}"
    VLLM_MODEL="${MODEL_ID}"
    if [[ -z "${OPENAI_API_KEY}" ]]; then
      echo "⚠️  OPENAI_API_KEY not set; add it to .env."
    fi
    ;;
  # OpenAI-compatible: local vLLM needs a process; remote gateways (e.g. Aiberm) are HTTP-only
  vllm)
    if [[ "${VLLM_BASE_URL:-}" == *"localhost"* ]] || [[ "${VLLM_BASE_URL:-}" == *"127.0.0.1"* ]]; then
      IS_LOCAL_PROVIDER="true"
      PROVIDER_NAME="Local vLLM (OpenAI-compatible)"
    else
      IS_LOCAL_PROVIDER="false"
      PROVIDER_NAME="OpenAI-compatible API (remote gateway)"
      if [[ -z "${VLLM_API_KEY}" ]]; then
        echo "⚠️  VLLM_API_KEY missing; set a real key in .env (e.g. sk-... for Aiberm)."
      fi
      if [[ -z "${VLLM_BASE_URL:-}" ]]; then
        echo "⚠️  VLLM_BASE_URL unset; set it in .env (e.g. https://aiberm.com/v1)."
      fi
    fi
    ;;
  # [disabled] kimi_api — kept for future switch
  # kimi_api)
  #   IS_LOCAL_PROVIDER="false"
  #   PROVIDER_NAME="Kimi 2.5 API"
  #   VLLM_BASE_URL="${KIMI_BASE_URL}"
  #   if [[ -n "${KIMI_API_KEY}" ]]; then
  #     VLLM_API_KEY="${KIMI_API_KEY}"
  #   fi
  #   MODEL_ID="${KIMI_MODEL}"
  #   VLLM_MODEL="${MODEL_ID}"
  #   if [[ -z "${VLLM_API_KEY}" ]]; then
  #     echo "⚠️  MOONSHOT_API_KEY or KIMI_API_KEY not set; add to .env."
  #   fi
  #   ;;
  *)
    PROVIDER_NAME="Local vLLM"
    ;;
esac

# Local vLLM server still needs some API key string; remote providers must set a real key in .env
if [[ "${IS_LOCAL_PROVIDER}" == "true" ]] && [[ -z "${VLLM_API_KEY:-}" ]]; then
  VLLM_API_KEY="token-abc123"
fi

# Optional tools
TAVILY_API_KEY="${TAVILY_API_KEY:-}"

# Web UI
UI_URL="http://localhost:${BACKEND_PORT}/ui/"

# ============================================
# Ensure log/pid dirs
# ============================================
mkdir -p "${LOG_DIR}" "${PID_DIR}"

# ============================================
# Export for child processes
# ============================================
export TAVILY_API_KEY
export DB_PATH
export MODEL_PROVIDER
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
    echo "Run first: bash ${ROOT_DIR}/install_s_project.sh"
    exit 1
  fi
}

activate_venv() {
  # shellcheck source=/dev/null
  if [[ -f "${VENV_ACTIVATE}" ]]; then
    source "${VENV_ACTIVATE}"
    return 0
  else
    echo "❌ Virtualenv activate script missing: ${VENV_ACTIVATE}"
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
  # Prefer VLLM_MODEL from env
  if [[ -n "${VLLM_MODEL}" ]]; then
    # If it looks like a path, require config.json
    if [[ -e "${ROOT_DIR}/${VLLM_MODEL}/config.json" ]]; then
      echo "${ROOT_DIR}/${VLLM_MODEL}"
      return 0
    elif [[ -e "${VLLM_MODEL}/config.json" ]]; then
      echo "${VLLM_MODEL}"
      return 0
    fi
    # Otherwise treat as Hugging Face model id (e.g. deepseek-chat)
    echo "${VLLM_MODEL}"
    return 0
  fi
  
  # No model configured
  echo ""
  return 1
}

# Per-probe timeouts: without --max-time, a zombie listener can make curl hang indefinitely.
CURL_CONNECT_TIMEOUT="${CURL_CONNECT_TIMEOUT:-3}"
CURL_MAX_TIME="${CURL_MAX_TIME:-8}"

_curl_probe() {
  local url=$1
  local header=$2
  if [[ -n "${header}" ]]; then
    curl -sSf --connect-timeout "${CURL_CONNECT_TIMEOUT}" --max-time "${CURL_MAX_TIME}" \
      -H "${header}" "${url}" >/dev/null 2>&1
  else
    curl -sSf --connect-timeout "${CURL_CONNECT_TIMEOUT}" --max-time "${CURL_MAX_TIME}" \
      "${url}" >/dev/null 2>&1
  fi
}

backend_port_in_use() {
  local port=$1
  if command -v ss >/dev/null 2>&1; then
    ss -tln 2>/dev/null | grep -qE ":${port}\\s" && return 0
    return 1
  fi
  if command -v netstat >/dev/null 2>&1; then
    netstat -tln 2>/dev/null | grep -qE ":${port}\\s" && return 0
    return 1
  fi
  return 1
}

wait_for_service() {
  local url=$1
  local header=$2
  local name=$3
  local timeout=${4:-90}

  echo "⏳  Waiting for ${name} at ${url} ..."
  for ((i=0; i<timeout; i++)); do
    if [[ -n "${header}" ]]; then
      if _curl_probe "${url}" "${header}"; then
        echo "✅ ${name} is ready"
        return 0
      fi
    else
      if _curl_probe "${url}" ""; then
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
  echo "⚠️  Warning: curl not found; health checks may fail"
fi

# ============================================
# start_vllm
# ============================================

start_vllm() {
  if [[ "${IS_LOCAL_PROVIDER}" != "true" ]]; then
    echo "ℹ️  Mode is ${PROVIDER_NAME}; skipping local vLLM."
    return 0
  fi

  if is_process_alive "${VLLM_PID_FILE}" >/dev/null; then
    echo "ℹ️  vLLM already running (PID $(cat "${VLLM_PID_FILE}"))"
    return 0
  fi

  if [[ -z "${VLLM_MODEL:-}" ]] && [[ -z "${MODEL_ID:-}" ]]; then
    echo "❌ VLLM_MODEL / MODEL_ID not configured; cannot start local vLLM"
    echo "   Use MODEL_PROVIDER=deepseek_api for API mode, or set VLLM_MODEL in .env"
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
    echo "❌ Model path empty; check VLLM_MODEL"
    return 1
  fi
  
  echo "▶️  Starting vLLM"
  echo "   Model: ${model_path}"
  echo "   Port: ${VLLM_PORT}"
  echo "   API key (prefix): ${VLLM_API_KEY:0:10}..."

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
    echo "   See: ${LOG_DIR}/vllm.log"
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
  if [[ "${HEARTBEAT_ENABLED}" != "true" ]]; then
    echo "ℹ️  Heartbeat disabled in settings/.env (HEARTBEAT_ENABLED=${HEARTBEAT_ENABLED}); skipping legacy heartbeat script."
    return 0
  fi

  if [[ "${LEGACY_HEARTBEAT_SCRIPT}" != "true" ]]; then
    echo "ℹ️  Using in-app HeartbeatService (interval=${HEARTBEAT_INTERVAL}s); not starting scripts/heartbeat.py."
    return 0
  fi

  if is_process_alive "${HEARTBEAT_PID_FILE}" >/dev/null; then
    echo "ℹ️  Heartbeat already running (PID $(cat "${HEARTBEAT_PID_FILE}"))"
    return 0
  fi

  echo "▶️  Starting legacy heartbeat script (compatibility mode)"

  local python_cmd
  if [[ -f "${ROOT_DIR}/.venv/bin/python" ]]; then
    python_cmd="${ROOT_DIR}/.venv/bin/python"
  else
    python_cmd="python3"
  fi

  nohup env \
    "API_BASE=http://localhost:${BACKEND_PORT}" \
    "HEARTBEAT_INTERVAL=${HEARTBEAT_INTERVAL}" \
    "${python_cmd}" "${ROOT_DIR}/scripts/heartbeat.py" \
    > "${LOG_DIR}/heartbeat.log" 2>&1 &

  local pid=$!
  echo "${pid}" > "${HEARTBEAT_PID_FILE}"
  echo "   PID: ${pid}"
  echo "   ✅ Legacy heartbeat started (interval: ${HEARTBEAT_INTERVAL}s)"
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
      echo "⚠️  vLLM not reachable yet; starting backend anyway."
    fi
  else
    wait_for_service "${VLLM_BASE_URL}/models" "Authorization: Bearer ${VLLM_API_KEY}" "${PROVIDER_NAME}" 10 || true
  fi

  local model_id
  model_id="$(detect_model_id)"
  echo "▶️  Starting backend"
  echo "   Database: ${DB_PATH}"
  echo "   Model ID: ${model_id}"
  echo "   Port: ${BACKEND_PORT}"
  echo "   Upstream base URL: ${VLLM_BASE_URL}"

  RELOAD_ARGS=()
  if [[ "${ENABLE_RELOAD:-false}" == "true" ]]; then
    RELOAD_ARGS=(--reload)
    echo "⚠️  Dev mode: --reload enabled (.cache churn may restart often)"
  else
    echo "ℹ️  Production mode: --reload disabled (avoids vLLM temp file reload storms)"
  fi
  
  mkdir -p "${LOG_DIR}"
  
  local python_cmd
  if [[ -f "${ROOT_DIR}/.venv/bin/python" ]]; then
    python_cmd="${ROOT_DIR}/.venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    python_cmd="python3"
  else
    echo "❌ No Python interpreter found"
    return 1
  fi
  
  if ! "${python_cmd}" -m uvicorn --help >/dev/null 2>&1; then
    echo "❌ uvicorn not installed in this environment"
    echo "   Try: source ${VENV_ACTIVATE} && pip install uvicorn"
    return 1
  fi

  # Syntax + import checks (cold start is slower; use SKIP_* to bypass — see usage)
  if [[ "${SKIP_BACKEND_SYNTAX_CHECK:-false}" != "true" ]]; then
    echo "⏳  compileall syntax check on backend/ ..."
    if ! bash "${ROOT_DIR}/scripts/check_backend.sh"; then
      echo "❌ Backend syntax check failed; aborting start."
      echo "   To skip (not recommended): export SKIP_BACKEND_SYNTAX_CHECK=true"
      return 1
    fi
  else
    echo "ℹ️  Skipping backend syntax check (SKIP_BACKEND_SYNTAX_CHECK=true)"
  fi

  if [[ "${SKIP_BACKEND_IMPORT_CHECK:-false}" != "true" ]]; then
    echo "⏳  Verifying import backend.app (same load path as uvicorn) ..."
    if ! bash "${ROOT_DIR}/scripts/verify_backend_imports.sh"; then
      echo "❌ Backend import check failed; aborting start."
      echo "   To skip (not recommended): export SKIP_BACKEND_IMPORT_CHECK=true"
      return 1
    fi
  else
    echo "ℹ️  Skipping backend import check (SKIP_BACKEND_IMPORT_CHECK=true)"
  fi

  if backend_port_in_use "${BACKEND_PORT}"; then
    echo "❌ Backend port ${BACKEND_PORT} is already in use."
    echo "   Stop the old process first: $(basename "$0") stop"
    echo "   Listener:"
    if command -v ss >/dev/null 2>&1; then
      ss -tlnp 2>/dev/null | grep -E ":${BACKEND_PORT}\\s" || true
    fi
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
  
  echo "   Launching uvicorn (${python_cmd})..."
  echo "   Env: DB_PATH=${DB_PATH}, VLLM_BASE_URL=${VLLM_BASE_URL}, MODEL_ID=${model_id}"
  
  local env_vars=(
    "DB_PATH=${DB_PATH}"
    "MODEL_PROVIDER=${MODEL_PROVIDER}"
    "VLLM_BASE_URL=${VLLM_BASE_URL}"
    "VLLM_API_KEY=${VLLM_API_KEY}"
    "MODEL_ID=${model_id}"
    "VLLM_MODEL=${model_id}"
    "PYTHONPATH=${PYTHONPATH}"
  )
  if [[ -n "${TAVILY_API_KEY:-}" ]]; then
    env_vars+=("TAVILY_API_KEY=${TAVILY_API_KEY}")
  fi
  # Forward email-related env to the backend process
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
    echo "   See: ${LOG_DIR}/backend.log"
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
  
  echo "   ✅ Backend process running"

  if ! wait_for_service "http://localhost:${BACKEND_PORT}/health" "" "Backend" 60; then
    echo "❌ Backend did not become healthy in time; see ${LOG_DIR}/backend.log"
    tail -n 40 "${LOG_DIR}/backend.log" || true
    return 1
  fi
}

open_ui() {
  echo "🌐  Opening UI: ${UI_URL}"
  if command -v wslview >/dev/null 2>&1; then
    wslview "${UI_URL}" >/dev/null 2>&1 || true
  elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "${UI_URL}" >/dev/null 2>&1 || true
  else
    echo "Open this URL in your browser: ${UI_URL}"
  fi
}

# ============================================
# stop_process / status
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
    echo "   Cleaning up vLLM child processes..."
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
  echo "🚀  Starting services..."
  echo "   Config: ${ROOT_DIR}/.env"
  echo "   Provider: ${PROVIDER_NAME}"
  echo "   Database: ${DB_PATH}"
  if [[ "${IS_LOCAL_PROVIDER}" == "true" ]]; then
    echo "   vLLM port: ${VLLM_PORT}"
  else
    echo "   vLLM port: (remote API — no local vLLM listener)"
  fi
  echo "   Backend port: ${BACKEND_PORT}"
  echo ""
  if [[ "${IS_LOCAL_PROVIDER}" == "true" ]]; then
    start_vllm
  else
    echo "ℹ️  Mode: ${PROVIDER_NAME} (skipping local vLLM)"
  fi
  start_backend
  if [[ "${SKIP_HEARTBEAT:-false}" == "true" ]]; then
    echo "ℹ️  SKIP_HEARTBEAT=true — not starting heartbeat (benchmarks / isolation)"
  else
    start_heartbeat
  fi
  open_ui
  echo ""
  echo "🎯  Done. Logs: ${LOG_DIR}/"
}

cmd_stop() {
  stop_process "${BACKEND_PID_FILE}" "Backend service"
  stop_process "${HEARTBEAT_PID_FILE}" "Heartbeat service"
  if [[ "${IS_LOCAL_PROVIDER}" == "true" ]]; then
    stop_process "${VLLM_PID_FILE}" "vLLM service"
  fi
  
  echo "🧹  Cleaning stray processes..."
  pkill -f "uvicorn backend.app:app" >/dev/null 2>&1 || true
  pkill -f "python -m uvicorn backend.app:app" >/dev/null 2>&1 || true
  pkill -f "scripts/heartbeat.py" >/dev/null 2>&1 || true
  pkill -f "vllm.entrypoints.openai.api_server" >/dev/null 2>&1 || true
  pkill -f "VLLM::EngineCore" >/dev/null 2>&1 || true
  
  sleep 2
  
  if command -v nvidia-smi >/dev/null 2>&1; then
    local gpu_mem
    gpu_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 || echo "0")
    if [[ -n "${gpu_mem}" ]] && [[ "${gpu_mem}" -gt 1000 ]]; then
      echo "⚠️  GPU memory still high (~${gpu_mem} MB); wait a few seconds for the driver to release."
    fi
  fi
  
  echo "🛑  All services stopped."
}

cmd_status() {
  echo "📊  Service status"
  if [[ "${IS_LOCAL_PROVIDER}" == "true" ]]; then
    status_process "${VLLM_PID_FILE}" "vLLM service"
  else
    echo "ℹ️  Mode: ${PROVIDER_NAME} (no local vLLM process)"
  fi
  status_process "${BACKEND_PID_FILE}" "Backend service"
  status_process "${HEARTBEAT_PID_FILE}" "Heartbeat service"
  echo ""
  echo "📄  Logs:"
  echo "  vLLM      -> ${LOG_DIR}/vllm.log"
  echo "  backend   -> ${LOG_DIR}/backend.log"
  echo "  heartbeat -> ${LOG_DIR}/heartbeat.log"
  echo ""
  echo "⚙️  Config:"
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
  start    Start vLLM (if local), backend, optional legacy heartbeat, open the web UI
  stop     Stop all managed services
  status   Show PIDs and log paths
  restart  stop then start
  open     Open the web UI only: ${UI_URL}

Environment:
  Values are read from ${ROOT_DIR}/.env (and optionally merged from config/settings.yaml).

  Common variables:
    MODEL_PROVIDER   deepseek_api | claude_api | openai_api | vllm
    DB_PATH          SQLite path (default: data.db under repo root)
    VLLM_PORT        Local vLLM port (default: 8000)
    BACKEND_PORT     FastAPI port (default: 8080)
    VLLM_BASE_URL    Upstream OpenAI-compatible base URL
    VLLM_API_KEY     Bearer token for that upstream
    MODEL_ID         Model name / HF id for routing
    HEARTBEAT_ENABLED        true|false (default true) — when false, skip scripts/heartbeat.py
    HEARTBEAT_INTERVAL       seconds (default 1800) — passed to legacy script if used
    LEGACY_HEARTBEAT_SCRIPT  true|false (default false) — true = run scripts/heartbeat.py; false = rely on in-app HeartbeatService

  Optional tuning:
    SKIP_BACKEND_SYNTAX_CHECK   true = skip compileall (faster, not recommended)
    SKIP_BACKEND_IMPORT_CHECK   true = skip import backend.app preflight
    SKIP_HEARTBEAT              true = skip start_heartbeat branch entirely
    CURL_CONNECT_TIMEOUT / CURL_MAX_TIME   per-probe limits for health checks (defaults: 3 / 8 seconds)

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
