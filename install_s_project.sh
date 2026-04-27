#!/usr/bin/env bash
# =============================================================================
# S project: one-shot install (venv + deps + optional DB/persona init)
#
# Usage (run from repo root, or invoke by absolute path):
#   bash install_s_project.sh
#   bash install_s_project.sh --china-mirror   # pip via Aliyun mirror (optional)
#   # Or set mirrors without the flag:
#   export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
#   export PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn   # if your mirror needs it
#   bash install_s_project.sh
#   bash install_s_project.sh --skip-init          # venv + pip only; skip init_*.py
#   bash install_s_project.sh --with-playwright    # also run playwright install (large)
#   bash install_s_project.sh --warm-embedder      # first-time embedder load/download (needs network)
#
# Backward compatible:
#   bash scripts/install_s_project.sh [...]   # forwards to this file
#
# Notes:
#   - This script is never run automatically; invoke it manually.
#   - Init scripts write data.db in the repo root; back up an existing DB before re-running init.
#   - API keys still belong in .env (copy from .env.example).
# =============================================================================
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

CHINA_MIRROR=0
SKIP_INIT=0
WITH_PLAYWRIGHT=0
WARM_EMBEDDER=0

for arg in "$@"; do
  case "$arg" in
    --china-mirror) CHINA_MIRROR=1 ;;
    --skip-init) SKIP_INIT=1 ;;
    --with-playwright) WITH_PLAYWRIGHT=1 ;;
    --warm-embedder) WARM_EMBEDDER=1 ;;
    -h|--help)
      grep '^#' "$0" | head -n 26 | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg (use --help)" >&2
      exit 1
      ;;
  esac
done

echo "=== S project install ==="
echo "Repo root: $REPO_ROOT"

# --- Python version ---
if ! command -v python3 >/dev/null 2>&1; then
  echo "Error: python3 not found; install Python 3.10+" >&2
  exit 1
fi
PY_VER="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' || {
  echo "Error: Python 3.10+ required; current: $PY_VER" >&2
  exit 1
}
echo "Python: $(command -v python3) ($PY_VER)"

# --- pip index (optional): --china-mirror OR pre-set PIP_INDEX_URL ---
if [[ "$CHINA_MIRROR" -eq 1 ]]; then
  export PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}"
  export PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-mirrors.aliyun.com}"
  echo "pip index: $PIP_INDEX_URL"
  PIP_EXTRA=( -i "$PIP_INDEX_URL" --trusted-host "$PIP_TRUSTED_HOST" )
elif [[ -n "${PIP_INDEX_URL:-}" ]]; then
  echo "pip index (from environment): $PIP_INDEX_URL"
  PIP_EXTRA=( -i "$PIP_INDEX_URL" )
  if [[ -n "${PIP_TRUSTED_HOST:-}" ]]; then
    PIP_EXTRA+=( --trusted-host "$PIP_TRUSTED_HOST" )
  fi
else
  PIP_EXTRA=()
fi

# --- Virtualenv .venv ---
VENV_PY="$REPO_ROOT/.venv/bin/python"
VENV_PIP="$REPO_ROOT/.venv/bin/pip"
if [[ ! -x "$VENV_PY" ]]; then
  echo "Creating venv: $REPO_ROOT/.venv"
  python3 -m venv "$REPO_ROOT/.venv"
fi
echo "Using interpreter: $VENV_PY"

echo "Upgrading pip / wheel..."
"$VENV_PIP" install -U pip wheel "${PIP_EXTRA[@]}"

echo "Installing requirements.txt ..."
"$VENV_PIP" install -r "$REPO_ROOT/requirements.txt" "${PIP_EXTRA[@]}"

# --- Embedder: local English BGE only (BAAI/bge-small-en-v1.5; fork default) ---
# On-disk names may use underscores instead of dots (e.g. bge-small-en-v1___5).
# We do not probe bge-small-zh here so a stale Chinese cache is never auto-selected.
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$REPO_ROOT/models}"
export EMBEDDER_MODEL_SCOPE="${EMBEDDER_MODEL_SCOPE:-BAAI/bge-small-en-v1.5}"
EMBED_LOCAL=""
for cand in \
  "$REPO_ROOT/models/BAAI/bge-small-en-v1___5" \
  "$REPO_ROOT/models/BAAI/bge-small-en-v1.5" \
  "$REPO_ROOT/models/BAAI/bge-small-en-v1_5"
do
  if [[ -d "$cand" ]]; then
    EMBED_LOCAL="$cand"
    break
  fi
done
if [[ -n "$EMBED_LOCAL" ]]; then
  export EMBEDDER_MODEL="$EMBED_LOCAL"
  echo "Set EMBEDDER_MODEL (local English BGE): $EMBEDDER_MODEL"
else
  echo "No local English BGE cache dir; first run will pull BAAI/bge-small-en-v1.5 (ModelScope or HF; see backend/embedder.py)."
  echo "Optional: MODELSCOPE_CACHE=$MODELSCOPE_CACHE EMBEDDER_MODEL_SCOPE=$EMBEDDER_MODEL_SCOPE"
fi

# --- .env: copy from template if missing (never overwrite) ---
if [[ ! -f "$REPO_ROOT/.env" && -f "$REPO_ROOT/.env.example" ]]; then
  cp "$REPO_ROOT/.env.example" "$REPO_ROOT/.env"
  echo "Created .env from .env.example; edit and set DEEPSEEK_API_KEY, etc."
elif [[ ! -f "$REPO_ROOT/.env" ]]; then
  echo "Note: neither .env nor .env.example found; configure environment variables yourself."
fi

# --- Playwright (optional browser tooling) ---
if [[ "$WITH_PLAYWRIGHT" -eq 1 ]]; then
  echo "Running playwright install (may take a while)..."
  "$REPO_ROOT/.venv/bin/playwright" install chromium || true
fi

export PYTHONPATH="$REPO_ROOT"

# --- DB + persona init (same order as README) ---
if [[ "$SKIP_INIT" -eq 1 ]]; then
  echo "Skipped init_*.py (--skip-init)."
else
  if [[ -f "$REPO_ROOT/data.db" ]]; then
    echo "Warning: data.db already exists. Running init may overwrite or append rules; back up first." >&2
    echo "Press Ctrl+C within 5 seconds to cancel..."
    sleep 5 || true
  fi
  echo "Running init_persona_core.py ..."
  "$VENV_PY" "$REPO_ROOT/scripts/init_persona_core.py"
  echo "Running init_emotion_motivation.py ..."
  "$VENV_PY" "$REPO_ROOT/scripts/init_emotion_motivation.py"
  echo "Running init_new_dimensions.py ..."
  "$VENV_PY" "$REPO_ROOT/scripts/init_new_dimensions.py"
  echo "Init scripts finished."
fi

# --- Optional embedder warm-up: trigger SentenceTransformer load ---
if [[ "$WARM_EMBEDDER" -eq 1 ]]; then
  echo "Warming up embedder..."
  "$VENV_PY" -c "
import os, sys
os.chdir('$REPO_ROOT')
sys.path.insert(0, '$REPO_ROOT')
from backend.embedder import get_embedder
get_embedder().encode('install_s_project warm-up')
print('embedder ok')
"
fi

echo ""
echo "=== Install finished ==="
echo "Activate venv: source $REPO_ROOT/.venv/bin/activate"
echo "Start services: cd $REPO_ROOT && python start_server.py"
echo "(Or set PYTHONPATH=$REPO_ROOT and run uvicorn; see README)"
