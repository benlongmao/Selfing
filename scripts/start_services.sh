#!/bin/bash
# Quick start helper for vLLM and the backend.

set -e

echo "=== Service launcher ==="

# Require virtualenv
if [ ! -f ".venv/bin/activate" ]; then
    echo "❌ Virtualenv missing. From repo root run: bash install_s_project.sh"
    exit 1
fi

source .venv/bin/activate

SERVICE=$1

if [ "$SERVICE" = "vllm" ]; then
    echo "Starting vLLM..."

    # Local model path (symlinks allowed)
    MODEL_PATH=""
    if [ -e "models/qwen/Qwen2.5-7B-Instruct" ] && [ -f "models/qwen/Qwen2.5-7B-Instruct/config.json" ]; then
        MODEL_PATH="$(pwd)/models/qwen/Qwen2.5-7B-Instruct"
        echo "✅ Using local model: $MODEL_PATH"
    elif [ -d "models/qwen/Qwen2___5-7B-Instruct" ] && [ -f "models/qwen/Qwen2___5-7B-Instruct/config.json" ]; then
        MODEL_PATH="$(pwd)/models/qwen/Qwen2___5-7B-Instruct"
        echo "✅ Using local model: $MODEL_PATH"
    else
        MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
        export HF_ENDPOINT="https://hf-mirror.com"
        echo "⚠️  Local model not found; will download from Hugging Face: $MODEL_PATH"
    fi

    python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL_PATH" \
        --trust-remote-code \
        --port 8000 \
        --api-key token-abc123
elif [ "$SERVICE" = "backend" ]; then
    echo "Starting backend..."
    export DB_PATH="data.db"
    export VLLM_BASE_URL="http://localhost:8000/v1"
    export VLLM_API_KEY="token-abc123"

    # Detect model id from vLLM
    MODEL_ID=$(curl -s -H "Authorization: Bearer token-abc123" http://localhost:8000/v1/models 2>/dev/null | python3 -c "import sys, json; d=json.load(sys.stdin); print(d['data'][0]['id'] if d.get('data') else 'Qwen/Qwen2.5-7B-Instruct')" 2>/dev/null || echo "Qwen/Qwen2.5-7B-Instruct")
    export MODEL_ID="$MODEL_ID"
    echo "Detected MODEL_ID: $MODEL_ID"

    # Backend in background, then try to open /ui
    uvicorn backend.app:app --host 0.0.0.0 --port 8080 --reload &
    BACKEND_PID=$!

    sleep 2
    echo "Trying to open UI in browser: http://localhost:8080/ui/"
    if command -v xdg-open >/dev/null 2>&1; then
        xdg-open "http://localhost:8080/ui/" >/dev/null 2>&1 || true
    elif command -v wslview >/dev/null 2>&1; then
        # Common on WSL
        wslview "http://localhost:8080/ui/" >/dev/null 2>&1 || true
    else
        # Fallback: Python webbrowser
        python -m webbrowser "http://localhost:8080/ui/" >/dev/null 2>&1 || true
    fi

    # Keep terminal attached to backend like before
    wait "$BACKEND_PID"
else
    echo "Usage: $0 [vllm|backend]"
    echo ""
    echo "Examples:"
    echo "  $0 vllm     # terminal 1: vLLM"
    echo "  $0 backend  # terminal 2: backend"
    exit 1
fi
