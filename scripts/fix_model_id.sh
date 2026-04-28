#!/bin/bash
# Print the model id reported by vLLM so you can export MODEL_ID.

echo "=== Fix / inspect MODEL_ID ==="

MODEL_ID=$(curl -s -H "Authorization: Bearer token-abc123" http://localhost:8000/v1/models 2>/dev/null | python3 -c "import sys, json; d=json.load(sys.stdin); print(d['data'][0]['id'] if d.get('data') else '')" 2>/dev/null)

if [ -z "$MODEL_ID" ]; then
    echo "❌ Could not read model id; ensure vLLM is running."
    exit 1
fi

echo "Detected MODEL_ID: $MODEL_ID"
echo ""
echo "Export when starting the backend:"
echo "  export MODEL_ID=\"$MODEL_ID\""
echo ""
echo "Or restart the backend helper (it auto-detects):"
echo "  bash scripts/start_services.sh backend"
