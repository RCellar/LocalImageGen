#!/usr/bin/env bash
set -euo pipefail

MODELS_DIR="/models"
INVOKEAI_ROOT="${INVOKEAI_ROOT:-/invokeai}"
INVOKEAI_PORT="${INVOKEAI_PORT:-9090}"

echo "=== InvokeAI Entrypoint ==="

# Ensure the InvokeAI root directory exists
mkdir -p "$INVOKEAI_ROOT"

# List available image models from shared mount
if [ -d "$MODELS_DIR" ]; then
    echo "Shared models directory contents:"
    ls -1 "$MODELS_DIR" 2>/dev/null || echo "  (empty)"
    echo ""
    echo "Import models via the Model Manager UI at http://localhost:${INVOKEAI_PORT}"
    echo "Point to /models/<model-name> to import shared models."
else
    echo "WARNING: Shared models directory $MODELS_DIR not found"
fi

echo "==========================="

# Execute the original InvokeAI entrypoint, which handles user setup
# and then runs the CMD (invokeai-web)
# InvokeAI defaults: host=0.0.0.0, port=9090 — no config file needed
exec /opt/invokeai/docker-entrypoint.sh invokeai-web --root "$INVOKEAI_ROOT"
