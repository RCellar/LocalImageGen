#!/usr/bin/env bash
set -euo pipefail

MODELS_DIR="/models"
INVOKEAI_ROOT="${INVOKEAI_ROOT:-/invokeai}"
INVOKEAI_PORT="${INVOKEAI_PORT:-9090}"

echo "=== InvokeAI Entrypoint ==="

# Ensure the InvokeAI root directory exists
mkdir -p "$INVOKEAI_ROOT"

# Set HuggingFace cache inside the InvokeAI root
export HF_HOME="${HF_HOME:-$INVOKEAI_ROOT/.cache/huggingface}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$INVOKEAI_ROOT/.matplotlib}"

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

# Run invokeai-web directly as container root.
# In rootless podman, container root = host user (no privilege escalation).
# This avoids the UID mapping issues with bind mounts that make files
# appear owned by different UIDs inside the container.
exec invokeai-web --root "$INVOKEAI_ROOT"
