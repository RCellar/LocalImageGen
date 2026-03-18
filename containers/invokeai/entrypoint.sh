#!/usr/bin/env bash
set -euo pipefail

MODELS_DIR="/models"
INVOKEAI_ROOT="${INVOKEAI_ROOT:-/invokeai}"

echo "=== InvokeAI Entrypoint ==="

# Ensure the InvokeAI root directory exists
mkdir -p "$INVOKEAI_ROOT"

# Register models from shared mount using InvokeAI's model installer
if [ -d "$MODELS_DIR" ]; then
    echo "Scanning $MODELS_DIR for models to register..."
    for model_dir in "$MODELS_DIR"/*/; do
        model_name=$(basename "$model_dir")
        if [ -f "$model_dir/model_index.json" ]; then
            echo "  Registering model: $model_name from $model_dir"
            # invokeai-model-install registers the model in InvokeAI's database
            # --yes skips confirmation; if already registered, this is a no-op
            invokeai-model-install --root "$INVOKEAI_ROOT" "$model_dir" --yes 2>/dev/null || \
                echo "  Note: Could not auto-register $model_name — import via Model Manager UI"
        fi
    done
else
    echo "WARNING: Shared models directory $MODELS_DIR not found"
fi

echo "Models can also be imported via the Model Manager UI at http://localhost:${INVOKEAI_PORT:-9090}"
echo "==========================="

# Execute the original InvokeAI entrypoint
# The official image uses invokeai-web as its command
exec invokeai-web --host 0.0.0.0 --port "${INVOKEAI_PORT:-9090}" --root "$INVOKEAI_ROOT"
