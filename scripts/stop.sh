#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# Source shared runtime detection
source "$SCRIPT_DIR/lib/runtime.sh"

COMPOSE_CMD=$(require_runtime)

echo "Stopping services..."

# Also stop any CDI-fallback containers
RUNTIME_CMD="${COMPOSE_CMD%% *}"
for name in localimggen-invokeai localimggen-cogvideo; do
    if $RUNTIME_CMD ps --format '{{.Names}}' 2>/dev/null | grep -q "$name"; then
        echo "Stopping CDI-fallback container: $name"
        $RUNTIME_CMD rm -f "$name" 2>/dev/null || true
    fi
done

if ! $COMPOSE_CMD down; then
    echo ""
    echo "ERROR: compose down failed."
    echo "Try manual cleanup:"
    echo "  podman ps --filter 'label=com.docker.compose.project'"
    echo "  podman rm -f <container_name>"
    exit 1
fi

# Report any remaining GPU processes
if command -v nvidia-smi &>/dev/null; then
    GPU_PROCS=$(nvidia-smi --query-compute-apps=pid,name --format=csv,noheader 2>/dev/null || true)
    if [ -n "$GPU_PROCS" ]; then
        echo ""
        echo "Note: GPU processes still running:"
        echo "$GPU_PROCS"
    fi
fi

echo "Services stopped."
