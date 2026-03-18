#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# Source shared runtime detection
source "$SCRIPT_DIR/lib/runtime.sh"

COMPOSE_CMD=$(require_runtime)

# Check which services are currently running
RUNNING_PROFILES=""
if $COMPOSE_CMD ps --format '{{.Name}}' 2>/dev/null | grep -q invokeai; then
    RUNNING_PROFILES="$RUNNING_PROFILES --profile image"
fi
if $COMPOSE_CMD ps --format '{{.Name}}' 2>/dev/null | grep -q cogvideo; then
    RUNNING_PROFILES="$RUNNING_PROFILES --profile video"
fi

# Pull/rebuild images
echo "=== Updating InvokeAI image ==="
RUNTIME_CMD="${COMPOSE_CMD%% *}"  # podman or docker
$RUNTIME_CMD pull ghcr.io/invoke-ai/invokeai

echo ""
echo "=== Rebuilding CogVideoX image ==="
$COMPOSE_CMD build cogvideo

# Optionally re-download models
if [ "${1:-}" = "--models" ]; then
    echo ""
    echo "=== Updating models ==="
    "$SCRIPT_DIR/download-models.sh"
fi

# Restart if services were running
if [ -n "$RUNNING_PROFILES" ]; then
    echo ""
    echo "=== Restarting services ==="
    $COMPOSE_CMD $RUNNING_PROFILES down
    $COMPOSE_CMD $RUNNING_PROFILES up -d
    echo "Services restarted. Run 'scripts/start.sh' for health check polling."
else
    echo ""
    echo "Images updated. No services were running — skipping restart."
fi

echo ""
echo "Update complete."
