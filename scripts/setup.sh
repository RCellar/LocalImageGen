#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

echo "=== LocalImageGen Setup ==="
echo ""

ERRORS=0

# --- Check prerequisites ---
echo "Checking prerequisites..."

# Container runtime
if podman compose version &>/dev/null 2>&1 || command -v podman-compose &>/dev/null; then
    echo "  [OK] Podman compose found"
elif docker compose version &>/dev/null 2>&1 || command -v docker-compose &>/dev/null; then
    echo "  [OK] Docker compose found"
else
    echo "  [MISSING] No container compose runtime found"
    echo "    Install: dnf install podman-compose"
    ERRORS=$((ERRORS + 1))
fi

# NVIDIA driver
if command -v nvidia-smi &>/dev/null; then
    DRIVER_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    echo "  [OK] NVIDIA driver $DRIVER_VER"
else
    echo "  [MISSING] nvidia-smi not found"
    echo "    Install NVIDIA drivers: dnf install akmod-nvidia"
    ERRORS=$((ERRORS + 1))
fi

# NVIDIA Container Toolkit / CDI
if [ -d "/etc/cdi" ] || [ -f "/etc/nvidia-container-runtime/config.toml" ]; then
    echo "  [OK] NVIDIA Container Toolkit / CDI configured"
else
    echo "  [MISSING] NVIDIA Container Toolkit not found"
    echo "    Install: dnf install nvidia-container-toolkit"
    echo "    Then: nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml"
    ERRORS=$((ERRORS + 1))
fi

# Python 3
if command -v python3 &>/dev/null; then
    PY_VER=$(python3 --version)
    echo "  [OK] $PY_VER"
else
    echo "  [MISSING] Python 3 not found"
    echo "    Install: dnf install python3"
    ERRORS=$((ERRORS + 1))
fi

# PyYAML
if python3 -c "import yaml" 2>/dev/null; then
    echo "  [OK] PyYAML installed"
else
    echo "  [MISSING] PyYAML not found"
    echo "    Install: dnf install python3-pyyaml"
    echo "    # or: pip install pyyaml"
    ERRORS=$((ERRORS + 1))
fi

# huggingface-cli
if command -v huggingface-cli &>/dev/null; then
    echo "  [OK] huggingface-cli found"
else
    echo "  [MISSING] huggingface-cli not found"
    echo "    Install: pip install huggingface-hub[cli]"
    ERRORS=$((ERRORS + 1))
fi

if [ $ERRORS -gt 0 ]; then
    echo ""
    echo "ERROR: $ERRORS prerequisite(s) missing. Fix the issues above and re-run setup."
    exit 1
fi

echo ""
echo "All prerequisites satisfied."

# --- Config ---
if [ ! -f config.yaml ]; then
    echo ""
    echo "Creating config.yaml from template..."
    cp config.example.yaml config.yaml
    echo "Created config.yaml — edit it to customize paths and enabled models."
else
    echo ""
    echo "config.yaml already exists — using existing configuration."
fi

# --- Download models ---
echo ""
echo "=== Downloading models ==="
"$SCRIPT_DIR/download-models.sh"

# --- Build/pull container images ---
echo ""
echo "=== Preparing container images ==="

# Detect runtime for pulling/building
if podman compose version &>/dev/null 2>&1; then
    RUNTIME_CMD="podman"
elif command -v podman-compose &>/dev/null; then
    RUNTIME_CMD="podman"
elif docker compose version &>/dev/null 2>&1; then
    RUNTIME_CMD="docker"
else
    RUNTIME_CMD="docker"
fi

echo "Pulling InvokeAI official image..."
$RUNTIME_CMD pull ghcr.io/invoke-ai/invokeai

echo ""
echo "Building CogVideoX container image..."
$RUNTIME_CMD build -t cogvideo-local:latest containers/cogvideo/

# --- GPU smoke test ---
echo ""
echo "=== GPU smoke test ==="
if $RUNTIME_CMD run --rm --device nvidia.com/gpu=all docker.io/nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi &>/dev/null; then
    echo "  [OK] GPU accessible inside containers"
else
    echo "  [WARN] GPU smoke test failed."
    echo "  Trying with --gpus all flag..."
    if $RUNTIME_CMD run --rm --gpus all docker.io/nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi &>/dev/null; then
        echo "  [OK] GPU accessible with --gpus all (deploy block may work)"
    else
        echo "  [FAIL] Cannot access GPU inside containers."
        echo ""
        echo "  Troubleshooting:"
        echo "    1. Check NVIDIA Container Toolkit: nvidia-ctk --version"
        echo "    2. Generate CDI spec: sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml"
        echo "    3. Verify CDI: $RUNTIME_CMD run --rm --device nvidia.com/gpu=all docker.io/nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi"
        echo ""
        echo "  Driver version: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo 'unknown')"
        echo "  CUDA version: $(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null || echo 'unknown')"
        exit 1
    fi
fi

# --- Done ---
echo ""
echo "============================================"
echo "  Setup complete!"
echo ""
echo "  Start services: scripts/start.sh"
echo "  Stop services:  scripts/stop.sh"
echo "============================================"
