#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# Source shared runtime detection
source "$SCRIPT_DIR/lib/runtime.sh"

CONFIG_FILE="$PROJECT_DIR/config.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: config.yaml not found. Run: cp config.example.yaml config.yaml"
    exit 1
fi

# --- Parse config and generate .env ---
echo "Parsing config.yaml..."
PROFILES=$(python3 "$SCRIPT_DIR/parse-config.py")
if [ -z "$PROFILES" ]; then
    echo "ERROR: No services enabled in config.yaml"
    exit 1
fi

# Source generated .env to get port variables
set -a
source .env
set +a

# --- Detect container runtime ---
RUNTIME_SETTING=$(python3 -c "
import yaml
config = yaml.safe_load(open('$CONFIG_FILE'))
print(config.get('gpu', {}).get('runtime', 'auto'))
")

case "$RUNTIME_SETTING" in
    auto)
        COMPOSE_CMD=$(require_runtime)
        ;;
    podman)
        if podman compose version &>/dev/null 2>&1; then
            COMPOSE_CMD="podman compose"
        elif command -v podman-compose &>/dev/null; then
            COMPOSE_CMD="podman-compose"
        else
            echo "ERROR: podman compose runtime not found. Install with:"
            echo "  dnf install podman-compose"
            exit 1
        fi
        ;;
    docker)
        if docker compose version &>/dev/null 2>&1; then
            COMPOSE_CMD="docker compose"
        elif command -v docker-compose &>/dev/null; then
            COMPOSE_CMD="docker-compose"
        else
            echo "ERROR: docker compose runtime not found."
            exit 1
        fi
        ;;
    *)
        echo "ERROR: Unknown runtime '$RUNTIME_SETTING' in config.yaml (expected: auto, podman, docker)"
        exit 1
        ;;
esac

echo "Using runtime: $COMPOSE_CMD"

# --- GPU memory guard ---
if command -v nvidia-smi &>/dev/null; then
    FREE_VRAM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ "$FREE_VRAM" -lt 4000 ] 2>/dev/null; then
        echo "WARNING: Only ${FREE_VRAM}MB VRAM free. Services may fail to load models."
        echo "Consider closing GPU-intensive applications first."
    fi

    # Warn about co-residency if both services enabled
    if echo "$PROFILES" | grep -q "image" && echo "$PROFILES" | grep -q "video"; then
        echo "WARNING: Both image and video services enabled."
        echo "Combined VRAM usage (~11GB+) may be unstable. Consider running one at a time."
    fi
fi

# --- Check for downloaded models ---
for profile in $PROFILES; do
    case "$profile" in
        image)
            if [ ! -d "$MODELS_DIR/sd3.5-medium" ]; then
                echo "WARNING: No image models found. InvokeAI will start but have no models."
                echo "Run: scripts/download-models.sh"
            fi
            ;;
        video)
            if [ ! -d "$MODELS_DIR/cogvideox-5b" ]; then
                echo "WARNING: No video models found. CogVideoX will fail to start."
                echo "Run: scripts/download-models.sh"
            fi
            ;;
    esac
done

# --- Build profile flags ---
PROFILE_FLAGS=""
for profile in $PROFILES; do
    PROFILE_FLAGS="$PROFILE_FLAGS --profile $profile"
done

# --- Launch ---
echo ""
echo "Starting services with profiles: $PROFILES"
$COMPOSE_CMD $PROFILE_FLAGS up -d

# --- GPU fallback: detect if containers started without GPU access ---
# Check if a running container can see the GPU
RUNTIME_CMD="${COMPOSE_CMD%% *}"  # extract "podman" or "docker"
GPU_CHECK_FAILED=false
for profile in $PROFILES; do
    case "$profile" in
        video)
            CONTAINER_NAME=$($COMPOSE_CMD ps --format '{{.Name}}' 2>/dev/null | grep cogvideo | head -1)
            if [ -n "$CONTAINER_NAME" ]; then
                if ! $RUNTIME_CMD exec "$CONTAINER_NAME" nvidia-smi &>/dev/null 2>&1; then
                    GPU_CHECK_FAILED=true
                fi
            fi
            ;;
    esac
done

if $GPU_CHECK_FAILED; then
    echo ""
    echo "WARNING: Containers started but GPU not accessible inside container."
    echo "The compose 'deploy.resources' block may not be supported by this runtime."
    echo "Attempting CDI fallback (--device nvidia.com/gpu=all)..."
    echo ""

    # Tear down and recreate with CDI device flags
    $COMPOSE_CMD $PROFILE_FLAGS down

    # Re-launch with explicit device passthrough
    # podman-compose doesn't support --device via compose, so use environment hint
    export NVIDIA_VISIBLE_DEVICES=all
    CDI_DEVICE="--device nvidia.com/gpu=all"

    for profile in $PROFILES; do
        case "$profile" in
            image)
                $RUNTIME_CMD run -d --name localimggen-invokeai \
                    $CDI_DEVICE \
                    -p "${INVOKEAI_PORT:-9090}:${INVOKEAI_PORT:-9090}" \
                    -v "${MODELS_DIR:-./models}:/models:ro" \
                    -v "${OUTPUTS_DIR:-./outputs}/images:/invokeai/outputs" \
                    -v "./containers/invokeai/entrypoint.sh:/entrypoint-wrapper.sh:ro" \
                    -e "INVOKEAI_PORT=${INVOKEAI_PORT:-9090}" \
                    -e "INVOKEAI_ROOT=/invokeai" \
                    --entrypoint /bin/bash \
                    ghcr.io/invoke-ai/invokeai \
                    /entrypoint-wrapper.sh
                ;;
            video)
                $RUNTIME_CMD run -d --name localimggen-cogvideo \
                    $CDI_DEVICE \
                    -p "${COGVIDEO_PORT:-7860}:${COGVIDEO_PORT:-7860}" \
                    -v "${MODELS_DIR:-./models}:/models:ro" \
                    -v "${OUTPUTS_DIR:-./outputs}/videos:/outputs/videos" \
                    -e "MODEL_PATH=/models/cogvideox-5b" \
                    -e "OUTPUT_DIR=/outputs/videos" \
                    -e "COGVIDEO_PORT=${COGVIDEO_PORT:-7860}" \
                    -e "COGVIDEO_QUANTIZATION=${COGVIDEO_QUANTIZATION:-none}" \
                    cogvideo-local:latest
                ;;
        esac
    done
    echo "Containers restarted with CDI device passthrough."
fi

# --- Wait for health ---
echo ""
echo "Waiting for services to become healthy (up to 120s)..."
TIMEOUT=120
ELAPSED=0
while [ $ELAPSED -lt $TIMEOUT ]; do
    ALL_HEALTHY=true
    for profile in $PROFILES; do
        case "$profile" in
            image)
                if ! curl -sf "http://localhost:${INVOKEAI_PORT}/api/v1/app/version" &>/dev/null; then
                    ALL_HEALTHY=false
                fi
                ;;
            video)
                if ! curl -sf "http://localhost:${COGVIDEO_PORT}/" &>/dev/null; then
                    ALL_HEALTHY=false
                fi
                ;;
        esac
    done

    if $ALL_HEALTHY; then
        break
    fi

    sleep 5
    ELAPSED=$((ELAPSED + 5))
    echo "  ...waiting ($ELAPSED/${TIMEOUT}s)"
done

if ! $ALL_HEALTHY; then
    echo ""
    echo "ERROR: Services did not become healthy within ${TIMEOUT}s"
    echo ""
    echo "=== Container logs ==="
    $COMPOSE_CMD $PROFILE_FLAGS logs --tail 50 2>/dev/null || \
        $RUNTIME_CMD logs localimggen-cogvideo 2>/dev/null || true
    echo ""
    echo "=== GPU status ==="
    nvidia-smi 2>/dev/null || echo "nvidia-smi not available"
    exit 1
fi

# --- Print URLs ---
echo ""
echo "=== Services ready ==="
for profile in $PROFILES; do
    case "$profile" in
        image)
            echo "  InvokeAI:  http://localhost:${INVOKEAI_PORT}"
            ;;
        video)
            echo "  CogVideoX: http://localhost:${COGVIDEO_PORT}"
            ;;
    esac
done
echo ""
