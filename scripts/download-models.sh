#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_FILE="$PROJECT_DIR/config.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: config.yaml not found. Run: cp config.example.yaml config.yaml"
    exit 1
fi

# Check for huggingface-cli
if ! command -v huggingface-cli &>/dev/null; then
    echo "ERROR: huggingface-cli not found. Install with:"
    echo "  pip install huggingface-hub[cli]"
    exit 1
fi

# Parse config with Python
MODELS_DIR=$(python3 -c "
import yaml, os
config = yaml.safe_load(open('$CONFIG_FILE'))
models_dir = config.get('paths', {}).get('models', './models')
if not os.path.isabs(models_dir):
    models_dir = os.path.join('$PROJECT_DIR', models_dir)
print(models_dir)
")

mkdir -p "$MODELS_DIR"

# Check if any gated models need a token
HAS_GATED=$(python3 -c "
import yaml
config = yaml.safe_load(open('$CONFIG_FILE'))
models = config.get('models', {})
gated = [k for k, v in models.items() if v.get('enabled') and v.get('gated')]
print('yes' if gated else 'no')
")

if [ "$HAS_GATED" = "yes" ]; then
    if ! huggingface-cli whoami &>/dev/null; then
        echo ""
        echo "Some enabled models require authentication (gated models)."
        echo "Please log in to HuggingFace:"
        echo ""
        huggingface-cli login
        echo ""
    fi
fi

# Download each enabled model
python3 -c "
import yaml
config = yaml.safe_load(open('$CONFIG_FILE'))
models = config.get('models', {})
for name, info in models.items():
    if info.get('enabled'):
        print(f\"{name}|{info['repo']}|{info.get('gated', False)}\")
" | while IFS='|' read -r name repo gated; do
    dest="$MODELS_DIR/$name"

    # Skip if directory exists and has model files (model_index.json or config.json + multiple files)
    if [ -d "$dest" ] && { [ -f "$dest/model_index.json" ] || [ -f "$dest/config.json" ]; }; then
        file_count=$(find "$dest" -type f | wc -l)
        if [ "$file_count" -ge 3 ]; then
            echo "Model '$name' already downloaded at $dest ($file_count files) — skipping"
            continue
        fi
    fi

    echo ""
    echo "=== Downloading $name from $repo ==="
    echo "Destination: $dest"
    echo ""

    if ! huggingface-cli download "$repo" --local-dir "$dest"; then
        echo ""
        echo "ERROR: Failed to download $name from $repo"
        if [ "$gated" = "True" ]; then
            echo "This is a gated model. Make sure you have:"
            echo "  1. Accepted the license at https://huggingface.co/$repo"
            echo "  2. Logged in with: huggingface-cli login"
        fi
        echo "You can re-run this script to retry."
        echo ""
        continue
    fi

    # Verify download — check for model_index.json or config.json and minimum file count
    if [ -f "$dest/model_index.json" ] || [ -f "$dest/config.json" ]; then
        file_count=$(find "$dest" -type f | wc -l)
        if [ "$file_count" -lt 3 ]; then
            echo "WARNING: Download may be incomplete ($file_count files). Re-run to retry."
        else
            echo "Successfully downloaded $name ($file_count files)"
        fi
    else
        echo "WARNING: Download may be incomplete — no model_index.json or config.json found"
        echo "Re-run this script to retry."
    fi
done

echo ""
echo "=== Download complete ==="
echo "Models directory: $MODELS_DIR"
ls -1 "$MODELS_DIR" 2>/dev/null
