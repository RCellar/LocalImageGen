#!/usr/bin/env python3
"""Parse config.yaml and generate .env for compose."""

import sys
import os

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML is required. Install with:", file=sys.stderr)
    print("  dnf install python3-pyyaml", file=sys.stderr)
    print("  # or: pip install pyyaml", file=sys.stderr)
    sys.exit(1)


def main():
    script_dir = os.path.dirname(os.path.realpath(__file__))
    config_path = os.path.join(script_dir, "..", "config.yaml")
    env_path = os.path.join(script_dir, "..", ".env")

    if not os.path.exists(config_path):
        print(f"ERROR: {config_path} not found.", file=sys.stderr)
        print("Run: cp config.example.yaml config.yaml", file=sys.stderr)
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Validate required sections
    for section in ("paths", "services", "gpu"):
        if section not in config:
            print(f"ERROR: Missing required section '{section}' in config.yaml", file=sys.stderr)
            sys.exit(1)

    paths = config["paths"]
    services = config["services"]
    gpu = config["gpu"]

    cogvideo_svc = services.get("cogvideo", {})
    env_lines = [
        "# Auto-generated from config.yaml — do not edit",
        f"MODELS_DIR={paths.get('models', './models')}",
        f"OUTPUTS_DIR={paths.get('outputs', './outputs')}",
        f"INVOKEAI_PORT={services.get('invokeai', {}).get('port', 9090)}",
        f"COGVIDEO_PORT={cogvideo_svc.get('port', 7860)}",
        f"COGVIDEO_QUANTIZATION={cogvideo_svc.get('quantization', 'none')}",
        f"I2V_MODEL_PATH={paths.get('models', './models')}/cogvideox-5b-i2v",
        f"GPU_DEVICE={gpu.get('device', 0)}",
        f"CONTAINER_UID={os.getuid()}",
        f"CONTAINER_GID={os.getgid()}",
    ]

    with open(env_path, "w") as f:
        f.write("\n".join(env_lines) + "\n")

    # Print profile flags for start.sh to consume
    profiles = []
    if services.get("invokeai", {}).get("enabled", False):
        profiles.append("image")
    if services.get("cogvideo", {}).get("enabled", False):
        profiles.append("video")

    if not profiles:
        print("WARNING: No services enabled in config.yaml", file=sys.stderr)
    else:
        # Output profiles as space-separated list on stdout
        print(" ".join(profiles))


if __name__ == "__main__":
    main()
