# LocalImageGen

Local AI image and video generation running entirely offline in containers. No cloud APIs, no subscriptions — just your GPU.

**Services included:**

- **InvokeAI** — Stable Diffusion 3.5 Medium for image generation (txt2img, img2img, inpainting, canvas, node editor)
- **CogVideoX-5B** — Text-to-video and image-to-video generation with a Gradio UI

## Prerequisites

| Requirement | Notes |
|---|---|
| **NVIDIA GPU** | 8GB+ VRAM recommended (tested on RTX 4080 16GB) |
| **NVIDIA Driver** | `nvidia-smi` must work. Fedora: `dnf install akmod-nvidia` |
| **NVIDIA Container Toolkit** | Fedora: `dnf install nvidia-container-toolkit` then `sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml` |
| **Podman + Podman Compose** *or* **Docker + Docker Compose** | Podman preferred. Runtime is auto-detected. |
| **Python 3 + PyYAML** | Fedora: `dnf install python3-pyyaml` |
| **huggingface-cli** | `pip install huggingface-hub[cli]` — required for model downloads |

## Quick Start

```bash
# 1. Run the setup wizard (checks prerequisites, downloads models, builds containers)
scripts/setup.sh

# 2. Start services
scripts/start.sh

# 3. Open in your browser
#    InvokeAI:  http://localhost:9090
#    CogVideoX: http://localhost:7860
```

## Setup

The setup script handles everything interactively:

```bash
scripts/setup.sh
```

This will:

1. Verify all prerequisites are installed
2. Create `config.yaml` from the template (if not present)
3. Prompt for HuggingFace login if gated models are enabled (SD 3.5 Medium requires acceptance of the [Stability AI license](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium))
4. Download models (~67GB total: ~46GB for SD 3.5 Medium, ~21GB for CogVideoX-5B)
5. Build/pull container images
6. Run a GPU smoke test

### Configuration

Edit `config.yaml` to customize your setup:

```yaml
paths:
  models: ./models          # Model storage location
  outputs: ./outputs        # Generated content output

models:
  sd3.5-medium:
    enabled: true           # Toggle Stable Diffusion
  cogvideox-5b:
    enabled: true           # Toggle CogVideoX

services:
  invokeai:
    port: 9090
    enabled: true
  cogvideo:
    port: 7860
    enabled: true
    quantization: none      # none (BF16, ~5GB VRAM) or int8 (~4.4GB VRAM)

gpu:
  device: 0                 # GPU index
  runtime: auto             # auto, podman, or docker
```

## Usage

### Starting Services

```bash
scripts/start.sh
```

The start script parses your configuration, checks GPU memory, launches the enabled services, validates GPU access inside containers, and waits for health checks before printing the service URLs.

### Stopping Services

```bash
scripts/stop.sh
```

Stops and removes all containers. Also cleans up any CDI-fallback containers.

### Updating

```bash
scripts/update.sh
```

Pulls the latest container images and re-downloads models if needed.

### Accessing Outputs

Generated content is saved to bind-mounted directories on the host:

- **Images:** `outputs/images/`
- **Videos:** `outputs/videos/`

## VRAM Considerations

| Service | VRAM Usage |
|---|---|
| SD 3.5 Medium (FP16) | ~6 GB |
| CogVideoX-5B (BF16) | ~5 GB |
| CogVideoX-5B (INT8) | ~4.4 GB |

Running both services simultaneously requires ~11GB+ VRAM. If you have limited VRAM, consider running one service at a time or enabling INT8 quantization for CogVideoX (`quantization: int8` in config).

## Project Structure

```
LocalImageGen/
├── compose.yaml                  # Container orchestration
├── config.example.yaml           # Configuration template
├── config.yaml                   # Your configuration (gitignored)
├── containers/
│   ├── invokeai/
│   │   └── entrypoint.sh        # Model registration wrapper
│   └── cogvideo/
│       ├── Containerfile         # CogVideoX container build
│       ├── requirements.txt      # Pinned Python dependencies
│       └── app.py               # Gradio video generation UI
├── models/                       # Downloaded model weights (gitignored)
├── outputs/                      # Generated content (gitignored)
└── scripts/
    ├── setup.sh                  # First-run setup wizard
    ├── start.sh                  # Launch services
    ├── stop.sh                   # Stop services
    ├── update.sh                 # Update services and models
    ├── download-models.sh        # Model downloader
    ├── parse-config.py           # YAML to .env converter
    └── lib/runtime.sh            # Container runtime detection
```

## GPU Passthrough

The project uses a two-tier strategy for GPU access in containers:

1. **Primary:** Standard Compose `deploy.resources.reservations.devices` syntax
2. **Fallback:** If GPU isn't accessible, containers are recreated with CDI device flags (`--device nvidia.com/gpu=all`)

GPU access is validated at startup by running `torch.cuda.is_available()` inside a container.

## Teardown

To stop services:

```bash
scripts/stop.sh
```

To remove generated content:

```bash
rm -rf outputs/*
```

To fully reset (models ~67GB will need to be re-downloaded):

```bash
rm -rf models/ outputs/ .env
scripts/setup.sh
```

To remove container images:

```bash
# Podman
podman rmi localimagegen-cogvideo
podman rmi ghcr.io/invoke-ai/invokeai

# Docker
docker rmi localimagegen-cogvideo
docker rmi ghcr.io/invoke-ai/invokeai
```

To remove the named data volume:

```bash
podman volume rm localimagegen_invokeai-data
# or: docker volume rm localimagegen_invokeai-data
```

## License

This project is licensed under the [MIT License](LICENSE).

## Model Licenses

The models downloaded by this project have their own licenses with specific terms and restrictions. By using this project, you are responsible for reviewing and complying with each model's license:

- **Stable Diffusion 3.5 Medium** — [Stability AI Community License](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium). Free for research and non-commercial use; commercial use free under $1M revenue/year.
- **CogVideoX-5B** — [CogVideoX License](https://huggingface.co/THUDM/CogVideoX-5b). Free for academic research; commercial use requires registration.

This project does not bundle or redistribute any models or container images — they are downloaded directly from their respective sources at setup time.
