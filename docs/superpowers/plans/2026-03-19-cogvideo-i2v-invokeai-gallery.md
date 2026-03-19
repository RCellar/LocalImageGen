# CogVideoX I2V + InvokeAI Gallery Fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix CogVideoX Image-to-Video generation, fix InvokeAI gallery persistence, and add interactive service selection to prevent GPU contention.

**Architecture:** Three independent changes sharing some files. Config and compose changes add the I2V model path. The CogVideo app.py gets a second model path, image resizing, error handling, and graceful degradation. InvokeAI's volume mount strategy is corrected to stop clobbering its internal directory structure. start.sh gets an interactive service selector and CDI fallback networking fixes.

**Tech Stack:** Bash, Python 3 (Gradio, diffusers, PIL), YAML config, podman/docker-compose

**Spec:** `docs/superpowers/specs/2026-03-19-cogvideo-i2v-invokeai-gallery-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `config.yaml` | Modify | Add `cogvideox-5b-i2v` model entry, re-enable InvokeAI |
| `config.example.yaml` | Modify | Add `cogvideox-5b-i2v` model entry |
| `compose.yaml` | Modify | Add `I2V_MODEL_PATH` env var to cogvideo; remove InvokeAI output bind mount overlay |
| `containers/cogvideo/app.py` | Modify | Route i2v to correct model, add image resizing, error handling, graceful degradation |
| `containers/invokeai/entrypoint.sh` | Modify | Pre-create output subdirectories |
| `scripts/start.sh` | Modify | Interactive service prompt, CDI fallback networking + i2v env var, model checks |
| `scripts/parse-config.py` | Modify | Add `I2V_MODEL_PATH` to `.env` output |

---

### Task 1: Add I2V model to config files

**Files:**
- Modify: `config.yaml`
- Modify: `config.example.yaml`

- [ ] **Step 1: Add cogvideox-5b-i2v to config.yaml**

Add after the existing `cogvideox-5b` entry in `config.yaml`:

```yaml
  cogvideox-5b-i2v:
    enabled: true
    gated: false
    repo: THUDM/CogVideoX-5b-I2V
    type: video
```

Also set `invokeai.enabled: true` (change from `false`) to re-enable InvokeAI for gallery testing.

- [ ] **Step 2: Add cogvideox-5b-i2v to config.example.yaml**

Add after the existing `cogvideox-5b` entry in `config.example.yaml`:

```yaml
  cogvideox-5b-i2v:
    repo: THUDM/CogVideoX-5b-I2V
    type: video
    gated: false
    enabled: true
```

- [ ] **Step 3: Verify config parses correctly**

Run: `python3 scripts/parse-config.py`
Expected: Outputs `image video` (both profiles enabled) and generates `.env` without errors.

- [ ] **Step 4: Commit**

```bash
git add config.yaml config.example.yaml
git commit -m "config: add CogVideoX-5B I2V model entry and re-enable InvokeAI"
```

---

### Task 2: Add I2V_MODEL_PATH to env generation and compose

**Files:**
- Modify: `scripts/parse-config.py:40-50`
- Modify: `compose.yaml:47-52`

- [ ] **Step 1: Add I2V_MODEL_PATH to parse-config.py**

In `scripts/parse-config.py`, add a new line to the `env_lines` list after the `COGVIDEO_QUANTIZATION` line (line 46):

```python
        f"I2V_MODEL_PATH={paths.get('models', './models')}/cogvideox-5b-i2v",
```

- [ ] **Step 2: Add I2V_MODEL_PATH to compose.yaml cogvideo environment**

In `compose.yaml`, add to the cogvideo service's `environment` block after `MODEL_PATH` (after line 48):

```yaml
      - I2V_MODEL_PATH=/models/cogvideox-5b-i2v
```

- [ ] **Step 3: Verify .env generation includes I2V_MODEL_PATH**

Run: `python3 scripts/parse-config.py && grep I2V_MODEL_PATH .env`
Expected: `I2V_MODEL_PATH=./models/cogvideox-5b-i2v`

- [ ] **Step 4: Commit**

```bash
git add scripts/parse-config.py compose.yaml
git commit -m "config: add I2V_MODEL_PATH to env generation and compose"
```

---

### Task 3: Fix CogVideoX app.py — route i2v to correct model

**Files:**
- Modify: `containers/cogvideo/app.py:13-17` (add I2V_MODEL_PATH)
- Modify: `containers/cogvideo/app.py:44-81` (_get_pipeline)
- Modify: `containers/cogvideo/app.py:91-144` (generation functions)
- Modify: `containers/cogvideo/app.py:147-213` (Gradio UI)

- [ ] **Step 1: Add I2V_MODEL_PATH and i2v model check**

After line 16 (`QUANTIZATION = ...`), add:

```python
I2V_MODEL_PATH = os.environ.get("I2V_MODEL_PATH", "/models/cogvideox-5b-i2v")

# Check if I2V model is available (has model_index.json or config.json)
I2V_AVAILABLE = os.path.isfile(os.path.join(I2V_MODEL_PATH, "model_index.json")) or \
                os.path.isfile(os.path.join(I2V_MODEL_PATH, "config.json"))
```

- [ ] **Step 2: Update _get_pipeline to use correct model path per mode**

Replace the print and pipeline loading section of `_get_pipeline()` (lines 59-67) with:

```python
    model_path = I2V_MODEL_PATH if mode == "img2vid" else MODEL_PATH
    print(f"Loading {mode} pipeline from {model_path}...")
    if mode == "txt2vid":
        pipe = CogVideoXPipeline.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
        )
    else:
        pipe = CogVideoXImageToVideoPipeline.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
        )
```

- [ ] **Step 3: Add error handling to generate_txt2vid**

Replace the body of `generate_txt2vid()` (lines 98-114) with:

```python
    if not prompt.strip():
        raise gr.Error("Please enter a prompt.")

    try:
        pipe = _get_pipeline("txt2vid")
        video_frames = pipe(
            prompt=prompt,
            num_videos_per_prompt=1,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
        ).frames[0]
    except torch.cuda.OutOfMemoryError:
        raise gr.Error("GPU out of memory. Try reducing inference steps or frame count, or close other GPU applications.")
    except Exception as e:
        if "out of memory" in str(e).lower():
            raise gr.Error("GPU out of memory. Try reducing inference steps or frame count, or close other GPU applications.")
        raise gr.Error(f"Generation failed: {e}")

    timestamp = int(time.time())
    output_path = os.path.join(OUTPUT_DIR, f"txt2vid_{timestamp}.mp4")
    save_video(video_frames, output_path, fps=8)

    return output_path
```

- [ ] **Step 4: Add error handling and image resizing to generate_img2vid**

Add `from PIL import Image` to the imports at the top of the file (after `import numpy as np`, line 11).

Replace the body of `generate_img2vid()` (lines 124-144) with:

```python
    if image is None:
        raise gr.Error("Please upload a starting image.")
    if not prompt.strip():
        raise gr.Error("Please enter a prompt.")

    # Resize to CogVideoX-5B I2V expected dimensions (480x720)
    target_size = (720, 480)  # PIL uses (width, height)
    if image.size != target_size:
        print(f"Resizing input image from {image.size} to {target_size}")
        image = image.resize(target_size, Image.LANCZOS)

    try:
        pipe = _get_pipeline("img2vid")
        video_frames = pipe(
            image=image,
            prompt=prompt,
            num_videos_per_prompt=1,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
        ).frames[0]
    except torch.cuda.OutOfMemoryError:
        raise gr.Error("GPU out of memory. Try reducing inference steps or frame count, or close other GPU applications.")
    except Exception as e:
        if "out of memory" in str(e).lower():
            raise gr.Error("GPU out of memory. Try reducing inference steps or frame count, or close other GPU applications.")
        raise gr.Error(f"Generation failed: {e}")

    timestamp = int(time.time())
    output_path = os.path.join(OUTPUT_DIR, f"img2vid_{timestamp}.mp4")
    save_video(video_frames, output_path, fps=8)

    return output_path
```

- [ ] **Step 5: Add graceful degradation to Image to Video tab**

Replace the Image to Video `TabItem` block (lines 183-213) with:

```python
        with gr.TabItem("Image to Video"):
            if I2V_AVAILABLE:
                with gr.Row():
                    with gr.Column(scale=2):
                        i2v_image = gr.Image(label="Starting Image", type="pil")
                        i2v_prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="The scene slowly comes to life...",
                            lines=3,
                        )
                        with gr.Row():
                            i2v_steps = gr.Slider(
                                minimum=10, maximum=100, value=50, step=1,
                                label="Inference Steps",
                            )
                            i2v_guidance = gr.Slider(
                                minimum=1.0, maximum=15.0, value=6.0, step=0.5,
                                label="Guidance Scale",
                            )
                            i2v_frames = gr.Slider(
                                minimum=9, maximum=49, value=49, step=8,
                                label="Number of Frames",
                            )
                        i2v_btn = gr.Button("Generate Video", variant="primary")
                    with gr.Column(scale=2):
                        i2v_output = gr.Video(label="Generated Video")

                i2v_btn.click(
                    fn=generate_img2vid,
                    inputs=[i2v_image, i2v_prompt, i2v_steps, i2v_guidance, i2v_frames],
                    outputs=[i2v_output],
                )
            else:
                gr.Markdown(
                    "**Image-to-Video model not downloaded.** "
                    "Run `scripts/setup.sh` to download the CogVideoX-5B I2V model."
                )
```

- [ ] **Step 6: Verify the file is syntactically valid**

Run: `python3 -c "import ast; ast.parse(open('containers/cogvideo/app.py').read()); print('OK')" `
Expected: `OK`

- [ ] **Step 7: Commit**

```bash
git add containers/cogvideo/app.py
git commit -m "fix: route CogVideoX i2v to correct model, add error handling and graceful degradation"
```

---

### Task 4: Fix InvokeAI gallery — remove output bind mount overlay

**Files:**
- Modify: `compose.yaml:10-14` (InvokeAI volumes)
- Modify: `containers/invokeai/entrypoint.sh`

- [ ] **Step 1: Investigate InvokeAI's output directory structure**

Start a temporary InvokeAI container and inspect the output directory structure:

```bash
RUNTIME_CMD="podman"  # or docker
$RUNTIME_CMD run --rm -it --entrypoint /bin/bash ghcr.io/invoke-ai/invokeai -c \
    "invokeai-web --root /tmp/invokeai &
     sleep 15 &&
     find /tmp/invokeai/outputs -type d 2>/dev/null &&
     find /tmp/invokeai/outputs -type f 2>/dev/null | head -20 &&
     cat /tmp/invokeai/invokeai.yaml 2>/dev/null || echo 'no invokeai.yaml'"
```

Record the directory structure. Based on the spec analysis, **Option B (remove the bind mount) is the expected outcome** — InvokeAI's output structure is likely too intertwined with its database to safely overlay. Use this to confirm:
- **Option A — Bind-mount a leaf subdirectory** (only if images go to a single isolated subdirectory like `/invokeai/outputs/images/` with no sibling metadata)
- **Option B — Remove the bind mount entirely** (expected — if the structure is complex or has metadata/thumbnails alongside images)
- **Fallback — Use invokeai.yaml to redirect** (if neither A nor B works)

- [ ] **Step 2: Apply the chosen mount strategy to compose.yaml**

**If removing the bind mount (Option B — simplest, most likely):**

Remove this line from the InvokeAI volumes in `compose.yaml` (line 12):
```yaml
      - ${OUTPUTS_DIR:-./outputs}/images:/invokeai/outputs
```

The InvokeAI volumes section becomes:
```yaml
    volumes:
      - ${MODELS_DIR:-./models}:/models
      - invokeai-data:/invokeai
      - ./containers/invokeai/entrypoint.sh:/entrypoint-wrapper.sh:ro
```

**If bind-mounting a leaf subdirectory (Option A):**

Replace line 12 with the correct deeper path discovered in Step 1, e.g.:
```yaml
      - ${OUTPUTS_DIR:-./outputs}/images:/invokeai/outputs/images
```

- [ ] **Step 3: Update entrypoint.sh to pre-create output directories**

In `containers/invokeai/entrypoint.sh`, add after `mkdir -p "$INVOKEAI_ROOT"` (line 11):

```bash
# Pre-create output directories InvokeAI expects
mkdir -p "$INVOKEAI_ROOT/outputs"
```

- [ ] **Step 4: Commit**

```bash
git add compose.yaml containers/invokeai/entrypoint.sh
git commit -m "fix: remove InvokeAI output bind mount overlay that broke gallery tracking"
```

---

### Task 5: Fix start.sh — interactive prompt, CDI fallback, model checks

**Files:**
- Modify: `scripts/start.sh:6` (add --all flag parsing after this line)
- Modify: `scripts/start.sh:79-108` (replace contiguous block: co-residency warning + nvidia-smi fi + model checks + profile flags)
- Modify: `scripts/start.sh:144-173` (CDI fallback containers)

- [ ] **Step 1: Add --all flag parsing**

After `cd "$PROJECT_DIR"` (line 6), add:

```bash
# Parse command-line flags
START_ALL=false
for arg in "$@"; do
    case "$arg" in
        --all) START_ALL=true ;;
    esac
done
```

- [ ] **Step 2: Replace lines 79–108 with interactive prompt, model checks, and profile flags**

Delete lines 79–108 as one contiguous block (the co-residency warning, the `fi` closing the nvidia-smi block, the model checks loop, and the profile flags builder). Replace with:

```bash
    # Interactive service selection when both are enabled
    if echo "$PROFILES" | grep -q "image" && echo "$PROFILES" | grep -q "video"; then
        if ! $START_ALL; then
            echo ""
            echo "Both InvokeAI and CogVideoX are enabled in config.yaml."
            echo "Running both simultaneously requires ~11GB+ VRAM."
            echo ""
            echo "Which services would you like to start?"
            echo "  1) InvokeAI only (image generation)"
            echo "  2) CogVideoX only (video generation)"
            echo "  3) Both (requires sufficient VRAM)"
            echo ""
            read -rp "Choice [1/2/3]: " choice
            case "$choice" in
                1) PROFILES="image" ;;
                2) PROFILES="video" ;;
                3) ;; # keep both
                *) echo "Invalid choice. Starting both." ;;
            esac
        fi  # end if ! $START_ALL
    fi  # end if both profiles enabled
fi  # end if command -v nvidia-smi

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
            if [ ! -d "$MODELS_DIR/cogvideox-5b-i2v" ]; then
                echo "WARNING: I2V model not found. Image-to-Video will be disabled."
            fi
            ;;
    esac
done

# --- Build profile flags ---
PROFILE_FLAGS=""
for profile in $PROFILES; do
    PROFILE_FLAGS="$PROFILE_FLAGS --profile $profile"
done
```

**Replacement boundaries:** Lines 79–108 inclusive. This is one contiguous block. Lines 72–78 (the VRAM free check) are kept as-is. The three `fi` statements close: (1) `if ! $START_ALL`, (2) `if echo "$PROFILES"...`, and (3) `if command -v nvidia-smi` (which opened at line 72). Each `fi` has an inline comment identifying which block it closes.

- [ ] **Step 3: Update CDI fallback — InvokeAI container**

Replace the InvokeAI CDI fallback block (lines 147-158) with:

```bash
            image)
                $RUNTIME_CMD run -d --name localimggen-invokeai \
                    $CDI_DEVICE $SELINUX_OPT \
                    --network=host \
                    -v "${MODELS_DIR:-./models}:/models" \
                    -v "localimagegen_invokeai-data:/invokeai" \
                    -v "./containers/invokeai/entrypoint.sh:/entrypoint-wrapper.sh:ro" \
                    -e "INVOKEAI_PORT=${INVOKEAI_PORT:-9090}" \
                    -e "INVOKEAI_ROOT=/invokeai" \
                    --entrypoint /bin/bash \
                    ghcr.io/invoke-ai/invokeai \
                    /entrypoint-wrapper.sh
                ;;
```

Changes from original:
- Removed `-p` port mapping (conflicts with `--network=host`)
- Added `--network=host`
- Removed the output bind mount line (`-v "${OUTPUTS_DIR:-./outputs}/images:/invokeai/outputs"`) — matches the compose.yaml fix from Task 4

- [ ] **Step 4: Update CDI fallback — CogVideoX container**

Replace the CogVideoX CDI fallback block (lines 160-171) with:

```bash
            video)
                $RUNTIME_CMD run -d --name localimggen-cogvideo \
                    $CDI_DEVICE $SELINUX_OPT \
                    --network=host \
                    -v "${MODELS_DIR:-./models}:/models:ro" \
                    -v "${OUTPUTS_DIR:-./outputs}/videos:/outputs/videos" \
                    -e "MODEL_PATH=/models/cogvideox-5b" \
                    -e "I2V_MODEL_PATH=/models/cogvideox-5b-i2v" \
                    -e "OUTPUT_DIR=/outputs/videos" \
                    -e "COGVIDEO_PORT=${COGVIDEO_PORT:-7860}" \
                    -e "COGVIDEO_QUANTIZATION=${COGVIDEO_QUANTIZATION:-none}" \
                    cogvideo-local:latest
                ;;
```

Changes from original:
- Removed `-p` port mapping (conflicts with `--network=host`)
- Added `--network=host`
- Added `-e "I2V_MODEL_PATH=/models/cogvideox-5b-i2v"`

- [ ] **Step 5: Verify start.sh syntax**

Run: `bash -n scripts/start.sh`
Expected: No output (no syntax errors).

- [ ] **Step 6: Commit**

```bash
git add scripts/start.sh
git commit -m "feat: add interactive service selection, fix CDI fallback networking and i2v model path"
```

---

### Task 6: Rebuild container and validate

This task requires GPU access and the downloaded models.

- [ ] **Step 1: Download the I2V model**

Run: `scripts/download-models.sh`
Expected: Downloads `cogvideox-5b-i2v` from `THUDM/CogVideoX-5b-I2V` to `models/cogvideox-5b-i2v/`. Skips already-downloaded models.

- [ ] **Step 2: Rebuild the CogVideoX container**

Run: `podman build -t cogvideo-local:latest containers/cogvideo/`
Expected: Build succeeds. The updated `app.py` is copied into the image.

- [ ] **Step 3: Validate interactive prompt**

Run: `scripts/start.sh`
Expected: Since both services are enabled in config.yaml, the interactive prompt appears. Select option 2 (CogVideoX only) to test video first.

- [ ] **Step 4: Validate CogVideoX Text-to-Video**

Open `http://localhost:7860`. Navigate to "Text to Video" tab. Enter a prompt and click "Generate Video".
Expected: Video generates and appears in the preview pane. File appears in `outputs/videos/` as `txt2vid_*.mp4`.

- [ ] **Step 5: Validate CogVideoX Image-to-Video**

Navigate to "Image to Video" tab. Upload an image, enter a prompt, click "Generate Video".
Expected: Image is resized automatically to 480x720. Video generates without error. File appears in `outputs/videos/` as `img2vid_*.mp4`.

- [ ] **Step 6: Validate graceful degradation**

Stop the service. Rename the I2V model directory:
```bash
mv models/cogvideox-5b-i2v models/cogvideox-5b-i2v.bak
```
Restart CogVideoX. Navigate to "Image to Video" tab.
Expected: Tab shows "Image-to-Video model not downloaded" message instead of inputs.

Restore the model:
```bash
mv models/cogvideox-5b-i2v.bak models/cogvideox-5b-i2v
```

- [ ] **Step 7: Stop CogVideoX and validate InvokeAI**

Stop CogVideoX:
```bash
scripts/stop.sh
```

Start InvokeAI only:
```bash
scripts/start.sh  # Select option 1
```

Open `http://localhost:9090`. Import SD 3.5 Medium via Model Manager (point to `/models/sd3.5-medium`, check "In-place install"). Generate an image.

Expected: Image appears in the gallery without manual page refresh. Thumbnail renders in the gallery grid.

- [ ] **Step 8: Validate gallery persistence**

Restart InvokeAI:
```bash
scripts/stop.sh && scripts/start.sh  # Select option 1
```

Expected: Previously generated images still appear in the gallery (named volume preserves database).

- [ ] **Step 9: Validate --all flag**

Run: `scripts/start.sh --all`
Expected: No interactive prompt. Both services start directly.

- [ ] **Step 10: Final commit**

If any fixes were needed during validation, commit them. Then:

```bash
git add -A
git commit -m "fix: validate CogVideoX i2v, InvokeAI gallery, and interactive service selection"
```
