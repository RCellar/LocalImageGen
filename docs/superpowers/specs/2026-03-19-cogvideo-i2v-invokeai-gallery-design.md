# Fix CogVideoX Image-to-Video and InvokeAI Gallery

**Date:** 2026-03-19
**Status:** Draft

## Overview

Two defects prevent core functionality from working, plus a UX enhancement to prevent GPU resource contention:

1. **CogVideoX Image-to-Video** — The Gradio frontend's Image to Video tab errors immediately when used. The wrong model is loaded; the base `CogVideoX-5b` text-to-video model lacks the image encoder required by `CogVideoXImageToVideoPipeline`.
2. **InvokeAI Gallery** — Generated images never appear in the gallery. Images render briefly in the preview pane during generation but are not tracked by InvokeAI's internal database. The gallery has never worked.
3. **Interactive Service Selection** — When both services are enabled, `start.sh` launches both without warning, risking GPU OOM. Add an interactive prompt to let the user choose which services to start.

## Defect 1: CogVideoX Image-to-Video

### Root Cause

The project downloads `THUDM/CogVideoX-5b` (text-to-video only). The code at `containers/cogvideo/app.py:65` attempts to load this model with `CogVideoXImageToVideoPipeline.from_pretrained()`, which expects the separate `THUDM/CogVideoX-5b-I2V` model containing an image encoder. The mismatch causes an immediate failure, surfaced as a bare "Error" in the Gradio preview pane.

### Fix

#### Add the I2V model as a separate download

**config.yaml / config.example.yaml** — Add a new model entry:

```yaml
models:
  cogvideox-5b:
    enabled: true
    gated: false
    repo: THUDM/CogVideoX-5b
    type: video
  cogvideox-5b-i2v:
    enabled: true
    gated: false
    repo: THUDM/CogVideoX-5b-I2V
    type: video
```

No changes to `download-models.sh` — it already iterates all enabled models. The I2V model adds approximately 25GB of disk usage.

#### Route each pipeline to its correct model

**compose.yaml** — Add `I2V_MODEL_PATH` to the cogvideo service environment:

```yaml
environment:
  - MODEL_PATH=/models/cogvideox-5b
  - I2V_MODEL_PATH=/models/cogvideox-5b-i2v
```

**start.sh** — Pass `I2V_MODEL_PATH` in the CDI fallback `podman run` command for the video container:

```bash
-e "I2V_MODEL_PATH=/models/cogvideox-5b-i2v"
```

**app.py** — Add a second model path variable and use it in `_get_pipeline()`:

```python
I2V_MODEL_PATH = os.environ.get("I2V_MODEL_PATH", "/models/cogvideox-5b-i2v")
```

When `mode == "img2vid"`, load from `I2V_MODEL_PATH`. When `mode == "txt2vid"`, load from `MODEL_PATH` (unchanged).

#### Graceful degradation

At startup, check whether the `I2V_MODEL_PATH` directory exists and contains model files (check for `model_index.json` or `config.json`, consistent with the download verification in `download-models.sh`):

- **Present and valid:** Render the Image to Video tab normally.
- **Absent or empty:** Render the tab with inputs disabled and a visible message: "Image-to-Video model not downloaded. Run scripts/setup.sh to download it."

#### Error handling

Wrap pipeline loading and inference calls in try/except blocks. Surface actionable messages via `gr.Error()`:

- Pipeline load failure: `"Failed to load model: {error}. Check container logs."`
- CUDA OOM: `"GPU out of memory. Try reducing inference steps or frame count, or close other GPU applications."`
- Image dimension mismatch: `"Image dimensions not supported. CogVideoX-5B I2V expects 480x720 input. Your image will be resized automatically."`
- General exceptions: `"Generation failed: {error}"`

Additionally, resize the input image to the model's expected dimensions (480x720) before passing it to the pipeline, to avoid cryptic tensor shape errors from diffusers.

This replaces the current bare "Error" with messages that tell the user what went wrong and what to do about it.

## Defect 2: InvokeAI Gallery

### Root Cause

InvokeAI manages outputs through an internal SQLite database at `/invokeai/databases/invokeai.db`. The gallery is populated from this database, not by filesystem scanning. The current volume mount strategy creates a conflict:

```
invokeai-data:/invokeai              # named volume (DB, config, cache, outputs)
outputs/images:/invokeai/outputs     # bind mount overlays the named volume's outputs dir
```

The bind mount overlays `/invokeai/outputs` on top of the named volume. This likely causes one or more of:

- InvokeAI's expected subdirectory structure under `/invokeai/outputs/` (e.g., `images/`, `thumbnails/`) doesn't exist in the empty host bind mount on first run
- Internal tracking files and thumbnails that InvokeAI writes alongside images are lost or misplaced
- Database records reference paths that don't resolve correctly through the overlay

A secondary issue may exist with WebSocket event delivery under CDI fallback container networking, preventing real-time gallery updates even if storage is fixed.

### Fix

The fix has two parts: storage (primary) and networking (secondary). Both must be addressed.

#### Part 1: Fix the output volume mount strategy

The current bind mount at `/invokeai/outputs` overlays the named volume's outputs subdirectory, clobbering InvokeAI's expected directory structure. The fix uses a two-step approach:

**Step 1 — Remove the bind mount overlay.** Remove the `${OUTPUTS_DIR:-./outputs}/images:/invokeai/outputs` volume from both `compose.yaml` and the CDI fallback in `start.sh`. Let InvokeAI manage its full `/invokeai/` tree within the named volume, including its outputs directory, thumbnails, and metadata.

**Step 2 — Copy outputs to the host directory.** To preserve the design goal of host-accessible output images, add a post-generation sync mechanism. Two options, in order of preference:

- **Option A (symlink in entrypoint):** In `entrypoint.sh`, create a symlink from a known host-mounted path to InvokeAI's output directory. Mount a host directory at a non-conflicting path (e.g., `/host-outputs/images`) and symlink InvokeAI's image output subdirectory there. This requires determining InvokeAI's actual output subdirectory structure at implementation time by inspecting the running container.
- **Option B (remove host output requirement):** Accept that InvokeAI outputs live in the named volume only, accessible via the InvokeAI UI. Images can be downloaded individually from the gallery. This is simpler and avoids the mount conflict entirely.

**Decision rule:** During implementation, inspect a clean InvokeAI container to determine the exact directory structure under `/invokeai/outputs/`. If InvokeAI writes to a single, leaf-level subdirectory for final images (e.g., `/invokeai/outputs/images/`), bind-mount only that subdirectory to the host. If the structure is more complex or InvokeAI writes metadata/thumbnails alongside images in the same directory, use Option B.

**Fallback:** If neither option works cleanly, configure InvokeAI's output path via `invokeai.yaml` (placed in the named volume via entrypoint) to write to a dedicated directory that doesn't conflict with InvokeAI's internal structure, and bind-mount that directory to the host.

#### Part 2: Ensure directory pre-creation

Update the entrypoint (`containers/invokeai/entrypoint.sh`) to create any expected subdirectories before InvokeAI starts, so directories exist regardless of mount strategy.

#### Part 3: Network fix for WebSocket events

Add `--network=host` to CDI fallback containers in `start.sh`. When using host networking, remove the explicit `-p` port mapping flags (they conflict with `--network=host`). Note: `--network=host` exposes all container ports directly on the host, changing the security posture — acceptable for a local-only generation tool.

For the compose-native path, verify that the default bridge networking with explicit port mapping passes WebSocket traffic correctly during testing. If it does not, add `network_mode: host` to the compose service definition and remove the `ports` mapping. This ensures both launch paths are covered.

## Script Changes

### parse-config.py

Add `I2V_MODEL_PATH` to the generated `.env`:

```python
f"I2V_MODEL_PATH={paths.get('models', './models')}/cogvideox-5b-i2v",
```

### start.sh — Interactive service selection

When both services are enabled in config.yaml, prompt the user before launching:

```
Both InvokeAI and CogVideoX are enabled in config.yaml.
Running both simultaneously requires ~11GB+ VRAM.

Which services would you like to start?
  1) InvokeAI only (image generation)
  2) CogVideoX only (video generation)
  3) Both (requires sufficient VRAM)

Choice [1/2/3]:
```

Behavior:

- If only one service is enabled in config, skip the prompt and launch directly.
- The prompt refines the session's profile flags — it does not modify config.yaml.
- Add a `--all` flag to bypass the prompt for scripted/non-interactive use (launches whatever config says).

### start.sh — Model existence checks

Extend the video profile model check to warn about the i2v model:

```bash
video)
    if [ ! -d "$MODELS_DIR/cogvideox-5b" ]; then
        echo "WARNING: No video models found. CogVideoX will fail to start."
        echo "Run: scripts/download-models.sh"
    fi
    if [ ! -d "$MODELS_DIR/cogvideox-5b-i2v" ]; then
        echo "WARNING: I2V model not found. Image-to-Video will be disabled."
    fi
    ;;
```

### start.sh — CDI fallback changes

- Pass `-e "I2V_MODEL_PATH=/models/cogvideox-5b-i2v"` to the video container
- Add `--network=host` to both CDI fallback containers
- Remove `-p` port mapping flags when using `--network=host`

### compose.yaml

Add `I2V_MODEL_PATH` to the cogvideo service environment block. Remove or adjust the InvokeAI output bind mount per the decision rule in Defect 2. If compose-native WebSockets fail during testing, add `network_mode: host` and remove `ports` mapping.

### config.example.yaml

Add the `cogvideox-5b-i2v` model entry so new setups include it by default.

## Testing & Validation

### CogVideoX i2v

1. Run `scripts/setup.sh` — verify `cogvideox-5b-i2v` model downloads alongside the base model.
2. Start CogVideoX, navigate to Image to Video tab.
3. Upload an image, enter a prompt, click Generate Video.
4. Verify: no error, video generates, file appears in `outputs/videos/` as `img2vid_*.mp4`.
5. Switch to Text to Video tab, generate a video — verify pipeline swap works without crash.
6. Test graceful degradation: remove/rename the i2v model directory, restart, verify the tab is disabled with a clear message.

### InvokeAI gallery

1. Start InvokeAI, import SD 3.5 Medium via Model Manager.
2. Generate an image.
3. Verify: image appears in gallery without manual refresh.
4. Verify: thumbnail renders in gallery grid.
5. Verify: if using Option A (bind mount), image file exists on host at `outputs/images/`. If using Option B (no bind mount), images are accessible only via the InvokeAI gallery UI.
6. Restart the container — verify gallery still shows previous generations.
7. Repeat under CDI fallback: stop, force CDI path, regenerate, verify gallery works.

### Interactive prompt

1. Enable both services in config, run `start.sh` — verify prompt appears.
2. Select each option (1, 2, 3) — verify only the chosen services start.
3. Disable one service in config — verify no prompt, direct launch.
4. Run `start.sh --all` — verify prompt is bypassed.

### Resource contention

1. Start both services, generate from each — verify no OOM or crashes on RTX 4080 (16GB).
2. If OOM occurs, verify the error message is actionable (not bare "Error").

## Files Modified

| File | Change |
|------|--------|
| `config.yaml` | Add `cogvideox-5b-i2v` model entry |
| `config.example.yaml` | Add `cogvideox-5b-i2v` model entry |
| `compose.yaml` | Add `I2V_MODEL_PATH` env var; fix InvokeAI output mount strategy; possibly add `network_mode: host` |
| `containers/cogvideo/app.py` | Use `I2V_MODEL_PATH` for img2vid pipeline; add input image resizing, graceful degradation, and error handling |
| `containers/invokeai/entrypoint.sh` | Pre-create expected output subdirectories |
| `scripts/start.sh` | Interactive service prompt; CDI fallback: pass i2v env var, add `--network=host`; model checks |
| `scripts/parse-config.py` | Add `I2V_MODEL_PATH` to `.env` output |
