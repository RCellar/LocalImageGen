#!/usr/bin/env python3
"""CogVideoX-5B Gradio UI for local video generation (txt2vid + img2vid)."""

import gc
import os
import time
import torch
import gradio as gr
from diffusers import CogVideoXPipeline, CogVideoXImageToVideoPipeline
import imageio
import numpy as np

MODEL_PATH = os.environ.get("MODEL_PATH", "/models/cogvideox-5b")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/outputs/videos")
PORT = int(os.environ.get("COGVIDEO_PORT", "7860"))
QUANTIZATION = os.environ.get("COGVIDEO_QUANTIZATION", "none")  # "none" or "int8"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_video(frames, output_path, fps=8):
    """Save video frames as H.264 MP4 for browser/Discord compatibility."""
    writer = imageio.get_writer(
        output_path,
        fps=fps,
        codec="libx264",
        output_params=["-pix_fmt", "yuv420p"],
    )
    for frame in frames:
        if hasattr(frame, "numpy"):
            frame = frame.numpy()
        if isinstance(frame, np.floating):
            frame = (frame * 255).astype(np.uint8)
        writer.append_data(np.array(frame))
    writer.close()

# Pipeline manager — loads one pipeline at a time to fit in VRAM
# Both pipelines share the same base model weights (~5GB BF16),
# but loading both simultaneously doubles VRAM usage.
_current_pipe = None
_current_mode = None


def _get_pipeline(mode: str):
    """Load the requested pipeline, swapping out the current one if different."""
    global _current_pipe, _current_mode

    if _current_mode == mode and _current_pipe is not None:
        return _current_pipe

    # Unload current pipeline
    if _current_pipe is not None:
        print(f"Unloading {_current_mode} pipeline...")
        _current_pipe.to("cpu")
        del _current_pipe
        gc.collect()
        torch.cuda.empty_cache()

    print(f"Loading {mode} pipeline from {MODEL_PATH}...")
    if mode == "txt2vid":
        pipe = CogVideoXPipeline.from_pretrained(
            MODEL_PATH, torch_dtype=torch.bfloat16,
        )
    else:
        pipe = CogVideoXImageToVideoPipeline.from_pretrained(
            MODEL_PATH, torch_dtype=torch.bfloat16,
        )

    if QUANTIZATION == "int8":
        from torchao.quantization import quantize_, int8_weight_only
        quantize_(pipe.transformer, int8_weight_only())
        print("Applied INT8 quantization via torchao")

    pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    _current_pipe = pipe
    _current_mode = mode
    print(f"{mode} pipeline loaded successfully.")
    return pipe


# Pre-load txt2vid on startup
print(f"Loading CogVideoX-5B from {MODEL_PATH}...")
print(f"Quantization: {QUANTIZATION}")
_get_pipeline("txt2vid")
print("Model loaded successfully.")


def generate_txt2vid(
    prompt: str,
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    num_frames: int = 49,
):
    """Generate a video from a text prompt."""
    if not prompt.strip():
        raise gr.Error("Please enter a prompt.")

    pipe = _get_pipeline("txt2vid")
    video_frames = pipe(
        prompt=prompt,
        num_videos_per_prompt=1,
        num_inference_steps=num_inference_steps,
        num_frames=num_frames,
        guidance_scale=guidance_scale,
    ).frames[0]

    timestamp = int(time.time())
    output_path = os.path.join(OUTPUT_DIR, f"txt2vid_{timestamp}.mp4")
    save_video(video_frames, output_path, fps=8)

    return output_path


def generate_img2vid(
    image,
    prompt: str,
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    num_frames: int = 49,
):
    """Generate a video from an image and text prompt."""
    if image is None:
        raise gr.Error("Please upload a starting image.")
    if not prompt.strip():
        raise gr.Error("Please enter a prompt.")

    pipe = _get_pipeline("img2vid")
    video_frames = pipe(
        image=image,
        prompt=prompt,
        num_videos_per_prompt=1,
        num_inference_steps=num_inference_steps,
        num_frames=num_frames,
        guidance_scale=guidance_scale,
    ).frames[0]

    timestamp = int(time.time())
    output_path = os.path.join(OUTPUT_DIR, f"img2vid_{timestamp}.mp4")
    save_video(video_frames, output_path, fps=8)

    return output_path


with gr.Blocks(title="CogVideoX-5B — Local Video Generation") as demo:
    gr.Markdown("# CogVideoX-5B — Local Video Generation")
    gr.Markdown("Generate short video clips from text or images. Running locally on your GPU.")

    with gr.Tabs():
        with gr.TabItem("Text to Video"):
            with gr.Row():
                with gr.Column(scale=2):
                    t2v_prompt = gr.Textbox(
                        label="Prompt",
                        placeholder="A golden retriever playing in a field of sunflowers...",
                        lines=3,
                    )
                    with gr.Row():
                        t2v_steps = gr.Slider(
                            minimum=10, maximum=100, value=50, step=1,
                            label="Inference Steps",
                        )
                        t2v_guidance = gr.Slider(
                            minimum=1.0, maximum=15.0, value=6.0, step=0.5,
                            label="Guidance Scale",
                        )
                        t2v_frames = gr.Slider(
                            minimum=9, maximum=49, value=49, step=8,
                            label="Number of Frames",
                        )
                    t2v_btn = gr.Button("Generate Video", variant="primary")
                with gr.Column(scale=2):
                    t2v_output = gr.Video(label="Generated Video")

            t2v_btn.click(
                fn=generate_txt2vid,
                inputs=[t2v_prompt, t2v_steps, t2v_guidance, t2v_frames],
                outputs=[t2v_output],
            )

        with gr.TabItem("Image to Video"):
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

# Gradio serves its UI at / which returns HTTP 200 — used as health check.
# Container workarounds:
# 1. Patch url_ok: Gradio's localhost self-check fails in rootless podman
# 2. Patch get_api_info: gradio_client crashes parsing diffusers pipeline
#    JSON schemas (bool additionalProperties). Return minimal valid API info.
import gradio.networking
gradio.networking.url_ok = lambda url: True

_original_get_api_info = demo.get_api_info
def _safe_get_api_info():
    try:
        return _original_get_api_info()
    except TypeError:
        return {"named_endpoints": {}, "unnamed_endpoints": {}}
demo.get_api_info = _safe_get_api_info

demo.queue()
demo.launch(
    server_name="0.0.0.0",
    server_port=PORT,
    share=False,
    allowed_paths=[OUTPUT_DIR],
)
