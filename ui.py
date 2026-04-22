"""
Video Uniquelizer — Gradio Web UI
==================================
Full settings panel mirroring every mutation stage from the mindmap.
Runs preflight → shows GPU report → lets user configure → runs pipeline.
"""

import os
import sys
import threading
import logging
import traceback

# ---------------------------------------------------------------------------
# Python 3.10+ compatibility: collections.MutableMapping was removed in 3.10.
# It was moved to collections.abc in Python 3.3. Many third-party packages
# still reference collections.MutableMapping, so we monkey-patch it back.
# ---------------------------------------------------------------------------
import collections
import collections.abc
for _attr in ('MutableMapping', 'MutableSequence', 'MutableSet',
              'Mapping', 'Sequence', 'Set', 'Callable', 'Iterable',
              'Iterator', 'MutableSet'):
    if not hasattr(collections, _attr) and hasattr(collections.abc, _attr):
        setattr(collections, _attr, getattr(collections.abc, _attr))

import gradio as gr

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preflight import run_preflight, format_report
from pipeline import Pipeline, PipelineConfig, PipelineContext

logger = logging.getLogger("uniquelizer")
logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")


# ---------------------------------------------------------------------------
# Pipeline runner (runs in background thread for Gradio)
# ---------------------------------------------------------------------------

_pipeline_result = {"running": False, "log": "", "output_path": ""}


def _run_pipeline_thread(input_path, work_dir, config_dict):
    """Run the pipeline in a background thread, update shared state."""
    global _pipeline_result
    _pipeline_result["running"] = True
    _pipeline_result["log"] = "Starting pipeline...\n"
    _pipeline_result["output_path"] = ""

    # Build config from UI values
    cfg = PipelineConfig(
        gaussian_sigma=config_dict["gaussian_sigma"],
        lsb_flip_count=config_dict["lsb_flip_count"],
        hue_shift=config_dict["hue_shift"],
        sat_shift=config_dict["sat_shift"],
        jnd_model=config_dict["jnd_model"],
        jnd_sensitivity=config_dict["jnd_sensitivity"],
        gan_enabled=config_dict["gan_enabled"],
        gan_resolution=config_dict["gan_resolution"],
        gan_blend_alpha=config_dict["gan_blend_alpha"],
        gan_latent_delta=config_dict["gan_latent_delta"],
        adv_enabled=config_dict["adv_enabled"],
        adv_method=config_dict["adv_method"],
        adv_model=config_dict["adv_model"],
        adv_epsilon=config_dict["adv_epsilon"],
        temporal_jitter_ms=config_dict["temporal_jitter_ms"],
        temporal_trim_frames=config_dict["temporal_trim_frames"],
        temporal_speed_shift=config_dict["temporal_speed_shift"],
        audio_enabled=config_dict["audio_enabled"],
        audio_ultrasonic_freq=config_dict["audio_ultrasonic_freq"],
        audio_ultrasonic_amp=config_dict["audio_ultrasonic_amp"],
        audio_noise_floor_db=config_dict["audio_noise_floor_db"],
        audio_phase_shift_deg=config_dict["audio_phase_shift_deg"],
        encoder=config_dict["encoder"],
        crf=config_dict["crf"],
        inject_uuid=config_dict["inject_uuid"],
        shuffle_moov=config_dict["shuffle_moov"],
        ssim_threshold=config_dict["ssim_threshold"],
    )

    pipe = Pipeline(config=cfg)

    # Capture logs
    class LogCapture(logging.Handler):
        def emit(self, record):
            _pipeline_result["log"] += record.getMessage() + "\n"

    handler = LogCapture()
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    try:
        ctx = pipe.run(input_path, output_dir=work_dir)
        _pipeline_result["output_path"] = ctx.output_path
        _pipeline_result["log"] += "\n=== PIPELINE COMPLETE ===\n"
        for entry in ctx.log:
            _pipeline_result["log"] += entry + "\n"
        _pipeline_result["log"] += "\nSSIM: {:.4f}\n".format(ctx.ssim_score)
        _pipeline_result["log"] += "Original hash:  {}\n".format(ctx.original_hash[:32])
        _pipeline_result["log"] += "Output hash:    {}\n".format(ctx.output_hash[:32])
        _pipeline_result["log"] += "Hash changed:   {}\n".format(
            "YES" if ctx.output_hash != ctx.original_hash else "NO"
        )
        _pipeline_result["log"] += "\nOutput file: {}\n".format(ctx.output_path)
    except Exception as e:
        _pipeline_result["log"] += "\n!!! PIPELINE ERROR !!!\n"
        _pipeline_result["log"] += traceback.format_exc()
    finally:
        logger.removeHandler(handler)
        _pipeline_result["running"] = False


# ---------------------------------------------------------------------------
# Gradio handlers
# ---------------------------------------------------------------------------

def run_preflight_check():
    r = run_preflight()
    return format_report(r)


def start_pipeline(video_file, *settings):
    global _pipeline_result
    if _pipeline_result["running"]:
        return "Pipeline already running. Please wait.", ""

    if video_file is None:
        return "Please upload a video file first.", ""

    # Unpack settings — order must match the UI layout
    config_dict = {
        "gaussian_sigma": settings[0],
        "lsb_flip_count": int(settings[1]),
        "hue_shift": settings[2],
        "sat_shift": settings[3],
        "jnd_model": settings[4],
        "jnd_sensitivity": settings[5],
        "gan_enabled": settings[6],
        "gan_resolution": int(settings[7]),
        "gan_blend_alpha": settings[8],
        "gan_latent_delta": settings[9],
        "adv_enabled": settings[10],
        "adv_method": settings[11],
        "adv_model": settings[12],
        "adv_epsilon": settings[13],
        "temporal_jitter_ms": settings[14],
        "temporal_trim_frames": int(settings[15]),
        "temporal_speed_shift": settings[16],
        "audio_enabled": settings[17],
        "audio_ultrasonic_freq": settings[18],
        "audio_ultrasonic_amp": settings[19],
        "audio_noise_floor_db": settings[20],
        "audio_phase_shift_deg": settings[21],
        "encoder": settings[22],
        "crf": int(settings[23]),
        "inject_uuid": settings[24],
        "shuffle_moov": settings[25],
        "ssim_threshold": settings[26],
    }

    work_dir = os.path.join("/kaggle/working", "uniquelizer_output")
    os.makedirs(work_dir, exist_ok=True)

    thread = threading.Thread(
        target=_run_pipeline_thread,
        args=(video_file, work_dir, config_dict),
        daemon=True
    )
    thread.start()
    return "Pipeline started! Click 'Refresh Log' to see progress.", ""


def refresh_log():
    return _pipeline_result.get("log", "No pipeline run yet.")


def get_output_file():
    path = _pipeline_result.get("output_path", "")
    if path and os.path.exists(path):
        return path
    return None


# ---------------------------------------------------------------------------
# Build the UI
# ---------------------------------------------------------------------------

def build_ui():
    with gr.Blocks(
        title="Video Uniquelizer",
        theme=gr.themes.Soft(
            primary_hue="violet",
            secondary_hue="cyan",
            neutral_hue="slate",
        ),
    ) as app:

        gr.Markdown(
            """
            # Video Uniquelizer
            ### Perceptually invisible video mutation — new file hash every run
            Upload a video, configure mutation layers, and get a visually identical
            output with a completely different SHA-256 hash.
            """
        )

        # ---- Preflight Tab ----
        with gr.Tab("1. Preflight Check"):
            pf_output = gr.Textbox(label="System Report", lines=18, interactive=False)
            pf_btn = gr.Button("Run Preflight Check", variant="secondary")
            pf_btn.click(fn=run_preflight_check, outputs=pf_output)

        # ---- Settings + Run Tab ----
        with gr.Tab("2. Configure & Run"):
            gr.Markdown("### Upload video and tweak mutation parameters")

            with gr.Row():
                video_input = gr.Video(label="Input Video", sources=["upload"])

            with gr.Row():
                with gr.Accordion("Pixel Mutation", open=True):
                    gr.Markdown(
                        "Below-JND pixel modifications. Gaussian noise is weighted by "
                        "the JND perceptual mask so more noise is added where it's invisible."
                    )
                    gaussian_sigma = gr.Slider(
                        0.0, 3.0, value=1.0, step=0.1,
                        label="Gaussian Sigma (noise strength)"
                    )
                    lsb_flip_count = gr.Slider(
                        0, 8, value=3, step=1,
                        label="LSB Flip Bits (0-8, lower = subtler)"
                    )
                    hue_shift = gr.Slider(
                        0.0, 2.0, value=0.5, step=0.1,
                        label="Hue Shift (degrees)"
                    )
                    sat_shift = gr.Slider(
                        0.0, 5.0, value=1.0, step=0.5,
                        label="Saturation Shift (%)"
                    )

                with gr.Accordion("JND Perceptual Mask", open=True):
                    gr.Markdown(
                        "Controls how the pipeline decides what perturbation is invisible. "
                        "Watson DCT is more accurate but slower. Simple luminance is faster."
                    )
                    jnd_model = gr.Dropdown(
                        choices=["watson_dct", "simple_luminance", "off"],
                        value="watson_dct",
                        label="JND Model"
                    )
                    jnd_sensitivity = gr.Slider(
                        0.1, 2.0, value=1.0, step=0.1,
                        label="JND Sensitivity (< 1 = more aggressive, > 1 = subtler)"
                    )

            with gr.Row():
                with gr.Accordion("GAN Perturbation", open=True):
                    gr.Markdown(
                        "Generates structured noise using a tiny CNN generator. "
                        "More effective at breaking perceptual hash than random noise. "
                        "Auto-downscaled to 512x512 on GPUs with < 6GB VRAM."
                    )
                    gan_enabled = gr.Checkbox(value=True, label="Enable GAN Perturbation")
                    gan_resolution = gr.Dropdown(
                        choices=[512, 1024],
                        value=512,
                        label="GAN Resolution (512 = safe for all GPUs)"
                    )
                    gan_blend_alpha = gr.Slider(
                        0.01, 0.2, value=0.08, step=0.01,
                        label="Blend Alpha (perturbation opacity)"
                    )
                    gan_latent_delta = gr.Slider(
                        0.001, 0.05, value=0.01, step=0.001,
                        label="Latent Walk Delta (per-frame variation)"
                    )

                with gr.Accordion("Adversarial Pattern", open=True):
                    gr.Markdown(
                        "FGSM perturbation shifts the video's embedding in feature space, "
                        "defeating perceptual hashing (pHash, dHash, aHash). "
                        "EfficientNet-B0 uses only ~800MB VRAM."
                    )
                    adv_enabled = gr.Checkbox(value=True, label="Enable Adversarial")
                    adv_method = gr.Dropdown(
                        choices=["fgsm", "random_uniform"],
                        value="fgsm",
                        label="Method (FGSM = gradient-based, random = no GPU needed)"
                    )
                    adv_model = gr.Dropdown(
                        choices=["efficientnet_b0", "mobilenetv3", "resnet50"],
                        value="efficientnet_b0",
                        label="Classifier Model (smaller = less VRAM)"
                    )
                    adv_epsilon = gr.Slider(
                        0.001, 0.02, value=0.005, step=0.001,
                        label="Epsilon (L-inf bound, lower = subtler)"
                    )

            with gr.Row():
                with gr.Accordion("Temporal Mutation", open=True):
                    gr.Markdown(
                        "Time-domain perturbation: frame jitter, micro-trim, speed shift. "
                        "These change the container structure without affecting visual content."
                    )
                    temporal_jitter_ms = gr.Slider(
                        0.0, 5.0, value=2.0, step=0.5,
                        label="Frame Jitter (±ms per frame)"
                    )
                    temporal_trim_frames = gr.Slider(
                        0, 10, value=3, step=1,
                        label="Micro-Trim (±frames from start/end)"
                    )
                    temporal_speed_shift = gr.Slider(
                        0.0, 2.0, value=0.5, step=0.1,
                        label="Speed Micro-Shift (±% playback rate)"
                    )

                with gr.Accordion("Audio Mutation", open=True):
                    gr.Markdown(
                        "Sonic fingerprint change: ultrasonic injection (above human hearing), "
                        "spectral noise floor, and phase rotation. All inaudible to humans "
                        "but change the audio fingerprint that YouTube Content ID checks."
                    )
                    audio_enabled = gr.Checkbox(value=True, label="Enable Audio Mutation")
                    audio_ultrasonic_freq = gr.Slider(
                        16000, 20000, value=19000, step=500,
                        label="Ultrasonic Frequency (Hz)"
                    )
                    audio_ultrasonic_amp = gr.Slider(
                        0.001, 0.05, value=0.01, step=0.001,
                        label="Ultrasonic Amplitude"
                    )
                    audio_noise_floor_db = gr.Slider(
                        -80, -40, value=-60, step=5,
                        label="Noise Floor (dB, lower = quieter)"
                    )
                    audio_phase_shift_deg = gr.Slider(
                        0.0, 30.0, value=10.0, step=1.0,
                        label="Phase Shift (degrees, all-pass filter)"
                    )

            with gr.Row():
                with gr.Accordion("Re-encode & Hash Guarantee", open=True):
                    gr.Markdown(
                        "Re-encoding alone produces a different bitstream (new hash). "
                        "UUID injection and MOOV shuffle provide additional hash randomization. "
                        "Combined: every run produces a unique file even with same settings."
                    )
                    encoder = gr.Dropdown(
                        choices=["auto", "h264_nvenc", "libx264"],
                        value="auto",
                        label="Video Encoder (auto = NVENC if available)"
                    )
                    crf = gr.Slider(
                        18, 28, value=20, step=1,
                        label="Quality CRF/CQP (lower = better quality)"
                    )
                    inject_uuid = gr.Checkbox(value=True, label="Inject Random UUID in Metadata")
                    shuffle_moov = gr.Checkbox(value=True, label="Shuffle MOOV Atom Order")

                with gr.Accordion("QA Verification", open=True):
                    gr.Markdown(
                        "After mutation, SSIM is computed between original and output. "
                        "If SSIM drops below threshold, the mutations are too aggressive. "
                        "Aim for SSIM > 0.998 (imperceptible difference)."
                    )
                    ssim_threshold = gr.Slider(
                        0.99, 1.0, value=0.998, step=0.001,
                        label="SSIM Threshold (minimum similarity)"
                    )

            # Run controls
            gr.Markdown("---")
            with gr.Row():
                run_btn = gr.Button("Run Pipeline", variant="primary", size="lg")
                refresh_btn = gr.Button("Refresh Log", variant="secondary")

            with gr.Row():
                status_output = gr.Textbox(label="Status", lines=2)
                log_output = gr.Textbox(label="Pipeline Log", lines=20, interactive=False)

            run_btn.click(
                fn=start_pipeline,
                inputs=[
                    video_input,
                    gaussian_sigma, lsb_flip_count, hue_shift, sat_shift,
                    jnd_model, jnd_sensitivity,
                    gan_enabled, gan_resolution, gan_blend_alpha, gan_latent_delta,
                    adv_enabled, adv_method, adv_model, adv_epsilon,
                    temporal_jitter_ms, temporal_trim_frames, temporal_speed_shift,
                    audio_enabled, audio_ultrasonic_freq, audio_ultrasonic_amp,
                    audio_noise_floor_db, audio_phase_shift_deg,
                    encoder, crf, inject_uuid, shuffle_moov, ssim_threshold,
                ],
                outputs=[status_output, log_output]
            )

            refresh_btn.click(fn=refresh_log, outputs=log_output)

        # ---- Results Tab ----
        with gr.Tab("3. Results"):
            gr.Markdown("### Download the uniquelized video")
            with gr.Row():
                download_btn = gr.Button("Get Output File")
            file_output = gr.File(label="Uniquelized Video")

            def _download():
                path = get_output_file()
                if path:
                    return path
                return None

            download_btn.click(fn=_download, outputs=file_output)

    return app


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def launch():
    app = build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
    )


if __name__ == "__main__":
    launch()
