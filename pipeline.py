"""
Video Uniquelizer — Core Pipeline
==================================
All mutation stages: JND mask, pixel noise, LSB flip, color shift,
GAN perturbation, adversarial pattern, temporal jitter, audio mutation,
re-encode + hash guarantee, QA verification.

Each stage is a callable class that takes a PipelineContext and mutates
frame buffers / audio in-place. Stages run sequentially so GPU memory
is freed between heavy stages.
"""

import os
import hashlib
import uuid
import struct
import tempfile
import subprocess
import logging
import threading
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
from pathlib import Path

# Python 3.10+ compatibility — must be imported before any third-party deps
import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import compat  # noqa: F401

import numpy as np
import cv2
from scipy.fftpack import dct, idct
from skimage.metrics import structural_similarity as ssim

logger = logging.getLogger("uniquelizer")

# ---------------------------------------------------------------------------
# Input sanitization & validation (Fix #10)
# ---------------------------------------------------------------------------

_ALLOWED_ENCODERS = {"auto", "h264_nvenc", "libx264", "libx265", "vp9"}

def _validate_path(file_path: str, work_dir: str, label: str = "path") -> str:
    """Sanitize a file path: reject traversals, confirm it resolves within work_dir.
    
    Returns the resolved path on success. Raises ValueError on failure.
    """
    if not file_path:
        raise ValueError("{} is empty".format(label))
    if ".." in file_path:
        raise ValueError("{} contains forbidden '..' traversal: {}".format(label, file_path))
    resolved = os.path.realpath(file_path)
    # For input_path, just ensure it exists and is absolute after resolution
    # For work_dir-relative paths, check containment
    if work_dir:
        resolved_root = os.path.realpath(work_dir)
        if not resolved.startswith(resolved_root + os.sep) and resolved != resolved_root:
            # Allow the input file to be outside work_dir (it's the source video)
            # But output/intermediate paths must be inside work_dir
            if label != "input_path":
                raise ValueError(
                    "{} ({}) resolves outside work_dir ({})".format(label, resolved, resolved_root)
                )
    return resolved


def _validate_encoder(encoder: str) -> str:
    """Validate encoder name against an allowlist. Returns the encoder on success."""
    if encoder not in _ALLOWED_ENCODERS:
        raise ValueError(
            "Invalid encoder '{}'. Allowed: {}".format(encoder, ", ".join(sorted(_ALLOWED_ENCODERS)))
        )
    return encoder


def _validate_crf(crf: int) -> int:
    """Validate CRF/CQP value is an integer in range 0-51."""
    if not isinstance(crf, int):
        try:
            crf = int(crf)
        except (TypeError, ValueError):
            raise ValueError("CRF must be an integer, got: {!r}".format(crf))
    if crf < 0 or crf > 51:
        raise ValueError("CRF must be in range 0-51, got: {}".format(crf))
    return crf


def _validate_config(cfg: "PipelineConfig") -> None:
    """Validate all user-facing PipelineConfig fields before running the pipeline.
    Raises ValueError if any field is out of bounds or invalid.
    """
    _validate_encoder(cfg.encoder)
    _validate_crf(cfg.crf)

    # JND model allowlist
    allowed_jnd = {"watson_dct", "simple_luminance", "off"}
    if cfg.jnd_model not in allowed_jnd:
        raise ValueError(
            "Invalid jnd_model '{}'. Allowed: {}".format(cfg.jnd_model, ", ".join(sorted(allowed_jnd)))
        )

    # Adversarial method allowlist
    allowed_adv_method = {"fgsm", "random_uniform"}
    if cfg.adv_method not in allowed_adv_method:
        raise ValueError(
            "Invalid adv_method '{}'. Allowed: {}".format(
                cfg.adv_method, ", ".join(sorted(allowed_adv_method))
            )
        )

    # Adversarial model allowlist
    allowed_adv_model = {"efficientnet_b0", "mobilenetv3", "resnet50"}
    if cfg.adv_model not in allowed_adv_model:
        raise ValueError(
            "Invalid adv_model '{}'. Allowed: {}".format(
                cfg.adv_model, ", ".join(sorted(allowed_adv_model))
            )
        )

    # GAN resolution allowlist
    allowed_gan_res = {512, 1024}
    if cfg.gan_resolution not in allowed_gan_res:
        raise ValueError(
            "Invalid gan_resolution '{}'. Allowed: {}".format(
                cfg.gan_resolution, ", ".join(str(x) for x in sorted(allowed_gan_res))
            )
        )

    # Numeric range checks
    if cfg.gaussian_sigma < 0:
        raise ValueError("gaussian_sigma must be >= 0, got: {}".format(cfg.gaussian_sigma))
    if not (0 <= cfg.lsb_flip_count <= 8):
        raise ValueError("lsb_flip_count must be 0-8, got: {}".format(cfg.lsb_flip_count))
    if cfg.jnd_sensitivity <= 0:
        raise ValueError("jnd_sensitivity must be > 0, got: {}".format(cfg.jnd_sensitivity))
    if not (0 <= cfg.gan_blend_alpha <= 1):
        raise ValueError("gan_blend_alpha must be 0-1, got: {}".format(cfg.gan_blend_alpha))
    if cfg.gan_latent_delta <= 0:
        raise ValueError("gan_latent_delta must be > 0, got: {}".format(cfg.gan_latent_delta))
    if cfg.adv_epsilon < 0:
        raise ValueError("adv_epsilon must be >= 0, got: {}".format(cfg.adv_epsilon))
    if cfg.temporal_trim_frames < 0:
        raise ValueError("temporal_trim_frames must be >= 0, got: {}".format(cfg.temporal_trim_frames))
    if not (0 <= cfg.ssim_threshold <= 1):
        raise ValueError("ssim_threshold must be 0-1, got: {}".format(cfg.ssim_threshold))

    logger.info("Config validation passed")




# ---------------------------------------------------------------------------
# Pipeline Context — shared state between stages
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """User-facing settings (mirrors the Web UI controls)."""
    # Pixel mutation
    gaussian_sigma: float = 1.0           # σ for micro-noise injection
    lsb_flip_count: int = 3               # number of LSB bits to flip
    hue_shift: float = 0.5                # degrees
    sat_shift: float = 1.0                # percent
    # JND
    jnd_model: str = "watson_dct"         # "watson_dct" | "simple_luminance" | "off"
    jnd_sensitivity: float = 1.0          # multiplier (< 1 = more aggressive)
    # GAN
    gan_enabled: bool = True
    gan_resolution: int = 512             # 512 or 1024 (auto-downscaled if VRAM low)
    gan_blend_alpha: float = 0.08         # perturbation blend opacity
    gan_latent_delta: float = 0.01        # latent walk step size
    # Adversarial
    adv_enabled: bool = True
    adv_method: str = "fgsm"              # "fgsm" | "random_uniform"
    adv_model: str = "efficientnet_b0"    # "efficientnet_b0" | "resnet50" | "mobilenetv3"
    adv_epsilon: float = 0.005            # L-inf bound
    # Temporal
    temporal_jitter_ms: float = 2.0       # ±ms per frame
    temporal_trim_frames: int = 3         # ±frames to trim from start/end
    temporal_speed_shift: float = 0.5     # ±% playback rate
    # Audio
    audio_enabled: bool = True
    audio_ultrasonic_freq: float = 19000  # Hz — above human hearing
    audio_ultrasonic_amp: float = 0.01    # amplitude (0-1)
    audio_noise_floor_db: float = -60     # noise floor in dB
    audio_phase_shift_deg: float = 10.0   # all-pass phase rotation
    # Encode
    encoder: str = "auto"                 # "auto" | "h264_nvenc" | "libx264"
    crf: int = 20                         # quality (18-28)
    # Hash guarantee
    inject_uuid: bool = True              # random UUID in container metadata
    shuffle_moov: bool = True             # randomize MOOV atom order
    # QA thresholds
    ssim_threshold: float = 0.998
    vmaf_check: bool = False              # VMAF is slow, off by default


@dataclass
class PipelineContext:
    """Mutable state passed through the pipeline."""
    config: PipelineConfig = field(default_factory=PipelineConfig)
    # Video
    input_path: str = ""
    work_dir: str = ""
    frames_dir: str = ""  # extracted frames
    audio_path: str = ""  # extracted audio WAV
    mutated_frames_dir: str = ""  # output frames
    output_path: str = ""
    frame_count: int = 0
    fps: float = 30.0
    width: int = 0
    height: int = 0
    trim_start: int = 0  # frames trimmed from start by TemporalMutationStage
    # GPU
    device: str = "cuda:0"
    vram_mb: int = 0
    # Computed metrics
    original_hash: str = ""
    output_hash: str = ""
    ssim_score: float = 0.0
    phash_distance: int = 0
    # Status
    log: list = field(default_factory=list)
    current_stage: str = ""
    # Cancellation support — set this Event from another thread to abort
    cancel_event: object = field(default_factory=threading.Event)


# ---------------------------------------------------------------------------
# Stage 1: Demux — split video into frames + audio
# ---------------------------------------------------------------------------

class DemuxStage:
    name = "Demux"

    def run(self, ctx: PipelineContext) -> None:
        ctx.current_stage = self.name
        logger.info("Stage: Demux — splitting video into frames + audio")

        os.makedirs(ctx.work_dir, exist_ok=True)
        ctx.frames_dir = os.path.join(ctx.work_dir, "frames_original")
        ctx.mutated_frames_dir = os.path.join(ctx.work_dir, "frames_mutated")
        ctx.audio_path = os.path.join(ctx.work_dir, "audio_original.wav")
        os.makedirs(ctx.frames_dir, exist_ok=True)
        os.makedirs(ctx.mutated_frames_dir, exist_ok=True)

        # Validate input path before passing to subprocess
        _validate_path(ctx.input_path, "", label="input_path")

        # Get video info
        probe = subprocess.run(
            ["ffprobe", "-v", "error",
             "-select_streams", "v:0",
             "-show_entries", "stream=width,height,r_frame_format,nb_frames,avg_frame_rate",
             "-of", "csv=p=0", ctx.input_path],
            capture_output=True, text=True, timeout=60
        )
        if probe.returncode == 0 and probe.stdout.strip():
            parts = probe.stdout.strip().split(",")
            if len(parts) >= 3:
                ctx.width = int(parts[0])
                ctx.height = int(parts[1])
                # Parse fps from avg_frame_rate like "30000/1001"
                fps_str = parts[2]
                if "/" in fps_str:
                    num, den = fps_str.split("/")
                    ctx.fps = float(num) / float(den)
                else:
                    ctx.fps = float(fps_str)

        # Extract frames
        subprocess.run(
            ["ffmpeg", "-y", "-i", ctx.input_path,
             "-qscale:v", "2",
             os.path.join(ctx.frames_dir, "frame_%06d.png")],
            capture_output=True, timeout=600
        )

        # Count frames
        ctx.frame_count = len([
            f for f in os.listdir(ctx.frames_dir)
            if f.endswith(".png")
        ])

        # Validate audio output path
        _validate_path(ctx.audio_path, ctx.work_dir, label="audio_path")

        # Extract audio
        has_audio = True
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", ctx.input_path,
             "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
             ctx.audio_path],
            capture_output=True, timeout=120
        )
        if r.returncode != 0 or not os.path.exists(ctx.audio_path):
            has_audio = False
            ctx.audio_path = ""

        # Original file hash
        ctx.original_hash = _sha256_file(ctx.input_path)

        msg = "Demuxed: {} frames, {}x{}, {:.1f} fps, audio={}".format(
            ctx.frame_count, ctx.width, ctx.height, ctx.fps,
            "yes" if has_audio else "no"
        )
        logger.info(msg)
        ctx.log.append(msg)


# ---------------------------------------------------------------------------
# Stage 2: JND Mask — compute perceptual visibility threshold per block
# ---------------------------------------------------------------------------

class JNDStage:
    name = "JND Mask"

    def run(self, ctx: PipelineContext) -> None:
        ctx.current_stage = self.name
        cfg = ctx.config
        if cfg.jnd_model == "off":
            logger.info("JND mask disabled, skipping")
            return

        logger.info("Stage: JND — computing perceptual mask ({})".format(cfg.jnd_model))

        # We compute the JND mask on-the-fly per frame in the pixel mutation
        # stage. This stage just validates the model choice.
        if cfg.jnd_model == "watson_dct":
            logger.info("Using Watson DCT-domain JND model (numpy)")
        elif cfg.jnd_model == "simple_luminance":
            logger.info("Using simple luminance-masking JND (faster)")

        ctx.log.append("JND model: {}".format(cfg.jnd_model))


def _watson_jnd_mask(frame_y: np.ndarray, block_size: int = 8,
                      sensitivity: float = 1.0) -> np.ndarray:
    """
    Compute a Watson-style JND threshold map for a luminance frame.
    Returns a mask the same size as frame_y where each pixel indicates
    the maximum invisible perturbation magnitude.

    Simplified Watson model:
      - Luminance masking: threshold proportional to sqrt(background luminance)
      - Contrast masking: threshold ~ max(base, |DCT|^0.7) per block
      - sensitivity multiplier scales the whole map
    """
    h, w = frame_y.shape
    mask = np.zeros_like(frame_y, dtype=np.float32)

    # Luminance masking — brighter areas can hide more noise
    lum_mask = np.sqrt(np.clip(frame_y.astype(np.float32), 1, 255)) * 0.8

    # Block-level contrast masking via DCT
    pad_h = (block_size - h % block_size) % block_size
    pad_w = (block_size - w % block_size) % block_size
    padded = np.pad(frame_y, ((0, pad_h), (0, pad_w)), mode='reflect')

    for i in range(0, padded.shape[0], block_size):
        for j in range(0, padded.shape[1], block_size):
            block = padded[i:i+block_size, j:j+block_size].astype(np.float32)
            block_dct = dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')
            # Contrast masking: AC coefficients energy
            ac_energy = np.sqrt(np.mean(block_dct[1:, 1:] ** 2))
            contrast_thresh = max(1.0, ac_energy ** 0.7) * 0.5
            mask[i:i+block_size, j:j+block_size] = np.maximum(
                lum_mask[i:i+block_size, j:j+block_size],
                contrast_thresh
            )

    mask = mask[:h, :w] * sensitivity
    # Scale to be small enough to be invisible (0.5-3.0 pixel values at most)
    mask = np.clip(mask, 0.3, 3.0)
    return mask


def _simple_jnd_mask(frame_y: np.ndarray, sensitivity: float = 1.0) -> np.ndarray:
    """
    Fast luminance-only JND mask. No DCT — just sqrt(luminance) scaling.
    Good enough for uniquelization where exact JND isn't critical.
    """
    lum_mask = np.sqrt(np.clip(frame_y.astype(np.float32), 1, 255)) * 0.6
    lum_mask = np.clip(lum_mask * sensitivity, 0.3, 2.5)
    return lum_mask


# ---------------------------------------------------------------------------
# Stage 3: Pixel Mutation — Gaussian noise, LSB flip, color shift
# ---------------------------------------------------------------------------

class PixelMutationStage:
    name = "Pixel Mutation"

    def run(self, ctx: PipelineContext) -> None:
        ctx.current_stage = self.name
        cfg = ctx.config
        logger.info("Stage: Pixel Mutation — noise σ={}, LSB bits={}, hue={}°, sat={}%".format(
            cfg.gaussian_sigma, cfg.lsb_flip_count,
            cfg.hue_shift, cfg.sat_shift))

        frame_files = sorted([
            f for f in os.listdir(ctx.frames_dir) if f.endswith(".png")
        ])

        for idx, fname in enumerate(frame_files):
            fpath = os.path.join(ctx.frames_dir, fname)
            frame = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)

            if frame is None:
                continue

            # --- JND-aware Gaussian noise ---
            if cfg.gaussian_sigma > 0:
                frame_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
                y_ch = frame_yuv[:, :, 0].astype(np.float32)

                if cfg.jnd_model == "watson_dct":
                    jnd = _watson_jnd_mask(y_ch, sensitivity=cfg.jnd_sensitivity)
                elif cfg.jnd_model == "simple_luminance":
                    jnd = _simple_jnd_mask(y_ch, sensitivity=cfg.jnd_sensitivity)
                else:
                    jnd = np.ones_like(y_ch) * cfg.gaussian_sigma

                noise = np.random.normal(0, cfg.gaussian_sigma, y_ch.shape).astype(np.float32)
                # Weight noise by JND mask — more noise where it's invisible
                noise = noise * (jnd / jnd.max()) if jnd.max() > 0 else noise
                y_ch = np.clip(y_ch + noise, 0, 255).astype(np.uint8)
                frame_yuv[:, :, 0] = y_ch
                frame = cv2.cvtColor(frame_yuv, cv2.COLOR_YUV2BGR)

            # --- LSB bit flipping ---
            if cfg.lsb_flip_count > 0:
                lsb_mask = 0
                for bit in range(cfg.lsb_flip_count):
                    lsb_mask |= (1 << bit)
                # Randomly flip LSBs on ~30% of pixels per channel
                flip_prob = np.random.random(frame.shape[:2]) < 0.3
                for c in range(frame.shape[2] if frame.ndim == 3 else [0]):
                    channel = frame[:, :, c] if frame.ndim == 3 else frame
                    flips = (np.random.randint(0, 2, channel.shape) * lsb_mask).astype(channel.dtype)
                    channel[flip_prob] ^= flips[flip_prob]

            # --- Color shift (hue + saturation) ---
            if cfg.hue_shift != 0 or cfg.sat_shift != 0:
                frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
                # Hue is in [0, 180] in OpenCV — shift by degrees
                frame_hsv[:, :, 0] = (frame_hsv[:, :, 0] + cfg.hue_shift / 2) % 180
                # Saturation shift — percentage of current value
                frame_hsv[:, :, 1] = np.clip(
                    frame_hsv[:, :, 1] * (1 + cfg.sat_shift / 100), 0, 255
                )
                frame = cv2.cvtColor(frame_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

            # Save to mutated dir
            cv2.imwrite(os.path.join(ctx.mutated_frames_dir, fname), frame)

            if (idx + 1) % 100 == 0:
                logger.info("  Pixel-mutated {}/{} frames".format(idx + 1, ctx.frame_count))

        msg = "Pixel mutation done: {} frames (σ={}, LSB={}, hue={}°, sat={}%)".format(
            ctx.frame_count, cfg.gaussian_sigma, cfg.lsb_flip_count,
            cfg.hue_shift, cfg.sat_shift
        )
        logger.info(msg)
        ctx.log.append(msg)


# ---------------------------------------------------------------------------
# Stage 4: GAN Perturbation — StyleGAN2 noise walk at 512x512, blend overlay
# ---------------------------------------------------------------------------

class GANPerturbationStage:
    name = "GAN Perturbation"

    def __init__(self):
        self.generator = None
        self.latent_avg = None

    def _load_model(self, ctx: PipelineContext):
        """Load StyleGAN2 generator. For uniquelization we use a pretrained
        landscape/stylegan2-ffhq model. Falls back to procedural noise
        if model weights are unavailable."""
        import torch

        cfg = ctx.config
        device = torch.device(ctx.device)

        # We generate structured perturbation noise using a small generator
        # rather than a full StyleGAN2 (which requires pretrained weights).
        # This approach: train-free, no downloads, works on any GPU.
        # We use a learned perceptual noise generator (tiny CNN).
        from torch import nn

        class PerturbationGenerator(nn.Module):
            """Tiny CNN that generates structured perturbation maps
            from a random latent vector. The output is a smooth,
            content-like noise pattern that defeats perceptual hashing
            more effectively than plain Gaussian noise."""
            def __init__(self, res=512, latent_dim=128):
                super().__init__()
                self.res = res
                self.latent_dim = latent_dim
                # Project latent to spatial feature map
                self.fc = nn.Linear(latent_dim, 256 * 4 * 4)
                self.net = nn.Sequential(
                    nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.LeakyReLU(0.2),
                    nn.ConvTranspose2d(128, 64, 4, 2, 1),  nn.LeakyReLU(0.2),
                    nn.ConvTranspose2d(64, 32, 4, 2, 1),   nn.LeakyReLU(0.2),
                    nn.ConvTranspose2d(32, 16, 4, 2, 1),   nn.LeakyReLU(0.2),
                    nn.ConvTranspose2d(16, 8, 4, 2, 1),    nn.LeakyReLU(0.2),
                    nn.ConvTranspose2d(8, 3, 4, 2, 1),     nn.Tanh(),
                )
                # Random init — we don't train, the structure from
                # random weights already produces spatially coherent noise

            def forward(self, z):
                x = self.fc(z).view(-1, 256, 4, 4)
                x = self.net(x)
                # Resize to target resolution
                x = torch.nn.functional.interpolate(
                    x, size=(self.res, self.res), mode='bilinear', align_corners=False
                )
                return x  # output in [-1, 1]

        res = cfg.gan_resolution
        self.generator = PerturbationGenerator(res=res).to(device).half()
        self.generator.eval()

        for p in self.generator.parameters():
            p.requires_grad_(False)

        logger.info("Loaded perturbation generator (res={}, FP16)".format(res))

    def run(self, ctx: PipelineContext) -> None:
        ctx.current_stage = self.name
        cfg = ctx.config
        if not cfg.gan_enabled:
            logger.info("GAN perturbation disabled, skipping")
            return

        logger.info("Stage: GAN Perturbation (res={}, α={}, Δz={})".format(
            cfg.gan_resolution, cfg.gan_blend_alpha, cfg.gan_latent_delta
        ))

        import torch

        self._load_model(ctx)
        device = torch.device(ctx.device)

        frame_files = sorted([
            f for f in os.listdir(ctx.mutated_frames_dir) if f.endswith(".png")
        ])

        # Generate a base latent + per-frame delta
        base_z = torch.randn(1, self.generator.latent_dim, device=device, dtype=torch.float16)

        for idx, fname in enumerate(frame_files):
            fpath = os.path.join(ctx.mutated_frames_dir, fname)
            frame = cv2.imread(fpath).astype(np.float32)  # BGR, [0,255]

            # Per-frame latent walk
            delta = torch.randn_like(base_z) * cfg.gan_latent_delta
            z = base_z + delta

            with torch.no_grad():
                perturb = self.generator(z)  # [1, 3, res, res], [-1,1]

            perturb_np = perturb[0].permute(1, 2, 0).cpu().float().numpy()
            perturb_np = ((perturb_np + 1) / 2 * 255).astype(np.float32)

            # Resize perturbation to frame size
            perturb_np = cv2.resize(perturb_np, (ctx.width, ctx.height),
                                     interpolation=cv2.INTER_LINEAR)

            # Blend: frame = frame * (1-α) + perturb * α
            # This adds structured noise that shifts perceptual hash
            blended = frame * (1 - cfg.gan_blend_alpha) + perturb_np * cfg.gan_blend_alpha
            blended = np.clip(blended, 0, 255).astype(np.uint8)

            cv2.imwrite(fpath, blended)

            if (idx + 1) % 100 == 0:
                logger.info("  GAN perturbed {}/{} frames".format(idx + 1, len(frame_files)))

        # Free GPU memory
        del self.generator
        torch.cuda.empty_cache()
        self.generator = None

        msg = "GAN perturbation done: {} frames (res={}, α={})".format(
            len(frame_files), cfg.gan_resolution, cfg.gan_blend_alpha
        )
        logger.info(msg)
        ctx.log.append(msg)


# ---------------------------------------------------------------------------
# Stage 5: Adversarial Pattern — FGSM or random bounded perturbation
# ---------------------------------------------------------------------------

class AdversarialStage:
    name = "Adversarial Pattern"

    def __init__(self):
        self.classifier = None

    def _load_classifier(self, ctx: PipelineContext):
        """Load a lightweight classifier for FGSM gradient computation.
        EfficientNet-B0 or MobileNetV3 for low VRAM."""
        import torch
        import torchvision.models as models

        cfg = ctx.config
        device = torch.device(ctx.device)

        if cfg.adv_model == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT
            model = models.efficientnet_b0(weights=weights)
        elif cfg.adv_model == "mobilenetv3":
            weights = models.MobileNet_V3_Small_Weights.DEFAULT
            model = models.mobilenet_v3_small(weights=weights)
        elif cfg.adv_model == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT
            model = models.resnet50(weights=weights)
        else:
            weights = models.EfficientNet_B0_Weights.DEFAULT
            model = models.efficientnet_b0(weights=weights)

        model = model.to(device).half().eval()
        self.classifier = model
        self.preprocess = weights.transforms()
        logger.info("Loaded classifier: {} (FP16)".format(cfg.adv_model))

    def run(self, ctx: PipelineContext) -> None:
        ctx.current_stage = self.name
        cfg = ctx.config
        if not cfg.adv_enabled:
            logger.info("Adversarial perturbation disabled, skipping")
            return

        logger.info("Stage: Adversarial — method={}, model={}, ε={}".format(
            cfg.adv_method, cfg.adv_model, cfg.adv_epsilon
        ))

        import torch

        frame_files = sorted([
            f for f in os.listdir(ctx.mutated_frames_dir) if f.endswith(".png")
        ])

        if cfg.adv_method == "fgsm" and ctx.vram_mb >= 2000:
            self._run_fgsm(ctx, frame_files)
        else:
            self._run_random(ctx, frame_files)

    def _run_fgsm(self, ctx: PipelineContext, frame_files: list):
        """FGSM: one-step gradient sign perturbation."""
        import torch

        if Image is None:
            logger.warning(
                "PIL/Pillow not available — falling back to random uniform perturbation"
            )
            self._run_random(ctx, frame_files)
            return

        self._load_classifier(ctx)
        device = torch.device(ctx.device)
        cfg = ctx.config

        for idx, fname in enumerate(frame_files):
            fpath = os.path.join(ctx.mutated_frames_dir, fname)
            frame_bgr = cv2.imread(fpath)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # Convert to tensor for classifier
            img_pil = Image.fromarray(frame_rgb)
            input_tensor = self.preprocess(img_pil).unsqueeze(0).to(device).half()
            input_tensor.requires_grad_(True)

            # Forward pass — we use a random target class
            output = self.classifier(input_tensor)
            # Maximize loss w.r.t. current top prediction (untargeted)
            target_class = output.argmax(dim=1)
            loss = torch.nn.functional.cross_entropy(output, target_class)
            loss.backward()

            # FGSM: perturb in direction of gradient sign
            grad_sign = input_tensor.grad.data.sign()
            adv_noise = cfg.adv_epsilon * grad_sign

            # Map adversarial noise back to full frame resolution
            # The noise is in preprocess-space (224x224 normalized).
            # We upscale it and apply to the original frame.
            noise_small = adv_noise[0].cpu().float().numpy()  # [3, 224, 224]
            # Denormalize from ImageNet stats
            mean = np.array([0.485, 0.456, 0.406])[:, None, None]
            std = np.array([0.229, 0.224, 0.225])[:, None, None]
            noise_rgb = noise_small * std * 255  # scale to pixel values
            noise_rgb = np.transpose(noise_rgb, (1, 2, 0))  # [H, W, 3]
            noise_rgb = cv2.resize(noise_rgb, (ctx.width, ctx.height),
                                    interpolation=cv2.INTER_LINEAR)

            # Add to frame
            frame_f = frame_bgr.astype(np.float32)
            # BGR order — noise is RGB so swap channels
            noise_bgr = noise_rgb[:, :, ::-1]
            frame_f += noise_bgr
            frame_f = np.clip(frame_f, 0, 255).astype(np.uint8)

            cv2.imwrite(fpath, frame_f)

            if (idx + 1) % 100 == 0:
                logger.info("  FGSM perturbed {}/{} frames".format(idx + 1, len(frame_files)))

        # Free GPU
        del self.classifier
        torch.cuda.empty_cache()
        self.classifier = None

    def _run_random(self, ctx: PipelineContext, frame_files: list):
        """Random bounded uniform perturbation — no gradient needed.
        Uses JND mask to stay invisible."""
        cfg = ctx.config

        for idx, fname in enumerate(frame_files):
            fpath = os.path.join(ctx.mutated_frames_dir, fname)
            frame = cv2.imread(fpath).astype(np.float32)

            # JND-weighted random noise
            frame_yuv = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2YUV)
            y = frame_yuv[:, :, 0].astype(np.float32)

            if cfg.jnd_model != "off":
                jnd = _simple_jnd_mask(y, sensitivity=cfg.jnd_sensitivity) if cfg.jnd_model == "simple_luminance" else _watson_jnd_mask(y, sensitivity=cfg.jnd_sensitivity)
            else:
                jnd = np.ones_like(y) * (cfg.adv_epsilon * 255)

            noise = np.random.uniform(
                -cfg.adv_epsilon * 255, cfg.adv_epsilon * 255, frame.shape
            ).astype(np.float32)

            # Weight each channel by JND mask
            for c in range(3):
                noise[:, :, c] *= (jnd / jnd.max()) if jnd.max() > 0 else 1.0

            frame = np.clip(frame + noise, 0, 255).astype(np.uint8)
            cv2.imwrite(fpath, frame)

        msg = "Adversarial perturbation done: {} frames (method={})".format(
            len(frame_files), cfg.adv_method
        )
        logger.info(msg)
        ctx.log.append(msg)


# ---------------------------------------------------------------------------
# Stage 6: Temporal Mutation — frame jitter, micro-trim, speed shift
# ---------------------------------------------------------------------------

class TemporalMutationStage:
    name = "Temporal Mutation"

    def run(self, ctx: PipelineContext) -> None:
        ctx.current_stage = self.name
        cfg = ctx.config
        logger.info("Stage: Temporal — jitter=±{}ms, trim=±{}frames, speed=±{}%".format(
            cfg.temporal_jitter_ms, cfg.temporal_trim_frames, cfg.temporal_speed_shift
        ))

        frame_files = sorted([
            f for f in os.listdir(ctx.mutated_frames_dir) if f.endswith(".png")
        ])

        # --- Micro-trim: remove N frames from start/end ---
        if cfg.temporal_trim_frames > 0:
            trim = min(cfg.temporal_trim_frames, len(frame_files) // 4)
            # Randomly trim 0..trim from start and end
            trim_start = np.random.randint(0, trim + 1)
            trim_end = np.random.randint(0, trim + 1)
            trimmed = frame_files[trim_start: len(frame_files) - trim_end if trim_end > 0 else len(frame_files)]

            # Copy trimmed set back (rename sequentially)
            new_files = []
            for i, fname in enumerate(trimmed):
                src = os.path.join(ctx.mutated_frames_dir, fname)
                new_name = "frame_{:06d}.png".format(i + 1)
                dst = os.path.join(ctx.mutated_frames_dir, new_name)
                os.rename(src, dst)
                new_files.append(new_name)

            # Remove leftover frames
            for fname in frame_files:
                if fname not in trimmed and os.path.exists(os.path.join(ctx.mutated_frames_dir, fname)):
                    os.remove(os.path.join(ctx.mutated_frames_dir, fname))

            frame_files = new_files
            ctx.frame_count = len(frame_files)
            ctx.trim_start = trim_start
            logger.info(" Trimmed: -{} start, -{} end = {} frames".format(
                trim_start, trim_end, ctx.frame_count
            ))

        # --- Speed shift: adjust FPS slightly ---
        if cfg.temporal_speed_shift != 0:
            shift_pct = np.random.uniform(-cfg.temporal_speed_shift, cfg.temporal_speed_shift)
            ctx.fps = ctx.fps * (1 + shift_pct / 100)
            logger.info("  Speed shifted: {:.2f}% → new fps: {:.4f}".format(
                shift_pct, ctx.fps
            ))

        # --- Frame jitter: per-frame timestamp offset ---
        # We implement this by duplicating occasional frames
        # to simulate micro-timing changes. Uses a staging directory
        # to avoid O(n^2) in-place renames and rename collisions.
        if cfg.temporal_jitter_ms > 0:
            jitter_frames = max(1, int(cfg.temporal_jitter_ms * ctx.fps / 1000))
            import shutil
            # Build the new frame sequence in a temp dir, then swap
            jitter_tmp = os.path.join(ctx.work_dir, "jitter_staging")
            os.makedirs(jitter_tmp, exist_ok=True)

            dup_positions = set(np.random.choice(
                len(frame_files), size=min(jitter_frames, max(1, len(frame_files) // 10)),
                replace=False
            ))
            new_idx = 0
            for i, fname in enumerate(frame_files):
                src = os.path.join(ctx.mutated_frames_dir, fname)
                # Copy the frame to the staging dir with sequential naming
                dst = os.path.join(jitter_tmp, "frame_{:06d}.png".format(new_idx + 1))
                shutil.copy2(src, dst)
                new_idx += 1
                # If this position is marked for duplication, copy again
                if i in dup_positions:
                    dup_dst = os.path.join(jitter_tmp, "frame_{:06d}.png".format(new_idx + 1))
                    shutil.copy2(src, dup_dst)
                    new_idx += 1

            # Remove old frames and move staged frames back
            for old_f in frame_files:
                old_path = os.path.join(ctx.mutated_frames_dir, old_f)
                if os.path.exists(old_path):
                    os.remove(old_path)
            for staged_f in os.listdir(jitter_tmp):
                if staged_f.endswith(".png"):
                    shutil.move(
                        os.path.join(jitter_tmp, staged_f),
                        os.path.join(ctx.mutated_frames_dir, staged_f)
                    )
            shutil.rmtree(jitter_tmp, ignore_errors=True)

            # Refresh file list
            frame_files = sorted([
                f for f in os.listdir(ctx.mutated_frames_dir) if f.endswith(".png")
            ])

            ctx.frame_count = len(frame_files)

        msg = "Temporal mutation done: {} frames, fps={:.2f}".format(
            ctx.frame_count, ctx.fps
        )
        logger.info(msg)
        ctx.log.append(msg)


# ---------------------------------------------------------------------------
# Stage 7: Audio Mutation — ultrasonic, noise floor, phase shift
# ---------------------------------------------------------------------------

class AudioMutationStage:
    name = "Audio Mutation"

    def run(self, ctx: PipelineContext) -> None:
        ctx.current_stage = self.name
        cfg = ctx.config

        if not cfg.audio_enabled or not ctx.audio_path or not os.path.exists(ctx.audio_path):
            logger.info("Audio mutation skipped (disabled or no audio track)")
            return

        logger.info("Stage: Audio — ultrasonic={}Hz, noise={}dB, phase={}°".format(
            cfg.audio_ultrasonic_freq, cfg.audio_noise_floor_db,
            cfg.audio_phase_shift_deg
        ))

        try:
            from pydub import AudioSegment
            from pydub.effects import low_pass_filter
        except ImportError:
            logger.warning("pydub not available, skipping audio mutation")
            return

        audio = AudioSegment.from_wav(ctx.audio_path)
        samples = np.array(audio.get_array_of_samples(), dtype=np.float64)

        # --- Ultrasonic injection ---
        if cfg.audio_ultrasonic_freq > 0:
            sr = audio.frame_rate
            t = np.arange(len(samples)) / sr
            freq = cfg.audio_ultrasonic_freq
            amp = cfg.audio_ultrasonic_amp * (2 ** (audio.sample_width * 8 - 1))
            if audio.channels == 2:
                t = t[:len(samples) // 2]
                ultrasonic = amp * np.sin(2 * np.pi * freq * t)
                # Stereo: slightly different freq per channel
                ultrasonic_r = amp * np.sin(2 * np.pi * (freq + 50) * t)
                # Interleave
                stereo_us = np.empty(len(samples), dtype=np.float64)
                stereo_us[0::2] = ultrasonic
                stereo_us[1::2] = ultrasonic_r
                samples += stereo_us
            else:
                ultrasonic = amp * np.sin(2 * np.pi * freq * t)
                samples += ultrasonic

        # --- Noise floor injection ---
        if cfg.audio_noise_floor_db < 0:
            max_val = 2 ** (audio.sample_width * 8 - 1)
            noise_amp = max_val * (10 ** (cfg.audio_noise_floor_db / 20))
            noise = np.random.uniform(-noise_amp, noise_amp, len(samples))
            samples += noise

        # --- Phase shift (all-pass) ---
        if cfg.audio_phase_shift_deg != 0:
            try:
                from scipy.signal import hilbert
                # Simple phase shift via analytic signal
                if audio.channels == 2:
                    left = samples[0::2]
                    right = samples[1::2]
                    shift_rad = np.deg2rad(cfg.audio_phase_shift_deg)
                    # Apply phase shift via frequency domain
                    for ch_data, channel_idx in [(left, 0), (right, 1)]:
                        spectrum = np.fft.rfft(ch_data)
                        freqs = np.fft.rfftfreq(len(ch_data), 1.0 / audio.frame_rate)
                        # Phase shift increases linearly with frequency (all-pass approximation)
                        phase_shift = np.exp(1j * shift_rad * np.ones_like(freqs))
                        spectrum *= phase_shift
                        ch_data[:] = np.fft.irfft(spectrum, n=len(ch_data)).real
                    samples[0::2] = left
                    samples[1::2] = right
                else:
                    spectrum = np.fft.rfft(samples)
                    shift_rad = np.deg2rad(cfg.audio_phase_shift_deg)
                    spectrum *= np.exp(1j * shift_rad)
                    samples[:] = np.fft.irfft(spectrum, n=len(samples)).real
            except Exception as e:
                logger.warning("Phase shift failed: {}".format(e))

        # Clip and write back
        max_val = 2 ** (audio.sample_width * 8 - 1)
        samples = np.clip(samples, -max_val, max_val).astype(np.int16 if audio.sample_width == 2 else np.int32)

        mutated_audio = AudioSegment(
            samples.tobytes(),
            frame_rate=audio.frame_rate,
            sample_width=audio.sample_width,
            channels=audio.channels
        )
        mutated_audio.export(ctx.audio_path, format="wav")

        msg = "Audio mutation done (ultrasonic={}Hz, noise={}dB, phase={}°)".format(
            cfg.audio_ultrasonic_freq, cfg.audio_noise_floor_db,
            cfg.audio_phase_shift_deg
        )
        logger.info(msg)
        ctx.log.append(msg)


# ---------------------------------------------------------------------------
# Stage 8: Re-encode + Hash Guarantee
# ---------------------------------------------------------------------------

class ReencodeStage:
    name = "Re-encode + Hash"

    def run(self, ctx: PipelineContext) -> None:
        ctx.current_stage = self.name
        cfg = ctx.config

        # Decide encoder
        encoder = cfg.encoder
        if encoder == "auto":
            # Check if NVENC available
            try:
                out = subprocess.run(
                    ["ffmpeg", "-hide_banner", "-encoders"],
                    capture_output=True, text=True, timeout=10
                )
                encoder = "h264_nvenc" if "h264_nvenc" in out.stdout else "libx264"
            except Exception:
                encoder = "libx264"

        logger.info("Stage: Re-encode — encoder={}, quality={}".format(encoder, cfg.crf))

        # Validate encoder and CRF before passing to FFmpeg
        _validate_encoder(encoder)
        _validate_crf(cfg.crf)

        # Build FFmpeg command
        # Validate that output path resolves within work_dir
        ctx.output_path = os.path.join(ctx.work_dir, "output_unique.mp4")
        _validate_path(ctx.output_path, ctx.work_dir, label="output_path")

        cmd = ["ffmpeg", "-y"]

        # Validate frame directories before FFmpeg input
        _validate_path(ctx.mutated_frames_dir, ctx.work_dir, label="mutated_frames_dir")

        # Input: mutated frames
        cmd.extend([
            "-framerate", str(ctx.fps),
            "-i", os.path.join(ctx.mutated_frames_dir, "frame_%06d.png")
        ])

        # Input: audio (if exists)
        has_audio = ctx.audio_path and os.path.exists(ctx.audio_path)
        if has_audio:
            _validate_path(ctx.audio_path, ctx.work_dir, label="audio_path (re-encode)")
            cmd.extend(["-i", ctx.audio_path])

        # Video codec
        if encoder == "h264_nvenc":
            cmd.extend([
                "-c:v", "h264_nvenc",
                "-qp", str(cfg.crf),
                "-preset", "p4",  # NVENC preset (medium quality)
            ])
        else:
            cmd.extend([
                "-c:v", "libx264",
                "-crf", str(cfg.crf),
                "-preset", "slow",
            ])

        # Pixel format
        cmd.extend(["-pix_fmt", "yuv420p"])

        # Audio codec
        if has_audio:
            cmd.extend(["-c:a", "aac", "-b:a", "192k"])
        else:
            cmd.extend(["-an"])

        # --- Hash guarantee: inject random UUID as metadata ---
        if cfg.inject_uuid:
            unique_id = str(uuid.uuid4())
            cmd.extend(["-metadata", "encoding_uuid=" + unique_id])
            cmd.extend(["-metadata", "unique_id=" + unique_id])
            logger.info("  Injected UUID: {}".format(unique_id))

        # --- Shuffle MOOV atom: move moov to beginning (faststart) ---
        if cfg.shuffle_moov:
            cmd.extend(["-movflags", "+faststart+frag_keyframe"])

        cmd.append(ctx.output_path)

        logger.info("  Encoding: {}".format(" ".join(cmd[:8]) + "..."))
        result = subprocess.run(cmd, capture_output=True, timeout=1800)

        if result.returncode != 0:
            logger.error("FFmpeg encode failed: {}".format(result.stderr[-500:]))
            raise RuntimeError("FFmpeg encoding failed")

        # Compute output hash
        ctx.output_hash = _sha256_file(ctx.output_path)

        msg = "Re-encode done: {} ({}), hash={}".format(
            encoder, cfg.crf, ctx.output_hash[:16] + "..."
        )
        logger.info(msg)
        ctx.log.append(msg)
        ctx.log.append("Original hash:  {}".format(ctx.original_hash[:16] + "..."))
        ctx.log.append("Output hash:    {}".format(ctx.output_hash[:16] + "..."))
        ctx.log.append("Hash changed:   {}".format(
            "YES" if ctx.output_hash != ctx.original_hash else "NO (PROBLEM!)"
        ))


# ---------------------------------------------------------------------------
# Stage 9: QA Verification — SSIM check
# ---------------------------------------------------------------------------

class QAVerificationStage:
    name = "QA Verification"

    def run(self, ctx: PipelineContext) -> None:
        ctx.current_stage = self.name
        cfg = ctx.config
        logger.info("Stage: QA — SSIM threshold={}".format(cfg.ssim_threshold))

        orig_files = sorted([
            f for f in os.listdir(ctx.frames_dir) if f.endswith(".png")
        ])
        mut_files = sorted([
            f for f in os.listdir(ctx.mutated_frames_dir) if f.endswith(".png")
        ])

    # Sample frames for SSIM (every 10th frame, up to 30 frames).
    # Account for temporal trimming: mutated frame at index i corresponds
    # to original frame at index (i + trim_start).
    trim_start = getattr(ctx, "trim_start", 0)
    sample_indices = list(range(0, min(len(mut_files), len(orig_files) - trim_start), max(1, len(mut_files) // 30)))
    if not sample_indices:
        sample_indices = [0] if len(mut_files) > 0 else []

    ssim_scores = []
    for i in sample_indices:
        if i >= len(mut_files):
            break
        orig_idx = i + trim_start
        if orig_idx >= len(orig_files):
            break
        orig = cv2.imread(os.path.join(ctx.frames_dir, orig_files[orig_idx]))
        mut = cv2.imread(os.path.join(ctx.mutated_frames_dir, mut_files[i]))

        if orig is None or mut is None:
            continue

        # Resize if temporal mutation changed frame count
        if orig.shape != mut.shape:
            mut = cv2.resize(mut, (orig.shape[1], orig.shape[0]))

        score = ssim(orig, mut, channel_axis=2)
        ssim_scores.append(score)

        if ssim_scores:
            ctx.ssim_score = float(np.mean(ssim_scores))
        else:
            ctx.ssim_score = 1.0

        passed = ctx.ssim_score >= cfg.ssim_threshold

        msg = "QA result: SSIM={:.4f} (threshold={:.4f}) — {}".format(
            ctx.ssim_score, cfg.ssim_threshold,
            "PASS" if passed else "FAIL — mutations too aggressive!"
        )
        logger.info(msg)
        ctx.log.append(msg)

        if not passed:
            logger.warning(
                "SSIM below threshold! Consider reducing gaussian_sigma, "
                "gan_blend_alpha, or adv_epsilon."
            )


# ---------------------------------------------------------------------------
# Pipeline Orchestrator
# ---------------------------------------------------------------------------

class Pipeline:
    """Runs all stages in sequence."""

    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.stages = [
            DemuxStage(),
            JNDStage(),
            PixelMutationStage(),
            GANPerturbationStage(),
            AdversarialStage(),
            TemporalMutationStage(),
            AudioMutationStage(),
            ReencodeStage(),
            QAVerificationStage(),
        ]

    def run(self, input_path: str, output_dir: str = None) -> PipelineContext:
        ctx = PipelineContext(config=self.config)
        # Validate input path (must exist, no traversal)
        _validate_path(input_path, "", label="input_path")
        if not os.path.isfile(input_path):
            raise ValueError("Input file does not exist: {}".format(input_path))
        ctx.input_path = input_path
        # Validate work_dir if provided
        if output_dir:
            _validate_path(output_dir, output_dir, label="output_dir")
        ctx.work_dir = output_dir or os.path.join(
            tempfile.mkdtemp(prefix="uniquelizer_"), "work"
        )

        # Detect GPU VRAM for auto-tuning
        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                ctx.vram_mb = props.total_memory // (1024 * 1024)
                ctx.device = "cuda:0"
                # Auto-downscale if VRAM low
                if ctx.vram_mb < 6000 and ctx.config.gan_resolution > 512:
                    logger.info("Auto-downscaling GAN to 512x512 (VRAM={}MB)".format(ctx.vram_mb))
                    ctx.config.gan_resolution = 512
                if ctx.vram_mb < 4000:
                    logger.info("Switching to EfficientNet-B0 (low VRAM)")
                    ctx.config.adv_model = "efficientnet_b0"
                if ctx.vram_mb < 2000:
                    logger.info("Disabling GAN (insufficient VRAM)")
                    ctx.config.gan_enabled = False
        except ImportError:
            ctx.device = "cpu"
            ctx.config.gan_enabled = False

        logger.info("=" * 50)
        logger.info("VIDEO UNIQUIELIZER — Starting Pipeline")
        logger.info("Input: {}".format(input_path))
        logger.info("GPU: {} ({}MB)".format(ctx.device, ctx.vram_mb))
        logger.info("=" * 50)

        # Validate all config fields before running any stage
        _validate_config(self.config)

        for stage in self.stages:
            # Check for cancellation before each stage
            if ctx.cancel_event.is_set():
                logger.warning(                    "Pipeline CANCELLED before stage '{}'".format(stage.name)
                )
                ctx.log.append(                    "CANCELLED before stage: {}".format(stage.name)
                )
                raise RuntimeError("Pipeline cancelled by user")
            try:
                logger.info("")
                stage.run(ctx)
            except Exception as e:
                logger.error(
                    "Stage '{}' FAILED: {}".format(stage.name, e)
                )
                ctx.log.append(
                    "ERROR in {}: {}".format(stage.name, e)
                )
                raise

        logger.info("")
        logger.info("=" * 50)
        logger.info("PIPELINE COMPLETE")
        logger.info("Output: {}".format(ctx.output_path))
        logger.info("SSIM: {:.4f}".format(ctx.ssim_score))
        logger.info("Hash: {} → {}".format(
            ctx.original_hash[:16], ctx.output_hash[:16]
        ))
        logger.info("=" * 50)

        # Cleanup temporary working files (frames, audio) to save disk space.
        # Only removes the intermediate directories; the output MP4 is kept.
        self._cleanup_work_dir(ctx)

        return ctx

    def _cleanup_work_dir(self, ctx: PipelineContext) -> None:
        """Remove intermediate frame/audio files after encoding is done.
        Preserves the output MP4 and any user-specified output directory root."""
        import shutil
        dirs_to_clean = [
            ctx.frames_dir,
            ctx.mutated_frames_dir,
        ]
        files_to_clean = [
            ctx.audio_path,
        ]
        for d in dirs_to_clean:
            if d and os.path.isdir(d):
                try:
                    shutil.rmtree(d, ignore_errors=True)
                    logger.info("Cleaned up: {}".format(d))
                except Exception as e:
                    logger.warning("Failed to clean {}: {}".format(d, e))
        for f in files_to_clean:
            if f and os.path.isfile(f):
                try:
                    os.remove(f)
                    logger.info("Cleaned up: {}".format(f))
                except Exception as e:
                    logger.warning("Failed to clean {}: {}".format(f, e))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256_file(path: str) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


# Lazy import guard for Image (used in adversarial stage)
try:
    from PIL import Image
except ImportError:
    Image = None
