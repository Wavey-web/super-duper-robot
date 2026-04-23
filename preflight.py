"""
Video Uniquelizer — Preflight Checks
=====================================
Validates GPU (CUDA + VRAM), FFmpeg, and Python dependencies
before the pipeline starts. Prints a clear report and raises
on hard failures.
"""

import subprocess
import sys
import importlib
import shutil
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GPUInfo:
    name: str = "N/A"
    vram_mb: int = 0
    cuda_version: str = "N/A"
    driver_version: str = "N/A"
    nvenc_available: bool = False
    fp16_support: bool = False


@dataclass
class PreflightResult:
    passed: bool = True
    gpu: GPUInfo = field(default_factory=GPUInfo)
    ffmpeg_path: Optional[str] = None
    ffmpeg_version: Optional[str] = None
    h264_nvenc: bool = False
    missing_packages: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    errors: list = field(default_factory=list)


# Minimum requirements
MIN_VRAM_MB = 2000          # 2 GB minimum
RECOMMENDED_VRAM_MB = 6000  # 6 GB for full pipeline
REQUIRED_PACKAGES = [
    "torch", "numpy", "cv2", "scipy", "skimage",
    "gradio", "PIL", "pydub",
]
OPTIONAL_PACKAGES = [
    "librosa", "moviepy", "torchvision",
]


def _check_gpu() -> GPUInfo:
    """Query nvidia-smi and torch for GPU details."""
    info = GPUInfo()

    # Try torch first (more reliable in-notebook)
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info.name = props.name
            info.vram_mb = props.total_memory // (1024 * 1024)
            info.cuda_version = torch.version.cuda or "N/A"
            info.fp16_support = True  # All CUDA GPUs support FP16 storage
    except Exception:
        pass

    # Supplement with nvidia-smi for driver + NVENC
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if out.returncode == 0:
            # May return multi-line if multiple GPUs — take first
            first_line = out.stdout.strip().split("\n")[0]
            parts = [p.strip() for p in first_line.split(",")]
            if len(parts) >= 3:
                if info.name == "N/A":
                    info.name = parts[0]
                if info.vram_mb == 0:
                    info.vram_mb = int(float(parts[1]))
                info.driver_version = parts[2]
    except Exception:
        pass

    # NVENC check via ffmpeg
    try:
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=10
        )
        info.nvenc_available = "h264_nvenc" in out.stdout
    except Exception:
        pass

    return info


def _check_ffmpeg() -> tuple:
    """Return (path, version_string, has_nvenc)."""
    path = shutil.which("ffmpeg")
    version = None
    has_nvenc = False

    if path:
        try:
            out = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True, text=True, timeout=10
            )
            first_line = out.stdout.split("\n")[0]
            version = first_line.split("version")[-1].strip() if "version" in first_line else out.stdout[:60]
        except Exception:
            pass

        try:
            out = subprocess.run(
                ["ffmpeg", "-hide_banner", "-encoders"],
                capture_output=True, text=True, timeout=10
            )
            has_nvenc = "h264_nvenc" in out.stdout
        except Exception:
            pass

    return path, version, has_nvenc


def _check_packages(required: list, optional: list) -> tuple:
    """Return (missing_required, missing_optional)."""
    missing_req = []
    missing_opt = []

    pkg_map = {
        "cv2": "opencv-python",
        "PIL": "Pillow",
        "skimage": "scikit-image",
    }

    for pkg in required:
        install_name = pkg_map.get(pkg, pkg)
        try:
            importlib.import_module(pkg)
        except ImportError:
            missing_req.append(install_name)

    for pkg in optional:
        install_name = pkg_map.get(pkg, pkg)
        try:
            importlib.import_module(pkg)
        except ImportError:
            missing_opt.append(install_name)

    return missing_req, missing_opt


def run_preflight() -> PreflightResult:
    """
    Run all preflight checks. Returns a PreflightResult.
    Sets result.passed = False on hard failures.
    """
    r = PreflightResult()

    # ---- GPU ----
    r.gpu = _check_gpu()
    if r.gpu.name == "N/A":
        r.errors.append("No CUDA GPU detected. Pipeline requires CUDA for ML stages.")
        r.passed = False
    elif r.gpu.vram_mb < MIN_VRAM_MB:
        r.errors.append(
            f"GPU VRAM {r.gpu.vram_mb}MB is below minimum {MIN_VRAM_MB}MB. "
            f"ML stages (GAN/FGSM) will not fit."
        )
        r.passed = False
    elif r.gpu.vram_mb < RECOMMENDED_VRAM_MB:
        r.warnings.append(
            f"GPU VRAM {r.gpu.vram_mb}MB is below recommended {RECOMMENDED_VRAM_MB}MB. "
            f"Pipeline will auto-downscale GAN to 512x512 and use FGSM+EfficientNet-B0."
        )

    # ---- FFmpeg ----
    r.ffmpeg_path, r.ffmpeg_version, r.h264_nvenc = _check_ffmpeg()
    if not r.ffmpeg_path:
        r.errors.append("FFmpeg not found on PATH. Required for demux/encode.")
        r.passed = False
    else:
        if not r.h264_nvenc:
            r.warnings.append(
                "h264_nvenc encoder not available. Will fall back to libx264 (CPU encoding)."
            )

    # ---- Packages ----
    missing_req, missing_opt = _check_packages(REQUIRED_PACKAGES, OPTIONAL_PACKAGES)
    r.missing_packages = missing_req + missing_opt
    if missing_req:
        r.errors.append(
            f"Missing required packages: {', '.join(missing_req)}. "
            f"Install with: pip install {' '.join(missing_req)}"
        )
        r.passed = False
    if missing_opt:
        r.warnings.append(
            f"Missing optional packages: {', '.join(missing_opt)}. Some features disabled."
        )

    return r


def format_report(r: PreflightResult) -> str:
    """Pretty-print the preflight report for console output."""
    lines = []
    lines.append("=" * 60)
    lines.append("  VIDEO UNIQUIELIZER — PREFLIGHT CHECK")
    lines.append("=" * 60)

    # GPU
    g = r.gpu
    lines.append("")
    lines.append("  GPU")
    lines.append("  ├── Name:       {}".format(g.name))
    lines.append("  ├── VRAM:       {} MB".format(g.vram_mb))
    lines.append("  ├── CUDA:       {}".format(g.cuda_version))
    lines.append("  ├── Driver:     {}".format(g.driver_version))
    lines.append("  ├── NVENC:      {}".format("YES" if g.nvenc_available else "NO"))
    lines.append("  └── FP16:       {}".format("YES" if g.fp16_support else "NO"))

    # FFmpeg
    lines.append("")
    lines.append("  FFMPEG")
    lines.append("  ├── Path:       {}".format(r.ffmpeg_path or "NOT FOUND"))
    lines.append("  ├── Version:    {}".format(r.ffmpeg_version or "N/A"))
    lines.append("  └── h264_nvenc: {}".format("YES" if r.h264_nvenc else "NO (will use libx264)"))

    # Pipeline recommendation based on VRAM
    lines.append("")
    lines.append("  PIPELINE CONFIG")
    if g.vram_mb >= RECOMMENDED_VRAM_MB:
        lines.append("  ├── GAN:        StyleGAN2-1024 FP16 (full quality)")
        lines.append("  ├── Adversarial: FGSM + ResNet-50")
        lines.append("  └── Encode:     {}".format("h264_nvenc" if r.h264_nvenc else "libx264"))
    elif g.vram_mb >= 4000:
        lines.append("  ├── GAN:        StyleGAN2-512 FP16 (auto-downscaled)")
        lines.append("  ├── Adversarial: FGSM + EfficientNet-B0 (lightweight)")
        lines.append("  └── Encode:     {}".format("h264_nvenc" if r.h264_nvenc else "libx264"))
    elif g.vram_mb >= MIN_VRAM_MB:
        lines.append("  ├── GAN:        SKIP (not enough VRAM, use JND-noise only)")
        lines.append("  ├── Adversarial: FGSM + MobileNetV3 (ultra-light)")
        lines.append("  └── Encode:     libx264 (CPU)")
    else:
        lines.append("  └── CANNOT RUN — insufficient VRAM")

    # Warnings / Errors
    if r.warnings:
        lines.append("")
        lines.append("  WARNINGS")
        for w in r.warnings:
            lines.append("  ⚠  {}".format(w))

    if r.errors:
        lines.append("")
        lines.append("  ERRORS")
        for e in r.errors:
            lines.append("  ✗  {}".format(e))

    # Verdict
    lines.append("")
    if r.passed:
        lines.append("  ✓  PREFLIGHT PASSED — pipeline ready")
    else:
        lines.append("  ✗  PREFLIGHT FAILED — fix errors above")
    lines.append("=" * 60)

    return "\n".join(lines)


if __name__ == "__main__":
    result = run_preflight()
    print(format_report(result))
    sys.exit(0 if result.passed else 1)
