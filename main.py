#!/usr/bin/env python3
"""
Video Uniquelizer — Main Entry Point
=====================================
1. Runs preflight checks (GPU, FFmpeg, deps)
2. Prints system report
3. Launches Gradio Web UI with full settings panel

Usage:
    python main.py              # launch with UI
    python main.py --preflight  # only run preflight, no UI
    python main.py --cli input.mp4   # headless CLI mode
"""

import sys
import os
import argparse
import logging

# Project root on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preflight import run_preflight, format_report
from pipeline import Pipeline, PipelineConfig

logger = logging.getLogger("uniquelizer")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)


def main():
    parser = argparse.ArgumentParser(
        description="Video Uniquelizer — imperceptible video mutation for unique file hashes"
    )
    parser.add_argument(
        "--preflight", action="store_true",
        help="Only run preflight check, then exit"
    )
    parser.add_argument(
        "--cli", type=str, default=None, metavar="INPUT_VIDEO",
        help="Run pipeline headlessly on the given video file (no UI)"
    )
    parser.add_argument(
        "--output", type=str, default=None, metavar="DIR",
        help="Output directory (CLI mode only)"
    )
    parser.add_argument(
        "--port", type=int, default=7860,
        help="Gradio server port (default: 7860)"
    )
    parser.add_argument(
        "--no-share", action="store_true",
        help="Do not create a public Gradio share link"
    )
    args = parser.parse_args()

    # ================================================================
    # STEP 1: Preflight
    # ================================================================
    print()
    print("  ╔══════════════════════════════════════════════╗")
    print("  ║         V I D E O   U N I Q U E L I Z E R    ║")
    print("  ║      Perceptually Invisible Video Mutation   ║")
    print("  ╚══════════════════════════════════════════════╝")
    print()

    result = run_preflight()
    print(format_report(result))

    if not result.passed:
        print("\n  Preflight FAILED. Fix the errors above before proceeding.\n")
        sys.exit(1)

    if args.preflight:
        sys.exit(0)

    # ================================================================
    # STEP 2a: CLI mode — run pipeline directly, no UI
    # ================================================================
    if args.cli:
        input_path = args.cli
        if not os.path.isfile(input_path):
            print("  ERROR: Input file not found: {}".format(input_path))
            sys.exit(1)

        output_dir = args.output or os.path.join(
            os.path.dirname(input_path), "uniquelizer_output"
        )
        os.makedirs(output_dir, exist_ok=True)

        # Default config — user can edit PipelineConfig here for CLI tweaks
        config = PipelineConfig()

        # Auto-tune based on GPU
        if result.gpu.vram_mb < 6000:
            config.gan_resolution = 512
        if result.gpu.vram_mb < 4000:
            config.adv_model = "efficientnet_b0"
        if result.gpu.vram_mb < 2000:
            config.gan_enabled = False
        if not result.h264_nvenc:
            config.encoder = "libx264"

        print("\n  Running pipeline in CLI mode...")
        print("  Input:  {}".format(input_path))
        print("  Output: {}".format(output_dir))
        print()

        pipe = Pipeline(config=config)
        ctx = pipe.run(input_path, output_dir=output_dir)

        print("\n  RESULTS")
        print("  ├── Output file: {}".format(ctx.output_path))
        print("  ├── SSIM:        {:.4f}".format(ctx.ssim_score))
        print("  ├── Orig hash:   {}".format(ctx.original_hash[:32]))
        print("  ├── New hash:    {}".format(ctx.output_hash[:32]))
        print("  └── Hash changed: {}".format(
            "YES" if ctx.output_hash != ctx.original_hash else "NO"
        ))
        sys.exit(0)

    # ================================================================
    # STEP 2b: UI mode — launch Gradio
    # ================================================================
    print("\n  Launching Web UI on port {}...".format(args.port))
    print("  Open the Gradio link below to configure and run the pipeline.\n")

    from ui import build_ui

    app = build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=not args.no_share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
