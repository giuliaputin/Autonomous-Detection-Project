"""CLI dispatcher for runnable application modes."""

from __future__ import annotations

import argparse
from typing import List, Optional

from apps import decode_live, image_debug, webcam_live
from core.config import LiveDecodeDefaults, LOG_LEVEL_CHOICES, parse_log_level


def build_parser() -> argparse.ArgumentParser:
    """Build top-level parser for application modes."""
    defaults = LiveDecodeDefaults()
    parser = argparse.ArgumentParser(description="Run webcam feed or QR live decode debug.")
    parser.add_argument("--decode-live", action="store_true", help="Enable live detector + decoder debug mode.")
    parser.add_argument("--image-debug", action="store_true", help="Run image debug mode from this entrypoint.")
    parser.add_argument("--input", help="Input file/folder for image debug mode.")
    parser.add_argument("--camera-index", type=int, default=defaults.camera_index, help="OpenCV camera index.")
    parser.add_argument("--width", type=int, default=defaults.width, help="Capture width.")
    parser.add_argument("--height", type=int, default=defaults.height, help="Capture height.")
    parser.add_argument("--model", default=defaults.model_path, help="Path to YOLO model file.")
    parser.add_argument("--fps", type=float, default=defaults.target_fps, help="Target FPS for decode-live mode.")
    parser.add_argument(
        "--log-level",
        choices=list(LOG_LEVEL_CHOICES),
        default="INFO",
        help="Log level for qr_decoder components in decode mode.",
    )
    parser.add_argument(
        "--min-consecutive-frames",
        type=int,
        default=defaults.min_consecutive_frames,
        help="Temporal confirmer threshold.",
    )
    parser.add_argument(
        "--cooldown-frames",
        type=int,
        default=defaults.cooldown_frames,
        help="Temporal confirmer cooldown.",
    )
    parser.add_argument("--show", action="store_true", help="Display image-debug windows.")
    parser.add_argument(
        "--full-frame-fallback",
        action="store_true",
        help="In image-debug mode, decode whole image when detector finds nothing.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Dispatch to selected application mode."""
    args = build_parser().parse_args(argv)

    if args.image_debug:
        if not args.input:
            raise SystemExit("--input is required when using --image-debug")
        return image_debug.main(
            [
                "--input",
                args.input,
                "--model",
                args.model,
                "--log-level",
                args.log_level,
                "--min-consecutive-frames",
                str(args.min_consecutive_frames),
                "--cooldown-frames",
                str(args.cooldown_frames),
            ]
            + (["--show"] if args.show else [])
            + (["--full-frame-fallback"] if args.full_frame_fallback else [])
        )

    if args.decode_live:
        decode_live.run(
            camera_index=args.camera_index,
            width=args.width,
            height=args.height,
            model_path=args.model,
            log_level=parse_log_level(args.log_level),
            target_fps=args.fps,
            min_consecutive_frames=args.min_consecutive_frames,
            cooldown_frames=args.cooldown_frames,
        )
        return 0

    webcam_live.run(camera_index=args.camera_index, width=args.width, height=args.height)
    return 0
