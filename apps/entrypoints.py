"""CLI dispatcher for runnable application modes."""

from __future__ import annotations

import argparse
from typing import Callable, List, Optional

from apps import decode_live, image_debug, webcam_live
from core.config import LiveDecodeDefaults, LOG_LEVEL_CHOICES, parse_log_level


def _add_shared_camera_args(parser: argparse.ArgumentParser, defaults: LiveDecodeDefaults) -> None:
    """Add common camera options used by webcam and live modes."""
    parser.add_argument("--camera-index", type=int, default=defaults.camera_index, help="OpenCV camera index.")
    parser.add_argument("--width", type=int, default=defaults.width, help="Capture width.")
    parser.add_argument("--height", type=int, default=defaults.height, help="Capture height.")


def _add_shared_decode_args(parser: argparse.ArgumentParser, defaults: LiveDecodeDefaults) -> None:
    """Add common detector/decoder runtime options used by live and image modes."""
    parser.add_argument("--model", default=defaults.model_path, help="Path to YOLO model file.")
    parser.add_argument(
        "--log-level",
        choices=list(LOG_LEVEL_CHOICES),
        default="INFO",
        help="Log level for qr_decoder components.",
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


def _run_webcam(args: argparse.Namespace) -> int:
    webcam_live.run(camera_index=args.camera_index, width=args.width, height=args.height)
    return 0


def _run_live(args: argparse.Namespace) -> int:
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


def _run_image(args: argparse.Namespace) -> int:
    return image_debug.run(
        input_path=args.input,
        model_path=args.model,
        log_level=args.log_level,
        min_consecutive_frames=args.min_consecutive_frames,
        cooldown_frames=args.cooldown_frames,
        full_frame_fallback=args.full_frame_fallback,
        show=args.show,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build top-level parser for application modes."""
    defaults = LiveDecodeDefaults()
    parser = argparse.ArgumentParser(description="Run QR project application modes.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    webcam_parser = subparsers.add_parser("webcam", help="Run webcam-only preview mode.")
    _add_shared_camera_args(webcam_parser, defaults)
    webcam_parser.set_defaults(handler=_run_webcam)

    live_parser = subparsers.add_parser("live", help="Run live detector + decoder mode.")
    _add_shared_camera_args(live_parser, defaults)
    _add_shared_decode_args(live_parser, defaults)
    live_parser.add_argument("--fps", type=float, default=defaults.target_fps, help="Target processing FPS.")
    live_parser.set_defaults(handler=_run_live)

    image_parser = subparsers.add_parser("image", help="Run image debug mode for one file or directory.")
    _add_shared_decode_args(image_parser, defaults)
    image_parser.add_argument(
        "--input",
        required=True,
        help="Image file or directory containing images (.jpg, .jpeg, .png, .bmp, .webp).",
    )
    image_parser.add_argument("--show", action="store_true", help="Display image-debug windows.")
    image_parser.add_argument(
        "--full-frame-fallback",
        action="store_true",
        help="When no detections are found, decode the whole image as one candidate.",
    )
    image_parser.set_defaults(handler=_run_image)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Dispatch to selected application mode."""
    args = build_parser().parse_args(argv)
    handler: Callable[[argparse.Namespace], int] = args.handler
    return handler(args)
