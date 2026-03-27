"""Manual image debug application for detection + decode workflows."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import cv2

from core.orchestration import DecodeOrchestrator
from qr_decoder.src import configure_qr_decoder_logging


def build_parser() -> argparse.ArgumentParser:
    """Create CLI parser for manual image debug mode."""
    parser = argparse.ArgumentParser(description="Manual QR debug runner for uploaded images.")
    parser.add_argument(
        "--input",
        required=True,
        help="Image file or directory containing images (.jpg, .jpeg, .png, .bmp, .webp).",
    )
    parser.add_argument("--model", default="yolov8n.pt", help="Path to YOLO detection model.")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="DEBUG",
        help="qr_decoder log level.",
    )
    parser.add_argument(
        "--min-consecutive-frames",
        type=int,
        default=1,
        help="Temporal confirmer threshold. Use 1 for static image evaluation.",
    )
    parser.add_argument(
        "--cooldown-frames",
        type=int,
        default=0,
        help="Temporal confirmer cooldown. Use 0 for static image evaluation.",
    )
    parser.add_argument(
        "--full-frame-fallback",
        action="store_true",
        help="When no detections are found, try decoding the entire image as one candidate.",
    )
    parser.add_argument("--show", action="store_true", help="Display annotated result windows.")
    return parser


def collect_images(input_path: Path) -> List[Path]:
    """Resolve image files from a single file path or a directory."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if input_path.is_file():
        return [input_path]

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted([p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in image_exts])
    if not images:
        raise ValueError(f"No supported images found in directory: {input_path}")

    return images


def print_candidate_summary(frame_index: int, image_path: Path, outcome: Dict) -> None:
    """Print decode and acceptance details for one image."""
    candidates = outcome["decode_candidates"]
    accepted = outcome["accepted"]

    print("=" * 90)
    print(f"Frame {frame_index} | Image: {image_path.name}")
    print(f"Filtered detections: {len(outcome['detections'])}")
    print(f"Decode candidates: {len(candidates)}")
    print(f"Accepted payload count: {len(accepted)}")

    for idx, candidate in enumerate(candidates, start=1):
        print(
            f"  Candidate {idx}: payload={candidate.payload!r} method={candidate.source_method} "
            f"variant={candidate.source_variant} conf={candidate.detection_confidence:.3f} "
            f"total_ms={candidate.total_latency_ms:.2f}"
        )
        for attempt in candidate.attempts:
            print(
                f"    Attempt: method={attempt.method} variant={attempt.variant} "
                f"ok={attempt.ok} latency_ms={attempt.latency_ms:.2f} payload={attempt.payload!r}"
            )

    for event in accepted:
        print(
            f"  ACCEPTED payload={event.payload!r} frame={event.frame_index} "
            f"method={event.source_method} bbox={event.bbox_xyxy}"
        )


def maybe_show(image_name: str, outcome: Dict, fallback_frame) -> None:
    """Display annotated frame and close safely even if user closes the window manually."""
    window_name = f"QR Debug - {image_name}"
    annotated = outcome.get("annotated_frame")
    to_show = annotated if annotated is not None else fallback_frame
    cv2.imshow(window_name, to_show)
    cv2.waitKey(0)
    try:
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1:
            cv2.destroyWindow(window_name)
    except cv2.error:
        # Window might already be closed by user interaction.
        pass


def main(argv: Optional[List[str]] = None) -> int:
    """Run manual detector + decoder processing for uploaded images."""
    parser = build_parser()
    args = parser.parse_args(argv)

    return run(
        input_path=args.input,
        model_path=args.model,
        log_level=args.log_level,
        min_consecutive_frames=args.min_consecutive_frames,
        cooldown_frames=args.cooldown_frames,
        full_frame_fallback=args.full_frame_fallback,
        show=args.show,
    )


def run(
    input_path: str,
    model_path: str = "yolov8n.pt",
    log_level: str = "DEBUG",
    min_consecutive_frames: int = 1,
    cooldown_frames: int = 0,
    full_frame_fallback: bool = False,
    show: bool = False,
) -> int:
    """Run manual detector + decoder processing for uploaded images."""
    level_name = log_level.upper()
    if level_name not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
        raise ValueError(f"Unsupported log level: {log_level}")

    configure_qr_decoder_logging(level=getattr(logging, level_name))
    logging.getLogger("ultralytics").setLevel(logging.WARNING)

    orchestrator = DecodeOrchestrator.build(
        model_path=model_path,
        min_consecutive_frames=min_consecutive_frames,
        cooldown_frames=cooldown_frames,
    )

    images = collect_images(Path(input_path))
    print(f"Processing {len(images)} image(s) from: {input_path}")

    for frame_index, image_path in enumerate(images):
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"[WARN] Could not read image: {image_path}")
            continue

        outcome = orchestrator.process_frame(frame=frame, frame_index=frame_index, return_annotated=True)

        if full_frame_fallback and not outcome["detections"]:
            full_frame_detection = {
                "class_name": "qr",
                "confidence": 1.0,
                "xyxy": [0, 0, frame.shape[1], frame.shape[0]],
            }
            outcome = orchestrator.decoder_interface.process_frame(
                frame=frame,
                frame_index=frame_index,
                detections=[full_frame_detection],
                return_annotated=True,
            )

        print_candidate_summary(frame_index=frame_index, image_path=image_path, outcome=outcome)

        if show:
            maybe_show(image_path.name, outcome, frame)

    if show:
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
