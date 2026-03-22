"""Manual QR image debug runner.

Use this script to manually validate detector + decoder behavior on uploaded
images (single file or directory). It prints detailed QR reader logs and a
human-readable per-image summary so you can inspect what happened.

Example usage
-------------
python scripts/manual_qr_image_debug.py --input data/images/test --log-level DEBUG --show
python scripts/manual_qr_image_debug.py --input some_image.jpg --min-consecutive-frames 1
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Iterable, List

import cv2

# Ensure imports resolve when running this script via
# `python scripts/manual_qr_image_debug.py` from project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qr_detection.detector import QRDetector
from qr_decoder.src import build_decoder_interface, configure_qr_decoder_logging


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for manual image debugging.

    Returns
    -------
    argparse.Namespace
        Parsed arguments for model, input path, decode policy, and display.
    """
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
    parser.add_argument("--show", action="store_true", help="Display annotated result windows.")
    return parser.parse_args()


def collect_images(input_path: Path) -> List[Path]:
    """Resolve image files from a single file path or a directory.

    Parameters
    ----------
    input_path : Path
        Source path provided by the user.

    Returns
    -------
    List[Path]
        Sorted list of image files to process.

    Raises
    ------
    FileNotFoundError
        If the input path does not exist.
    ValueError
        If no supported images are found.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if input_path.is_file():
        return [input_path]

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted([p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in image_exts])

    if not images:
        raise ValueError(f"No supported images found in directory: {input_path}")

    return images


def print_candidate_summary(frame_index: int, image_path: Path, outcome: dict) -> None:
    """Print concise decode and acceptance details for one image/frame.

    Parameters
    ----------
    frame_index : int
        Current frame index used by pipeline.
    image_path : Path
        Image currently being processed.
    outcome : dict
        Result dictionary returned by QRDecoderInterface.process_frame.
    """
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


def maybe_show(image_name: str, outcome: dict, fallback_frame) -> None:
    """Display an annotated frame and block until key press.

    Parameters
    ----------
    image_name : str
        Name used for the OpenCV window title.
    outcome : dict
        Decoder pipeline output dictionary.
    fallback_frame : np.ndarray
        Raw frame shown when annotation is unavailable.
    """
    annotated = outcome.get("annotated_frame")
    to_show = annotated if annotated is not None else fallback_frame
    cv2.imshow(f"QR Debug - {image_name}", to_show)
    cv2.waitKey(0)
    cv2.destroyWindow(f"QR Debug - {image_name}")


def run() -> None:
    """Run manual detector + decoder processing for uploaded images."""
    args = parse_args()

    log_level = getattr(logging, args.log_level)
    configure_qr_decoder_logging(level=log_level)
    logging.getLogger("ultralytics").setLevel(logging.WARNING)

    detector = QRDetector(model_path=args.model)
    decoder = build_decoder_interface(
        min_consecutive_frames=args.min_consecutive_frames,
        cooldown_frames=args.cooldown_frames,
    )

    images = collect_images(Path(args.input))
    print(f"Processing {len(images)} image(s) from: {args.input}")

    for frame_index, image_path in enumerate(images):
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"[WARN] Could not read image: {image_path}")
            continue

        yolo_result = detector.predict_frame(frame)
        detections = detector.extract_detections(yolo_result)

        outcome = decoder.process_frame(
            frame=frame,
            frame_index=frame_index,
            detections=detections,
            return_annotated=True,
        )

        print_candidate_summary(frame_index=frame_index, image_path=image_path, outcome=outcome)

        if args.show:
            maybe_show(image_path.name, outcome, frame)

    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
