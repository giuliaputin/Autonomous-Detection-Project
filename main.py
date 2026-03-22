"""Application entrypoint for webcam display and optional QR decode debug.

Default behavior remains unchanged: webcam feed display only.
Use ``--decode-live`` to enable detector + QR reader pipeline logging in the
same loop for manual end-to-end debugging.
"""

from __future__ import annotations

import argparse
import logging
from time import perf_counter

import cv2

from vision.camera import Camera


def run_camera_feed(camera_index: int = 0, width: int = 640, height: int = 480) -> None:
    """Run the original webcam-only display loop.

    Parameters
    ----------
    camera_index : int, optional
        OpenCV camera device index.
    width : int, optional
        Capture width.
    height : int, optional
        Capture height.
    """
    camera = Camera(camera_index=camera_index, width=width, height=height)

    try:
        while True:
            frame = camera.read()
            if frame is None:
                print("Warning: frame is None, retrying...")
                continue

            cv2.imshow("Webcam Feed", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

    except KeyboardInterrupt:
        print("Exiting camera feed.")
    finally:
        camera.release()
        cv2.destroyAllWindows()


def run_live_decode_debug(
    camera_index: int = 0,
    width: int = 640,
    height: int = 480,
    model_path: str = "yolov8n.pt",
    log_level: int = logging.INFO,
    target_fps: float = 5.0,
) -> None:
    """Run live detector + QR decoder debugging on webcam frames.

    Parameters
    ----------
    camera_index : int, optional
        OpenCV camera device index.
    width : int, optional
        Capture width.
    height : int, optional
        Capture height.
    model_path : str, optional
        Path to YOLO detection model used by QR detector.
    log_level : int, optional
        Logging level for qr_decoder instrumentation.
    target_fps : float, optional
        Target render/process rate for the live debug loop.
    """
    from qr_detection.detector import QRDetector
    from qr_decoder.src import build_decoder_interface, configure_qr_decoder_logging

    configure_qr_decoder_logging(level=log_level)
    logging.getLogger("ultralytics").setLevel(logging.WARNING)

    detector = QRDetector(model_path=model_path)
    decoder = build_decoder_interface(min_consecutive_frames=2, cooldown_frames=25)
    camera = Camera(camera_index=camera_index, width=width, height=height)
    frame_budget_s = 1.0 / max(target_fps, 0.1)

    frame_index = 0
    try:
        while True:
            frame_started = perf_counter()
            frame = camera.read()
            if frame is None:
                print("Warning: frame is None, retrying...")
                continue

            yolo_result = detector.predict_frame(frame)
            detections = detector.extract_detections(yolo_result)
            outcome = decoder.process_frame(
                frame=frame,
                frame_index=frame_index,
                detections=detections,
                return_annotated=True,
            )

            for accepted in outcome["accepted"]:
                print(
                    f"ACCEPTED payload={accepted.payload} frame={accepted.frame_index} "
                    f"method={accepted.source_method} bbox={accepted.bbox_xyxy}"
                )

            frame_to_show = outcome["annotated_frame"] if outcome["annotated_frame"] is not None else frame
            cv2.imshow("Webcam Feed (QR Debug)", frame_to_show)

            # Slow the loop to make detections/decodes easier to inspect visually.
            elapsed_s = perf_counter() - frame_started
            wait_ms = max(1, int((frame_budget_s - elapsed_s) * 1000.0))
            key = cv2.waitKey(wait_ms) & 0xFF
            if key == 27:
                break

            frame_index += 1

    except KeyboardInterrupt:
        print("Exiting live decode debug.")
    finally:
        camera.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for display and optional decode debug modes."""
    parser = argparse.ArgumentParser(description="Run webcam feed or QR live decode debug.")
    parser.add_argument("--decode-live", action="store_true", help="Enable live detector + decoder debug mode.")
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument("--width", type=int, default=640, help="Capture width.")
    parser.add_argument("--height", type=int, default=480, help="Capture height.")
    parser.add_argument("--model", default="yolov8n.pt", help="Path to YOLO model file.")
    parser.add_argument("--fps", type=float, default=5.0, help="Target FPS for decode-live mode.")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Log level for qr_decoder components in decode mode.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    level = getattr(logging, args.log_level)

    if args.decode_live:
        run_live_decode_debug(
            camera_index=args.camera_index,
            width=args.width,
            height=args.height,
            model_path=args.model,
            log_level=level,
            target_fps=args.fps,
        )
    else:
        run_camera_feed(camera_index=args.camera_index, width=args.width, height=args.height)
