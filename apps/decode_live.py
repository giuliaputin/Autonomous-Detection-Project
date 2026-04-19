"""Live webcam mode with QR detection and decoding."""

from __future__ import annotations

import logging
from time import perf_counter

import cv2

from core.orchestration import DecodeOrchestrator
from qr_decoder.src import configure_qr_decoder_logging
from vision.camera import Camera


def run(
    camera_index: int = 0,
    width: int = 640,
    height: int = 480,
    model_path: str = "yolov8n.pt",
    log_level: int = logging.INFO,
    target_fps: float = 5.0,
    min_consecutive_frames: int = 2,
    cooldown_frames: int = 25,
) -> None:
    """Stream webcam frames through detection and temporal decode pipeline.

    Parameters
    ----------
    camera_index : int, optional
        OpenCV camera device index.
    width : int, optional
        Requested capture width in pixels.
    height : int, optional
        Requested capture height in pixels.
    model_path : str, optional
        Path to YOLO model weights used for QR detection.
    log_level : int, optional
        Logging module level constant (for example ``logging.DEBUG``).
    target_fps : float, optional
        Target frame rate. Loop timing uses this to compute a per-frame budget.
    min_consecutive_frames : int, optional
        Consecutive-frame threshold for temporal acceptance.
    cooldown_frames : int, optional
        Cooldown in frames before same payload can be emitted again.

    Notes
    -----
    Exit with ``Esc`` or by closing the OpenCV window via the title-bar button.
    """
    configure_qr_decoder_logging(level=log_level)
    logging.getLogger("ultralytics").setLevel(logging.WARNING)

    orchestrator = DecodeOrchestrator.build(
        model_path=model_path,
        min_consecutive_frames=min_consecutive_frames,
        cooldown_frames=cooldown_frames,
    )
    camera = Camera(camera_index=camera_index, width=width, height=height)
    # Frame pacing keeps processing close to target_fps without busy waiting.
    frame_budget_s = 1.0 / max(target_fps, 0.1)
    window_name = "Webcam Feed (QR Debug)"

    frame_index = 0
    try:
        while True:
            frame_started = perf_counter()
            frame = camera.read()
            if frame is None:
                print("Warning: frame is None, retrying...")
                continue

            outcome = orchestrator.process_frame(
                frame=frame, frame_index=frame_index, return_annotated=True
            )
            for accepted in outcome["accepted"]:
                print(
                    f"ACCEPTED payload={accepted.payload} frame={accepted.frame_index} "
                    f"method={accepted.source_method} bbox={accepted.bbox_xyxy}"
                )

            frame_to_show = (
                outcome["annotated_frame"] if outcome["annotated_frame"] is not None else frame
            )
            cv2.imshow(window_name, frame_to_show)

            # Exit when the user closes the window using the title-bar close button.
            # Some OpenCV backends raise cv2.error if the window was already destroyed.
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break

            elapsed_s = perf_counter() - frame_started
            wait_ms = max(1, int((frame_budget_s - elapsed_s) * 1000.0))
            if (cv2.waitKey(wait_ms) & 0xFF) == 27:
                break

            frame_index += 1
    except KeyboardInterrupt:
        print("Exiting live decode debug.")
    finally:
        camera.release()
        cv2.destroyAllWindows()
