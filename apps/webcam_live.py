"""Webcam-only application mode."""

from __future__ import annotations

import cv2

from vision.camera import Camera


def run(camera_index: int = 0, width: int = 640, height: int = 480) -> None:
    """Run webcam preview without detection or decode."""
    camera = Camera(camera_index=camera_index, width=width, height=height)

    try:
        while True:
            frame = camera.read()
            if frame is None:
                print("Warning: frame is None, retrying...")
                continue

            cv2.imshow("Webcam Feed", frame)
            if (cv2.waitKey(1) & 0xFF) == 27:
                break
    except KeyboardInterrupt:
        print("Exiting camera feed.")
    finally:
        camera.release()
        cv2.destroyAllWindows()
