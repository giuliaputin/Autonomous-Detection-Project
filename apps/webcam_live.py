"""Webcam-only application mode."""

from __future__ import annotations

import cv2

from vision.camera import Camera


def run(camera_index: int = 0, width: int = 640, height: int = 480) -> None:
    """Display raw webcam frames without detection or decode.

    Parameters
    ----------
    camera_index : int, optional
        OpenCV camera device index.
    width : int, optional
        Requested capture width in pixels.
    height : int, optional
        Requested capture height in pixels.

    Notes
    -----
    Exit with ``Esc`` or by closing the OpenCV window via the title-bar button.
    """
    camera = Camera(camera_index=camera_index, width=width, height=height)
    window_name = "Webcam Feed"

    try:
        while True:
            frame = camera.read()
            if frame is None:
                print("Warning: frame is None, retrying...")
                continue

            cv2.imshow(window_name, frame)

            # Exit when the user closes the window using the title-bar close button.
            # Some OpenCV backends raise cv2.error if the window was already destroyed.
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break

            if (cv2.waitKey(1) & 0xFF) == 27:
                break
    except KeyboardInterrupt:
        print("Exiting camera feed.")
    finally:
        camera.release()
        cv2.destroyAllWindows()
