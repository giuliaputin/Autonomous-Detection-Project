"""Webcam capture utilities.

This module provides a thin wrapper over ``cv2.VideoCapture`` with backend
fallback logic tuned for Windows camera compatibility.
"""

import cv2


class Camera:
    """OpenCV webcam capture wrapper with backend fallback behavior."""

    def __init__(self, camera_index=0, width=640, height=480):
        """Initialize webcam capture and apply requested resolution.

        Parameters
        ----------
        camera_index : int, optional
            Camera device index (``0`` is typically the default webcam).
        width : int, optional
            Requested frame width in pixels.
        height : int, optional
            Requested frame height in pixels.

        Raises
        ------
        RuntimeError
            If no backend can open the requested camera device.
        """

        # Probe common Windows backends in a stable order before falling back to CAP_ANY.
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        self.cap = None
        for backend in backends:
            cap = cv2.VideoCapture(camera_index, backend)
            if cap.isOpened():
                self.cap = cap
                break
            else:
                # Release unsuccessful handles to avoid leaking camera locks.
                cap.release()

        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError(
                "Error: Could not open webcam. Ensure no other app is using it "
                "and your camera device is available."
            )

        # Apply preferred capture resolution; backend may clamp to supported values.
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.width = width
        self.height = height

    def read(self):
        """Capture one frame from webcam.

        Returns
        -------
        ndarray
            Captured BGR frame.

        Raises
        ------
        RuntimeError
            If frame capture fails.
        """

        ret, frame = self.cap.read()

        if not ret:
            raise RuntimeError("Error: Failed to capture frame from camera.")

        return frame

    def release(self):
        """Release the underlying camera handle if it exists."""
        if self.cap is not None:
            self.cap.release()
