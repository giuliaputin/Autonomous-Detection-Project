"""
camera.py

Module responsible for capturing frames from a webcam.

This module provides a Camera class that:
- Initializes the webcam
- Captures frames
- Releases the camera properly

"""

import cv2


class Camera:
    def __init__(self, camera_index=0, width=640, height=480):
        """
        Initialize the webcam.

        Parameters
        ----------
        camera_index : int
            Index of the camera (0 is the default webcam for my hp)
        width : int
            Desired frame width
        height : int
            Desired frame height
        """

        # common Windows backends to avoid camera being blocked by unsupported backend
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        self.cap = None
        for backend in backends:
            cap = cv2.VideoCapture(camera_index, backend)
            if cap.isOpened():
                self.cap = cap
                break
            else:
                cap.release()

        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError(
                "Error: Could not open webcam. Ensure no other app is using it and your camera device is available."
            )

        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.width = width
        self.height = height

    def read(self):
        """
        Capture a single frame from the webcam.

        Returns
        -------
        frame : ndarray
            The captured image frame
        """

        ret, frame = self.cap.read()

        if not ret:
            raise RuntimeError("Error: Failed to capture frame from camera.")

        return frame

    def release(self):
        """Release the camera resource."""
        if self.cap is not None:
            self.cap.release()


