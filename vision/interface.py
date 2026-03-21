"""Simple interface helpers for webcam access in the vision module."""

from __future__ import annotations

from .camera import Camera


def build_camera(camera_index: int = 0, width: int = 640, height: int = 480) -> Camera:
    """Return a configured Camera instance for webcam operations."""
    return Camera(camera_index=camera_index, width=width, height=height)