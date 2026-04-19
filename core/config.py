"""Centralized runtime defaults for application entrypoints."""

from __future__ import annotations

import logging
from dataclasses import dataclass


@dataclass(frozen=True)
class LiveDecodeDefaults:
    """Default runtime values for webcam and live decode modes.

    Attributes
    ----------
    camera_index : int
        Default camera device index.
    width : int
        Default capture width in pixels.
    height : int
        Default capture height in pixels.
    model_path : str
        Default detector model path.
    target_fps : float
        Default processing frame rate for live mode.
    min_consecutive_frames : int
        Default temporal acceptance threshold.
    cooldown_frames : int
        Default cooldown before re-emitting same payload.
    """

    camera_index: int = 0
    width: int = 640
    height: int = 480
    model_path: str = "yolov8n.pt"
    target_fps: float = 5.0
    min_consecutive_frames: int = 2
    cooldown_frames: int = 25


LOG_LEVEL_CHOICES = ("DEBUG", "INFO", "WARNING", "ERROR")


def parse_log_level(level_name: str) -> int:
    """Convert CLI log-level name to logging module constant.

    Parameters
    ----------
    level_name : str
        Case-insensitive log level name (for example ``DEBUG``).

    Returns
    -------
    int
        Logging level constant; defaults to ``logging.INFO`` for unknown values.
    """
    return getattr(logging, level_name.upper(), logging.INFO)
