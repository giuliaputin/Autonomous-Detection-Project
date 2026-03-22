"""Centralized runtime defaults for application entrypoints."""

from __future__ import annotations

from dataclasses import dataclass
import logging


@dataclass(frozen=True)
class LiveDecodeDefaults:
    """Default runtime values for live decode mode."""

    camera_index: int = 0
    width: int = 640
    height: int = 480
    model_path: str = "yolov8n.pt"
    target_fps: float = 5.0
    min_consecutive_frames: int = 2
    cooldown_frames: int = 25


LOG_LEVEL_CHOICES = ("DEBUG", "INFO", "WARNING", "ERROR")


def parse_log_level(level_name: str) -> int:
    """Convert CLI log-level name to logging module constant."""
    return getattr(logging, level_name.upper(), logging.INFO)
