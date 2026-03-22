"""QR Detection Module - Object detection for QR codes using YOLOv8."""

from .src.detector import QRDetector
from .src.image_processor import QRImageProcessor, quick_process

__all__ = [
    "QRDetector",
    "QRImageProcessor",
    "quick_process",
]
