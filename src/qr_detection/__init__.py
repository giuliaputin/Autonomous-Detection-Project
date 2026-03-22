"""
QR Detection Package

Main interfaces:
- cli: Command-line interface for model management and batch inference
- image_processor: Simple image processing and detection interface
- detector: Core YOLOv8 detector class
"""

from .src.detector import QRDetector
from .src.image_processor import QRImageProcessor, quick_process

__all__ = ["QRDetector", "QRImageProcessor", "quick_process"]
