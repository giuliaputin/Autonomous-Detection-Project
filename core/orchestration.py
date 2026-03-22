"""Shared detector-to-decoder orchestration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from qr_detection.detector import QRDetector
from qr_decoder.src import build_decoder_interface


@dataclass
class DecodeOrchestrator:
    """Compose QR detection and QR decode pipeline for frame processing."""

    detector: QRDetector
    decoder_interface: Any

    @classmethod
    def build(
        cls,
        model_path: str,
        min_consecutive_frames: int = 2,
        cooldown_frames: int = 25,
    ) -> "DecodeOrchestrator":
        """Build a default detector + decoder pair."""
        detector = QRDetector(model_path=model_path)
        decoder_interface = build_decoder_interface(
            min_consecutive_frames=min_consecutive_frames,
            cooldown_frames=cooldown_frames,
        )
        return cls(detector=detector, decoder_interface=decoder_interface)

    def process_frame(self, frame: np.ndarray, frame_index: int, return_annotated: bool = True) -> Dict[str, Any]:
        """Run detector inference and decode pipeline for one frame."""
        yolo_result = self.detector.predict_frame(frame)
        detections: List[Dict[str, Any]] = self.detector.extract_detections(yolo_result)
        return self.decoder_interface.process_frame(
            frame=frame,
            frame_index=frame_index,
            detections=detections,
            return_annotated=return_annotated,
        )
