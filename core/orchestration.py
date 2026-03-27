"""Shared detector-to-decoder orchestration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from qr_decoder.src import build_decoder_interface
from qr_detection.detector import QRDetector


@dataclass
class DecodeOrchestrator:
    """Compose QR detection and QR decode pipeline for frame processing.

    Attributes
    ----------
    detector : QRDetector
        Detector instance responsible for bounding-box inference.
    decoder_interface : Any
        Decoder pipeline interface returned by ``build_decoder_interface``.
    """

    detector: QRDetector
    decoder_interface: Any

    @classmethod
    def build(
        cls,
        model_path: str,
        min_consecutive_frames: int = 2,
        cooldown_frames: int = 25,
    ) -> "DecodeOrchestrator":
        """Build a default detector + decoder pair.

        Parameters
        ----------
        model_path : str
            Path to YOLO model weights for detection.
        min_consecutive_frames : int, optional
            Temporal acceptance threshold used by decoder pipeline.
        cooldown_frames : int, optional
            Temporal cooldown used by decoder pipeline.

        Returns
        -------
        DecodeOrchestrator
            Configured orchestrator instance.
        """
        detector = QRDetector(model_path=model_path)
        decoder_interface = build_decoder_interface(
            min_consecutive_frames=min_consecutive_frames,
            cooldown_frames=cooldown_frames,
        )
        return cls(detector=detector, decoder_interface=decoder_interface)

    def process_frame(
        self, frame: np.ndarray, frame_index: int, return_annotated: bool = True
    ) -> Dict[str, Any]:
        """Run detector inference and decode pipeline for one frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR frame from camera or image loader.
        frame_index : int
            Monotonic frame index used by temporal confirmer.
        return_annotated : bool, optional
            If True, include annotated frame in returned payload.

        Returns
        -------
        Dict[str, Any]
            Decoder interface payload including detections, candidates, and
            accepted events.
        """
        yolo_result = self.detector.predict_frame(frame)
        detections: List[Dict[str, Any]] = self.detector.extract_detections(yolo_result)
        return self.decoder_interface.process_frame(
            frame=frame,
            frame_index=frame_index,
            detections=detections,
            return_annotated=return_annotated,
        )
