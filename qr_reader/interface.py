"""Accessible public interface for the standalone QR decoder pipeline."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from .decoder import QRDecoder
from .pipeline import QRDecodePipeline, TemporalQRConfirmer


class QRDecoderInterface:
    """Stable entrypoint used by other parts of the project."""

    def __init__(self, min_consecutive_frames: int = 2, cooldown_frames: int = 25) -> None:
        decoder = QRDecoder()
        confirmer = TemporalQRConfirmer(
            min_consecutive_frames=min_consecutive_frames,
            cooldown_frames=cooldown_frames,
        )
        self.pipeline = QRDecodePipeline(decoder=decoder, confirmer=confirmer)

    def process_frame(
        self,
        frame: np.ndarray,
        frame_index: int,
        detections: List[Dict[str, Any]],
        return_annotated: bool = True,
    ) -> Dict[str, Any]:
        """Run QR decode over detections and return accepted/provisional results."""
        return self.pipeline.process_frame(
            frame=frame,
            frame_index=frame_index,
            detections=detections,
            return_annotated=return_annotated,
        )


def build_decoder_interface(min_consecutive_frames: int = 2, cooldown_frames: int = 25) -> QRDecoderInterface:
    """Factory function for creating a standalone decoder interface."""
    return QRDecoderInterface(
        min_consecutive_frames=min_consecutive_frames,
        cooldown_frames=cooldown_frames,
    )
