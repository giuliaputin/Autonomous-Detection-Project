"""Public integration interface for the QR decode pipeline.

This module is the preferred entrypoint for external callers. It exposes a
stable interface that wraps decoder and temporal confirmation internals while
keeping extension points available through constructor parameters.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np

from .decoder import QRDecoder
from .pipeline import QRDecodePipeline, TemporalQRConfirmer


logger = logging.getLogger(__name__)


def configure_qr_decoder_logging(level: int = logging.INFO) -> None:
    """Configure package-scoped logging for QR decoder components.

    Parameters
    ----------
    level : int, optional
        Logging level (for example ``logging.DEBUG`` or ``logging.INFO``).

    Notes
    -----
    This function configures the ``qr_decoder`` logger namespace with a single
    stream handler when none is present. It avoids duplicate handlers when
    called multiple times.
    """
    root_logger = logging.getLogger("qr_decoder")
    root_logger.setLevel(level)

    if not root_logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)

    logger.debug("Configured qr_decoder logging at level=%s.", level)


class QRDecoderInterface:
    """Stable QR decode entrypoint used by external modules.

    Parameters
    ----------
    min_consecutive_frames : int, optional
        Number of consecutive-frame matches required before acceptance.
    cooldown_frames : int, optional
        Minimum frame gap before re-emitting the same payload.
    """

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
        """Process one frame and return decode/acceptance outputs.

        Parameters
        ----------
        frame : np.ndarray
            Current frame in BGR format.
        frame_index : int
            Frame index used for temporal confirmer state.
        detections : List[Dict[str, Any]]
            Detector outputs adhering to ``{class_name, confidence, xyxy}``.
        return_annotated : bool, optional
            If True, include an annotated frame in output dictionary.

        Returns
        -------
        Dict[str, Any]
            Pipeline output containing candidates, accepted payloads, and
            optional visualization frame.
        """
        logger.debug(
            "Interface process_frame called: frame_index=%d detections=%d annotated=%s",
            frame_index,
            len(detections),
            return_annotated,
        )
        return self.pipeline.process_frame(
            frame=frame,
            frame_index=frame_index,
            detections=detections,
            return_annotated=return_annotated,
        )


def build_decoder_interface(min_consecutive_frames: int = 2, cooldown_frames: int = 25) -> QRDecoderInterface:
    """Construct the default QR decoder interface.

    Parameters
    ----------
    min_consecutive_frames : int, optional
        Consecutive frame threshold required by temporal confirmer.
    cooldown_frames : int, optional
        Post-acceptance suppression window for duplicate payload events.

    Returns
    -------
    QRDecoderInterface
        Ready-to-use interface instance.
    """
    return QRDecoderInterface(
        min_consecutive_frames=min_consecutive_frames,
        cooldown_frames=cooldown_frames,
    )
