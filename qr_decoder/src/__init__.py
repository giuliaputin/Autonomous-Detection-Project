"""QR reader package public API.

This package contains a detector-agnostic QR decoding pipeline that receives
already computed detections and attempts to decode payloads from their bounding
boxes. The package intentionally separates low-level image preprocessing,
decode retries, temporal confirmation, and app-facing integration.

Public exports are grouped so callers can either:

1. Use the high-level interface/factory for normal application usage.
2. Use lower-level classes directly for custom orchestration or tuning.
"""

from .decoder import DecodeAttempt, DecodeCandidate, QRDecoder
from .interface import QRDecoderInterface, build_decoder_interface, configure_qr_decoder_logging
from .pipeline import AcceptedQR, QRDecodePipeline, TemporalQRConfirmer

__all__ = [
    "DecodeAttempt",
    "DecodeCandidate",
    "QRDecoder",
    "QRDecodePipeline",
    "TemporalQRConfirmer",
    "AcceptedQR",
    "QRDecoderInterface",
    "build_decoder_interface",
    "configure_qr_decoder_logging",
]
