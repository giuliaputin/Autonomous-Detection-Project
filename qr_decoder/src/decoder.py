"""QR decoding engine with telemetry-friendly fallback behavior.

This module focuses on decoding payloads from detector-provided bounding boxes
in a single frame. The decoding strategy is intentionally layered:

1. Expand/crop the target ROI from detector geometry.
2. Try direct decode on the raw crop.
3. If needed, retry a bounded set of preprocessing variants.
4. Prefer zxing-cpp or OpenCV first depending on configuration.

The decoder records attempt-level telemetry to simplify debugging and to make
runtime behavior transparent during tuning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .utils import build_preprocessing_variants, crop_from_bbox, expand_bbox_with_margin

try:
    import zxingcpp  # type: ignore
except ImportError:  # pragma: no cover - environment dependent
    zxingcpp = None


logger = logging.getLogger(__name__)


@dataclass
class DecodeAttempt:
    """One decode attempt executed with a specific method and image variant.

    Attributes
    ----------
    method : str
        Decoder backend name, typically ``"zxing"`` or ``"opencv"``.
    variant : str
        Image variant label (for example ``"raw"`` or ``"clahe"``).
    ok : bool
        True when the attempt produced a non-empty payload.
    payload : str
        Payload text returned by the backend for this attempt.
    latency_ms : float
        End-to-end backend call latency in milliseconds.
    """

    method: str
    variant: str
    ok: bool
    payload: str
    latency_ms: float


@dataclass
class DecodeCandidate:
    """Aggregated decode result for one detection within one frame.

    Attributes
    ----------
    payload : Optional[str]
        Decoded QR payload if any backend/variant succeeds.
    symbology : Optional[str]
        Symbol format as reported by backend, usually ``"QR_CODE"``.
    bbox_xyxy : Tuple[int, int, int, int]
        Expanded and clamped ROI used for decoding.
    detection_confidence : float
        Original detector confidence for this region.
    accepted_in_frame : bool
        True when a payload was decoded in this frame for this candidate.
    source_method : Optional[str]
        Backend that produced the winning payload.
    source_variant : Optional[str]
        Preprocessing variant that produced the winning payload.
    total_latency_ms : float
        Total decode latency including retries.
    attempts : List[DecodeAttempt]
        Ordered history of all backend/variant attempts.
    """

    payload: Optional[str]
    symbology: Optional[str]
    bbox_xyxy: Tuple[int, int, int, int]
    detection_confidence: float
    accepted_in_frame: bool
    source_method: Optional[str]
    source_variant: Optional[str]
    total_latency_ms: float
    attempts: List[DecodeAttempt]


class QRDecoder:
    """Decode payloads from detector-provided QR candidate boxes.

    Parameters
    ----------
    margin_ratio : float, optional
        Expansion ratio applied to detector boxes before cropping.
    max_preprocess_variants : int, optional
        Maximum number of preprocessing variants to evaluate after raw decode.
    enable_preprocessing : bool, optional
        If True, use additional preprocessing variants when raw decode fails.
    prefer_zxing : bool, optional
        If True, try zxing-cpp before OpenCV for each variant.

    Notes
    -----
    The decoder does not run object detection itself; it expects caller-provided
    detection dictionaries containing at least ``xyxy`` and ``confidence``.
    """

    def __init__(
        self,
        margin_ratio: float = 0.25,
        max_preprocess_variants: int = 5,
        enable_preprocessing: bool = True,
        prefer_zxing: bool = True,
    ) -> None:
        self.margin_ratio = margin_ratio
        self.max_preprocess_variants = max_preprocess_variants
        self.enable_preprocessing = enable_preprocessing
        self.prefer_zxing = prefer_zxing
        self._opencv_detector = cv2.QRCodeDetector()

    @property
    def has_zxing(self) -> bool:
        """Whether zxing-cpp bindings are available in the runtime."""
        return zxingcpp is not None

    def decode_detection(self, frame: np.ndarray, detection: Dict[str, Any]) -> DecodeCandidate:
        """Decode one detector region and return full telemetry.

        Parameters
        ----------
        frame : np.ndarray
            Full frame in BGR format.
        detection : Dict[str, Any]
            Detection dictionary containing ``xyxy`` and ``confidence``.

        Returns
        -------
        DecodeCandidate
            Structured decode result with attempt history and timing data.
        """
        frame_start = perf_counter()

        raw_xyxy = detection.get("xyxy", [0, 0, 1, 1])
        conf = float(detection.get("confidence", 0.0))
        expanded_xyxy = expand_bbox_with_margin(
            raw_xyxy, frame.shape, margin_ratio=self.margin_ratio
        )
        crop = crop_from_bbox(frame, expanded_xyxy)

        logger.debug(
            "Decode start: conf=%.3f raw_xyxy=%s expanded_xyxy=%s crop_shape=%s",
            conf,
            raw_xyxy,
            expanded_xyxy,
            tuple(crop.shape) if crop.size else (0, 0),
        )

        attempts: List[DecodeAttempt] = []

        if crop.size == 0:
            logger.debug("Decode skipped: empty crop after bbox expansion.")
            return DecodeCandidate(
                payload=None,
                symbology=None,
                bbox_xyxy=expanded_xyxy,
                detection_confidence=conf,
                accepted_in_frame=False,
                source_method=None,
                source_variant=None,
                total_latency_ms=(perf_counter() - frame_start) * 1000.0,
                attempts=attempts,
            )

        decoded = self._try_decode(crop, variant_name="raw", attempts=attempts)

        if decoded is None and self.enable_preprocessing:
            logger.debug(
                "Raw decode failed; trying preprocessing variants (max=%d).",
                self.max_preprocess_variants,
            )
            variants = build_preprocessing_variants(crop, max_variants=self.max_preprocess_variants)
            for variant_name, variant_img in variants:
                decoded = self._try_decode(
                    variant_img, variant_name=variant_name, attempts=attempts
                )
                if decoded is not None:
                    break

        total_ms = (perf_counter() - frame_start) * 1000.0

        if decoded is None:
            logger.debug("Decode failed after %d attempts (total=%.2fms).", len(attempts), total_ms)
            return DecodeCandidate(
                payload=None,
                symbology=None,
                bbox_xyxy=expanded_xyxy,
                detection_confidence=conf,
                accepted_in_frame=False,
                source_method=None,
                source_variant=None,
                total_latency_ms=total_ms,
                attempts=attempts,
            )

        payload, symbology, source_method, source_variant = decoded
        logger.debug(
            "Decode success: payload='%s' method=%s variant=%s attempts=%d total=%.2fms",
            payload,
            source_method,
            source_variant,
            len(attempts),
            total_ms,
        )
        return DecodeCandidate(
            payload=payload,
            symbology=symbology,
            bbox_xyxy=expanded_xyxy,
            detection_confidence=conf,
            accepted_in_frame=True,
            source_method=source_method,
            source_variant=source_variant,
            total_latency_ms=total_ms,
            attempts=attempts,
        )

    def decode_detections(
        self, frame: np.ndarray, detections: List[Dict[str, Any]]
    ) -> List[DecodeCandidate]:
        """Decode all candidate detections in one frame.

        Parameters
        ----------
        frame : np.ndarray
            Full frame in BGR format.
        detections : List[Dict[str, Any]]
            Detector candidate list for the current frame.

        Returns
        -------
        List[DecodeCandidate]
            One candidate result per input detection in original order.
        """
        logger.debug("Decoding %d detections in current frame.", len(detections))
        return [self.decode_detection(frame, detection) for detection in detections]

    def _try_decode(
        self,
        image: np.ndarray,
        variant_name: str,
        attempts: List[DecodeAttempt],
    ) -> Optional[Tuple[str, str, str, str]]:
        """Try decode backends in configured order for one image variant.

        Parameters
        ----------
        image : np.ndarray
            Candidate crop variant to decode.
        variant_name : str
            Human-readable variant label for telemetry.
        attempts : List[DecodeAttempt]
            Mutable attempt list that will be appended in call order.

        Returns
        -------
        Optional[Tuple[str, str, str, str]]
            Winning decode tuple ``(payload, symbology, method, variant)`` or
            None when all methods fail.
        """
        order = ["zxing", "opencv"] if self.prefer_zxing else ["opencv", "zxing"]
        logger.debug("Trying decode methods %s for variant='%s'.", order, variant_name)

        for method_name in order:
            if method_name == "zxing":
                decoded = self._decode_with_zxing(
                    image, variant_name=variant_name, attempts=attempts
                )
            else:
                decoded = self._decode_with_opencv(
                    image, variant_name=variant_name, attempts=attempts
                )

            if decoded is not None:
                logger.debug("Variant '%s' succeeded with method '%s'.", variant_name, method_name)
                return decoded

        return None

    def _decode_with_zxing(
        self,
        image: np.ndarray,
        variant_name: str,
        attempts: List[DecodeAttempt],
    ) -> Optional[Tuple[str, str, str, str]]:
        """Decode one variant using zxing-cpp when available.

        Parameters
        ----------
        image : np.ndarray
            Input crop variant.
        variant_name : str
            Variant label recorded in telemetry.
        attempts : List[DecodeAttempt]
            Mutable telemetry list appended with this attempt result.

        Returns
        -------
        Optional[Tuple[str, str, str, str]]
            Winning decode tuple on success, otherwise None.
        """
        if zxingcpp is None:
            logger.debug("Skipping zxing decode: dependency not available.")
            return None

        started = perf_counter()
        payload = ""
        try:
            results = zxingcpp.read_barcodes(image)
            if not results:
                latency_ms = (perf_counter() - started) * 1000.0
                attempts.append(DecodeAttempt("zxing", variant_name, False, "", latency_ms))
                logger.debug("zxing miss: variant='%s' latency=%.2fms", variant_name, latency_ms)
                return None

            first = results[0]
            payload = str(getattr(first, "text", "") or "").strip()
            fmt = str(getattr(first, "format", "QR_CODE"))
            latency_ms = (perf_counter() - started) * 1000.0

            ok = bool(payload)
            attempts.append(DecodeAttempt("zxing", variant_name, ok, payload, latency_ms))
            logger.debug(
                "zxing attempt: variant='%s' ok=%s fmt=%s latency=%.2fms",
                variant_name,
                ok,
                fmt,
                latency_ms,
            )
            if not ok:
                return None

            return payload, fmt, "zxing", variant_name
        except Exception:
            latency_ms = (perf_counter() - started) * 1000.0
            attempts.append(DecodeAttempt("zxing", variant_name, False, payload, latency_ms))
            logger.exception("zxing exception on variant='%s'.", variant_name)
            return None

    def _decode_with_opencv(
        self,
        image: np.ndarray,
        variant_name: str,
        attempts: List[DecodeAttempt],
    ) -> Optional[Tuple[str, str, str, str]]:
        """Decode one variant using OpenCV QRCodeDetector fallback.

        Parameters
        ----------
        image : np.ndarray
            Input crop variant.
        variant_name : str
            Variant label recorded in telemetry.
        attempts : List[DecodeAttempt]
            Mutable telemetry list appended with this attempt result.

        Returns
        -------
        Optional[Tuple[str, str, str, str]]
            Winning decode tuple on success, otherwise None.
        """
        started = perf_counter()
        payload = ""
        try:
            payload, _points, _straight = self._opencv_detector.detectAndDecode(image)
            payload = (payload or "").strip()
            latency_ms = (perf_counter() - started) * 1000.0
            ok = bool(payload)
            attempts.append(DecodeAttempt("opencv", variant_name, ok, payload, latency_ms))
            logger.debug(
                "opencv attempt: variant='%s' ok=%s latency=%.2fms", variant_name, ok, latency_ms
            )
            if not ok:
                return None
            return payload, "QR_CODE", "opencv", variant_name
        except Exception:
            latency_ms = (perf_counter() - started) * 1000.0
            attempts.append(DecodeAttempt("opencv", variant_name, False, payload, latency_ms))
            logger.exception("opencv exception on variant='%s'.", variant_name)
            return None
