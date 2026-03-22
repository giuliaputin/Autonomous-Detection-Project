"""QR decoding engine with zxing-cpp primary and OpenCV fallback."""

from __future__ import annotations

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


@dataclass
class DecodeAttempt:
    method: str
    variant: str
    ok: bool
    payload: str
    latency_ms: float


@dataclass
class DecodeCandidate:
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
    """Decode a QR payload from detector-provided bounding boxes."""

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
        return zxingcpp is not None

    def decode_detection(self, frame: np.ndarray, detection: Dict[str, Any]) -> DecodeCandidate:
        """Decode one detection box and return detailed attempt metadata."""
        frame_start = perf_counter()

        raw_xyxy = detection.get("xyxy", [0, 0, 1, 1])
        conf = float(detection.get("confidence", 0.0))
        expanded_xyxy = expand_bbox_with_margin(raw_xyxy, frame.shape, margin_ratio=self.margin_ratio)
        crop = crop_from_bbox(frame, expanded_xyxy)

        attempts: List[DecodeAttempt] = []

        if crop.size == 0:
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
            variants = build_preprocessing_variants(crop, max_variants=self.max_preprocess_variants)
            for variant_name, variant_img in variants:
                decoded = self._try_decode(variant_img, variant_name=variant_name, attempts=attempts)
                if decoded is not None:
                    break

        total_ms = (perf_counter() - frame_start) * 1000.0

        if decoded is None:
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

    def decode_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> List[DecodeCandidate]:
        """Decode all detections in a single frame."""
        return [self.decode_detection(frame, detection) for detection in detections]

    def _try_decode(
        self,
        image: np.ndarray,
        variant_name: str,
        attempts: List[DecodeAttempt],
    ) -> Optional[Tuple[str, str, str, str]]:
        order = ["zxing", "opencv"] if self.prefer_zxing else ["opencv", "zxing"]

        for method_name in order:
            if method_name == "zxing":
                decoded = self._decode_with_zxing(image, variant_name=variant_name, attempts=attempts)
            else:
                decoded = self._decode_with_opencv(image, variant_name=variant_name, attempts=attempts)

            if decoded is not None:
                return decoded

        return None

    def _decode_with_zxing(
        self,
        image: np.ndarray,
        variant_name: str,
        attempts: List[DecodeAttempt],
    ) -> Optional[Tuple[str, str, str, str]]:
        if zxingcpp is None:
            return None

        started = perf_counter()
        payload = ""
        try:
            results = zxingcpp.read_barcodes(image)
            if not results:
                latency_ms = (perf_counter() - started) * 1000.0
                attempts.append(DecodeAttempt("zxing", variant_name, False, "", latency_ms))
                return None

            first = results[0]
            payload = str(getattr(first, "text", "") or "").strip()
            fmt = str(getattr(first, "format", "QR_CODE"))
            latency_ms = (perf_counter() - started) * 1000.0

            ok = bool(payload)
            attempts.append(DecodeAttempt("zxing", variant_name, ok, payload, latency_ms))
            if not ok:
                return None

            return payload, fmt, "zxing", variant_name
        except Exception:
            latency_ms = (perf_counter() - started) * 1000.0
            attempts.append(DecodeAttempt("zxing", variant_name, False, payload, latency_ms))
            return None

    def _decode_with_opencv(
        self,
        image: np.ndarray,
        variant_name: str,
        attempts: List[DecodeAttempt],
    ) -> Optional[Tuple[str, str, str, str]]:
        started = perf_counter()
        payload = ""
        try:
            payload, _points, _straight = self._opencv_detector.detectAndDecode(image)
            payload = (payload or "").strip()
            latency_ms = (perf_counter() - started) * 1000.0
            ok = bool(payload)
            attempts.append(DecodeAttempt("opencv", variant_name, ok, payload, latency_ms))
            if not ok:
                return None
            return payload, "QR_CODE", "opencv", variant_name
        except Exception:
            latency_ms = (perf_counter() - started) * 1000.0
            attempts.append(DecodeAttempt("opencv", variant_name, False, payload, latency_ms))
            return None
