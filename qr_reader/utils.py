"""Utilities for QR crop geometry and lightweight preprocessing variants."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import cv2
import numpy as np


def clamp_bbox(xyxy: Sequence[float], frame_shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
    """Clamp a bounding box to valid image coordinates."""
    frame_h, frame_w = frame_shape[:2]
    x1, y1, x2, y2 = xyxy

    x1_i = max(0, min(frame_w - 1, int(round(x1))))
    y1_i = max(0, min(frame_h - 1, int(round(y1))))
    x2_i = max(0, min(frame_w, int(round(x2))))
    y2_i = max(0, min(frame_h, int(round(y2))))

    if x2_i <= x1_i:
        x2_i = min(frame_w, x1_i + 1)
    if y2_i <= y1_i:
        y2_i = min(frame_h, y1_i + 1)

    return x1_i, y1_i, x2_i, y2_i


def expand_bbox_with_margin(
    xyxy: Sequence[float],
    frame_shape: Tuple[int, int, int],
    margin_ratio: float = 0.25,
) -> Tuple[int, int, int, int]:
    """Expand bbox by ratio of width/height while preserving image bounds."""
    x1, y1, x2, y2 = clamp_bbox(xyxy, frame_shape)
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)

    margin_x = int(round(box_w * margin_ratio))
    margin_y = int(round(box_h * margin_ratio))

    expanded = (x1 - margin_x, y1 - margin_y, x2 + margin_x, y2 + margin_y)
    return clamp_bbox(expanded, frame_shape)


def crop_from_bbox(frame: np.ndarray, xyxy: Sequence[int]) -> np.ndarray:
    """Return an image crop using xyxy pixel coordinates."""
    x1, y1, x2, y2 = xyxy
    return frame[y1:y2, x1:x2]


def build_preprocessing_variants(crop_bgr: np.ndarray, max_variants: int = 5) -> List[Tuple[str, np.ndarray]]:
    """Generate cheap decode variants ordered from least to most expensive."""
    if crop_bgr.size == 0:
        return []

    variants: List[Tuple[str, np.ndarray]] = []

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    variants.append(("gray", gray))

    upscaled = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    variants.append(("gray_upscaled_2x", upscaled))

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    variants.append(("clahe", clahe))

    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        3,
    )
    variants.append(("adaptive_threshold", adaptive))

    denoised = cv2.fastNlMeansDenoising(gray, None, h=8, templateWindowSize=7, searchWindowSize=21)
    variants.append(("denoise", denoised))

    return variants[: max(0, max_variants)]
