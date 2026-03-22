"""Utility helpers for QR crop geometry and decode preprocessing.

This module provides deterministic image-space helpers used by the QR decoder
pipeline:

1. Bounding-box normalization and clamping to valid image coordinates.
2. Margin expansion around detector boxes to preserve QR quiet zones.
3. Lightweight preprocessing variants ordered by computational cost.

All helpers are pure functions with no external state, which keeps behavior
easy to reason about and debug.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import cv2
import numpy as np


def clamp_bbox(xyxy: Sequence[float], frame_shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
    """Clamp an XYXY bounding box to valid image coordinates.

    Parameters
    ----------
    xyxy : Sequence[float]
        Bounding box in ``[x1, y1, x2, y2]`` format, typically from detector
        output. Values can be float or int and may extend outside image bounds.
    frame_shape : Tuple[int, int, int]
        Full input frame shape as returned by NumPy/OpenCV
        ``(height, width, channels)``.

    Returns
    -------
    Tuple[int, int, int, int]
        Pixel-aligned and clipped bounding box ``(x1, y1, x2, y2)`` where
        ``x2 > x1`` and ``y2 > y1`` is always enforced.

    Notes
    -----
    The function guarantees at least a 1-pixel-wide and 1-pixel-high region so
    downstream slicing operations never produce invalid geometry due solely to
    inverted or zero-area inputs.
    """
    frame_h, frame_w = frame_shape[:2]
    x1, y1, x2, y2 = xyxy

    # Round first, then clamp so detector float boxes map cleanly to pixel grid.
    x1_i = max(0, min(frame_w - 1, int(round(x1))))
    y1_i = max(0, min(frame_h - 1, int(round(y1))))
    x2_i = max(0, min(frame_w, int(round(x2))))
    y2_i = max(0, min(frame_h, int(round(y2))))

    # Enforce non-empty ROI to prevent empty crops in later slicing.
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
    """Expand a box by a width/height ratio while preserving image bounds.

    Parameters
    ----------
    xyxy : Sequence[float]
        Original bounding box in ``[x1, y1, x2, y2]`` format.
    frame_shape : Tuple[int, int, int]
        Frame shape as ``(height, width, channels)``.
    margin_ratio : float, optional
        Fraction of current box width/height to add on each side.
        For example, ``0.25`` adds 25% margin left/right and top/bottom.

    Returns
    -------
    Tuple[int, int, int, int]
        Expanded, clamped XYXY box in integer pixel coordinates.

    Notes
    -----
    Margin expansion helps preserve the quiet zone around a QR code,
    which often improves decoder reliability when detector boxes are tight.
    """
    x1, y1, x2, y2 = clamp_bbox(xyxy, frame_shape)
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)

    margin_x = int(round(box_w * margin_ratio))
    margin_y = int(round(box_h * margin_ratio))

    expanded = (x1 - margin_x, y1 - margin_y, x2 + margin_x, y2 + margin_y)
    return clamp_bbox(expanded, frame_shape)


def crop_from_bbox(frame: np.ndarray, xyxy: Sequence[int]) -> np.ndarray:
    """Extract a rectangular crop from an image using XYXY coordinates.

    Parameters
    ----------
    frame : np.ndarray
        Source frame in OpenCV BGR layout.
    xyxy : Sequence[int]
        Bounding box in integer pixel coordinates ``(x1, y1, x2, y2)``.

    Returns
    -------
    np.ndarray
        Cropped region view/copy depending on NumPy slicing semantics.
    """
    x1, y1, x2, y2 = xyxy
    return frame[y1:y2, x1:x2]


def build_preprocessing_variants(crop_bgr: np.ndarray, max_variants: int = 5) -> List[Tuple[str, np.ndarray]]:
    """Generate preprocessing variants ordered from cheap to expensive.

    Parameters
    ----------
    crop_bgr : np.ndarray
        Input crop in BGR color order.
    max_variants : int, optional
        Maximum number of generated variants to return. The predefined order is
        preserved and truncated to this value.

    Returns
    -------
    List[Tuple[str, np.ndarray]]
        Ordered ``(variant_name, image)`` pairs. Variant names are stable,
        making debug logs and telemetry easier to interpret.

    Notes
    -----
    Variant order is intentional:

    1. Grayscale conversion as low-cost baseline.
    2. 2x upscale for small or distant QR codes.
    3. CLAHE for local contrast enhancement.
    4. Adaptive threshold for challenging lighting.
    5. Mild denoising for noisy crops.
    """
    if crop_bgr.size == 0:
        return []

    variants: List[Tuple[str, np.ndarray]] = []

    # 1) Baseline grayscale variant.
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    variants.append(("gray", gray))

    # 2) Upscale can improve module visibility for tiny QR patterns.
    upscaled = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    variants.append(("gray_upscaled_2x", upscaled))

    # 3) CLAHE boosts local contrast in uneven illumination.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    variants.append(("clahe", clahe))

    # 4) Adaptive threshold helps isolate modules under variable brightness.
    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        3,
    )
    variants.append(("adaptive_threshold", adaptive))

    # 5) Light denoise reduces high-frequency sensor noise before decode.
    denoised = cv2.fastNlMeansDenoising(gray, None, h=8, templateWindowSize=7, searchWindowSize=21)
    variants.append(("denoise", denoised))

    return variants[: max(0, max_variants)]
