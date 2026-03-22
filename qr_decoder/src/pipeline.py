"""Decode orchestration and temporal confirmation logic.

This module turns per-frame detector outputs into stable QR acceptance events.
It is intentionally detector-agnostic: callers provide frame detections, and
the pipeline handles filtering, decode attempts, temporal confirmation, and
optional frame annotation.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from time import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from .decoder import DecodeCandidate, QRDecoder


logger = logging.getLogger(__name__)


@dataclass
class AcceptedQR:
    """A temporally accepted QR payload event.

    Attributes
    ----------
    payload : str
        Final payload accepted by temporal confirmation policy.
    frame_index : int
        Frame index where acceptance occurred.
    timestamp_s : float
        Timestamp associated with the frame processing moment.
    bbox_xyxy : tuple[int, int, int, int]
        Bounding box used for the accepted candidate.
    source_method : Optional[str]
        Decode backend responsible for the accepted payload.
    """
    payload: str
    frame_index: int
    timestamp_s: float
    bbox_xyxy: tuple[int, int, int, int]
    source_method: Optional[str]


class TemporalQRConfirmer:
    """Accept payloads only after short consecutive-frame agreement.

    Parameters
    ----------
    min_consecutive_frames : int, optional
        Number of consecutive frames required before first acceptance.
    cooldown_frames : int, optional
        Minimum frame gap before the same payload can be emitted again.

    Notes
    -----
    State is tracked per payload string using ``last_seen``, ``streak``, and
    ``last_emitted`` frame indices. This makes acceptance deterministic and
    easy to inspect in debug logs.
    """

    def __init__(self, min_consecutive_frames: int = 2, cooldown_frames: int = 25) -> None:
        self.min_consecutive_frames = min_consecutive_frames
        self.cooldown_frames = cooldown_frames
        self._state: Dict[str, Dict[str, int]] = {}

    def update(self, frame_index: int, candidates: List[DecodeCandidate]) -> List[str]:
        """Update temporal state with current frame candidates.

        Parameters
        ----------
        frame_index : int
            Current monotonically increasing frame index.
        candidates : List[DecodeCandidate]
            Candidate results for current frame after decode stage.

        Returns
        -------
        List[str]
            Payload strings accepted at this frame.
        """
        accepted: List[str] = []

        by_payload: Dict[str, DecodeCandidate] = {}
        for candidate in candidates:
            if not candidate.payload:
                continue
            # Keep strongest detection when same payload appears multiple times.
            best = by_payload.get(candidate.payload)
            if best is None or candidate.detection_confidence > best.detection_confidence:
                by_payload[candidate.payload] = candidate

        for payload in by_payload:
            state = self._state.get(payload, {"last_seen": -10_000, "streak": 0, "last_emitted": -10_000})

            if frame_index == state["last_seen"] + 1:
                state["streak"] += 1
            else:
                state["streak"] = 1

            state["last_seen"] = frame_index

            ready = state["streak"] >= self.min_consecutive_frames
            cooled = (frame_index - state["last_emitted"]) >= self.cooldown_frames
            if ready and cooled:
                accepted.append(payload)
                state["last_emitted"] = frame_index
                logger.info(
                    "Accepted payload '%s' at frame=%d (streak=%d, cooldown=%d).",
                    payload,
                    frame_index,
                    state["streak"],
                    self.cooldown_frames,
                )
            else:
                logger.debug(
                    "Payload '%s' state at frame=%d: streak=%d ready=%s cooled=%s.",
                    payload,
                    frame_index,
                    state["streak"],
                    ready,
                    cooled,
                )

            self._state[payload] = state

        return accepted


class QRDecodePipeline:
    """Detection-agnostic frame pipeline for QR decoding.

    Parameters
    ----------
    decoder : Optional[QRDecoder], optional
        Decoder instance used for candidate decoding.
    confirmer : Optional[TemporalQRConfirmer], optional
        Temporal confirmer instance used for acceptance decisions.
    filter_class_names : Optional[List[str]], optional
        Class names considered QR targets. Comparison is case-insensitive.

    Notes
    -----
    The pipeline relies on a minimal detection dictionary contract:
    ``{class_name, confidence, xyxy}``.
    """

    def __init__(
        self,
        decoder: Optional[QRDecoder] = None,
        confirmer: Optional[TemporalQRConfirmer] = None,
        filter_class_names: Optional[List[str]] = None,
    ) -> None:
        self.decoder = decoder or QRDecoder()
        self.confirmer = confirmer or TemporalQRConfirmer()
        self.filter_class_names = {name.lower() for name in (filter_class_names or ["qr", "qrcode"]) }

    def process_frame(
        self,
        frame: np.ndarray,
        frame_index: int,
        detections: List[Dict[str, Any]],
        timestamp_s: Optional[float] = None,
        return_annotated: bool = True,
    ) -> Dict[str, Any]:
        """Process a frame and return decode and acceptance outputs.

        Parameters
        ----------
        frame : np.ndarray
            Current frame in BGR format.
        frame_index : int
            Monotonic index for temporal tracking.
        detections : List[Dict[str, Any]]
            Detector outputs for this frame.
        timestamp_s : Optional[float], optional
            Optional timestamp override. Uses current wall-clock time if None.
        return_annotated : bool, optional
            If True, include ``annotated_frame`` in return payload.

        Returns
        -------
        Dict[str, Any]
            Structured result containing filtered detections, decode candidates,
            accepted events, and optional annotated frame.
        """
        timestamp_s = time() if timestamp_s is None else timestamp_s

        filtered = [d for d in detections if self._is_target_qr_class(d)]
        if detections and not filtered:
            seen_classes = [str(d.get("class_name", "")).strip().lower() for d in detections]
            logger.debug(
                "Frame %d filtered all detections. Allowed=%s Seen=%s",
                frame_index,
                sorted(self.filter_class_names),
                seen_classes,
            )
        candidates = self.decoder.decode_detections(frame, filtered)
        accepted_payloads = self.confirmer.update(frame_index, candidates)

        logger.info(
            "Frame %d summary: detections=%d filtered=%d decoded=%d accepted=%d",
            frame_index,
            len(detections),
            len(filtered),
            len(candidates),
            len(accepted_payloads),
        )
        if accepted_payloads:
            logger.info("Frame %d accepted payloads: %s", frame_index, accepted_payloads)

        accepted_items: List[AcceptedQR] = []
        for payload in accepted_payloads:
            for candidate in candidates:
                if candidate.payload == payload:
                    accepted_items.append(
                        AcceptedQR(
                            payload=payload,
                            frame_index=frame_index,
                            timestamp_s=timestamp_s,
                            bbox_xyxy=candidate.bbox_xyxy,
                            source_method=candidate.source_method,
                        )
                    )
                    break

        annotated = self._annotate(frame, filtered, candidates) if return_annotated else None

        return {
            "frame_index": frame_index,
            "timestamp_s": timestamp_s,
            "detections": filtered,
            "decode_candidates": candidates,
            "accepted": accepted_items,
            "annotated_frame": annotated,
        }

    def _is_target_qr_class(self, detection: Dict[str, Any]) -> bool:
        """Return True when a detection belongs to configured QR classes."""
        class_name = str(detection.get("class_name", "")).strip().lower()
        if not self.filter_class_names:
            return True
        return class_name in self.filter_class_names

    def _annotate(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        candidates: List[DecodeCandidate],
    ) -> np.ndarray:
        """Create an annotated frame for visualization and debugging.

        Parameters
        ----------
        frame : np.ndarray
            Original frame in BGR format.
        detections : List[Dict[str, Any]]
            Filtered QR detections drawn in green.
        candidates : List[DecodeCandidate]
            Decode candidates drawn with success/failure labels.

        Returns
        -------
        np.ndarray
            Annotated frame copy.
        """
        annotated = frame.copy()
        for detection in detections:
            x1, y1, x2, y2 = [int(v) for v in detection.get("xyxy", [0, 0, 1, 1])]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        for candidate in candidates:
            x1, y1, x2, y2 = candidate.bbox_xyxy
            if candidate.payload:
                text = f"{candidate.payload} ({candidate.source_method})"
                color = (0, 255, 255)
            else:
                text = "decode_failed"
                color = (0, 120, 255)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated,
                text,
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

        return annotated
