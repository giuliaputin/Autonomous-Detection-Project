"""Decoupled orchestration for decode -> temporal confirm."""

from __future__ import annotations

from dataclasses import dataclass
from time import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from .decoder import DecodeCandidate, QRDecoder


@dataclass
class AcceptedQR:
    payload: str
    frame_index: int
    timestamp_s: float
    bbox_xyxy: tuple[int, int, int, int]
    source_method: Optional[str]


class TemporalQRConfirmer:
    """Accept payloads only after short consecutive-frame agreement."""

    def __init__(self, min_consecutive_frames: int = 2, cooldown_frames: int = 25) -> None:
        self.min_consecutive_frames = min_consecutive_frames
        self.cooldown_frames = cooldown_frames
        self._state: Dict[str, Dict[str, int]] = {}

    def update(self, frame_index: int, candidates: List[DecodeCandidate]) -> List[str]:
        accepted: List[str] = []

        by_payload: Dict[str, DecodeCandidate] = {}
        for candidate in candidates:
            if not candidate.payload:
                continue
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

            self._state[payload] = state

        return accepted


class QRDecodePipeline:
    """Detection-agnostic pipeline: caller provides detections per frame."""

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
        """Process a frame and detector outputs and return decode results."""
        timestamp_s = time() if timestamp_s is None else timestamp_s

        filtered = [d for d in detections if self._is_target_qr_class(d)]
        candidates = self.decoder.decode_detections(frame, filtered)
        accepted_payloads = self.confirmer.update(frame_index, candidates)

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
