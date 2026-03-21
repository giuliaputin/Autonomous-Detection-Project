# QR Decoding Pipeline

## Objective

Build a robust and understandable QR reading stage for the drone inventory workflow. Detection is assumed to be provided by a separate component (coworker branch). This pipeline starts from detected QR bounding boxes and returns accepted decoded payloads.

## Final Stack

- Detection source: external YOLOv8 detections (pixel xyxy boxes).
- Primary decoder: zxing-cpp Python bindings.
- Fallback decoder: OpenCV QRCodeDetector.
- Robustness strategy: crop margin + lightweight preprocessing retries + short temporal confirmation.

## Why This Works In Practice

- YOLO localization is used for coarse region finding only.
- Margin expansion protects QR quiet zone and prevents tight-crop decode failures.
- zxing-cpp is fast and robust for normal cases.
- OpenCV fallback catches additional cases when zxing-cpp misses.
- Multi-frame confirmation suppresses one-frame false reads in motion/blur.

## Module Structure

- qr_reader/utils.py
  - BBox clamping and expansion.
  - Crop extraction.
  - Lightweight preprocessing variants.

- qr_reader/decoder.py
  - Decoder engine and attempt telemetry.
  - zxing-cpp first, OpenCV fallback.
  - Variant retry loop.

- qr_reader/pipeline.py
  - Orchestration for decode -> temporal confirm.
  - Accepted payload filtering and frame annotation.

- qr_reader/interface.py
  - Public decoder interface entrypoint for external modules.

- main.py
  - Webcam-only runtime loop (decoder not coupled yet).

## Input Contract

Each detection is expected as a dictionary with at least:

- xyxy: [x1, y1, x2, y2] in pixel coordinates.
- confidence: detector confidence float.
- class_name: class label string.

Detections must be provided by the caller (detection-agnostic design).

## Processing Flow

1. Receive frame and detections.
2. Filter detections to QR class names.
3. Expand each bbox by configurable margin.
4. Decode raw crop with zxing-cpp.
5. Fallback decode with OpenCV QRCodeDetector.
6. If both fail, retry 3 to 5 cheap variants:
   - grayscale
   - 2x upscale grayscale
   - CLAHE
   - adaptive threshold
   - mild denoise
7. Collect attempt telemetry and per-frame decode candidates.
8. Confirm payloads across consecutive frames.
9. Emit accepted payloads (currently logging in main loop).

## Temporal Confirmation Policy

Current defaults:

- min_consecutive_frames = 2
- cooldown_frames = 25

Interpretation:

- A payload must appear in two consecutive frames before acceptance.
- After acceptance, the same payload is suppressed for 25 frames to avoid duplicate inventory events.

## Configuration Knobs

Primary tuning parameters in QRDecoder:

- margin_ratio (default 0.25)
- max_preprocess_variants (default 5)
- enable_preprocessing (default true)
- prefer_zxing (default true)

Temporal tuning in TemporalQRConfirmer:

- min_consecutive_frames
- cooldown_frames

## Dependencies

Install required packages in your active Python environment:

```bash
pip install ultralytics opencv-python zxing-cpp
```

Notes:

- opencv-python already includes QRCodeDetector.
- If zxing-cpp is unavailable, pipeline still runs with OpenCV fallback only.

## Runtime Usage

The decoder is intentionally standalone and not wired into main.py yet.

Use the public interface from qr_reader/interface.py:

- Build interface: build_decoder_interface(...)
- Call process_frame(frame, frame_index, detections)
- Read accepted payloads from result["accepted"]

This keeps vision webcam code independent while detection and integration are still under development.

## Testing Checklist

1. Unit tests
   - BBox expansion and clipping at image boundaries.
   - Temporal confirmer streak and cooldown behavior.
   - Variant ordering and stop-on-first-success behavior.

2. Integration tests
   - Known-good QR crops.
   - Hard cases: blur, low light, angle, small QR in frame.
   - False-positive suppression across noisy frames.

3. Live validation
   - Verify accepted payloads match package labels.
   - Verify same package is not repeatedly accepted every frame.
   - Check frame loop responsiveness and stable latency.

## Engineering Notes

- Keep detector and decoder loosely coupled through a minimal detection contract.
- Preserve deterministic behavior in confirmation logic to simplify testing.
- Log attempt-level telemetry early; tuning is much faster with real measurements.
