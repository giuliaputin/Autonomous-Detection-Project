# CLAUDE.md

This file is the project memory and agent-facing guidance for this repository.
Use this file for project memory. Do not create or update `AGENTS.md` for this
project.

## Project Overview

AEROSCAN (Autonomous Detection Project) is a real-time QR code detection and
decoding system for autonomous drone-based inventory scanning. It uses YOLOv8
for detection and zxing-cpp for decoding, with Crazyflie drone integration.

## Commands

```bash
# Install
make install
make install-dev

# Run
python main.py webcam
python main.py live --model <model.pt>
python main.py image --input <path>

# Make shortcuts
make run-webcam
make run-decode-live
make run-image-debug
make run-image-debug-fallback

# Training
make train-detector

# Code quality
make lint
make format
make pre-commit

# Tests
make test
python -m pytest tests -v
python -m pytest tests/path/test.py
tox
tox -e lint

# Cleanup
make clean
```

## Architecture

```text
main.py -> apps/entrypoints.py -> apps/{webcam_live,decode_live,image_debug}.py
                                      |
                                      v
                              core/orchestration.py
                              (DecodeOrchestrator)
                                |              |
                                v              v
                         qr_detection/   qr_decoder/src/
                         detector.py     interface.py -> pipeline.py -> decoder.py
                         (YOLOv8)        (temporal confirmer + decode engine)
                                |
                                v
                         vision/camera.py    drone_controls/
                         (VideoCapture)      (Crazyflie takeoff/landing)
```

## Key Modules

- `core/orchestration.py`: `DecodeOrchestrator` composes the detector and
  decoder via `DecodeOrchestrator.build(model_path, min_consecutive_frames,
  cooldown_frames)` and exposes `process_frame(frame, frame_index)`.
- `core/config.py`: `LiveDecodeDefaults` dataclass with canonical runtime
  defaults: camera 0, 640x480, 5 FPS, YOLOv8n, 2 consecutive frames, and
  25-frame cooldown.
- `qr_detection/detector.py`: `QRDetector` wraps Ultralytics YOLOv8.
  `predict_frame(frame)` returns a raw result. `extract_detections(result)`
  returns a list of `{xyxy, confidence, class_name}` dictionaries.
- `qr_detection/train.py`: training entrypoint; expects `data/data.yaml`.
- `qr_decoder/src/interface.py`: public decoder surface.
  `QRDecoderInterface` and `build_decoder_interface()` provide
  `process_frame(frame, frame_index, detections)` returning `{candidates,
  accepted_payloads}`.
- `qr_decoder/src/pipeline.py`: `QRDecodePipeline` filters detections, accepts
  class names `qr` and `qrcode`, expands bboxes with a configurable margin, and
  feeds cropped regions to the decoder. `TemporalQRConfirmer` requires
  `min_consecutive_frames` matches before emitting a payload and suppresses
  re-emission for `cooldown_frames`.
- `qr_decoder/src/decoder.py`: `QRDecoder` tries zxing-cpp first, falls back to
  OpenCV `QRCodeDetector`, and retries with preprocessing variants such as
  CLAHE, adaptive threshold, denoise, and upscale. It records `DecodeAttempt`
  telemetry.
- `qr_decoder/src/utils.py`: bbox clamping, crop extraction, and preprocessing
  helpers.
- `apps/`: runnable modes; thin wrappers around core orchestration.
- `vision/camera.py`: `Camera` wraps `cv2.VideoCapture` with a Windows backend
  fallback chain: DSHOW -> MSMF -> ANY.
- `drone_controls/`: Crazyflie `CrazyflieTakeoff` and `land()`.

## Data Flow Rules

- `apps/` may import from `core/`, `qr_detection/`, `qr_decoder/`, `vision/`,
  and `drone_controls/`; never the reverse.
- `core/` must not import from `apps/`.
- `qr_detection/` and `qr_decoder/` are independently testable; neither imports
  from the other.

## Tooling

- Ruff is the linter and formatter, configured in `pyproject.toml`.
- Pre-commit runs Ruff auto-fix and format hooks on `git commit`.
- Python 3.9+ is required.
- Use LF line endings and 4-space indents for Python.
- YOLO weights (`.pt`) and training runs (`runs/`) are git-ignored; place
  weights locally.
- Training data is git-ignored; YOLO training expects `data/data.yaml`.

## Project Notes

- If logs show `detections > 0` and `filtered = 0`, class names from detector do
  not match the decoder filter policy (`qr`, `qrcode`).
- Prefer trained model weights for QR workflows over generic COCO models.
