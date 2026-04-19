# Autonomous Detection Project

Computer-vision workspace for QR detection and QR payload decoding.

## Architecture

- `apps/`: runnable application flows.
- `core/`: shared orchestration and runtime configuration.
- `qr_detection/`: YOLO detector training and inference.
- `qr_decoder/src/`: QR decoding engine and temporal acceptance pipeline.
- `vision/`: camera access utilities.
- `scripts/`: thin executable wrappers that delegate to app modules.

## Common Commands

```bash
# Webcam only
python main.py webcam

# Live detection + decode (5 FPS default)
python main.py live --model qr_detection/runs/detect/qr_detector_v17/weights/best.pt --log-level DEBUG

# Manual image debug for a file or folder
python main.py image --input qr_decoder/data --model qr_detection/runs/detect/qr_detector_v17/weights/best.pt --log-level DEBUG
```

```bash
# Convenience task aliases
make run-webcam
make run-decode-live
make run-image-debug
make run-image-debug-fallback
make train-detector
```

## Tooling

- `Makefile`: common developer tasks.
- `pyproject.toml`: packaging and tool configuration.
- `.editorconfig`: formatting conventions.
- `.pre-commit-config.yaml`: commit-time checks.
- `tox.ini`: repeatable lint/test environments.

## Notes

- If logs show `detections > 0` and `filtered = 0`, class names from detector do not match the decoder filter policy (`qr`, `qrcode`).
- Prefer trained model weights for QR workflows over generic COCO models.
