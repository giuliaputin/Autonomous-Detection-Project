# Project Documentation

This folder contains technical documentation for the Autonomous Detection Project.

## Documents

- `ARCHITECTURE.md`: Module ownership, dependency boundaries, and runtime flow.
- `qr-detection-pipeline.md`: Detector training and inference workflows.
- `qr-decoding-pipeline.md`: Decoder pipeline, interfaces, and tuning behavior.

## Manual Debug Quick Start

Use these commands from project root for manual validation:

```bash
# Manual uploaded-image debug (single file or folder)
python scripts/manual_qr_image_debug.py --input data/images/test --log-level DEBUG --show

# Optional live webcam decode debug
python main.py --decode-live --model qr_detection/runs/detect/qr_detector_v17/weights/best.pt --log-level DEBUG --fps 5
```

Notes:

- `scripts/manual_qr_image_debug.py` is the recommended first step for controlled tests.
- Keep `--min-consecutive-frames 1 --cooldown-frames 0` for static image batches.
- Use `--log-level INFO` for concise output or `DEBUG` for attempt-level detail.
- If detections are found but filtered to zero, class names from detector outputs do not match accepted QR classes.

## Documentation Scope

- Architecture and module responsibilities.
- Runtime behavior and tuning parameters.
- Manual testing and validation workflow.
- Integration notes for connecting detection and inventory workflows.
