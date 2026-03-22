# Project Documentation

This folder contains technical documentation for the Autonomous Detection Project.

## Documents

- [QR Decoding Pipeline](qr-decoding-pipeline.md): Design and implementation guide for QR reading using detector-provided bounding boxes.

## Manual Debug Quick Start

Use these commands from project root for manual validation:

```bash
# Manual uploaded-image debug (single file or folder)
python manual_qr_image_debug.py --input data/images/test --log-level DEBUG --show

# Optional live webcam decode debug
python main.py --decode-live --log-level DEBUG
```

Notes:

- `manual_qr_image_debug.py` is the recommended first step for controlled tests.
- Keep `--min-consecutive-frames 1 --cooldown-frames 0` for static image batches.
- Use `--log-level INFO` for concise output or `DEBUG` for attempt-level detail.

## Documentation Scope

- Architecture and module responsibilities.
- Runtime behavior and tuning parameters.
- Manual testing and validation workflow.
- Integration notes for connecting detection and inventory workflows.
