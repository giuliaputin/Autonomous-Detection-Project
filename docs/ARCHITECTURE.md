# Architecture

This document describes the runtime boundaries and module ownership for the project.

## Design Goals

- Keep entry scripts thin and deterministic.
- Centralize runtime orchestration so behavior is implemented once.
- Keep QR decoding logic isolated in `qr_decoder`.
- Allow detector and decoder evolution with minimal coupling.

## Directory Ownership

- `main.py`
  - Single top-level delegate.
  - Routes to CLI entrypoint logic in `apps`.
- `apps/`
  - User-facing runnable modes.
  - Argument parsing and mode dispatch.
  - No heavy business logic.
- `core/`
  - Shared orchestration used by app modes.
  - Runtime config defaults and logging bootstrap.
- `qr_detection/`
  - Detection model train/infer interfaces.
- `qr_decoder/`
  - QR decode pipeline package.
  - `src/` contains implementation modules.
  - `data/` stores manual debug inputs.
- `scripts/`
  - Thin wrappers only.
  - Should call into `apps` or package entrypoints.

## Runtime Flow

1. User runs `main.py` or a thin wrapper under `scripts/`.
2. `apps.entrypoints` resolves selected mode.
3. App mode calls shared helpers from `core.orchestration`.
4. Detector returns candidate bounding boxes.
5. `qr_decoder` pipeline filters, decodes, and confirms payloads.
6. Accepted payloads are emitted to stdout/logs.

## Dependency Rules

- `main.py` can import `apps` only.
- `apps` can import `core`, `vision`, `qr_detection`, and `qr_decoder`.
- `core` must remain generic and import-light.
- `qr_decoder` must not depend on `apps` or `main.py`.
- `scripts` must not contain business logic.

## Logging

- Use `logging` with namespace loggers:
  - `qr_decoder.*` for decode internals.
  - `apps.*` and `core.*` for runtime orchestration.
- Prefer structured, short messages with key runtime counters.

## Manual Testing

- Place test images in `qr_decoder/data/`.
- Run:
  - `make run-image-debug`
  - `make run-image-debug-fallback`
  - `make run-decode-live`

## Future Extensions

- Add formal package entrypoints via `python -m ...` for all app modes.
- Add automated tests for:
  - class-filter behavior,
  - temporal confirmation,
  - decode fallback ordering.
