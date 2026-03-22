"""Backward-compatible delegate to the QR detector training entrypoint."""

from __future__ import annotations

from qr_detection.train import main


if __name__ == "__main__":
    main()