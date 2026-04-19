"""Backward-compatible delegate to detector training entrypoint.

This module preserves the historical import path for tooling that still calls
``python -m qr_detection.interface``.
"""

from __future__ import annotations

from qr_detection.train import main

if __name__ == "__main__":
    main()
