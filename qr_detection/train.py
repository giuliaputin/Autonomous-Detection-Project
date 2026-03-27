"""Training entrypoint for QR detector model."""

from __future__ import annotations

from multiprocessing import freeze_support
from pathlib import Path

from qr_detection.detector import QRDetector


def main() -> None:
    """Train detector using repository dataset configuration.

    Notes
    -----
    Uses ``data/data.yaml`` from project root and writes run artifacts into
    ``qr_detection/runs/detect``.

    Raises
    ------
    FileNotFoundError
        If expected dataset configuration file does not exist.
    """
    project_root = Path(__file__).resolve().parents[1]
    data_yaml = project_root / "data" / "data.yaml"

    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_yaml}")

    qr_detection_folder = Path(__file__).resolve().parent
    runs_output = qr_detection_folder / "runs"

    # Start from a compact pretrained checkpoint for faster iteration.
    detector = QRDetector("yolov8n.pt")
    detector.train(
        data_yaml=str(data_yaml),
        epochs=50,
        batch=16,
        imgsz=640,
        optimizer="auto",
        project=str(runs_output / "detect"),
        name="qr_detector_v1",
    )


if __name__ == "__main__":
    freeze_support()
    main()
