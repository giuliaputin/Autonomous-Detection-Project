from pathlib import Path
from detector import QRDetector

# Project root is now 2 levels up (was in vision/, now in QR detection/)
project_root = Path(__file__).resolve().parents[2]
data_yaml = project_root / "data" / "data.yaml"

# QR detection folder for saving runs and outputs
qr_detection_folder = Path(__file__).resolve().parent
runs_output = qr_detection_folder / "runs"

detector = QRDetector("yolov8n.pt")
detector.train(
    data_yaml=str(data_yaml),
    epochs=50,
    batch=16,
    imgsz=640,
    optimizer="auto",
    project=str(runs_output / "detect"),
    name="qr_detector_v1"
)