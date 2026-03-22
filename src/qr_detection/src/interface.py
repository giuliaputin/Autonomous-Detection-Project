from pathlib import Path
from multiprocessing import freeze_support
from .detector import QRDetector

def main() -> None:
    # interface.py is in src/qr_detection/src/interface.py
    # Go up 3 levels to project root (src -> qr_detection -> src)
    project_root = Path(__file__).resolve().parent.parent.parent.parent
    
    # data is at qr_detection level, up one level from src/
    qr_detection_folder = Path(__file__).resolve().parent.parent
    data_yaml = qr_detection_folder / "data" / "data.yaml"

    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_yaml}")

    # QR detection folder for saving runs and outputs
    runs_output = qr_detection_folder / "runs"

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
