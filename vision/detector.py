"""
detector.py

YOLOv8/Ultralytics detector for QR-code object detection.

This version is written for the REAL object-detection dataset format used by
Ultralytics YOLO detect models such as yolov8n.pt.

Dataset structure expected
--------------------------
data/
├─ images/
│  ├─ train/
│  ├─ val/
│  └─ test/
├─ labels/
│  ├─ train/
│  ├─ val/
│  └─ test/
└─ data.yaml

Important:
- The images do NOT contain the labels "inside" the image file.
- Each image has a matching .txt file in labels/... with the same filename stem.
- Example:
    images/train/img001.jpg
    labels/train/img001.txt

Label format in each .txt file
------------------------------
One object per line:
<class_id> <x_center> <y_center> <width> <height>

All coordinates are NORMALIZED between 0 and 1 relative to image size.

For a single-class QR-code dataset:
- class_id is always 0
- names in data.yaml should be: 0: QR

Example label file for one QR code:
0 0.512500 0.430000 0.225000 0.180000

If an image has NO QR code:
- create an EMPTY .txt file with the same stem
  Example:
    images/train/no_qr_01.jpg
    labels/train/no_qr_01.txt   # empty file

This lets YOLO learn both positive and negative scenes.

Training notes
--------------
- We use transfer learning from COCO-pretrained yolov8n.pt.
- Start with mostly defaults and only a few chosen hyperparameters.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import cv2
from ultralytics import YOLO


class QRDetector:
    """Ultralytics YOLOv8 detector for QR codes."""

    def __init__(self, model_path: str = "yolov8n.pt", conf_threshold: float = 0.5):
        """
        Load a pretrained YOLO detection model.

        Parameters
        ----------
        model_path : str
            Path to a YOLO detection model. For transfer learning, start with
            'yolov8n.pt' (nano model pretrained on COCO).
        conf_threshold : float
            Minimum confidence used when filtering predictions for display.
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    @staticmethod
    def create_dataset_structure(base_dir: str = "data") -> None:
        """
        Create the folder structure required by Ultralytics YOLO detection.

        Parameters
        ----------
        base_dir : str
            Root dataset directory.
        """
        base = Path(base_dir)
        folders = [
            base / "images" / "train",
            base / "images" / "val",
            base / "images" / "test",
            base / "labels" / "train",
            base / "labels" / "val",
            base / "labels" / "test",
        ]
        for folder in folders:
            folder.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def write_data_yaml(base_dir: str = "data", class_names: Optional[Dict[int, str]] = None) -> Path:
        """
        Create data.yaml for Ultralytics training.

        Parameters
        ----------
        base_dir : str
            Root dataset directory.
        class_names : dict[int, str] | None
            Class dictionary. For this project use {0: "QR"}.

        Returns
        -------
        Path
            Path to the written YAML file.
        """
        if class_names is None:
            class_names = {0: "QR"}

        base = Path(base_dir)
        yaml_path = base / "data.yaml"

        lines = [
            f"path: {base.resolve().as_posix()}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "",
            f"nc: {len(class_names)}",
            "names:",
        ]
        for idx, name in class_names.items():
            lines.append(f"  {idx}: {name}")

        yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return yaml_path

    def train(
        self,
        data_yaml: str = "data/data.yaml",
        epochs: int = 50,
        batch: int = 16,
        imgsz: int = 640,
        optimizer: str = "auto",
        project: str = "runs/detect",
        name: str = "qr_detector_v1",
    ):
        """
        Train the detector on the custom QR dataset.

        Parameters
        ----------
        data_yaml : str
            Path to the dataset YAML file.
        epochs : int
            Start with 50 as discussed.
        batch : int
            Batch size. Reduce if the GPU runs out of memory.
        imgsz : int
            Input image size. 640 is a common default for detection.
        optimizer : str
            Keep 'auto' first; tune later only if needed.
        project : str
            Output project folder.
        name : str
            Run name.

        Returns
        -------
        Training results object from Ultralytics.
        """
        return self.model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            optimizer=optimizer,
            project=project,
            name=name,
            pretrained=True,
        )

    def validate(self, data_yaml: str = "data/data.yaml"):
        """
        Validate the currently loaded model on the dataset.

        Parameters
        ----------
        data_yaml : str
            Path to dataset YAML.
        """
        return self.model.val(data=data_yaml)

    def predict_image(self, image_path: str, save: bool = False):
        """
        Run inference on a single image.

        Parameters
        ----------
        image_path : str
            Path to image file.
        save : bool
            Whether to save YOLO visual outputs.
        """
        return self.model.predict(source=image_path, conf=self.conf_threshold, save=save)

    def predict_frame(self, frame):
        """
        Run inference on an OpenCV frame (NumPy array).

        Parameters
        ----------
        frame : np.ndarray
            Input BGR frame.

        Returns
        -------
        ultralytics.engine.results.Results
            First result object.
        """
        results = self.model.predict(source=frame, conf=self.conf_threshold, verbose=False)
        return results[0]

    def extract_detections(self, result) -> List[Dict]:
        """
        Convert YOLO results into a simpler Python list.

        Parameters
        ----------
        result : ultralytics.engine.results.Results
            YOLO result object for one image/frame.

        Returns
        -------
        list[dict]
            List with bounding boxes, confidence, class id and class name.
        """
        detections: List[Dict] = []

        if result.boxes is None:
            return detections

        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append(
                {
                    "class_id": cls_id,
                    "class_name": result.names.get(cls_id, str(cls_id)),
                    "confidence": conf,
                    "xyxy": [x1, y1, x2, y2],
                    "center": [(x1 + x2) / 2.0, (y1 + y2) / 2.0],
                    "width": x2 - x1,
                    "height": y2 - y1,
                }
            )

        return detections

    def annotate_frame(self, frame, result):
        """
        Draw YOLO bounding boxes and labels on a frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image.
        result : ultralytics.engine.results.Results
            YOLO result object.

        Returns
        -------
        np.ndarray
            Annotated frame.
        """
        annotated = frame.copy()
        detections = self.extract_detections(result)

        for det in detections:
            if det["confidence"] < self.conf_threshold:
                continue

            x1, y1, x2, y2 = map(int, det["xyxy"])
            cx, cy = map(int, det["center"])
            label = f'{det["class_name"]}: {det["confidence"]:.2f}'

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(
                annotated,
                label,
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        return annotated


def main() -> None:
    """
    Prepare the dataset structure and YAML file.

    This does not train automatically, because you first need to place your
    images and matching label files into the dataset folders.
    """
    QRDetector.create_dataset_structure("data")
    yaml_path = QRDetector.write_data_yaml("data", {0: "QR"})
    print(f"Dataset folders ready. YAML written to: {yaml_path}")
    print("Add your images and matching YOLO .txt label files, then train with:")
    print()
    print("from detector import QRDetector")
    print("detector = QRDetector('yolov8n.pt')")
    print("detector.train(data_yaml='data/data.yaml', epochs=50, batch=16, imgsz=640)")


if __name__ == "__main__":
    main()
