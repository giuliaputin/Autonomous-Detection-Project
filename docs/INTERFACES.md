# QR Detection Module Interfaces

## Overview

The QR Detection module provides object detection capabilities for QR codes using YOLOv8. It's organized into two main components:

1. **Detector** - Model training and inference
2. **Image Processor** - High-level image processing interface

---

## QRDetector Class

Located in: `qr_detection/src/detector.py`

### Purpose
Handles model training, validation, and inference for QR code object detection using Ultralytics YOLOv8.

### Dataset Format
Uses the **Ultralytics detection format** (YOLO object detection):

```
data/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
```

Each image has a matching `.txt` label file with normalized YOLO format:
```
<class_id> <x_center> <y_center> <width> <height>
```

For empty images (no QR codes), create an empty `.txt` file to train the model on negative examples.

### Key Methods

#### `__init__(model_path: str = "yolov8n.pt", conf_threshold: float = 0.5)`
Initialize the detector with a YOLO model.

```python
detector = QRDetector("yolov8n.pt", conf_threshold=0.5)
```

#### `create_dataset_structure(base_dir: str = "data")`
Create the required folder structure for the dataset.

```python
QRDetector.create_dataset_structure("data")
```

#### `write_data_yaml(base_dir: str = "data", class_names: Dict[int, str] | None = None)`
Generate `data.yaml` for training configuration.

```python
yaml_path = QRDetector.write_data_yaml("data", {0: "QR"})
```

#### `train(data_yaml: str, epochs: int = 50, batch: int = 16, imgsz: int = 640, ...)`
Train the model on the custom QR dataset.

```python
detector = QRDetector("yolov8n.pt")
results = detector.train(
    data_yaml="data/data.yaml",
    epochs=50,
    batch=16,
    imgsz=640,
    project="runs/detect",
    name="qr_detector_v1"
)
```

#### `predict_image(image_path: str, save: bool = False)`
Run inference on a single image file.

```python
results = detector.predict_image("photo.jpg")
```

#### `predict_frame(frame: np.ndarray)`
Run inference on an OpenCV frame (NumPy array).

```python
results = detector.predict_frame(cv2_frame)
```

#### `extract_detections(result) -> List[Dict]`
Convert YOLO results into a structured format.

Returns a list of detections, each with:
- `class_id`: Integer class identifier
- `class_name`: String class name
- `confidence`: Confidence score (0-1)
- `xyxy`: [x1, y1, x2, y2] bounding box coordinates
- `center`: [cx, cy] center coordinates
- `width`: Bounding box width in pixels
- `height`: Bounding box height in pixels

```python
result = detector.predict_frame(frame)
detections = detector.extract_detections(result)

for det in detections:
    print(f"QR found at {det['center']} with {det['confidence']:.2%} confidence")
```

#### `annotate_frame(frame: np.ndarray, result) -> np.ndarray`
Draw bounding boxes and labels on a frame.

```python
annotated = detector.annotate_frame(frame, result)
cv2.imshow("Detections", annotated)
```

---

## QRImageProcessor Class

Located in: `qr_detection/src/image_processor.py`

### Purpose
High-level interface for processing images and generating reports.

### Key Methods

#### `__init__(model_path: str = "yolov8n.pt", conf_threshold: float = 0.5)`
Initialize the processor with a model.

```python
processor = QRImageProcessor("./runs/detect/qr_detector_v1/weights/best.pt")
```

#### `process_image(image_path: str) -> Dict`
Process a single image and detect QR codes.

Returns:
```python
{
    "timestamp": "2024-01-15T10:30:45.123456",
    "image_path": "/path/to/image.jpg",
    "image_name": "image.jpg",
    "qr_count": 3,
    "model_used": "yolov8n.pt",
    "confidence_threshold": 0.5,
    "detections": [
        {
            "class_id": 0,
            "class_name": "QR",
            "confidence": 0.95,
            "xyxy": [100.0, 150.0, 200.0, 250.0],
            "center": [150.0, 200.0],
            "width": 100.0,
            "height": 100.0
        },
        # ... more detections
    ]
}
```

#### `get_summary(results: Dict) -> Dict`
Get statistics about detections.

```python
summary = processor.get_summary(results)
print(f"High confidence: {summary['high_confidence']}")
print(f"Average confidence: {summary['average_confidence']:.2%}")
```

#### `print_results(results: Dict)`
Print results in human-readable format.

#### `save_annotated_image(image_path: str, results: Dict, output_path: str | None = None) -> str`
Save image with QR bounding boxes drawn.

```python
output_path = processor.save_annotated_image("photo.jpg", results)
```

#### `save_results_json(results: Dict, output_path: str | None = None) -> str`
Save detection results as JSON.

```python
json_path = processor.save_results_json(results, "results.json")
```

#### `create_summary_report(image_paths: List[str], output_dir: str | None = None) -> Dict`
Process multiple images and create a batch report.

```python
image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
report = processor.create_summary_report(image_paths, output_dir="./reports")
```

---

## Quick Functions

### `quick_process(image_path: str, model_path: str = "yolov8n.pt", save_annotated: bool = False) -> Dict`

Simplest way to process an image:

```python
from qr_detection import quick_process

results = quick_process("my_image.jpg", save_annotated=True)
print(f"Found {results['qr_count']} QR codes")
```

---

## Usage Examples

### Example 1: Train a Custom QR Detector

```python
from qr_detection import QRDetector

# Setup dataset structure
QRDetector.create_dataset_structure("data")
yaml_path = QRDetector.write_data_yaml("data", {0: "QR"})

# Train model
detector = QRDetector("yolov8n.pt")
results = detector.train(
    data_yaml="data/data.yaml",
    epochs=50,
    batch=16,
    imgsz=640,
    project="runs/detect",
    name="qr_detector_v1"
)
```

### Example 2: Process a Single Image

```python
from qr_detection import QRImageProcessor

processor = QRImageProcessor("./runs/detect/qr_detector_v1/weights/best.pt")
results = processor.process_image("photo.jpg")
processor.print_results(results)

# Save outputs
processor.save_annotated_image("photo.jpg", results, "photo_annotated.jpg")
processor.save_results_json(results, "detections.json")
```

### Example 3: Batch Processing

```python
from pathlib import Path
from qr_detection import QRImageProcessor

processor = QRImageProcessor()
images = list(Path("test_images/").glob("*.jpg"))

report = processor.create_summary_report(
    [str(img) for img in images],
    output_dir="./batch_results"
)
```

### Example 4: Real-time Video Processing

```python
import cv2
from qr_detection import QRDetector

detector = QRDetector("./runs/detect/qr_detector_v1/weights/best.pt")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = detector.predict_frame(frame)
    detections = detector.extract_detections(result)
    annotated = detector.annotate_frame(frame, result)

    print(f"QR codes found: {len(detections)}")
    cv2.imshow("QR Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

## Dataset Preparation

### Label Format
Create `.txt` files matching each image:

```
# image.jpg has image.txt with:
0 0.5 0.5 0.3 0.4
0 0.7 0.3 0.2 0.25
```

Where each line is: `class_id center_x center_y width height` (normalized 0-1)

### Empty Images
For images without QR codes, create an empty `.txt` file:

```bash
touch labels/train/empty_001.txt
```

This helps the model learn negative examples.

### Tools for Labeling
- **Roboflow**: Cloud-based annotation tool
- **LabelImg**: Desktop annotation tool
- **CVAT**: Web-based annotation platform

---

## Configuration

### Confidence Threshold
Controls which detections are kept. Default is 0.5 (50%):

```python
processor = QRImageProcessor(conf_threshold=0.7)  # Only keep 70%+ confidence
```

### Training Hyperparameters
Use defaults first, tune only if needed:

```python
detector.train(
    epochs=50,        # Number of training epochs
    batch=16,         # Batch size (reduce if GPU memory limited)
    imgsz=640,        # Input image size
    optimizer="auto"  # Keep as "auto" initially
)
```

---

## Error Handling

```python
from qr_detection import QRImageProcessor

processor = QRImageProcessor()

try:
    results = processor.process_image("photo.jpg")
except FileNotFoundError:
    print("Image file not found")
except ValueError as e:
    print(f"Invalid image format: {e}")
except Exception as e:
    print(f"Detection failed: {e}")
```

---

## Performance Tips

1. **Use GPU**: YOLO will automatically use CUDA if available
2. **Batch Processing**: Process multiple images to maximize throughput
3. **Image Size**: Smaller images (416px) are faster, larger (640px) are more accurate
4. **Model Size**: 
   - `yolov8n.pt`: Fast, less accurate
   - `yolov8s.pt`: Balanced
   - `yolov8m.pt`: More accurate, slower

Example using GPU and batch processing:

```python
import torch
print(f"GPU available: {torch.cuda.is_available()}")

processor = QRImageProcessor("yolov8s.pt")  # Smaller/faster model

images = ["img1.jpg", "img2.jpg", "img3.jpg"]
report = processor.create_summary_report(images)
```

---

## Integration with Other Modules

### With QR Reader
After detection, pass QR coordinates to the decoder:

```python
from qr_detection import QRImageProcessor
from qr_reader import QRDecoder

processor = QRImageProcessor()
results = processor.process_image("photo.jpg")

decoder = QRDecoder()
for det in results["detections"]:
    x1, y1, x2, y2 = map(int, det["xyxy"])
    qr_crop = frame[y1:y2, x1:x2]
    decoded = decoder.decode(qr_crop)
    print(f"QR Content: {decoded}")
```

### With Vision System
Use detector in camera pipeline:

```python
from qr_detection import QRDetector
from vision.camera import Camera

camera = Camera()
detector = QRDetector()

frame = camera.capture()
result = detector.predict_frame(frame)
detections = detector.extract_detections(result)

for det in detections:
    print(f"QR at {det['center']}: {det['confidence']:.2%}")
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Very slow inference | Use smaller model (yolov8n.pt) or reduce imgsz |
| Low detection accuracy | Train longer (more epochs) or use more labeled data |
| GPU not detected | Check PyTorch CUDA installation |
| Out of memory | Reduce batch size during training |
| No detections found | Verify model path and image format |

---

## Version Information

- **Framework**: Ultralytics YOLOv8
- **Python**: 3.8+
- **Dependencies**: opencv-python, ultralytics, numpy

---

*Last updated: 2024 - QR Detection Module Documentation*
