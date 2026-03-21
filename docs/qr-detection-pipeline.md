# QR Detection Pipeline - Complete Guide

## 📋 Overview

The **QR Detection Pipeline** is a machine learning-based system for detecting QR codes in images and video frames. It uses YOLOv8 (You Only Look Once), a state-of-the-art object detection model, to quickly and accurately find QR codes in real-time.

### What is YOLO?
YOLO is an object detection framework that:
- **Detects objects** in one pass (very fast)
- **Works in real-time** (can process video frames)
- **Transfer learns** from pre-trained COCO weights (starts with knowledge about millions of objects)
- **Outputs** bounding boxes with confidence scores

---

## 📁 Folder Structure

```
qr_detection/
├── detector.py          # Core QR detection engine (the brain)
├── interface.py         # Training script (how to train the model)
└── runs/                # Output folder (where results are saved)
    └── detect/
        └── qr_detector_v1/
            ├── weights/
            │   ├── best.pt          # Best trained model
            │   └── last.pt          # Last checkpoint
            ├── training_results/    # Metrics and plots
            ├── predictions/         # Inference outputs
            └── logs/                # Training logs
```

### What Each File Does

| File | Purpose |
|------|---------|
| `detector.py` | Contains the `QRDetector` class - handles model training, inference, and predictions |
| `interface.py` | Main entry point - configures and runs training automatically |
| `runs/` | Stores all training and inference outputs |

---

## 🔧 detector.py - The Core Engine

This is the main Python module that does all the heavy lifting.

### The QRDetector Class

```python
from detector import QRDetector

detector = QRDetector("yolov8n.pt")
```

#### **1. Initialize the Detector**
```python
detector = QRDetector(
    model_path="yolov8n.pt",      # Which YOLO model to use
    conf_threshold=0.5             # Confidence threshold (0-1, lower = more detections)
)
```

**Parameters:**
- `model_path`: Path to the YOLO model file
  - `yolov8n.pt` = Nano (small, fast, less accurate)
  - `yolov8s.pt` = Small (balanced)
  - `yolov8m.pt` = Medium (more accurate but slower)
  - `yolov8l.pt` = Large (most accurate)

- `conf_threshold`: Minimum confidence to accept a detection
  - `0.5` (default) = Require 50% confidence
  - `0.9` = Only show very confident detections

---

### **Core Methods**

#### **Training**
Train the model on your QR code dataset:

```python
detector.train(
    data_yaml="data/data.yaml",        # Path to dataset config
    epochs=50,                          # Number of training loops
    batch=16,                           # Images per training step
    imgsz=640,                          # Image size for training
    optimizer="auto",                   # Optimization algorithm
    project="qr_detection/runs/detect", # Where to save results
    name="qr_detector_v1"               # Name of this training run
)
```

**What Training Does:**
1. Loads your labeled images from `data/images/train/`
2. Compares predictions against labels from `data/labels/train/`
3. Adjusts model weights to improve accuracy
4. Tests on validation set (`data/images/val/`)
5. Saves the best model as `best.pt`

**Training Time:**
- 50 epochs on GPU: ~30-60 minutes (depends on GPU)
- Creates checkpoints every few epochs (can resume if interrupted)

---

#### **Validation**
Test the model on a validation dataset:

```python
results = detector.validate(data_yaml="data/data.yaml")
```

Returns metrics like:
- **mAP** (mean Average Precision) - overall accuracy
- **Precision** - how many detections were correct
- **Recall** - how many actual QR codes were found

---

#### **Inference on Images**
Detect QR codes in a single image:

```python
results = detector.predict_image(
    image_path="path/to/image.jpg",
    save=True  # Save annotated image with boxes
)
```

**Returns:** YOLO results object containing all detections

---

#### **Inference on Video Frames**
Detect QR codes in a live video frame (for real-time detection):

```python
import cv2

cap = cv2.VideoCapture(0)  # Open webcam
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run detection on this frame
    result = detector.predict_frame(frame)
    
    # Draw boxes on frame
    annotated = detector.annotate_frame(frame, result)
    
    cv2.imshow("QR Detection", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

---

#### **Extract Detections**
Convert YOLO results into a simple list you can use:

```python
result = detector.predict_image("image.jpg")
detections = detector.extract_detections(result)

for det in detections:
    print(f"QR Code found!")
    print(f"  Confidence: {det['confidence']:.2f} (0-1 scale)")
    print(f"  Position: {det['xyxy']} (x1, y1, x2, y2)")
    print(f"  Center: {det['center']}")
    print(f"  Size: {det['width']} x {det['height']}")
```

**Detection Dictionary Structure:**
```python
{
    "class_id": 0,                      # Always 0 for QR codes
    "class_name": "QR",                 # What it detected
    "confidence": 0.95,                 # How sure (0-1)
    "xyxy": [x1, y1, x2, y2],          # Bounding box corners (pixels)
    "center": [cx, cy],                 # Center point (pixels)
    "width": 100,                       # Width in pixels
    "height": 100                       # Height in pixels
}
```

---

#### **Annotate Frames**
Draw detection boxes on images/frames:

```python
result = detector.predict_image("image.jpg")
annotated_image = detector.annotate_frame(frame, result)

# Now annotated_image shows green boxes around QR codes
```

**Drawing Style:**
- 🟢 Green rectangle = QR code bounding box
- 🔴 Red dot = Center point
- 📝 Text = "QR: 0.95" (label and confidence)

---

#### **Dataset Setup (Utilities)**

```python
# Create folder structure for training data
QRDetector.create_dataset_structure("data")

# Create data.yaml file for YOLO
QRDetector.write_data_yaml("data", {0: "QR"})
```

---

## 📝 interface.py - The Training Script

This is the entry point that automatically configures and runs everything:

```python
from pathlib import Path
from detector import QRDetector

# Find the project root (go up 2 folders from qr_detection/)
project_root = Path(__file__).resolve().parents[2]
data_yaml = project_root / "data" / "data.yaml"

# Setup output folder
qr_detection_folder = Path(__file__).resolve().parent
runs_output = qr_detection_folder / "runs"

# Create detector
detector = QRDetector("yolov8n.pt")

# Train!
detector.train(
    data_yaml=str(data_yaml),
    epochs=50,
    batch=16,
    imgsz=640,
    optimizer="auto",
    project=str(runs_output / "detect"),
    name="qr_detector_v1"
)
```

### How to Use It

**From command line:**
```bash
cd "QR detection"
python interface.py
```

**What Happens:**
1. Checks if `data/data.yaml` exists
2. Loads the YOLOv8 nano model
3. Starts training for 50 epochs
4. Saves all results to `qr_detection/runs/detect/qr_detector_v1/`
5. Displays real-time training metrics

---

## 📊 Data Format

The training system expects data in a specific format:

### Folder Structure
```
data/
├── data.yaml              # Configuration file
├── images/
│   ├── train/             # Training images (70%)
│   ├── val/               # Validation images (20%)
│   └── test/              # Test images (10%)
└── labels/
    ├── train/             # Training labels
    ├── val/               # Validation labels
    └── test/              # Test labels
```

### Image Format
- **Supported:** JPG, PNG, BMP, TIFF
- **Recommended:** 640x640 pixels (can be any size, YOLO resizes)
- **Naming:** `image_001.jpg`

### Label Format (`.txt` files)
Each image has a matching `.txt` file with the same name:

```
image_001.jpg    →    image_001.txt
```

**Label File Contents:**
Each line represents one QR code found in the image:
```
<class_id> <x_center> <y_center> <width> <height>
```

**Example:**
```
0 0.5 0.5 0.2 0.2
```

This means:
- `0` = Class 0 (QR code)
- `0.5` = Center X at 50% across image
- `0.5` = Center Y at 50% down image
- `0.2` = Width is 20% of image width
- `0.2` = Height is 20% of image height

**All coordinates are NORMALIZED (0-1 scale), not pixels!**

### If Image Has No QR Codes
Create an empty `.txt` file:
```
image_no_qr.jpg    →    image_no_qr.txt (empty)
```

### data.yaml Configuration
```yaml
path: /full/path/to/data
train: images/train
val: images/val
test: images/test

nc: 1
names:
  0: QR
```

---

## 🚀 Step-by-Step Training Guide

### Step 1: Prepare Your Data
```
1. Collect QR code images (hundreds recommended)
2. Split into: 70% train, 20% val, 10% test
3. Create bounding boxes (use labeling tool like Roboflow)
4. Place images in data/images/{train,val,test}/
5. Place labels in data/labels/{train,val,test}/
```

**Recommended Labeling Tools:**
- Roboflow Annotate (online)
- CVAT (open source)
- LabelImg (desktop)

### Step 2: Create data.yaml
Run this once:
```python
from detector import QRDetector
QRDetector.write_data_yaml("data", {0: "QR"})
```

### Step 3: Train the Model
```bash
python qr_detection/interface.py
```

**During Training You'll See:**
- Epoch 1/50: loss = 2.3, val_loss = 2.1
- Epoch 2/50: loss = 1.9, val_loss = 1.8
- ...continues...

Lower loss numbers = better model

### Step 4: Check Results
```
qr_detection/runs/detect/qr_detector_v1/
├── weights/
│   ├── best.pt            # ← Use this for inference!
│   └── last.pt
├── results.csv            # Training metrics
├── confusion_matrix.png   # Accuracy visualization
└── training_plots/        # Graphs and charts
```

### Step 5: Use the Trained Model
```python
from detector import QRDetector

# Load the best trained model
detector = QRDetector("qr_detection/runs/detect/qr_detector_v1/weights/best.pt")

# Detect QR codes
result = detector.predict_image("test_image.jpg")
detections = detector.extract_detections(result)

for det in detections:
    print(f"Found QR with {det['confidence']:.1%} confidence")
```

---

## 📈 Understanding Training Metrics

When training, you'll see several metrics:

| Metric | What It Means |
|--------|--------------|
| **Loss** | How wrong predictions are (lower is better) |
| **val_loss** | Loss on validation set |
| **mAP50** | Accuracy at 50% threshold (higher is better) |
| **Precision** | Of detected QR codes, how many were correct |
| **Recall** | Of all QR codes, how many were detected |

**Good Training:**
- Loss decreases over time
- val_loss follows loss (not too different)
- mAP50 increases over time

**Bad Training:**
- val_loss keeps increasing (overfitting)
- Loss becomes NaN (learning rate too high)
- mAP50 plateaus or decreases

---

## 🎯 Common Use Cases

### Use Case 1: Train Once, Deploy Forever
```python
# 1. Train (do this once)
detector = QRDetector("yolov8n.pt")
detector.train(data_yaml="data/data.yaml", epochs=50)

# 2. Load trained model (do this in production)
detector = QRDetector("qr_detection/runs/detect/qr_detector_v1/weights/best.pt")

# 3. Use it
result = detector.predict_image("drone_image.jpg")
```

### Use Case 2: Real-Time Video Detection
```python
import cv2
from detector import QRDetector

detector = QRDetector("qr_detection/runs/detect/qr_detector_v1/weights/best.pt")

cap = cv2.VideoCapture(0)  # Webcam
while True:
    ret, frame = cap.read()
    result = detector.predict_frame(frame)
    annotated = detector.annotate_frame(frame, result)
    
    cv2.imshow("QR Detection", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

### Use Case 3: Batch Process Images
```python
from pathlib import Path
from detector import QRDetector

detector = QRDetector("qr_detection/runs/detect/qr_detector_v1/weights/best.pt")

for image_file in Path("batch_images").glob("*.jpg"):
    result = detector.predict_image(str(image_file))
    detections = detector.extract_detections(result)
    print(f"{image_file.name}: {len(detections)} QR codes found")
```

---

## 🔍 Troubleshooting

### Model Not Training / CUDA Error
**Problem:** "CUDA out of memory" or no GPU detected

**Solution:**
1. Reduce batch size: `batch=8` (instead of 16)
2. Use smaller model: `yolov8n.pt` (instead of yolov8m)
3. Reduce image size: `imgsz=416` (instead of 640)

### Low Accuracy (mAP < 0.5)
**Problem:** Model isn't learning

**Solutions:**
1. Check labels are correct (not inverted/scaled wrong)
2. Use more training data (need at least 50-100 images)
3. Train longer: `epochs=100` (instead of 50)
4. Increase image size: `imgsz=800`

### Training Stops Mid-Way
**Problem:** Interrupted training

**Solution:** Rerun training - YOLO will use last checkpoint

### Detections Too Noisy
**Problem:** Too many false positives

**Solution:**
```python
detector = QRDetector("model.pt", conf_threshold=0.7)  # Higher threshold
```

---

## 📚 Output Files Explained

### weights/best.pt
- Your trained model
- ~100MB file
- Use this for inference after training
- Contains learned patterns for QR codes

### results.csv
- Training metrics for each epoch
- Can plot with Excel or Python

### confusion_matrix.png
- Shows detection accuracy
- Diagonal = correct detections
- Off-diagonal = mistakes

### results.png
- All training graphs in one image
- Loss curves, accuracy, precision, recall

---

## 🚀 Performance Tips

### Speed Up Training
- Use GPU (NVIDIA GPU recommended)
- Use smaller model: `yolov8n.pt`
- Reduce image size: `imgsz=416`
- Reduce batch size: `batch=8`

### Improve Accuracy
- Use more training data
- Use larger model: `yolov8m.pt`
- Increase training time: `epochs=100`
- Increase image size: `imgsz=800`

### Balance Performance
```python
# Fast and decent (recommended for real-time)
detector = QRDetector("yolov8n.pt")
detector.train(..., imgsz=640, epochs=50, batch=16)

# Slow but accurate (recommended for high accuracy)
detector = QRDetector("yolov8m.pt")
detector.train(..., imgsz=800, epochs=100, batch=32)
```

---

## 📖 Complete Example: Train and Deploy

```python
from detector import QRDetector
from pathlib import Path

# Step 1: Setup
print("Setting up...")
project_root = Path(".")
data_yaml = project_root / "data" / "data.yaml"

# Step 2: Create detector
print("Loading model...")
detector = QRDetector("yolov8n.pt")

# Step 3: Train
print("Training...")
detector.train(
    data_yaml=str(data_yaml),
    epochs=50,
    batch=16,
    imgsz=640,
    optimizer="auto",
    project="qr_detection/runs/detect",
    name="qr_detector_v1"
)

# Step 4: Validate
print("Validating...")
results = detector.validate(str(data_yaml))

# Step 5: Test on image
print("Testing on image...")
detector_trained = QRDetector("qr_detection/runs/detect/qr_detector_v1/weights/best.pt")
result = detector_trained.predict_image("test_image.jpg", save=True)
detections = detector_trained.extract_detections(result)

print(f"Found {len(detections)} QR codes!")
for i, det in enumerate(detections):
    print(f"  QR {i+1}: {det['confidence']:.1%} confidence at {det['center']}")
```

---

## 🔗 Summary

```
┌─────────────────────────────────────────┐
│   Training Pipeline                      │
├─────────────────────────────────────────┤
│ 1. detector.py (model & methods)        │
│ 2. interface.py (training script)       │
│ 3. data/ (training images & labels)     │
│ 4. ↓ Run training                       │
│ 5. ↓ Save results to runs/              │
│ 6. ↓ Get best.pt (trained model)        │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│   Inference Pipeline                    │
├─────────────────────────────────────────┤
│ 1. Load trained model (best.pt)         │
│ 2. Create QRDetector instance           │
│ 3. Predict on image/frame               │
│ 4. Extract detections                   │
│ 5. Use detections (coordinates, conf)   │
└─────────────────────────────────────────┘

