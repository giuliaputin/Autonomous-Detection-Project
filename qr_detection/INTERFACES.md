# QR Code Detection - Complete Interfaces Guide

## What is QR Code Detection?

This project uses a trained **YOLOv8 machine learning model** to automatically find and locate QR codes in images. The model:
- 📍 **Detects** where QR codes are located (coordinates)
- 🔢 **Counts** how many QR codes are in the image
- 📊 **Shows confidence** (how sure the model is about each detection)
- 📁 **Works with** individual images or entire folders

---

## 🚀 Quick Start (Choose Your Way)

### **Option 1: GUI Interface (Easiest) ✨**
Point and click! No terminal commands needed.

```bash
python main.py
# Choose: [2] GUI Interface
```

**What you can do:**
- ✅ See all available models with training info
- ✅ Upload and analyze single images
- ✅ Process entire folders at once
- ✅ View detailed results

---

### **Option 2: Command Line (CLI)**
Use commands in your terminal.

```bash
# See all available models with training info
python -m qr_detection.cli --list

# Analyze a single image (interactive)
python -m qr_detection.cli --infer image.jpg

# Process entire folder
python -m qr_detection.cli --batch ./my_images/
```

---

### **Option 3: Python Code**
Use the model in your own Python scripts.

```python
from qr_detection import quick_process

# Simple one-liner to detect QR codes
results = quick_process("image.jpg")

print(f"Found {results['qr_count']} QR codes")
for detection in results['detections']:
    print(f"  - Position: {detection['xyxy']}")
    print(f"  - Confidence: {detection['confidence']:.1%}")
```

---

## 📋 Interface Details

### **1. GUI Interface** (`gui.py`) - Easiest to Use

A graphical interface with buttons for common tasks.

**Features:**
- 📂 File browser to select images
- 🎯 Click buttons to run detection
- 📊 View results in scrollable windows
- 📈 Batch process multiple images

**Launch it:**
```bash
python qr_detection/gui.py
# Or through main.py menu: Option 2
```

---

### **2. CLI Interface** (`cli.py`) - Powerful & Flexible

A command-line tool for advanced users who like terminal commands.

**List models with training info:**
```bash
python -m qr_detection.cli --list
```

Output shows:
- Model name and version
- **Training Information:**
  - Epochs (how many times model trained)
  - Batch size (images processed at once)
  - Image size (resolution used)
  - Dataset (what data was used)

**Process single image:**
```bash
# Interactive - choose model from menu
python -m qr_detection.cli --infer image.jpg

# Specify exact model
python -m qr_detection.cli --infer image.jpg --model ./runs/detect/qr_detector_v17/weights/best.pt

# Save results to disk
python -m qr_detection.cli --infer image.jpg --save-results
```

**Batch process directory:**
```bash
# Process all images in folder
python -m qr_detection.cli --batch ./images/

# With specific model and save results
python -m qr_detection.cli --batch ./images/ --model ./runs/detect/qr_detector_v17/weights/best.pt --save-results
```

**Example output:**
```
==============================================================
🎯 DETECTION RESULTS
==============================================================
QR Codes Detected: 3

  QR Code #1:
    Position: (123, 456) to (234, 567)
    Size: 111x111 pixels
    Confidence: 95.42%

  QR Code #2:
    Position: (500, 200) to (645, 345)
    Size: 145x145 pixels
    Confidence: 88.76%
==============================================================
```

---

### **3. Python API** (`image_processor.py`) - For Developers

Import and use the model in your own Python code.

**Simple approach:**
```python
from qr_detection import quick_process

# Detect QR codes in one line
results = quick_process("my_image.jpg", save_annotated=True)
print(f"Found: {results['qr_count']} QR codes")
```

**More control with QRImageProcessor class:**
```python
from qr_detection import QRImageProcessor

# Initialize processor with specific model
processor = QRImageProcessor(
    model_path="./runs/detect/qr_detector_v17/weights/best.pt"
)

# Process image
results = processor.process_image("photo.jpg")

# Print results
processor.print_results(results)

# Save annotated image (shows QR boxes)
output_path = processor.save_annotated_image("photo.jpg", results)
print(f"Saved to: {output_path}")

# Save as JSON
json_path = processor.save_results_json(results, "output.json")

# Get statistics
summary = processor.get_summary(results)
print(f"Average confidence: {summary['average_confidence']:.1%}")
```

**Process many images:**
```python
from qr_detection import QRImageProcessor
from pathlib import Path

processor = QRImageProcessor()

# Get all images
images = list(Path("./images").glob("*.jpg"))

# Process all with summary report
report = processor.create_summary_report(
    [str(img) for img in images],
    output_dir="./results"
)

print(f"Total QR codes: {report['total_qrs_detected']}")
print(f"Per image average: {report['average_qrs_per_image']:.1f}")
```

**Result structure (what data you get back):**
```json
{
  "timestamp": "2024-03-22T10:30:45",
  "image_path": "photo.jpg",
  "qr_count": 3,
  "model_used": "qr_detector_v17",
  "detections": [
    {
      "class_name": "QR",
      "confidence": 0.95,
      "xyxy": [123, 456, 234, 567],
      "center": [178.5, 511.5],
      "width": 111,
      "height": 111
    }
  ]
}
```

---

## 🔍 Available Models

Models are automatically discovered in: `qr_detection/runs/detect/`

**Base model (always available):**
- `yolov8n.pt` - Pre-trained on public data (6.2 MB)

**Trained models:**
- `qr_detector_v1` through `qr_detector_v17` - Custom trained on your data
- Each has `best.pt` (best performance) and `last.pt` (final checkpoint)

**How to see all models:**
```bash
# CLI method
python -m qr_detection.cli --list

# Python method
from qr_detection import QRDetectionCLI
cli = QRDetectionCLI()
cli.display_available_models()
```

---

## 📊 Understanding Model Training Info

When you use `--list`, you see:
- **Epochs**: Number of times model trained on dataset
  - Higher = more training (usually better but slower)
- **Batch Size**: How many images trained at once
  - Affects memory usage and training speed
- **Image Size**: Resolution used (640 = 640x640 pixels)
- **Dataset**: What data was used for training

**Example:**
```
[1] qr_detector_v17 (best)
    Training Info:
      • Epochs: 100
      • Batch Size: 16
      • Image Size: 640
      • Dataset: Custom QR Dataset
```

---

## ❓ Common Tasks

### Detect QR codes in one image
```bash
# Using CLI
python -m qr_detection.cli --infer my_photo.jpg

# Using Python
from qr_detection import quick_process
results = quick_process("my_photo.jpg")
```

### Detect QR codes in many images
```bash
# Using CLI (entire folder)
python -m qr_detection.cli --batch ./my_images/

# Using Python
from qr_detection import QRImageProcessor
processor = QRImageProcessor()
processor.create_summary_report(
    ["image1.jpg", "image2.jpg", "image3.jpg"],
    output_dir="./results"
)
```

### Use best model vs latest model
```bash
# Best model (recommended)
python -m qr_detection.cli --infer image.jpg --model ./runs/detect/qr_detector_v17/weights/best.pt

# Latest model
python -m qr_detection.cli --infer image.jpg --model ./runs/detect/qr_detector_v17/weights/last.pt
```

### Save results
```bash
# CLI automatically saves with --save-results
python -m qr_detection.cli --infer image.jpg --save-results

# Python - save annotated image
processor.save_annotated_image("image.jpg", results)

# Python - save as JSON
processor.save_results_json(results, "detections.json")
```

---

## ⚙️ Confidence Threshold

The model assigns a confidence score (0-100%) to each detection. You can filter:

- **Default**: 50% confidence or higher
- **In Python code**:
  ```python
  processor = QRImageProcessor(conf_threshold=0.8)  # Only 80%+ confidence
  ```

Higher threshold = fewer but more confident detections

---

## 🆘 Troubleshooting

**"No models found"**
- Check that trained models exist in: `qr_detection/runs/detect/`
- The base model `yolov8n.pt` should always be available

**"CUDA/GPU errors"**
- Model will fall back to CPU automatically
- CPU is slower but works everywhere

**"File not found"**
- Use full paths or relative from current directory
- Example: `./data/my_image.jpg` not just `my_image.jpg`

**GUI window won't start**
- Use terminal command instead: `python -m qr_detection.cli --list`
- Check that tkinter is installed (should be with Python)

---

## Examples

### Full Workflow

1. **List models**:
   ```bash
   python cli.py --list
   ```

2. **Test on single image**:
   ```bash
   python cli.py --infer test_image.jpg
   ```

3. **Batch process and save**:
   ```bash
   python cli.py --batch ./data/images/test/ --save-results
   ```

4. **Use in automation**:
   ```python
   from image_processor import quick_process
   
   for image_file in Path("./images").glob("*.jpg"):
       results = quick_process(str(image_file))
       if results['qr_count'] > 0:
           print(f"{image_file.name}: {results['qr_count']} QR codes")
   ```

---

## Performance Notes

- **Processing Speed**: ~100-200ms per image (depends on GPU/CPU)
- **Accuracy**: ~95%+ confidence on clear QR codes
- **Model Size**: ~6.2MB for yolov8n

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Model not found | Check path in `runs/detect/` or use `--list` to see available models |
| Image not recognized | Ensure image is JPG, PNG, BMP, or TIFF format |
| Low confidence | Try adjusting confidence threshold or using a better-trained model |
| GPU not used | Install CUDA-enabled PyTorch (see requirements.txt) |

