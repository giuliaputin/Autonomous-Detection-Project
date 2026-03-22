"""
Image Processor for QR Code Detection

A simple interface for uploading images and detecting QR codes.
This module provides functions to:
- Load and process images
- Run detection on images
- Save results (annotated images, JSON data)
- Provide summary statistics about detections
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from datetime import datetime

from .detector import QRDetector


class QRImageProcessor:
    """Process images for QR code detection."""

    def __init__(self, model_path: str = "yolov8n.pt", conf_threshold: float = 0.5):
        """
        Initialize the image processor with a model.

        Parameters
        ----------
        model_path : str
            Path to the YOLO model to use.
        conf_threshold : float
            Minimum confidence threshold for detections.
        """
        self.detector = QRDetector(model_path, conf_threshold=conf_threshold)
        self.model_path = model_path
        self.last_results = None
        self.last_image_path = None

    def process_image(self, image_path: str) -> Dict:
        """
        Process a single image and detect QR codes.

        Parameters
        ----------
        image_path : str
            Path to the image file.

        Returns
        -------
        dict
            Detection results with QR count, locations, and confidence scores.

        Raises
        ------
        FileNotFoundError
            If the image file does not exist.
        """
        image_file = Path(image_path)

        if not image_file.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        if not image_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            raise ValueError(f"Unsupported image format: {image_file.suffix}")

        # Run inference
        results = self.detector.predict_image(str(image_file))

        if not results:
            raise RuntimeError("Failed to process image with model")

        result = results[0]
        detections = self.detector.extract_detections(result)

        # Store for later use
        self.last_results = detections
        self.last_image_path = str(image_file)

        # Create output
        output = {
            "timestamp": datetime.now().isoformat(),
            "image_path": str(image_file),
            "image_name": image_file.name,
            "qr_count": len(detections),
            "model_used": self.model_path,
            "confidence_threshold": self.detector.conf_threshold,
            "detections": detections,
        }

        return output

    def get_summary(self, results: Dict) -> Dict:
        """
        Get a summary of detection results.

        Parameters
        ----------
        results : dict
            Detection results from process_image().

        Returns
        -------
        dict
            Summary statistics.
        """
        detections = results.get("detections", [])

        if not detections:
            return {
                "total_qrs": 0,
                "high_confidence": 0,
                "medium_confidence": 0,
                "low_confidence": 0,
                "average_confidence": 0,
                "max_confidence": 0,
                "min_confidence": 0,
            }

        confidences = [d["confidence"] for d in detections]

        return {
            "total_qrs": len(detections),
            "high_confidence": sum(1 for c in confidences if c >= 0.8),
            "medium_confidence": sum(1 for c in confidences if 0.6 <= c < 0.8),
            "low_confidence": sum(1 for c in confidences if c < 0.6),
            "average_confidence": sum(confidences) / len(confidences),
            "max_confidence": max(confidences),
            "min_confidence": min(confidences),
        }

    def print_results(self, results: Dict) -> None:
        """
        Print results in a human-readable format.

        Parameters
        ----------
        results : dict
            Detection results from process_image().
        """
        print("\n" + "=" * 70)
        print("🎯 QR CODE DETECTION RESULTS")
        print("=" * 70)
        print(f"Time: {results['timestamp']}")
        print(f"Image: {results['image_name']}")
        print(f"Model: {results['model_used']}")
        print(f"Confidence Threshold: {results['confidence_threshold']}")

        detections = results.get("detections", [])
        qr_count = results.get("qr_count", 0)

        print(f"\n📊 QR Codes Found: {qr_count}")

        if qr_count == 0:
            print("   No QR codes detected in this image.")
        else:
            summary = self.get_summary(results)
            print(f"   High Confidence (≥80%): {summary['high_confidence']}")
            print(f"   Medium Confidence (60-80%): {summary['medium_confidence']}")
            print(f"   Low Confidence (<60%): {summary['low_confidence']}")
            print(f"   Average Confidence: {summary['average_confidence']:.2%}")

            print("\n📍 QR Code Locations:")
            for idx, det in enumerate(detections, 1):
                x1, y1, x2, y2 = map(int, det["xyxy"])
                cx, cy = map(int, det["center"])
                print(f"\n   QR #{idx}:")
                print(f"      Bounding Box: X: [{x1}, {x2}], Y: [{y1}, {y2}]")
                print(f"      Center: ({cx}, {cy})")
                print(f"      Size: {det['width']:.0f} × {det['height']:.0f} pixels")
                print(f"      Confidence: {det['confidence']:.2%}")

        print("=" * 70 + "\n")

    def save_annotated_image(self, image_path: str, results: Dict, output_path: Optional[str] = None) -> str:
        """
        Save an annotated version of the image with QR detections drawn.

        Parameters
        ----------
        image_path : str
            Path to the original image.
        results : dict
            Detection results from process_image().
        output_path : str, optional
            Path for the output file. If None, saves with _qr_detected suffix.

        Returns
        -------
        str
            Path to the saved annotated image.
        """
        # Read original image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        # Run detection again to get the annotated version
        detection_result = self.detector.predict_image(image_path)
        annotated_frame = self.detector.annotate_frame(image, detection_result[0])

        # Determine output path
        if output_path is None:
            image_file = Path(image_path)
            output_path = image_file.parent / f"{image_file.stem}_qr_detected{image_file.suffix}"

        # Save annotated image
        cv2.imwrite(str(output_path), annotated_frame)
        return str(output_path)

    def save_results_json(self, results: Dict, output_path: Optional[str] = None) -> str:
        """
        Save detection results as JSON.

        Parameters
        ----------
        results : dict
            Detection results from process_image().
        output_path : str, optional
            Path for JSON file. If None, uses timestamp-based name.

        Returns
        -------
        str
            Path to saved JSON file.
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"qr_detection_results_{timestamp}.json"

        with open(output_path, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            results_serializable = {k: v for k, v in results.items()}
            results_serializable["detections"] = [
                {
                    "class_id": int(d["class_id"]),
                    "class_name": str(d["class_name"]),
                    "confidence": float(d["confidence"]),
                    "xyxy": [float(x) for x in d["xyxy"]],
                    "center": [float(x) for x in d["center"]],
                    "width": float(d["width"]),
                    "height": float(d["height"]),
                }
                for d in results_serializable.get("detections", [])
            ]

            json.dump(results_serializable, f, indent=2)

        return str(output_path)

    def create_summary_report(
        self, image_paths: List[str], output_dir: Optional[str] = None
    ) -> Dict:
        """
        Process multiple images and create a summary report.

        Parameters
        ----------
        image_paths : list[str]
            List of image file paths to process.
        output_dir : str, optional
            Directory to save results. If None, uses current directory.

        Returns
        -------
        dict
            Summary report with statistics across all images.
        """
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        report = {
            "timestamp": datetime.now().isoformat(),
            "total_images": len(image_paths),
            "total_qrs_detected": 0,
            "images_with_qrs": 0,
            "average_qrs_per_image": 0,
            "average_confidence": 0,
            "model_used": self.model_path,
            "image_results": [],
        }

        total_confidence = 0
        confidence_count = 0

        print(f"\n📸 Processing {len(image_paths)} images...\n")

        for idx, image_path in enumerate(image_paths, 1):
            try:
                print(f"  [{idx}/{len(image_paths)}] Processing {Path(image_path).name}...", end=" ")
                results = self.process_image(image_path)

                detections = results["detections"]
                qr_count = results["qr_count"]

                print(f"✅ {qr_count} QR(s) found")

                # Update report
                report["total_qrs_detected"] += qr_count
                if qr_count > 0:
                    report["images_with_qrs"] += 1

                # Collect confidence scores
                for det in detections:
                    total_confidence += det["confidence"]
                    confidence_count += 1

                report["image_results"].append({
                    "image": Path(image_path).name,
                    "qr_count": qr_count,
                    "summary": self.get_summary(results),
                })

            except Exception as e:
                print(f"❌ Error: {str(e)}")

        # Calculate averages
        if len(image_paths) > 0:
            report["average_qrs_per_image"] = report["total_qrs_detected"] / len(image_paths)

        if confidence_count > 0:
            report["average_confidence"] = total_confidence / confidence_count

        # Display and save report
        print("\n" + "=" * 70)
        print("📊 BATCH PROCESSING REPORT")
        print("=" * 70)
        print(f"Total Images: {report['total_images']}")
        print(f"Total QR Codes: {report['total_qrs_detected']}")
        print(f"Images with QR Codes: {report['images_with_qrs']}")
        print(f"Average QRs per Image: {report['average_qrs_per_image']:.2f}")
        if confidence_count > 0:
            print(f"Average Confidence: {report['average_confidence']:.2%}")
        print("=" * 70 + "\n")

        # Save report
        if output_dir:
            report_path = Path(output_dir) / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"✅ Report saved to: {report_path}")

        return report


def quick_process(image_path: str, model_path: str = "yolov8n.pt", save_annotated: bool = False) -> Dict:
    """
    Quick function to process an image and get QR detection results.

    This is the simplest way to use the processor:

    ```python
    results = quick_process("my_image.jpg")
    print(f"Found {results['qr_count']} QR codes")
    ```

    Parameters
    ----------
    image_path : str
        Path to the image to process.
    model_path : str
        Path to the model to use.
    save_annotated : bool
        Whether to save annotated image with detections marked.

    Returns
    -------
    dict
        Detection results.
    """
    processor = QRImageProcessor(model_path)
    results = processor.process_image(image_path)
    processor.print_results(results)

    if save_annotated:
        output_path = processor.save_annotated_image(image_path, results)
        print(f"✅ Annotated image saved to: {output_path}")

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python image_processor.py <image_path> [--model MODEL_PATH] [--save-annotated]")
        print("\nExample:")
        print("  python image_processor.py photo.jpg")
        print("  python image_processor.py photo.jpg --model ./runs/detect/qr_detector_v1/weights/best.pt --save-annotated")
        sys.exit(1)

    image_path = sys.argv[1]
    model_path = "yolov8n.pt"
    save_annotated = False

    # Parse optional arguments
    if "--model" in sys.argv:
        idx = sys.argv.index("--model")
        if idx + 1 < len(sys.argv):
            model_path = sys.argv[idx + 1]

    if "--save-annotated" in sys.argv:
        save_annotated = True

    try:
        quick_process(image_path, model_path, save_annotated)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
