"""
CLI for QR Code Detection Model Management and Inference

This module provides a command-line interface to:
- List and select trained models from the runs directory
- Run inference on images or directories
- Train new models
- Validate models on datasets
- View model statistics and performance metrics
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional
import json

from .detector import QRDetector


class QRDetectionCLI:
    """Command-line interface for QR code detection model management."""

    def __init__(self):
        """Initialize the CLI."""
        self.qr_detection_root = Path(__file__).resolve().parent.parent  # Go up to qr_detection folder
        self.project_root = self.qr_detection_root.parent.parent  # Go up 2 more levels to project root
        self.runs_dir = self.qr_detection_root / "runs" / "detect"
        self.data_yaml = self.qr_detection_root / "data" / "data.yaml"  # Data is at qr_detection level
        self.base_model_path = self.project_root / "yolov8n.pt"

    def get_model_training_info(self, run_folder: Path) -> dict:
        """
        Extract training information from args.yaml file.

        Parameters
        ----------
        run_folder : Path
            Path to the model run folder.

        Returns
        -------
        dict
            Training information.
        """
        args_file = run_folder / "args.yaml"
        info = {"epochs": "?", "batch": "?", "imgsz": "640", "dataset": "Custom QR Dataset"}

        if args_file.exists():
            try:
                with open(args_file) as f:
                    for line in f:
                        if "epochs:" in line:
                            info["epochs"] = line.split(":")[-1].strip()
                        elif "batch:" in line:
                            info["batch"] = line.split(":")[-1].strip()
                        elif "imgsz:" in line:
                            info["imgsz"] = line.split(":")[-1].strip()
                        elif "data:" in line:
                            dataset = line.split(":")[-1].strip()
                            if dataset:
                                info["dataset"] = dataset
            except Exception:
                pass

        return info

    def list_available_models(self) -> List[dict]:
        """
        Discover and list all available trained models.

        Returns
        -------
        list[dict]
            List of available models with paths and metadata.
        """
        models = []

        # Add base pretrained model
        models.append({
            "name": "yolov8n (base)",
            "path": str(self.base_model_path),
            "type": "base",
            "description": "YOLOv8 Nano pretrained on COCO",
            "epochs": "Pretrained",
            "batch": "N/A",
            "imgsz": "640",
            "dataset": "COCO Dataset",
        })

        # Scan runs directory for trained models
        if self.runs_dir.exists():
            for run_folder in sorted(self.runs_dir.iterdir()):
                if run_folder.is_dir():
                    weights_dir = run_folder / "weights"
                    best_weights = weights_dir / "best.pt" if weights_dir.exists() else None
                    last_weights = weights_dir / "last.pt" if weights_dir.exists() else None

                    training_info = self.get_model_training_info(run_folder)

                    if best_weights and best_weights.exists():
                        models.append({
                            "name": f"{run_folder.name} (best)",
                            "path": str(best_weights),
                            "type": "trained",
                            "description": f"Best weights from {run_folder.name}",
                            "run_folder": run_folder.name,
                            "epochs": training_info.get("epochs", "?"),
                            "batch": training_info.get("batch", "?"),
                            "imgsz": training_info.get("imgsz", "640"),
                            "dataset": training_info.get("dataset", "Custom"),
                        })

                    if last_weights and last_weights.exists():
                        models.append({
                            "name": f"{run_folder.name} (last)",
                            "path": str(last_weights),
                            "type": "trained",
                            "description": f"Last checkpoint from {run_folder.name}",
                            "run_folder": run_folder.name,
                            "epochs": training_info.get("epochs", "?"),
                            "batch": training_info.get("batch", "?"),
                            "imgsz": training_info.get("imgsz", "640"),
                            "dataset": training_info.get("dataset", "Custom"),
                        })

        return models

    def display_available_models(self) -> None:
        """Display all available models in a formatted table."""
        models = self.list_available_models()

        if not models:
            print("❌ No models found!")
            return

        print("\n" + "=" * 100)
        print("📋 AVAILABLE QR DETECTION MODELS")
        print("=" * 100)

        for idx, model in enumerate(models, 1):
            print(f"\n[{idx}] {model['name']}")
            print(f"    Type: {model['type']}")
            print(f"    Path: {model['path']}")
            print(f"    Description: {model['description']}")
            
            # Show training info if available
            if model.get("epochs") and model["epochs"] != "N/A":
                print(f"    Training Info:")
                print(f"      • Epochs: {model.get('epochs', '?')}")
                print(f"      • Batch Size: {model.get('batch', '?')}")
                print(f"      • Image Size: {model.get('imgsz', '640')}")
                print(f"      • Dataset: {model.get('dataset', '?')}")

        print("\n" + "=" * 100 + "\n")

    def select_model_interactive(self) -> Optional[str]:
        """
        Interactively select a model from available options.

        Returns
        -------
        str or None
            Path to selected model, or None if cancelled.
        """
        models = self.list_available_models()

        if not models:
            print("❌ No models found!")
            return None

        self.display_available_models()

        while True:
            try:
                choice = input("Select a model (enter number) or 'q' to quit: ").strip()

                if choice.lower() == 'q':
                    return None

                idx = int(choice) - 1
                if 0 <= idx < len(models):
                    selected = models[idx]
                    print(f"\n✅ Selected: {selected['name']}")
                    return selected['path']
                else:
                    print("❌ Invalid selection. Please try again.")
            except ValueError:
                print("❌ Invalid input. Please enter a number or 'q'.")

    def run_inference(self, model_path: str, image_path: str, save_results: bool = False) -> dict:
        """
        Run inference on an image using the selected model.

        Parameters
        ----------
        model_path : str
            Path to the model file.
        image_path : str
            Path to the image to process.
        save_results : bool
            Whether to save annotated images.

        Returns
        -------
        dict
            Detection results including count and details.
        """
        try:
            print(f"\n🔄 Loading model: {model_path}")
            detector = QRDetector(model_path)

            print(f"📸 Processing image: {image_path}")
            results = detector.predict_image(image_path, save=save_results)

            if not results:
                return {"success": False, "error": "No results returned from model"}

            result = results[0]
            detections = detector.extract_detections(result)

            output = {
                "success": True,
                "image_path": image_path,
                "model": model_path,
                "qr_count": len(detections),
                "detections": detections,
                "confidence_threshold": detector.conf_threshold,
            }

            # Display results
            print("\n" + "=" * 60)
            print("🎯 DETECTION RESULTS")
            print("=" * 60)
            print(f"QR Codes Detected: {len(detections)}")

            for idx, det in enumerate(detections, 1):
                x1, y1, x2, y2 = map(int, det["xyxy"])
                print(f"\n  QR Code #{idx}:")
                print(f"    Position: ({x1}, {y1}) to ({x2}, {y2})")
                print(f"    Size: {det['width']:.0f}x{det['height']:.0f} pixels")
                print(f"    Center: {det['center']}")
                print(f"    Confidence: {det['confidence']:.2%}")

            print("\n" + "=" * 60)

            if save_results:
                print("✅ Annotated image saved to results folder")

            return output

        except Exception as e:
            return {"success": False, "error": str(e)}

    def run_batch_inference(self, model_path: str, image_dir: str, save_results: bool = False) -> dict:
        """
        Run inference on all images in a directory.

        Parameters
        ----------
        model_path : str
            Path to the model file.
        image_dir : str
            Directory containing images.
        save_results : bool
            Whether to save annotated images.

        Returns
        -------
        dict
            Summary of results across all images.
        """
        image_path = Path(image_dir)

        if not image_path.is_dir():
            return {"success": False, "error": f"Directory not found: {image_dir}"}

        # Find all image files
        image_files = list(image_path.glob("*.jpg")) + list(image_path.glob("*.png"))

        if not image_files:
            return {"success": False, "error": f"No images found in {image_dir}"}

        print(f"\n🔄 Loading model: {model_path}")
        detector = QRDetector(model_path)

        print(f"📸 Processing {len(image_files)} images from {image_dir}\n")

        summary = {
            "success": True,
            "total_images": len(image_files),
            "total_qrs_detected": 0,
            "images_with_qrs": 0,
            "average_confidence": 0,
            "results": [],
        }

        total_confidence = 0
        confidence_count = 0

        for image_file in image_files:
            print(f"  Processing: {image_file.name}...", end=" ")
            results = detector.predict_image(str(image_file), save=save_results)

            if results:
                result = results[0]
                detections = detector.extract_detections(result)

                print(f"✅ {len(detections)} QR(s) found")

                summary["results"].append({
                    "image": image_file.name,
                    "qr_count": len(detections),
                    "detections": detections,
                })

                summary["total_qrs_detected"] += len(detections)
                if len(detections) > 0:
                    summary["images_with_qrs"] += 1
                    for det in detections:
                        total_confidence += det["confidence"]
                        confidence_count += 1
            else:
                print("⚠️  No results")

        if confidence_count > 0:
            summary["average_confidence"] = total_confidence / confidence_count

        # Display summary
        print("\n" + "=" * 60)
        print("📊 BATCH PROCESSING SUMMARY")
        print("=" * 60)
        print(f"Total Images Processed: {summary['total_images']}")
        print(f"Total QR Codes Detected: {summary['total_qrs_detected']}")
        print(f"Images with QR Codes: {summary['images_with_qrs']}")
        if confidence_count > 0:
            print(f"Average Confidence: {summary['average_confidence']:.2%}")
        print("=" * 60 + "\n")

        return summary


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="QR Code Detection Model CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive model selection and image inference
  python cli.py --infer image.jpg

  # Use specific model
  python cli.py --infer image.jpg --model ./runs/detect/qr_detector_v1/weights/best.pt

  # Batch process directory
  python cli.py --batch ./images/ --model ./runs/detect/qr_detector_v1/weights/best.pt

  # List available models
  python cli.py --list

  # Save annotated results
  python cli.py --infer image.jpg --save-results
        """,
    )

    parser.add_argument("--list", action="store_true", help="List all available models")
    parser.add_argument("--infer", type=str, help="Run inference on a single image")
    parser.add_argument("--batch", type=str, help="Run inference on all images in a directory")
    parser.add_argument("--model", type=str, help="Path to model (if not specified, will prompt)")
    parser.add_argument("--save-results", action="store_true", help="Save annotated images")

    args = parser.parse_args()

    cli = QRDetectionCLI()

    # Show available models
    if args.list:
        cli.display_available_models()
        return

    # Inference on single image
    if args.infer:
        # Select model if not specified
        model_path = args.model
        if not model_path:
            model_path = cli.select_model_interactive()
            if not model_path:
                print("❌ No model selected. Exiting.")
                return

        result = cli.run_inference(model_path, args.infer, args.save_results)

        if result["success"]:
            # Save result as JSON if requested
            result_json = json.dumps(result, indent=2, default=str)
            print("\n📄 JSON Output:")
            print(result_json)
        else:
            print(f"❌ Error: {result['error']}")
            sys.exit(1)

    # Batch inference
    elif args.batch:
        # Select model if not specified
        model_path = args.model
        if not model_path:
            model_path = cli.select_model_interactive()
            if not model_path:
                print("❌ No model selected. Exiting.")
                return

        summary = cli.run_batch_inference(model_path, args.batch, args.save_results)

        if summary["success"]:
            # Save summary as JSON
            summary_json = json.dumps(summary, indent=2, default=str)
            print("\n📄 JSON Summary:")
            print(summary_json)
        else:
            print(f"❌ Error: {summary['error']}")
            sys.exit(1)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
