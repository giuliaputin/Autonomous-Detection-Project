"""
Main entry point for QR Code Detection System.

Three main interfaces are available:
1. CLI Interface (cli.py) - Model selection, batch processing, inference management
2. GUI Interface (gui.py) - Graphical interface with buttons
3. Documentation - View interface usage guide

See docs/INTERFACES.md for detailed usage documentation.
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Launch the QR detection main menu."""
    project_root = Path(__file__).resolve().parent
    qr_detection_main = project_root / "src" / "qr_detection" / "src" / "main.py"

    if qr_detection_main.exists():
        subprocess.run([sys.executable, str(qr_detection_main)], cwd=project_root)
    else:
        print("❌ Error: src/qr_detection/src/main.py not found!")
        sys.exit(1)


if __name__ == "__main__":
    main()
