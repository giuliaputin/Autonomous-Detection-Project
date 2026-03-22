# -*- coding: utf-8 -*-
"""
QR Code Detection - Welcome & Interface Guide

This is the main entry point for the QR Code Detection system.
Choose between two powerful interfaces:

1. CLI Interface - Command line for model selection and batch processing
2. GUI Interface - Graphical interface with buttons
"""

import sys
import subprocess
import io
from pathlib import Path

# Fix Unicode encoding on Windows
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def print_welcome():
    """Display welcome banner."""
    print("\n" + "=" * 70)
    print("🎯 QR CODE DETECTION SYSTEM")
    print("=" * 70)
    print("\nWelcome! Choose how you'd like to work:\n")


def print_options():
    """Display available options."""
    options = {
        "1": {
            "label": "CLI Interface",
            "description": "List models, batch process, manage inference",
            "command": "src.qr_detection.cli",
            "examples": [
                "List available models: python -m src.qr_detection.cli --list",
                "Detect QR in image: python -m src.qr_detection.cli --infer photo.jpg",
                "Batch process folder: python -m src.qr_detection.cli --batch ./images/",
            ],
        },
        "2": {
            "label": "GUI Interface",
            "description": "Graphical interface with clickable buttons",
            "command": "src.qr_detection.gui",
            "examples": [
                "Launch GUI: python -m src.qr_detection.gui",
                "Click buttons to interact with detection system",
            ],
        },
        "3": {
            "label": "Documentation",
            "description": "View detailed interface documentation",
            "command": "src/qr_detection/INTERFACES.md",
        },
        "4": {
            "label": "Exit",
            "description": "Close this menu",
        },
    }

    for key, opt in options.items():
        print(f"  [{key}] {opt['label']}")
        print(f"      {opt['description']}\n")

    return options


def run_interactive_menu():
    """Run interactive menu."""
    project_root = Path(__file__).resolve().parent

    while True:
        print_welcome()
        options = print_options()

        choice = input("Enter your choice [1-4]: ").strip()

        if choice == "1":
            print("\n✅ Launching CLI Interface...\n")
            subprocess.run(
                [sys.executable, "-m", "src.qr_detection.cli", "--list"],
                cwd=project_root,
            )
            print("\n📌 Next Steps:")
            print("   • Detect QR in image: python -m src.qr_detection.cli --infer photo.jpg")
            print("   • Batch process folder: python -m src.qr_detection.cli --batch ./images/")
            print("   • For more help: python -m src.qr_detection.cli --help\n")

        elif choice == "2":
            print("\n✅ Launching GUI Interface...\n")
            subprocess.run(
                [sys.executable, "-m", "src.qr_detection.gui"],
                cwd=project_root,
            )
            print("\n📌 GUI Interface closed\n")

        elif choice == "3":
            doc_path = project_root / "src" / "qr_detection" / "INTERFACES.md"
            if doc_path.exists():
                print(f"\n📖 Opening {doc_path}...\n")
                with open(doc_path) as f:
                    print(f.read())
            else:
                print(f"❌ Documentation not found: {doc_path}\n")

        elif choice == "4":
            print("\n👋 Goodbye!\n")
            break

        else:
            print("❌ Invalid choice. Please try again.\n")


def main():
    """Main entry point - can be used as script or from Python."""
    if len(sys.argv) > 1:
        # If arguments provided, pass to CLI
        subprocess.run([sys.executable, "-m", "src.qr_detection.cli"] + sys.argv[1:])
    else:
        # Otherwise show interactive menu
        try:
            run_interactive_menu()
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!\n")
            sys.exit(0)


if __name__ == "__main__":
    main()
