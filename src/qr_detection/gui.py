# -*- coding: utf-8 -*-
"""
GUI Interface for QR Code Detection

A graphical interface with clickable buttons instead of terminal input.
Uses tkinter (built-in to Python, no extra dependencies).
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import subprocess
import sys
from pathlib import Path
import threading


class QRDetectionGUI:
    """Graphical user interface for QR code detection."""

    def __init__(self, root):
        """Initialize the GUI."""
        self.root = root
        self.root.title("🎯 QR Code Detection System")
        self.root.geometry("750x650")
        self.root.resizable(True, True)

        self.project_root = Path(__file__).resolve().parent.parent.parent  # Go up 3 levels: qr_detection -> src -> project root
        self.create_widgets()

    def create_widgets(self):
        """Create GUI widgets."""
        # Header Frame
        header_frame = ttk.Frame(self.root)
        header_frame.pack(fill=tk.X, padx=20, pady=20)

        title = ttk.Label(
            header_frame,
            text="🎯 QR CODE DETECTION SYSTEM",
            font=("Arial", 20, "bold"),
        )
        title.pack()

        subtitle = ttk.Label(
            header_frame,
            text="Click a button to get started",
            font=("Arial", 12),
        )
        subtitle.pack(pady=(10, 0))

        # Separator
        ttk.Separator(self.root, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=20)

        # Main content frame
        content_frame = ttk.Frame(self.root)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Button 1: List Models
        btn1 = ttk.Button(
            content_frame,
            text="📋 LIST AVAILABLE MODELS\nSee all trained models with training info",
            command=self.list_models,
        )
        btn1.pack(fill=tk.BOTH, expand=True, pady=8)

        # Button 2: Detect QR in Image
        btn2 = ttk.Button(
            content_frame,
            text="🖼️  DETECT QR IN IMAGE\nProcess a single image to find QR codes",
            command=self.infer_image,
        )
        btn2.pack(fill=tk.BOTH, expand=True, pady=8)

        # Button 3: Batch Process
        btn3 = ttk.Button(
            content_frame,
            text="📁 BATCH PROCESS FOLDER\nProcess multiple images at once",
            command=self.batch_process,
        )
        btn3.pack(fill=tk.BOTH, expand=True, pady=8)

        # Button 4: Documentation
        btn4 = ttk.Button(
            content_frame,
            text="📖 DOCUMENTATION\nView detailed usage guide",
            command=self.show_documentation,
        )
        btn4.pack(fill=tk.BOTH, expand=True, pady=8)

        # Button 5: Exit
        btn5 = ttk.Button(
            content_frame,
            text="❌ EXIT",
            command=self.root.quit,
        )
        btn5.pack(fill=tk.BOTH, expand=True, pady=8)

        # Separator
        ttk.Separator(self.root, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=20)

        # Status bar
        self.status = ttk.Label(
            self.root,
            text="Ready",
            font=("Arial", 10),
        )
        self.status.pack(fill=tk.X, padx=20, pady=10)

    def update_status(self, message):
        """Update status bar and refresh."""
        self.status.config(text=message)
        self.root.update()

    def show_output_window(self, title, content):
        """Show output in a new scrollable window."""
        window = tk.Toplevel(self.root)
        window.title(title)
        window.geometry("800x700")

        # Header
        header = ttk.Frame(window)
        header.pack(fill=tk.X, padx=10, pady=10)
        ttk.Label(header, text=title, font=("Arial", 14, "bold")).pack()

        # Output text
        text_widget = scrolledtext.ScrolledText(
            window, height=30, width=100, font=("Courier", 10)
        )
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        text_widget.insert(tk.END, content)
        text_widget.config(state=tk.DISABLED)

        # Close button
        ttk.Button(window, text="Close", command=window.destroy).pack(pady=10)

    def run_command(self, *args):
        """Run CLI command in background thread."""
        def run():
            try:
                self.update_status("Processing... please wait")
                result = subprocess.run(
                    [sys.executable, "-m", "src.qr_detection.cli"] + list(args),
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )

                if result.returncode == 0:
                    output = result.stdout
                else:
                    output = result.stdout + "\n--- ERRORS ---\n" + result.stderr

                self.show_output_window("Results", output)
                self.update_status("✅ Completed successfully")

            except subprocess.TimeoutExpired:
                messagebox.showerror("Error", "Operation timed out")
                self.update_status("❌ Timed out")
            except Exception as e:
                messagebox.showerror("Error", f"Error: {str(e)}")
                self.update_status(f"❌ Error: {str(e)}")

        # Run in background to keep GUI responsive
        thread = threading.Thread(target=run, daemon=True)
        thread.start()

    def list_models(self):
        """List available models with training info."""
        self.run_command("--list")

    def infer_image(self):
        """Detect QR codes in a single image."""
        file_path = filedialog.askopenfilename(
            title="Select Image to Process",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("All Files", "*.*")],
        )

        if file_path:
            self.update_status(f"Processing: {Path(file_path).name}...")
            self.run_command("--infer", file_path)

    def batch_process(self):
        """Batch process all images in a directory."""
        folder_path = filedialog.askdirectory(title="Select Folder with Images")

        if folder_path:
            self.update_status(f"Batch processing: {Path(folder_path).name}...")
            self.run_command("--batch", folder_path)

    def show_documentation(self):
        """Display documentation."""
        doc_path = self.project_root / "src" / "qr_detection" / "INTERFACES.md"

        if not doc_path.exists():
            messagebox.showerror("Error", f"Documentation not found:\n{doc_path}")
            return

        try:
            with open(doc_path, "r", encoding="utf-8") as f:
                doc_content = f.read()

            self.show_output_window("📖 Documentation", doc_content)
            self.update_status("✅ Documentation displayed")

        except Exception as e:
            messagebox.showerror("Error", f"Could not load documentation:\n{str(e)}")


def main():
    """Main entry point for GUI application."""
    root = tk.Tk()
    app = QRDetectionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
