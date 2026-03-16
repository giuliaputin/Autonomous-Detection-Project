"""
detector.py

Neural network model and detector wrapper for smiley detection.

First version:
- Binary classifier: smiley vs not_smiley
- Uses a small CNN in PyTorch
- Can be trained later with a dataset of labeled images

Expected usage:
    detector = Detector(model_path="models/smiley_cnn.pth")
    result = detector.predict(frame)

Output format:
{
    "found": True/False,
    "confidence": float,
    "label": "smiley" or "not_smiley"
}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn


@dataclass
class DetectorConfig:
    image_size: int = 64
    threshold: float = 0.7
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class SmileyCNN(nn.Module):
    """
    Small convolutional neural network for binary classification.

    Input:
        [batch, 3, 64, 64]

    Output:
        logits of shape [batch, 2]
        class 0 -> not_smiley
        class 1 -> smiley
    """

    def __init__(self) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 64 -> 32

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 32 -> 16

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),   # 16 -> 8
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x


class Detector:
    """
    Wrapper around the neural network model.

    This first version classifies the whole frame.
    Later, you can extend it to produce bounding boxes and centers.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        config: Optional[DetectorConfig] = None,
    ) -> None:
        self.config = config or DetectorConfig()
        self.device = torch.device(self.config.device)

        self.model = SmileyCNN().to(self.device)
        self.model.eval()

        if model_path is not None:
            self.load(model_path)

    def load(self, model_path: str) -> None:
        """Load trained weights from disk."""
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def preprocess(self, frame: np.ndarray) -> torch.Tensor:
        """
        Convert OpenCV frame to model input tensor.

        Steps:
        - BGR -> RGB
        - resize to fixed input size
        - normalize to [0,1]
        - HWC -> CHW
        - add batch dimension
        """
        if frame is None:
            raise ValueError("Input frame is None.")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.config.image_size, self.config.image_size))
        normalized = resized.astype(np.float32) / 255.0
        chw = np.transpose(normalized, (2, 0, 1))
        tensor = torch.tensor(chw, dtype=torch.float32).unsqueeze(0)
        return tensor.to(self.device)

    @torch.no_grad()
    def predict(self, frame: np.ndarray) -> Dict[str, object]:
        """
        Predict whether the target smiley is present in the frame.

        Returns a dictionary compatible with your future pipeline.
        """
        x = self.preprocess(frame)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)[0]

        not_smiley_prob = float(probs[0].item())
        smiley_prob = float(probs[1].item())

        found = smiley_prob >= self.config.threshold

        return {
            "found": found,
            "confidence": smiley_prob,
            "label": "smiley" if found else "not_smiley",
            "raw_probs": {
                "not_smiley": not_smiley_prob,
                "smiley": smiley_prob,
            },
        }