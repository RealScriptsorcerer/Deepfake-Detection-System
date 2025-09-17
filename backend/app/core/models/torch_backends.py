from __future__ import annotations
from typing import List, Optional
import os

try:
    import torch
    import torchvision
except Exception:
    torch = None

from .base import DeepModelBackend


class XceptionVideoBackend(DeepModelBackend):
    def load(self) -> None:
        if torch is None:
            self.available = False
            return
        # Placeholder: assume weights exist at MODEL_DIR/xception_video.pt
        weights = os.path.join(self.model_dir, "xception_video.pt")
        self.available = os.path.exists(weights)

    def predict_video_crops(self, crops_rgb: List, fps: float) -> Optional[float]:
        if not self.available or torch is None:
            return None
        # Placeholder scoring: length-based proxy
        return min(1.0, 0.5 + 0.5 * (len(crops_rgb) / 300.0))


class RawNet2AudioBackend(DeepModelBackend):
    def load(self) -> None:
        if torch is None:
            self.available = False
            return
        weights = os.path.join(self.model_dir, "rawnet2_audio.pt")
        self.available = os.path.exists(weights)

    def predict_audio(self, audio_samples, sample_rate: int) -> Optional[float]:
        if not self.available or torch is None:
            return None
        # Placeholder scoring: simple energy proxy
        import numpy as np
        energy = float(np.mean(np.square(audio_samples)))
        return float(max(0.0, min(1.0, 0.3 + energy)))

