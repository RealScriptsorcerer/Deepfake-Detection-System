from __future__ import annotations
from typing import List, Optional
import os


class DeepModelBackend:
    def __init__(self, model_dir: str | None = None) -> None:
        self.model_dir = model_dir or os.environ.get("MODEL_DIR", "/models")
        self.available: bool = False

    def load(self) -> None:
        raise NotImplementedError

    def predict_video_crops(self, crops_rgb: List, fps: float) -> Optional[float]:
        return None

    def predict_audio(self, audio_samples, sample_rate: int) -> Optional[float]:
        return None

