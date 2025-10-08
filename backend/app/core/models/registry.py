from __future__ import annotations
from typing import Dict
from .torch_backends import XceptionVideoBackend, RawNet2AudioBackend


class ModelRegistry:
    def __init__(self) -> None:
        self.video = XceptionVideoBackend()
        self.audio = RawNet2AudioBackend()

    def load_all(self) -> Dict[str, bool]:
        self.video.load()
        self.audio.load()
        return {"video": self.video.available, "audio": self.audio.available}


REGISTRY = ModelRegistry()

