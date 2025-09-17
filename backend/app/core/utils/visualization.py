from typing import List
import numpy as np
import matplotlib.pyplot as plt
from .common import figure_to_base64_png
import cv2
import base64


def plot_video_suspicion_timeline(per_frame_scores: List[float]) -> str:
    if not per_frame_scores:
        return ""
    x = np.arange(len(per_frame_scores))
    y = np.array(per_frame_scores, dtype=np.float32)
    fig, ax = plt.subplots(figsize=(8, 2.5))
    ax.plot(x, y, color="#d62728", linewidth=1.5)
    ax.fill_between(x, y, color="#ff9896", alpha=0.3)
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Suspicion")
    ax.set_title("Per-frame Suspicion Timeline")
    b64 = figure_to_base64_png(fig)
    return b64


def overlay_heatmap(frame_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    heatmap_norm = (heatmap - heatmap.min()) / (heatmap.ptp() + 1e-8)
    heatmap_color = cv2.applyColorMap((heatmap_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = ((1 - alpha) * frame_rgb.astype(np.float32) + alpha * heatmap_color.astype(np.float32)).astype(np.uint8)
    return overlay


def encode_image_rgb_base64_png(image_rgb: np.ndarray) -> str:
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    success, buf = cv2.imencode(".png", image_bgr)
    if not success:
        return ""
    return base64.b64encode(buf.tobytes()).decode("utf-8")

