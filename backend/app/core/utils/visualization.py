from typing import List
import numpy as np
import matplotlib.pyplot as plt
from .common import figure_to_base64_png


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

