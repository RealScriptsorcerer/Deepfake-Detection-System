from typing import List, Tuple
import numpy as np
import cv2


def compute_temporal_difference_scores(frames_rgb: List[np.ndarray]) -> List[float]:
    scores: List[float] = []
    if len(frames_rgb) < 2:
        return [0.0] * len(frames_rgb)
    prev_gray = cv2.cvtColor(frames_rgb[0], cv2.COLOR_RGB2GRAY)
    for idx in range(1, len(frames_rgb)):
        gray = cv2.cvtColor(frames_rgb[idx], cv2.COLOR_RGB2GRAY)
        diff = cv2.absdiff(gray, prev_gray)
        score = float(np.mean(diff) / 255.0)
        scores.append(score)
        prev_gray = gray
    # Prepend a neutral score for the first frame to align lengths
    return [scores[0] if scores else 0.0] + scores


def compute_dct_highfreq_ratio(frame_rgb: np.ndarray, low_keep: int = 16) -> float:
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    gray_float = np.float32(gray) / 255.0
    dct = cv2.dct(gray_float)
    h, w = dct.shape
    low_region = dct[:low_keep, :low_keep]
    total_energy = float(np.mean(np.abs(dct)) + 1e-8)
    low_energy = float(np.mean(np.abs(low_region)) + 1e-8)
    high_energy = max(total_energy - low_energy, 1e-8)
    ratio = high_energy / total_energy
    return float(np.clip(ratio, 0.0, 1.0))


def compute_frequency_artifact_scores(frames_rgb: List[np.ndarray]) -> List[float]:
    return [compute_dct_highfreq_ratio(f) for f in frames_rgb]


def combine_video_scores(temporal: List[float], freq: List[float]) -> Tuple[List[float], float]:
    temporal_arr = np.array(temporal, dtype=np.float32)
    freq_arr = np.array(freq, dtype=np.float32)
    # Heuristic: high frequency ratio and inconsistent temporal diffs may indicate manipulation
    # Normalize temporal by its moving average to get instability
    if temporal_arr.size == 0:
        return [], 0.0
    window = max(3, int(len(temporal_arr) * 0.05))
    kernel = np.ones(window, dtype=np.float32) / float(window)
    mov_avg = np.convolve(temporal_arr, kernel, mode="same")
    instability = np.abs(temporal_arr - mov_avg)
    per_frame = 0.4 * freq_arr + 0.6 * instability
    per_frame = np.clip(per_frame, 0.0, 1.0)
    overall = float(np.clip(np.mean(per_frame) * 1.5, 0.0, 1.0))
    return per_frame.tolist(), overall

