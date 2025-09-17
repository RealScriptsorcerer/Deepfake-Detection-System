from typing import List, Tuple, Optional
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


def compute_optical_flow_inconsistency(frames_rgb: List[np.ndarray]) -> List[float]:
    if len(frames_rgb) < 2:
        return [0.0] * len(frames_rgb)
    scores: List[float] = []
    prev_gray = cv2.cvtColor(frames_rgb[0], cv2.COLOR_RGB2GRAY)
    for idx in range(1, len(frames_rgb)):
        gray = cv2.cvtColor(frames_rgb[idx], cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # Spatial variance of flow magnitude captures localized jitter/warping
        mag_norm = (mag - mag.min()) / (mag.ptp() + 1e-8)
        score = float(np.clip(np.var(mag_norm), 0.0, 1.0))
        scores.append(score)
        prev_gray = gray
    return [scores[0] if scores else 0.0] + scores


def combine_video_scores_with_flow(
    temporal: List[float],
    freq: List[float],
    flow: Optional[List[float]] = None,
) -> Tuple[List[float], float]:
    temporal_arr = np.array(temporal, dtype=np.float32)
    freq_arr = np.array(freq, dtype=np.float32)
    if temporal_arr.size == 0:
        return [], 0.0
    window = max(3, int(len(temporal_arr) * 0.05))
    kernel = np.ones(window, dtype=np.float32) / float(window)
    mov_avg = np.convolve(temporal_arr, kernel, mode="same")
    instability = np.abs(temporal_arr - mov_avg)

    # Base fusion
    per_frame = 0.35 * freq_arr + 0.45 * instability
    if flow is not None and len(flow) == len(temporal):
        flow_arr = np.array(flow, dtype=np.float32)
        per_frame += 0.20 * flow_arr
    per_frame = np.clip(per_frame, 0.0, 1.0)
    overall = float(np.clip(np.mean(per_frame) * 1.5, 0.0, 1.0))
    return per_frame.tolist(), overall

