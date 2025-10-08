from typing import Dict, Any, List, Tuple
import cv2
import numpy as np

from .heuristics import (
    compute_temporal_difference_scores,
    compute_frequency_artifact_scores,
    compute_optical_flow_inconsistency,
    combine_video_scores_with_flow,
)
from ..utils.visualization import plot_video_suspicion_timeline
from .landmarks import analyze_face_landmarks
from ..av.sync import compute_av_sync_score
from ..utils.visualization import overlay_heatmap, encode_image_rgb_base64_png
from ..models.registry import REGISTRY


def _extract_frames_rgb(video_path: str, max_frames: int = 600, target_short_side: int = 256) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    frames: List[np.ndarray] = []
    try:
        if not cap.isOpened():
            return frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        stride = max(total_frames // max_frames, 1) if total_frames > 0 else 1
        index = 0
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            if index % stride != 0:
                index += 1
                continue
            h, w = frame_bgr.shape[:2]
            if min(h, w) != target_short_side:
                scale = target_short_side / float(min(h, w))
                new_w = int(round(w * scale))
                new_h = int(round(h * scale))
                frame_bgr = cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            index += 1
            if len(frames) >= max_frames:
                break
    finally:
        cap.release()
    return frames


def analyze_video_file(video_path: str) -> Dict[str, Any]:
    frames = _extract_frames_rgb(video_path)
    if not frames:
        return {"label": "unknown", "score": 0.0, "detail": {"error": "Could not read frames"}}

    temporal = compute_temporal_difference_scores(frames)
    freq = compute_frequency_artifact_scores(frames)
    flow = compute_optical_flow_inconsistency(frames)
    lm = analyze_face_landmarks(frames)
    per_frame, overall = combine_video_scores_with_flow(temporal, freq, flow)

    label = "fake" if overall >= 0.5 else "real"
    timeline_png_b64 = plot_video_suspicion_timeline(per_frame)
    # Top suspicious frames and approximate timestamps
    fps = _safe_fps(video_path)
    top_idx = np.argsort(np.array(per_frame))[-10:][::-1].tolist()
    top_ts = [round(i / max(fps, 1.0), 3) for i in top_idx]
    av_sync = compute_av_sync_score(video_path, lm.get("lip_aperture", []), fps)

    # Deep model score if available (placeholder backend)
    deep_score = REGISTRY.video.predict_video_crops(frames, fps= _safe_fps(video_path)) if hasattr(REGISTRY, 'video') else None

    # Generate heatmap overlays for top suspicious frames
    overlays: List[Dict[str, Any]] = []
    # Precompute optical flow magnitudes per frame pair for heatmap
    flow_mags: List[np.ndarray] = []
    if len(frames) > 1:
        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
        flow_mags.append(np.zeros_like(prev_gray, dtype=np.float32))
        for idx in range(1, len(frames)):
            gray = cv2.cvtColor(frames[idx], cv2.COLOR_RGB2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            flow_mags.append(mag.astype(np.float32))
            prev_gray = gray
    else:
        flow_mags = [np.zeros(frames[0].shape[:2], dtype=np.float32)]

    face_hulls = lm.get("face_hulls", []) or [None] * len(frames)
    for i in top_idx[:5]:
        if i < 0 or i >= len(frames):
            continue
        # Build a simple heatmap: combination of flow magnitude and Laplacian edges
        mag = flow_mags[i]
        gray = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
        lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        lap_abs = np.abs(lap)
        # Normalize components
        mag_n = (mag - mag.min()) / (mag.ptp() + 1e-8)
        lap_n = (lap_abs - lap_abs.min()) / (lap_abs.ptp() + 1e-8)
        hm = 0.6 * mag_n + 0.4 * lap_n
        # Mask to face hull if available
        hull = face_hulls[i]
        if hull is not None and isinstance(hull, np.ndarray):
            mask = np.zeros_like(gray, dtype=np.uint8)
            cv2.fillConvexPoly(mask, hull.astype(np.int32), 255)
            hm = hm * (mask.astype(np.float32) / 255.0)
        overlay = overlay_heatmap(frames[i], hm)
        b64 = encode_image_rgb_base64_png(overlay)
        overlays.append({"frame_index": i, "timestamp": round(i / max(fps, 1.0), 3), "overlay_png_base64": b64})

    return {
        "modality": "video",
        "label": label,
        "score": round(float(overall if deep_score is None else 0.7 * overall + 0.3 * float(deep_score)), 4),
        "explanations": {
            "per_frame_suspicion": per_frame,
            "temporal_diff": temporal,
            "frequency_ratio": freq,
            "optical_flow_inconsistency": flow,
            "ear_left": lm.get("ear_left", []),
            "ear_right": lm.get("ear_right", []),
            "blink_indices": lm.get("blink_indices", []),
            "lip_aperture": lm.get("lip_aperture", []),
            "timeline_png_base64": timeline_png_b64,
            "top_suspicious_frames": top_idx,
            "top_suspicious_timestamps": top_ts,
            "av_sync": av_sync,
            "deep_video_score": deep_score,
            "heatmap_overlays": overlays,
        },
    }


def analyze_image_file(image_path: str) -> Dict[str, Any]:
    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        return {"label": "unknown", "score": 0.0, "detail": {"error": "Could not read image"}}
    h, w = image_bgr.shape[:2]
    target_short = 512
    if min(h, w) != target_short:
        scale = target_short / float(min(h, w))
        image_bgr = cv2.resize(image_bgr, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    # Reuse frequency artifact score as image score
    from .heuristics import compute_dct_highfreq_ratio
    freq_ratio = compute_dct_highfreq_ratio(image_rgb)
    score = float(np.clip(freq_ratio * 1.6, 0.0, 1.0))
    label = "fake" if score >= 0.5 else "real"
    return {
        "modality": "image",
        "label": label,
        "score": round(score, 4),
        "explanations": {
            "frequency_ratio": freq_ratio,
        },
    }


def _safe_fps(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if not np.isfinite(fps) or fps <= 0:
            return 25.0
        return fps
    finally:
        cap.release()

