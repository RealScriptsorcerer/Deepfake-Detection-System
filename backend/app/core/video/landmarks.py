from typing import Dict, Any, List, Tuple
import numpy as np
import cv2
import mediapipe as mp


_mp_face_mesh = mp.solutions.face_mesh


def _landmarks_to_numpy(landmarks, image_shape: Tuple[int, int]) -> np.ndarray:
    h, w = image_shape[:2]
    pts = np.array([(lm.x * w, lm.y * h) for lm in landmarks.landmark], dtype=np.float32)
    return pts


def _eye_opening_ratio(pts: np.ndarray, left: bool = True) -> float:
    # Mediapipe indices (approx): left eye [33, 160, 158, 133, 153, 144], right eye [362, 385, 387, 263, 373, 380]
    if left:
        top_idx, bottom_idx, left_idx, right_idx = 159, 145, 33, 133
    else:
        top_idx, bottom_idx, left_idx, right_idx = 386, 374, 362, 263
    vertical = np.linalg.norm(pts[top_idx] - pts[bottom_idx])
    horizontal = np.linalg.norm(pts[left_idx] - pts[right_idx]) + 1e-6
    return float(np.clip(vertical / horizontal, 0.0, 1.5))


def _lip_aperture_ratio(pts: np.ndarray) -> float:
    # Upper/lower lip center indices: 13 (upper), 14 (lower)
    # Normalize by inter-ocular distance
    upper, lower = pts[13], pts[14]
    left_eye, right_eye = pts[33], pts[263]
    eye_dist = np.linalg.norm(left_eye - right_eye) + 1e-6
    return float(np.clip(np.linalg.norm(upper - lower) / eye_dist, 0.0, 1.5))


def analyze_face_landmarks(frames_rgb: List[np.ndarray]) -> Dict[str, Any]:
    if not frames_rgb:
        return {"ear_left": [], "ear_right": [], "blink_indices": [], "lip_aperture": [], "face_hulls": []}
    ear_left: List[float] = []
    ear_right: List[float] = []
    lip_aperture: List[float] = []
    blink_indices: List[int] = []
    face_hulls: List[np.ndarray] = []

    with _mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as mesh:
        for idx, img in enumerate(frames_rgb):
            res = mesh.process(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            if not res.multi_face_landmarks:
                ear_left.append(0.0)
                ear_right.append(0.0)
                lip_aperture.append(0.0)
                face_hulls.append(None)
                continue
            lm = res.multi_face_landmarks[0]
            pts = _landmarks_to_numpy(lm, img.shape[:2])
            ear_l = _eye_opening_ratio(pts, True)
            ear_r = _eye_opening_ratio(pts, False)
            lip = _lip_aperture_ratio(pts)
            ear_left.append(ear_l)
            ear_right.append(ear_r)
            lip_aperture.append(lip)
            # Face hull: use a subset around cheeks and jaw
            hull_idx = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 152, 148, 176, 149, 150, 136, 172]
            hull = cv2.convexHull(pts[hull_idx].astype(np.float32)) if pts.shape[0] > max(hull_idx) else None
            face_hulls.append(hull)

    # Blink detection: low EAR frame indices
    ear = (np.array(ear_left) + np.array(ear_right)) / 2.0
    if ear.size:
        thresh = max(0.15, float(np.percentile(ear, 10)))
        blink_indices = np.where(ear < thresh)[0].tolist()
    return {
        "ear_left": ear_left,
        "ear_right": ear_right,
        "lip_aperture": lip_aperture,
        "blink_indices": blink_indices,
        "face_hulls": face_hulls,
    }

