from typing import Dict, Any, Tuple, List
import numpy as np
import os
import subprocess
import tempfile
import soundfile as sf


def _decode_audio(input_path: str, sr: int = 16000) -> np.ndarray:
    tmp_fd, tmp_wav = tempfile.mkstemp(suffix=".wav")
    os.close(tmp_fd)
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ac", "1", "-ar", str(sr), "-f", "wav", tmp_wav,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        y, sr_read = sf.read(tmp_wav, dtype="float32")
        if y.ndim > 1:
            y = y.mean(axis=1)
        return y.astype(np.float32)
    except Exception:
        return np.zeros(1, dtype=np.float32)
    finally:
        try:
            os.remove(tmp_wav)
        except OSError:
            pass


def _frame_energy(y: np.ndarray, sr: int, hop_ms: float = 40.0) -> np.ndarray:
    hop = int(sr * hop_ms / 1000.0)
    hop = max(hop, 1)
    if y.size < hop:
        return np.array([0.0], dtype=np.float32)
    energies: List[float] = []
    for i in range(0, len(y) - hop + 1, hop):
        seg = y[i:i + hop]
        energies.append(float(np.mean(seg * seg)))
    arr = np.array(energies, dtype=np.float32)
    arr = (arr - arr.min()) / (arr.ptp() + 1e-8)
    return arr


def compute_av_sync_score(video_path: str, lip_aperture_series: List[float], fps: float) -> Dict[str, Any]:
    # Map lip series to ~25Hz energy series for correlation
    if not lip_aperture_series or fps <= 0:
        return {"sync_score": 0.0, "lag_seconds": 0.0}
    y = _decode_audio(video_path, sr=16000)
    if y.size <= 1:
        return {"sync_score": 0.0, "lag_seconds": 0.0}
    audio_energy = _frame_energy(y, 16000, hop_ms=40.0)  # ~25Hz
    lip = np.array(lip_aperture_series, dtype=np.float32)
    # Downsample lip to match length
    target_len = len(audio_energy)
    if target_len <= 1:
        return {"sync_score": 0.0, "lag_seconds": 0.0}
    lip_idx = (np.linspace(0, len(lip) - 1, target_len)).astype(int)
    lip_ds = lip[lip_idx]
    # Normalize
    def _norm(x):
        x = (x - x.mean()) / (x.std() + 1e-8)
        return x
    a = _norm(audio_energy)
    b = _norm(lip_ds)
    # Cross-correlation
    xcorr = np.correlate(a, b, mode='full')
    lag_idx = int(np.argmax(xcorr)) - (len(a) - 1)
    sync_score = float(np.max(xcorr) / (len(a)))
    lag_seconds = float(lag_idx * 0.04)
    # Clip reasonable range
    return {"sync_score": round(max(min(sync_score, 1.0), -1.0), 4), "lag_seconds": round(lag_seconds, 3)}

