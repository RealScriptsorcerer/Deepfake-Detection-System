from typing import Dict, Any, Tuple
import os
import subprocess
import tempfile
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt

from ..utils.common import figure_to_base64_png


def _decode_to_wav_mono_16k(input_path: str) -> Tuple[str, int]:
    tmp_fd, tmp_wav = tempfile.mkstemp(suffix=".wav")
    os.close(tmp_fd)
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ac", "1", "-ar", "16000", "-f", "wav", tmp_wav,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return tmp_wav, 16000
    except Exception:
        # Fall back to original file; may still be readable by soundfile
        return input_path, 16000


def _load_audio_mono(path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    wav_path, sr = _decode_to_wav_mono_16k(path)
    try:
        y, sr_read = sf.read(wav_path, dtype="float32", always_2d=False)
        if y is None or (isinstance(y, np.ndarray) and y.size == 0):
            return np.zeros(1, dtype=np.float32), target_sr
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        return y.astype(np.float32), int(sr_read)
    except Exception:
        return np.zeros(1, dtype=np.float32), target_sr
    finally:
        if wav_path != path:
            try:
                os.remove(wav_path)
            except OSError:
                pass


def _frame_audio(signal: np.ndarray, frame_length: int = 1024, hop_length: int = 512) -> np.ndarray:
    if signal.size < frame_length:
        pad_width = frame_length - signal.size
        signal = np.pad(signal, (0, pad_width), mode="constant")
    num_frames = 1 + (len(signal) - frame_length) // hop_length
    frames = np.lib.stride_tricks.as_strided(
        signal,
        shape=(num_frames, frame_length),
        strides=(signal.strides[0] * hop_length, signal.strides[0]),
        writeable=False,
    )
    window = np.hanning(frame_length).astype(np.float32)
    return frames * window[None, :]


def _compute_baseline_scores(y: np.ndarray, sr: int) -> Dict[str, float]:
    frames = _frame_audio(y, 1024, 512)
    # Zero Crossing Rate per frame
    zc = np.mean((frames[:, 1:] * frames[:, :-1]) < 0, axis=1)
    zcr_mean = float(np.mean(zc))

    # Magnitude spectra
    fft = np.fft.rfft(frames, axis=1)
    mag = np.abs(fft) + 1e-8
    freqs = np.fft.rfftfreq(frames.shape[1], d=1.0 / sr)
    nyquist = 0.5 * sr

    # Spectral centroid and bandwidth
    centroid = np.sum(freqs[None, :] * mag, axis=1) / np.sum(mag, axis=1)
    centroid_mean_norm = float(np.mean(centroid) / nyquist)
    # Bandwidth: sqrt of weighted second central moment
    spread = np.sqrt(np.sum(((freqs[None, :] - centroid[:, None]) ** 2) * mag, axis=1) / np.sum(mag, axis=1))
    bandwidth_mean_norm = float(np.mean(spread) / nyquist)

    # Spectral flatness (geometric mean / arithmetic mean)
    geom = np.exp(np.mean(np.log(mag), axis=1))
    arith = np.mean(mag, axis=1)
    flatness = np.clip(geom / arith, 0.0, 1.0)
    flatness_mean = float(np.mean(flatness))

    # Phase inconsistency heuristic (approximate neural vocoder artifact):
    # compute frame-wise phase delta variance across bins
    fft_c = np.fft.rfft(frames, axis=1)
    phase = np.angle(fft_c)
    dphi = np.diff(phase, axis=0)
    phase_var = float(np.mean(np.var(dphi, axis=1)))

    # Heuristic score
    score = 0.0
    score += 0.45 * np.clip(centroid_mean_norm, 0.0, 1.0)
    score += 0.35 * np.clip(bandwidth_mean_norm, 0.0, 1.0)
    score += 0.20 * (1.0 - np.clip(zcr_mean, 0.0, 1.0))
    score = float(np.clip(score, 0.0, 1.0))

    return {
        "zcr_mean": zcr_mean,
        "centroid_mean_norm": centroid_mean_norm,
        "bandwidth_mean_norm": bandwidth_mean_norm,
        "flatness_mean": flatness_mean,
        "phase_delta_var": phase_var,
        "heuristic_score": score,
    }


def _spectrogram_png_b64(y: np.ndarray, sr: int) -> str:
    fig, ax = plt.subplots(figsize=(6, 3))
    Pxx, freqs, bins, im = ax.specgram(y, NFFT=1024, Fs=sr, noverlap=768, cmap="magma")
    ax.set_ylim([0, sr / 2])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Spectrogram")
    b64 = figure_to_base64_png(fig)
    return b64


def analyze_audio_file(audio_path: str) -> Dict[str, Any]:
    y, sr = _load_audio_mono(audio_path)
    if y.size == 0:
        return {"label": "unknown", "score": 0.0, "detail": {"error": "Could not read audio"}}
    feats = _compute_baseline_scores(y, sr)
    score = float(np.clip(1.1 * feats["heuristic_score"], 0.0, 1.0))
    label = "fake" if score >= 0.5 else "real"
    spec_png_b64 = _spectrogram_png_b64(y, sr)
    return {
        "modality": "audio",
        "label": label,
        "score": round(score, 4),
        "explanations": {
            **feats,
            "spectrogram_png_base64": spec_png_b64,
        },
    }

