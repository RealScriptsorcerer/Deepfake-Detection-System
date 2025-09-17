#!/usr/bin/env python3
import argparse
import cv2
import numpy as np
import time


def compute_temporal_diff(prev_gray, gray):
    return float(np.mean(cv2.absdiff(prev_gray, gray)) / 255.0)


def compute_freq_ratio(gray):
    gray_f = np.float32(gray) / 255.0
    dct = cv2.dct(gray_f)
    h, w = dct.shape
    low = dct[:16, :16]
    total = float(np.mean(np.abs(dct)) + 1e-8)
    low_e = float(np.mean(np.abs(low)) + 1e-8)
    high = max(total - low_e, 1e-8)
    return float(np.clip(high / total, 0.0, 1.0))


def main():
    parser = argparse.ArgumentParser(description="Live deepfake suspicion heuristic")
    parser.add_argument("source", help="0 for webcam, or RTSP/URL/path")
    args = parser.parse_args()

    src = 0 if args.source == "0" else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print("Failed to open source")
        return

    prev_gray = None
    ema_score = 0.0
    alpha = 0.1

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, (640, int(frame.shape[0] * 640 / frame.shape[1])), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is None:
            prev_gray = gray
        tdiff = compute_temporal_diff(prev_gray, gray)
        freq = compute_freq_ratio(gray)
        score = float(np.clip(0.6 * tdiff + 0.4 * freq, 0.0, 1.0))
        ema_score = (1 - alpha) * ema_score + alpha * score
        prev_gray = gray

        color = (0, 0, 255) if ema_score >= 0.5 else (0, 255, 0)
        cv2.putText(frame, f"Suspicion: {ema_score:.3f}", (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.imshow("Live Deepfake Suspicion", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

