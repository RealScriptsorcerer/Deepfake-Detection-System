from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import numpy as np
import cv2
import base64
from typing import Any, Dict

from ..core.video.heuristics import compute_dct_highfreq_ratio


router = APIRouter()


@router.websocket("/ws/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            message = await ws.receive_json()
            # Expect base64 PNG of frame
            b64 = message.get("frame_png_base64", "")
            if not b64:
                await ws.send_json({"error": "missing frame"})
                continue
            data = base64.b64decode(b64)
            arr = np.frombuffer(data, dtype=np.uint8)
            img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img_bgr is None:
                await ws.send_json({"error": "decode failed"})
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            freq_ratio = compute_dct_highfreq_ratio(img_rgb)
            score = float(np.clip(freq_ratio * 1.6, 0.0, 1.0))
            await ws.send_json({"score": score, "freq_ratio": freq_ratio})
    except WebSocketDisconnect:
        pass

