import os
import tempfile
import base64
from typing import Tuple
from fastapi import UploadFile


def get_temp_dir() -> str:
    temp_dir = os.path.join(tempfile.gettempdir(), "deepfake_detect")
    os.makedirs(temp_dir, exist_ok=True)
    return temp_dir


def safe_filename(filename: str) -> str:
    return os.path.basename(filename).replace("..", ".")


def save_upload_to_temp(upload: UploadFile) -> str:
    temp_dir = get_temp_dir()
    filename = safe_filename(upload.filename or "upload.bin")
    temp_path = os.path.join(temp_dir, filename)
    with open(temp_path, "wb") as f:
        f.write(upload.file.read())
    return temp_path


def image_to_base64_png(image_bgr) -> str:
    import cv2
    success, buf = cv2.imencode(".png", image_bgr)
    if not success:
        return ""
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def figure_to_base64_png(fig) -> str:
    import io
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

