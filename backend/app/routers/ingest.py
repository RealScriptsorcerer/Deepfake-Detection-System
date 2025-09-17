from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import os

from ..core.ingest.downloader import download_media_to_temp
from ..core.video.processing import analyze_video_file
from ..core.audio.processing import analyze_audio_file
from ..core.storage.db import save_result


router = APIRouter()


@router.post("/url")
def ingest_url(body: dict) -> JSONResponse:
    url = str(body.get("url", "")).strip()
    if not url:
        raise HTTPException(status_code=400, detail="url is required")
    path, cth = download_media_to_temp(url)
    try:
        if cth == 'video':
            result = analyze_video_file(path)
            rid = save_result("video", url, result.get("label", "unknown"), float(result.get("score", 0.0)), result)
        else:
            result = analyze_audio_file(path)
            rid = save_result("audio", url, result.get("label", "unknown"), float(result.get("score", 0.0)), result)
        return JSONResponse({"id": rid, **result})
    finally:
        try:
            os.remove(path)
        except Exception:
            pass

