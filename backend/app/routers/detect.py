from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import Any, Dict
import os

from ..core.utils.common import save_upload_to_temp
from ..core.video.processing import analyze_video_file, analyze_image_file
from ..core.audio.processing import analyze_audio_file
from ..core.ensemble import SimpleEnsembleAggregator
from ..core.storage.db import save_result


router = APIRouter()

aggregator = SimpleEnsembleAggregator()

@router.post("/image")
async def detect_image(file: UploadFile = File(...)) -> JSONResponse:
    if file.content_type is None or not str(file.content_type).startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload a valid image file")
    temp_path = save_upload_to_temp(file)
    try:
        result = analyze_image_file(temp_path)
        result_id = save_result("image", file.filename, result.get("label", "unknown"), float(result.get("score", 0.0)), result)
        return JSONResponse({"id": result_id, **result})
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


@router.post("/video")
async def detect_video(file: UploadFile = File(...)) -> JSONResponse:
    if file.content_type is None or not str(file.content_type).startswith("video/"):
        raise HTTPException(status_code=400, detail="Please upload a valid video file")
    temp_path = save_upload_to_temp(file)
    try:
        result = analyze_video_file(temp_path)
        result_id = save_result("video", file.filename, result.get("label", "unknown"), float(result.get("score", 0.0)), result)
        return JSONResponse({"id": result_id, **result})
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


@router.post("/audio")
async def detect_audio(file: UploadFile = File(...)) -> JSONResponse:
    if file.content_type is None or not (str(file.content_type).startswith("audio/") or file.filename.lower().endswith((".wav", ".mp3", ".flac", ".m4a", ".ogg"))):
        raise HTTPException(status_code=400, detail="Please upload a valid audio file")
    temp_path = save_upload_to_temp(file)
    try:
        result = analyze_audio_file(temp_path)
        result_id = save_result("audio", file.filename, result.get("label", "unknown"), float(result.get("score", 0.0)), result)
        return JSONResponse({"id": result_id, **result})
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


@router.post("/av")
async def detect_av(video: UploadFile = File(None), audio: UploadFile = File(None)) -> JSONResponse:
    if video is None and audio is None:
        raise HTTPException(status_code=400, detail="Provide at least one of video or audio")
    video_path = audio_path = None
    try:
        video_result: Dict[str, Any] | None = None
        audio_result: Dict[str, Any] | None = None
        if video is not None:
            if video.content_type is None or not str(video.content_type).startswith("video/"):
                raise HTTPException(status_code=400, detail="Invalid video file")
            video_path = save_upload_to_temp(video)
            video_result = analyze_video_file(video_path)
        if audio is not None:
            if audio.content_type is None and not audio.filename.lower().endswith((".wav", ".mp3", ".flac", ".m4a", ".ogg")):
                raise HTTPException(status_code=400, detail="Invalid audio file")
            audio_path = save_upload_to_temp(audio)
            audio_result = analyze_audio_file(audio_path)

        scores = {}
        if video_result is not None:
            scores["video"] = float(video_result.get("score", 0.0))
        if audio_result is not None:
            scores["audio"] = float(audio_result.get("score", 0.0))
        agg = aggregator.aggregate(scores) if scores else 0.0
        label = "fake" if agg >= 0.5 else "real"
        result_payload = {
            "modality": "av",
            "label": label,
            "score": round(float(agg), 4),
            "components": {
                "video": video_result,
                "audio": audio_result,
            },
        }
        result_id = save_result("av", video.filename if video else (audio.filename if audio else ""), label, float(agg), result_payload)
        return JSONResponse({"id": result_id, **result_payload})


@router.post("/batch")
async def detect_batch(files: list[UploadFile] = File(...)) -> JSONResponse:
    results = []
    for f in files:
        try:
            temp_path = save_upload_to_temp(f)
            if f.content_type and f.content_type.startswith("image/"):
                res = analyze_image_file(temp_path)
                rid = save_result("image", f.filename, res.get("label", "unknown"), float(res.get("score", 0.0)), res)
            elif f.content_type and f.content_type.startswith("video/"):
                res = analyze_video_file(temp_path)
                rid = save_result("video", f.filename, res.get("label", "unknown"), float(res.get("score", 0.0)), res)
            elif f.content_type and f.content_type.startswith("audio/"):
                res = analyze_audio_file(temp_path)
                rid = save_result("audio", f.filename, res.get("label", "unknown"), float(res.get("score", 0.0)), res)
            else:
                continue
            results.append({"id": rid, **res})
        finally:
            try:
                os.remove(temp_path)
            except Exception:
                pass
    return JSONResponse({"items": results})
    finally:
        for p in (video_path, audio_path):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass

