from fastapi import APIRouter
from fastapi.responses import JSONResponse
from ..core.models.registry import REGISTRY


router = APIRouter()


@router.get("/models/status")
def status() -> JSONResponse:
    return JSONResponse({
        "video_backend_available": REGISTRY.video.available,
        "audio_backend_available": REGISTRY.audio.available,
    })

