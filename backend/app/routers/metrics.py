from fastapi import APIRouter
from fastapi.responses import PlainTextResponse
from ..core.metrics import METRICS


router = APIRouter()


@router.get("/metrics")
def scrape() -> PlainTextResponse:
    return PlainTextResponse(METRICS.scrape(), media_type="text/plain; version=0.0.4; charset=utf-8")

