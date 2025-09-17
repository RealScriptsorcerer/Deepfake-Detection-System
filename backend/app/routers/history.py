from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from typing import Any

from ..core.storage.db import list_results, get_result, set_truth_label, compute_metrics


router = APIRouter()


@router.get("/list")
def list_recent(limit: int = 50) -> JSONResponse:
    return JSONResponse({"items": list_results(limit)})


@router.get("/get/{result_id}")
def get_one(result_id: int) -> JSONResponse:
    result = get_result(result_id)
    if not result:
        raise HTTPException(status_code=404, detail="Not Found")
    return JSONResponse(result)


@router.post("/label/{result_id}")
def label_one(result_id: int, body: dict) -> JSONResponse:
    truth = str(body.get("truth", "")).lower()
    if truth not in ("real", "fake"):
        raise HTTPException(status_code=400, detail="truth must be 'real' or 'fake'")
    ok = set_truth_label(result_id, truth)
    if not ok:
        raise HTTPException(status_code=404, detail="Not Found")
    return JSONResponse({"ok": True})


@router.get("/metrics")
def metrics() -> JSONResponse:
    return JSONResponse(compute_metrics())

