from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from typing import Any

from ..core.storage.db import list_results, get_result


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

