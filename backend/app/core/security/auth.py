import os
import time
from typing import Callable, Iterable
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


API_KEY = os.environ.get("DETECT_API_KEY", "").strip()
OPEN_PATHS: tuple[str, ...] = (
    "/health",
    "/metrics",
    "/docs",
    "/redoc",
    "/openapi.json",
    "/ws/stream",
)


class ApiKeyAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable):
        # Bypass when no key configured
        if API_KEY == "":
            return await call_next(request)

        # Allow open paths
        path = request.url.path
        if path.startswith(OPEN_PATHS):
            return await call_next(request)

        key = request.headers.get("x-api-key", "")
        if key != API_KEY:
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)
        return await call_next(request)

