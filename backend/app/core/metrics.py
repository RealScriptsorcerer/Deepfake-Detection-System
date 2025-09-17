from __future__ import annotations
from typing import Callable
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import time


class SimpleMetrics:
    def __init__(self):
        self.request_count: int = 0
        self.request_duration_sum_ms: float = 0.0

    def scrape(self) -> str:
        lines = [
            f"deepfake_requests_total {self.request_count}",
            f"deepfake_request_duration_ms_sum {self.request_duration_sum_ms:.3f}",
        ]
        return "\n".join(lines) + "\n"


METRICS = SimpleMetrics()


class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable):
        start = time.time()
        response: Response = await call_next(request)
        dur_ms = (time.time() - start) * 1000.0
        METRICS.request_count += 1
        METRICS.request_duration_sum_ms += dur_ms
        return response

