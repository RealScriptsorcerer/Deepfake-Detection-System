from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import detect
from .routers import history
from .routers import metrics as metrics_router
from .routers import ws as ws_router
from .routers import ingest as ingest_router
from .routers import report as report_router
from .core.security.auth import ApiKeyAuthMiddleware
from .core.metrics import MetricsMiddleware


def create_app() -> FastAPI:
    app = FastAPI(title="Deepfake Detection API", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(ApiKeyAuthMiddleware)
    app.add_middleware(MetricsMiddleware)

    app.include_router(detect.router, prefix="/detect", tags=["detect"]) 
    app.include_router(history.router, prefix="/history", tags=["history"]) 
    app.include_router(metrics_router.router, tags=["metrics"]) 
    app.include_router(ws_router.router, tags=["ws"]) 
    app.include_router(ingest_router.router, prefix="/ingest", tags=["ingest"]) 
    app.include_router(report_router.router, tags=["report"]) 

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    return app


app = create_app()

