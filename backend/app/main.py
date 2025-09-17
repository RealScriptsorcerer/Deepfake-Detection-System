from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import detect


def create_app() -> FastAPI:
    app = FastAPI(title="Deepfake Detection API", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(detect.router, prefix="/detect", tags=["detect"]) 

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    return app


app = create_app()

