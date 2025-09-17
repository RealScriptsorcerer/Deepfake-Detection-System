# Deepfake Detection System (MVP)

## Overview
A real-time oriented backend + simple UI to analyze images, videos, and audio for potential deepfakes. Provides a baseline heuristic detector and visual explanations. Designed to be extended with ensemble models (EfficientNet/Xception, RawNet2, etc.).

## Structure
```
backend/
  app/
    main.py
    routers/detect.py
    core/
      video/{processing.py, heuristics.py}
      audio/processing.py
      utils/{common.py, visualization.py}
  requirements.txt
  Dockerfile
frontend/
  streamlit_app.py
  requirements.txt

docker-compose.yml
```

## Quickstart (Docker)
```
docker compose up --build
```
- API: http://localhost:8000/docs
- UI: http://localhost:8501
 - Metrics: http://localhost:8000/metrics
 - Report: http://localhost:8000/report/{id}

## Run locally (backend)
```
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## Run locally (frontend)
```
pip install -r frontend/requirements.txt
export API_URL=http://localhost:8000
export API_KEY=  # set if API has key auth enabled
streamlit run frontend/streamlit_app.py
```

## API Endpoints
- POST `/detect/image` (multipart `file`)
- POST `/detect/video` (multipart `file`)
- POST `/detect/audio` (multipart `file`)
- POST `/detect/av` (multipart `video` and/or `audio`)
- POST `/detect/batch` (multipart `files[]`)
- POST `/ingest/url` (body `{url}`)
- GET `/history/list`, `/history/get/{id}`, POST `/history/label/{id}`
- GET `/report/{id}` (HTML)
- GET `/metrics` (Prometheus)
- WS `/ws/stream` (send/recv JSON with `frame_png_base64`)

## Notes
- This MVP uses fast heuristics (temporal diffs, DCT frequency ratio; MFCC/ZCR-derived audio score). Replace with trained models via the ensemble interface in future phases.
- ffmpeg is included for container-side codec support.
 - API key: set `DETECT_API_KEY` to require `x-api-key` header (empty to disable)
 - Database: set `DEEPFAKE_DB` to persist (compose mounts `/data` volume)
 - Webhook: set `DF_WEBHOOK_URL` to receive notifications on score >= 0.8
- Models: place weights in `MODEL_DIR` (default `/models`); `/models/status` shows availability
- Live detection (local): `python scripts/live_detect.py 0` or an RTSP URL
- Dataset ingestion: `python scripts/ingest_dataset.py <folder> http://localhost:8000`
- Download models (scaffold): `python scripts/download_models.py --dir /models`

## Next Steps
- Integrate pretrained backbones and an ensemble aggregator
- Add facial landmark/eye-blink analysis via MediaPipe
- Add real-time webcam/microphone streaming modes
- Persist results and logs in a database
- Evaluate on benchmark datasets