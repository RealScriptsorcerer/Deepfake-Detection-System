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
streamlit run frontend/streamlit_app.py
```

## API Endpoints
- POST `/detect/image` (multipart `file`)
- POST `/detect/video` (multipart `file`)
- POST `/detect/audio` (multipart `file`)
- POST `/detect/av` (multipart `video` and/or `audio`)

## Notes
- This MVP uses fast heuristics (temporal diffs, DCT frequency ratio; MFCC/ZCR-derived audio score). Replace with trained models via the ensemble interface in future phases.
- ffmpeg is included for container-side codec support.

## Next Steps
- Integrate pretrained backbones and an ensemble aggregator
- Add facial landmark/eye-blink analysis via MediaPipe
- Add real-time webcam/microphone streaming modes
- Persist results and logs in a database
- Evaluate on benchmark datasets