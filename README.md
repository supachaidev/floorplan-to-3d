# Floorplan to 3D

Upload a 2D floor plan image, automatically detect rooms and doors, edit boundaries interactively, and visualize in 3D.

## Architecture

```
Floor plan image
      |
  FastAPI /upload
      |
  Preprocess (deskew) -> Detect rooms & doors -> Simplify polygons -> Normalize to meters
      |
  JSON response
      |
  Frontend: 2D editor (Canvas) + 3D viewer (Three.js)
```

**Detection pipeline:** OpenCV for clean printed plans, Claude Vision API as fallback for complex/hand-drawn plans.

## Project Structure

```
backend/
  main.py                  # FastAPI server (POST /upload, GET /health)
  requirements.txt
  pipeline/
    preprocess.py           # Deskew & perspective correction
    detect.py               # Room/door detection (OpenCV + Claude Vision)
    polygons.py             # Polygon simplification & meter normalization
    export.py               # Final JSON schema builder
  evaluate.py               # Accuracy evaluation against ground truth
  convert_cubicasa.py        # CubiCasa5K dataset converter
  run_eval.sh               # End-to-end evaluation pipeline

frontend/
  index.html                # Split-panel UI
  editor.js                 # 2D interactive polygon editor (Canvas)
  viewer.js                 # 3D room visualization (Three.js)
```

## Getting Started

### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

Server starts at `http://localhost:8000`.

Set `ANTHROPIC_API_KEY` environment variable to enable Claude Vision fallback.

### Frontend

Serve the `frontend/` directory on port 5500 (configured in CORS):

```bash
cd frontend
python -m http.server 5500
```

Open `http://localhost:5500` in your browser.

## API

### `POST /upload`

Upload a floor plan image. Returns detected rooms and doors as JSON.

| Parameter     | Type  | Description                          |
|---------------|-------|--------------------------------------|
| `file`        | file  | Image file (PNG, JPG, BMP, TIFF)     |
| `force_claude`| query | Set `true` to use Claude Vision API  |

### `GET /health`

Health check endpoint.

## Evaluation

Run accuracy evaluation against the CubiCasa5K dataset:

```bash
cd backend

# Default: 5 samples, OpenCV mode
./run_eval.sh

# Random samples each run
./run_eval.sh --shuffle

# Custom options
./run_eval.sh --limit 20 --shuffle --claude --output results.json

# Clean up downloaded dataset
./run_eval.sh --clean
```

The script downloads CubiCasa5K (~5.5 GB), converts SVG annotations to ground truth, and reports precision, recall, F1, and IoU metrics for rooms and doors.
