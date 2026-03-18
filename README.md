# Floorplan to 3D

Upload a 2D floor plan image, automatically detect rooms and doors, edit boundaries interactively, and visualize in 3D.

Two detection pipelines are available:

- **OpenCV pipeline** — classical image processing for clean printed plans
- **VLM pipeline** — Qwen2.5-VL vision-language model for structured vectorization with a wall-first schema

## Architecture

```
Floor plan image
      |
  ┌───────────────────────────┐
  │  FastAPI /upload           │  OpenCV pipeline
  │  Preprocess → Detect rooms │  (deskew, contours, polygon simplification)
  │  & doors → Simplify →     │
  │  Normalize to meters       │
  └───────────────────────────┘
              OR
  ┌───────────────────────────┐
  │  FastAPI /upload-vlm       │  VLM pipeline
  │  Qwen2.5-VL structured    │  (walls → rooms → openings)
  │  extraction                │
  └───────────────────────────┘
      |
  JSON response
      |
  Frontend: 2D editor (Canvas) + 3D viewer (Three.js)
```

## Project Structure

```
backend/
  main.py                  # FastAPI server (POST /upload, POST /upload-vlm, GET /health)
  requirements.txt
  pipeline/
    preprocess.py           # Deskew & perspective correction
    detect.py               # Room/door detection (pure OpenCV)
    vlm_vectorize.py        # VLM-based vectorization (Qwen2.5-VL)
    polygons.py             # Polygon simplification & meter normalization
    export.py               # Final JSON schema builder
  evaluate.py               # Accuracy evaluation against ground truth
  convert_cubicasa.py        # CubiCasa5K dataset converter
  run_eval.sh               # End-to-end evaluation pipeline

frontend/
  index.html                # Split-panel UI with pipeline selector
  editor.js                 # 2D interactive polygon editor (Canvas)
  viewer.js                 # 3D room & wall visualization (Three.js)
```

## Getting Started

### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload
```

Server starts at `http://localhost:8000`.

To use the VLM pipeline, install the optional dependencies (requires a CUDA-capable GPU):

```bash
pip install transformers>=4.45.0 torch>=2.0.0 accelerate qwen-vl-utils
```

### Frontend

Serve the `frontend/` directory on port 5500 (configured in CORS):

```bash
cd frontend
python -m http.server 5500
```

Open `http://localhost:5500` in your browser.

## API

### `POST /upload`

Upload a floor plan image. Detects rooms and doors using the OpenCV pipeline.

| Parameter | Type | Description                      |
|-----------|------|----------------------------------|
| `file`    | file | Image file (PNG, JPG, BMP, TIFF) |

### `POST /upload-vlm`

Upload a floor plan image. Extracts walls, rooms, and openings using the Qwen2.5-VL model.

| Parameter | Type  | Description                                |
|-----------|-------|--------------------------------------------|
| `file`    | file  | Image file (PNG, JPG, BMP, TIFF)           |
| `model`   | query | Model name (default: `Qwen/Qwen2.5-VL-3B-Instruct`) |

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
