# Floorplan to 3D — Claude Code Context

## Project Goal
Convert 2D floor plan images (hand-drawn or CAD) into structured JSON
that drives a Three.js 3D viewer. Backend: FastAPI + OpenCV + Gemini API.
Frontend: Vanilla JS + Three.js r128.

## Team & Stakeholders
- Team name: All in
- University: Bangkok University (senior project)
- Industry partner: SCG (Siam Cement Group)
- Progress reports delivered to SCG advisors in bilingual Thai/English .docx format

---

## Architecture Principle (CRITICAL)
**OpenCV owns geometry precision. VLM owns semantic understanding.**
- OpenCV detects room polygons, wall positions, door arcs — all geometry
- Gemini Vision receives the image + OpenCV-detected room positions (as text)
  and returns only room labels/types — NO pixel coordinates from VLM
- This separation prevents coordinate hallucination

---

## Project Structure

```
floorplan-to-3d/
├── backend/
│   ├── main.py                    # FastAPI — POST /upload + POST /upload-vlm
│   ├── requirements.txt           # opencv, fastapi, uvicorn, numpy, Pillow
│   ├── pipeline/
│   │   ├── preprocess.py          # CLAHE contrast + deskew + perspective correction
│   │   ├── detect.py              # Room detection (flood-fill) + door detection (arc finder)
│   │   ├── polygons.py            # Polygon simplification + coordinate normalization to meters
│   │   ├── export.py              # Final JSON schema builder
│   │   └── vlm_vectorize.py       # Gemini hybrid: label_rooms_with_vlm()
│   ├── evaluate.py                # Accuracy evaluation (IoU, F1, precision, recall)
│   ├── convert_cubicasa.py        # CubiCasa5K SVG → ground truth JSON converter
│   └── run_eval.sh                # Automated eval pipeline
├── frontend/
│   ├── index.html
│   ├── editor.js                  # Interactive 2D polygon editor (Canvas API)
│   └── viewer.js                  # 3D renderer (Three.js r128)
├── README.md
├── ARCHITECTURE.md
└── CLAUDE.md                      # This file
```

---

## Two API Endpoints

### POST /upload (OpenCV only)
```
Image → deskew → detect_rooms_cv() → simplify → meters → JSON
```
Room labels are assigned by geometry heuristics (area ratios, aspect ratio).

### POST /upload-vlm (Hybrid: OpenCV geometry + Gemini labels)
```
Image → deskew → detect_rooms_cv(classify=False) → label_rooms_with_vlm() → simplify → meters → JSON
```
OpenCV detects polygons. Gemini labels each room using:
- Room center position (as normalized x, y fraction)
- Room size (% of total floor area)
- Number of corners
- The original image (for visual context — fixtures, counters, text labels)

Gemini returns ONLY labels and types — never coordinates.

---

## Current Benchmark Results (30 samples, CubiCasa5K)

```
Avg Room F1:  46.2%
Avg Room IoU: 46.8%
Avg Door F1:  32.0%
```

These are OpenCV-only results from /upload. The /upload-vlm endpoint exists
but its impact on F1 has not been measured yet (labeling does not affect polygon
accuracy, only type classification accuracy).

---

## Key Files — What They Do

### pipeline/vlm_vectorize.py
- `label_rooms_with_vlm(image_rgb, rooms, model_name)` → list[dict]
- Sends image + room center positions to Gemini
- Model default: `gemini-2.5-flash`
- Requires: `GEMINI_API_KEY` environment variable
- Fallback: if Gemini fails, keeps geometry-based labels silently
- Cleans up internal keys (`_area_frac`, `_aspect`) before returning

### pipeline/detect.py
- `detect_rooms_cv(image, classify=True)` — classify=False skips geometry-based
  type assignment (used in /upload-vlm path since Gemini handles labels)
- `detect_doors_cv(image)` — quarter-circle arc detector
- `detect_rooms(image)` — calls both, returns `{"rooms": [...], "doors": [...]}`

### pipeline/preprocess.py
- `deskew(image)` — CLAHE + perspective correction + rotation fix

### pipeline/polygons.py
- `simplify_polygon()` — Ramer-Douglas-Peucker
- `normalize_to_meters()` — pixel coords → meters (longest dim = 15m)
- `normalize_doors_to_meters()` — door positions + width → meters

### pipeline/export.py
- `build_floorplan_json(rooms, doors, w, h)` → final API response dict

---

## JSON Schema (API Response)

```json
{
  "floorplan": {
    "scale": 0.01,
    "units": "meters",
    "image_width_m": 15.0,
    "image_height_m": 12.5,
    "rooms": [
      {
        "id": "room_1",
        "label": "Living Room",
        "type": "living",
        "height": 3.0,
        "polygon": [{"x": 1.2, "y": 0.8}, {"x": 7.5, "y": 0.8}]
      }
    ],
    "doors": [
      {
        "id": "door_1",
        "position": {"x": 7.5, "y": 3.0},
        "width": 0.9,
        "angle": 180,
        "connects": []
      }
    ]
  }
}
```

---

## Current Investigation Task

### Hypothesis
Before deciding on further VLM improvements, we need to understand WHERE
OpenCV's geometry detection fails. The 46% Room F1 is the bottleneck — not
room labeling (which Gemini already handles).

Hypothesis: OpenCV works well (≥65% F1) on clean digital plans but fails on
messy/hand-drawn/photo inputs. If true, we improve preprocessing for those
categories instead of investing more in VLM.

### Image Categories to Profile

| Category | Description |
|---|---|
| `clean_digital` | High contrast, straight lines, exported from CAD/architectural software |
| `low_contrast` | Faded, aged scans, uneven lighting, washed-out lines |
| `hand_drawn` | Sketchy lines, inconsistent line thickness, imprecise corners |
| `photo` | Photographed plan, perspective distortion, glare, shadows |

### Task: Create backend/pipeline/profiler.py

```python
def profile_image(image: np.ndarray) -> dict:
    """
    Returns:
    {
      "category": "clean_digital" | "low_contrast" | "hand_drawn" | "photo",
      "contrast_score": float,      # std dev of grayscale pixels (0-255)
      "line_straightness": float,   # 0.0-1.0, higher = straighter
      "edge_regularity": float,     # 0.0-1.0, higher = more uniform edges
      "skew_angle_deg": float       # residual skew after deskew()
    }
    """
```

Category assignment thresholds:
- `contrast_score < 40`    → `low_contrast`
- `edge_regularity < 0.45` → `hand_drawn`
- `skew_angle_deg > 3.0`   → `photo`
- else                      → `clean_digital`

Heuristics (OpenCV only, no new dependencies):
- **contrast_score**: std deviation of grayscale pixel values
- **line_straightness**: ratio of HoughLinesP-detected line pixels to total Canny edge pixels
- **edge_regularity**: 1 minus normalized variance of local edge gradient orientations
- **skew_angle_deg**: residual rotation angle detected after deskew() has already run

### Task: Modify backend/evaluate.py

- Import `profile_image` from `pipeline.profiler`
- Call `profile_image()` on each sample image after loading (before deskew)
- After all samples, print a category breakdown table:

```
Category         | Count | Avg Room F1 | Avg Door F1 | Avg IoU
clean_digital    |    -- |         --% |         --% |      --%
low_contrast     |    -- |         --% |         --% |      --%
hand_drawn       |    -- |         --% |         --% |      --%
photo            |    -- |         --% |         --% |      --%
```

- If `--output` is used, include `"image_profile"` key in each sample's JSON

### Do NOT change
- `detect.py`, `main.py`, `polygons.py`, `export.py`, `preprocess.py`, `vlm_vectorize.py`
- Use only OpenCV in profiler.py — no new pip dependencies

### Run After Implementation

```bash
cd backend
./run_eval.sh --limit 50 --shuffle --output results_profiled.json
```

### Decision Table

| Result | Action |
|---|---|
| `clean_digital` F1 ≥ 65% | OpenCV is fine on clean inputs. Improve preprocessing for other categories. |
| `clean_digital` F1 < 50% | Core detection problem. Fix detect.py parameter sweep or flood-fill logic. |
| `hand_drawn` F1 < 30% | Need a dedicated preprocessing chain for sketched inputs. |
| All categories low and equal | Universal bottleneck — likely polygon matching threshold or simplification. |

---

## Running the Project

```bash
# Backend (OpenCV only)
cd backend
pip install -r requirements.txt
uvicorn main:app --reload                  # → http://localhost:8000

# Backend (with Gemini labeling)
export GEMINI_API_KEY="..."
uvicorn main:app --reload
# Use POST /upload-vlm instead of /upload

# Frontend
cd frontend
python3 -m http.server 5500                # → http://localhost:5500

# Evaluate (OpenCV only)
cd backend
./run_eval.sh --limit 30 --shuffle

# Evaluate with profiling (after profiler.py is implemented)
./run_eval.sh --limit 50 --shuffle --output results_profiled.json
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.11+, FastAPI, Uvicorn |
| Computer Vision | OpenCV (opencv-python-headless) |
| Image processing | Pillow (PIL) |
| VLM (room labeling) | Gemini API — model: `gemini-2.5-flash` |
| Frontend | Vanilla JS, HTML5 Canvas API |
| 3D Viewer | Three.js r128 (ExtrudeGeometry for walls) |
| Evaluation | CubiCasa5K dataset, custom IoU/F1 pipeline |
| Dev Environment | MacBook Air M2 |

---

## Key Constraints

- No GPU available — no local model inference
- No Node.js on backend
- Do not modify frontend/ unless explicitly asked
- Do not modify vlm_vectorize.py, polygons.py, or export.py unless explicitly asked
- All coordinate math uses normalized 0–1 range internally; meters only in final output
- Door width is clamped to 0.6m–1.5m in polygons.py
- Gemini prompt must never ask for pixel coordinates — only labels and types