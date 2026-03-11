# Floorplan to 3D — Architecture & Codebase Overview

## What This Project Does

Takes a 2D architectural floor plan image as input, detects rooms and doors using computer vision, and renders an interactive 3D model in the browser. Users can edit detected polygons in a 2D editor before viewing the final 3D result.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3, FastAPI, Uvicorn |
| Computer Vision | OpenCV (adaptive threshold, flood-fill, morphology, contour detection) |
| AI Fallback | Anthropic Claude Vision API (claude-sonnet-4-20250514) |
| Frontend | Vanilla JavaScript, HTML5 Canvas (2D editor), Three.js r128 (3D viewer) |
| Evaluation | CubiCasa5K dataset, custom accuracy pipeline |

## Project Structure

```
floorplan-to-3d/
├── backend/
│   ├── main.py                    # FastAPI server — POST /upload endpoint
│   ├── requirements.txt           # opencv, fastapi, uvicorn, anthropic, numpy
│   ├── pipeline/
│   │   ├── preprocess.py          # CLAHE contrast + deskew + perspective correction
│   │   ├── detect.py              # Room detection (flood-fill) + door detection (arc finder)
│   │   ├── polygons.py            # Polygon simplification + coordinate normalization to meters
│   │   └── export.py              # Final JSON schema builder
│   ├── evaluate.py                # Accuracy evaluation (IoU, F1, precision, recall)
│   ├── convert_cubicasa.py        # CubiCasa5K SVG → ground truth JSON converter
│   ├── run_eval.sh                # Automated eval pipeline (download dataset → evaluate)
│   └── tests/fixtures/            # Test image + ground truth JSON pairs
│
├── frontend/
│   ├── index.html                 # Split-panel UI (2D editor | 3D viewer)
│   ├── editor.js                  # Interactive 2D polygon editor (Canvas API)
│   └── viewer.js                  # 3D renderer (Three.js) with wall/door geometry
│
├── README.md                      # Setup instructions and API reference
└── ARCHITECTURE.md                # This file
```

## Detection Pipeline

The core processing happens in `backend/pipeline/`. Here is the flow when a user uploads an image:

```
Image Upload (PNG/JPG)
       │
       ▼
┌─────────────────────┐
│  1. PREPROCESS       │  preprocess.py
│  ─ CLAHE contrast    │  Normalizes lighting differences across
│  ─ Perspective fix   │  scanned/photographed floor plans
│  ─ Deskew rotation   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  2. DETECT ROOMS     │  detect.py :: detect_rooms_cv()
│  ─ Adaptive thresh   │  Converts image to binary (walls = black, rooms = white)
│  ─ Morphology close  │  Seals small wall gaps with configurable kernel (3x3 or 5x5)
│  ─ Flood-fill        │  Fills background from corners, leaving enclosed rooms
│  ─ Contour extract   │  Finds room boundaries from remaining white regions
│  ─ Parameter sweep   │  Tests 24 parameter combinations, scores by quality
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  3. DETECT DOORS     │  detect.py :: detect_doors_cv()
│  ─ Otsu threshold    │  Finds filled quarter-circle shapes (door swing arcs)
│  ─ Geometric filter  │  Filters by area, aspect ratio, solidity, arc ratio
│  ─ Hinge detection   │  Sharpest convex hull corner = hinge point on wall
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  4. CLASSIFY ROOMS   │  detect.py :: _classify_rooms_by_geometry()
│  ─ Area heuristics   │  >35% of total → living, >15% → bedroom, <4% → bathroom
│  ─ Aspect ratio      │  Narrow rooms (aspect < 0.3) → hallway
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  5. NORMALIZE        │  polygons.py
│  ─ Simplify polygons │  Ramer-Douglas-Peucker (epsilon = 2% of perimeter)
│  ─ Scale to meters   │  Longest image dimension → 15 meters
│  ─ Assign heights    │  3.0m standard, 2.7m bathroom, 2.8m kitchen
│  ─ Clamp door width  │  0.6m–1.5m realistic range
└──────────┬──────────┘
           │
           ▼
       JSON Response
```

If OpenCV detection finds fewer than 2 rooms, the pipeline falls back to the **Claude Vision API**, which analyzes the image and returns room polygons, types, and door positions via a structured prompt.

## Key Algorithms Explained

### Room Detection: Flood-Fill Approach

The core insight is that rooms in a floor plan are **enclosed white regions between dark walls**.

1. **Adaptive threshold** converts the grayscale image to binary, making walls white
2. **Morphological close + dilate** seals small gaps in walls (e.g., where doors are drawn)
3. **Bitwise inversion** flips so rooms are white, walls are black
4. A **white border** is added, then **flood-fill from (0,0)** turns the connected background black
5. What remains white are the enclosed rooms — extracted as contours

The algorithm sweeps 24 parameter combinations `(block_size × c_value × kernel_size)` and picks the result with the best **quality score**, which balances:
- Coverage: total room area should be 40–90% of the image
- Count: more rooms is better (log-scaled)
- Fragmentation penalty: many tiny rooms (<1% area) suggest noise
- Overlap penalty: rooms shouldn't overlap each other

### Door Detection: Quarter-Circle Arc Finder

Architectural floor plans represent doors as filled quarter-circle sectors showing the swing path. The detector:
1. Finds all contours in the binary image
2. Filters by geometric properties that match a quarter-circle:
   - Bounding box fill ratio: 45–88% (a quarter circle fills ~78.5% of its bbox)
   - Area-to-quarter-circle ratio: 50–120%
   - Convex hull solidity: ≥80%
   - Radius: 2–15% of image size
3. The **hinge point** (pivot on the wall) is the sharpest corner of the convex hull

### Polygon Simplification: Conditional Convex Hull

Detected room contours can be noisy. The simplification strategy preserves room shape:
- If solidity > 92% (nearly convex): apply convex hull, then approximate — this cleans up noise notches from door arcs
- If solidity ≤ 92% (L-shaped, T-shaped): approximate the raw contour directly — this preserves significant concavities

## Frontend Architecture

### 2D Editor (`editor.js`)

A Canvas-based interactive editor that displays the uploaded floor plan image with detected polygons overlaid. Supports:
- **Drag** polygon corners to adjust room boundaries
- **Double-click** an edge to insert a new vertex
- **Right-click** a vertex to delete it (minimum 3 vertices)
- **Scroll** on a door to rotate its swing angle
- **"Confirm & View 3D"** sends the edited data to the 3D viewer

Coordinates are stored in meters and converted to screen pixels via `toScreen()`/`fromScreen()` transforms based on the image dimensions.

### 3D Viewer (`viewer.js`)

Renders the floor plan as a 3D model using Three.js:

**Walls** — Each edge of a room polygon becomes a wall segment with 0.15m thickness, built as a box mesh (6 faces: outer, inner, top, bottom, two caps). When a door is detected on a wall segment:
- The wall is split into solid sections before/after the door
- A gap is left at door height (2.1m)
- A lintel piece fills from door height to ceiling

**Doors** — Each door is rendered as:
- A thin panel (BoxGeometry) pivoted from the hinge point
- Two frame posts (hinge side and latch side)
- A top beam connecting the posts
- A quarter-circle arc on the floor showing the swing path

**Scene** — PerspectiveCamera with OrbitControls, ambient + directional + hemisphere lighting, and a grid helper. Camera auto-fits to the bounding box of all geometry.

## Data Flow Between Frontend and Backend

```
Frontend                          Backend
────────                          ───────
                POST /upload
User drops ──────────────────►  main.py receives image
image file    (FormData)         │
                                 ├─ preprocess.py :: deskew()
                                 ├─ detect.py :: detect_rooms()
                                 ├─ polygons.py :: simplify + normalize
                                 ├─ export.py :: build JSON
                                 │
              JSON response      │
editor.js  ◄──────────────────  returns floorplan schema
draws 2D       { floorplan:
polygons        { rooms, doors,
                  scale, units }}
    │
    │ user clicks "Confirm"
    ▼
viewer.js
renders 3D
```

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
        "polygon": [
          { "x": 1.2, "y": 0.8 },
          { "x": 7.5, "y": 0.8 },
          { "x": 7.5, "y": 5.2 },
          { "x": 1.2, "y": 5.2 }
        ]
      }
    ],
    "doors": [
      {
        "id": "door_1",
        "position": { "x": 7.5, "y": 3.0 },
        "width": 0.9,
        "angle": 180,
        "connects": []
      }
    ]
  }
}
```

## Evaluation Pipeline

Measures detection accuracy against the **CubiCasa5K** dataset (5,000 real architectural floor plans with SVG annotations).

### How to Run

```bash
cd backend
./run_eval.sh                    # default: 5 samples, OpenCV mode
./run_eval.sh --limit 20         # evaluate 20 samples
./run_eval.sh --claude           # use Claude Vision API
./run_eval.sh --shuffle          # randomize sample selection
```

### Metrics

| Metric | What It Measures |
|--------|-----------------|
| Room F1 | Harmonic mean of precision (correct detections / total detections) and recall (correct detections / total ground truth) |
| Room IoU | Intersection over Union — how well detected room shapes overlap with ground truth polygons |
| Door F1 | Same as Room F1 but for doors, matched by position proximity (threshold: 0.08 normalized distance) |
| Type Accuracy | Among matched rooms, what fraction have the correct type label |

### Matching Strategy

- **Rooms**: Greedy matching using an IoU matrix. A detection matches ground truth if IoU ≥ 0.25. Best pairs are matched first.
- **Doors**: Nearest-neighbor matching by Euclidean distance between positions. Threshold: 0.08 (normalized to image dimensions).

### Current Results (30 samples)

```
Avg Room F1:  46.2%
Avg Room IoU: 46.8%
Avg Door F1:  32.0%
```

### Ground Truth Conversion

`convert_cubicasa.py` parses CubiCasa5K SVG files:
- Extracts `<polygon>` elements inside `<g class="Space ...">` groups as rooms
- Maps room types (e.g., `LivingRoom` → `living`, `Bath` → `bathroom`)
- Extracts door polygons and computes their centroids as positions
- Normalizes all coordinates to 0–1 range using the SVG viewBox dimensions

## How to Run the Project

### Backend

```bash
cd backend
pip install -r requirements.txt
export ANTHROPIC_API_KEY="sk-..."    # only needed for Claude Vision fallback
uvicorn main:app --reload            # starts on http://localhost:8000
```

### Frontend

Serve the `frontend/` directory on port 5500 (configured in CORS):

```bash
cd frontend
python3 -m http.server 5500
# or use VS Code Live Server extension
```

Open `http://localhost:5500` in browser, upload a floor plan image.
