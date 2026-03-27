"""
VLM-based floor plan vectorization using Google Gemini API.

Produces a wall-first JSON schema where:
- Walls are defined as independent primitives with unique IDs
- Rooms reference wall IDs (dependency-ordered)
- Openings (doors/windows) reference parent walls with position offsets
"""

import base64
import io
import json
import logging
import os
import re

from PIL import Image

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert architectural floor plan digitizer.
You convert raster floor plan images into precise, structured JSON geometry.
You are meticulous about coordinate accuracy and never miss rooms."""

USER_PROMPT = """Analyze this floor plan image and extract its geometry as JSON.

## Coordinate system

All coordinates are NORMALIZED fractions from 0.0 to 1.0 relative to the image:
- (0.0, 0.0) = top-left corner
- (1.0, 1.0) = bottom-right corner
- The image center is (0.5, 0.5)

Be precise — estimate coordinates to 2-3 decimal places by carefully looking at
where each wall and corner sits relative to the full image width and height.

## Step-by-step process

Follow these steps IN ORDER:

### Step 1: Inventory all rooms
Scan the entire image. List EVERY distinct enclosed space you can see — bedrooms,
bathrooms, kitchen, living room, dining room, hallways, corridors, closets,
walk-in closets, storage rooms, laundry, pantry, balconies, foyer, garage,
utility rooms, etc. Do NOT skip small spaces. A typical apartment has 8-15 spaces.

### Step 2: Trace walls
For each wall segment in the floor plan, define it with a start and end point.
- Walls should run along the CENTER LINE of the drawn wall
- Where walls meet at corners, their endpoints should share the same coordinates
- Typical wall thickness is 0.015–0.025 in normalized coordinates
- Include both exterior walls and interior partition walls

### Step 3: Define room polygons
For each room, trace its floor_polygon as the INNER boundary of the room:
- Walk around the room clockwise, listing each corner point
- Corners should be at the INNER edges of wall intersections
- Rectangular rooms have exactly 4 corners
- L-shaped rooms have 6 corners, etc.
- The polygon must be CLOSED (first point connects back to last)

### Step 4: Find windows
Scan exterior walls for windows:
- Windows appear as thin parallel lines or hatching across a wall segment
- They look like a narrow rectangle embedded in the wall

For each window:
- "position" is a fraction (0.0–1.0) along the parent wall from its start to end
- "width" is the opening width as a fraction of the image (typically 0.04–0.08)

Note: doors are detected separately — you only need to output windows as openings.

## Output format

Return ONLY this JSON structure — no explanation, no markdown fences:
{
  "walls": [
    {"id": "w1", "start": [0.10, 0.15], "end": [0.65, 0.15], "thickness": 0.02}
  ],
  "rooms": [
    {
      "id": "r1",
      "label": "Living Room",
      "wall_ids": ["w1", "w2", "w3", "w4"],
      "floor_polygon": [[0.11, 0.16], [0.64, 0.16], [0.64, 0.48], [0.11, 0.48]]
    }
  ],
  "openings": [
    {"id": "o1", "type": "window", "wall_id": "w2", "position": 0.5, "width": 0.08}
  ]
}

## Important reminders
- Include EVERY room — missing rooms is the most common mistake
- Coordinates must be 0.0–1.0 normalized, NOT pixels
- Room polygons should be tight to inner wall edges, not overlapping other rooms
- Adjacent rooms should share wall edges (no gaps between rooms)
- Label rooms with descriptive names (e.g. "Master Bedroom", "Bathroom 1")
- Doors are detected separately — only include windows in openings"""

DEFAULT_MODEL = "gemini-2.5-flash"
MAX_IMAGE_PIXELS = 2048 * 2048


def _get_api_key() -> str:
    """Get Gemini API key from environment."""
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is not set. "
            "Get a free API key at https://aistudio.google.com/apikey"
        )
    return key


def _resize_if_needed(image: Image.Image) -> Image.Image:
    """Downscale image if it exceeds MAX_IMAGE_PIXELS to reduce token usage."""
    w, h = image.size
    if w * h <= MAX_IMAGE_PIXELS:
        return image
    scale = (MAX_IMAGE_PIXELS / (w * h)) ** 0.5
    new_w, new_h = int(w * scale), int(h * scale)
    logger.info("Resizing image from %dx%d to %dx%d for VLM", w, h, new_w, new_h)
    return image.resize((new_w, new_h), Image.LANCZOS)


def _image_to_base64(image: Image.Image) -> str:
    """Convert PIL image to base64-encoded JPEG."""
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


REFINE_PROMPT = """Look at the floor plan image again and review your previous JSON output below.

{previous_json}

## Check for these common issues:

1. **Missing rooms**: Count every enclosed space in the image. Did you miss any
   small rooms like closets, hallways, storage, pantry, laundry, or utility rooms?

2. **Coordinate accuracy**: Check that the polygon coordinates actually match
   where the rooms are in the image. Are rooms in the correct relative positions?
   Are adjacent room polygons sharing edges without gaps or overlaps?

3. **Wall connectivity**: Do walls that should meet at corners share the same
   endpoint coordinates? Are there any gaps in the wall network?

4. **Missing windows**: Look for parallel lines or hatching on exterior walls.

Output the COMPLETE corrected JSON (not just the changes). Keep the same structure.
If everything looks correct, output the same JSON unchanged."""


def _call_gemini_api(payload: dict, model_name: str) -> str:
    """Make a single Gemini API call and return the text response."""
    import urllib.request

    api_key = _get_api_key()
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}"
        f":generateContent?key={api_key}"
    )

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini API error ({e.code}): {body}") from e

    # Extract text from response — with thinking mode, skip thought parts
    try:
        parts = result["candidates"][0]["content"]["parts"]
        for part in reversed(parts):
            if "text" in part and not part.get("thought"):
                return part["text"]
        raise KeyError("No text part found")
    except (KeyError, IndexError) as e:
        logger.error("Unexpected Gemini response: %s", result)
        raise RuntimeError("Gemini returned an unexpected response format") from e


def _make_generation_config(json_mode: bool = True) -> dict:
    config = {
        "temperature": 0,
        "maxOutputTokens": 16384,
        "thinkingConfig": {"thinkingBudget": 8192},
    }
    if json_mode:
        config["responseMimeType"] = "application/json"
    return config


def _call_gemini(image: Image.Image, model_name: str) -> str:
    """Call Gemini with initial extraction then a refinement pass."""
    image_b64 = _image_to_base64(image)
    image_part = {
        "inline_data": {"mime_type": "image/jpeg", "data": image_b64}
    }

    # --- Pass 1: Initial extraction ---
    logger.info("VLM pass 1: initial extraction")
    payload = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [
            {"role": "user", "parts": [image_part, {"text": USER_PROMPT}]}
        ],
        "generationConfig": _make_generation_config(),
    }
    initial_json = _call_gemini_api(payload, model_name)
    logger.info("VLM pass 1 output length: %d chars", len(initial_json))

    # Quick-validate before refinement
    initial_data = _try_parse(initial_json)
    if initial_data is None:
        logger.warning("Pass 1 output failed to parse, skipping refinement")
        return initial_json

    n_rooms = len(initial_data.get("rooms", []))
    n_openings = len(initial_data.get("openings", []))
    logger.info("VLM pass 1: %d rooms, %d openings", n_rooms, n_openings)

    # --- Pass 2: Refinement with the image shown again ---
    logger.info("VLM pass 2: refinement")
    refine_text = REFINE_PROMPT.format(previous_json=initial_json)
    payload_refine = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [
            {"role": "user", "parts": [image_part, {"text": USER_PROMPT}]},
            {"role": "model", "parts": [{"text": initial_json}]},
            {"role": "user", "parts": [image_part, {"text": refine_text}]},
        ],
        "generationConfig": _make_generation_config(),
    }
    refined_json = _call_gemini_api(payload_refine, model_name)
    logger.info("VLM pass 2 output length: %d chars", len(refined_json))

    # Use refined output if it parses, otherwise fall back to initial
    refined_data = _try_parse(refined_json)
    if refined_data is None:
        logger.warning("Pass 2 output failed to parse, using pass 1 result")
        return initial_json

    n_rooms_r = len(refined_data.get("rooms", []))
    n_openings_r = len(refined_data.get("openings", []))
    logger.info("VLM pass 2: %d rooms, %d openings", n_rooms_r, n_openings_r)

    return refined_json


def _try_parse(raw: str) -> dict | None:
    """Try to parse JSON, return dict or None."""
    clean = re.sub(r"```(?:json)?|```", "", raw).strip()
    try:
        data = json.loads(clean)
        if isinstance(data, dict) and "walls" in data and "rooms" in data:
            return data
    except (json.JSONDecodeError, TypeError):
        pass
    return None


def vectorize_floorplan(image_path: str, model_name: str | None = None) -> dict | None:
    """
    Vectorize a floor plan image using Gemini API.

    Args:
        image_path: Path to the floor plan image.
        model_name: Optional model override (default: gemini-2.0-flash).

    Returns:
        Parsed JSON dict with walls, rooms, and openings, or None on failure.
    """
    image = Image.open(image_path).convert("RGB")
    image = _resize_if_needed(image)

    raw = _call_gemini(image, model_name or DEFAULT_MODEL)
    logger.info("VLM raw output length: %d chars", len(raw))
    return parse_vlm_output(raw)


def vectorize_floorplan_from_array(
    image_rgb,
    model_name: str | None = None,
) -> dict | None:
    """
    Vectorize from a numpy array (RGB) — used by the upload endpoint.

    Args:
        image_rgb: numpy array in RGB format (H, W, 3).
        model_name: Optional model override.

    Returns:
        Parsed JSON dict or None.
    """
    image = Image.fromarray(image_rgb)
    image = _resize_if_needed(image)

    raw = _call_gemini(image, model_name or DEFAULT_MODEL)
    logger.info("VLM raw output length: %d chars", len(raw))
    return parse_vlm_output(raw)


def parse_vlm_output(raw: str) -> dict | None:
    """Parse and validate VLM text output into the wall-first schema."""
    # Strip markdown fences if model wraps output
    clean = re.sub(r"```(?:json)?|```", "", raw).strip()

    try:
        data = json.loads(clean)
    except json.JSONDecodeError as e:
        logger.error("JSON parse failed: %s", e)
        logger.debug("Raw VLM output: %s", raw)
        return None

    # Validate required top-level keys
    if not isinstance(data, dict):
        logger.error("VLM output is not a JSON object")
        return None

    if "walls" not in data or "rooms" not in data:
        logger.error("VLM output missing 'walls' or 'rooms'")
        return None

    # Validate and normalize walls
    wall_ids = set()
    for wall in data["walls"]:
        if "id" not in wall or "start" not in wall or "end" not in wall:
            logger.warning("Skipping wall with missing fields: %s", wall)
            continue
        wall_ids.add(wall["id"])
        # Ensure start/end are lists of 2 numbers
        wall["start"] = [float(wall["start"][0]), float(wall["start"][1])]
        wall["end"] = [float(wall["end"][0]), float(wall["end"][1])]
        wall["thickness"] = float(wall.get("thickness", 0.02))

    # Validate rooms
    for room in data["rooms"]:
        if "id" not in room:
            logger.warning("Room missing 'id': %s", room)
        # Ensure floor_polygon is a list of [x,y] pairs
        if "floor_polygon" in room:
            room["floor_polygon"] = [
                [float(p[0]), float(p[1])] for p in room["floor_polygon"]
            ]

    # Ensure openings exist (may be empty)
    data.setdefault("openings", [])
    data.setdefault("scale", {"pixels_per_meter": 50})

    logger.info(
        "VLM detected %d walls, %d rooms, %d openings",
        len(data["walls"]),
        len(data["rooms"]),
        len(data["openings"]),
    )
    for room in data["rooms"]:
        logger.info("  Room %s: %s (%d polygon pts)",
                     room.get("id"), room.get("label"),
                     len(room.get("floor_polygon", [])))

    return data


def _project_point_to_segment(px, py, ax, ay, bx, by):
    """Project point (px,py) onto segment (ax,ay)-(bx,by). Return (t, dist)."""
    dx, dy = bx - ax, by - ay
    len2 = dx * dx + dy * dy
    if len2 < 1e-12:
        return 0.5, ((px - ax) ** 2 + (py - ay) ** 2) ** 0.5
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / len2))
    proj_x = ax + t * dx
    proj_y = ay + t * dy
    dist = ((px - proj_x) ** 2 + (py - proj_y) ** 2) ** 0.5
    return t, dist


def merge_cv_doors(vlm_data: dict, cv_doors: list[dict]) -> dict:
    """Replace VLM openings with CV-detected doors snapped to VLM walls.

    CV doors have accurate positions from arc detection. This function
    projects each CV door onto the nearest VLM wall to produce openings
    in the wall-first schema format.

    Args:
        vlm_data: Parsed VLM output with walls, rooms, openings.
        cv_doors: List of CV-detected doors with normalized position {x, y},
                  width, and angle.

    Returns:
        Updated vlm_data with merged openings.
    """
    walls = vlm_data.get("walls", [])
    if not walls:
        logger.warning("No VLM walls to snap CV doors to")
        return vlm_data

    # Keep VLM windows, replace doors
    vlm_windows = [o for o in vlm_data.get("openings", []) if o.get("type") == "window"]

    new_openings = list(vlm_windows)
    snap_threshold = 0.08  # max distance to snap a door to a wall

    for i, door in enumerate(cv_doors):
        dx = door["position"]["x"]
        dy = door["position"]["y"]

        best_wall = None
        best_t = 0.5
        best_dist = snap_threshold

        for wall in walls:
            ax, ay = wall["start"]
            bx, by = wall["end"]
            t, dist = _project_point_to_segment(dx, dy, ax, ay, bx, by)
            if dist < best_dist:
                best_dist = dist
                best_wall = wall
                best_t = t

        if best_wall is not None:
            new_openings.append({
                "id": f"o_door_{i + 1}",
                "type": "door",
                "wall_id": best_wall["id"],
                "position": round(best_t, 4),
                "width": round(door["width"], 4),
            })
            logger.info(
                "CV door %d snapped to wall %s at t=%.3f (dist=%.4f)",
                i + 1, best_wall["id"], best_t, best_dist,
            )
        else:
            logger.warning(
                "CV door %d at (%.3f, %.3f) too far from any wall, skipped",
                i + 1, dx, dy,
            )

    vlm_data["openings"] = new_openings
    logger.info(
        "Merged: %d CV doors + %d VLM windows = %d openings",
        len(new_openings) - len(vlm_windows),
        len(vlm_windows),
        len(new_openings),
    )
    return vlm_data
