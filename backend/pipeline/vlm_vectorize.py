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

SYSTEM_PROMPT = """You are an architectural floor plan vectorization assistant.
Your task is to analyze a raster floor plan image and output a structured JSON
representing the geometric topology.

Rules:
1. Define all walls FIRST as independent primitives with unique IDs (w1, w2, ...).
2. Define rooms SECOND, referencing wall IDs only.
3. Define openings (doors/windows) LAST, each referencing a parent wall.
4. ALL coordinates must be NORMALIZED fractions from 0.0 to 1.0, where:
   - (0.0, 0.0) = top-left corner of the image
   - (1.0, 1.0) = bottom-right corner of the image
   For example, a point in the center of the image is (0.5, 0.5).
5. Wall thickness is a normalized fraction (e.g. 0.02 for a typical wall).
6. You MUST detect EVERY room in the floor plan. Do not skip any room, even small
   ones like closets, hallways, storage, laundry, pantry, or utility rooms.
7. Common room types: bedroom, bathroom, kitchen, living, dining, hallway, closet,
   balcony, laundry, storage, garage, office, foyer, pantry, utility, other.
8. Each room's floor_polygon must trace the INNER boundary of that room tightly.
   Polygons should be closed (last point connects back to first).
9. Output ONLY valid JSON. No explanation, no markdown, no extra text."""

USER_PROMPT = """Vectorize this floor plan image.

IMPORTANT: All coordinates must be normalized fractions between 0.0 and 1.0
relative to the image dimensions. (0,0) is top-left, (1,1) is bottom-right.

First, carefully scan the ENTIRE image and count how many distinct enclosed spaces
(rooms) exist. Include every room regardless of size — bedrooms, bathrooms, kitchen,
living room, hallways, closets, storage, balcony, etc. Then output ALL of them.

For each room, trace its floor_polygon by following the inner edges of its walls,
producing a tight polygon with corners at wall intersections.

Output the following JSON structure exactly:
{
  "walls": [
    { "id": "w1", "start": [x, y], "end": [x, y], "thickness": <fraction> }
  ],
  "rooms": [
    {
      "id": "r1",
      "label": "<room type>",
      "wall_ids": ["w1", ...],
      "floor_polygon": [[x,y], ...]
    }
  ],
  "openings": [
    { "id": "o1", "type": "door|window", "wall_id": "w1", "position": <fraction along wall 0-1>, "width": <fraction> }
  ]
}"""

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


def _call_gemini(image: Image.Image, model_name: str) -> str:
    """Call Gemini API with the image and return raw text response."""
    import urllib.request

    api_key = _get_api_key()
    image_b64 = _image_to_base64(image)

    user_prompt = USER_PROMPT

    payload = {
        "system_instruction": {
            "parts": [{"text": SYSTEM_PROMPT}]
        },
        "contents": [
            {
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": image_b64,
                        }
                    },
                    {"text": user_prompt},
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": 8192,
            "responseMimeType": "application/json",
        },
    }

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
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini API error ({e.code}): {body}") from e

    # Extract text from response
    try:
        return result["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as e:
        logger.error("Unexpected Gemini response: %s", result)
        raise RuntimeError("Gemini returned an unexpected response format") from e


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
