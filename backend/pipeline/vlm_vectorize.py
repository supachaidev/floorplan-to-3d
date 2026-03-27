"""
Hybrid VLM pipeline: OpenCV geometry + Gemini room labeling.

Uses OpenCV for accurate room polygon and door detection, then sends
the image to Gemini to classify/label each detected room with a
descriptive name (e.g. "Master Bedroom", "Kitchen", "Bathroom 1").
"""

import base64
import io
import json
import logging
import os
import re

from PIL import Image

logger = logging.getLogger(__name__)

LABEL_SYSTEM_PROMPT = """You are an expert at reading architectural floor plans.
You will be given a floor plan image and a list of detected rooms with their
approximate positions. Your job is to identify what each room is and give it
a descriptive label."""

LABEL_USER_PROMPT = """Look at this floor plan image. I have detected {n_rooms} rooms
using computer vision. Each room is described by its approximate center position
(as fraction of image width/height, where 0,0 is top-left and 1,1 is bottom-right)
and its relative size.

Here are the detected rooms:
{room_descriptions}

For each room, determine what type of room it is based on:
- Its position in the floor plan
- Its size relative to other rooms
- Any text labels visible in the image
- Typical floor plan conventions (e.g. bathrooms have fixtures, kitchens have counters)

Common room types: bedroom, bathroom, kitchen, living room, dining room, hallway,
corridor, closet, walk-in closet, storage, laundry, pantry, balcony, foyer,
garage, office, utility room, master bedroom, guest room, en-suite.

Return a JSON array with one object per room, in the SAME ORDER as the input:
[
  {{"room_index": 0, "label": "Living Room", "type": "living"}},
  {{"room_index": 1, "label": "Master Bedroom", "type": "bedroom"}},
  ...
]

Valid types: bedroom, bathroom, kitchen, living, dining, hallway, closet, balcony,
laundry, storage, garage, office, foyer, pantry, utility, other.

Return ONLY the JSON array. No explanation, no markdown fences."""

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
        with urllib.request.urlopen(req, timeout=120) as resp:
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


def _build_room_descriptions(rooms: list[dict]) -> str:
    """Build a text description of detected rooms for the VLM prompt."""
    lines = []
    total_area = sum(r.get("_area_frac", 0) for r in rooms)
    for i, room in enumerate(rooms):
        poly = room["polygon"]
        # Compute centroid
        cx = sum(p["x"] for p in poly) / len(poly)
        cy = sum(p["y"] for p in poly) / len(poly)
        area_frac = room.get("_area_frac", 0)
        area_pct = (area_frac / total_area * 100) if total_area > 0 else 0
        n_corners = len(poly)
        lines.append(
            f"Room {i}: center=({cx:.2f}, {cy:.2f}), "
            f"{area_pct:.0f}% of total floor area, "
            f"{n_corners} corners"
        )
    return "\n".join(lines)


def label_rooms_with_vlm(
    image_rgb,
    rooms: list[dict],
    model_name: str | None = None,
) -> list[dict]:
    """Use Gemini to label CV-detected rooms.

    Args:
        image_rgb: numpy array (RGB) or PIL Image.
        rooms: list of room dicts with "polygon" and "_area_frac" keys.
        model_name: Gemini model name.

    Returns:
        The rooms list with updated "label" and "type" fields.
    """
    if not rooms:
        return rooms

    model = model_name or DEFAULT_MODEL

    # Convert to PIL if needed
    if not isinstance(image_rgb, Image.Image):
        image = Image.fromarray(image_rgb)
    else:
        image = image_rgb
    image = _resize_if_needed(image)
    image_b64 = _image_to_base64(image)

    room_descriptions = _build_room_descriptions(rooms)
    prompt = LABEL_USER_PROMPT.format(
        n_rooms=len(rooms),
        room_descriptions=room_descriptions,
    )

    payload = {
        "system_instruction": {"parts": [{"text": LABEL_SYSTEM_PROMPT}]},
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"inline_data": {"mime_type": "image/jpeg", "data": image_b64}},
                    {"text": prompt},
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": 4096,
            "responseMimeType": "application/json",
        },
    }

    try:
        raw = _call_gemini_api(payload, model)
        logger.info("VLM labeling output: %s", raw[:500])

        # Parse response
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        labels = json.loads(clean)

        if not isinstance(labels, list):
            logger.warning("VLM labels not a list, ignoring")
            return rooms

        # Apply labels to rooms
        for item in labels:
            idx = item.get("room_index", -1)
            if 0 <= idx < len(rooms):
                rooms[idx]["label"] = item.get("label", rooms[idx]["label"])
                rooms[idx]["type"] = item.get("type", rooms[idx].get("type", "other"))

        logger.info("VLM labeled %d/%d rooms", len(labels), len(rooms))

    except Exception as e:
        logger.warning("VLM labeling failed, falling back to geometry labels: %s", e)
        # Fall back — assign generic labels
        for i, room in enumerate(rooms):
            if room.get("label", "").startswith("Room "):
                room["type"] = room.get("type", "other")

    # Clean up internal keys used for labeling
    for room in rooms:
        room.pop("_area_frac", None)
        room.pop("_aspect", None)
        # Ensure every room has height and type
        room.setdefault("type", "other")
        room.setdefault("height", 3.0)

    return rooms
