"""
VLM-guided room segmentation: Gemini identifies room centers,
OpenCV flood-fills from those seeds for precise polygon extraction.

This inverts the existing hybrid pipeline — instead of OpenCV finding
rooms and Gemini labeling them, Gemini tells us WHERE rooms are and
OpenCV extracts precise boundaries at those locations.
"""

import json
import logging
import re

import cv2
import numpy as np
from PIL import Image

from pipeline.vlm_vectorize import (
    _call_gemini_api,
    _get_api_key,
    _image_to_base64,
    _resize_if_needed,
)
from pipeline.detect import _remove_small_components, detect_doors_cv

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-2.5-flash"
_MAX_DETECT_DIM = 1200

SEED_SYSTEM_PROMPT = """You are an expert at reading architectural floor plans.
You identify every room in a floor plan and pinpoint its center position."""

SEED_USER_PROMPT = """Look at this floor plan image carefully. Identify every
enclosed room or space (bedrooms, bathrooms, kitchen, living room, dining room,
hallway, closet, balcony, storage, etc.).

For each room, provide:
- "label": a descriptive name (e.g. "Master Bedroom", "Kitchen")
- "type": one of: bedroom, bathroom, kitchen, living, dining, hallway, closet,
  balcony, laundry, storage, garage, office, foyer, pantry, utility, other
- "center_x": the approximate x-position of the room's center as a fraction
  of image width (0.0 = left edge, 1.0 = right edge)
- "center_y": the approximate y-position of the room's center as a fraction
  of image height (0.0 = top edge, 1.0 = bottom edge)

Be precise with center positions — they must land INSIDE the room, not on a wall.

Return ONLY a JSON array. No explanation, no markdown fences.
Example:
[
  {"label": "Living Room", "type": "living", "center_x": 0.35, "center_y": 0.40},
  {"label": "Kitchen", "type": "kitchen", "center_x": 0.70, "center_y": 0.55}
]"""

# Flood fill parameters
_SEED_NUDGE_RADIUS = 20
_MIN_ROOM_AREA_FRAC = 0.003
_MAX_ROOM_AREA_FRAC = 0.8


def get_room_seeds_from_vlm(
    image_rgb, model_name: str | None = None,
) -> list[dict]:
    """Ask Gemini to identify rooms and their center positions."""
    model = model_name or DEFAULT_MODEL

    if not isinstance(image_rgb, Image.Image):
        image = Image.fromarray(image_rgb)
    else:
        image = image_rgb
    image = _resize_if_needed(image)
    image_b64 = _image_to_base64(image)

    payload = {
        "system_instruction": {"parts": [{"text": SEED_SYSTEM_PROMPT}]},
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"inline_data": {"mime_type": "image/jpeg", "data": image_b64}},
                    {"text": SEED_USER_PROMPT},
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": 4096,
            "responseMimeType": "application/json",
        },
    }

    raw = _call_gemini_api(payload, model)
    logger.info("VLM seed response: %s", raw[:500])

    clean = re.sub(r"```(?:json)?|```", "", raw).strip()
    seeds = json.loads(clean)

    if not isinstance(seeds, list):
        raise RuntimeError("Gemini returned non-list response for room seeds")

    # Validate and clamp
    valid = []
    for s in seeds[:30]:  # cap at 30 rooms
        try:
            cx = max(0.02, min(0.98, float(s["center_x"])))
            cy = max(0.02, min(0.98, float(s["center_y"])))
            valid.append({
                "label": str(s.get("label", "Room")),
                "type": str(s.get("type", "other")),
                "cx": cx,
                "cy": cy,
            })
        except (KeyError, ValueError, TypeError):
            continue

    if not valid:
        raise RuntimeError("Gemini returned no valid room seeds")

    logger.info("VLM identified %d room seeds", len(valid))
    return valid


def _prepare_binary_for_flood(gray: np.ndarray) -> np.ndarray:
    """Create wall mask from grayscale image. Walls = 255, rooms = 0."""
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 4,
    )
    binary = _remove_small_components(binary)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    walls = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
    walls_h = cv2.morphologyEx(walls, cv2.MORPH_CLOSE, h_kernel, iterations=1)
    walls_v = cv2.morphologyEx(walls, cv2.MORPH_CLOSE, v_kernel, iterations=1)
    walls = cv2.bitwise_or(walls_h, walls_v)

    walls = cv2.dilate(walls, kernel, iterations=1)
    return walls


def _prepare_binary_thin(gray: np.ndarray) -> np.ndarray:
    """Wall mask for thin-line drawings."""
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), np.uint8)
    walls = cv2.dilate(binary, kernel, iterations=2)
    return walls


def _find_valid_seed(walls: np.ndarray, px: int, py: int) -> tuple[int, int] | None:
    """Find a non-wall pixel near (px, py). BFS spiral outward."""
    h, w = walls.shape
    px = max(0, min(w - 1, px))
    py = max(0, min(h - 1, py))

    if walls[py, px] == 0:
        return (px, py)

    # BFS search for nearest non-wall pixel
    for r in range(1, _SEED_NUDGE_RADIUS + 1):
        for dx in range(-r, r + 1):
            for dy in [-r, r]:
                nx, ny = px + dx, py + dy
                if 0 <= nx < w and 0 <= ny < h and walls[ny, nx] == 0:
                    return (nx, ny)
        for dy in range(-r + 1, r):
            for dx in [-r, r]:
                nx, ny = px + dx, py + dy
                if 0 <= nx < w and 0 <= ny < h and walls[ny, nx] == 0:
                    return (nx, ny)

    return None


def _flood_fill_room(
    walls: np.ndarray, sx: int, sy: int, w: int, h: int,
) -> np.ndarray | None:
    """Flood fill from seed on inverted wall mask. Returns room binary mask."""
    rooms_mask = cv2.bitwise_not(walls)

    flood_img = rooms_mask.copy()
    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

    # Fill with a marker value (128)
    cv2.floodFill(flood_img, mask, (sx, sy), 128)

    # Extract just the filled region
    room_mask = np.where(flood_img == 128, 255, 0).astype(np.uint8)

    area = cv2.countNonZero(room_mask)
    total = w * h
    area_frac = area / total

    if area_frac < _MIN_ROOM_AREA_FRAC or area_frac > _MAX_ROOM_AREA_FRAC:
        return None

    return room_mask


def _mask_to_polygon(
    mask: np.ndarray, w: int, h: int,
) -> list[dict] | None:
    """Extract polygon from binary mask, return normalized coordinates."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.015 * peri, True)

    if len(approx) < 3:
        return None

    polygon = []
    for pt in approx:
        px, py = pt[0]
        polygon.append({"x": round(px / w, 4), "y": round(py / h, 4)})

    return polygon


def segment_rooms_with_vlm(
    image_bgr: np.ndarray, model_name: str | None = None,
) -> dict:
    """VLM-guided room segmentation.

    1. Ask Gemini to identify room centers
    2. Flood fill from each center on thresholded image
    3. Extract precise polygons

    Returns {"rooms": [...], "doors": [...]}.
    """
    h, w = image_bgr.shape[:2]

    # Downscale for processing
    scale_factor = 1.0
    if max(h, w) > _MAX_DETECT_DIM:
        scale_factor = _MAX_DETECT_DIM / max(h, w)
        new_w, new_h = int(w * scale_factor), int(h * scale_factor)
        proc = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        h, w = new_h, new_w
    else:
        proc = image_bgr

    # Get room seeds from Gemini
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    seeds = get_room_seeds_from_vlm(image_rgb, model_name)

    # Prepare wall mask
    gray = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
    dark_ratio = np.sum(gray < 128) / gray.size
    white_ratio = np.sum(gray > 230) / gray.size

    if dark_ratio < 0.02 and white_ratio > 0.85:
        logger.info("Thin-line image detected for VLM segmentation")
        walls = _prepare_binary_thin(gray)
    else:
        walls = _prepare_binary_for_flood(gray)

    # Flood fill from each seed
    rooms = []
    claimed = walls.copy()  # track claimed regions

    for seed in seeds:
        px = int(seed["cx"] * w)
        py = int(seed["cy"] * h)

        valid = _find_valid_seed(claimed, px, py)
        if valid is None:
            logger.warning(
                "Seed for '%s' at (%.2f, %.2f) landed on wall, skipping",
                seed["label"], seed["cx"], seed["cy"],
            )
            continue

        sx, sy = valid
        room_mask = _flood_fill_room(claimed, sx, sy, w, h)
        if room_mask is None:
            logger.warning(
                "Seed for '%s' produced invalid region, skipping",
                seed["label"],
            )
            continue

        polygon = _mask_to_polygon(room_mask, w, h)
        if polygon is None:
            continue

        # Mark this region as claimed so other seeds don't overlap
        claimed[room_mask > 0] = 255

        area_frac = cv2.countNonZero(room_mask) / (w * h)
        rooms.append({
            "label": seed["label"],
            "type": seed["type"],
            "polygon": polygon,
            "_area_frac": area_frac,
            "_aspect": 1.0,
            "height": 3.0,
        })

    logger.info(
        "VLM segmentation: %d/%d seeds produced rooms",
        len(rooms), len(seeds),
    )

    # Detect doors (reuse existing CV detector)
    doors = detect_doors_cv(proc)

    return {"rooms": rooms, "doors": doors}
