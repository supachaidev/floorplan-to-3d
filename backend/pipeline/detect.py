import base64
import json
import logging
import math
import re

import anthropic
import cv2
import numpy as np

logger = logging.getLogger(__name__)


def detect_rooms_cv(image: np.ndarray) -> list[dict] | None:
    """Detect rooms using OpenCV for clean printed plans.

    Uses flood-fill to find enclosed white regions (rooms) between walls.
    Sweeps multiple threshold parameters and picks the best result.

    Returns a list of room dicts or None if detection fails.
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Normalize contrast for varying image quality
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    best_rooms = None
    best_score = -1.0

    # Sweep threshold and morphology parameters
    for block_size in [11, 15, 21, 31]:
        for c_val in [3, 5, 8]:
            for morph_k in [3, 5]:
                rooms = _find_enclosed_rooms(
                    enhanced, w, h, block_size, c_val, morph_k,
                )
                if rooms is None:
                    continue
                score = _score_room_set(rooms)
                if score > best_score:
                    best_score = score
                    best_rooms = rooms

    if best_rooms is None:
        return None

    _classify_rooms_by_geometry(best_rooms)
    return best_rooms


def _score_room_set(rooms: list[dict]) -> float:
    """Score a set of detected rooms by quality, not just count.

    Prefers results where total room area covers 40-90% of the image
    with minimal mutual overlap and uniform room sizes (not fragmented).
    """
    if not rooms:
        return -1.0

    areas = [r["_area_frac"] for r in rooms]
    total_area = sum(areas)

    # Penalize coverage outside the sweet spot (40-90% of image)
    if total_area < 0.4:
        coverage_score = total_area / 0.4
    elif total_area > 0.9:
        coverage_score = max(0, 1.0 - (total_area - 0.9) / 0.5)
    else:
        coverage_score = 1.0

    # Reward more rooms (log scale, capped to avoid runaway)
    count_score = min(math.log2(max(len(rooms), 1)) / 5.0, 0.5)

    # Penalize fragmentation: if many rooms are tiny (< 1% of image),
    # the result is likely noise from aggressive thresholding
    tiny_count = sum(1 for a in areas if a < 0.01)
    frag_penalty = 0.3 * (tiny_count / max(len(rooms), 1))

    # Penalize overlap between rooms using mask-based comparison
    overlap_penalty = 0.0
    if len(rooms) > 1:
        size = 200
        masks = []
        for r in rooms:
            mask = np.zeros((size, size), dtype=np.uint8)
            pts = np.array(
                [[int(p["x"] * size), int(p["y"] * size)] for p in r["polygon"]],
                dtype=np.int32,
            )
            if len(pts) >= 3:
                cv2.fillPoly(mask, [pts], 255)
            masks.append(mask)
        combined = np.zeros((size, size), dtype=np.uint8)
        double_count = np.zeros((size, size), dtype=np.uint8)
        for m in masks:
            double_count[np.logical_and(combined > 0, m > 0)] = 255
            combined = np.maximum(combined, m)
        if combined.sum() > 0:
            overlap_penalty = 1.5 * double_count.sum() / combined.sum()

    return coverage_score + count_score - frag_penalty - overlap_penalty


def _find_enclosed_rooms(
    gray: np.ndarray, w: int, h: int, block_size: int, c_val: int,
    morph_k: int = 3,
) -> list[dict] | None:
    """Find rooms as enclosed white regions between detected walls."""
    # Detect walls (dark lines become white)
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block_size, c_val,
    )

    # Close small gaps in walls, then dilate to seal door openings
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_k, morph_k))
    walls = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    walls = cv2.dilate(walls, kernel, iterations=1)

    # Invert: rooms become white (255), walls become black (0)
    rooms_mask = cv2.bitwise_not(walls)

    # Flood fill background from edges to isolate enclosed rooms.
    # Add a white border so the background is guaranteed to be connected
    # to the corner, then flood fill from (0,0) to turn background black.
    bordered = cv2.copyMakeBorder(
        rooms_mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255,
    )
    flood_mask = np.zeros(
        (bordered.shape[0] + 2, bordered.shape[1] + 2), dtype=np.uint8,
    )
    cv2.floodFill(bordered, flood_mask, (0, 0), 0)
    # Remove added border — only enclosed rooms remain white
    rooms_only = bordered[1:-1, 1:-1]

    contours, _ = cv2.findContours(
        rooms_only, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )

    min_area = (w * h) * 0.005  # 0.5% — catch small rooms like closets
    max_area = (w * h) * 0.8

    rooms = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        # Approximate contour, preserving significant concavities (L-shapes)
        # but smoothing noise. Use hull only when shape is nearly convex.
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) < 4:
            continue

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 1.0
        # If shape is >92% convex, use hull (removes noise notches)
        # Otherwise keep the concave approximation (L/T shapes)
        if solidity > 0.92:
            approx = cv2.approxPolyDP(hull, 0.015 * peri, True)
            if len(approx) < 4:
                continue

        # Compute aspect ratio for classification
        _, _, bw, bh = cv2.boundingRect(cnt)
        aspect = min(bw, bh) / max(bw, bh) if max(bw, bh) > 0 else 1.0

        polygon = []
        for pt in approx:
            px, py = pt[0]
            polygon.append({"x": round(px / w, 4), "y": round(py / h, 4)})

        rooms.append({
            "label": f"Room {len(rooms) + 1}",
            "polygon": polygon,
            "type": "other",
            "_area_frac": area / (w * h),
            "_aspect": aspect,
        })

    return rooms if len(rooms) >= 2 else None


def _classify_rooms_by_geometry(rooms: list[dict]) -> None:
    """Assign room types based on area and aspect ratio heuristics.

    Mutates rooms in place. This is approximate — the Claude Vision
    path provides more reliable classification.
    """
    if not rooms:
        return

    total_area = sum(r["_area_frac"] for r in rooms)

    # First pass: assign types
    for room in rooms:
        frac = room.pop("_area_frac")
        aspect = room.pop("_aspect")
        share = frac / total_area if total_area > 0 else 0

        if aspect < 0.3:
            room["type"] = "hallway"
        elif share > 0.35:
            room["type"] = "living"
        elif share > 0.15:
            room["type"] = "bedroom"
        elif share > 0.08:
            room["type"] = "kitchen"
        elif share < 0.04:
            room["type"] = "bathroom"
        else:
            room["type"] = "other"

    # Second pass: assign numbered labels per type
    _TYPE_LABELS = {
        "hallway": "Hallway",
        "living": "Living Room",
        "bedroom": "Bedroom",
        "kitchen": "Kitchen",
        "bathroom": "Bathroom",
        "other": "Room",
    }
    type_counts: dict[str, int] = {}
    for room in rooms:
        rtype = room["type"]
        type_counts[rtype] = type_counts.get(rtype, 0) + 1

    type_idx: dict[str, int] = {}
    for room in rooms:
        rtype = room["type"]
        base = _TYPE_LABELS.get(rtype, "Room")
        if type_counts[rtype] > 1:
            type_idx[rtype] = type_idx.get(rtype, 0) + 1
            room["label"] = f"{base} {type_idx[rtype]}"
        else:
            room["label"] = base


# Door detection thresholds — moderately relaxed for imperfect arcs
_DOOR_MIN_AREA = 80
_DOOR_ASPECT_MIN = 0.55
_DOOR_BBOX_FILL_MIN = 0.45
_DOOR_BBOX_FILL_MAX = 0.88
_DOOR_QTR_AREA_RATIO_MIN = 0.50
_DOOR_QTR_AREA_RATIO_MAX = 1.20
_DOOR_SOLIDITY_MIN = 0.80
_DOOR_RADIUS_FRAC_MIN = 0.02   # fraction of max image dimension
_DOOR_RADIUS_FRAC_MAX = 0.15


def detect_doors_cv(image: np.ndarray) -> list[dict]:
    """Detect doors by finding filled quarter-circle sector shapes.

    In architectural floor plans, doors are drawn as filled pie/sector
    shapes (quarter circles) showing the door swing. This function finds
    contours matching that geometry.

    The hinge point is found as the sharpest corner of the convex hull.
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Use Otsu's threshold instead of fixed value for robustness
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    max_dim = max(w, h)
    min_radius = max_dim * _DOOR_RADIUS_FRAC_MIN
    max_radius = max_dim * _DOOR_RADIUS_FRAC_MAX

    doors = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < _DOOR_MIN_AREA:
            continue

        bx, by, bw, bh = cv2.boundingRect(cnt)
        if bw == 0 or bh == 0:
            continue

        aspect = min(bw, bh) / max(bw, bh)
        if aspect < _DOOR_ASPECT_MIN:
            continue

        radius_est = max(bw, bh)
        if radius_est < min_radius or radius_est > max_radius:
            continue

        bbox_fill = area / (bw * bh)
        if bbox_fill < _DOOR_BBOX_FILL_MIN or bbox_fill > _DOOR_BBOX_FILL_MAX:
            continue

        quarter_area = math.pi * radius_est * radius_est / 4
        area_ratio = area / quarter_area
        if area_ratio < _DOOR_QTR_AREA_RATIO_MIN or area_ratio > _DOOR_QTR_AREA_RATIO_MAX:
            continue

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        if solidity < _DOOR_SOLIDITY_MIN:
            continue

        hull_pts = hull.reshape(-1, 2).astype(np.float64)
        n = len(hull_pts)
        if n < 3:
            continue

        min_angle = 999.0
        hinge_idx = 0
        for j in range(n):
            p0 = hull_pts[(j - 1) % n]
            p1 = hull_pts[j]
            p2 = hull_pts[(j + 1) % n]
            v1 = p0 - p1
            v2 = p2 - p1
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 < 1 or norm2 < 1:
                continue
            cos_a = np.dot(v1, v2) / (norm1 * norm2)
            angle = math.degrees(math.acos(np.clip(cos_a, -1.0, 1.0)))
            if angle < min_angle:
                min_angle = angle
                hinge_idx = j

        hinge = hull_pts[hinge_idx]

        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        mass_x = M["m10"] / M["m00"]
        mass_y = M["m01"] / M["m00"]
        arc_angle = math.degrees(math.atan2(mass_y - hinge[1], mass_x - hinge[0]))

        pts = cnt.reshape(-1, 2).astype(np.float64)
        dists = np.sqrt((pts[:, 0] - hinge[0]) ** 2 + (pts[:, 1] - hinge[1]) ** 2)
        radius = float(np.max(dists))

        doors.append({
            "id": f"door_{len(doors) + 1}",
            "position": {
                "x": round(float(hinge[0]) / w, 4),
                "y": round(float(hinge[1]) / h, 4),
            },
            "width": round(radius / max_dim, 4),
            "angle": round(arc_angle % 360, 1),
        })

    return doors


def _extract_json(text: str) -> str:
    """Extract JSON from Claude response, handling markdown fences."""
    # Try to find JSON in fenced code block
    match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def detect_rooms_claude(image: np.ndarray) -> dict:
    """Use Claude Vision API to detect rooms AND doors from the floor plan image."""
    success, buf = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError("Failed to encode image as PNG")

    b64 = base64.b64encode(buf).decode("utf-8")

    client = anthropic.Anthropic()

    prompt = (
        "This is a 2D architectural floor plan image. Analyze it carefully and return ONLY a JSON object "
        "with two arrays: \"rooms\" and \"doors\".\n\n"
        "For each ROOM provide:\n"
        "- label: room name in English (e.g. Bedroom, Bathroom, Kitchen, Living Room)\n"
        "- polygon: array of {x, y} normalized coordinates (0.0 to 1.0) relative to image width/height, "
        "tracing the room boundary corners clockwise\n"
        "- type: one of bedroom/bathroom/kitchen/living/dining/hallway/closet/balcony/other\n\n"
        "For each DOOR (the quarter-circle arc symbols in the floor plan):\n"
        "- position: {x, y} normalized coordinates (0.0 to 1.0) of the door HINGE point. "
        "The hinge is where the arc's two straight radii meet - the pivot corner of the door, "
        "which sits on the wall at the edge of the door opening. Look for the point where the "
        "quarter-circle arc originates from.\n"
        "- width: the door width as a fraction of image width (the radius of the arc, typically 0.03-0.08)\n"
        "- angle: the direction from hinge toward the CENTER of the arc sweep, in degrees "
        "(0=right, 90=down, 180=left, 270=up). This is the midpoint of the arc's angular range.\n"
        "- connects: array of two room labels this door connects (e.g. [\"Bedroom\", \"Hallway\"])\n\n"
        "IMPORTANT: Look carefully at every quarter-circle arc in the image - each one is a door. "
        "The hinge point must be precisely at the corner where the arc starts, on the wall.\n\n"
        "Return only valid JSON: {\"rooms\": [...], \"doors\": [...]}\n"
        "No explanation, no markdown fences."
    )

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            timeout=60.0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": b64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
    except anthropic.APIError as e:
        raise RuntimeError(f"Claude API error: {e}") from e

    if not response.content:
        raise RuntimeError("Claude returned empty response")

    text = response.content[0].text.strip()
    text = _extract_json(text)

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        logger.error("Claude returned invalid JSON: %s", text[:500])
        raise RuntimeError(f"Claude returned invalid JSON: {e}") from e

    # Handle both formats: {rooms, doors} or just an array of rooms
    if isinstance(data, list):
        raw_rooms = data
        raw_doors = []
    else:
        raw_rooms = data.get("rooms", [])
        raw_doors = data.get("doors", [])

    # Normalize rooms
    rooms = []
    for room in raw_rooms:
        polygon = room.get("polygon", [])
        normalized_polygon = []
        for pt in polygon:
            if isinstance(pt, list):
                normalized_polygon.append({"x": round(pt[0], 4), "y": round(pt[1], 4)})
            elif isinstance(pt, dict):
                normalized_polygon.append({
                    "x": round(pt.get("x", 0), 4),
                    "y": round(pt.get("y", 0), 4),
                })
        rooms.append({
            "label": room.get("label", "Room"),
            "polygon": normalized_polygon,
            "type": room.get("type", "other"),
        })

    # Normalize doors
    doors = []
    for i, door in enumerate(raw_doors):
        pos = door.get("position", {})
        if isinstance(pos, list):
            pos = {"x": pos[0], "y": pos[1]}
        connects = door.get("connects", [])
        doors.append({
            "id": f"door_{i + 1}",
            "position": {
                "x": round(pos.get("x", 0), 4),
                "y": round(pos.get("y", 0), 4),
            },
            "width": round(door.get("width", 0.05), 4),
            "angle": door.get("angle", 0),
            "connects": connects,
        })

    return {"rooms": rooms, "doors": doors}


def detect_rooms(image: np.ndarray, force_claude: bool = False) -> dict:
    """Detect rooms and doors. Returns {"rooms": [...], "doors": [...]}.

    Uses OpenCV first, falls back to Claude Vision API.
    """
    if not force_claude:
        cv_rooms = detect_rooms_cv(image)
        if cv_rooms is not None:
            cv_doors = detect_doors_cv(image)
            return {"rooms": cv_rooms, "doors": cv_doors}

    try:
        return detect_rooms_claude(image)
    except (TypeError, RuntimeError) as e:
        if "authentication" in str(e).lower() or "api_key" in str(e).lower():
            raise RuntimeError(
                "OpenCV detected fewer than 2 rooms and Claude Vision API "
                "key is not configured. Set ANTHROPIC_API_KEY or try a "
                "cleaner floor plan image."
            ) from e
        raise
