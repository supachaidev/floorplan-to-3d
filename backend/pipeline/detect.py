import logging
import math

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _remove_small_components(binary: np.ndarray, max_frac: float = 0.003) -> np.ndarray:
    """Remove small connected components (text, annotations, symbols).

    Components smaller than max_frac of total image area are removed.
    This prevents text labels from creating false walls/rooms.
    """
    h, w = binary.shape[:2]
    max_area = w * h * max_frac
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    cleaned = binary.copy()
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < max_area:
            cleaned[labels == i] = 0
    return cleaned


_MAX_DETECT_DIM = 1200  # downscale large images for speed


def detect_rooms_cv(image: np.ndarray) -> list[dict] | None:
    """Detect rooms using OpenCV for clean printed plans.

    Uses flood-fill to find enclosed white regions (rooms) between walls.
    Sweeps multiple threshold parameters and picks the best result.

    Returns a list of room dicts or None if detection fails.
    """
    h, w = image.shape[:2]

    # Downscale large images — detection works on relative shapes,
    # so high resolution just wastes time.
    scale_factor = 1.0
    if max(h, w) > _MAX_DETECT_DIM:
        scale_factor = _MAX_DETECT_DIM / max(h, w)
        new_w, new_h = int(w * scale_factor), int(h * scale_factor)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        h, w = new_h, new_w

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Normalize contrast for varying image quality
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Prepare blurred variant to sweep over
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

    best_rooms = None
    best_score = -1.0

    # Parameter sweep including blur variant
    for img in [enhanced, blurred]:
        for block_size in [11, 15, 21, 31]:
            for c_val in [2, 4, 6, 8]:
                for morph_k in [3, 5]:
                    rooms = _find_enclosed_rooms(
                        img, w, h, block_size, c_val, morph_k,
                    )
                    if rooms is None:
                        continue
                    score = _score_room_set(rooms)
                    if score > best_score:
                        best_score = score
                        best_rooms = rooms

    if best_rooms is None:
        return None

    # Merge rooms that overlap significantly (IoU > 0.5)
    best_rooms = _merge_overlapping_rooms(best_rooms)

    _classify_rooms_by_geometry(best_rooms)
    return best_rooms


def _merge_overlapping_rooms(rooms: list[dict], iou_threshold: float = 0.5) -> list[dict]:
    """Merge room pairs that overlap significantly, keeping the larger one."""
    if len(rooms) <= 1:
        return rooms

    size = 300
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

    drop = set()
    for i in range(len(rooms)):
        if i in drop:
            continue
        for j in range(i + 1, len(rooms)):
            if j in drop:
                continue
            inter = np.logical_and(masks[i], masks[j]).sum()
            union = np.logical_or(masks[i], masks[j]).sum()
            if union > 0 and inter / union > iou_threshold:
                # Keep the larger room
                if rooms[i]["_area_frac"] >= rooms[j]["_area_frac"]:
                    drop.add(j)
                else:
                    drop.add(i)
                    break

    return [r for idx, r in enumerate(rooms) if idx not in drop]


def _score_room_set(rooms: list[dict]) -> float:
    """Score a set of detected rooms by quality, not just count.

    Prefers results where total room area covers 30-90% of the image
    with minimal mutual overlap and uniform room sizes (not fragmented).
    """
    if not rooms:
        return -1.0

    areas = [r["_area_frac"] for r in rooms]
    total_area = sum(areas)

    # Penalize coverage outside the sweet spot (30-90% of image)
    if total_area < 0.30:
        coverage_score = total_area / 0.30
    elif total_area > 0.90:
        coverage_score = max(0, 1.0 - (total_area - 0.90) / 0.5)
    else:
        coverage_score = 1.0

    # Reward more rooms (log scale, capped to avoid runaway)
    count_score = min(math.log2(max(len(rooms), 1)) / 5.0, 0.5)

    # Penalize fragmentation: if many rooms are tiny (< 1% of image),
    # the result is likely noise from aggressive thresholding
    tiny_count = sum(1 for a in areas if a < 0.01)
    frag_penalty = 0.4 * (tiny_count / max(len(rooms), 1))

    # Reward room regularity — rooms with aspect ratio > 0.4 are more
    # likely real rooms (not slivers of noise)
    regular_count = sum(1 for r in rooms if r.get("_aspect", 0.5) > 0.4)
    regularity_bonus = 0.15 * (regular_count / max(len(rooms), 1))

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
            overlap_penalty = 2.0 * double_count.sum() / combined.sum()

    return coverage_score + count_score + regularity_bonus - frag_penalty - overlap_penalty


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

    # Remove small connected components (text/annotations) before closing
    binary = _remove_small_components(binary)

    # Close small gaps in walls using both square and directional kernels
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_k, morph_k))
    walls = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Use directional kernels to reconnect horizontal/vertical wall segments
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_k * 2 + 1, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, morph_k * 2 + 1))
    walls_h = cv2.morphologyEx(walls, cv2.MORPH_CLOSE, h_kernel, iterations=1)
    walls_v = cv2.morphologyEx(walls, cv2.MORPH_CLOSE, v_kernel, iterations=1)
    walls = cv2.bitwise_or(walls_h, walls_v)

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
_DOOR_ASPECT_MIN = 0.55
_DOOR_BBOX_FILL_MIN = 0.45
_DOOR_BBOX_FILL_MAX = 0.88
_DOOR_QTR_AREA_RATIO_MIN = 0.50
_DOOR_QTR_AREA_RATIO_MAX = 1.20
_DOOR_SOLIDITY_MIN = 0.80
_DOOR_RADIUS_FRAC_MIN = 0.02   # fraction of max image dimension
_DOOR_RADIUS_FRAC_MAX = 0.15
# Minimum distance (normalized) between distinct doors
_DOOR_DEDUP_DIST = 0.025


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
    min_area = 80

    candidates = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
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

        candidates.append({
            "position": {
                "x": round(float(hinge[0]) / w, 4),
                "y": round(float(hinge[1]) / h, 4),
            },
            "width": round(radius / max_dim, 4),
            "angle": round(arc_angle % 360, 1),
            "_area": area,
        })

    # Deduplicate nearby doors — keep the one with larger area
    candidates.sort(key=lambda d: d["_area"], reverse=True)
    doors = []
    for cand in candidates:
        too_close = False
        for existing in doors:
            dx = cand["position"]["x"] - existing["position"]["x"]
            dy = cand["position"]["y"] - existing["position"]["y"]
            if math.sqrt(dx * dx + dy * dy) < _DOOR_DEDUP_DIST:
                too_close = True
                break
        if not too_close:
            doors.append({
                "id": f"door_{len(doors) + 1}",
                "position": cand["position"],
                "width": cand["width"],
                "angle": cand["angle"],
            })

    return doors


def detect_rooms(image: np.ndarray) -> dict:
    """Detect rooms and doors using OpenCV.

    Returns {"rooms": [...], "doors": [...]}.
    Returns empty rooms list if detection fails.
    """
    cv_rooms = detect_rooms_cv(image)
    cv_doors = detect_doors_cv(image)
    return {"rooms": cv_rooms or [], "doors": cv_doors}
