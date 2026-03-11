import cv2
import numpy as np


def simplify_polygon(polygon: list[dict], epsilon_factor: float = 0.02) -> list[dict]:
    """Simplify a polygon using cv2.approxPolyDP.

    Args:
        polygon: List of {"x": float, "y": float} normalized coords (0-1).
        epsilon_factor: Approximation accuracy as fraction of perimeter.

    Returns:
        Simplified polygon in the same format.
    """
    if len(polygon) < 3:
        return polygon

    # Scale up for better precision with integer coordinates
    scale = 10000
    pts = np.array(
        [[int(p["x"] * scale), int(p["y"] * scale)] for p in polygon],
        dtype=np.int32,
    )

    peri = cv2.arcLength(pts, True)
    epsilon = epsilon_factor * peri
    approx = cv2.approxPolyDP(pts, epsilon, True)

    result = []
    for pt in approx:
        x, y = pt[0] if len(pt.shape) > 1 else pt
        result.append({"x": round(x / scale, 4), "y": round(y / scale, 4)})

    return result


def compute_scale(image_width: int, image_height: int) -> float:
    """Compute a shared scale factor (meters per pixel).

    Maps the longest image dimension to ~15 meters.
    """
    max_dim = max(image_width, image_height)
    if max_dim == 0:
        return 1.0
    return 15.0 / max_dim


def normalize_to_meters(
    rooms: list[dict],
    image_width: int,
    image_height: int,
    scale_m_per_px: float | None = None,
) -> list[dict]:
    """Convert normalized polygon coordinates to meter-based coordinates."""
    if scale_m_per_px is None:
        scale_m_per_px = compute_scale(image_width, image_height)

    result = []
    for i, room in enumerate(rooms):
        polygon_m = []
        for pt in room["polygon"]:
            polygon_m.append({
                "x": round(pt["x"] * image_width * scale_m_per_px, 2),
                "y": round(pt["y"] * image_height * scale_m_per_px, 2),
            })

        height = 3.0
        room_type = room.get("type", "other")
        if room_type == "bathroom":
            height = 2.7
        elif room_type == "kitchen":
            height = 2.8

        result.append({
            "id": f"room_{i + 1}",
            "label": room["label"],
            "polygon": polygon_m,
            "height": height,
            "type": room_type,
        })

    return result


def normalize_doors_to_meters(
    doors: list[dict],
    image_width: int,
    image_height: int,
    scale_m_per_px: float | None = None,
) -> list[dict]:
    """Convert normalized door positions to meter-based coordinates."""
    if scale_m_per_px is None:
        scale_m_per_px = compute_scale(image_width, image_height)

    result = []
    for i, door in enumerate(doors):
        pos = door["position"]
        # Door width is normalized to image max dimension
        width_m = door.get("width", 0.05) * max(image_width, image_height) * scale_m_per_px
        # Clamp to reasonable door width (0.6m - 1.5m)
        width_m = max(0.6, min(1.5, width_m))

        result.append({
            "id": door.get("id", f"door_{i + 1}"),
            "position": {
                "x": round(pos["x"] * image_width * scale_m_per_px, 2),
                "y": round(pos["y"] * image_height * scale_m_per_px, 2),
            },
            "width": round(width_m, 2),
            "angle": door.get("angle", 0),
            "connects": door.get("connects", []),
        })

    return result
