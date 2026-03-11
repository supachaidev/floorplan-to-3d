import json


def build_floorplan_json(
    rooms: list[dict],
    doors: list[dict] | None = None,
    image_width_m: float = 0,
    image_height_m: float = 0,
) -> dict:
    """Build the final floorplan JSON schema."""
    result = {
        "floorplan": {
            "scale": 0.01,
            "units": "meters",
            "image_width_m": round(image_width_m, 2),
            "image_height_m": round(image_height_m, 2),
            "rooms": rooms,
            "doors": doors or [],
        }
    }
    return result


def export_json(rooms: list[dict], doors: list[dict] | None = None, path: str | None = None) -> str:
    """Export floorplan as JSON string, optionally write to file."""
    data = build_floorplan_json(rooms, doors)
    text = json.dumps(data, indent=2)
    if path:
        with open(path, "w") as f:
            f.write(text)
    return text
