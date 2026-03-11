import json


def build_floorplan_json(rooms: list[dict], doors: list[dict] | None = None) -> dict:
    """Build the final floorplan JSON schema."""
    result = {
        "floorplan": {
            "scale": 0.01,
            "units": "meters",
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
