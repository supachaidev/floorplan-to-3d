"""Convert CubiCasa5K SVG annotations to ground truth JSON for evaluate.py.

Usage:
    # Convert a few samples from CubiCasa5K into tests/fixtures/
    python convert_cubicasa.py --cubicasa-dir /path/to/CubiCasa5k/data --limit 5

    # Convert specific subfolder
    python convert_cubicasa.py --cubicasa-dir /path/to/CubiCasa5k/data/1 --limit 3

Expected CubiCasa5K structure:
    data/
      1/        (high quality)
      2/        (medium quality)
      3/        (low quality)
      Each contains subfolders like:
        <id>/
          F1_original.png   (floorplan image)
          model.svg         (annotation)
"""

import argparse
import json
import random
import re
import shutil
import sys
from pathlib import Path
from xml.dom import minidom


# Map CubiCasa room types to our simplified types
ROOM_TYPE_MAP = {
    "Bedroom": "bedroom",
    "Bathroom": "bathroom",
    "Bath": "bathroom",
    "Kitchen": "kitchen",
    "LivingRoom": "living",
    "Living": "living",
    "DiningRoom": "dining",
    "Dining": "dining",
    "Hallway": "hallway",
    "Hall": "hallway",
    "Corridor": "hallway",
    "Entry": "hallway",
    "Closet": "closet",
    "Storage": "closet",
    "Balcony": "balcony",
    "Terrace": "balcony",
    "Outdoor": "balcony",
    "Garage": "other",
    "Office": "other",
    "Room": "other",
    "Undefined": "other",
}


def parse_polygon_points(points_str: str) -> list[tuple[float, float]]:
    """Parse SVG polygon points attribute into list of (x, y) tuples."""
    points = []
    # Format: "x1,y1 x2,y2 x3,y3" or "x1 y1 x2 y2 x3 y3"
    parts = points_str.strip().split()
    for part in parts:
        if "," in part:
            x, y = part.split(",")
            points.append((float(x), float(y)))
        else:
            # Space-separated pairs: accumulate
            points.append(float(part))

    # If we got flat numbers, pair them up
    if points and isinstance(points[0], float):
        paired = []
        for i in range(0, len(points) - 1, 2):
            paired.append((points[i], points[i + 1]))
        return paired

    return points


def get_svg_bounds(doc: minidom.Document) -> tuple[float, float]:
    """Get the SVG width and height from the root element."""
    svg = doc.getElementsByTagName("svg")[0]

    # Try viewBox first
    viewbox = svg.getAttribute("viewBox")
    if viewbox:
        parts = viewbox.split()
        if len(parts) == 4:
            return float(parts[2]), float(parts[3])

    # Fall back to width/height attributes
    w = svg.getAttribute("width")
    h = svg.getAttribute("height")
    if w and h:
        # Strip units like "px"
        w = float(re.sub(r"[^\d.]", "", w))
        h = float(re.sub(r"[^\d.]", "", h))
        return w, h

    return 0, 0


def parse_svg_rooms(svg_path: Path) -> tuple[list[dict], list[dict], float, float]:
    """Parse CubiCasa5K SVG file and extract rooms and doors.

    Returns: (rooms, doors, svg_width, svg_height)
    """
    doc = minidom.parse(str(svg_path))
    svg_w, svg_h = get_svg_bounds(doc)

    rooms = []
    doors = []

    for g in doc.getElementsByTagName("g"):
        cls = g.getAttribute("class") or ""
        gid = g.getAttribute("id") or ""

        # Parse rooms (Space elements)
        if cls.startswith("Space"):
            parts = cls.split(" ", 1)
            room_type_raw = parts[1].strip() if len(parts) > 1 else "Undefined"
            room_type = ROOM_TYPE_MAP.get(room_type_raw, "other")

            # Find polygon child
            polygons = g.getElementsByTagName("polygon")
            if not polygons:
                continue

            points_str = polygons[0].getAttribute("points")
            if not points_str:
                continue

            raw_points = parse_polygon_points(points_str)
            if len(raw_points) < 3:
                continue

            # Normalize to 0-1
            if svg_w > 0 and svg_h > 0:
                polygon = [
                    {"x": round(x / svg_w, 4), "y": round(y / svg_h, 4)}
                    for x, y in raw_points
                ]
            else:
                # Can't normalize without bounds, use raw
                polygon = [{"x": x, "y": y} for x, y in raw_points]

            # Use raw type as label, mapped type as type
            label = room_type_raw.replace("Room", " Room").replace("  ", " ").strip()
            rooms.append({
                "label": label,
                "type": room_type,
                "polygon": polygon,
            })

        # Parse doors
        elif "Door" in gid.split() or "Door" in cls.split():
            polygons = g.getElementsByTagName("polygon")
            if not polygons:
                continue

            points_str = polygons[0].getAttribute("points")
            if not points_str:
                continue

            raw_points = parse_polygon_points(points_str)
            if len(raw_points) < 2:
                continue

            # Use centroid as door position
            cx = sum(p[0] for p in raw_points) / len(raw_points)
            cy = sum(p[1] for p in raw_points) / len(raw_points)

            # Estimate door width from bounding box
            xs = [p[0] for p in raw_points]
            ys = [p[1] for p in raw_points]
            door_w = max(xs) - min(xs)
            door_h = max(ys) - min(ys)
            width = max(door_w, door_h)

            if svg_w > 0 and svg_h > 0:
                doors.append({
                    "position": {
                        "x": round(cx / svg_w, 4),
                        "y": round(cy / svg_h, 4),
                    },
                    "width": round(width / max(svg_w, svg_h), 4),
                    "angle": 0,
                })

    doc.unlink()
    return rooms, doors, svg_w, svg_h


def find_cubicasa_samples(
    data_dir: Path, limit: int = 5, shuffle: bool = False
) -> list[dict]:
    """Find image + SVG pairs in CubiCasa5K data directory."""
    samples = []

    # CubiCasa structure: data/{1,2,3}/<id>/F1_original.png + model.svg
    for svg_path in sorted(data_dir.rglob("model.svg")):
        sample_dir = svg_path.parent

        # Look for the floorplan image
        img_path = None
        for name in ["F1_original.png", "F1_scaled.png"]:
            candidate = sample_dir / name
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            # Try any png
            pngs = list(sample_dir.glob("*.png"))
            if pngs:
                img_path = pngs[0]

        if img_path is None:
            continue

        samples.append({
            "id": sample_dir.name,
            "image": img_path,
            "svg": svg_path,
        })

    if shuffle:
        random.shuffle(samples)

    return samples[:limit]


def convert_sample(sample: dict, output_dir: Path) -> bool:
    """Convert a single CubiCasa sample to test fixture."""
    svg_path = sample["svg"]
    img_path = sample["image"]
    sample_id = sample["id"]

    try:
        rooms, doors, svg_w, svg_h = parse_svg_rooms(svg_path)
    except Exception as e:
        print(f"  Error parsing {svg_path}: {e}")
        return False

    if not rooms:
        print(f"  No rooms found in {svg_path}, skipping")
        return False

    # Copy image
    out_img = output_dir / f"cubicasa_{sample_id}.png"
    shutil.copy2(img_path, out_img)

    # Write ground truth JSON
    gt = {"rooms": rooms, "doors": doors}
    out_json = output_dir / f"cubicasa_{sample_id}.json"
    with open(out_json, "w") as f:
        json.dump(gt, f, indent=2)

    print(f"  {out_img.name}: {len(rooms)} rooms, {len(doors)} doors")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert CubiCasa5K samples to evaluate.py ground truth format"
    )
    parser.add_argument(
        "--cubicasa-dir",
        type=Path,
        required=True,
        help="Path to CubiCasa5K data directory (e.g., CubiCasa5k/data)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tests/fixtures"),
        help="Output directory for test fixtures (default: tests/fixtures)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Max number of samples to convert (default: 5)",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Randomly shuffle samples before selecting (default: deterministic order)",
    )
    args = parser.parse_args()

    if not args.cubicasa_dir.exists():
        print(f"Error: directory not found: {args.cubicasa_dir}")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {args.cubicasa_dir} for samples...")
    samples = find_cubicasa_samples(args.cubicasa_dir, limit=args.limit, shuffle=args.shuffle)

    if not samples:
        print("No CubiCasa5K samples found. Expected structure:")
        print("  data/{1,2,3}/<id>/F1_original.png + model.svg")
        sys.exit(1)

    print(f"Found {len(samples)} sample(s), converting...\n")

    converted = 0
    for sample in samples:
        print(f"Converting {sample['id']}...")
        if convert_sample(sample, args.output_dir):
            converted += 1

    print(f"\nDone! Converted {converted}/{len(samples)} samples to {args.output_dir}")
    print(f"\nRun evaluation with:")
    print(f"  python evaluate.py --test-dir {args.output_dir}")


if __name__ == "__main__":
    main()
