"""Evaluate floorplan detection accuracy against ground truth.

Usage:
    python evaluate.py --test-dir tests/fixtures

Expected directory structure:
    tests/fixtures/
        plan_01.png          # floorplan image
        plan_01.json         # ground truth
        plan_02.png
        plan_02.json
        ...

Ground truth JSON format:
    {
        "rooms": [
            {
                "label": "Bedroom",
                "type": "bedroom",
                "polygon": [
                    {"x": 0.05, "y": 0.10},
                    {"x": 0.45, "y": 0.10},
                    {"x": 0.45, "y": 0.55},
                    {"x": 0.05, "y": 0.55}
                ]
            }
        ],
        "doors": [
            {
                "position": {"x": 0.25, "y": 0.10},
                "width": 0.05,
                "angle": 90
            }
        ]
    }

Coordinates are normalized (0.0 to 1.0) relative to image dimensions.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

from pipeline.preprocess import deskew
from pipeline.detect import detect_rooms
from pipeline.polygons import simplify_polygon


def polygon_to_mask(polygon: list[dict], size: int = 500) -> np.ndarray:
    """Render a normalized polygon as a binary mask."""
    mask = np.zeros((size, size), dtype=np.uint8)
    pts = np.array(
        [[int(p["x"] * size), int(p["y"] * size)] for p in polygon],
        dtype=np.int32,
    )
    if len(pts) >= 3:
        cv2.fillPoly(mask, [pts], 255)
    return mask


def compute_iou(poly_a: list[dict], poly_b: list[dict], size: int = 500) -> float:
    """Compute Intersection over Union between two polygons."""
    mask_a = polygon_to_mask(poly_a, size)
    mask_b = polygon_to_mask(poly_b, size)
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def match_rooms(
    gt_rooms: list[dict], pred_rooms: list[dict], iou_threshold: float = 0.25
) -> list[dict]:
    """Match predicted rooms to ground truth using IoU.

    Returns a list of match dicts with gt index, pred index, and IoU.
    Uses greedy matching: best IoU pair first.
    """
    if not gt_rooms or not pred_rooms:
        return []

    # Build IoU matrix
    iou_matrix = np.zeros((len(gt_rooms), len(pred_rooms)))
    for i, gt in enumerate(gt_rooms):
        for j, pred in enumerate(pred_rooms):
            iou_matrix[i, j] = compute_iou(gt["polygon"], pred["polygon"])

    matches = []
    used_gt = set()
    used_pred = set()

    while True:
        best = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
        best_iou = iou_matrix[best]
        if best_iou < iou_threshold:
            break

        i, j = int(best[0]), int(best[1])
        matches.append({"gt_idx": i, "pred_idx": j, "iou": float(best_iou)})
        used_gt.add(i)
        used_pred.add(j)

        iou_matrix[i, :] = 0
        iou_matrix[:, j] = 0

    return matches


def point_distance(a: dict, b: dict) -> float:
    """Euclidean distance between two normalized points."""
    return math.sqrt((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2)


def match_doors(
    gt_doors: list[dict], pred_doors: list[dict], dist_threshold: float = 0.08
) -> list[dict]:
    """Match predicted doors to ground truth by position proximity."""
    if not gt_doors or not pred_doors:
        return []

    matches = []
    used_pred = set()

    for i, gt in enumerate(gt_doors):
        best_j = -1
        best_dist = dist_threshold
        for j, pred in enumerate(pred_doors):
            if j in used_pred:
                continue
            d = point_distance(gt["position"], pred["position"])
            if d < best_dist:
                best_dist = d
                best_j = j

        if best_j >= 0:
            matches.append({
                "gt_idx": i,
                "pred_idx": best_j,
                "distance": best_dist,
            })
            used_pred.add(best_j)

    return matches


def evaluate_image(
    image_path: Path, gt_path: Path, force_claude: bool = False
) -> dict:
    """Run pipeline on a single image and compare with ground truth."""
    image = cv2.imread(str(image_path))
    if image is None:
        return {"error": f"Could not read image: {image_path}"}

    with open(gt_path) as f:
        gt = json.load(f)

    gt_rooms = gt.get("rooms", [])
    gt_doors = gt.get("doors", [])

    # Run pipeline
    processed = deskew(image)
    detection = detect_rooms(processed, force_claude=force_claude)
    pred_rooms = detection["rooms"]
    pred_doors = detection["doors"]

    # Simplify predicted polygons (as the main pipeline does)
    for room in pred_rooms:
        room["polygon"] = simplify_polygon(room["polygon"])

    # --- Room metrics ---
    room_matches = match_rooms(gt_rooms, pred_rooms)
    matched_gt = {m["gt_idx"] for m in room_matches}
    matched_pred = {m["pred_idx"] for m in room_matches}

    room_precision = len(room_matches) / len(pred_rooms) if pred_rooms else 0
    room_recall = len(room_matches) / len(gt_rooms) if gt_rooms else 0
    room_f1 = (
        2 * room_precision * room_recall / (room_precision + room_recall)
        if (room_precision + room_recall) > 0
        else 0
    )
    mean_iou = (
        sum(m["iou"] for m in room_matches) / len(room_matches)
        if room_matches
        else 0
    )

    # Room type accuracy (among matched rooms)
    type_correct = 0
    for m in room_matches:
        gt_type = gt_rooms[m["gt_idx"]].get("type", "other")
        pred_type = pred_rooms[m["pred_idx"]].get("type", "other")
        if gt_type == pred_type:
            type_correct += 1
    type_accuracy = type_correct / len(room_matches) if room_matches else 0

    # --- Door metrics ---
    door_matches = match_doors(gt_doors, pred_doors)

    door_precision = len(door_matches) / len(pred_doors) if pred_doors else 0
    door_recall = len(door_matches) / len(gt_doors) if gt_doors else 0
    door_f1 = (
        2 * door_precision * door_recall / (door_precision + door_recall)
        if (door_precision + door_recall) > 0
        else 0
    )
    mean_door_dist = (
        sum(m["distance"] for m in door_matches) / len(door_matches)
        if door_matches
        else 0
    )

    return {
        "image": image_path.name,
        "rooms": {
            "gt_count": len(gt_rooms),
            "pred_count": len(pred_rooms),
            "matched": len(room_matches),
            "precision": round(room_precision, 3),
            "recall": round(room_recall, 3),
            "f1": round(room_f1, 3),
            "mean_iou": round(mean_iou, 3),
            "type_accuracy": round(type_accuracy, 3),
        },
        "doors": {
            "gt_count": len(gt_doors),
            "pred_count": len(pred_doors),
            "matched": len(door_matches),
            "precision": round(door_precision, 3),
            "recall": round(door_recall, 3),
            "f1": round(door_f1, 3),
            "mean_position_error": round(mean_door_dist, 4),
        },
        "details": {
            "room_matches": room_matches,
            "missed_rooms": [
                i for i in range(len(gt_rooms)) if i not in matched_gt
            ],
            "extra_rooms": [
                i for i in range(len(pred_rooms)) if i not in matched_pred
            ],
        },
    }


def print_report(results: list[dict]) -> None:
    """Print a summary report."""
    print("=" * 60)
    print("FLOORPLAN DETECTION ACCURACY REPORT")
    print("=" * 60)

    all_room_f1 = []
    all_room_iou = []
    all_door_f1 = []

    for r in results:
        if "error" in r:
            print(f"\n{r.get('image', '?')}: ERROR - {r['error']}")
            continue

        rooms = r["rooms"]
        doors = r["doors"]
        all_room_f1.append(rooms["f1"])
        all_room_iou.append(rooms["mean_iou"])
        all_door_f1.append(doors["f1"])

        print(f"\n--- {r['image']} ---")
        print(f"  Rooms: {rooms['pred_count']} detected / {rooms['gt_count']} actual "
              f"({rooms['matched']} matched)")
        print(f"    Precision: {rooms['precision']:.1%}  "
              f"Recall: {rooms['recall']:.1%}  "
              f"F1: {rooms['f1']:.1%}")
        print(f"    Mean IoU: {rooms['mean_iou']:.1%}  "
              f"Type accuracy: {rooms['type_accuracy']:.1%}")

        print(f"  Doors: {doors['pred_count']} detected / {doors['gt_count']} actual "
              f"({doors['matched']} matched)")
        print(f"    Precision: {doors['precision']:.1%}  "
              f"Recall: {doors['recall']:.1%}  "
              f"F1: {doors['f1']:.1%}")
        print(f"    Mean position error: {doors['mean_position_error']:.4f}")

        missed = r["details"]["missed_rooms"]
        extra = r["details"]["extra_rooms"]
        if missed:
            print(f"    Missed rooms (gt indices): {missed}")
        if extra:
            print(f"    Extra rooms (pred indices): {extra}")

    if all_room_f1:
        print("\n" + "=" * 60)
        print("AGGREGATE")
        print(f"  Images evaluated: {len(all_room_f1)}")
        print(f"  Avg Room F1:  {sum(all_room_f1) / len(all_room_f1):.1%}")
        print(f"  Avg Room IoU: {sum(all_room_iou) / len(all_room_iou):.1%}")
        print(f"  Avg Door F1:  {sum(all_door_f1) / len(all_door_f1):.1%}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate floorplan detection accuracy")
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=Path("tests/fixtures"),
        help="Directory containing test images and ground truth JSONs",
    )
    parser.add_argument(
        "--force-claude",
        action="store_true",
        help="Force Claude Vision API instead of OpenCV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Save detailed results to JSON file",
    )
    args = parser.parse_args()

    if not args.test_dir.exists():
        print(f"Error: test directory not found: {args.test_dir}")
        print(f"\nCreate it and add test cases:")
        print(f"  mkdir -p {args.test_dir}")
        print(f"  # Add pairs: image.png + image.json (ground truth)")
        sys.exit(1)

    # Find image/json pairs
    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    pairs = []
    for img_path in sorted(args.test_dir.iterdir()):
        if img_path.suffix.lower() not in image_exts:
            continue
        gt_path = img_path.with_suffix(".json")
        if gt_path.exists():
            pairs.append((img_path, gt_path))
        else:
            print(f"Warning: no ground truth for {img_path.name}, skipping")

    if not pairs:
        print(f"No test cases found in {args.test_dir}")
        print("Add image + JSON pairs (e.g., plan_01.png + plan_01.json)")
        sys.exit(1)

    print(f"Found {len(pairs)} test case(s)\n")

    results = []
    for img_path, gt_path in pairs:
        print(f"Processing {img_path.name}...", flush=True)
        result = evaluate_image(img_path, gt_path, force_claude=args.force_claude)
        results.append(result)

    print_report(results)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to {args.output}")


if __name__ == "__main__":
    main()
