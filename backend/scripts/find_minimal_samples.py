"""Find minimal floor plans from CubiCasa5K by parsing ground-truth SVG annotations.

This avoids running the slow OpenCV parameter sweep. Each model.svg has <g class="Space ...">
elements for rooms — counting them directly gives us low-room-count candidates fast.
Then we filter for single-floor images (no F2_original) and squarish aspect ratios.
"""

import argparse
import re
import shutil
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


SPACE_RE = re.compile(r'class="Space\s+([^"]*)"')


def count_spaces(svg_path: Path) -> tuple[int, list[str]]:
    """Count Space elements in a CubiCasa model.svg and return their subclasses."""
    try:
        text = svg_path.read_text(errors="ignore")
    except Exception:
        return 0, []
    matches = SPACE_RE.findall(text)
    return len(matches), matches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path,
                        default=Path("backend/.cubicasa/CubiCasa5k/high_quality_architectural"))
    parser.add_argument("--out", type=Path, default=Path("test_samples"))
    parser.add_argument("--min-rooms", type=int, default=3)
    parser.add_argument("--max-rooms", type=int, default=6)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--prefix", default="minimal_")
    parser.add_argument("--single-floor-only", action="store_true", default=True)
    parser.add_argument("--min-dim", type=int, default=500,
                        help="minimum image width/height in pixels")
    parser.add_argument("--max-aspect", type=float, default=2.0,
                        help="reject if max(w/h, h/w) exceeds this")
    args = parser.parse_args()

    all_dirs = [d for d in args.dataset.iterdir() if d.is_dir()]
    print(f"Scanning {len(all_dirs)} sample directories...")

    candidates = []
    for d in all_dirs:
        img_path = d / "F1_original.png"
        svg_path = d / "model.svg"
        if not img_path.exists() or not svg_path.exists():
            continue
        if args.single_floor_only and (d / "F2_original.png").exists():
            continue

        n_rooms, subclasses = count_spaces(svg_path)
        if not (args.min_rooms <= n_rooms <= args.max_rooms):
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        if min(w, h) < args.min_dim:
            continue
        aspect = max(w / h, h / w)
        if aspect > args.max_aspect:
            continue

        candidates.append({
            "id": d.name,
            "path": img_path,
            "n_rooms": n_rooms,
            "subclasses": subclasses,
            "w": w, "h": h,
            "aspect": aspect,
        })

    # Rank: prefer exactly 4 rooms, squarish aspect, larger images
    def rank_key(c: dict) -> tuple:
        room_pref = abs(c["n_rooms"] - 4)  # prefer 4 rooms (ideal demo)
        return (room_pref, c["aspect"], -min(c["w"], c["h"]))

    candidates.sort(key=rank_key)

    print(f"\nFound {len(candidates)} candidates in room range [{args.min_rooms}, {args.max_rooms}]")
    print(f"Top {args.top}:")
    print(f"{'rank':>4} {'id':>8} {'rooms':>5} {'w':>5} {'h':>5} {'aspect':>6}  subclasses")

    args.out.mkdir(parents=True, exist_ok=True)
    top = candidates[: args.top]
    for rank, c in enumerate(top, 1):
        dst = args.out / f"{args.prefix}{c['id']}.png"
        shutil.copy2(c["path"], dst)
        subs = ", ".join(s.split()[-1] for s in c["subclasses"])
        print(f"{rank:>4} {c['id']:>8} {c['n_rooms']:>5} {c['w']:>5} {c['h']:>5} "
              f"{c['aspect']:>6.2f}  {subs[:60]}")


if __name__ == "__main__":
    main()
