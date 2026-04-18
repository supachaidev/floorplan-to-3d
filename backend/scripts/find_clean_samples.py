"""Find the cleanest floorplan samples from CubiCasa5K high_quality_architectural.

Scores samples using the existing profiler (contrast + edge regularity + low skew),
filters for clean_digital category, and copies the top-N to test_samples/.
"""

import argparse
import random
import shutil
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.profiler import profile_image


def score(profile: dict) -> float:
    """Higher = cleaner. Favor high contrast, straight lines, regular edges, low skew."""
    return (
        profile["contrast_score"] * 1.0
        + profile["line_straightness"] * 100.0
        + profile["edge_regularity"] * 500.0
        - profile["skew_angle_deg"] * 5.0
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path,
                        default=Path("backend/.cubicasa/CubiCasa5k/high_quality_architectural"))
    parser.add_argument("--out", type=Path, default=Path("test_samples"))
    parser.add_argument("--scan", type=int, default=400, help="number of random samples to profile")
    parser.add_argument("--top", type=int, default=12, help="how many to copy")
    parser.add_argument("--min-size-kb", type=int, default=100, help="skip tiny images")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    all_dirs = [d for d in args.dataset.iterdir() if d.is_dir()]
    random.shuffle(all_dirs)
    candidates = all_dirs[: args.scan]

    print(f"Profiling {len(candidates)} candidates from {args.dataset}...")

    scored = []
    for i, d in enumerate(candidates, 1):
        img_path = d / "F1_original.png"
        if not img_path.exists():
            continue
        if img_path.stat().st_size < args.min_size_kb * 1024:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        profile = profile_image(img)
        if profile["category"] != "clean_digital":
            continue
        scored.append((score(profile), d.name, img_path, profile))
        if i % 50 == 0:
            print(f"  ...{i}/{len(candidates)} ({len(scored)} clean)")

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[: args.top]

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"\nTop {len(top)} cleanest samples → {args.out}/")
    print(f"{'rank':>4} {'id':>8} {'score':>8}  {'contrast':>9} {'lines':>6} {'edges':>6} {'skew':>5}")
    for rank, (sc, sample_id, src, prof) in enumerate(top, 1):
        dst = args.out / f"clean_{sample_id}.png"
        shutil.copy2(src, dst)
        print(f"{rank:>4} {sample_id:>8} {sc:>8.1f}  "
              f"{prof['contrast_score']:>9.1f} {prof['line_straightness']:>6.3f} "
              f"{prof['edge_regularity']:>6.3f} {prof['skew_angle_deg']:>5.1f}")


if __name__ == "__main__":
    main()
