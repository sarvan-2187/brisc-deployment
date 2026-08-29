"""Builds a reproducible, class-proportional random sample of the BRISC2025
test split for the paired accuracy evaluation in eval_paired_accuracy.py.

Sampling is proportional to the real test-split class distribution (glioma
254, meningioma 306, pituitary 300, no_tumor 140 out of 1000 - see the
project's EDA_Findings_Report.pdf), not equal-count-per-class, using the
largest-remainder method so the per-class counts always sum to exactly
--sample-size. A fixed --seed (default 42) makes the sample reproducible
run to run.

Ground-truth segmentation masks only exist for the 3 tumor-bearing classes
(data/segmentation_task/test/masks/); no_tumor has none. The manifest
records has_mask per row so eval_paired_accuracy.py knows which images to
include in mIoU/Dice.

Usage:
    python eval_dataset_sample.py                        # 500-image sample, seed 42
    python eval_dataset_sample.py --sample-size 20        # smoke-test sample
    python eval_dataset_sample.py --sample-size 500 --seed 7 --out my_manifest.csv
"""

import argparse
import csv
import random
from pathlib import Path

CLASS_NAMES = ["glioma", "meningioma", "pituitary", "no_tumor"]
TUMOR_CLASSES = {"glioma", "meningioma", "pituitary"}


def proportional_counts(class_totals, sample_size):
    """Largest-remainder apportionment: floor each class's proportional share,
    then hand out the leftover seats to the classes with the biggest
    fractional remainder, so the counts always sum to exactly sample_size."""
    total = sum(class_totals.values())
    raw = {c: sample_size * n / total for c, n in class_totals.items()}
    counts = {c: int(v) for c, v in raw.items()}
    remainder = sample_size - sum(counts.values())
    for c in sorted(class_totals, key=lambda c: raw[c] - counts[c], reverse=True)[:remainder]:
        counts[c] += 1
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample-size", type=int, default=500,
                         help="Total images to sample across all 4 classes (default 500)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default 42)")
    parser.add_argument("--classification-dir", default="data/classification_task/test",
                         help="Folder of <class>/*.jpg (default data/classification_task/test)")
    parser.add_argument("--masks-dir", default="data/segmentation_task/test/masks",
                         help="Folder of ground-truth <stem>.png masks (default data/segmentation_task/test/masks)")
    parser.add_argument("--out", default="eval_sample_manifest.csv", help="Output manifest CSV path")
    args = parser.parse_args()

    cls_dir = Path(args.classification_dir)
    masks_dir = Path(args.masks_dir)

    class_files = {c: sorted((cls_dir / c).glob("*.jpg")) for c in CLASS_NAMES}
    class_totals = {c: len(files) for c, files in class_files.items()}
    if any(n == 0 for n in class_totals.values()):
        raise SystemExit(f"One or more classes have 0 images under {cls_dir} - check the path. Counts: {class_totals}")

    counts = proportional_counts(class_totals, args.sample_size)
    print(f"Test-split totals: {class_totals} (total {sum(class_totals.values())})")
    print(f"Sampling {args.sample_size} proportionally (seed {args.seed}): {counts}")

    rng = random.Random(args.seed)
    rows = []
    for c in CLASS_NAMES:
        chosen = rng.sample(class_files[c], counts[c])
        for path in chosen:
            mask_path = masks_dir / f"{path.stem}.png"
            has_mask = c in TUMOR_CLASSES and mask_path.exists()
            rows.append({
                "filename": path.name,
                "true_class": c,
                "has_mask": has_mask,
                "image_path": str(path),
                "mask_path": str(mask_path) if has_mask else "",
            })

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "true_class", "has_mask", "image_path", "mask_path"])
        writer.writeheader()
        writer.writerows(rows)

    n_with_mask = sum(1 for r in rows if r["has_mask"])
    print(f"Wrote {len(rows)} rows to {args.out} ({n_with_mask} with ground-truth masks, "
          f"{len(rows) - n_with_mask} without - no_tumor images have no mask by design).")


if __name__ == "__main__":
    main()
