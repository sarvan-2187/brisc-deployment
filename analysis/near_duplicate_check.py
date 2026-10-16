"""Train/test near-duplicate audit for BRISC 2025 classification images (ISSUES.md H-02).

For every test image, finds the training image with the closest 256-bit
perceptual hash, then confirms with a mean absolute pixel difference at
256x256 greyscale and an exact md5 match. Writes one CSV row per test image
whose nearest train neighbour is within --max-hamming bits.

    pip install imagehash pillow numpy
    python near_duplicate_check.py --data ../../data --out results/near_duplicate_pairs.csv

A pair with mean_abs_pixel_diff_256px < 5 (on a 0-255 scale) is treated as a
duplicate in the paper; inspect a sample by eye before relying on it.
"""

import argparse
import collections
import csv
import hashlib
from pathlib import Path

import imagehash
import numpy as np
from PIL import Image


def images(root, split):
    return sorted((root / "classification_task" / split).glob("*/*.jpg"))


def phash_bits(path):
    return imagehash.phash(Image.open(path).convert("L"), hash_size=16).hash.flatten()


def grey256(path):
    return np.asarray(Image.open(path).convert("L").resize((256, 256)), dtype=np.float32)


def md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-hamming", type=int, default=16)
    ap.add_argument("--dup-mad", type=float, default=5.0)
    args = ap.parse_args()

    train, test = images(args.data, "train"), images(args.data, "test")
    train_md5 = {md5(p) for p in train}
    train_h = np.array([phash_bits(p) for p in train])
    test_h = np.array([phash_bits(p) for p in test])
    dist = (test_h[:, None, :] != train_h[None, :, :]).sum(-1)
    nearest, nearest_d = dist.argmin(1), dist.min(1)

    seg_test = {p.name for p in (args.data / "segmentation_task" / "test" / "images").glob("*.jpg")}
    rows = []
    for i in np.where(nearest_d <= args.max_hamming)[0]:
        t, tr = test[i], train[nearest[i]]
        mad = float(np.abs(grey256(t) - grey256(tr)).mean())
        rows.append({
            "test_file": t.name, "nearest_train_file": tr.name, "phash_hamming_256": int(nearest_d[i]),
            "mean_abs_pixel_diff_256px": round(mad, 2), "class": t.parent.name,
            "byte_identical": md5(t) in train_md5, "in_seg_test": t.name in seg_test,
        })
    rows.sort(key=lambda r: r["mean_abs_pixel_diff_256px"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    dups = [r for r in rows if r["mean_abs_pixel_diff_256px"] < args.dup_mad]
    print(f"train={len(train)} test={len(test)} candidate pairs={len(rows)}")
    print(f"duplicates (MAD < {args.dup_mad}): {len(dups)} test images ({len(dups) / len(test):.1%})")
    print(f"  byte-identical: {sum(r['byte_identical'] for r in dups)}")
    print(f"  by class: {dict(collections.Counter(r['class'] for r in dups))}")
    print(f"  in segmentation test set: {sum(r['in_seg_test'] for r in dups)}")
    print(f"  label mismatches: {sum(1 for r in dups if not r['nearest_train_file'].split('_')[3] == r['test_file'].split('_')[3])}")


if __name__ == "__main__":
    main()
