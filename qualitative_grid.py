"""Builds a 2x4 collage of real classify+segment predictions, two images per
class (glioma, meningioma, pituitary, no_tumor), for a qualitative Results
figure. Reuses the deployed pipeline's own classify_slice/segment_slice and
annotate_result, so what's drawn is exactly what the app would show, not a
separate reimplementation.

Usage:
    python qualitative_grid.py --classifier mobilenet_v3 --segmenter unet \
        --threshold 0.70 --out qualitative_grid.png
"""

import argparse
from pathlib import Path

from PIL import Image

from benchmark import load_cls_model, load_seg_model
from case_report import CLASS_NAMES, classify_slice, segment_slice
from report import annotate_result

CLASS_DIR = Path("data/classification_task/test")
CELL_SIZE = 396  # working resolution per cell; keeps annotate_result's font legible in print
GRID_COLS, GRID_ROWS = 4, 2
GAP = 6


def pick_two_per_class():
    paths = []
    for cls in CLASS_NAMES:
        files = sorted((CLASS_DIR / cls).glob("*.jpg"))[:2]
        if len(files) < 2:
            raise SystemExit(f"Need at least 2 images in {CLASS_DIR / cls}, found {len(files)}")
        paths.extend(files)
    return paths  # 8 paths: glioma x2, meningioma x2, pituitary x2, no_tumor x2


def build_cell(path, cls_model, seg_model, threshold):
    image, probs, top_class, top_conf, verdict_kind = classify_slice(path, cls_model, threshold)
    mask = segment_slice(image, seg_model) if verdict_kind == "tumor" else None
    label = f"{top_class.replace('_', ' ').title()} - {top_conf:.1%}"
    cell = annotate_result(image.resize((CELL_SIZE, CELL_SIZE)), mask, label)
    return cell.resize((CELL_SIZE, CELL_SIZE))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--classifier", default="mobilenet_v3", choices=["efficientnet_b0", "mobilenet_v3"])
    parser.add_argument("--segmenter", default="unet", choices=["deeplabv3plus", "unet"])
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--out", default="qualitative_grid.png")
    args = parser.parse_args()

    cls_model = load_cls_model(args.classifier)
    seg_model = load_seg_model(args.segmenter)

    paths = pick_two_per_class()
    cells = [build_cell(p, cls_model, seg_model, args.threshold) for p in paths]

    grid_w = GRID_COLS * CELL_SIZE + (GRID_COLS - 1) * GAP
    grid_h = GRID_ROWS * CELL_SIZE + (GRID_ROWS - 1) * GAP
    grid = Image.new("RGB", (grid_w, grid_h), "white")
    for i, cell in enumerate(cells):
        row, col = divmod(i, GRID_COLS)
        x = col * (CELL_SIZE + GAP)
        y = row * (CELL_SIZE + GAP)
        grid.paste(cell, (x, y))

    grid.save(args.out, dpi=(300, 300))
    print(f"Saved {len(cells)}-image collage to {args.out} ({grid_w}x{grid_h}px, 300 DPI)")


if __name__ == "__main__":
    main()
