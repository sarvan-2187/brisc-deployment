"""Paired accuracy/segmentation-quality comparison between the two model
tiers, on the identical held-out image sample from eval_dataset_sample.py -
the defensible, apples-to-apples number for the paper's Results section,
replacing any earlier accuracy comparison that used mismatched or repeating
image sets.

  high_accuracy: EfficientNetB0 + DeepLabV3+
  lightweight:   MobileNetV3 + UNet

Every sampled image is run through BOTH tiers, in the same order, with no
repeats - unlike benchmark_adaptive.py's stress test, this is purely about
accuracy on a fixed set, not timing or thermal behavior.

Classification metrics (accuracy, precision/recall/F1 per-class + weighted +
macro average, full confusion matrix) are computed over ALL sampled images.
Segmentation metrics (mIoU, mean Dice) are computed only over images with a
ground-truth mask (the 3 tumor-bearing classes - no_tumor has none), and are
computed unconditionally for every such image regardless of what either
tier's classifier predicted - this isolates segmenter quality from
classifier confidence gating, rather than conflating the two.

Ground-truth masks are resized to 256x256 with nearest-neighbor and
thresholded at >127, exactly matching train_colab.ipynb's preprocessing, so
predicted and ground-truth masks are compared on identical terms.

Reuses benchmark.py's model loading (load_cls_model/load_seg_model,
CLASS_NAMES), case_report.py's classify_slice/segment_slice, and
adaptive_controller.py's HIGH_ACCURACY/LIGHTWEIGHT tier definitions - no
inference logic is reimplemented here. Predicted masks are not saved to
disk (metrics only) to avoid the disk cost of ~2 x (number of tumor images)
256x256 mask files; ask if you want them saved too.

Usage:
    python eval_dataset_sample.py --sample-size 20
    python eval_paired_accuracy.py --manifest eval_sample_manifest.csv --out-dir eval_results_smoke
        # smoke test - confirm the output format before the full run

    python eval_dataset_sample.py --sample-size 500
    python eval_paired_accuracy.py
        # full run
"""

import argparse
import csv
import json
from pathlib import Path

from adaptive_controller import HIGH_ACCURACY, LIGHTWEIGHT
from benchmark import CLASS_NAMES, load_cls_model, load_seg_model
from case_report import classify_slice, segment_slice
from PIL import Image
import numpy as np

TIERS = {"high_accuracy": HIGH_ACCURACY, "lightweight": LIGHTWEIGHT}


def load_manifest(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["has_mask"] = r["has_mask"] == "True"
    return rows


def load_gt_mask(mask_path, size=256):
    m = Image.open(mask_path).convert("L").resize((size, size), Image.NEAREST)
    return np.array(m) > 127


def iou_dice(pred_mask, gt_mask):
    pred, gt = pred_mask.astype(bool), gt_mask.astype(bool)
    intersection = int(np.logical_and(pred, gt).sum())
    union = int(np.logical_or(pred, gt).sum())
    iou = 1.0 if union == 0 else intersection / union
    denom = int(pred.sum()) + int(gt.sum())
    dice = 1.0 if denom == 0 else 2 * intersection / denom
    return iou, dice


def confusion_matrix(y_true, y_pred, classes):
    cm = {t: {p: 0 for p in classes} for t in classes}
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm


def classification_metrics(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred, classes)
    per_class = {}
    for c in classes:
        tp = cm[c][c]
        fp = sum(cm[t][c] for t in classes if t != c)
        fn = sum(cm[c][p] for p in classes if p != c)
        support = sum(cm[c].values())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[c] = {"precision": precision, "recall": recall, "f1": f1, "support": support}

    total = len(y_true)
    weighted = {
        m: sum(per_class[c][m] * per_class[c]["support"] for c in classes) / total if total else 0.0
        for m in ("precision", "recall", "f1")
    }
    macro = {m: sum(per_class[c][m] for c in classes) / len(classes) for m in ("precision", "recall", "f1")}
    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / total if total else 0.0
    return {
        "accuracy": accuracy, "per_class": per_class, "weighted": weighted, "macro": macro,
        "confusion_matrix": cm,
    }


def evaluate_tier(pair, manifest_rows, cls_models, seg_models, threshold):
    y_true, y_pred = [], []
    per_image = []
    ious, dices = [], []
    cls_model = cls_models[pair["classifier"]]
    seg_model = seg_models[pair["segmenter"]]

    for row in manifest_rows:
        image, _probs, top_class, _top_conf, _verdict = classify_slice(Path(row["image_path"]), cls_model, threshold)
        y_true.append(row["true_class"])
        y_pred.append(top_class)
        entry = {"predicted_class": top_class, "correct": top_class == row["true_class"], "iou": None, "dice": None}

        if row["has_mask"]:
            pred_mask = segment_slice(image, seg_model)
            gt_mask = load_gt_mask(Path(row["mask_path"]))
            iou, dice = iou_dice(pred_mask, gt_mask)
            entry["iou"], entry["dice"] = iou, dice
            ious.append(iou)
            dices.append(dice)
        per_image.append(entry)

    cls_metrics = classification_metrics(y_true, y_pred, CLASS_NAMES)
    seg_metrics = {
        "n_images": len(ious),
        "miou": (sum(ious) / len(ious)) if ious else None,
        "mean_dice": (sum(dices) / len(dices)) if dices else None,
    }
    return cls_metrics, seg_metrics, per_image


def save_confusion_matrix_png(cm, classes, tier_name, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = np.array([[cm[t][p] for p in classes] for t in classes])
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix: {tier_name}")
    peak = matrix.max() if matrix.size else 0
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                     color="white" if matrix[i, j] > peak / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_per_image_csv(path, manifest_rows, results):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "true_class", "has_mask",
                          "high_accuracy_predicted_class", "high_accuracy_correct",
                          "high_accuracy_iou", "high_accuracy_dice",
                          "lightweight_predicted_class", "lightweight_correct",
                          "lightweight_iou", "lightweight_dice"])
        ha, lw = results["high_accuracy"]["per_image"], results["lightweight"]["per_image"]
        for row, a, b in zip(manifest_rows, ha, lw):
            writer.writerow([
                row["filename"], row["true_class"], row["has_mask"],
                a["predicted_class"], a["correct"], a["iou"], a["dice"],
                b["predicted_class"], b["correct"], b["iou"], b["dice"],
            ])


def write_summary_csv(path, summary):
    rows = [["metric", "high_accuracy", "lightweight"]]
    ha, lw = summary["tiers"]["high_accuracy"], summary["tiers"]["lightweight"]
    rows.append(["accuracy", f"{ha['accuracy']:.4f}", f"{lw['accuracy']:.4f}"])
    for scope in ("weighted", "macro"):
        for m in ("precision", "recall", "f1"):
            rows.append([f"{m}_{scope}", f"{ha[scope][m]:.4f}", f"{lw[scope][m]:.4f}"])
    rows.append(["miou", f"{ha['miou']:.4f}" if ha["miou"] is not None else "",
                 f"{lw['miou']:.4f}" if lw["miou"] is not None else ""])
    rows.append(["mean_dice", f"{ha['mean_dice']:.4f}" if ha["mean_dice"] is not None else "",
                 f"{lw['mean_dice']:.4f}" if lw["mean_dice"] is not None else ""])
    rows.append(["miou_n_images", ha["miou_n_images"], lw["miou_n_images"]])
    rows.append(["sample_size", summary["sample_size"], summary["sample_size"]])
    rows.append(["delta_f1_weighted_ha_minus_lw", f"{summary['deltas']['f1_weighted']:.4f}", ""])
    rows.append(["delta_f1_macro_ha_minus_lw", f"{summary['deltas']['f1_macro']:.4f}", ""])
    rows.append(["delta_miou_ha_minus_lw",
                 f"{summary['deltas']['miou']:.4f}" if summary["deltas"]["miou"] is not None else "", ""])
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(rows)


def print_report(summary):
    ha, lw = summary["tiers"]["high_accuracy"], summary["tiers"]["lightweight"]
    print(f"\n=== PAIRED ACCURACY EVALUATION ({summary['sample_size']} images) ===\n")
    print(f"{'Metric':30s}{'High-accuracy':>16s}{'Lightweight':>16s}")
    print(f"{'  (EfficientNetB0+DeepLabV3+)':30s}{'':>16s}{'(MobileNetV3+UNet)':>16s}")
    print(f"{'Accuracy':30s}{ha['accuracy']:16.4f}{lw['accuracy']:16.4f}")
    print(f"{'F1 (weighted)':30s}{ha['weighted']['f1']:16.4f}{lw['weighted']['f1']:16.4f}")
    print(f"{'F1 (macro)':30s}{ha['macro']['f1']:16.4f}{lw['macro']['f1']:16.4f}")
    print(f"{'Precision (weighted)':30s}{ha['weighted']['precision']:16.4f}{lw['weighted']['precision']:16.4f}")
    print(f"{'Recall (weighted)':30s}{ha['weighted']['recall']:16.4f}{lw['weighted']['recall']:16.4f}")
    miou_a = f"{ha['miou']:.4f}" if ha["miou"] is not None else "n/a"
    miou_b = f"{lw['miou']:.4f}" if lw["miou"] is not None else "n/a"
    dice_a = f"{ha['mean_dice']:.4f}" if ha["mean_dice"] is not None else "n/a"
    dice_b = f"{lw['mean_dice']:.4f}" if lw["mean_dice"] is not None else "n/a"
    print(f"{'mIoU (' + str(ha['miou_n_images']) + ' images)':30s}{miou_a:>16s}{miou_b:>16s}")
    print(f"{'Mean Dice':30s}{dice_a:>16s}{dice_b:>16s}")
    print(f"\nDelta-F1 (weighted, high-accuracy - lightweight): {summary['deltas']['f1_weighted']:+.4f}")
    print(f"Delta-F1 (macro, high-accuracy - lightweight): {summary['deltas']['f1_macro']:+.4f}")
    if summary["deltas"]["miou"] is not None:
        print(f"Delta-mIoU (high-accuracy - lightweight): {summary['deltas']['miou']:+.4f}")
    excl = summary["excluded_from_miou"]
    print(f"\nExcluded from mIoU/Dice: {excl['count']} images ({excl['reason']})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default="eval_sample_manifest.csv",
                         help="Manifest CSV from eval_dataset_sample.py")
    parser.add_argument("--threshold", type=float, default=0.70,
                         help="Confidence threshold passed to classify_slice (default 0.70, matches app.py); "
                              "does not affect predicted class or these metrics, only the verdict_kind gate")
    parser.add_argument("--out-dir", default="eval_results", help="Output directory for CSVs/JSON/PNGs")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = load_manifest(args.manifest)
    if not manifest_rows:
        raise SystemExit(f"{args.manifest} is empty or missing - run eval_dataset_sample.py first.")

    n_with_mask = sum(1 for r in manifest_rows if r["has_mask"])
    print(f"Loaded {len(manifest_rows)} images from {args.manifest} ({n_with_mask} with ground-truth masks)")

    print("Loading all 4 checkpoints...")
    cls_models = {name: load_cls_model(name) for name in ("efficientnet_b0", "mobilenet_v3")}
    seg_models = {name: load_seg_model(name) for name in ("deeplabv3plus", "unet")}

    results = {}
    for tier_name, pair in TIERS.items():
        print(f"Evaluating {tier_name} ({pair['classifier']}+{pair['segmenter']}) "
              f"on {len(manifest_rows)} images...")
        cls_metrics, seg_metrics, per_image = evaluate_tier(pair, manifest_rows, cls_models, seg_models,
                                                              args.threshold)
        results[tier_name] = {"cls": cls_metrics, "seg": seg_metrics, "per_image": per_image}

    tiers_summary = {}
    for tier_name, pair in TIERS.items():
        cls_metrics, seg_metrics = results[tier_name]["cls"], results[tier_name]["seg"]
        tiers_summary[tier_name] = {
            "classifier": pair["classifier"], "segmenter": pair["segmenter"],
            "accuracy": cls_metrics["accuracy"],
            "weighted": cls_metrics["weighted"], "macro": cls_metrics["macro"],
            "per_class": cls_metrics["per_class"], "confusion_matrix": cls_metrics["confusion_matrix"],
            "miou": seg_metrics["miou"], "mean_dice": seg_metrics["mean_dice"],
            "miou_n_images": seg_metrics["n_images"],
        }

    ha, lw = tiers_summary["high_accuracy"], tiers_summary["lightweight"]
    deltas = {
        "f1_weighted": ha["weighted"]["f1"] - lw["weighted"]["f1"],
        "f1_macro": ha["macro"]["f1"] - lw["macro"]["f1"],
        "miou": (ha["miou"] - lw["miou"]) if ha["miou"] is not None and lw["miou"] is not None else None,
    }
    n_excluded = sum(1 for r in manifest_rows if not r["has_mask"])
    summary = {
        "sample_size": len(manifest_rows),
        "manifest": args.manifest,
        "tiers": tiers_summary,
        "deltas": deltas,
        "excluded_from_miou": {
            "count": n_excluded,
            "reason": "no ground-truth mask available (no_tumor class has no tumor to segment)",
        },
    }

    print_report(summary)

    with open(out_dir / "paired_comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    write_summary_csv(out_dir / "paired_comparison_summary.csv", summary)
    write_per_image_csv(out_dir / "per_image_results.csv", manifest_rows, results)
    for tier_name in TIERS:
        save_confusion_matrix_png(tiers_summary[tier_name]["confusion_matrix"], CLASS_NAMES, tier_name,
                                   out_dir / f"confusion_matrix_{tier_name}.png")

    print(f"\nWrote {out_dir / 'paired_comparison_summary.json'}, "
          f"{out_dir / 'paired_comparison_summary.csv'}, {out_dir / 'per_image_results.csv'}, "
          f"and 2 confusion matrix PNGs to {out_dir}/")


if __name__ == "__main__":
    main()
