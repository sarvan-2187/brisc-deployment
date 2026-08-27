"""Classify every slice belonging to a single case (one patient's scan),
and build ONE consolidated PDF report - not one report per slice.

Since all slices in --input come from the same scan, this aggregates
across them: classifies every slice, then picks the single most
representative "best view" - the slice with the strongest evidence of a
tumor if any slice is confidently a tumor, otherwise the most confident
slice overall - and only segments that one. Priority order per slice:
confidently a tumor (class + confidence >= threshold) beats uncertain
(tumor class but below threshold) beats no_tumor; within the same tier,
higher confidence wins ("best view" = clearest evidence).

Usage:
    python case_report.py --input incoming/ --output reports/case_report.pdf
    python case_report.py --input incoming/ --output reports/case_report.pdf \
        --classifier mobilenet_v3 --segmenter unet --threshold 0.7
"""

import argparse
import sys
import time
from pathlib import Path

import segmentation_models_pytorch as smp
import timm
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v3_large

from pi_thermal import read_temp_c, read_throttled_flags, wait_for_safe_temp
from report import build_case_pdf_report

CKPT_DIR = Path("models")
CLASS_NAMES = ["glioma", "meningioma", "pituitary", "no_tumor"]
TUMOR_CLASSES = {"glioma", "meningioma", "pituitary"}
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
IMG_SIZE_CLS = 224
IMG_SIZE_SEG = 256
DEVICE = torch.device("cpu")

TIER = {"tumor": 2, "uncertain": 1, "no_tumor": 0}

cls_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE_CLS, IMG_SIZE_CLS)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
seg_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE_SEG, IMG_SIZE_SEG)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def load_cls_model(name):
    if name == "efficientnet_b0":
        m = timm.create_model("efficientnet_b0", pretrained=False, num_classes=len(CLASS_NAMES))
    else:
        m = mobilenet_v3_large(weights=None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, len(CLASS_NAMES))
    m.load_state_dict(torch.load(CKPT_DIR / f"{name}_best.pt", map_location=DEVICE, weights_only=False))
    return m.to(DEVICE).eval()


def load_seg_model(name):
    cls = smp.Unet if name == "unet" else smp.DeepLabV3Plus
    m = cls(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
    m.load_state_dict(torch.load(CKPT_DIR / f"{name}_best.pt", map_location=DEVICE, weights_only=False))
    return m.to(DEVICE).eval()


def classify_slice(path, model, threshold):
    image = Image.open(path).convert("RGB")
    x = cls_tf(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
    top_idx = int(probs.argmax())
    top_class = CLASS_NAMES[top_idx]
    top_conf = float(probs[top_idx])
    gate = top_class in TUMOR_CLASSES and top_conf >= threshold
    verdict_kind = "tumor" if gate else ("uncertain" if top_class in TUMOR_CLASSES else "no_tumor")
    return image, probs, top_class, top_conf, verdict_kind


def segment_slice(image, model):
    x = seg_tf(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        return torch.sigmoid(model(x))[0, 0].cpu().numpy() > 0.5


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Folder of PNG/JPG slices, all from one case")
    parser.add_argument("--output", required=True, help="Path to write the single consolidated PDF to")
    parser.add_argument("--classifier", default="efficientnet_b0", choices=["efficientnet_b0", "mobilenet_v3"])
    parser.add_argument("--segmenter", default="deeplabv3plus", choices=["deeplabv3plus", "unet"])
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--max-temp", type=float, default=75.0,
                         help="Pause between slices if SoC temp reaches this many C (default 75)")
    args = parser.parse_args()

    in_dir = Path(args.input)
    images = sorted(
        list(in_dir.glob("*.png")) + list(in_dir.glob("*.jpg")) + list(in_dir.glob("*.jpeg"))
    )
    if not images:
        print(f"No .png/.jpg/.jpeg files found in {in_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading classifier: {args.classifier}")
    cls_model = load_cls_model(args.classifier)

    start_temp = read_temp_c()
    if start_temp is not None:
        print(f"SoC temp at start: {start_temp:.1f}C")
    print(f"Classifying {len(images)} slice(s)...\n")

    inference_start = time.monotonic()
    slice_results = []
    best = None  # (rank_key, path, image, probs, top_class, top_conf, verdict_kind)
    for i, path in enumerate(images, start=1):
        wait_for_safe_temp(args.max_temp)
        image, probs, top_class, top_conf, verdict_kind = classify_slice(path, cls_model, args.threshold)
        slice_results.append({
            "filename": path.name, "top_class": top_class,
            "top_conf": top_conf, "verdict_kind": verdict_kind,
        })
        print(f"[{i}/{len(images)}] {path.name}: {verdict_kind} ({top_class}, {top_conf:.1%})")

        rank_key = (TIER[verdict_kind], top_conf)
        if best is None or rank_key > best[0]:
            best = (rank_key, path, image, probs, top_class, top_conf, verdict_kind)

    _, best_path, best_image, best_probs, best_top_class, best_top_conf, overall_verdict = best

    best_mask = None
    if overall_verdict == "tumor":
        print(f"\nBest view: {best_path.name} ({best_top_class}, {best_top_conf:.1%}) - segmenting...")
        seg_model = load_seg_model(args.segmenter)
        best_mask = segment_slice(best_image, seg_model)
    else:
        print(f"\nBest view: {best_path.name} ({overall_verdict}: {best_top_class}, {best_top_conf:.1%}) - "
              "no confident tumor found across any slice, segmentation skipped.")

    inference_seconds = time.monotonic() - inference_start
    end_temp = read_temp_c()
    flags = read_throttled_flags()

    device_info = {
        "device": "Raspberry Pi (CPU)",
        "inference_seconds": inference_seconds,
        "temp_before_c": start_temp,
        "temp_after_c": end_temp,
        "notes": flags or None,
    }

    pdf_bytes = build_case_pdf_report(
        slice_results, best_image, best_mask, overall_verdict,
        best_top_class, best_top_conf, best_probs, CLASS_NAMES,
        args.classifier, args.segmenter if best_mask is not None else None, args.threshold,
        device_info=device_info,
    )
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(pdf_bytes)

    print(f"\nInference time: {inference_seconds:.1f}s")
    if end_temp is not None:
        note = f" (started at {start_temp:.1f}C)" if start_temp is not None else ""
        print(f"SoC temp at end: {end_temp:.1f}C{note}")

    if flags:
        print("WARNING: throttling occurred during this run - the result above may reflect "
              "slower-than-normal inference:")
        for f in flags:
            print(f"  - {f}")

    print(f"\nDone. Case report ({overall_verdict}) written to {out_path}")


if __name__ == "__main__":
    main()
