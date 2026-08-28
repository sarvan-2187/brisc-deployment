"""Classify every slice belonging to a single case (one patient's scan),
and build ONE consolidated PDF report - the Jetson counterpart to
case_report.py.

Same aggregation logic as case_report.py (classifies every slice, picks
the single most representative "best view" - confidently a tumor beats
uncertain beats no_tumor, highest confidence wins within a tier - and
segments only that one), but uses CUDA when available instead of forcing
CPU, since a Jetson's whole point is the onboard GPU. This is the script
send_to_jetson.py (in the companion brain-tumor repo) runs over SSH.

Usage:
    python case_report_jetson.py --input incoming/ --output reports/case_report.pdf
    python case_report_jetson.py --input incoming/ --output reports/case_report.pdf \
        --classifier mobilenet_v3 --segmenter unet --threshold 0.7
    python case_report_jetson.py --input incoming/ --output reports/case_report.pdf --cpu
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

from jetson_thermal import read_tegrastats_line, read_thermal_zones, wait_for_safe_temp
from report import build_case_pdf_report

CKPT_DIR = Path("models")
CLASS_NAMES = ["glioma", "meningioma", "pituitary", "no_tumor"]
TUMOR_CLASSES = {"glioma", "meningioma", "pituitary"}
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
IMG_SIZE_CLS = 224
IMG_SIZE_SEG = 256

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


def load_cls_model(name, device):
    if name == "efficientnet_b0":
        m = timm.create_model("efficientnet_b0", pretrained=False, num_classes=len(CLASS_NAMES))
    else:
        m = mobilenet_v3_large(weights=None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, len(CLASS_NAMES))
    m.load_state_dict(torch.load(CKPT_DIR / f"{name}_best.pt", map_location=device))
    return m.to(device).eval()


def load_seg_model(name, device):
    cls = smp.Unet if name == "unet" else smp.DeepLabV3Plus
    m = cls(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
    m.load_state_dict(torch.load(CKPT_DIR / f"{name}_best.pt", map_location=device))
    return m.to(device).eval()


def classify_slice(path, model, threshold, device):
    image = Image.open(path).convert("RGB")
    x = cls_tf(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
    top_idx = int(probs.argmax())
    top_class = CLASS_NAMES[top_idx]
    top_conf = float(probs[top_idx])
    gate = top_class in TUMOR_CLASSES and top_conf >= threshold
    verdict_kind = "tumor" if gate else ("uncertain" if top_class in TUMOR_CLASSES else "no_tumor")
    return image, probs, top_class, top_conf, verdict_kind


def segment_slice(image, model, device):
    x = seg_tf(image).unsqueeze(0).to(device)
    with torch.no_grad():
        return torch.sigmoid(model(x))[0, 0].cpu().numpy() > 0.5


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Folder of PNG/JPG slices, all from one case")
    parser.add_argument("--output", required=True, help="Path to write the single consolidated PDF to")
    parser.add_argument("--classifier", default="mobilenet_v3", choices=["efficientnet_b0", "mobilenet_v3"])
    parser.add_argument("--segmenter", default="deeplabv3plus", choices=["deeplabv3plus", "unet"])
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--max-temp", type=float, default=80.0,
                         help="Pause between slices if the hottest thermal zone reaches this many C (default 80)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))

    in_dir = Path(args.input)
    images = sorted(
        list(in_dir.glob("*.png")) + list(in_dir.glob("*.jpg")) + list(in_dir.glob("*.jpeg"))
    )
    if not images:
        print(f"No .png/.jpg/.jpeg files found in {in_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    if device.type == "cpu" and not args.cpu:
        print("WARNING: CUDA not available - falling back to CPU. If this is a Jetson, check "
              "that the JetPack-matched PyTorch wheel is installed (see requirements-jetson-nano.txt).")

    print(f"Loading classifier: {args.classifier}")
    cls_model = load_cls_model(args.classifier, device)

    zones = read_thermal_zones()
    if zones:
        print("Thermal zones at start: " + ", ".join(f"{n}={t:.1f}C" for n, t in zones.items()))
    print(f"Classifying {len(images)} slice(s)...\n")

    inference_start = time.monotonic()
    slice_results = []
    best = None  # (rank_key, path, image, probs, top_class, top_conf, verdict_kind)
    for i, path in enumerate(images, start=1):
        wait_for_safe_temp(args.max_temp)
        image, probs, top_class, top_conf, verdict_kind = classify_slice(path, cls_model, args.threshold, device)
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
        seg_model = load_seg_model(args.segmenter, device)
        best_mask = segment_slice(best_image, seg_model, device)
    else:
        print(f"\nBest view: {best_path.name} ({overall_verdict}: {best_top_class}, {best_top_conf:.1%}) - "
              "no confident tumor found across any slice, segmentation skipped.")

    inference_seconds = time.monotonic() - inference_start
    end_zones = read_thermal_zones()
    tegra_line = read_tegrastats_line()

    hottest_before = max(zones.values()) if zones else None
    hottest_after = max(end_zones.values()) if end_zones else None
    notes = [f"tegrastats: {tegra_line}"] if tegra_line else None

    device_info = {
        "device": f"Jetson ({device.type.upper()})",
        "inference_seconds": inference_seconds,
        "temp_before_c": hottest_before,
        "temp_after_c": hottest_after,
        "notes": notes,
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
    if end_zones:
        print("Thermal zones at end: " + ", ".join(f"{n}={t:.1f}C" for n, t in end_zones.items()))

    if tegra_line:
        print(f"tegrastats snapshot: {tegra_line}")

    print(f"\nDone. Case report ({overall_verdict}) written to {out_path}")


if __name__ == "__main__":
    main()
