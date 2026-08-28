"""Batch classify + segment every image in a folder, writing one PDF
report per image.

The Pi side of the wireless round trip: send_to_pi.py (on the local
computer) copies slices into --input, runs this script over SSH, then
copies --output back. This is the "Send to AI Model" step of the
pipeline - same single classifier+segmenter pipeline app.py runs per
image, just applied to a whole folder without needing the web UI.

Usage:
    python batch_infer.py --input incoming/ --output reports/
    python batch_infer.py --input incoming/ --output reports/ \
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
from report import build_pdf_report

CKPT_DIR = Path("models")
CLASS_NAMES = ["glioma", "meningioma", "pituitary", "no_tumor"]
TUMOR_CLASSES = {"glioma", "meningioma", "pituitary"}
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
IMG_SIZE_CLS = 224
IMG_SIZE_SEG = 256
DEVICE = torch.device("cpu")

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


def process_image(path, cls_model, seg_model, classifier_name, segmenter_name, threshold):
    image = Image.open(path).convert("RGB")

    x = cls_tf(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(cls_model(x), dim=1)[0].cpu().numpy()
    top_idx = int(probs.argmax())
    top_class = CLASS_NAMES[top_idx]
    top_conf = float(probs[top_idx])

    mask = None
    if top_class in TUMOR_CLASSES and top_conf >= threshold:
        verdict_kind = "tumor"
        x_seg = seg_tf(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            mask = torch.sigmoid(seg_model(x_seg))[0, 0].cpu().numpy() > 0.5
    elif top_class in TUMOR_CLASSES:
        verdict_kind = "uncertain"
    else:
        verdict_kind = "no_tumor"

    pdf_bytes = build_pdf_report(
        image, verdict_kind, top_class, top_conf, probs, CLASS_NAMES,
        mask, classifier_name, segmenter_name if mask is not None else None, threshold,
    )
    return verdict_kind, top_class, top_conf, pdf_bytes


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Folder of PNG/JPG slices to process")
    parser.add_argument("--output", required=True, help="Folder to write PDF reports into")
    parser.add_argument("--classifier", default="mobilenet_v3", choices=["efficientnet_b0", "mobilenet_v3"])
    parser.add_argument("--segmenter", default="deeplabv3plus", choices=["deeplabv3plus", "unet"])
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--max-temp", type=float, default=75.0,
                         help="Pause between images if SoC temp reaches this many C (default 75)")
    args = parser.parse_args()

    in_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(
        list(in_dir.glob("*.png")) + list(in_dir.glob("*.jpg")) + list(in_dir.glob("*.jpeg"))
    )
    if not images:
        print(f"No .png/.jpg/.jpeg files found in {in_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {args.classifier} + {args.segmenter}...")
    cls_model = load_cls_model(args.classifier)
    seg_model = load_seg_model(args.segmenter)

    start_temp = read_temp_c()
    if start_temp is not None:
        print(f"SoC temp at start: {start_temp:.1f}C")
    print(f"Processing {len(images)} image(s)...\n")

    for i, path in enumerate(images, start=1):
        wait_for_safe_temp(args.max_temp)
        t0 = time.perf_counter()
        verdict_kind, top_class, top_conf, pdf_bytes = process_image(
            path, cls_model, seg_model, args.classifier, args.segmenter, args.threshold,
        )
        elapsed = time.perf_counter() - t0

        report_path = out_dir / f"{path.stem}_report.pdf"
        report_path.write_bytes(pdf_bytes)
        print(f"[{i}/{len(images)}] {path.name}: {verdict_kind} "
              f"({top_class}, {top_conf:.1%}) -> {report_path.name}  ({elapsed:.1f}s)")

    end_temp = read_temp_c()
    if end_temp is not None:
        note = f" (started at {start_temp:.1f}C)" if start_temp is not None else ""
        print(f"\nSoC temp at end: {end_temp:.1f}C{note}")

    flags = read_throttled_flags()
    if flags:
        print("WARNING: throttling occurred during this batch - some reports above may "
              "reflect slower-than-normal inference:")
        for f in flags:
            print(f"  - {f}")

    print(f"\nDone. {len(images)} report(s) written to {out_dir}/")


if __name__ == "__main__":
    main()
