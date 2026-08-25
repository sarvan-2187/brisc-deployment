"""Measure real inference latency for all 4 classifier x segmenter
combinations on this device (both classifiers, both segmenters, and each
of the 4 combined pipelines).

No Streamlit dependency, so it can run standalone on a Raspberry Pi.
Requires all 4 checkpoints pulled via `git lfs pull` (models/*.pt).

Thermal safety: a Pi 4 without a heatsink can hit its throttle point within
minutes of sustained multi-core load, which is exactly what a long benchmark
run is. This script checks the SoC temperature before every timed iteration
and pauses if it's running hot, and reports whether throttling actually
occurred so the numbers you get can be trusted.

Usage:
    python benchmark.py
    python benchmark.py --runs 20 --cooldown 1.0 --max-temp 70
    python benchmark.py --image demo_images/glioma/some_image.jpg
"""

import argparse
import statistics
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

try:
    import resource
except ImportError:
    resource = None  # not available on Windows; benchmark is meant for Linux/Pi

CKPT_DIR = Path("models")
CLASS_NAMES = ["glioma", "meningioma", "pituitary", "no_tumor"]
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
IMG_SIZE_CLS = 224
IMG_SIZE_SEG = 256
DEVICE = torch.device("cpu")

CLASSIFIER_CHOICES = ["efficientnet_b0", "mobilenet_v3"]
SEGMENTER_CHOICES = ["deeplabv3plus", "unet"]
DISPLAY_NAMES = {
    "efficientnet_b0": "EfficientNetB0",
    "mobilenet_v3": "MobileNetV3",
    "unet": "UNet",
    "deeplabv3plus": "DeepLabV3+",
}

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


def time_calls(fn, warmup, runs, cooldown, max_temp):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        wait_for_safe_temp(max_temp)
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
        if cooldown > 0:
            time.sleep(cooldown)
    return times


def summarize(label, times):
    ms = sorted(t * 1000 for t in times)
    mean = statistics.mean(ms)
    median = statistics.median(ms)
    p95 = ms[min(len(ms) - 1, int(len(ms) * 0.95))]
    print(f"{label:38s} mean {mean:7.1f} ms  median {median:7.1f} ms  "
          f"p95 {p95:7.1f} ms  min {ms[0]:7.1f} ms  max {ms[-1]:7.1f} ms")


def peak_memory_mb():
    if resource is None:
        return None
    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return kb / 1024  # ru_maxrss is KB on Linux


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", default=None, help="Test image path; defaults to the first demo image found")
    parser.add_argument("--warmup", type=int, default=2, help="Untimed warmup runs per stage (default 2)")
    parser.add_argument("--runs", type=int, default=10, help="Timed runs to average per stage (default 10)")
    parser.add_argument("--cooldown", type=float, default=0.5,
                         help="Seconds to sleep between each timed run (default 0.5, 0 to disable)")
    parser.add_argument("--max-temp", type=float, default=75.0,
                         help="Pause and wait if SoC temp reaches this many C (default 75)")
    args = parser.parse_args()

    image_path = Path(args.image) if args.image else next(Path("demo_images").rglob("*.jpg"))
    image = Image.open(image_path).convert("RGB")
    cls_input = cls_tf(image).unsqueeze(0)
    seg_input = seg_tf(image).unsqueeze(0)

    start_temp = read_temp_c()
    print(f"Threads available to torch: {torch.get_num_threads()}")
    print(f"Test image: {image_path}")
    print(f"Warmup: {args.warmup}  Runs: {args.runs}  Cooldown: {args.cooldown}s  Max temp: {args.max_temp}C")
    if start_temp is not None:
        print(f"SoC temp at start: {start_temp:.1f}C")
    else:
        print("SoC temp: unavailable on this device - thermal pausing is disabled.")
    print()

    print("Loading all 4 checkpoints...")
    cls_models = {}
    for name in CLASSIFIER_CHOICES:
        print(f"  classifier: {name}")
        cls_models[name] = load_cls_model(name)
    seg_models = {}
    for name in SEGMENTER_CHOICES:
        print(f"  segmenter:  {name}")
        seg_models[name] = load_seg_model(name)
    print()

    def run_cls(name):
        with torch.no_grad():
            cls_models[name](cls_input)

    def run_seg(name):
        with torch.no_grad():
            seg_models[name](seg_input)

    print("--- Individual models ---")
    for name in CLASSIFIER_CHOICES:
        summarize(f"Classifier: {DISPLAY_NAMES[name]}",
                  time_calls(lambda n=name: run_cls(n), args.warmup, args.runs, args.cooldown, args.max_temp))
    for name in SEGMENTER_CHOICES:
        summarize(f"Segmenter: {DISPLAY_NAMES[name]}",
                  time_calls(lambda n=name: run_seg(n), args.warmup, args.runs, args.cooldown, args.max_temp))

    print("\n--- All 4 classifier + segmenter combinations (full pipeline) ---")
    for cls_name in CLASSIFIER_CHOICES:
        for seg_name in SEGMENTER_CHOICES:
            def run_pipeline(c=cls_name, s=seg_name):
                run_cls(c)
                run_seg(s)

            label = f"{DISPLAY_NAMES[cls_name]} + {DISPLAY_NAMES[seg_name]}"
            summarize(label, time_calls(run_pipeline, args.warmup, args.runs, args.cooldown, args.max_temp))

    end_temp = read_temp_c()
    if end_temp is not None:
        note = f" (started at {start_temp:.1f}C)" if start_temp is not None else ""
        print(f"\nSoC temp at end: {end_temp:.1f}C{note}")

    flags = read_throttled_flags()
    if flags is None:
        print("(vcgencmd not available - can't confirm whether throttling occurred.)")
    elif flags:
        print("WARNING: throttling flags active during this run - numbers above may be "
              "slower than the hardware's real capability:")
        for f in flags:
            print(f"  - {f}")
    else:
        print("No throttling occurred - these numbers reflect full, unthrottled performance.")

    peak = peak_memory_mb()
    if peak is not None:
        print(f"Peak RSS this process: {peak:.0f} MB (all 4 models loaded simultaneously)")
    else:
        print("(Peak RSS reporting needs the `resource` module - Linux/Pi only, not available here.)")


if __name__ == "__main__":
    main()
