"""Measure real classification/segmentation inference latency on this device.

No Streamlit dependency, so it can run standalone on a Raspberry Pi to
validate (or correct) the feasibility estimates before relying on them.

Thermal safety: a Pi 4 without a heatsink can hit its throttle point within
minutes of sustained multi-core load, which is exactly what a long benchmark
run is. This script checks the SoC temperature before every timed iteration
and pauses if it's running hot, and reports whether throttling actually
occurred so the numbers you get can be trusted.

Usage:
    python benchmark.py
    python benchmark.py --classifier mobilenet_v3 --segmenter unet --runs 20
    python benchmark.py --image demo_images/glioma/some_image.jpg
    python benchmark.py --cooldown 1.0 --max-temp 70
"""

import argparse
import statistics
import subprocess
import time
from pathlib import Path

import segmentation_models_pytorch as smp
import timm
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v3_large

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

THERMAL_ZONE = Path("/sys/class/thermal/thermal_zone0/temp")
THROTTLE_FLAGS = {
    0: "under-voltage now",
    1: "arm frequency capped now",
    2: "currently throttled",
    3: "soft temperature limit active now",
    16: "under-voltage has occurred",
    17: "arm frequency capping has occurred",
    18: "throttling has occurred",
    19: "soft temperature limit has occurred",
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


def read_temp_c():
    try:
        return int(THERMAL_ZONE.read_text().strip()) / 1000
    except (FileNotFoundError, ValueError):
        return None


def read_throttled_flags():
    try:
        out = subprocess.run(
            ["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=2,
        )
        value = int(out.stdout.strip().split("=")[1], 16)
    except (FileNotFoundError, subprocess.SubprocessError, IndexError, ValueError):
        return None
    return [desc for bit, desc in THROTTLE_FLAGS.items() if value & (1 << bit)]


def wait_for_safe_temp(max_temp, poll_seconds=5):
    temp = read_temp_c()
    if temp is None or temp < max_temp:
        return
    print(f"  SoC at {temp:.1f}C (limit {max_temp:.1f}C) - pausing to cool down...")
    while temp is not None and temp >= max_temp:
        time.sleep(poll_seconds)
        temp = read_temp_c()
    if temp is not None:
        print(f"  Back to {temp:.1f}C, resuming.\n")


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
    print(f"{label:32s} mean {mean:7.1f} ms  median {median:7.1f} ms  "
          f"p95 {p95:7.1f} ms  min {ms[0]:7.1f} ms  max {ms[-1]:7.1f} ms")


def peak_memory_mb():
    if resource is None:
        return None
    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return kb / 1024  # ru_maxrss is KB on Linux


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--classifier", default="efficientnet_b0", choices=["efficientnet_b0", "mobilenet_v3"])
    parser.add_argument("--segmenter", default="deeplabv3plus", choices=["deeplabv3plus", "unet"])
    parser.add_argument("--image", default=None, help="Test image path; defaults to the first demo image found")
    parser.add_argument("--warmup", type=int, default=2, help="Untimed warmup runs (default 2)")
    parser.add_argument("--runs", type=int, default=10, help="Timed runs to average (default 10)")
    parser.add_argument("--cooldown", type=float, default=0.5,
                         help="Seconds to sleep between each timed run (default 0.5, 0 to disable)")
    parser.add_argument("--max-temp", type=float, default=75.0,
                         help="Pause and wait if SoC temp reaches this many C (default 75)")
    args = parser.parse_args()

    image_path = Path(args.image) if args.image else next(Path("demo_images").rglob("*.jpg"))
    image = Image.open(image_path).convert("RGB")

    start_temp = read_temp_c()
    print(f"Threads available to torch: {torch.get_num_threads()}")
    print(f"Test image: {image_path}")
    print(f"Warmup runs: {args.warmup}  Timed runs: {args.runs}  "
          f"Cooldown: {args.cooldown}s  Max temp: {args.max_temp}C")
    if start_temp is not None:
        print(f"SoC temp at start: {start_temp:.1f}C")
    else:
        print("SoC temp: unavailable on this device (no /sys/class/thermal/thermal_zone0) - "
              "thermal pausing is disabled.")
    print()

    print(f"Loading classifier: {args.classifier}")
    cls_model = load_cls_model(args.classifier)
    print(f"Loading segmenter:  {args.segmenter}\n")
    seg_model = load_seg_model(args.segmenter)

    cls_input = cls_tf(image).unsqueeze(0)
    seg_input = seg_tf(image).unsqueeze(0)

    def run_cls():
        with torch.no_grad():
            cls_model(cls_input)

    def run_seg():
        with torch.no_grad():
            seg_model(seg_input)

    def run_pipeline():
        run_cls()
        run_seg()

    summarize(f"Classification ({args.classifier})",
              time_calls(run_cls, args.warmup, args.runs, args.cooldown, args.max_temp))
    summarize(f"Segmentation ({args.segmenter})",
              time_calls(run_seg, args.warmup, args.runs, args.cooldown, args.max_temp))
    summarize("Full pipeline",
              time_calls(run_pipeline, args.warmup, args.runs, args.cooldown, args.max_temp))

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
        print(f"Peak RSS this process: {peak:.0f} MB")
    else:
        print("(Peak RSS reporting needs the `resource` module - Linux/Pi only, not available here.)")


if __name__ == "__main__":
    main()
