"""Measure real inference latency for all 4 classifier x segmenter
combinations on an NVIDIA Jetson (Nano or newer), CUDA if available.

This is the Jetson counterpart to benchmark.py (which targets the
Raspberry Pi's CPU-only case). A Jetson Nano has an actual CUDA GPU, so
this script uses it when present, and its thermal/power tooling differs
from the Pi's (no vcgencmd; multiple named thermal zones instead of one;
tegrastats instead of jtop). Kept as a separate script rather than
reusing pi_thermal.py, since that module's thermal-zone and vcgencmd
assumptions are Broadcom/Pi specific and would silently mislead on Jetson
hardware.

Prerequisite: install PyTorch from NVIDIA's JetPack-matched wheel, not the
generic pip/CPU wheel used on the Pi - the standard PyPI wheels are not
built for Jetson's ARM+CUDA combination. See:
https://developer.nvidia.com/embedded/downloads (search "PyTorch for
Jetson") and pick the wheel matching your JetPack/L4T version.

Usage:
    python benchmark_jetson.py
    python benchmark_jetson.py --runs 20 --cooldown 1.0 --max-temp 70
    python benchmark_jetson.py --image demo_images/glioma/some_image.jpg
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

from jetson_thermal import read_tegrastats_line, read_thermal_zones, wait_for_safe_temp

try:
    import resource
except ImportError:
    resource = None  # not available on Windows; this script targets Linux/L4T

CKPT_DIR = Path("models")
CLASS_NAMES = ["glioma", "meningioma", "pituitary", "no_tumor"]
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
IMG_SIZE_CLS = 224
IMG_SIZE_SEG = 256

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


def load_cls_model(name, device):
    if name == "efficientnet_b0":
        m = timm.create_model("efficientnet_b0", pretrained=False, num_classes=len(CLASS_NAMES))
    else:
        m = mobilenet_v3_large(weights=None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, len(CLASS_NAMES))
    m.load_state_dict(torch.load(CKPT_DIR / f"{name}_best.pt", map_location=device, weights_only=False))
    return m.to(device).eval()


def load_seg_model(name, device):
    cls = smp.Unet if name == "unet" else smp.DeepLabV3Plus
    m = cls(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
    m.load_state_dict(torch.load(CKPT_DIR / f"{name}_best.pt", map_location=device, weights_only=False))
    return m.to(device).eval()


def time_calls(fn, warmup, runs, cooldown, max_temp, synchronize):
    for _ in range(warmup):
        fn()
    if synchronize:
        torch.cuda.synchronize()
    times = []
    for _ in range(runs):
        wait_for_safe_temp(max_temp)
        if synchronize:
            torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        if synchronize:
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
        if cooldown > 0:
            time.sleep(cooldown)
    return times


def summarize(label, times):
    ms = sorted(t * 1000 for t in times)
    mean = statistics.mean(ms)
    median = statistics.median(ms)
    p95 = ms[min(len(ms) - 1, int(len(ms) * 0.95))]
    print(f"{label:38s} mean {mean:7.2f} ms  median {median:7.2f} ms  "
          f"p95 {p95:7.2f} ms  min {ms[0]:7.2f} ms  max {ms[-1]:7.2f} ms")


def peak_memory_mb():
    if resource is None:
        return None
    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return kb / 1024  # ru_maxrss is KB on Linux


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image", default=None, help="Test image path; defaults to the first demo image found")
    parser.add_argument("--warmup", type=int, default=5, help="Untimed warmup runs per stage (default 5)")
    parser.add_argument("--runs", type=int, default=20, help="Timed runs to average per stage (default 20)")
    parser.add_argument("--cooldown", type=float, default=0.2,
                         help="Seconds to sleep between each timed run (default 0.2, 0 to disable)")
    parser.add_argument("--max-temp", type=float, default=80.0,
                         help="Pause and wait if the hottest thermal zone reaches this many C (default 80)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu"))
    synchronize = device.type == "cuda"

    image_path = Path(args.image) if args.image else next(Path("demo_images").rglob("*.jpg"))
    image = Image.open(image_path).convert("RGB")

    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if synchronize else ""))
    if not synchronize and not args.cpu:
        print("WARNING: CUDA not available - falling back to CPU. If this is a Jetson, "
              "check that the JetPack-matched PyTorch wheel is installed, not a generic "
              "CPU-only build (see this script's docstring).")
    print(f"Test image: {image_path}")
    print(f"Warmup: {args.warmup}  Runs: {args.runs}  Cooldown: {args.cooldown}s  Max temp: {args.max_temp}C")

    zones = read_thermal_zones()
    if zones:
        print("Thermal zones at start: " + ", ".join(f"{n}={t:.1f}C" for n, t in zones.items()))
    else:
        print("Thermal zones: unavailable on this device - thermal pausing is disabled.")
    print()

    print("Loading all 4 checkpoints...")
    cls_models = {name: load_cls_model(name, device) for name in CLASSIFIER_CHOICES}
    seg_models = {name: load_seg_model(name, device) for name in SEGMENTER_CHOICES}
    print()

    cls_input = cls_tf(image).unsqueeze(0).to(device)
    seg_input = seg_tf(image).unsqueeze(0).to(device)

    def run_cls(name):
        with torch.no_grad():
            cls_models[name](cls_input)

    def run_seg(name):
        with torch.no_grad():
            seg_models[name](seg_input)

    print("--- Individual models ---")
    for name in CLASSIFIER_CHOICES:
        summarize(f"Classifier: {DISPLAY_NAMES[name]}",
                  time_calls(lambda n=name: run_cls(n), args.warmup, args.runs,
                             args.cooldown, args.max_temp, synchronize))
    for name in SEGMENTER_CHOICES:
        summarize(f"Segmenter: {DISPLAY_NAMES[name]}",
                  time_calls(lambda n=name: run_seg(n), args.warmup, args.runs,
                             args.cooldown, args.max_temp, synchronize))

    print("\n--- All 4 classifier + segmenter combinations (full pipeline) ---")
    for cls_name in CLASSIFIER_CHOICES:
        for seg_name in SEGMENTER_CHOICES:
            def run_pipeline(c=cls_name, s=seg_name):
                run_cls(c)
                run_seg(s)

            label = f"{DISPLAY_NAMES[cls_name]} + {DISPLAY_NAMES[seg_name]}"
            summarize(label, time_calls(run_pipeline, args.warmup, args.runs,
                                         args.cooldown, args.max_temp, synchronize))

    end_zones = read_thermal_zones()
    if end_zones:
        print("\nThermal zones at end: " + ", ".join(f"{n}={t:.1f}C" for n, t in end_zones.items()))

    tegra_line = read_tegrastats_line()
    if tegra_line:
        print(f"tegrastats snapshot: {tegra_line}")
    else:
        print("(tegrastats not available - power draw not captured. Run `sudo jtop` or "
              "`tegrastats` separately alongside this benchmark for power/GPU-load figures.)")

    peak = peak_memory_mb()
    if peak is not None:
        print(f"Peak RSS this process: {peak:.0f} MB (all 4 models loaded simultaneously)")
    else:
        print("(Peak RSS reporting needs the `resource` module - Linux only, not available here.)")


if __name__ == "__main__":
    main()
