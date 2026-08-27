# BRISC Brain Tumor MRI - Streamlit Demo

Standalone deployment repo for the BRISC 2025 baseline models: upload an MRI slice
(or pick a demo image) and run classification (EfficientNetB0 / MobileNetV3) or
segmentation (UNet / DeepLabV3+).

## Contents

- `app.py` - Streamlit web app (single image at a time, browser upload)
- `report.py` - PDF report generation, shared by the app and the two scripts below
- `batch_infer.py` - processes a folder of independent images, one PDF report each
- `case_report.py` - processes a folder of slices from **one** scan, picks the single
  "best view" slice, and writes **one** consolidated PDF report (used by `send_to_pi.py`
  in the companion `brain-tumor` repo - see "Wireless pipeline" below)
- `pi_thermal.py` - shared SoC-temperature/throttle-status helpers (used by
  `benchmark.py` and `case_report.py`/`batch_infer.py`); no-ops off-Pi
- `benchmark.py` - standalone inference-latency benchmark, all 4 model combinations
  (no Streamlit needed)
- `case_report_jetson.py` - Jetson counterpart to `case_report.py` (used by
  `send_to_jetson.py` in the companion `brain-tumor` repo)
- `jetson_thermal.py` - shared thermal-zone/`tegrastats` helpers for Jetson scripts;
  no-ops off-Jetson
- `benchmark_jetson.py` - standalone inference-latency benchmark on Jetson (CUDA if
  available), all 4 model combinations
- `requirements.txt` - Python dependencies
- `requirements-pi.txt` - dependencies for ARM / Raspberry Pi (CPU-only torch)
- `requirements-jetson-nano.txt` - dependencies for Jetson Nano / JetPack (see the
  file's own comments for the full torch/torchvision-from-source setup)
- `models/` - trained checkpoints (`*_best.pt`), tracked via Git LFS
- `demo_images/<class>/` - sample MRI slices per class (glioma, meningioma,
  no_tumor, pituitary) for the "Use this image" demo buttons

## Wireless pipeline (NIfTI scan -> Pi -> report)

This repo runs on the Pi and is the receiving end of a pipeline that starts on a
separate local computer, in the companion **`brain-tumor`** repo:

```
.nii scan  --[nii_to_slices.py]-->  PNG slices  --[send_to_pi.py, SSH/SCP]-->
    (this repo's case_report.py runs here on the Pi)  --[SCP]-->  one PDF report
```

`send_to_pi.py` (and its browser wrapper, `nii_web_app.py`) live in the `brain-tumor`
repo, not here - see that repo's README for the local-computer side.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Run on Raspberry Pi (or other ARM CPU-only device)

Use `requirements-pi.txt` instead of `requirements.txt` - it installs the
CPU-only PyTorch build so pip doesn't pull in `nvidia-cuda-*`/cublas/cudnn
packages that are irrelevant on ARM:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-pi.txt
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Before relying on it, check real inference latency on the device (also pauses
if the SoC gets hot, and reports whether throttling occurred):

```bash
python benchmark.py
```

### Measured on a Raspberry Pi 4 (4GB, CPU-only)

Test image: `demo_images/meningioma/brisc2025_test_00287_me_ax_t1.jpg` - 2 warmup runs,
10 timed runs per stage.

Individual models (ms):

| Model | Mean | Median | p95 | Min | Max |
|---|---|---|---|---|---|
| Classifier: EfficientNetB0 | 282.0 | 280.1 | 308.7 | 274.8 | 308.7 |
| Classifier: MobileNetV3 | 189.3 | 187.2 | 207.5 | 186.2 | 207.5 |
| Segmenter: DeepLabV3+ | 1007.6 | 1004.5 | 1045.7 | 992.6 | 1045.7 |
| Segmenter: UNet | 926.5 | 924.3 | 938.4 | 918.6 | 938.4 |

All 4 classifier + segmenter combinations, full pipeline (ms):

| Combination | Mean | Median | p95 | Min | Max |
|---|---|---|---|---|---|
| EfficientNetB0 + DeepLabV3+ (app default) | 1295.5 | 1292.7 | 1325.5 | 1288.7 | 1325.5 |
| EfficientNetB0 + UNet | 1203.9 | 1203.1 | 1223.0 | 1194.1 | 1223.0 |
| MobileNetV3 + DeepLabV3+ | 1196.3 | 1197.8 | 1200.5 | 1188.5 | 1200.5 |
| MobileNetV3 + UNet | 1114.6 | 1114.8 | 1125.0 | 1102.2 | 1125.0 |

Note: checkpoint file size doesn't predict latency here - UNet (93.4 MB) is
faster than DeepLabV3+ (85.8 MB) on this hardware, since DeepLabV3+'s ASPP
decoder is more compute-heavy per-FLOP than UNet's simpler decoder. The
app's default pair (EfficientNetB0 + DeepLabV3+) is kept for best reported
accuracy (99.2% classification F1, 76.3% segmentation mIoU) even though
it's the slowest of the 4 on this device; MobileNetV3 + UNet is ~14%
faster with nearly identical segmentation accuracy (75.7% mIoU) but a real
drop in classification accuracy (94.5% F1). Switch the default in
`app.py`'s Advanced Settings if latency matters more than accuracy for
your use case.

SoC temperature rose 44.3C -> 55.0C over the full benchmark with no
throttling (peak RSS ~676 MB with all 4 models loaded simultaneously).
Comfortably within the 4GB/32GB budget - see `benchmark.py` to reproduce or
re-check after any dependency/model changes.

## Run on Jetson Nano

Use `requirements-jetson-nano.txt` instead of `requirements.txt` - see that file's own
comments for the full setup (apt-installed numpy/Pillow/matplotlib, a
`--system-site-packages` venv, NVIDIA's JetPack-matched PyTorch wheel, and torchvision
built from source, since none of these have usable prebuilt wheels for Jetson's
ARM+CUDA combination on older JetPack images):

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements-jetson-nano.txt
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Before relying on it, check real inference latency on the device (uses CUDA
automatically if available, pauses if a thermal zone gets hot):

```bash
python benchmark_jetson.py
```

### Measured on a Jetson Nano (4GB, CUDA)

Test image: `demo_images/meningioma/brisc2025_test_00413_me_co_t1.jpg` - 5 warmup runs,
20 timed runs per stage, `torch.cuda.synchronize()` around every timed call since GPU
kernel launches are asynchronous.

Individual models (ms):

| Model | Mean | Median | p95 | Min | Max |
|---|---|---|---|---|---|
| Classifier: MobileNetV3 | 40.7 | 39.8 | 48.2 | 34.3 | 48.2 |
| Classifier: EfficientNetB0 | 51.4 | 50.5 | 59.3 | 46.2 | 59.3 |
| Segmenter: UNet | 100.7 | 100.3 | 105.1 | 99.2 | 105.1 |
| Segmenter: DeepLabV3+ | 111.7 | 111.6 | 113.7 | 111.1 | 113.7 |

All 4 classifier + segmenter combinations, full pipeline (ms):

| Combination | Mean | Median | p95 | Min | Max |
|---|---|---|---|---|---|
| MobileNetV3 + UNet | 141.1 | 141.2 | 143.5 | 137.1 | 143.5 |
| EfficientNetB0 + UNet | 151.7 | 151.8 | 158.0 | 146.6 | 158.0 |
| MobileNetV3 + DeepLabV3+ | 151.8 | 152.0 | 154.8 | 149.0 | 154.8 |
| EfficientNetB0 + DeepLabV3+ (app default) | 162.9 | 163.6 | 167.0 | 158.9 | 167.0 |

Roughly **8-9x faster than the Pi** on the same default pair (162.9ms vs 1295.5ms),
since the Nano's Tegra X1 CUDA GPU can actually parallelize these convolution-heavy
models instead of walking them through 4 CPU threads one op at a time. Peak RSS with
all four models loaded was **3023-3041 MB** (vs 676 MB on the Pi) - close enough to the
Nano's 3964 MB total that spawning even a small extra subprocess while all four models
are resident can fail with `OSError: Cannot allocate memory`; `jetson_thermal.py`
degrades gracefully instead of crashing when that happens.

Captured with `tegrastats` running concurrently in a separate SSH session during the
benchmark: peak instantaneous power draw **~7.3W** (`POM_5V_IN`), peak GPU power
**~4.4W**, steady running average **~2.4-2.7W**, idle baseline **~0.9-1.5W**. Thermal
zones peaked at 48.5C (AO-therm) with no throttling - comfortably below this SoC's
throttle point in the 90s.

### Pi 4 (CPU) vs Colab T4 (GPU)

Same 4 model combinations, same methodology, run on a Colab T4 GPU via
`colab_gpu_benchmark.ipynb` (in the `brain-tumor` repo) for a direct
hardware comparison:

| Model | Pi 4 mean (ms) | T4 mean (ms) | Speedup |
|---|---|---|---|
| Classifier: EfficientNetB0 | 282.0 | 8.76 | 32.2x |
| Classifier: MobileNetV3 | 189.3 | 6.10 | 31.0x |
| Segmenter: DeepLabV3+ | 1007.6 | 9.76 | 103.2x |
| Segmenter: UNet | 926.5 | 6.51 | 142.3x |

| Combination | Pi 4 mean (ms) | T4 mean (ms) | Speedup |
|---|---|---|---|
| EfficientNetB0 + DeepLabV3+ (app default) | 1295.5 | 14.82 | 87.4x |
| EfficientNetB0 + UNet | 1203.9 | 13.87 | 86.8x |
| MobileNetV3 + DeepLabV3+ | 1196.3 | 12.48 | 95.9x |
| MobileNetV3 + UNet | 1114.6 | 12.52 | 89.0x |

Segmentation gains far more from the GPU than classification (103-142x vs
31-32x) - the heavier ResNet34-encoder segmenters parallelize better than
the lighter classifiers. The UNet-faster-than-DeepLabV3+ finding holds on
both platforms, confirming it's an architecture-compute difference (ASPP's
dilated convolutions cost more per FLOP), not a CPU-specific quirk.

Correctness check on the T4: all 20 demo images correctly classified
(matching their folder labels), with sensible tumor-area segmentation on
every tumor-class image and segmentation correctly skipped on every
no_tumor image.

Methodology note: the two runs used different single test images for the
warmup/timed-run loop (Pi: a meningioma slice, `..._00287`; T4: a
different meningioma slice, `..._00413`) and different run counts (Pi: 2
warmup + 10 timed; T4: 5 warmup + 20 timed, with `torch.cuda.synchronize()`
around every call since GPU kernel launches are asynchronous). Both are
representative same-format slices, so the comparison is fair, but it
wasn't literally the same file on both runs.

Source project / training notebook: https://github.com/sarvan-2187/brain-tumor-edge
