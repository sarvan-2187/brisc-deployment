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
- `requirements.txt` - Python dependencies
- `requirements-pi.txt` - dependencies for ARM / Raspberry Pi (CPU-only torch)
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

Source project / training notebook: https://github.com/sarvan-2187/brain-tumor-edge
