# BRISC Brain Tumor MRI — Streamlit Demo

Standalone deployment repo for the BRISC 2025 baseline models: upload an MRI slice
(or pick a demo image) and run classification (EfficientNetB0 / MobileNetV3) or
segmentation (UNet / DeepLabV3+).

## Contents

- `app.py` — Streamlit app
- `report.py` — PDF report generation
- `benchmark.py` — standalone inference-latency benchmark (no Streamlit needed)
- `requirements.txt` — Python dependencies
- `requirements-pi.txt` — dependencies for ARM / Raspberry Pi (CPU-only torch)
- `models/` — trained checkpoints (`*_best.pt`), tracked via Git LFS
- `demo_images/<class>/` — sample MRI slices per class (glioma, meningioma,
  no_tumor, pituitary) for the "Use this image" demo buttons

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Run on Raspberry Pi (or other ARM CPU-only device)

Use `requirements-pi.txt` instead of `requirements.txt` — it installs the
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

Individual models:

| Model | Latency |
|---|---|
| Classifier: EfficientNetB0 | ~282 ms |
| Classifier: MobileNetV3 | ~189 ms |
| Segmenter: DeepLabV3+ | ~1008 ms |
| Segmenter: UNet | ~927 ms |

All 4 classifier + segmenter combinations (full pipeline):

| Combination | Full pipeline |
|---|---|
| EfficientNetB0 + DeepLabV3+ (app default) | ~1296 ms |
| EfficientNetB0 + UNet | ~1204 ms |
| MobileNetV3 + DeepLabV3+ | ~1196 ms |
| MobileNetV3 + UNet | ~1115 ms |

Note: checkpoint file size doesn't predict latency here — UNet (93.4 MB) is
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
Comfortably within the 4GB/32GB budget — see `benchmark.py` to reproduce or
re-check after any dependency/model changes.

Source project / training notebook: https://github.com/sarvan-2187/brain-tumor-edge
