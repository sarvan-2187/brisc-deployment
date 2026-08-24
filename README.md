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

### Measured on a Raspberry Pi 4 (4GB, EfficientNetB0 + DeepLabV3+, CPU-only)

| Stage | Latency |
|---|---|
| Classification | ~0.29 s |
| Segmentation | ~1.0 s |
| Full pipeline | ~1.28 s |

SoC temperature rose 46.7C -> 56.0C over a 10-run benchmark with no
throttling (peak RSS ~552 MB for the benchmark process alone). Comfortably
within the 4GB/32GB budget — see `benchmark.py` to reproduce or re-check
after any dependency/model changes.

Source project / training notebook: https://github.com/sarvan-2187/brain-tumor-edge
