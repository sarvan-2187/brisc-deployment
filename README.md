# BRISC Brain Tumor MRI — Streamlit Demo

Standalone deployment repo for the BRISC 2025 baseline models: upload an MRI slice
(or pick a demo image) and run classification (EfficientNetB0 / MobileNetV3) or
segmentation (UNet / DeepLabV3+).

## Contents

- `app.py` — Streamlit app
- `requirements.txt` — Python dependencies
- `models/` — trained checkpoints (`*_best.pt`), tracked via Git LFS
- `demo_images/<class>/` — 5 sample MRI slices per class (glioma, meningioma,
  no_tumor, pituitary) for the "Use this image" demo buttons

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Source project / training notebook: https://github.com/sarvan-2187/brain-tumor-edge
