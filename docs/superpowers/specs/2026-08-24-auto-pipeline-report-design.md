# Auto Classify→Segment Pipeline + PDF Report — Design

Date: 2026-08-24
Repo: brisc-deployment (brain-tumor-deploy)

## Problem

The current app (`app.py`) makes the user pick "Classification" or "Segmentation"
as separate tasks and separate models. That's not how the clinical workflow
works, and it's not simple. We want: user picks/uploads one image, the app
figures out the rest.

## Goals

- One automatic pipeline: classify → segment (only if a tumor is found with
  high enough confidence) → render a clean on-screen report → offer a PDF
  download of that report.
- Keep model choice available (all 4 permutations: 2 classifiers × 2
  segmenters) but tucked into an "Advanced settings" expander, not the main
  flow.
- Confidence threshold (adjustable) gates whether segmentation runs.
- No "Run inference" button — selecting/uploading an image is the trigger.

## Non-goals

- No patient data collection, no auth, no persistence/database.
- No new model training or architecture changes.
- No multi-page Streamlit app — stays a single page.

## Pipeline logic

1. Run the selected classifier on the image → softmax probabilities over
   `["glioma", "meningioma", "pituitary", "no_tumor"]`.
2. `top_class, top_conf = argmax(probs)`.
3. Verdict:
   - `top_class == "no_tumor"` → **No tumor detected** (green). Segmentation
     skipped.
   - `top_class != "no_tumor"` and `top_conf >= threshold` → **Tumor
     detected: {class}** (red). Run the selected segmentation model, compute
     binary mask, tumor-area % of frame.
   - `top_class != "no_tumor"` and `top_conf < threshold` → **Uncertain —
     confidence below threshold** (yellow). Segmentation skipped. Report
     recommends clinical/radiologist review.
4. Threshold default: 0.70, adjustable 0.50–0.95 via slider.

## UI

- Header + one-line caption.
- Demo image gallery: tabs per class (Glioma / Meningioma / No Tumor /
  Pituitary), 5 thumbnails each, "Use this image" button per thumbnail.
- File uploader below the gallery ("...or upload your own MRI slice").
- Collapsed `st.expander("Advanced settings")`:
  - classifier dropdown: `efficientnet_b0` (default) / `mobilenet_v3`
  - segmenter dropdown: `deeplabv3plus` (default) / `unet`
  - confidence threshold slider: default 0.70
- As soon as an image is selected (upload or demo click), the pipeline runs
  and renders:
  - Verdict banner (colored `st.success` / `st.error` / `st.warning`)
  - Input image; if segmented, mask overlay (red) next to it
  - Confidence bar chart (all 4 classes, `st.bar_chart`)
  - Tumor-area metric (if segmented): `st.metric`
  - `st.download_button("Download PDF Report", ...)`

Caching: `st.cache_resource` for model loading (unchanged from current app).
Inference results cached per `(image_bytes_hash, classifier_name,
segmenter_name, threshold)` via `st.cache_data` so re-running the script
(e.g. from an unrelated widget interaction) doesn't redo work.

## PDF report (`report.py`)

New module, function `build_pdf_report(image, verdict, probs, class_names,
mask, tumor_area_pct, classifier_name, segmenter_name, threshold) -> bytes`.

Library: `fpdf2` (pure Python, no system dependencies — matters for
Streamlit Community Cloud). Confidence bar chart rendered to a PNG via
`matplotlib` (Agg backend, headless) and embedded as an image.

Layout (single page, A4):

1. Header: title, generation timestamp, thin rule.
2. Images row: input MRI slice; mask overlay next to it if segmentation ran
   (otherwise this column is omitted and the input image is shown larger).
3. Verdict box: colored background matching the on-screen banner, verdict
   text + confidence %.
4. Confidence breakdown: embedded bar chart image (all 4 classes).
5. Tumor metrics table (only if segmented): tumor pixel count, frame pixel
   count, tumor area %.
6. Metadata table: classifier used, segmenter used (or "not run"),
   confidence threshold applied, timestamp.
7. Footer disclaimer: "Research/demo tool — not a diagnostic device. Not for
   clinical use."

`report.py` includes a `if __name__ == "__main__":` self-check that builds a
report from synthetic data (blank image, fake probs, fake mask) and asserts
the output is non-empty PDF bytes (`%PDF` header).

## Files touched

- `app.py` — rewritten: single pipeline flow replacing task-selector UI.
- `report.py` — new.
- `requirements.txt` — add `fpdf2`, `matplotlib`.

## Deployment

Streamlit Community Cloud reads directly from the GitHub repo
(`sarvan-2187/brisc-deployment`), so once this code is committed and pushed,
deployment is: go to https://share.streamlit.io, "New app", pick the repo/
branch, set main file to `app.py`, deploy. This step needs the user's GitHub
login and can't be done on their behalf — instructions will be handed off
after the code is pushed.

## Testing

- `report.py` self-check (`python report.py`) runs without torch/model
  dependencies — verifies PDF generation logic in isolation.
- Full pipeline (`streamlit run app.py`) needs `torch`, `timm`,
  `segmentation-models-pytorch` installed — to be smoke-tested by the user
  locally before relying on Streamlit Cloud's build.
