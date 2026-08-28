"""PDF report generation for BRISC-Edge pipeline results. No Streamlit/torch deps
so it can be smoke-tested standalone (see __main__ below).
"""

import io
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from PIL import Image

MODEL_DISPLAY_NAMES = {
    "efficientnet_b0": "EfficientNetB0",
    "mobilenet_v3": "MobileNetV3",
    "unet": "UNet",
    "deeplabv3plus": "DeepLabV3+",
}

ACCENT_COLOR = (44, 62, 80)
VERDICT_COLORS = {
    "tumor": (192, 57, 43),
    "no_tumor": (39, 174, 96),
    "uncertain": (211, 133, 8),
}
VERDICT_LABELS = {
    "tumor": "Tumor detected",
    "no_tumor": "No tumor detected",
    "uncertain": "Uncertain, review recommended",
}

# Geist (SIL OFL) - bundled as .ttf so the PDF doesn't depend on fonts being
# installed on whatever machine renders it. Regular + Bold only: Geist ships
# no italic weight, so italic-styled text below uses Regular at a muted color
# instead of a synthetic/missing italic.
FONTS_DIR = Path(__file__).resolve().parent / "fonts"


def display_name(model_key):
    return MODEL_DISPLAY_NAMES.get(model_key, model_key)


def overlay_mask(image, mask):
    size = mask.shape[::-1]
    base = np.array(image.resize(size).convert("RGB")).astype(np.float32)
    overlay = base.copy()
    overlay[mask] = 0.5 * overlay[mask] + 0.5 * np.array([255, 0, 0])
    return Image.fromarray(overlay.astype(np.uint8))


def _confidence_chart(probs, class_names):
    fig, ax = plt.subplots(figsize=(5.5, 2.6))
    colors = ["#27ae60" if n == "no_tumor" else "#c0392b" for n in class_names]
    labels = [n.replace("_", " ").title() for n in class_names]
    x = np.arange(len(class_names))
    ax.bar(x, probs * 100, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Confidence (%)")
    ax.set_ylim(0, 100)
    for i, p in enumerate(probs * 100):
        ax.text(i, p + 2, f"{p:.1f}%", ha="center", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def _temperature_chart(temp_before_c, temp_after_c):
    """Before/after bar chart of edge-device temperature across the inference
    run. Only called when both readings are available.
    """
    fig, ax = plt.subplots(figsize=(5.5, 2.6))
    labels = ["Before inference", "After inference"]
    values = [temp_before_c, temp_after_c]
    delta = temp_after_c - temp_before_c
    rising_color = "#c0392b" if delta >= 0 else "#27ae60"
    ax.bar(labels, values, color=["#2c3e50", rising_color], width=0.5)
    ax.set_ylabel("Temperature (C)")
    headroom = max(values) * 0.15 or 1
    ax.set_ylim(0, max(values) + headroom)
    for i, v in enumerate(values):
        ax.text(i, v + headroom * 0.15, f"{v:.1f}C", ha="center", fontsize=9)
    ax.set_title(f"{delta:+.1f}C over the run", fontsize=9, color=rising_color)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


class ReportPDF(FPDF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_font("Geist", "", str(FONTS_DIR / "Geist-Regular.ttf"))
        self.add_font("Geist", "B", str(FONTS_DIR / "Geist-Bold.ttf"))

    def header(self):
        self.set_fill_color(*ACCENT_COLOR)
        self.rect(0, 0, self.w, 4, style="F")
        self.set_y(12)

    def footer(self):
        self.set_y(-15)
        self.set_draw_color(220, 220, 220)
        self.line(10, self.get_y(), 200, self.get_y())
        self.set_font("Geist", "", 8)
        self.set_text_color(140, 140, 140)
        self.cell(0, 10, f"Page {self.page_no()} of {{nb}}", align="C")


def _section_title(pdf, title):
    pdf.set_font("Geist", "B", 16)
    pdf.cell(0, 9, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Geist", "", 8)
    pdf.set_text_color(120, 120, 120)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    pdf.cell(0, 5, f"Generated {timestamp}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(1)
    pdf.set_draw_color(*ACCENT_COLOR)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)


def _subheading(pdf, text):
    pdf.set_font("Geist", "B", 11)
    pdf.set_text_color(*ACCENT_COLOR)
    pdf.cell(0, 7, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(0, 0, 0)


def _kv_table(pdf, rows, col1=55, col2=125):
    for label, value in rows:
        pdf.set_font("Geist", "B", 9.5)
        pdf.set_fill_color(245, 246, 248)
        pdf.cell(col1, 7, f"  {label}", border=1, fill=True)
        pdf.set_font("Geist", "", 9.5)
        pdf.cell(col2, 7, f"  {value}", border=1, new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def _verdict_box(pdf, verdict_kind, top_class, top_conf):
    color = VERDICT_COLORS[verdict_kind]
    pdf.set_fill_color(*color)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Geist", "B", 12)
    label = VERDICT_LABELS[verdict_kind]
    if verdict_kind == "no_tumor":
        text = f"  {label}, {top_conf:.1%} confidence"
    else:
        text = f"  {label}: {top_class.replace('_', ' ').title()}, {top_conf:.1%} confidence"
    pdf.cell(0, 10, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)


def _image_row(pdf, image, mask):
    img_w = 85
    y0 = pdf.get_y()
    pdf.image(image.resize((512, 512)), x=10, y=y0, w=img_w)
    pdf.set_xy(10, y0 + img_w + 2)
    pdf.set_font("Geist", "", 8)
    pdf.set_text_color(120, 120, 120)
    pdf.cell(img_w, 5, "Input MRI slice", align="C")
    pdf.set_text_color(0, 0, 0)

    if mask is not None:
        overlay = overlay_mask(image, mask)
        pdf.image(overlay, x=10 + img_w + 10, y=y0, w=img_w)
        pdf.set_xy(10 + img_w + 10, y0 + img_w + 2)
        pdf.set_text_color(120, 120, 120)
        pdf.cell(img_w, 5, "Predicted tumor mask (highlighted)", align="C")
        pdf.set_text_color(0, 0, 0)

    pdf.set_y(y0 + img_w + 14)


def _disclaimer(pdf):
    pdf.set_font("Geist", "", 8.5)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(0, 4.5, "Research and demonstration tool only. Not a certified diagnostic device "
                          "and not intended for clinical use. Findings should always be reviewed by "
                          "a qualified radiologist.")
    pdf.set_text_color(0, 0, 0)


def build_pdf_report(image, verdict_kind, top_class, top_conf, probs, class_names,
                      mask, classifier_name, segmenter_name, threshold):
    """Returns the report as PDF bytes.

    mask=None means segmentation didn't run (no_tumor verdict, or uncertain
    classification below threshold); segmenter_name is ignored in that case.
    """
    clf_label = display_name(classifier_name)
    seg_label = display_name(segmenter_name) if segmenter_name else None

    pdf = ReportPDF(unit="mm", format="A4")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    _section_title(pdf, "BRISC-Edge: Brain Tumor MRI Analysis Report")
    _image_row(pdf, image, mask)
    _verdict_box(pdf, verdict_kind, top_class, top_conf)

    _subheading(pdf, "Classification confidence")
    chart = _confidence_chart(np.asarray(probs), class_names)
    pdf.image(chart, x=10, w=140)
    pdf.ln(4)

    if mask is not None:
        _subheading(pdf, "Tumor metrics")
        tumor_px = int(mask.sum())
        frame_px = int(mask.size)
        _kv_table(pdf, [
            ("Tumor area", f"{100 * tumor_px / frame_px:.1f}% of frame"),
            ("Tumor pixels", f"{tumor_px} / {frame_px}"),
        ])
        pdf.ln(3)

    _subheading(pdf, "Run details")
    _kv_table(pdf, [
        ("Classifier", clf_label),
        ("Segmenter", seg_label if mask is not None else "Not run"),
        ("Confidence threshold", f"{threshold:.0%}"),
    ])
    pdf.ln(6)

    _disclaimer(pdf)

    return bytes(pdf.output())


def _device_performance_table(pdf, device_info):
    """device_info: dict with 'device' (str), 'inference_seconds' (float),
    'temp_before_c'/'temp_after_c' (float or None), and optional 'notes'
    (list of str, e.g. throttling warnings). Any temp field may be None if
    unavailable on that hardware (e.g. running off-device) - rendered as
    "n/a" rather than omitted, so the report is honest about what wasn't
    measured. Just the key/value table + notes; the temperature chart (when
    both readings exist) is drawn separately, on the image/graph page.
    """
    _subheading(pdf, "Device & performance")
    rows = [
        ("Edge device", device_info["device"]),
        ("Inference time", f"{device_info['inference_seconds']:.1f}s"),
    ]
    temp_before, temp_after = device_info.get("temp_before_c"), device_info.get("temp_after_c")
    if temp_before is not None and temp_after is not None:
        rows.append(("Temperature", f"{temp_before:.1f}C -> {temp_after:.1f}C "
                                     f"({temp_after - temp_before:+.1f}C)"))
    else:
        rows.append(("Temperature", "n/a"))
    _kv_table(pdf, rows)
    notes = device_info.get("notes")
    if notes:
        pdf.ln(2)
        pdf.set_font("Geist", "", 8.5)
        pdf.set_text_color(180, 60, 20)
        for note in notes:
            pdf.multi_cell(0, 4.5, f"Note: {note}")
        pdf.set_text_color(0, 0, 0)
    pdf.ln(3)


def _charts_row(pdf, confidence_img, temperature_img):
    """Confidence chart (and, when available, the temperature chart) placed
    side by side rather than stacked - two full-width stacked charts don't
    fit page 1's height budget alongside the verdict box and slice images,
    so this keeps everything on one page instead of spilling an awkward
    near-empty page in between.
    """
    if temperature_img is not None:
        conf_w, temp_w = 100, 78
        y0 = pdf.get_y()
        pdf.image(confidence_img, x=10, y=y0, w=conf_w)
        pdf.image(temperature_img, x=190 - temp_w, y=y0, w=temp_w)
        conf_h = conf_w * confidence_img.height / confidence_img.width
        temp_h = temp_w * temperature_img.height / temperature_img.width
        pdf.set_y(y0 + max(conf_h, temp_h) + 4)
    else:
        pdf.image(confidence_img, x=10, w=140)
        pdf.ln(4)


def build_case_pdf_report(slice_results, best_image, best_mask, overall_verdict,
                           best_top_class, best_top_conf, best_probs, class_names,
                           classifier_name, segmenter_name, threshold, device_info=None):
    """One consolidated report for every slice belonging to a single case
    (e.g. one patient's scan), instead of one report per slice.

    Page 1 is all images and graphs (best-view slice + mask, classification
    confidence chart, temperature-change chart if device_info has both
    readings) so the visual summary is skimmable at a glance; page 2 is
    every table (tumor metrics, run details, device performance, the
    full per-slice breakdown) for anyone who wants the numbers.

    slice_results: list of dicts with 'filename', 'top_class', 'top_conf',
    'verdict_kind' - one entry per slice, for the per-slice traceability
    table. best_image/best_mask: the single most representative ("best
    view") slice's image and segmentation mask (best_mask is None if no
    slice was confidently a tumor, so segmentation never ran).

    device_info: optional dict (see _device_performance_table) adding an
    "edge device" section to the report - device name, inference wall-clock
    time, and temperature before/after. Omitted entirely if None, so
    build_case_pdf_report keeps working for any caller that doesn't have
    this data (e.g. a non-edge run).
    """
    clf_label = display_name(classifier_name)
    seg_label = display_name(segmenter_name) if segmenter_name else None

    pdf = ReportPDF(unit="mm", format="A4")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    # --- Page 1: images and graphs ---
    pdf.add_page()

    _section_title(pdf, "BRISC-Edge: Brain Tumor MRI Case Report")
    _verdict_box(pdf, overall_verdict, best_top_class, best_top_conf)

    _subheading(pdf, f"Best view ({len(slice_results)} slice(s) analyzed)")
    _image_row(pdf, best_image, best_mask)

    temp_before = temp_after = None
    if device_info is not None:
        temp_before, temp_after = device_info.get("temp_before_c"), device_info.get("temp_after_c")
    has_temp_chart = temp_before is not None and temp_after is not None

    _subheading(pdf, "Classification confidence & temperature change (best view)"
                     if has_temp_chart else "Classification confidence (best view)")
    confidence_chart = _confidence_chart(np.asarray(best_probs), class_names)
    temperature_chart = _temperature_chart(temp_before, temp_after) if has_temp_chart else None
    _charts_row(pdf, confidence_chart, temperature_chart)

    # --- Page 2: every table ---
    pdf.add_page()

    if best_mask is not None:
        _subheading(pdf, "Tumor metrics (best view)")
        tumor_px = int(best_mask.sum())
        frame_px = int(best_mask.size)
        _kv_table(pdf, [
            ("Tumor area", f"{100 * tumor_px / frame_px:.1f}% of frame"),
            ("Tumor pixels", f"{tumor_px} / {frame_px}"),
        ])
        pdf.ln(3)

    _subheading(pdf, "Run details")
    _kv_table(pdf, [
        ("Slices analyzed", str(len(slice_results))),
        ("Classifier", clf_label),
        ("Segmenter", seg_label if best_mask is not None else "Not run"),
        ("Confidence threshold", f"{threshold:.0%}"),
    ])
    pdf.ln(4)

    if device_info is not None:
        _device_performance_table(pdf, device_info)

    _subheading(pdf, "Per-slice results")
    pdf.set_font("Geist", "B", 8.5)
    pdf.set_fill_color(245, 246, 248)
    pdf.cell(75, 6, "  Slice", border=1, fill=True)
    pdf.cell(45, 6, "  Predicted class", border=1, fill=True)
    pdf.cell(30, 6, "  Confidence", border=1, fill=True)
    pdf.cell(30, 6, "  Result", border=1, new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
    pdf.set_font("Geist", "", 8.5)
    for r in slice_results:
        pdf.cell(75, 6, f"  {r['filename']}", border=1)
        pdf.cell(45, 6, f"  {r['top_class'].replace('_', ' ').title()}", border=1)
        pdf.cell(30, 6, f"  {r['top_conf']:.1%}", border=1)
        pdf.cell(30, 6, f"  {r['verdict_kind']}", border=1, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    _disclaimer(pdf)

    return bytes(pdf.output())


if __name__ == "__main__":
    dummy_img = Image.new("RGB", (256, 256), (120, 120, 120))
    dummy_probs = np.array([0.82, 0.05, 0.05, 0.08])
    dummy_mask = np.zeros((256, 256), dtype=bool)
    dummy_mask[80:160, 80:160] = True
    class_names = ["glioma", "meningioma", "pituitary", "no_tumor"]

    pdf_bytes = build_pdf_report(
        dummy_img, "tumor", "glioma", 0.82, dummy_probs, class_names,
        dummy_mask, "efficientnet_b0", "deeplabv3plus", 0.70,
    )
    assert pdf_bytes[:4] == b"%PDF", "output is not a valid PDF"
    assert len(pdf_bytes) > 1000, "PDF suspiciously small"

    pdf_bytes_no_seg = build_pdf_report(
        dummy_img, "no_tumor", "no_tumor", 0.95, dummy_probs, class_names,
        None, "efficientnet_b0", None, 0.70,
    )
    assert pdf_bytes_no_seg[:4] == b"%PDF"

    slice_results = [
        {"filename": "slice_0070.png", "top_class": "no_tumor", "top_conf": 0.55, "verdict_kind": "no_tumor"},
        {"filename": "slice_0072.png", "top_class": "glioma", "top_conf": 0.61, "verdict_kind": "uncertain"},
        {"filename": "slice_0074.png", "top_class": "glioma", "top_conf": 0.91, "verdict_kind": "tumor"},
        {"filename": "slice_0076.png", "top_class": "glioma", "top_conf": 0.78, "verdict_kind": "tumor"},
    ]
    case_pdf_bytes = build_case_pdf_report(
        slice_results, dummy_img, dummy_mask, "tumor", "glioma", 0.91, dummy_probs,
        class_names, "efficientnet_b0", "deeplabv3plus", 0.70,
    )
    assert case_pdf_bytes[:4] == b"%PDF"

    case_pdf_bytes_with_device = build_case_pdf_report(
        slice_results, dummy_img, dummy_mask, "tumor", "glioma", 0.91, dummy_probs,
        class_names, "efficientnet_b0", "deeplabv3plus", 0.70,
        device_info={
            "device": "Raspberry Pi 4 (CPU)", "inference_seconds": 12.4,
            "temp_before_c": 44.1, "temp_after_c": 58.7,
            "notes": ["currently throttled"],
        },
    )
    assert case_pdf_bytes_with_device[:4] == b"%PDF"
    assert len(case_pdf_bytes_with_device) > len(case_pdf_bytes), \
        "device_info section didn't add any content"

    case_pdf_bytes_no_temp = build_case_pdf_report(
        slice_results, dummy_img, dummy_mask, "tumor", "glioma", 0.91, dummy_probs,
        class_names, "efficientnet_b0", "deeplabv3plus", 0.70,
        device_info={"device": "Jetson Nano (CUDA)", "inference_seconds": 3.2,
                     "temp_before_c": None, "temp_after_c": None},
    )
    assert case_pdf_bytes_no_temp[:4] == b"%PDF"
    assert len(case_pdf_bytes) > 1000

    print(f"OK - tumor report {len(pdf_bytes)} bytes, no-tumor report {len(pdf_bytes_no_seg)} bytes, "
          f"case report {len(case_pdf_bytes)} bytes, case report with device chart "
          f"{len(case_pdf_bytes_with_device)} bytes")
