"""PDF report generation for BRISC pipeline results. No Streamlit/torch deps
so it can be smoke-tested standalone (see __main__ below).
"""

import io
from datetime import datetime, timezone

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
    ax.bar(labels, probs * 100, color=colors)
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


class ReportPDF(FPDF):
    def header(self):
        self.set_fill_color(*ACCENT_COLOR)
        self.rect(0, 0, self.w, 4, style="F")
        self.set_y(12)

    def footer(self):
        self.set_y(-15)
        self.set_draw_color(220, 220, 220)
        self.line(10, self.get_y(), 200, self.get_y())
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(140, 140, 140)
        self.cell(0, 10, f"Page {self.page_no()} of {{nb}}", align="C")


def _section_title(pdf, title):
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 9, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(120, 120, 120)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    pdf.cell(0, 5, f"Generated {timestamp}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(1)
    pdf.set_draw_color(*ACCENT_COLOR)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(5)


def _subheading(pdf, text):
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(*ACCENT_COLOR)
    pdf.cell(0, 7, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(0, 0, 0)


def _kv_table(pdf, rows, col1=55, col2=125):
    for label, value in rows:
        pdf.set_font("Helvetica", "B", 9.5)
        pdf.set_fill_color(245, 246, 248)
        pdf.cell(col1, 7, f"  {label}", border=1, fill=True)
        pdf.set_font("Helvetica", "", 9.5)
        pdf.cell(col2, 7, f"  {value}", border=1, new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def _verdict_box(pdf, verdict_kind, top_class, top_conf):
    color = VERDICT_COLORS[verdict_kind]
    pdf.set_fill_color(*color)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 12)
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
    pdf.set_font("Helvetica", "I", 8)
    pdf.cell(img_w, 5, "Input MRI slice", align="C")

    if mask is not None:
        overlay = overlay_mask(image, mask)
        pdf.image(overlay, x=10 + img_w + 10, y=y0, w=img_w)
        pdf.set_xy(10 + img_w + 10, y0 + img_w + 2)
        pdf.cell(img_w, 5, "Predicted tumor mask (highlighted)", align="C")

    pdf.set_y(y0 + img_w + 14)


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

    _section_title(pdf, "Brain Tumor MRI Analysis Report")
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

    pdf.set_font("Helvetica", "I", 8.5)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(0, 4.5, "Research and demonstration tool only. Not a certified diagnostic device "
                          "and not intended for clinical use. Findings should always be reviewed by "
                          "a qualified radiologist.")
    pdf.set_text_color(0, 0, 0)

    return bytes(pdf.output())


def build_case_pdf_report(slice_results, best_image, best_mask, overall_verdict,
                           best_top_class, best_top_conf, best_probs, class_names,
                           classifier_name, segmenter_name, threshold):
    """One consolidated report for every slice belonging to a single case
    (e.g. one patient's scan), instead of one report per slice.

    slice_results: list of dicts with 'filename', 'top_class', 'top_conf',
    'verdict_kind' - one entry per slice, for the per-slice traceability
    table. best_image/best_mask: the single most representative ("best
    view") slice's image and segmentation mask (best_mask is None if no
    slice was confidently a tumor, so segmentation never ran).
    """
    clf_label = display_name(classifier_name)
    seg_label = display_name(segmenter_name) if segmenter_name else None

    pdf = ReportPDF(unit="mm", format="A4")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    _section_title(pdf, "Brain Tumor MRI Case Report")
    _verdict_box(pdf, overall_verdict, best_top_class, best_top_conf)

    _subheading(pdf, f"Best view ({len(slice_results)} slice(s) analyzed)")
    _image_row(pdf, best_image, best_mask)

    _subheading(pdf, "Classification confidence (best view)")
    chart = _confidence_chart(np.asarray(best_probs), class_names)
    pdf.image(chart, x=10, w=140)
    pdf.ln(4)

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

    _subheading(pdf, "Per-slice results")
    pdf.set_font("Helvetica", "B", 8.5)
    pdf.set_fill_color(245, 246, 248)
    pdf.cell(75, 6, "  Slice", border=1, fill=True)
    pdf.cell(45, 6, "  Predicted class", border=1, fill=True)
    pdf.cell(30, 6, "  Confidence", border=1, fill=True)
    pdf.cell(30, 6, "  Result", border=1, new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
    pdf.set_font("Helvetica", "", 8.5)
    for r in slice_results:
        pdf.cell(75, 6, f"  {r['filename']}", border=1)
        pdf.cell(45, 6, f"  {r['top_class'].replace('_', ' ').title()}", border=1)
        pdf.cell(30, 6, f"  {r['top_conf']:.1%}", border=1)
        pdf.cell(30, 6, f"  {r['verdict_kind']}", border=1, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    pdf.set_font("Helvetica", "I", 8.5)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(0, 4.5, "Research and demonstration tool only. Not a certified diagnostic device "
                          "and not intended for clinical use. Findings should always be reviewed by "
                          "a qualified radiologist.")
    pdf.set_text_color(0, 0, 0)

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
    assert len(case_pdf_bytes) > 1000

    print(f"OK - tumor report {len(pdf_bytes)} bytes, no-tumor report {len(pdf_bytes_no_seg)} bytes, "
          f"case report {len(case_pdf_bytes)} bytes")
