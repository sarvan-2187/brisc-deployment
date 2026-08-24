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

VERDICT_COLORS = {
    "tumor": (231, 76, 60),
    "no_tumor": (39, 174, 96),
    "uncertain": (243, 156, 18),
}
VERDICT_LABELS = {
    "tumor": "Tumor detected",
    "no_tumor": "No tumor detected",
    "uncertain": "Uncertain - review recommended",
}


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
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def build_pdf_report(image, verdict_kind, top_class, top_conf, probs, class_names,
                      mask, classifier_name, segmenter_name, threshold):
    """Returns the report as PDF bytes.

    mask=None means segmentation didn't run (no_tumor verdict, or uncertain
    classification below threshold); segmenter_name is ignored in that case.
    """
    pdf = FPDF(unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 10, "Brain Tumor MRI Analysis Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(100, 100, 100)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    pdf.cell(0, 6, f"Generated {timestamp}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(2)
    pdf.set_draw_color(200, 200, 200)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(6)

    # ponytail: assumes roughly square input images (true for BRISC slices) so
    # the caption below each image lines up; a wildly non-square upload would
    # just misalign the caption a few mm, not break the report.
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
        pdf.cell(img_w, 5, "Predicted tumor mask (red)", align="C")

    pdf.set_y(y0 + img_w + 14)

    color = VERDICT_COLORS[verdict_kind]
    pdf.set_fill_color(*color)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 13)
    label = VERDICT_LABELS[verdict_kind]
    if verdict_kind == "no_tumor":
        text = f"  {label} - {top_conf:.1%} confidence"
    else:
        text = f"  {label}: {top_class.replace('_', ' ').title()} - {top_conf:.1%} confidence"
    pdf.cell(0, 12, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT, fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(6)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Classification confidence", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    chart = _confidence_chart(np.asarray(probs), class_names)
    pdf.image(chart, x=10, w=140)
    pdf.ln(4)

    if mask is not None:
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "Tumor metrics", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", "", 10)
        tumor_px = int(mask.sum())
        frame_px = int(mask.size)
        pdf.cell(0, 6, f"Tumor area: {100 * tumor_px / frame_px:.1f}% of frame "
                       f"({tumor_px} / {frame_px} px)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(2)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 7, "Run details", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Classifier: {classifier_name}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 6, f"Segmenter: {segmenter_name if mask is not None else 'not run'}",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 6, f"Confidence threshold: {threshold:.0%}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_y(-20)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(0, 4, "Research/demo tool only - not a diagnostic device. "
                         "Not for clinical use. Always consult a qualified radiologist.")

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

    print(f"OK — tumor report {len(pdf_bytes)} bytes, no-tumor report {len(pdf_bytes_no_seg)} bytes")
