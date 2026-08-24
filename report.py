"""PDF report generation for BRISC pipeline results. No Streamlit/torch deps
so it can be smoke-tested standalone (see __main__ below).

A "combo" is one (classifier, segmenter) pipeline run: classification is
independent per classifier; segmentation (if the classifier's verdict gates
it on) is independent per segmenter. The report covers all 4 combinations,
one page each, followed by a consolidated summary page.
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


def summarize_classifiers(combos):
    """One representative result per classifier_name, in first-seen order."""
    seen = {}
    for c in combos:
        seen.setdefault(c["classifier_name"], c)
    return list(seen.values())


def describe_consensus(cls_results):
    """cls_results: the 2 unique per-classifier dicts from summarize_classifiers."""
    a, b = cls_results[0], cls_results[1]
    agree = a["top_class"] == b["top_class"]
    if agree:
        text = (
            f"Both classifiers agree: predicted class is "
            f"{a['top_class'].replace('_', ' ').title()} "
            f"({display_name(a['classifier_name'])}: {a['top_conf']:.1%}, "
            f"{display_name(b['classifier_name'])}: {b['top_conf']:.1%})."
        )
    else:
        text = (
            f"Classifiers disagree. {display_name(a['classifier_name'])} predicts "
            f"{a['top_class'].replace('_', ' ').title()} ({a['top_conf']:.1%}); "
            f"{display_name(b['classifier_name'])} predicts "
            f"{b['top_class'].replace('_', ' ').title()} ({b['top_conf']:.1%}). "
            f"Review all combinations for detail."
        )
    return agree, text


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


def _comparison_chart(cls_results, class_names):
    fig, ax = plt.subplots(figsize=(5.8, 2.8))
    x = np.arange(len(class_names))
    width = 0.35
    colors = ["#2c3e50", "#2980b9"]
    for i, c in enumerate(cls_results):
        offset = (i - (len(cls_results) - 1) / 2) * width
        ax.bar(x + offset, c["probs"] * 100, width,
               label=display_name(c["classifier_name"]), color=colors[i % len(colors)])
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", " ").title() for n in class_names])
    ax.set_ylabel("Confidence (%)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8, frameon=False)
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


def _combo_page(pdf, index, total, combo, class_names, threshold):
    clf_label = display_name(combo["classifier_name"])
    seg_label = display_name(combo["segmenter_name"])

    pdf.add_page()
    _section_title(pdf, f"Combination {index} of {total}: {clf_label} + {seg_label}")

    _image_row(pdf, combo["image"], combo["mask"])
    _verdict_box(pdf, combo["verdict_kind"], combo["top_class"], combo["top_conf"])

    _subheading(pdf, "Classification confidence")
    chart = _confidence_chart(np.asarray(combo["probs"]), class_names)
    pdf.image(chart, x=10, w=140)
    pdf.ln(4)

    if combo["mask"] is not None:
        _subheading(pdf, "Tumor metrics")
        tumor_px = int(combo["mask"].sum())
        frame_px = int(combo["mask"].size)
        _kv_table(pdf, [
            ("Tumor area", f"{100 * tumor_px / frame_px:.1f}% of frame"),
            ("Tumor pixels", f"{tumor_px} / {frame_px}"),
        ])
        pdf.ln(3)

    _subheading(pdf, "Run details")
    _kv_table(pdf, [
        ("Classifier", clf_label),
        ("Segmenter", seg_label if combo["mask"] is not None else "Not run"),
        ("Confidence threshold", f"{threshold:.0%}"),
    ])


def _summary_page(pdf, combos, class_names, threshold):
    cls_results = summarize_classifiers(combos)
    _, consensus_text = describe_consensus(cls_results)

    pdf.add_page()
    _section_title(pdf, "Consolidated Summary")

    _subheading(pdf, "Classifier agreement")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6, consensus_text)
    pdf.ln(3)

    _subheading(pdf, "Confidence by classifier")
    chart = _comparison_chart(cls_results, class_names)
    pdf.image(chart, x=10, w=150)
    pdf.ln(4)

    _subheading(pdf, "Per-classifier results")
    rows = []
    for c in cls_results:
        seg_ran = any(x["classifier_name"] == c["classifier_name"] and x["mask"] is not None for x in combos)
        rows.append((
            display_name(c["classifier_name"]),
            f"{c['top_class'].replace('_', ' ').title()}, {c['top_conf']:.1%} "
            f"(segmentation {'run' if seg_ran else 'skipped'})",
        ))
    _kv_table(pdf, rows)
    pdf.ln(3)

    _subheading(pdf, "Report parameters")
    _kv_table(pdf, [("Confidence threshold", f"{threshold:.0%}")])
    pdf.ln(6)

    pdf.set_font("Helvetica", "I", 8.5)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(0, 4.5, "Research and demonstration tool only. Not a certified diagnostic device "
                          "and not intended for clinical use. Findings should always be reviewed by "
                          "a qualified radiologist.")
    pdf.set_text_color(0, 0, 0)


def build_pdf_report(image, combos, class_names, threshold):
    """combos: list of 4 dicts, one per (classifier, segmenter) pair, each with
    classifier_name, segmenter_name, probs, top_class, top_conf, verdict_kind,
    mask (numpy bool array or None). Renders one page per combination (in
    order) followed by a consolidated summary page.
    """
    pdf = ReportPDF(unit="mm", format="A4")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)

    for i, combo in enumerate(combos, start=1):
        _combo_page(pdf, i, len(combos), {**combo, "image": image}, class_names, threshold)

    _summary_page(pdf, combos, class_names, threshold)

    return bytes(pdf.output())


if __name__ == "__main__":
    dummy_img = Image.new("RGB", (256, 256), (120, 120, 120))
    mask_a = np.zeros((256, 256), dtype=bool)
    mask_a[80:160, 80:160] = True
    class_names = ["glioma", "meningioma", "pituitary", "no_tumor"]

    probs_a = np.array([0.82, 0.05, 0.05, 0.08])
    probs_b = np.array([0.61, 0.10, 0.09, 0.20])

    combos = [
        {"classifier_name": "efficientnet_b0", "segmenter_name": "deeplabv3plus",
         "probs": probs_a, "top_class": "glioma", "top_conf": 0.82,
         "verdict_kind": "tumor", "mask": mask_a},
        {"classifier_name": "efficientnet_b0", "segmenter_name": "unet",
         "probs": probs_a, "top_class": "glioma", "top_conf": 0.82,
         "verdict_kind": "tumor", "mask": mask_a},
        {"classifier_name": "mobilenet_v3", "segmenter_name": "deeplabv3plus",
         "probs": probs_b, "top_class": "glioma", "top_conf": 0.61,
         "verdict_kind": "uncertain", "mask": None},
        {"classifier_name": "mobilenet_v3", "segmenter_name": "unet",
         "probs": probs_b, "top_class": "glioma", "top_conf": 0.61,
         "verdict_kind": "uncertain", "mask": None},
    ]

    pdf_bytes = build_pdf_report(dummy_img, combos, class_names, 0.70)
    assert pdf_bytes[:4] == b"%PDF", "output is not a valid PDF"
    assert len(pdf_bytes) > 1000, "PDF suspiciously small"
    print(f"OK - comprehensive report {len(pdf_bytes)} bytes, {len(combos) + 1} pages")
