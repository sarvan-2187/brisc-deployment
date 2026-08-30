"""Streamlit demo: automatic brain-tumor classify -> segment pipeline.

Select or upload one MRI slice; the app classifies it and, only if a tumor
type is predicted with high enough confidence, runs segmentation too. Model
architectures/preprocessing mirror train_colab.ipynb exactly so the
checkpoints load cleanly.
"""

import io
import time
from pathlib import Path

import segmentation_models_pytorch as smp
import streamlit as st
import streamlit.components.v1 as components
import timm
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v3_large

from adaptive_controller import HIGH_ACCURACY, LIGHTWEIGHT, AdaptiveController
from pi_thermal import read_temp_c
from report import build_pdf_report, display_name, overlay_mask

CKPT_DIR = Path("models")
DEMO_DIR = Path("demo_images")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["glioma", "meningioma", "pituitary", "no_tumor"]
TUMOR_CLASSES = {"glioma", "meningioma", "pituitary"}
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
IMG_SIZE_CLS = 224
IMG_SIZE_SEG = 256

# Adaptive Model Controller's two tiers (see adaptive_controller.py for the
# rolling-latency / SoC-temperature / hysteresis switching logic itself).
TIER_CHOICES = {
    "High-Accuracy Tier (EfficientNetB0 + DeepLabV3+)": HIGH_ACCURACY,
    "Lightweight Tier (MobileNetV3 + UNet)": LIGHTWEIGHT,
}

# Verdict banner colors. Every one of these carries white body text, so each is
# darkened until it clears WCAG AA (4.5:1): the lighter #27AE60 / #D38508 that
# report.py uses for PDF fills only reach 2.87:1 and 2.95:1 against white.
# The word in VERDICT_TEXT does the work on its own, so the color is never the
# only thing distinguishing one verdict from another.
VERDICT_COLOR = {"tumor": "#C0392B", "no_tumor": "#1E8449", "uncertain": "#8B5E04"}
VERDICT_TEXT = {
    "tumor": "Tumor detected",
    "no_tumor": "No tumor detected",
    "uncertain": "Uncertain, clinical review recommended",
}

cls_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE_CLS, IMG_SIZE_CLS)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
seg_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE_SEG, IMG_SIZE_SEG)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


@st.cache_resource
def load_cls_model(name):
    if name == "efficientnet_b0":
        m = timm.create_model("efficientnet_b0", pretrained=False, num_classes=len(CLASS_NAMES))
    else:
        m = mobilenet_v3_large(weights=None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, len(CLASS_NAMES))
    m.load_state_dict(torch.load(CKPT_DIR / f"{name}_best.pt", map_location=DEVICE, weights_only=False))
    return m.to(DEVICE).eval()


@st.cache_resource
def load_seg_model(name):
    cls = smp.Unet if name == "unet" else smp.DeepLabV3Plus
    m = cls(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
    m.load_state_dict(torch.load(CKPT_DIR / f"{name}_best.pt", map_location=DEVICE, weights_only=False))
    return m.to(DEVICE).eval()


def select_tier(controller, manual_override, manual_pair):
    """Adaptive Model Controller stage: pick the active tier. Manual
    override bypasses the controller with a fixed tier choice; otherwise
    the controller's rolling-latency/SoC-temperature hysteresis rule
    decides (see adaptive_controller.py)."""
    if manual_override:
        return manual_pair["classifier"], manual_pair["segmenter"], None
    pair = controller.select_pair()
    return pair["classifier"], pair["segmenter"], controller.current_tier


@st.cache_data
def list_demo_images():
    return {
        cls_name: sorted((DEMO_DIR / cls_name).glob("*.jpg"))
        for cls_name in CLASS_NAMES
        if (DEMO_DIR / cls_name).is_dir()
    }


@st.cache_data(show_spinner="Running pipeline...")
def run_classification_and_segmentation(image_bytes, classifier_name, segmenter_name, threshold):
    """Classification & Segmentation stage: classify with the tier's
    classifier, then segment with the tier's segmenter only if a tumor
    class clears the confidence threshold."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    cls_model = load_cls_model(classifier_name)
    x = cls_tf(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(cls_model(x), dim=1)[0].cpu().numpy()
    top_idx = int(probs.argmax())
    top_class = CLASS_NAMES[top_idx]
    top_conf = float(probs[top_idx])

    mask = None
    if top_class in TUMOR_CLASSES and top_conf >= threshold:
        verdict_kind = "tumor"
        seg_model = load_seg_model(segmenter_name)
        x_seg = seg_tf(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            mask = torch.sigmoid(seg_model(x_seg))[0, 0].cpu().numpy() > 0.5
    elif top_class in TUMOR_CLASSES:
        verdict_kind = "uncertain"
    else:
        verdict_kind = "no_tumor"

    return {
        "probs": probs,
        "top_class": top_class,
        "top_conf": top_conf,
        "verdict_kind": verdict_kind,
        "mask": mask,
    }


st.set_page_config(page_title="BRISC-Edge", layout="wide")

st.markdown(
    """
    <style>
    #MainMenu, footer, header {visibility: hidden;}

    .block-container {
        padding-top: 2.5rem;
        max-width: 1180px;
    }

    h1 {
        font-weight: 600;
        letter-spacing: -0.02em;
    }

    div[data-testid="stMetricValue"] {
        font-weight: 600;
    }

    /* Scan images carry the result, so give them a visible frame and let them
       fill their column rather than sitting small in a centered narrow page. */
    div[data-testid="stImage"] img {
        border-radius: 8px;
        border: 1px solid rgba(128, 128, 128, 0.35);
        background: #000;
    }

    .verdict {
        border-radius: 10px;
        padding: 1.1rem 1.4rem;
        margin: 0.4rem 0 1.4rem 0;
        color: #ffffff;
    }
    .verdict .headline {
        font-size: 1.55rem;
        font-weight: 700;
        line-height: 1.25;
    }
    .verdict .detail {
        font-size: 1rem;
        opacity: 0.95;
        margin-top: 0.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("BRISC-Edge: Brain Tumor MRI Analysis")
st.caption("Upload or select an MRI slice. The system classifies it and segments the tumor automatically.")

if "image_bytes" not in st.session_state:
    st.session_state.image_bytes = None
if "last_upload_id" not in st.session_state:
    st.session_state.last_upload_id = None
if "just_selected" not in st.session_state:
    st.session_state.just_selected = False


def _scroll_to_results():
    components.html(
        """
        <script>
        const el = window.parent.document.getElementById('results-anchor');
        if (el) { el.scrollIntoView({behavior: 'smooth', block: 'start'}); }
        </script>
        """,
        height=0,
    )


with st.expander("Try a demo image", expanded=(st.session_state.image_bytes is None)):
    demo_images = list_demo_images()
    tabs = st.tabs([c.replace("_", " ").title() for c in demo_images])
    for tab, (cls_name, paths) in zip(tabs, demo_images.items()):
        with tab:
            cols = st.columns(len(paths))
            for col, path in zip(cols, paths):
                with col:
                    st.image(str(path), width="stretch")
                    if st.button("Use this image", key=f"use_{path.name}"):
                        st.session_state.image_bytes = path.read_bytes()
                        st.session_state.just_selected = True

uploaded = st.file_uploader("Or upload your own MRI slice (jpg/png)", type=["jpg", "jpeg", "png"])
if uploaded is not None:
    upload_id = (uploaded.name, uploaded.size)
    if upload_id != st.session_state.last_upload_id:
        st.session_state.image_bytes = uploaded.getvalue()
        st.session_state.last_upload_id = upload_id
        st.session_state.just_selected = True

if "adaptive_controller" not in st.session_state:
    st.session_state.adaptive_controller = AdaptiveController()
controller = st.session_state.adaptive_controller

with st.expander("Advanced settings"):
    manual_override = st.checkbox("Manual override (force a specific tier)")
    manual_pair = TIER_CHOICES[st.selectbox("Tier", list(TIER_CHOICES))] if manual_override else None
    classifier_name, segmenter_name, active_tier = select_tier(controller, manual_override, manual_pair)
    if active_tier is not None:
        st.caption(
            f"Adaptive tier: **{active_tier.replace('_', ' ')}** "
            f"({classifier_name} + {segmenter_name}) - last switch reason: {controller.last_reason}"
        )
    threshold = st.slider("Confidence threshold for running segmentation", 0.50, 0.95, 0.70, 0.05)

if st.session_state.image_bytes is not None:
    st.markdown('<div id="results-anchor"></div>', unsafe_allow_html=True)
    if st.session_state.just_selected:
        _scroll_to_results()
        st.session_state.just_selected = False

    image = Image.open(io.BytesIO(st.session_state.image_bytes)).convert("RGB")

    _t0 = time.perf_counter()
    result = run_classification_and_segmentation(
        st.session_state.image_bytes, classifier_name, segmenter_name, threshold
    )
    # Feed the measured latency back into the Adaptive Model Controller.
    if not manual_override:
        controller.record_inference((time.perf_counter() - _t0) * 1000)
    verdict_kind = result["verdict_kind"]
    top_class, top_conf = result["top_class"], result["top_conf"]
    mask = result["mask"]

    pretty_class = top_class.replace("_", " ").title()
    headline = pretty_class if verdict_kind != "no_tumor" else VERDICT_TEXT["no_tumor"]
    if verdict_kind == "tumor":
        detail = f"{VERDICT_TEXT['tumor']} at {top_conf:.1%} confidence"
    elif verdict_kind == "no_tumor":
        detail = f"{top_conf:.1%} confidence"
    else:
        detail = (f"Top prediction {pretty_class} at {top_conf:.1%}, below the "
                  f"{threshold:.0%} threshold. Segmentation skipped, clinical review recommended.")

    st.markdown(
        f'<div class="verdict" style="background:{VERDICT_COLOR[verdict_kind]}">'
        f'<div class="headline">{headline}</div>'
        f'<div class="detail">{detail}</div></div>',
        unsafe_allow_html=True,
    )

    # Input beside result, so the mask is readable against the untouched slice.
    left, right = st.columns(2, gap="medium")
    with left:
        st.image(image, caption="Input MRI slice", width="stretch")
    with right:
        if mask is not None:
            st.image(overlay_mask(image, mask),
                     caption="Predicted tumor region, shown in red", width="stretch")
        else:
            reason = ("the slice was classified as no tumor" if verdict_kind == "no_tumor"
                      else f"confidence stayed below the {threshold:.0%} threshold")
            st.info(f"Segmentation did not run because {reason}.")

    st.subheader("Classification confidence")
    # Native progress bars instead of a Vega chart: one less browser-side
    # rendering library for the Pi to serve, and the numbers stay readable.
    for name, p in zip(CLASS_NAMES, result["probs"].tolist()):
        st.progress(p, text=f"{name.replace('_', ' ').title()}  {p:.1%}")

    st.subheader("Device")
    soc_temp = read_temp_c()
    cols = st.columns(4)
    cols[0].metric("Active tier", "Manual" if manual_override else (active_tier or "").replace("_", " ").title())
    cols[1].metric("Models", f"{display_name(classifier_name)} + {display_name(segmenter_name)}")
    cols[2].metric("SoC temperature", f"{soc_temp:.1f} C" if soc_temp is not None else "n/a")
    cols[3].metric("Tumor area", f"{mask.mean():.1%} of frame" if mask is not None else "n/a")

    # Report Generation stage: assemble verdict, confidence, and mask into one PDF.
    pdf_bytes = build_pdf_report(
        image, verdict_kind, top_class, top_conf, result["probs"], CLASS_NAMES,
        mask, classifier_name, segmenter_name if mask is not None else None, threshold,
    )
    st.download_button(
        "Download PDF report", data=pdf_bytes,
        file_name="brain_tumor_report.pdf", mime="application/pdf",
    )
else:
    st.info("Pick a demo image above or upload your own MRI slice to get started.")
