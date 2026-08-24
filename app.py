"""Streamlit demo: automatic brain-tumor classify -> segment pipeline.

Select or upload one MRI slice; the app classifies it and, only if a tumor
type is predicted with high enough confidence, runs segmentation too. Model
architectures/preprocessing mirror train_colab.ipynb exactly so the
checkpoints load cleanly.
"""

import io
from pathlib import Path

import segmentation_models_pytorch as smp
import streamlit as st
import timm
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v3_large

from report import build_pdf_report, overlay_mask

CKPT_DIR = Path("models")
DEMO_DIR = Path("demo_images")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["glioma", "meningioma", "pituitary", "no_tumor"]
TUMOR_CLASSES = {"glioma", "meningioma", "pituitary"}
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
IMG_SIZE_CLS = 224
IMG_SIZE_SEG = 256

CLASSIFIER_CHOICES = ["efficientnet_b0", "mobilenet_v3"]
SEGMENTER_CHOICES = ["deeplabv3plus", "unet"]

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
    m.load_state_dict(torch.load(CKPT_DIR / f"{name}_best.pt", map_location=DEVICE))
    return m.to(DEVICE).eval()


@st.cache_resource
def load_seg_model(name):
    cls = smp.Unet if name == "unet" else smp.DeepLabV3Plus
    m = cls(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
    m.load_state_dict(torch.load(CKPT_DIR / f"{name}_best.pt", map_location=DEVICE))
    return m.to(DEVICE).eval()


@st.cache_data
def list_demo_images():
    return {
        cls_name: sorted((DEMO_DIR / cls_name).glob("*.jpg"))
        for cls_name in CLASS_NAMES
        if (DEMO_DIR / cls_name).is_dir()
    }


@st.cache_data(show_spinner="Running pipeline...")
def run_pipeline(image_bytes, classifier_name, segmenter_name, threshold):
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
            mask = (torch.sigmoid(seg_model(x_seg))[0, 0].cpu().numpy() > 0.5)
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


st.set_page_config(page_title="BRISC Brain Tumor MRI", layout="centered")

st.markdown(
    """
    <style>
    @import url('https://cdn.jsdelivr.net/npm/@fontsource/geist@5/index.css');
    @import url('https://cdn.jsdelivr.net/npm/@fontsource/geist@5/500.css');
    @import url('https://cdn.jsdelivr.net/npm/@fontsource/geist@5/600.css');

    html, body, [class*="css"] {
        font-family: 'Geist', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    #MainMenu, footer, header {visibility: hidden;}

    .block-container {
        padding-top: 3rem;
        max-width: 760px;
    }

    h1 {
        font-weight: 600;
        letter-spacing: -0.02em;
    }

    div[data-testid="stMetricValue"] {
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Brain Tumor MRI Analysis")
st.caption("Upload or select an MRI slice. The system classifies it and segments the tumor automatically.")

if "image_bytes" not in st.session_state:
    st.session_state.image_bytes = None
if "last_upload_id" not in st.session_state:
    st.session_state.last_upload_id = None

with st.expander("Try a demo image"):
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

uploaded = st.file_uploader("Or upload your own MRI slice (jpg/png)", type=["jpg", "jpeg", "png"])
if uploaded is not None:
    upload_id = (uploaded.name, uploaded.size)
    if upload_id != st.session_state.last_upload_id:
        st.session_state.image_bytes = uploaded.getvalue()
        st.session_state.last_upload_id = upload_id

with st.expander("Advanced settings"):
    classifier_name = st.selectbox("Classifier", CLASSIFIER_CHOICES)
    segmenter_name = st.selectbox("Segmenter", SEGMENTER_CHOICES)
    threshold = st.slider("Confidence threshold for running segmentation", 0.50, 0.95, 0.70, 0.05)

if st.session_state.image_bytes is not None:
    image = Image.open(io.BytesIO(st.session_state.image_bytes)).convert("RGB")

    result = run_pipeline(st.session_state.image_bytes, classifier_name, segmenter_name, threshold)
    verdict_kind = result["verdict_kind"]
    top_class, top_conf = result["top_class"], result["top_conf"]
    mask = result["mask"]

    if verdict_kind == "tumor":
        st.error(f"Tumor detected: **{top_class.title()}** — {top_conf:.1%} confidence")
    elif verdict_kind == "no_tumor":
        st.success(f"No tumor detected — {top_conf:.1%} confidence")
    else:
        st.warning(
            f"Uncertain: top prediction **{top_class.title()}** at {top_conf:.1%}, "
            f"below the {threshold:.0%} threshold. Segmentation skipped — clinical review recommended."
        )

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Input", width="stretch")
    if mask is not None:
        with col2:
            overlay = overlay_mask(image, mask)
            st.image(overlay, caption="Predicted tumor mask (red)", width="stretch")
        st.metric("Tumor area", f"{mask.mean():.1%} of frame")

    st.subheader("Classification confidence")
    st.bar_chart({"confidence": dict(zip(CLASS_NAMES, result["probs"].tolist()))})

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
