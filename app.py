"""Streamlit demo: automatic brain-tumor classify -> segment pipeline.

Select or upload one MRI slice. All 2 classifiers x 2 segmenters (4
combinations) run automatically; segmentation for a given combination is
skipped unless its classifier predicts a tumor type above the confidence
threshold. Model architectures/preprocessing mirror train_colab.ipynb
exactly so the checkpoints load cleanly.
"""

import io
from pathlib import Path

import pandas as pd
import segmentation_models_pytorch as smp
import streamlit as st
import timm
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v3_large

from report import build_pdf_report, display_name, overlay_mask, summarize_classifiers, describe_consensus

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


@st.cache_data(show_spinner=False)
def classify_image(image_bytes, classifier_name):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    model = load_cls_model(classifier_name)
    x = cls_tf(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
    return probs


@st.cache_data(show_spinner=False)
def segment_image(image_bytes, segmenter_name):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    model = load_seg_model(segmenter_name)
    x = seg_tf(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        mask = torch.sigmoid(model(x))[0, 0].cpu().numpy() > 0.5
    return mask


@st.cache_data(show_spinner="Running all model combinations...")
def run_all_combos(image_bytes, threshold):
    cls_results = {}
    for classifier_name in CLASSIFIER_CHOICES:
        probs = classify_image(image_bytes, classifier_name)
        top_idx = int(probs.argmax())
        top_class = CLASS_NAMES[top_idx]
        top_conf = float(probs[top_idx])
        gate = top_class in TUMOR_CLASSES and top_conf >= threshold
        verdict_kind = "tumor" if gate else ("uncertain" if top_class in TUMOR_CLASSES else "no_tumor")
        cls_results[classifier_name] = {
            "classifier_name": classifier_name, "probs": probs, "top_class": top_class,
            "top_conf": top_conf, "verdict_kind": verdict_kind, "gate": gate,
        }

    seg_cache = {}
    combos = []
    for classifier_name in CLASSIFIER_CHOICES:
        cr = cls_results[classifier_name]
        for segmenter_name in SEGMENTER_CHOICES:
            mask = None
            if cr["gate"]:
                if segmenter_name not in seg_cache:
                    seg_cache[segmenter_name] = segment_image(image_bytes, segmenter_name)
                mask = seg_cache[segmenter_name]
            combos.append({
                "classifier_name": classifier_name,
                "segmenter_name": segmenter_name,
                "probs": cr["probs"],
                "top_class": cr["top_class"],
                "top_conf": cr["top_conf"],
                "verdict_kind": cr["verdict_kind"],
                "mask": mask,
            })
    return combos


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
        max-width: 800px;
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
st.caption(
    "Upload or select an MRI slice. All classifier and segmentation model combinations "
    "run automatically and are compiled into one report."
)

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
    threshold = st.slider("Confidence threshold for running segmentation", 0.50, 0.95, 0.70, 0.05)

if st.session_state.image_bytes is not None:
    image = Image.open(io.BytesIO(st.session_state.image_bytes)).convert("RGB")
    combos = run_all_combos(st.session_state.image_bytes, threshold)
    cls_results = summarize_classifiers(combos)
    agree, consensus_text = describe_consensus(cls_results)

    st.subheader("Consolidated summary")
    if agree:
        top_class = cls_results[0]["top_class"]
        (st.success if top_class == "no_tumor" else st.error)(consensus_text)
    else:
        st.warning(consensus_text)

    chart_df = pd.DataFrame(
        {display_name(c["classifier_name"]): c["probs"] for c in cls_results},
        index=[n.replace("_", " ").title() for n in CLASS_NAMES],
    )
    st.bar_chart(chart_df)

    st.subheader("All model combinations")
    tab_labels = [f"{display_name(c['classifier_name'])} + {display_name(c['segmenter_name'])}" for c in combos]
    tabs = st.tabs(tab_labels)
    for tab, combo in zip(tabs, combos):
        with tab:
            verdict_kind, top_class, top_conf = combo["verdict_kind"], combo["top_class"], combo["top_conf"]
            if verdict_kind == "tumor":
                st.error(f"Tumor detected: **{top_class.title()}**, {top_conf:.1%} confidence")
            elif verdict_kind == "no_tumor":
                st.success(f"No tumor detected, {top_conf:.1%} confidence")
            else:
                st.warning(
                    f"Uncertain: top prediction **{top_class.title()}** at {top_conf:.1%}, "
                    f"below the {threshold:.0%} threshold. Segmentation skipped."
                )

            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Input", width="stretch")
            if combo["mask"] is not None:
                with col2:
                    overlay = overlay_mask(image, combo["mask"])
                    st.image(overlay, caption="Predicted tumor mask", width="stretch")
                st.metric("Tumor area", f"{combo['mask'].mean():.1%} of frame")

    pdf_bytes = build_pdf_report(image, combos, CLASS_NAMES, threshold)
    st.download_button(
        "Download comprehensive PDF report", data=pdf_bytes,
        file_name="brain_tumor_report.pdf", mime="application/pdf",
    )
else:
    st.info("Pick a demo image above or upload your own MRI slice to get started.")
