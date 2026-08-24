"""Streamlit demo for BRISC 2025 brain tumor classification + segmentation.

Standalone deployment repo: model checkpoints live in models/, demo MRI
slices live in demo_images/<class>/. Architectures and preprocessing mirror
the training notebook exactly (same encoder, input size, normalization).
"""

from pathlib import Path

import numpy as np
import segmentation_models_pytorch as smp
import streamlit as st
import timm
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v3_large

CKPT_DIR = Path("models")
DEMO_DIR = Path("demo_images")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["glioma", "meningioma", "pituitary", "no_tumor"]
IMAGENET_MEAN, IMAGENET_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
IMG_SIZE_CLS = 224
IMG_SIZE_SEG = 256

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


def predict_cls(model, image):
    x = cls_tf(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
    return probs


def predict_seg(model, image):
    x = seg_tf(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        mask = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    return mask > 0.5


def overlay_mask(image, mask):
    base = np.array(image.resize((IMG_SIZE_SEG, IMG_SIZE_SEG)).convert("RGB")).astype(np.float32)
    overlay = base.copy()
    overlay[mask] = 0.5 * overlay[mask] + 0.5 * np.array([255, 0, 0])
    return overlay.astype(np.uint8)


st.set_page_config(page_title="BRISC Brain Tumor MRI", layout="centered")
st.title("Brain Tumor MRI — Classification & Segmentation")
st.caption("BRISC 2025 baselines (EfficientNetB0 / MobileNetV3 / UNet / DeepLabV3+)")

if "selected_image_path" not in st.session_state:
    st.session_state.selected_image_path = None

task = st.sidebar.radio("Task", ["Classification", "Segmentation"])
if task == "Classification":
    model_name = st.sidebar.selectbox("Model", ["efficientnet_b0", "mobilenet_v3"])
else:
    model_name = st.sidebar.selectbox("Model", ["unet", "deeplabv3plus"])

st.subheader("Demo images")
demo_images = list_demo_images()
tabs = st.tabs([c.replace("_", " ").title() for c in demo_images])
for tab, (cls_name, paths) in zip(tabs, demo_images.items()):
    with tab:
        cols = st.columns(len(paths))
        for col, path in zip(cols, paths):
            with col:
                st.image(str(path), use_container_width=True)
                if st.button("Use this image", key=f"use_{path.name}"):
                    st.session_state.selected_image_path = path

st.subheader("Input")
uploaded = st.file_uploader("...or upload your own MRI slice (jpg/png)", type=["jpg", "jpeg", "png"])
if uploaded is not None:
    st.session_state.selected_image_path = None
    image = Image.open(uploaded).convert("RGB")
elif st.session_state.selected_image_path is not None:
    image = Image.open(st.session_state.selected_image_path).convert("RGB")
else:
    image = None

if image is not None:
    st.image(image, caption="Selected input", use_container_width=True)

    if st.button("Run inference"):
        if task == "Classification":
            model = load_cls_model(model_name)
            probs = predict_cls(model, image)
            pred_idx = int(probs.argmax())
            st.success(f"Prediction: **{CLASS_NAMES[pred_idx]}** ({probs[pred_idx]:.1%})")
            st.bar_chart({"probability": dict(zip(CLASS_NAMES, probs.tolist()))})
        else:
            model = load_seg_model(model_name)
            mask = predict_seg(model, image)
            st.image(overlay_mask(image, mask), caption="Predicted tumor mask (red)", use_container_width=True)
            st.caption(f"Tumor pixels: {mask.sum()} / {mask.size} ({mask.mean():.1%} of frame)")
