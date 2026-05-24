"""
streamlit_app.py
-----------------
Streamlit web app for animal disease detection with Grad-CAM.

"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json
import numpy as np
import torch
import streamlit as st
from PIL import Image

from inference import load_model_and_classes, predict

st.set_page_config(
    page_title="Animal Disease Detector",
    page_icon="🐾",
    layout="centered",
)

st.markdown("""
<style>
    .main-title   { font-size: 2rem; font-weight: 700; color: #1f4e79; }
    .subtitle     { font-size: 1rem; color: #555; margin-bottom: 1.5rem; }
    .pred-box     { padding: 1rem 1.5rem; border-radius: 10px; margin-top: 1rem; }
    .pred-healthy { background: #57ad6c; border-left: 5px solid #28a745; }
    .pred-disease { background: #de6671; border-left: 5px solid #dc3545; }
    .info-text    { font-size: 0.85rem; color: #777; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(species: str, model_dir="models/"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model, class_names = load_model_and_classes(model_dir, species, device)
        return model, class_names, device
    except FileNotFoundError:
        return None, None, None


st.markdown('<div class="main-title">🐾 Animal Disease Detector</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Upload an image — the model will classify the condition '
    'and highlight the affected region using <strong>Grad-CAM</strong>.</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("⚙️ Settings")
    species    = st.selectbox("Species", ["cats", "dogs"], index=0)
    model_dir  = st.text_input("Model directory", value="models/")
    img_size   = st.slider("Image size", 128, 384, 224, step=32)
    st.divider()

    if species == "cats":
        st.caption("**Cat classes:** healthy | mange | acne | abscess")
    else:
        st.caption("**Dog classes:** healthy dog | dog with bacterial dermatosis | dog with lupus | dog with alopecia")

    st.caption("Built with EfficientNet-B0 + Grad-CAM")

model, class_names, device = load_model(species, model_dir)

if model is None:
    st.error(
        f"Model not found for **{species}**. Train it first:\n\n"
        f"```bash\npython src/train.py --species {species}\n```"
    )
    st.stop()

st.success(f"**{species.capitalize()}** model loaded  |  Classes: **{', '.join(class_names)}**")

uploaded_file = st.file_uploader(
    f"Upload a {species[:-1]} image", 
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    help="Clear, close-up photo of a single animal works best.",
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    with st.spinner("Running inference …"):
        pred_class, confidence, all_probs, cam_image, rgb_img = predict(
            image, model, class_names, device, img_size
        )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original image")
        st.image(image, use_container_width=True)
    with col2:
        st.subheader("Grad-CAM heatmap")
        st.image(cam_image, use_container_width=True)
        st.caption("🔴 Warmer = model focused here for the prediction")

    healthy_keywords = ["healthy"]
    is_healthy = any(kw in pred_class.lower() for kw in healthy_keywords)
    box_class  = "pred-healthy" if is_healthy else "pred-disease"
    emoji      = "✅" if is_healthy else "⚠️"

    st.markdown(
        f'<div class="pred-box {box_class}">'
        f'<strong>{emoji} Prediction: {pred_class.upper()}</strong><br>'
        f'Confidence: {confidence:.1%}'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.subheader("Class probabilities")
    for cls, prob in sorted(all_probs.items(), key=lambda x: -x[1]):
        st.markdown(f"**{cls}**  —  {prob:.1%}")
        st.progress(float(prob))

    st.divider()
    st.markdown(
        '<span class="info-text">⚕️ For research and educational purposes only. '
        'Consult a licensed veterinarian for medical diagnosis.</span>',
        unsafe_allow_html=True,
    )
