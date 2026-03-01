#!/usr/bin/env python3
"""
Streamlit Web Interface — Thermal Pattern Analysis

Upload a thermal image → see preprocessing → get anomaly prediction + Grad-CAM.

Usage:
    streamlit run app.py
"""

import streamlit as st
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import io

from src.utils.config import load_config, setup_device
from src.preprocessing.image_processor import ThermalImageProcessor
from src.models.anomaly_detector import ThermalPatternPipeline

# ──────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Thermal Pattern Analyzer",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ──────────────────────────────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    .main { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }

    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        transition: transform 0.3s ease;
    }
    .metric-card:hover { transform: translateY(-2px); }

    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #f093fb, #f5576c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .result-normal {
        background: linear-gradient(135deg, rgba(0,200,83,0.15), rgba(0,200,83,0.05));
        border: 2px solid rgba(0,200,83,0.4);
        border-radius: 16px;
        padding: 30px;
        text-align: center;
    }
    .result-abnormal {
        background: linear-gradient(135deg, rgba(255,61,87,0.15), rgba(255,61,87,0.05));
        border: 2px solid rgba(255,61,87,0.4);
        border-radius: 16px;
        padding: 30px;
        text-align: center;
    }

    .header-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .header-sub {
        color: rgba(255,255,255,0.6);
        font-size: 1.1rem;
        margin-top: 0;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────
# Load model (cached)
# ──────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load the trained model and config."""
    config = load_config("configs/config.yaml")
    device = setup_device(config)

    model = ThermalPatternPipeline.from_config(config)
    model = model.to(device)

    # Load classifier
    import torch.nn as nn
    classifier = nn.Linear(config.model.feature_extractor.embedding_dim, 2).to(device)

    # Load checkpoint
    ckpt_path = Path("checkpoints/best_model.pt")
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        classifier.load_state_dict(ckpt["classifier_state_dict"])
        epoch = ckpt.get("epoch", "?")
        st.sidebar.success(f"✓ Model loaded (epoch {epoch})")
    else:
        st.sidebar.warning("⚠ No checkpoint found. Run train.py first.")
        return None, None, None, None

    model.eval()
    classifier.eval()

    processor = ThermalImageProcessor.from_config(config)
    return model, classifier, processor, device


def compute_gradcam(model, input_tensor, device):
    """Compute Grad-CAM heatmap from the feature extractor."""
    model.eval()

    # Get the last conv layer of ResNet
    target_layer = model.feature_extractor.backbone.layer4[-1].conv2

    activations = {}
    gradients = {}

    def forward_hook(module, input, output):
        activations["value"] = output.detach()

    def backward_hook(module, grad_input, grad_output):
        gradients["value"] = grad_output[0].detach()

    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    try:
        # Forward single image
        img = input_tensor.unsqueeze(0).to(device)  # [1, 1, H, W]
        features = model.feature_extractor(img)      # [1, embedding_dim]

        # Backward on the max feature
        model.zero_grad()
        features.max().backward()

        # Compute Grad-CAM
        acts = activations["value"].squeeze(0)   # [C, h, w]
        grads = gradients["value"].squeeze(0)    # [C, h, w]
        weights = grads.mean(dim=(1, 2))         # [C]
        cam = (weights[:, None, None] * acts).sum(dim=0)  # [h, w]
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = cam.cpu().numpy()

        # Resize to input size
        cam_resized = cv2.resize(cam, (224, 224))
        return cam_resized
    finally:
        fh.remove()
        bh.remove()


def preprocess_uploaded_image(uploaded_file, processor):
    """Preprocess an uploaded image file."""
    # Read the file
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return None, None, None, None, None

    # Convert to grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    original = gray.copy()

    # Preprocess steps
    resized = processor.resize(gray)
    denoised = processor.denoise(resized)
    enhanced = processor.enhance_contrast(denoised)

    # Normalize to [0, 1] float32
    normalized = enhanced.astype(np.float32) / 255.0

    return original, resized, denoised, enhanced, normalized


# ──────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────

def main():
    # Header
    st.markdown('<h1 class="header-title">🔥 Thermal Pattern Analyzer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="header-sub">CNN-Based Thermal Pattern Analysis of Power Transformers</p>', unsafe_allow_html=True)
    st.markdown("---")

    # Load model
    result = load_model()
    if result[0] is None:
        st.error("Model not loaded. Please run `python train.py` first.")
        return
    model, classifier, processor, device = result

    # Sidebar
    st.sidebar.markdown("### ⚙️ Settings")
    threshold = st.sidebar.slider("Anomaly Threshold", 0.0, 1.0, 0.5, 0.05)
    show_gradcam = st.sidebar.checkbox("Show Grad-CAM Heatmap", value=True)
    show_preprocessing = st.sidebar.checkbox("Show Preprocessing Steps", value=True)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Model Info")
    st.sidebar.markdown("- **Backbone**: ResNet-18")
    st.sidebar.markdown("- **Sequence**: Bi-LSTM + Attention")
    st.sidebar.markdown("- **Input**: 224×224 grayscale")

    # Upload
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📤 Upload Thermal Image")
        uploaded_file = st.file_uploader(
            "Drop a thermal image here",
            type=["jpg", "jpeg", "png", "bmp", "tiff"],
            help="Upload an infrared thermal image of power equipment"
        )

        # Or use a sample
        use_sample = st.checkbox("Use sample image from dataset")
        if use_sample:
            import glob
            samples = glob.glob("data/raw/Power Transformers/*.jpg")[:10]
            if samples:
                selected = st.selectbox("Select sample", [Path(s).name for s in samples])
                selected_path = next(s for s in samples if Path(s).name == selected)
                uploaded_file = open(selected_path, "rb")

    if uploaded_file is not None:
        # Preprocess
        original, resized, denoised, enhanced, normalized = preprocess_uploaded_image(
            uploaded_file, processor
        )

        if original is None:
            st.error("Could not read the image. Please try another file.")
            return

        with col1:
            st.image(original, caption="Uploaded Image", use_container_width=True, clamp=True)

        # Run inference
        with torch.no_grad():
            # Create a sequence of the same image (for single-image inference)
            img_tensor = torch.from_numpy(normalized).unsqueeze(0)  # [1, H, W]
            seq_len = 5
            sequence = img_tensor.unsqueeze(0).repeat(1, seq_len, 1, 1)  # [1, seq_len, H, W]
            sequence = sequence.unsqueeze(2)  # [1, seq_len, 1, H, W]
            sequence = sequence.to(device)

            results = model(sequence)
            logits = classifier(results["encoding"])
            probs = torch.softmax(logits, dim=1)
            anomaly_prob = probs[0, 1].item()
            prediction = "ABNORMAL" if anomaly_prob > threshold else "NORMAL"
            confidence = max(anomaly_prob, 1 - anomaly_prob) * 100

        with col2:
            st.markdown("### 🎯 Analysis Result")

            if prediction == "NORMAL":
                st.markdown(f"""
                <div class="result-normal">
                    <h2 style="color: #00c853; margin:0;">✅ NORMAL</h2>
                    <p style="color: rgba(255,255,255,0.8); font-size: 1.2rem;">
                        No thermal anomalies detected
                    </p>
                    <p style="color: rgba(255,255,255,0.6);">
                        Confidence: {confidence:.1f}%
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-abnormal">
                    <h2 style="color: #ff3d57; margin:0;">⚠️ ABNORMAL</h2>
                    <p style="color: rgba(255,255,255,0.8); font-size: 1.2rem;">
                        Thermal anomaly detected!
                    </p>
                    <p style="color: rgba(255,255,255,0.6);">
                        Confidence: {confidence:.1f}%
                    </p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("")

            # Metrics cards
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Anomaly Score", f"{anomaly_prob:.3f}")
            with m2:
                st.metric("Confidence", f"{confidence:.1f}%")
            with m3:
                st.metric("Threshold", f"{threshold:.2f}")

        # Grad-CAM
        if show_gradcam:
            st.markdown("---")
            st.markdown("### 🔬 Grad-CAM Attention Map")
            st.markdown("*Highlights the regions the model focuses on for its decision*")

            try:
                img_for_cam = torch.from_numpy(normalized).unsqueeze(0)  # [1, H, W]
                cam = compute_gradcam(model, img_for_cam, device)

                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                fig.patch.set_facecolor('#1a1a2e')

                # Original
                axes[0].imshow(enhanced, cmap='inferno')
                axes[0].set_title("Input Image", color='white', fontsize=14)
                axes[0].axis('off')

                # Grad-CAM heatmap
                axes[1].imshow(cam, cmap='jet')
                axes[1].set_title("Grad-CAM Heatmap", color='white', fontsize=14)
                axes[1].axis('off')

                # Overlay
                heatmap_colored = cv2.applyColorMap(
                    (cam * 255).astype(np.uint8), cv2.COLORMAP_JET
                )
                # Convert grayscale to BGR for overlay
                base_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
                overlay = cv2.addWeighted(base_bgr, 0.6, heatmap_colored, 0.4, 0)
                overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

                axes[2].imshow(overlay_rgb)
                axes[2].set_title("Overlay", color='white', fontsize=14)
                axes[2].axis('off')

                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            except Exception as e:
                st.warning(f"Grad-CAM generation failed: {e}")

        # Preprocessing steps
        if show_preprocessing:
            st.markdown("---")
            st.markdown("### 📋 Preprocessing Pipeline")

            p1, p2, p3, p4 = st.columns(4)
            with p1:
                st.image(resized, caption="1. Resized (224×224)", use_container_width=True, clamp=True)
            with p2:
                st.image(denoised, caption="2. Denoised", use_container_width=True, clamp=True)
            with p3:
                st.image(enhanced, caption="3. CLAHE Enhanced", use_container_width=True, clamp=True)
            with p4:
                st.image(normalized, caption="4. Normalized [0,1]", use_container_width=True, clamp=True)

    else:
        # Landing state
        st.markdown("""
        <div style="text-align: center; padding: 80px 0; color: rgba(255,255,255,0.5);">
            <h2>👆 Upload a thermal image to get started</h2>
            <p>Supports JPG, PNG, BMP, TIFF formats</p>
            <p style="margin-top: 20px; font-size: 0.9rem;">
                The system analyses infrared thermal images of power transformers
                to detect abnormal heat distribution patterns that may indicate
                equipment faults or maintenance needs.
            </p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
