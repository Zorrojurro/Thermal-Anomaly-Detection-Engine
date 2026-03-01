#!/usr/bin/env python3
"""
Flask Web Application — Thermal Pattern Analysis Interface.

Usage:
    python web_app.py
    → Open http://localhost:5000
"""

import os
import io
import base64
import torch
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.nn as nn
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

from src.utils.config import load_config, setup_device
from src.preprocessing.image_processor import ThermalImageProcessor
from src.models.anomaly_detector import ThermalPatternPipeline

app = Flask(__name__)
CORS(app)

# ── Global model state ───────────────────────────────────────────────
MODEL = None
CLASSIFIER = None
PROCESSOR = None
DEVICE = None


def load_model():
    """Load model, classifier, and processor at startup."""
    global MODEL, CLASSIFIER, PROCESSOR, DEVICE

    config = load_config("configs/config.yaml")
    DEVICE = setup_device(config)

    MODEL = ThermalPatternPipeline.from_config(config).to(DEVICE)
    CLASSIFIER = nn.Linear(config.model.feature_extractor.embedding_dim, 2).to(DEVICE)

    ckpt_path = Path("checkpoints/best_model.pt")
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        MODEL.load_state_dict(ckpt["model_state_dict"])
        CLASSIFIER.load_state_dict(ckpt["classifier_state_dict"])
        print(f"  ✓ Model loaded from {ckpt_path}")
    else:
        print(f"  ✗ No checkpoint at {ckpt_path}")

    MODEL.eval()
    CLASSIFIER.eval()
    PROCESSOR = ThermalImageProcessor.from_config(config)


def img_to_base64(img, cmap=None):
    """Convert numpy image to base64-encoded PNG for HTML display."""
    # Normalize to 0-255 uint8 if needed
    if img.dtype == np.float32 or img.dtype == np.float64:
        img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8) if img.max() <= 1.0 else np.clip(img, 0, 255).astype(np.uint8)
    else:
        img_u8 = img.astype(np.uint8)

    if cmap == 'jet':
        # Grad-CAM heatmap
        colored = cv2.applyColorMap(img_u8, cv2.COLORMAP_JET)
    elif len(img_u8.shape) == 2:
        # Grayscale → apply thermal inferno colormap
        colored = cv2.applyColorMap(img_u8, cv2.COLORMAP_INFERNO)
    else:
        # Already colored (like overlay)
        colored = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR) if img_u8.shape[2] == 3 else img_u8

    _, buf = cv2.imencode('.png', colored)
    return base64.b64encode(buf.tobytes()).decode('utf-8')


def compute_gradcam(input_tensor):
    """Compute Grad-CAM heatmap."""
    target_layer = MODEL.feature_extractor.layer4[-1].conv2
    activations, gradients = {}, {}

    def fwd_hook(m, i, o): activations["v"] = o.detach()
    def bwd_hook(m, gi, go): gradients["v"] = go[0].detach()

    fh = target_layer.register_forward_hook(fwd_hook)
    bh = target_layer.register_full_backward_hook(bwd_hook)

    try:
        img = input_tensor.unsqueeze(0).to(DEVICE)
        features = MODEL.feature_extractor(img)
        MODEL.zero_grad()
        features.max().backward()

        acts = activations["v"].squeeze(0)
        grads = gradients["v"].squeeze(0)
        weights = grads.mean(dim=(1, 2))
        cam = torch.relu((weights[:, None, None] * acts).sum(0))
        cam = cam / (cam.max() + 1e-8)
        cam = cam.cpu().numpy()
        return cv2.resize(cam, (224, 224))
    finally:
        fh.remove()
        bh.remove()


# ── Routes ────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Cannot read image"}), 400

    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    original = gray.copy()

    # Preprocessing steps
    resized = PROCESSOR.resize(gray)
    denoised = PROCESSOR.denoise(resized)
    enhanced = PROCESSOR.enhance_contrast(denoised)
    normalized = enhanced.astype(np.float32) / 255.0

    # Inference
    with torch.no_grad():
        img_tensor = torch.from_numpy(normalized).unsqueeze(0)  # [1, H, W]
        sequence = img_tensor.unsqueeze(0).repeat(1, 5, 1, 1).unsqueeze(2)  # [1, 5, 1, H, W]
        sequence = sequence.to(DEVICE)

        results = MODEL(sequence)
        logits = CLASSIFIER(results["encoding"])
        probs = torch.softmax(logits, dim=1)
        anomaly_score = probs[0, 1].item()
        prediction = "ABNORMAL" if anomaly_score > 0.5 else "NORMAL"
        confidence = max(anomaly_score, 1 - anomaly_score) * 100

    # Grad-CAM
    gradcam = compute_gradcam(img_tensor)

    # Create overlay
    heatmap_colored = cv2.applyColorMap((gradcam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    base_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(base_bgr, 0.6, heatmap_colored, 0.4, 0)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    # Encode images
    response = {
        "prediction": prediction,
        "anomaly_score": round(anomaly_score, 4),
        "confidence": round(confidence, 1),
        "images": {
            "original": img_to_base64(original),
            "resized": img_to_base64(resized),
            "denoised": img_to_base64(denoised),
            "enhanced": img_to_base64(enhanced),
            "normalized": img_to_base64(normalized),
            "gradcam": img_to_base64(gradcam, cmap='jet'),
            "overlay": img_to_base64(overlay_rgb),
        }
    }

    return jsonify(response)


@app.route("/sample_images")
def sample_images():
    """Return list of sample images from the dataset."""
    import glob
    samples = glob.glob("data/raw/Power Transformers/*.jpg")[:12]
    names = [Path(s).name for s in samples]
    return jsonify(names)


@app.route("/analyze_sample/<filename>")
def analyze_sample(filename):
    """Analyze a sample image from the dataset."""
    path = Path("data/raw/Power Transformers") / filename
    if not path.exists():
        return jsonify({"error": "Sample not found"}), 404

    with open(path, "rb") as f:
        from werkzeug.datastructures import FileStorage
        file = FileStorage(f, filename=filename)
        # Read the file manually
        file_bytes = np.frombuffer(f.read(), np.uint8)

    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Cannot read image"}), 400

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    original = gray.copy()
    resized = PROCESSOR.resize(gray)
    denoised = PROCESSOR.denoise(resized)
    enhanced = PROCESSOR.enhance_contrast(denoised)
    normalized = enhanced.astype(np.float32) / 255.0

    with torch.no_grad():
        img_tensor = torch.from_numpy(normalized).unsqueeze(0)
        sequence = img_tensor.unsqueeze(0).repeat(1, 5, 1, 1).unsqueeze(2)
        sequence = sequence.to(DEVICE)
        results = MODEL(sequence)
        logits = CLASSIFIER(results["encoding"])
        probs = torch.softmax(logits, dim=1)
        anomaly_score = probs[0, 1].item()
        prediction = "ABNORMAL" if anomaly_score > 0.5 else "NORMAL"
        confidence = max(anomaly_score, 1 - anomaly_score) * 100

    gradcam = compute_gradcam(img_tensor)
    heatmap_colored = cv2.applyColorMap((gradcam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    base_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(base_bgr, 0.6, heatmap_colored, 0.4, 0)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    return jsonify({
        "prediction": prediction,
        "anomaly_score": round(anomaly_score, 4),
        "confidence": round(confidence, 1),
        "images": {
            "original": img_to_base64(original),
            "resized": img_to_base64(resized),
            "denoised": img_to_base64(denoised),
            "enhanced": img_to_base64(enhanced),
            "normalized": img_to_base64(normalized),
            "gradcam": img_to_base64(gradcam, cmap='jet'),
            "overlay": img_to_base64(overlay_rgb),
        }
    })


if __name__ == "__main__":
    print("Loading model...")
    load_model()
    print("Starting server at http://localhost:5000")
    app.run(debug=False, host="0.0.0.0", port=5000)
