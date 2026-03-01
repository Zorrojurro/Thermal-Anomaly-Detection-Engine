# Developer Guide: Thermal Anomaly Detection Engine

This guide covers the technical architecture, training pipeline, and how to extend the project for new thermal patterns or equipment.

---

## 1. Core Architecture

The system uses a **Hybrid CNN + LSTM** architecture:

1.  **Feature Extractor (`src/models/feature_extractor.py`)**:
    - A modified **ResNet-18** backbone.
    - The first convolutional layer is changed from 3-channel (RGB) to 1-channel (Grayscale).
    - Pretrained weights are adapted by averaging the original RGB channels.
    - Outputs a **256-dimensional embedding** per frame.

2.  **Sequence Analyzer (`src/models/sequence_analyzer.py`)**:
    - A **Bidirectional LSTM** (Bi-LSTM).
    - Processes temporal features from a sequence of images (default length: 5).
    - Includes a **Self-Attention** layer to weigh critical frames (e.g., when a hot spot is most visible).

3.  **Anomaly Detector (`src/models/anomaly_detector.py`)**:
    - Combines the CNN and LSTM into a single pipeline.
    - During training, it uses **Cross-Entropy Loss** for direct classification.

---

## 2. Data Preparation & Synthetic Anomalies

Thermal anomalies are rare. We use `prepare_data_v4.py` to generate a balanced training set:

- **6 Anomaly Types**: Hotspots, dark spots, stripes, asymmetric heat, ring patterns, and noise patches.
- **Injection Layer**: Anomalies are injected *after* preprocessing but before normalization. This ensures the model learns the "post-processed" signature of a defect.
- **Adjusting Difficulty**: Look for `intensity_range` in the `_inject_anomaly` methods to make patterns more subtle or obvious.

```bash
# To generate a new synthetic dataset:
python prepare_data_v4.py --config configs/config.yaml
```

---

## 3. Training & Evaluation

The training loop (`train.py`) includes:
- **Early Stopping**: Halts if validation loss doesn't improve for 25 epochs.
- **Learning Rate Scheduling**: `ReduceLROnPlateau` for fine-tuning towards the end.
- **Visualizations**: Confusion Matrix and ROC Curves are auto-generated in `results/visualizations/`.

```bash
# To start training:
python train.py
```

---

## 4. Web Application Flow

The app is built in two layers to handle heavy ML runtimes:

1.  **Backend (`web_app.py`)**:
    - Loads the PyTorch model into memory once.
    - Computes **Grad-CAM** heatmaps using backward gradients from the last conv layer of the ResNet.
    - Serves Base64-encoded images to the frontend.

2.  **Frontend (`frontend/`)**:
    - **Next.js 15+** with App Router.
    - Uses `@paper-design/shaders-react` for the Warp shader background.
    - Uses `lucide-react` for iconography.
    - Tailored for dark-mode contrast with high-opacity text and `backdrop-blur`.

---

## 5. How to Extend

### Add a New Equipment Category
1. Add images to `data/raw/[CATEGORY_NAME]`.
2. Update the `EquipmentType` mapping in `prepare_data_v4.py`.
3. Add a specific anomaly logic (e.g., cooling fin blockage is a diffuse warm patch).

### Swap the CNN Backbone
1. In `src/models/feature_extractor.py`, you can replace `models.resnet18` with `resnet50`, `EfficientNet`, or `Vision Transformer (ViT)`.
2. Ensure the final `fc` layer outputs 512 dimensions to match the existing LSTM projection, or update the projection head.

### Tune Grad-CAM
- The target layer is currently `layer4[-1].conv2`. For deeper or different networks, find the final convolutional layer where spatial resolution is still preserved.

---

## 6. Environment Troubleshooting

- **Large Model Files**: The checkpoint `best_model.pt` is 140MB. If using GitHub, ensure it's in `.gitignore` or use Git LFS.
- **Memory**: For cloud deployment, the `torch` + `opencv` environment needs at least 1GB - 2GB RAM.
