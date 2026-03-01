# CNN-Based Thermal Pattern Analysis of Power Transformers

> A computer vision system that analyses infrared thermal images of power transformers to detect abnormal heat distribution patterns using deep learning.

---

## Overview

This project applies **image processing** and **computer vision** techniques to infrared thermal images of industrial power transformers for automated anomaly detection.

### Pipeline Architecture

```
Thermal Images → Preprocessing → CNN Feature Extraction → Sequence Analysis → Anomaly Detection
```

| Stage | Component | Technology |
|-------|-----------|------------|
| 1. Preprocessing | Resize, denoise, CLAHE, normalize | OpenCV |
| 2. Feature Extraction | Modified ResNet-18 (grayscale) | PyTorch |
| 3. Sequence Analysis | Bi-LSTM + Self-Attention | PyTorch |
| 4. Anomaly Detection | Cosine similarity scoring | PyTorch |

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset

Organise thermal images into the following structure:

```
data/sequences/
├── normal/
│   ├── seq_001/
│   │   ├── img_001.png
│   │   ├── img_002.png
│   │   └── ...
│   └── seq_002/
│       └── ...
└── abnormal/
    ├── seq_010/
    │   └── ...
    └── ...
```

### 3. Train

```bash
python train.py --config configs/config.yaml
```

### 4. Inference

```bash
python inference.py \
    --checkpoint checkpoints/best_model.pt \
    --sequence data/sequences/normal/seq_001
```

### 5. Start Web Interface (Premium UI)

The project includes a premium Next.js frontend with a Flask backend.

**Terminal 1 (Backend):**
```bash
python web_app.py
```

**Terminal 2 (Frontend):**
```bash
cd frontend
npm install
npm run dev
```
→ Open **http://localhost:3000**

---

## Project Structure

```
thermal-pattern-analysis/
├── configs/
│   └── config.yaml              # All hyperparameters
├── src/
│   ├── preprocessing/
│   │   ├── image_processor.py   # Resize, denoise, CLAHE, normalize
│   │   └── augmentation.py      # Rotation, flip, brightness shift
│   ├── models/
│   │   ├── feature_extractor.py # Modified ResNet-18
│   │   ├── sequence_analyzer.py # Bi-LSTM + Self-Attention
│   │   └── anomaly_detector.py  # Cosine similarity + pipeline
│   ├── training/
│   │   ├── train.py             # Training loop + early stopping
│   │   └── losses.py            # Contrastive + Triplet + CE loss
│   ├── evaluation/
│   │   ├── metrics.py           # Accuracy, Precision, Recall, F1, AUC-ROC
│   │   └── visualize.py         # Grad-CAM, attention, confusion matrix
│   └── utils/
│       ├── config.py            # YAML config management
│       └── dataset.py           # PyTorch Dataset for sequences
├── train.py                     # Main training entry point
├── inference.py                 # Inference entry point
├── requirements.txt
└── README.md
```

---

## Configuration

All hyperparameters are in `configs/config.yaml`:

| Category | Key Parameters |
|----------|---------------|
| **Data** | `image_size: 224×224`, `sequence_length: 20` |
| **Model** | ResNet-18, Bi-LSTM (hidden=128, 2 layers), embedding=256 |
| **Training** | AdamW (lr=1e-4), CosineAnnealing, 50 epochs |
| **Detection** | Cosine similarity, threshold=0.7 |

---

## Evaluation Metrics

| Metric | Purpose |
|--------|---------|
| Accuracy | Overall classification correctness |
| Precision | Abnormal detection reliability |
| Recall | Abnormal pattern capture rate |
| F1-Score | Balanced performance measure |
| AUC-ROC | Threshold-independent performance |

---

## Technologies

- **Python 3.9+** — Implementation language
- **PyTorch** — Deep learning framework
- **OpenCV** — Image preprocessing
- **scikit-learn** — Evaluation metrics
- **Albumentations** — Data augmentation
- **Matplotlib / Seaborn** — Visualization
- **TensorBoard** — Training monitoring

---

## Dataset

**Primary**: [SciDB — Infrared Thermal Image Dataset of High Voltage Electrical Power Equipment](https://www.scidb.cn/en/detail?dataSetId=e416c488169f484485ad7575dcfc43ce)
- 895 IR images, 640×480 pixels
- Power Transformer subset: 178 images
- Captured at different times and load conditions

---

## Deployment

The project is structured for a hybrid cloud deployment:
1. **Frontend (Next.js)**: Optimized for [Vercel](https://vercel.com).
2. **Backend (Flask + ML)**: Optimized for [Render](https://render.com) or [Railway](https://railway.app).

**[See Step-by-Step Deployment Guide (DEPLOYMENT.md)](./DEPLOYMENT.md)**

---

## Repository

[GitHub: Thermal-Anomaly-Detection-Engine](https://github.com/Zorrojurro/Thermal-Anomaly-Detection-Engine.git)

---

## License

This project is created for academic purposes (IPCV coursework).
