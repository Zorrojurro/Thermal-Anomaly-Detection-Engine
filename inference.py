#!/usr/bin/env python3
"""
Inference script for the CNN-Based Thermal Pattern Analysis system.

Loads a trained model checkpoint and runs anomaly detection on a single
thermal image sequence (directory of images).

Usage:
    python inference.py --checkpoint checkpoints/best_model.pt \
                        --sequence data/sequences/normal/seq_001 \
                        --config configs/config.yaml
"""

import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List

from src.utils.config import load_config, setup_device, set_seed
from src.preprocessing import ThermalImageProcessor
from src.models.anomaly_detector import ThermalPatternPipeline
from src.evaluation.visualize import Visualizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference on a thermal image sequence"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pt)",
    )
    parser.add_argument(
        "--sequence",
        type=str,
        required=True,
        help="Path to directory containing a thermal image sequence",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/inference",
        help="Directory to save inference results",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override anomaly threshold (default: from config)",
    )
    return parser.parse_args()


def load_sequence(
    sequence_dir: str,
    processor: ThermalImageProcessor,
    sequence_length: int,
) -> tuple:
    """
    Load and preprocess a directory of thermal images.

    Returns:
        tensor: (1, T, 1, H, W) batch tensor.
        images: list of raw numpy images (for visualisation).
    """
    seq_path = Path(sequence_dir)
    img_files = sorted([
        str(f) for f in seq_path.iterdir()
        if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    ])

    if not img_files:
        raise FileNotFoundError(
            f"No images found in {sequence_dir}"
        )

    raw_images = []
    processed_images = []

    for fp in img_files[:sequence_length]:
        raw = processor.load_image(fp)
        raw_images.append(raw)
        processed = processor.process(fp)
        processed_images.append(processed)

    # Pad if needed
    while len(processed_images) < sequence_length:
        processed_images.append(processed_images[-1].copy())
        raw_images.append(raw_images[-1].copy())

    tensors = [
        torch.from_numpy(img).unsqueeze(0) for img in processed_images
    ]
    sequence_tensor = torch.stack(tensors, dim=0).unsqueeze(0)  # (1, T, 1, H, W)

    return sequence_tensor, raw_images


def main():
    args = parse_args()

    # ── Config ───────────────────────────────────────────────────────
    config = load_config(args.config)
    device = setup_device(config)
    set_seed(config.get("seed", 42))

    threshold = args.threshold or config.model.anomaly_detector.threshold
    seq_len = config.data.sequence_length

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ───────────────────────────────────────────────────
    print("Loading model...")
    model = ThermalPatternPipeline.from_config(config).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("  ✓ Model loaded from", args.checkpoint)

    # ── Load & preprocess sequence ───────────────────────────────────
    print(f"Processing sequence: {args.sequence}")
    processor = ThermalImageProcessor.from_config(config)
    sequence_tensor, raw_images = load_sequence(
        args.sequence, processor, seq_len
    )
    sequence_tensor = sequence_tensor.to(device)

    # ── Inference ────────────────────────────────────────────────────
    print("Running inference...")
    with torch.no_grad():
        results = model(sequence_tensor)

    similarity = results["similarity_score"].item()
    anomaly_score = results["anomaly_score"].item()
    is_normal = results["is_normal"].item()
    confidence = results["confidence"].item()
    attn_weights = results["attention_weights"]

    # ── Display results ──────────────────────────────────────────────
    status = "✅ NORMAL" if is_normal else "🚨 ABNORMAL"

    print(f"\n{'='*50}")
    print(f"  Sequence      : {args.sequence}")
    print(f"  Prediction    : {status}")
    print(f"  Similarity    : {similarity:.4f}")
    print(f"  Anomaly Score : {anomaly_score:.4f}")
    print(f"  Confidence    : {confidence:.4f}")
    print(f"  Threshold     : {threshold}")
    print(f"{'='*50}")

    # ── Visualise attention weights ──────────────────────────────────
    if attn_weights is not None:
        vis = Visualizer(output_dir=str(output_dir))
        weights_np = attn_weights.squeeze(0).cpu().numpy()

        # Resize raw images for display
        display_images = [
            img[:, :min(img.shape[1], 150)] if img.ndim == 2 else img
            for img in raw_images[:seq_len]
        ]
        vis.plot_attention_weights(
            display_images, weights_np,
            filename="inference_attention.png",
        )
        print(f"\n  Attention plot saved → {output_dir}/inference_attention.png")

    # ── Save results as JSON ─────────────────────────────────────────
    import json
    result_json = {
        "sequence": args.sequence,
        "prediction": "normal" if is_normal else "abnormal",
        "similarity_score": round(similarity, 4),
        "anomaly_score": round(anomaly_score, 4),
        "confidence": round(confidence, 4),
        "threshold": threshold,
    }
    json_path = output_dir / "inference_result.json"
    with open(json_path, "w") as f:
        json.dump(result_json, f, indent=2)
    print(f"  Results JSON  saved → {json_path}")

    print("\n✅  Inference complete.")


if __name__ == "__main__":
    main()
