#!/usr/bin/env python3
"""
Main training script for the CNN-Based Thermal Pattern Analysis system.

Usage:
    python train.py                          # default config
    python train.py --config path/to/cfg.yaml
    python train.py --resume checkpoints/best_model.pt
"""

import argparse
import torch

from src.utils.config import load_config, setup_device, set_seed, ensure_dirs
from src.preprocessing import ThermalImageProcessor, ThermalAugmentor
from src.models.anomaly_detector import ThermalPatternPipeline
from src.utils.dataset import create_dataloaders
from src.training.train import Trainer
from src.evaluation.metrics import MetricsCalculator
from src.evaluation.visualize import Visualizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the Thermal Pattern Analysis pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to a checkpoint to resume training from",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── 1. Load config ───────────────────────────────────────────────
    print("Loading configuration...")
    config = load_config(args.config)
    device = setup_device(config)
    set_seed(config.get("seed", 42))
    ensure_dirs(config)
    print(f"  Device : {device}")
    print(f"  Seed   : {config.get('seed', 42)}")

    # ── 2. Preprocessing & augmentation ──────────────────────────────
    print("Setting up preprocessing pipeline...")
    processor = ThermalImageProcessor.from_config(config)
    augmentor = ThermalAugmentor.from_config(config)

    # ── 3. Data loaders ──────────────────────────────────────────────
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        config, processor, augmentor
    )
    print(f"  Train batches : {len(train_loader)}")
    print(f"  Val batches   : {len(val_loader)}")
    print(f"  Test batches  : {len(test_loader)}")

    # ── 4. Build model ───────────────────────────────────────────────
    print("Building model pipeline...")
    model = ThermalPatternPipeline.from_config(config)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params     : {total_params:,}")
    print(f"  Trainable params : {trainable:,}")

    # ── 5. Trainer ───────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    # ── 6. Train ─────────────────────────────────────────────────────
    best_metrics = trainer.train()

    # ── 7. Test evaluation ───────────────────────────────────────────
    print("\nEvaluating on test set...")

    # Collect raw predictions from test set
    model.eval()
    trainer.classifier.eval()
    all_preds, all_labels, all_scores = [], [], []

    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            results = model(sequences)
            logits = trainer.classifier(results["encoding"])
            preds = logits.argmax(dim=1)
            probs = torch.softmax(logits, dim=1)[:, 1]

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(probs.cpu().numpy())

    metrics_calc = MetricsCalculator()
    test_metrics = metrics_calc.compute_all(all_labels, all_preds, all_scores)

    print("\n╔══════════════════════════════════╗")
    print("║      Test Set Results            ║")
    print("╠══════════════════════════════════╣")
    print(metrics_calc.format_metrics(test_metrics))
    print("╚══════════════════════════════════╝")

    # ── 8. Visualizations ────────────────────────────────────────────
    vis_dir = config.paths.visualizations
    vis = Visualizer(output_dir=vis_dir)

    # Confusion Matrix
    try:
        vis.plot_confusion_matrix(
            all_labels, all_preds,
            labels=["Normal", "Abnormal"],
            filename="confusion_matrix.png",
        )
        print(f"  ✓ Confusion matrix saved")
    except Exception as e:
        print(f"  ✗ Confusion matrix failed: {e}")

    # ROC Curve
    try:
        vis.plot_roc_curve(
            all_labels, all_scores,
            filename="roc_curve.png",
        )
        print(f"  ✓ ROC curve saved")
    except Exception as e:
        print(f"  ✗ ROC curve failed: {e}")

    # Preprocessing steps demo
    try:
        sample_img_path = None
        import glob
        img_files = glob.glob("data/raw/Power Transformers/*.jpg")
        if img_files:
            sample_img_path = img_files[0]
            import cv2
            original = cv2.imread(sample_img_path, cv2.IMREAD_GRAYSCALE)
            resized = processor.resize(original)
            denoised = processor.denoise(resized)
            enhanced = processor.enhance_contrast(denoised)
            normalized = processor.normalize_image(enhanced)
            vis.plot_preprocessing_steps(
                original, resized, denoised, enhanced, normalized,
                filename="preprocessing_steps.png",
            )
            print(f"  ✓ Preprocessing steps saved")
    except Exception as e:
        print(f"  ✗ Preprocessing steps failed: {e}")

    # Anomaly score distribution
    try:
        import numpy as np
        labels_arr = np.array(all_labels)
        scores_arr = np.array(all_scores)
        normal_scores = scores_arr[labels_arr == 0].tolist()
        abnormal_scores = scores_arr[labels_arr == 1].tolist()
        if normal_scores and abnormal_scores:
            vis.plot_anomaly_distribution(
                normal_scores, abnormal_scores,
                threshold=0.5,
                filename="anomaly_distribution.png",
            )
            print(f"  ✓ Anomaly distribution saved")
    except Exception as e:
        print(f"  ✗ Anomaly distribution failed: {e}")

    print(f"\nVisualizations saved to {vis_dir}/")
    print("\n✅  Training pipeline complete.")


if __name__ == "__main__":
    main()
