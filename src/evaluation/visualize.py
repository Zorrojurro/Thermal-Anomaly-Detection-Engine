"""
Visualization utilities for the Thermal Pattern Analysis project.

Provides:
    - Preprocessing step visualisation
    - Confusion matrix heatmap
    - ROC curve
    - Attention weights over a sequence
    - Grad-CAM heatmap overlay
    - Training history plots
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, List
from sklearn.metrics import confusion_matrix, roc_curve, auc


class Visualizer:
    """Static visualisation helpers; all methods save to disk."""

    def __init__(self, output_dir: str = "results/visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.style.use("seaborn-v0_8-darkgrid")

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def plot_preprocessing_steps(
        self,
        original: np.ndarray,
        resized: np.ndarray,
        denoised: np.ndarray,
        enhanced: np.ndarray,
        normalized: np.ndarray,
        filename: str = "preprocessing_steps.png",
    ):
        """Visual comparison of each preprocessing stage."""
        stages = [
            ("Original", original),
            ("Resized", resized),
            ("Denoised", denoised),
            ("CLAHE Enhanced", enhanced),
            ("Normalized", normalized),
        ]
        fig, axes = plt.subplots(1, len(stages), figsize=(20, 4))
        for ax, (title, img) in zip(axes, stages):
            ax.imshow(img, cmap="inferno")
            ax.set_title(title, fontsize=12)
            ax.axis("off")

        plt.suptitle("Image Preprocessing Pipeline", fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()

    # ------------------------------------------------------------------
    # Confusion Matrix
    # ------------------------------------------------------------------

    def plot_confusion_matrix(
        self,
        y_true: list,
        y_pred: list,
        labels: list = None,
        filename: str = "confusion_matrix.png",
    ):
        """Plot a confusion matrix heatmap."""
        if labels is None:
            labels = ["Normal", "Abnormal"]

        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
        )
        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("Actual", fontsize=12)
        ax.set_title("Confusion Matrix", fontsize=14)
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150)
        plt.close()

    # ------------------------------------------------------------------
    # ROC Curve
    # ------------------------------------------------------------------

    def plot_roc_curve(
        self,
        y_true: list,
        y_scores: list,
        filename: str = "roc_curve.png",
    ):
        """Plot the receiver operating characteristic curve."""
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color="#4C72B0", lw=2, label=f"AUC = {roc_auc:.4f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.05])
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title("ROC Curve", fontsize=14)
        ax.legend(loc="lower right", fontsize=12)
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150)
        plt.close()

    # ------------------------------------------------------------------
    # Attention Weights
    # ------------------------------------------------------------------

    def plot_attention_weights(
        self,
        images: list,
        weights: np.ndarray,
        filename: str = "attention_weights.png",
    ):
        """
        Visualise attention weights over a sequence of images.

        Args:
            images:  List of (H, W) numpy arrays.
            weights: 1-D array of attention weights, len = len(images).
        """
        n = len(images)
        fig, axes = plt.subplots(2, 1, figsize=(max(n * 2, 12), 6), gridspec_kw={"height_ratios": [3, 1]})

        # Top: images
        ax_img = axes[0]
        concat = np.concatenate(images, axis=1)
        ax_img.imshow(concat, cmap="inferno")
        ax_img.set_title("Sequence Frames", fontsize=12)
        ax_img.axis("off")

        # Bottom: bar chart of weights
        ax_bar = axes[1]
        colors = plt.cm.RdYlGn_r(weights / (weights.max() + 1e-8))
        ax_bar.bar(range(n), weights, color=colors, edgecolor="black", linewidth=0.5)
        ax_bar.set_xlabel("Frame Index", fontsize=11)
        ax_bar.set_ylabel("Attention", fontsize=11)
        ax_bar.set_title("Attention Weights (higher = more important)", fontsize=12)

        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150)
        plt.close()

    # ------------------------------------------------------------------
    # Grad-CAM
    # ------------------------------------------------------------------

    def plot_gradcam(
        self,
        original_image: np.ndarray,
        heatmap: np.ndarray,
        filename: str = "gradcam.png",
    ):
        """
        Overlay a Grad-CAM heatmap on the original image.

        Args:
            original_image: (H, W) normalised float image.
            heatmap:        (H, W) Grad-CAM activation map.
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(original_image, cmap="gray")
        axes[0].set_title("Original", fontsize=12)
        axes[0].axis("off")

        axes[1].imshow(heatmap, cmap="jet")
        axes[1].set_title("Grad-CAM Heatmap", fontsize=12)
        axes[1].axis("off")

        axes[2].imshow(original_image, cmap="gray")
        axes[2].imshow(heatmap, cmap="jet", alpha=0.5)
        axes[2].set_title("Overlay", fontsize=12)
        axes[2].axis("off")

        plt.suptitle("Grad-CAM Visualization", fontsize=14)
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()

    # ------------------------------------------------------------------
    # Training history
    # ------------------------------------------------------------------

    def plot_training_history(
        self,
        train_losses: list,
        val_losses: list,
        train_accs: list = None,
        val_accs: list = None,
        filename: str = "training_history.png",
    ):
        """Plot loss and accuracy curves over epochs."""
        n_plots = 2 if train_accs else 1
        fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 5))
        if n_plots == 1:
            axes = [axes]

        # Loss
        axes[0].plot(train_losses, label="Train", linewidth=2)
        axes[0].plot(val_losses, label="Validation", linewidth=2)
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training & Validation Loss")
        axes[0].legend()

        # Accuracy
        if train_accs:
            axes[1].plot(train_accs, label="Train", linewidth=2)
            axes[1].plot(val_accs, label="Validation", linewidth=2)
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Accuracy")
            axes[1].set_title("Training & Validation Accuracy")
            axes[1].legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150)
        plt.close()

    # ------------------------------------------------------------------
    # Anomaly score distribution
    # ------------------------------------------------------------------

    def plot_anomaly_distribution(
        self,
        normal_scores: list,
        abnormal_scores: list,
        threshold: float = 0.7,
        filename: str = "anomaly_distribution.png",
    ):
        """
        Plot the distribution of anomaly scores for normal vs abnormal
        sequences with the decision threshold.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(normal_scores, bins=30, alpha=0.6, label="Normal", color="#4C72B0")
        ax.hist(abnormal_scores, bins=30, alpha=0.6, label="Abnormal", color="#C44E52")
        ax.axvline(
            x=threshold, color="black", linestyle="--",
            linewidth=2, label=f"Threshold ({threshold})"
        )

        ax.set_xlabel("Similarity Score", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title("Anomaly Score Distribution", fontsize=14)
        ax.legend(fontsize=11)
        plt.tight_layout()
        plt.savefig(self.output_dir / filename, dpi=150)
        plt.close()
