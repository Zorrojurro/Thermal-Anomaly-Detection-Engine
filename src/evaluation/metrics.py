"""
Evaluation metrics for anomaly detection performance.

Computes accuracy, precision, recall, F1-score, and AUC-ROC.
"""

import numpy as np
from typing import Optional, List
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


class MetricsCalculator:
    """Calculates all evaluation metrics for binary classification."""

    @staticmethod
    def compute_all(
        y_true: list | np.ndarray,
        y_pred: list | np.ndarray,
        y_scores: Optional[list | np.ndarray] = None,
    ) -> dict:
        """
        Compute all metrics.

        Args:
            y_true:   Ground-truth labels (0 = normal, 1 = abnormal).
            y_pred:   Predicted labels.
            y_scores: Predicted probabilities for the positive class
                      (required for AUC-ROC).

        Returns:
            Dictionary of metric_name → value.
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
        }

        if y_scores is not None and len(np.unique(y_true)) > 1:
            metrics["auc_roc"] = roc_auc_score(y_true, y_scores)

        return metrics

    @staticmethod
    def get_confusion_matrix(
        y_true: list | np.ndarray,
        y_pred: list | np.ndarray,
    ) -> np.ndarray:
        """Return confusion matrix as a 2×2 numpy array."""
        return confusion_matrix(y_true, y_pred)

    @staticmethod
    def get_classification_report(
        y_true: list | np.ndarray,
        y_pred: list | np.ndarray,
        target_names: list = None,
    ) -> str:
        """Return a formatted classification report."""
        if target_names is None:
            target_names = ["Normal", "Abnormal"]
        return classification_report(
            y_true, y_pred, target_names=target_names, zero_division=0
        )

    @staticmethod
    def format_metrics(metrics: dict) -> str:
        """Pretty-print metrics to a formatted string."""
        lines = []
        for name, value in metrics.items():
            if isinstance(value, float):
                lines.append(f"  {name:>15s}: {value:.4f}")
            else:
                lines.append(f"  {name:>15s}: {value}")
        return "\n".join(lines)
