"""
Anomaly Detector — Cosine-similarity-based anomaly scoring.

Compares temporal pattern encodings against a learned "normal"
baseline to flag abnormal heat-distribution sequences.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class AnomalyDetector(nn.Module):
    """
    Anomaly detector using cosine similarity against a
    reference baseline embedding.

    During training, the baseline is updated as a running mean of
    embeddings from *normal* sequences.  At inference, an anomaly
    score is produced:  1 − cosine_similarity.

    Attributes:
        threshold: similarity below this → abnormal.
        baseline:  registered buffer; running-average normal embedding.
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        threshold: float = 0.7,
        momentum: float = 0.99,
    ):
        super().__init__()
        self.threshold = threshold
        self.momentum = momentum

        # The normal-pattern baseline (non-trainable, persisted with model)
        self.register_buffer(
            "baseline", torch.zeros(embedding_dim)
        )
        self.register_buffer(
            "baseline_initialised", torch.tensor(False)
        )

    @classmethod
    def from_config(cls, config) -> "AnomalyDetector":
        """Construct from a Config object."""
        ad = config.model.anomaly_detector
        fe = config.model.feature_extractor
        return cls(
            embedding_dim=fe.embedding_dim,
            threshold=ad.threshold,
        )

    # ------------------------------------------------------------------
    # Baseline management
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update_baseline(self, normal_embeddings: torch.Tensor):
        """
        Update the running-average baseline with new normal embeddings.

        Args:
            normal_embeddings: (N, D) embeddings from normal sequences.
        """
        batch_mean = normal_embeddings.mean(dim=0)
        if not self.baseline_initialised:
            self.baseline.copy_(batch_mean)
            self.baseline_initialised.fill_(True)
        else:
            self.baseline.mul_(self.momentum).add_(
                batch_mean, alpha=1.0 - self.momentum
            )

    def set_baseline(self, baseline: torch.Tensor):
        """Directly set the baseline embedding."""
        self.baseline.copy_(baseline)
        self.baseline_initialised.fill_(True)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def compute_similarity(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Cosine similarity between each embedding and the baseline.

        Args:
            embeddings: (B, D)

        Returns:
            similarities: (B,) in range [-1, 1].
        """
        baseline = self.baseline.unsqueeze(0)  # (1, D)
        return F.cosine_similarity(embeddings, baseline, dim=1)

    def compute_anomaly_score(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Anomaly score = 1 − similarity.
        Higher score → more abnormal.
        """
        return 1.0 - self.compute_similarity(embeddings)

    def forward(self, embeddings: torch.Tensor) -> dict:
        """
        Full anomaly detection inference.

        Args:
            embeddings: (B, D) temporal pattern encodings.

        Returns:
            dict with keys:
                similarity_score: (B,)
                anomaly_score:    (B,)
                is_normal:        (B,) boolean
                confidence:       (B,) distance from threshold
        """
        similarity = self.compute_similarity(embeddings)
        anomaly_score = 1.0 - similarity
        is_normal = similarity >= self.threshold
        confidence = torch.abs(similarity - self.threshold)

        return {
            "similarity_score": similarity,
            "anomaly_score": anomaly_score,
            "is_normal": is_normal,
            "confidence": confidence,
        }


class ThermalPatternPipeline(nn.Module):
    """
    End-to-end pipeline combining all three stages:
        1. ThermalFeatureExtractor  (CNN)
        2. SequenceAnalyzer         (LSTM + Attention)
        3. AnomalyDetector          (Cosine similarity)

    Accepts raw image sequences and returns anomaly predictions.
    """

    def __init__(self, feature_extractor, sequence_analyzer, anomaly_detector):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.sequence_analyzer = sequence_analyzer
        self.anomaly_detector = anomaly_detector

    @classmethod
    def from_config(cls, config) -> "ThermalPatternPipeline":
        """Build the entire pipeline from a Config object."""
        from src.models.feature_extractor import ThermalFeatureExtractor
        from src.models.sequence_analyzer import SequenceAnalyzer

        fe = ThermalFeatureExtractor.from_config(config)
        sa = SequenceAnalyzer.from_config(config)
        ad = AnomalyDetector.from_config(config)
        return cls(fe, sa, ad)

    def forward(self, sequences: torch.Tensor) -> dict:
        """
        End-to-end forward pass.

        Args:
            sequences: (B, T, 1, H, W)

        Returns:
            dict with anomaly_detector outputs + attention_weights.
        """
        # 1. Extract per-frame features  →  (B, T, D)
        features = self.feature_extractor.extract_features_from_sequence(
            sequences
        )

        # 2. Temporal analysis  →  (B, D), (B, T) attention
        encoding, attn_weights = self.sequence_analyzer(features)

        # 3. Anomaly detection
        results = self.anomaly_detector(encoding)
        results["attention_weights"] = attn_weights
        results["encoding"] = encoding

        return results
