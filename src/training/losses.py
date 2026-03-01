"""
Custom loss functions for thermal pattern analysis training.

Implements:
    - ContrastiveLoss  — pushes same-class pairs together, different-class apart
    - TripletLoss      — anchor / positive / negative margin ranking
    - CombinedLoss     — weighted sum of both
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss (Chopra et al., 2005).

    For a pair of embeddings (e1, e2) with label y ∈ {0, 1}:
        y=0 → same class   → loss = ½ · D²
        y=1 → diff class   → loss = ½ · max(0, margin − D)²
    where D = ‖e1 − e2‖₂.
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        embeddings1: torch.Tensor,
        embeddings2: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            embeddings1: (B, D)
            embeddings2: (B, D)
            labels:      (B,)  — 0 if same class, 1 if different

        Returns:
            Scalar loss.
        """
        distance = F.pairwise_distance(embeddings1, embeddings2)
        loss = (
            (1 - labels) * distance.pow(2)
            + labels * F.relu(self.margin - distance).pow(2)
        )
        return 0.5 * loss.mean()


class TripletLoss(nn.Module):
    """
    Triplet margin loss with optional hard-negative mining.

    loss = max(0, d(a, p) − d(a, n) + margin)
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.loss_fn = nn.TripletMarginLoss(margin=margin, p=2)

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            anchor:   (B, D)
            positive: (B, D)  — same class as anchor
            negative: (B, D)  — different class from anchor

        Returns:
            Scalar loss.
        """
        return self.loss_fn(anchor, positive, negative)


class CombinedLoss(nn.Module):
    """
    Weighted combination of Contrastive and Triplet losses,
    with a standard cross-entropy classification head.

    total = α·contrastive + β·triplet + γ·classification
    """

    def __init__(
        self,
        contrastive_weight: float = 0.3,
        triplet_weight: float = 0.3,
        classification_weight: float = 0.4,
        triplet_margin: float = 1.0,
        contrastive_margin: float = 1.0,
    ):
        super().__init__()
        self.contrastive_weight = contrastive_weight
        self.triplet_weight = triplet_weight
        self.classification_weight = classification_weight

        self.contrastive_loss = ContrastiveLoss(margin=contrastive_margin)
        self.triplet_loss = TripletLoss(margin=triplet_margin)
        self.classification_loss = nn.CrossEntropyLoss()

    @classmethod
    def from_config(cls, config) -> "CombinedLoss":
        """Construct from a Config object."""
        loss_cfg = config.training.loss
        return cls(
            contrastive_weight=loss_cfg.contrastive_weight,
            triplet_weight=loss_cfg.triplet_weight,
            classification_weight=1.0 - loss_cfg.contrastive_weight - loss_cfg.triplet_weight,
            triplet_margin=loss_cfg.triplet_margin,
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        logits: torch.Tensor | None = None,
    ) -> dict:
        """
        Compute the combined loss.

        Uses in-batch pair and triplet mining for efficiency.

        Args:
            embeddings: (B, D)
            labels:     (B,) integer class labels
            logits:     (B, num_classes) or None

        Returns:
            dict with total_loss, contrastive, triplet, classification.
        """
        total = torch.tensor(0.0, device=embeddings.device)
        result = {}

        # ------- Contrastive: generate in-batch pairs -------
        B = embeddings.size(0)
        if B >= 2:
            idx = torch.randperm(B, device=embeddings.device)
            e1, e2 = embeddings, embeddings[idx]
            pair_labels = (labels != labels[idx]).float()

            c_loss = self.contrastive_loss(e1, e2, pair_labels)
            total = total + self.contrastive_weight * c_loss
            result["contrastive"] = c_loss.item()

        # ------- Triplet: mine anchor / pos / neg -------
        anchors, positives, negatives = self._mine_triplets(embeddings, labels)
        if anchors is not None:
            t_loss = self.triplet_loss(anchors, positives, negatives)
            total = total + self.triplet_weight * t_loss
            result["triplet"] = t_loss.item()

        # ------- Classification -------
        if logits is not None:
            cls_loss = self.classification_loss(logits, labels)
            total = total + self.classification_weight * cls_loss
            result["classification"] = cls_loss.item()

        result["total_loss"] = total
        return result

    @staticmethod
    def _mine_triplets(
        embeddings: torch.Tensor, labels: torch.Tensor
    ) -> tuple:
        """Simple in-batch triplet mining."""
        unique_labels = labels.unique()
        if len(unique_labels) < 2:
            return None, None, None

        anchors, positives, negatives = [], [], []

        for label in unique_labels:
            mask_pos = labels == label
            mask_neg = labels != label

            pos_idx = mask_pos.nonzero(as_tuple=True)[0]
            neg_idx = mask_neg.nonzero(as_tuple=True)[0]

            if len(pos_idx) < 2 or len(neg_idx) < 1:
                continue

            for i in range(min(len(pos_idx) - 1, 4)):  # limit per class
                anchors.append(embeddings[pos_idx[i]])
                positives.append(embeddings[pos_idx[i + 1]])
                neg_i = neg_idx[torch.randint(len(neg_idx), (1,)).item()]
                negatives.append(embeddings[neg_i])

        if not anchors:
            return None, None, None

        return (
            torch.stack(anchors),
            torch.stack(positives),
            torch.stack(negatives),
        )
