"""
CNN Feature Extractor — Modified ResNet-18 for grayscale thermal images.

Takes single-channel (grayscale) 224×224 images and outputs 256-dim
feature embeddings suitable for downstream sequence analysis.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class ThermalFeatureExtractor(nn.Module):
    """
    Modified ResNet-18 that accepts 1-channel grayscale input
    and produces a compact feature embedding.

    Architecture:
        Input (1, 224, 224)
          → Conv1 (1→64, 7×7)  (replaces the default 3→64)
          → ResNet-18 layers 1-4
          → AdaptiveAvgPool → (512,)
          → FC(512→256) + BatchNorm + ReLU + Dropout
          → 256-dim embedding
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        pretrained: bool = True,
        in_channels: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Load pretrained ResNet-18
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)

        # Replace the first conv layer: 3-channel → 1-channel
        original_conv = resnet.conv1
        self.conv1 = nn.Conv2d(
            in_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

        # If pretrained, initialise from the mean of the RGB weights
        if pretrained:
            with torch.no_grad():
                self.conv1.weight = nn.Parameter(
                    original_conv.weight.mean(dim=1, keepdim=True)
                )

        # Keep the rest of ResNet-18 up to avgpool
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool

        # Projection head: 512 → embedding_dim
        self.projection = nn.Sequential(
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

    @classmethod
    def from_config(cls, config) -> "ThermalFeatureExtractor":
        """Construct from a Config object."""
        fe = config.model.feature_extractor
        return cls(
            embedding_dim=fe.embedding_dim,
            pretrained=fe.pretrained,
            in_channels=fe.in_channels,
            dropout=config.model.sequence_analyzer.dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor of shape (B, 1, 224, 224).

        Returns:
            Embedding tensor of shape (B, embedding_dim).
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # (B, 512)
        x = self.projection(x)   # (B, embedding_dim)
        return x

    def extract_features_from_sequence(
        self, sequence: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract features for a batch of sequences.

        Args:
            sequence: (B, T, 1, H, W) — batch of image sequences.

        Returns:
            (B, T, embedding_dim)
        """
        B, T, C, H, W = sequence.shape
        # Flatten batch and time → (B*T, C, H, W)
        x = sequence.view(B * T, C, H, W)
        features = self.forward(x)  # (B*T, D)
        return features.view(B, T, self.embedding_dim)
