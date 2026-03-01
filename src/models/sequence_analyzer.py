"""
Sequence Analyzer — Bidirectional LSTM with Self-Attention.

Consumes a sequence of CNN feature embeddings and produces a single
temporal pattern encoding that captures heat-pattern evolution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """
    Additive (Bahdanau-style) self-attention over a sequence of hidden states.

    Learns which timesteps are most informative and produces a
    weighted context vector.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention_fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1, bias=False),
        )

    def forward(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: (B, T, H)

        Returns:
            context:  (B, H)  — weighted sum
            weights:  (B, T)  — attention weights (for visualisation)
        """
        scores = self.attention_fc(hidden_states).squeeze(-1)  # (B, T)
        weights = F.softmax(scores, dim=1)                     # (B, T)
        context = torch.bmm(
            weights.unsqueeze(1), hidden_states
        ).squeeze(1)  # (B, H)
        return context, weights


class SequenceAnalyzer(nn.Module):
    """
    Bidirectional LSTM + Self-Attention for temporal analysis
    of CNN feature sequences.

    Architecture:
        Input features (B, T, D)
          → LayerNorm
          → Bi-LSTM (2 layers, hidden=128)
          → Self-Attention → context (B, 2*hidden)
          → FC projection → (B, output_dim=256)
    """

    def __init__(
        self,
        input_dim: int = 256,
        hidden_size: int = 128,
        num_layers: int = 2,
        output_dim: int = 256,
        bidirectional: bool = True,
        dropout: float = 0.3,
        use_attention: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.num_directions = 2 if bidirectional else 1

        # Normalise input features
        self.input_norm = nn.LayerNorm(input_dim)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        lstm_output_dim = hidden_size * self.num_directions

        # Attention
        if self.use_attention:
            self.attention = SelfAttention(lstm_output_dim)

        # Projection to output_dim
        self.projection = nn.Sequential(
            nn.Linear(lstm_output_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )

    @classmethod
    def from_config(cls, config) -> "SequenceAnalyzer":
        """Construct from a Config object."""
        sa = config.model.sequence_analyzer
        fe = config.model.feature_extractor
        return cls(
            input_dim=fe.embedding_dim,
            hidden_size=sa.hidden_size,
            num_layers=sa.num_layers,
            output_dim=fe.embedding_dim,
            bidirectional=sa.bidirectional,
            dropout=sa.dropout,
            use_attention=sa.attention,
        )

    def forward(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            features: (B, T, D)  — sequence of CNN embeddings.

        Returns:
            encoding:          (B, output_dim) — temporal pattern encoding.
            attention_weights: (B, T) or None  — per-timestep importance.
        """
        # Normalise
        normed = self.input_norm(features)

        # LSTM
        lstm_out, _ = self.lstm(normed)  # (B, T, H*num_directions)

        # Aggregate
        if self.use_attention:
            context, attn_weights = self.attention(lstm_out)
        else:
            # Fallback: use the last hidden state
            context = lstm_out[:, -1, :]
            attn_weights = None

        # Project
        encoding = self.projection(context)
        return encoding, attn_weights
