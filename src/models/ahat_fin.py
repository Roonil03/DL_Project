"""A small hierarchical attention model for financial signal prediction.

The model combines two views of a multivariate lookback window: attention over
time steps and attention over input features. A learned gate adapts the mix of
those views for each example before producing bounded per-asset signals.
"""

import torch
import torch.nn as nn

from .dlinear import TradingSignalModel


class AHATFinSignalModel(TradingSignalModel):
    """Adaptive Hierarchical Attention model for ``[batch, time, features]`` input.

    This compact AHAT-Fin-inspired variant uses a temporal Transformer branch,
    a feature Transformer branch, learned attention pooling in each branch, and
    a per-example gate to fuse the resulting representations.
    """

    def __init__(
        self,
        lookback: int,
        num_features: int,
        num_assets: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        if lookback <= 0 or num_features <= 0 or num_assets <= 0:
            raise ValueError("lookback, num_features, and num_assets must be positive")
        if d_model <= 0 or n_heads <= 0 or d_model % n_heads != 0:
            raise ValueError("d_model must be positive and divisible by n_heads")
        if n_layers <= 0:
            raise ValueError("n_layers must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.lookback = lookback
        self.num_features = num_features

        self.temporal_input = nn.Linear(num_features, d_model)
        self.temporal_position = nn.Parameter(torch.zeros(1, lookback, d_model))
        self.temporal_encoder = self._encoder(d_model, n_heads, n_layers, dropout)
        self.temporal_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.temporal_pool = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        # Each feature is a token whose values span the lookback window.
        self.feature_input = nn.Linear(lookback, d_model)
        self.feature_encoder = self._encoder(d_model, n_heads, n_layers, dropout)
        self.feature_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.feature_pool = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        self.gate = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.Sigmoid())
        self.output = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, num_assets))

    @staticmethod
    def _encoder(d_model, n_heads, n_layers, dropout):
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        return nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("input must have shape [batch, lookback, num_features]")
        if x.shape[1] != self.lookback or x.shape[2] != self.num_features:
            raise ValueError(
                f"expected [batch, {self.lookback}, {self.num_features}], "
                f"got {list(x.shape)}"
            )

        temporal = self.temporal_input(x) + self.temporal_position
        temporal = self.temporal_encoder(temporal)
        temporal, _ = self.temporal_pool(
            self.temporal_query.expand(x.shape[0], -1, -1), temporal, temporal,
            need_weights=False,
        )
        temporal = temporal.squeeze(1)

        features = self.feature_input(x.transpose(1, 2))
        features = self.feature_encoder(features)
        features, _ = self.feature_pool(
            self.feature_query.expand(x.shape[0], -1, -1), features, features,
            need_weights=False,
        )
        features = features.squeeze(1)

        gate = self.gate(torch.cat((temporal, features), dim=-1))
        fused = gate * temporal + (1.0 - gate) * features
        return torch.tanh(self.output(fused))
