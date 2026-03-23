from __future__ import annotations

import torch.nn as nn

from project.models.torch_binary import TorchBinaryClassifier


class MLPBackbone(nn.Module):
    def __init__(self, signal_length: int, hidden_dims=(256, 128, 64), dropout: float = 0.2):
        super().__init__()
        layers = []
        in_features = signal_length
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_features, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )
            in_features = hidden_dim
        layers.append(nn.Linear(in_features, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)


class MLPModel(TorchBinaryClassifier):
    """Simple multilayer perceptron baseline for signal detection."""

    def __init__(self, hidden_dims=(256, 128, 64), dropout: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dims = tuple(int(dim) for dim in hidden_dims)
        self.dropout = float(dropout)

    def build_network(self, signal_length: int) -> nn.Module:
        return MLPBackbone(
            signal_length=signal_length,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
        )
