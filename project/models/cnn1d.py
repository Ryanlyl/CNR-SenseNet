from __future__ import annotations

import torch
import torch.nn as nn

from project.models.torch_binary import TorchBinaryClassifier


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CNN1DBackbone(nn.Module):
    def __init__(self, dropout: float = 0.2):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 32, kernel_size=5),
            ConvBlock(32, 64, kernel_size=5),
            ConvBlock(64, 128, kernel_size=3),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.features(x)
        return self.classifier(x).squeeze(-1)


class CNN1DModel(TorchBinaryClassifier):
    """Shallow 1D CNN baseline on raw interleaved IQ samples."""

    def __init__(self, dropout: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        self.dropout = float(dropout)

    def build_network(self, signal_length: int) -> nn.Module:
        del signal_length
        return CNN1DBackbone(dropout=self.dropout)
