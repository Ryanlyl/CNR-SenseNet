from __future__ import annotations

import torch
import torch.nn as nn

from project.models.torch_binary import TorchBinaryClassifier


class LSTMBackbone(nn.Module):
    def __init__(
        self,
        signal_length: int,
        hidden_size: int = 48,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        if signal_length % 2 != 0:
            raise ValueError("signal_length must be even for IQ-pair reshaping.")

        self.sequence_length = signal_length // 2
        self.feature_dim = 2
        self.hidden_size = int(hidden_size)
        self.bidirectional = bool(bidirectional)
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.encoder = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
            bidirectional=self.bidirectional,
        )
        output_dim = self.hidden_size * (2 if self.bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Linear(output_dim * 2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = x.view(batch_size, self.sequence_length, self.feature_dim)
        encoded, _ = self.encoder(x)
        mean_feature = encoded.mean(dim=1)
        last_feature = encoded[:, -1, :]
        features = torch.cat([mean_feature, last_feature], dim=1)
        return self.classifier(features).squeeze(-1)


class LSTMModel(TorchBinaryClassifier):
    """Bidirectional LSTM baseline over IQ sample pairs."""

    def __init__(
        self,
        hidden_size: int = 48,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = int(hidden_size)
        self.num_layers = int(num_layers)
        self.bidirectional = bool(bidirectional)
        self.dropout = float(dropout)

    def build_network(self, signal_length: int) -> nn.Module:
        return LSTMBackbone(
            signal_length=signal_length,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            dropout=self.dropout,
        )
