from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from project.models.base import BaseDetector


class DSConv1dBlock(nn.Module):
    """Depthwise separable 1D convolution block."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class RawBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
        )
        self.conv2 = DSConv1dBlock(16, 32, kernel_size=3)
        self.conv3 = DSConv1dBlock(32, 64, kernel_size=3)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.pool(x).squeeze(-1)


class DiffBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
        )
        self.conv2 = DSConv1dBlock(16, 32, kernel_size=3)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return self.pool(x).squeeze(-1)


class EnergyBranch(nn.Module):
    def __init__(self, num_windows: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_windows, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class CNRSenseNet(nn.Module):
    """Backbone network extracted from the notebook implementation."""

    def __init__(self, signal_length: int = 256, energy_window: int = 8, dropout: float = 0.2):
        super().__init__()
        if signal_length % energy_window != 0:
            raise ValueError("signal_length must be divisible by energy_window")

        self.signal_length = int(signal_length)
        self.energy_window = int(energy_window)
        self.num_windows = self.signal_length // self.energy_window

        self.raw_branch = RawBranch()
        self.diff_branch = DiffBranch()
        self.energy_branch = EnergyBranch(self.num_windows)
        self.classifier = nn.Sequential(
            nn.Linear(64 + 32 + 32, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def compute_local_energy(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _ = x.shape
        x_reshaped = x.view(batch_size, self.num_windows, self.energy_window)
        return torch.sum(x_reshaped**2, dim=-1)

    @staticmethod
    def compute_diff(x: torch.Tensor) -> torch.Tensor:
        return x[:, 1:] - x[:, :-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_raw = x.unsqueeze(1)
        h_raw = self.raw_branch(x_raw)

        x_energy = self.compute_local_energy(x)
        h_energy = self.energy_branch(x_energy)

        x_diff = self.compute_diff(x).unsqueeze(1)
        h_diff = self.diff_branch(x_diff)

        features = torch.cat([h_raw, h_energy, h_diff], dim=1)
        return self.classifier(features).squeeze(-1)


@dataclass(slots=True)
class TrainingHistory:
    train_loss: list[float]
    val_loss: list[float]


class CNRSenseNetModel(BaseDetector):
    """Unified detector interface around the CNRSenseNet backbone."""

    def __init__(
        self,
        signal_length: int | None = None,
        energy_window: int = 8,
        dropout: float = 0.2,
        lr: float = 1e-3,
        batch_size: int = 256,
        epochs: int = 10,
        weight_decay: float = 0.0,
        device: str | None = None,
    ):
        super().__init__(
            signal_length=signal_length,
            energy_window=energy_window,
            dropout=dropout,
            lr=lr,
            batch_size=batch_size,
            epochs=epochs,
            weight_decay=weight_decay,
            device=device,
        )
        self.signal_length = signal_length
        self.energy_window = int(energy_window)
        self.dropout = float(dropout)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.weight_decay = float(weight_decay)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model: CNRSenseNet | None = None
        self.history = TrainingHistory(train_loss=[], val_loss=[])

    @staticmethod
    def _unpack_batch(batch):
        if not isinstance(batch, (list, tuple)) or len(batch) < 2:
            raise ValueError("Expected dataset items to be tuples like (x, y, ...).")
        return batch[0], batch[1]

    @staticmethod
    def _prepare_inputs(x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        return x.float().view(x.shape[0], -1)

    @staticmethod
    def _prepare_targets(y: torch.Tensor) -> torch.Tensor:
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32)
        return y.float().view(-1)

    def _infer_signal_length(self, dataset) -> int:
        sample = dataset[0]
        x, _ = self._unpack_batch(sample)
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        return int(x.numel())

    def _ensure_model(self, signal_length: int | None = None) -> CNRSenseNet:
        inferred_length = int(signal_length or self.signal_length or 0)
        if inferred_length <= 0:
            raise ValueError("signal_length could not be inferred.")
        if self.model is None or self.model.signal_length != inferred_length:
            self.model = CNRSenseNet(
                signal_length=inferred_length,
                energy_window=self.energy_window,
                dropout=self.dropout,
            ).to(self.device)
            self.signal_length = inferred_length
        return self.model

    def _make_loader(self, dataset, shuffle: bool) -> DataLoader:
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def _run_epoch(
        self,
        loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> float:
        if self.model is None:
            raise RuntimeError("Model is not initialized.")

        is_train = optimizer is not None
        self.model.train(mode=is_train)
        total_loss = 0.0
        total_count = 0

        context = torch.enable_grad() if is_train else torch.no_grad()
        with context:
            for batch in loader:
                x, y = self._unpack_batch(batch)
                x = self._prepare_inputs(x).to(self.device)
                y = self._prepare_targets(y).to(self.device)

                logits = self.model(x)
                loss = criterion(logits, y)

                if is_train:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

                batch_count = y.shape[0]
                total_loss += loss.item() * batch_count
                total_count += batch_count

        return total_loss / max(total_count, 1)

    def fit(self, train_dataset, val_dataset=None, **kwargs):
        epochs = int(kwargs.get("epochs", self.epochs))
        lr = float(kwargs.get("lr", self.lr))
        batch_size = int(kwargs.get("batch_size", self.batch_size))
        weight_decay = float(kwargs.get("weight_decay", self.weight_decay))
        verbose = bool(kwargs.get("verbose", False))

        self.batch_size = batch_size
        signal_length = self.signal_length or self._infer_signal_length(train_dataset)
        self._ensure_model(signal_length=signal_length)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        train_loader = self._make_loader(train_dataset, shuffle=True)
        val_loader = self._make_loader(val_dataset, shuffle=False) if val_dataset is not None else None

        self.history = TrainingHistory(train_loss=[], val_loss=[])
        for epoch in range(epochs):
            train_loss = self._run_epoch(train_loader, criterion, optimizer=optimizer)
            self.history.train_loss.append(float(train_loss))

            if val_loader is not None:
                val_loss = self._run_epoch(val_loader, criterion, optimizer=None)
                self.history.val_loss.append(float(val_loss))
            elif self.history.val_loss:
                self.history.val_loss.append(self.history.val_loss[-1])

            if verbose:
                if val_loader is None:
                    print(f"[Epoch {epoch + 1}/{epochs}] train_loss={train_loss:.6f}")
                else:
                    print(
                        f"[Epoch {epoch + 1}/{epochs}] train_loss={train_loss:.6f} "
                        f"val_loss={self.history.val_loss[-1]:.6f}"
                    )

        return self

    def predict_scores(self, dataset):
        signal_length = self.signal_length or self._infer_signal_length(dataset)
        self._ensure_model(signal_length=signal_length)
        loader = self._make_loader(dataset, shuffle=False)
        self.model.eval()

        scores: list[np.ndarray] = []
        with torch.no_grad():
            for batch in loader:
                x, _ = self._unpack_batch(batch)
                x = self._prepare_inputs(x).to(self.device)
                logits = self.model(x)
                probs = torch.sigmoid(logits)
                scores.append(probs.detach().cpu().numpy())

        if not scores:
            return np.asarray([], dtype=np.float32)
        return np.concatenate(scores, axis=0).astype(np.float32)

    def predict_proba(self, dataset):
        scores = self.predict_scores(dataset)
        return np.column_stack([1.0 - scores, scores])

    def state_dict(self):
        if self.model is None:
            raise RuntimeError("Model is not initialized.")
        return self.model.state_dict()

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]):
        if self.model is None:
            if self.signal_length is None:
                raise ValueError("Set signal_length before loading weights into CNRSenseNetModel.")
            self._ensure_model(signal_length=self.signal_length)
        self.model.load_state_dict(state_dict)
        return self


__all__ = [
    "CNRSenseNet",
    "CNRSenseNetModel",
    "DSConv1dBlock",
    "DiffBranch",
    "EnergyBranch",
    "RawBranch",
    "TrainingHistory",
]
