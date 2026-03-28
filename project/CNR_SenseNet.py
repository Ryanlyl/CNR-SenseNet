from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from project.models.base import BaseDetector, CalibratedThresholdResult, fit_binary_threshold


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
            nn.Conv1d(2, 16, kernel_size=5, padding=2, bias=False),
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
            nn.Conv1d(2, 16, kernel_size=5, padding=2, bias=False),
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
        if signal_length % 2 != 0:
            raise ValueError("signal_length must be even for interleaved IQ samples")

        self.signal_length = int(signal_length)
        self.num_iq_samples = self.signal_length // 2
        self.energy_window = int(energy_window)
        if self.num_iq_samples % self.energy_window != 0:
            raise ValueError("number of IQ samples must be divisible by energy_window")
        self.num_windows = self.num_iq_samples // self.energy_window

        self.raw_branch = RawBranch()
        self.diff_branch = DiffBranch()
        self.energy_branch = EnergyBranch(self.num_windows)
        self.classifier = nn.Sequential(
            nn.Linear(64 + 32 + 32, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def reshape_iq(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, signal_length = x.shape
        if signal_length != self.signal_length:
            raise ValueError(
                f"Expected flattened input length {self.signal_length}, got {signal_length}."
            )
        return x.view(batch_size, self.num_iq_samples, 2).transpose(1, 2).contiguous()

    def compute_local_energy(self, x_iq: torch.Tensor) -> torch.Tensor:
        batch_size, _, num_iq_samples = x_iq.shape
        if num_iq_samples != self.num_iq_samples:
            raise ValueError(
                f"Expected {self.num_iq_samples} IQ samples, got {num_iq_samples}."
            )
        point_energy = torch.sum(x_iq**2, dim=1)
        energy_windows = point_energy.view(batch_size, self.num_windows, self.energy_window)
        return torch.sum(energy_windows, dim=-1)

    @staticmethod
    def compute_diff(x_iq: torch.Tensor) -> torch.Tensor:
        return x_iq[:, :, 1:] - x_iq[:, :, :-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_iq = self.reshape_iq(x)
        h_raw = self.raw_branch(x_iq)

        x_energy = self.compute_local_energy(x_iq)
        h_energy = self.energy_branch(x_energy)

        x_diff = self.compute_diff(x_iq)
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
        threshold: float = 0.5,
        threshold_mode: str | None = None,
        target_pfa: float = 0.1,
        calibration_split: str = "val",
        decision_threshold: float | None = None,
        snr_loss_weighting: str | None = "two_band",
        low_snr_cutoff: int = -10,
        low_snr_positive_weight: float = 3.0,
        mid_snr_cutoff: int = -6,
        mid_snr_positive_weight: float = 2.0,
        device: str | None = None,
    ):
        threshold_mode = self._normalize_threshold_mode(threshold_mode)
        snr_loss_weighting = self._normalize_snr_loss_weighting(snr_loss_weighting)
        super().__init__(
            signal_length=signal_length,
            energy_window=energy_window,
            dropout=dropout,
            lr=lr,
            batch_size=batch_size,
            epochs=epochs,
            weight_decay=weight_decay,
            threshold=threshold,
            threshold_mode=threshold_mode,
            target_pfa=target_pfa,
            calibration_split=calibration_split,
            decision_threshold=decision_threshold,
            snr_loss_weighting=snr_loss_weighting,
            low_snr_cutoff=low_snr_cutoff,
            low_snr_positive_weight=low_snr_positive_weight,
            mid_snr_cutoff=mid_snr_cutoff,
            mid_snr_positive_weight=mid_snr_positive_weight,
            device=device,
        )
        self.signal_length = signal_length
        self.energy_window = int(energy_window)
        self.dropout = float(dropout)
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.weight_decay = float(weight_decay)
        self.threshold = float(threshold)
        self.threshold_mode = threshold_mode
        self.target_pfa = float(target_pfa)
        self.calibration_split = str(calibration_split)
        self.decision_threshold = None if decision_threshold is None else float(decision_threshold)
        self.snr_loss_weighting = snr_loss_weighting
        self.low_snr_cutoff = int(low_snr_cutoff)
        self.low_snr_positive_weight = float(low_snr_positive_weight)
        self.mid_snr_cutoff = int(mid_snr_cutoff)
        self.mid_snr_positive_weight = float(mid_snr_positive_weight)
        self._validate_snr_loss_config()
        self.prefers_internal_threshold = self.decision_threshold is not None and self.threshold_mode is not None
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model: CNRSenseNet | None = None
        self.history = TrainingHistory(train_loss=[], val_loss=[])
        self.fit_result: CalibratedThresholdResult | None = None

    @staticmethod
    def _normalize_threshold_mode(threshold_mode: str | None) -> str | None:
        if threshold_mode is None:
            return None
        normalized = str(threshold_mode).strip().lower()
        if normalized in {"", "none", "fixed"}:
            return None
        return normalized

    @staticmethod
    def _normalize_snr_loss_weighting(snr_loss_weighting: str | None) -> str:
        if snr_loss_weighting is None:
            return "none"
        normalized = str(snr_loss_weighting).strip().lower()
        if normalized in {"", "none", "off", "disabled", "fixed"}:
            return "none"
        if normalized in {"two_band", "two-band", "2band"}:
            return "two_band"
        raise ValueError("snr_loss_weighting must be one of: none, two_band.")

    def _validate_snr_loss_config(self) -> None:
        if self.mid_snr_cutoff < self.low_snr_cutoff:
            raise ValueError("mid_snr_cutoff must be greater than or equal to low_snr_cutoff.")
        if self.low_snr_positive_weight <= 0.0 or self.mid_snr_positive_weight <= 0.0:
            raise ValueError("Positive sample weights must be greater than 0.")

    @staticmethod
    def _to_numpy(values) -> np.ndarray:
        if isinstance(values, torch.Tensor):
            return values.detach().cpu().numpy()
        return np.asarray(values)

    @staticmethod
    def _unpack_batch(batch):
        if not isinstance(batch, (list, tuple)) or len(batch) < 2:
            raise ValueError("Expected dataset items to be tuples like (x, y, ...).")
        return batch[0], batch[1]

    @staticmethod
    def _extract_snr(batch):
        if isinstance(batch, (list, tuple)) and len(batch) >= 3:
            return batch[2]
        return None

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

    @staticmethod
    def _prepare_snr(snr: torch.Tensor | None) -> torch.Tensor | None:
        if snr is None:
            return None
        if not isinstance(snr, torch.Tensor):
            snr = torch.tensor(snr, dtype=torch.int64)
        return snr.to(dtype=torch.int64).view(-1)

    def _infer_signal_length(self, dataset) -> int:
        sample = dataset[0]
        x, _ = self._unpack_batch(sample)
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        return int(x.numel())

    def _dataset_labels(self, dataset) -> np.ndarray:
        if hasattr(dataset, "y"):
            labels = self._to_numpy(dataset.y)
        elif hasattr(dataset, "dataset") and hasattr(dataset.dataset, "y") and hasattr(dataset, "indices"):
            labels = self._to_numpy(dataset.dataset.y)[np.asarray(dataset.indices, dtype=np.int64)]
        else:
            labels = [dataset[idx][1] for idx in range(len(dataset))]
        return self._to_numpy(labels).reshape(-1).astype(np.int64, copy=False)

    def _choose_calibration_dataset(self, train_dataset, val_dataset):
        if self.calibration_split == "val" and val_dataset is not None:
            return val_dataset, "val"
        return train_dataset, "train"

    def _compute_sample_weights(
        self,
        y: torch.Tensor,
        snr: torch.Tensor | None,
        device: torch.device,
    ) -> torch.Tensor:
        weights = torch.ones_like(y, dtype=torch.float32, device=device)
        if self.snr_loss_weighting != "two_band" or snr is None:
            return weights

        positive_mask = y >= 0.5
        low_mask = positive_mask & (snr <= self.low_snr_cutoff)
        mid_mask = positive_mask & (snr > self.low_snr_cutoff) & (snr <= self.mid_snr_cutoff)

        if self.low_snr_positive_weight != 1.0:
            weights = torch.where(
                low_mask,
                torch.full_like(weights, self.low_snr_positive_weight),
                weights,
            )
        if self.mid_snr_positive_weight != 1.0:
            weights = torch.where(
                mid_mask,
                torch.full_like(weights, self.mid_snr_positive_weight),
                weights,
            )
        return weights

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
                snr = self._extract_snr(batch)
                x = self._prepare_inputs(x).to(self.device)
                y = self._prepare_targets(y).to(self.device)
                snr = self._prepare_snr(snr)
                if snr is not None:
                    snr = snr.to(self.device)

                logits = self.model(x)
                losses = criterion(logits, y)
                sample_weights = self._compute_sample_weights(y, snr, device=y.device)
                loss = torch.sum(losses * sample_weights) / torch.clamp(sample_weights.sum(), min=1.0)

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
        threshold_mode = self._normalize_threshold_mode(kwargs.get("threshold_mode", self.threshold_mode))
        target_pfa = float(kwargs.get("target_pfa", self.target_pfa))
        calibration_split = str(kwargs.get("calibration_split", self.calibration_split))
        snr_loss_weighting = self._normalize_snr_loss_weighting(
            kwargs.get("snr_loss_weighting", self.snr_loss_weighting)
        )
        low_snr_cutoff = int(kwargs.get("low_snr_cutoff", self.low_snr_cutoff))
        low_snr_positive_weight = float(
            kwargs.get("low_snr_positive_weight", self.low_snr_positive_weight)
        )
        mid_snr_cutoff = int(kwargs.get("mid_snr_cutoff", self.mid_snr_cutoff))
        mid_snr_positive_weight = float(
            kwargs.get("mid_snr_positive_weight", self.mid_snr_positive_weight)
        )
        verbose = bool(kwargs.get("verbose", False))

        if calibration_split not in {"train", "val"}:
            raise ValueError("calibration_split must be either 'train' or 'val'.")

        self.batch_size = batch_size
        self.threshold_mode = threshold_mode
        self.target_pfa = target_pfa
        self.calibration_split = calibration_split
        self.snr_loss_weighting = snr_loss_weighting
        self.low_snr_cutoff = low_snr_cutoff
        self.low_snr_positive_weight = low_snr_positive_weight
        self.mid_snr_cutoff = mid_snr_cutoff
        self.mid_snr_positive_weight = mid_snr_positive_weight
        self._validate_snr_loss_config()
        self.config["lr"] = lr
        self.config["batch_size"] = batch_size
        self.config["weight_decay"] = weight_decay
        self.config["threshold_mode"] = threshold_mode
        self.config["target_pfa"] = target_pfa
        self.config["calibration_split"] = calibration_split
        self.config["snr_loss_weighting"] = snr_loss_weighting
        self.config["low_snr_cutoff"] = low_snr_cutoff
        self.config["low_snr_positive_weight"] = low_snr_positive_weight
        self.config["mid_snr_cutoff"] = mid_snr_cutoff
        self.config["mid_snr_positive_weight"] = mid_snr_positive_weight

        signal_length = self.signal_length or self._infer_signal_length(train_dataset)
        self._ensure_model(signal_length=signal_length)

        criterion = nn.BCEWithLogitsLoss(reduction="none")
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

        self.fit_result = None
        self.decision_threshold = None
        self.prefers_internal_threshold = False
        self.config["decision_threshold"] = None

        if self.threshold_mode is not None:
            calibration_dataset, calibration_source = self._choose_calibration_dataset(
                train_dataset,
                val_dataset,
            )
            calibration_scores = self.predict_scores(calibration_dataset)
            calibration_labels = self._dataset_labels(calibration_dataset)
            self.fit_result = fit_binary_threshold(
                scores=calibration_scores,
                labels=calibration_labels,
                threshold_mode=self.threshold_mode,
                target_pfa=self.target_pfa,
                calibration_source=calibration_source,
            )
            self.decision_threshold = float(self.fit_result.threshold)
            self.prefers_internal_threshold = True
            self.config["decision_threshold"] = self.decision_threshold

            if verbose:
                print(
                    f"[Threshold Calibration] threshold={self.decision_threshold:.6f} "
                    f"Pd={self.fit_result.Pd:.4f} Pfa={self.fit_result.Pfa:.4f} "
                    f"BA={self.fit_result.balanced_accuracy:.4f} "
                    f"({self.fit_result.calibration_source})"
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

    def get_evaluation_threshold(self, threshold=None):
        if threshold is not None:
            return float(threshold)
        if self.prefers_internal_threshold and self.decision_threshold is not None:
            return float(self.decision_threshold)
        return float(self.threshold)

    def predict(self, dataset, threshold=None):
        threshold = self.get_evaluation_threshold(threshold)
        scores = self.predict_scores(dataset)
        return (scores >= threshold).astype(np.int64)

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
        self.prefers_internal_threshold = self.decision_threshold is not None and self.threshold_mode is not None
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
