from __future__ import annotations

from abc import abstractmethod
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from project.models.base import BaseDetector, CalibratedThresholdResult, fit_binary_threshold


@dataclass(slots=True)
class TrainingHistory:
    train_loss: list[float]
    val_loss: list[float]


class TorchBinaryClassifier(BaseDetector):
    """Reusable trainer for binary PyTorch baselines."""

    def __init__(
        self,
        signal_length: int | None = None,
        lr: float = 1e-3,
        batch_size: int = 1024,
        epochs: int = 5,
        weight_decay: float = 0.0,
        threshold: float = 0.5,
        threshold_mode: str | None = None,
        target_pfa: float = 0.1,
        calibration_split: str = "val",
        decision_threshold: float | None = None,
        device: str | None = None,
    ):
        threshold_mode = self._normalize_threshold_mode(threshold_mode)
        super().__init__(
            signal_length=signal_length,
            lr=lr,
            batch_size=batch_size,
            epochs=epochs,
            weight_decay=weight_decay,
            threshold=threshold,
            threshold_mode=threshold_mode,
            target_pfa=target_pfa,
            calibration_split=calibration_split,
            decision_threshold=decision_threshold,
            device=device,
        )
        self.signal_length = signal_length
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.weight_decay = float(weight_decay)
        self.threshold = float(threshold)
        self.threshold_mode = threshold_mode
        self.target_pfa = float(target_pfa)
        self.calibration_split = str(calibration_split)
        self.decision_threshold = None if decision_threshold is None else float(decision_threshold)
        self.prefers_internal_threshold = self.decision_threshold is not None and self.threshold_mode is not None
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model: nn.Module | None = None
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
    def _prepare_targets(y: torch.Tensor) -> torch.Tensor:
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32)
        return y.float().view(-1)

    def _prepare_inputs(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        return x.float().view(x.shape[0], -1)

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

    @abstractmethod
    def build_network(self, signal_length: int) -> nn.Module:
        """Create the model backbone for a given signal length."""

    def _ensure_model(self, signal_length: int | None = None) -> nn.Module:
        inferred_length = int(signal_length or self.signal_length or 0)
        if inferred_length <= 0:
            raise ValueError("signal_length could not be inferred.")
        if self.model is None or self.signal_length != inferred_length:
            self.model = self.build_network(inferred_length).to(self.device)
            self.signal_length = inferred_length
        return self.model

    def _make_loader(self, dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=self.device.type == "cuda",
        )

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
        patience = kwargs.get("patience")
        threshold_mode = self._normalize_threshold_mode(kwargs.get("threshold_mode", self.threshold_mode))
        target_pfa = float(kwargs.get("target_pfa", self.target_pfa))
        calibration_split = str(kwargs.get("calibration_split", self.calibration_split))
        verbose = bool(kwargs.get("verbose", False))

        if calibration_split not in {"train", "val"}:
            raise ValueError("calibration_split must be either 'train' or 'val'.")

        self.batch_size = batch_size
        self.threshold_mode = threshold_mode
        self.target_pfa = target_pfa
        self.calibration_split = calibration_split
        self.config["lr"] = lr
        self.config["batch_size"] = batch_size
        self.config["weight_decay"] = weight_decay
        self.config["threshold_mode"] = threshold_mode
        self.config["target_pfa"] = target_pfa
        self.config["calibration_split"] = calibration_split

        signal_length = self.signal_length or self._infer_signal_length(train_dataset)
        self._ensure_model(signal_length=signal_length)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        train_loader = self._make_loader(train_dataset, shuffle=True)
        val_loader = self._make_loader(val_dataset, shuffle=False) if val_dataset is not None else None

        best_state = deepcopy(self.model.state_dict())
        best_val_loss = float("inf")
        epochs_without_improvement = 0
        self.history = TrainingHistory(train_loss=[], val_loss=[])

        for epoch in range(epochs):
            train_loss = self._run_epoch(train_loader, criterion, optimizer=optimizer)
            self.history.train_loss.append(float(train_loss))

            if val_loader is None:
                if verbose:
                    print(f"[Epoch {epoch + 1}/{epochs}] train_loss={train_loss:.6f}")
                continue

            val_loss = self._run_epoch(val_loader, criterion, optimizer=None)
            self.history.val_loss.append(float(val_loss))

            if val_loss < best_val_loss:
                best_val_loss = float(val_loss)
                best_state = deepcopy(self.model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if verbose:
                print(
                    f"[Epoch {epoch + 1}/{epochs}] train_loss={train_loss:.6f} "
                    f"val_loss={val_loss:.6f}"
                )

            if patience is not None and epochs_without_improvement >= int(patience):
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}.")
                break

        if val_loader is not None:
            self.model.load_state_dict(best_state)

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

    def predict_logits(self, dataset):
        signal_length = self.signal_length or self._infer_signal_length(dataset)
        self._ensure_model(signal_length=signal_length)
        loader = self._make_loader(dataset, shuffle=False)
        self.model.eval()

        logits_list: list[np.ndarray] = []
        with torch.no_grad():
            for batch in loader:
                x, _ = self._unpack_batch(batch)
                x = self._prepare_inputs(x).to(self.device)
                logits = self.model(x)
                logits_list.append(logits.detach().cpu().numpy())

        if not logits_list:
            return np.asarray([], dtype=np.float32)
        return np.concatenate(logits_list, axis=0).astype(np.float32)

    def predict_scores(self, dataset):
        logits = self.predict_logits(dataset)
        return 1.0 / (1.0 + np.exp(-logits))

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
                raise ValueError("Set signal_length before loading weights.")
            self._ensure_model(signal_length=self.signal_length)
        self.model.load_state_dict(state_dict)
        self.prefers_internal_threshold = self.decision_threshold is not None and self.threshold_mode is not None
        return self
