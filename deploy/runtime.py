from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


DEFAULT_INPUT_KEY = "X"


@dataclass(slots=True)
class EdgeMetadata:
    artifact_format: str
    model_name: str
    signal_length: int
    num_iq_samples: int
    input_layout: str
    score_activation: str
    decision_threshold: float
    config: dict[str, Any]
    checkpoint_path: str | None = None
    exported_model_path: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EdgeMetadata":
        config = dict(payload.get("config", {}))
        signal_length = int(payload.get("signal_length") or config.get("signal_length") or 0)
        if signal_length <= 0:
            raise ValueError("Metadata must include a positive signal_length.")

        decision_threshold = payload.get("decision_threshold")
        if decision_threshold is None:
            decision_threshold = config.get("decision_threshold", config.get("threshold", 0.5))

        return cls(
            artifact_format=str(payload.get("artifact_format", "cnr-sensenet-edge-v1")),
            model_name=str(payload.get("model_name", "unknown")),
            signal_length=signal_length,
            num_iq_samples=int(payload.get("num_iq_samples", signal_length // 2)),
            input_layout=str(payload.get("input_layout", "interleaved_iq")),
            score_activation=str(payload.get("score_activation", "sigmoid")),
            decision_threshold=float(decision_threshold),
            config=config,
            checkpoint_path=payload.get("checkpoint_path"),
            exported_model_path=payload.get("exported_model_path"),
        )


def load_metadata(path: str | Path) -> EdgeMetadata:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected metadata payload to be a dict, got {type(payload)!r}.")
    return EdgeMetadata.from_dict(payload)


def load_input_array(path: str | Path, input_key: str = DEFAULT_INPUT_KEY) -> np.ndarray:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        array = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if input_key not in archive.files:
                available = ", ".join(sorted(archive.files))
                raise KeyError(
                    f"Key '{input_key}' was not found in {path}. Available keys: {available}"
                )
            array = archive[input_key]
    else:
        raise ValueError(f"Unsupported input file type '{suffix}'. Expected .npy or .npz")
    return np.asarray(array, dtype=np.float32)


def normalize_iq_input(array: np.ndarray, signal_length: int) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)

    if values.ndim == 1:
        if values.shape[0] != signal_length:
            raise ValueError(f"Expected 1D input length {signal_length}, got {values.shape[0]}.")
        return values.reshape(1, signal_length)

    if values.ndim == 2:
        if values.shape == (2, signal_length // 2):
            return interleave_iq_batch(values[None, ...])
        if values.shape[1] != signal_length:
            raise ValueError(
                f"Expected 2D input shape [batch, {signal_length}] or [2, {signal_length // 2}], "
                f"got {tuple(values.shape)}."
            )
        return values.astype(np.float32, copy=False)

    if values.ndim == 3:
        if values.shape[1:] != (2, signal_length // 2):
            raise ValueError(
                f"Expected 3D input shape [batch, 2, {signal_length // 2}], got {tuple(values.shape)}."
            )
        return interleave_iq_batch(values)

    raise ValueError(
        "Expected input with shape [signal_length], [batch, signal_length], "
        "[2, num_iq_samples], or [batch, 2, num_iq_samples]."
    )


def interleave_iq_batch(values: np.ndarray) -> np.ndarray:
    batch = np.asarray(values, dtype=np.float32)
    if batch.ndim != 3 or batch.shape[1] != 2:
        raise ValueError("Expected IQ input with shape [batch, 2, num_iq_samples].")

    batch_size, _, num_iq_samples = batch.shape
    output = np.empty((batch_size, num_iq_samples * 2), dtype=np.float32)
    output[:, 0::2] = batch[:, 0, :]
    output[:, 1::2] = batch[:, 1, :]
    return output


class EdgeModel:
    """Minimal CPU-first runtime for exported TorchScript artifacts."""

    def __init__(
        self,
        model_path: str | Path,
        metadata_path: str | Path | None = None,
        device: str = "cpu",
        num_threads: int | None = None,
    ) -> None:
        self.model_path = Path(model_path)
        self.metadata_path = (
            Path(metadata_path)
            if metadata_path is not None
            else self.model_path.with_name("metadata.json")
        )
        self.device = torch.device(device)

        if num_threads is not None:
            torch.set_num_threads(int(num_threads))

        self.metadata = load_metadata(self.metadata_path)
        self.model = torch.jit.load(str(self.model_path), map_location=self.device)
        self.model.eval()

    @property
    def signal_length(self) -> int:
        return int(self.metadata.signal_length)

    @property
    def threshold(self) -> float:
        return float(self.metadata.decision_threshold)

    def prepare_array(self, array: np.ndarray) -> np.ndarray:
        return normalize_iq_input(array, signal_length=self.signal_length)

    def predict_scores_array(self, array: np.ndarray, batch_size: int = 256) -> np.ndarray:
        prepared = self.prepare_array(array)
        outputs: list[np.ndarray] = []

        with torch.no_grad():
            for start in range(0, prepared.shape[0], batch_size):
                chunk = prepared[start : start + batch_size]
                tensor = torch.from_numpy(chunk).to(self.device)
                logits = self.model(tensor)
                scores = torch.sigmoid(logits)
                outputs.append(scores.detach().cpu().numpy().reshape(-1))

        if not outputs:
            return np.asarray([], dtype=np.float32)
        return np.concatenate(outputs, axis=0).astype(np.float32, copy=False)

    def predict_labels_array(
        self,
        array: np.ndarray,
        threshold: float | None = None,
        batch_size: int = 256,
    ) -> np.ndarray:
        scores = self.predict_scores_array(array, batch_size=batch_size)
        active_threshold = self.threshold if threshold is None else float(threshold)
        return (scores >= active_threshold).astype(np.int64)

    def predict_from_path(
        self,
        input_path: str | Path,
        input_key: str = DEFAULT_INPUT_KEY,
        threshold: float | None = None,
        batch_size: int = 256,
    ) -> dict[str, Any]:
        array = load_input_array(input_path, input_key=input_key)
        scores = self.predict_scores_array(array, batch_size=batch_size)
        active_threshold = self.threshold if threshold is None else float(threshold)
        labels = (scores >= active_threshold).astype(np.int64)
        return {
            "input_path": str(Path(input_path).resolve()),
            "num_samples": int(scores.shape[0]),
            "signal_length": self.signal_length,
            "threshold": float(active_threshold),
            "scores": scores,
            "labels": labels,
        }
