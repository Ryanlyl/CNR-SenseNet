from dataclasses import dataclass
from typing import Any

import numpy as np
from abc import ABC, abstractmethod


@dataclass(slots=True)
class CalibratedThresholdResult:
    threshold: float
    Pd: float
    Pfa: float
    balanced_accuracy: float
    youden: float
    calibration_source: str
    num_positive: int
    num_negative: int


def candidate_thresholds(neg_scores: np.ndarray, pos_scores: np.ndarray) -> np.ndarray:
    all_scores = np.unique(np.sort(np.concatenate([neg_scores, pos_scores])))
    if all_scores.size == 1:
        return all_scores.astype(np.float64, copy=True)

    mids = (all_scores[:-1] + all_scores[1:]) / 2.0
    return np.concatenate(
        [
            np.asarray([all_scores[0] - 1e-12], dtype=np.float64),
            mids.astype(np.float64, copy=False),
            np.asarray([all_scores[-1] + 1e-12], dtype=np.float64),
        ]
    )


def threshold_metrics(
    neg_scores: np.ndarray,
    pos_scores: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    pfa = float(np.mean(neg_scores >= threshold)) if neg_scores.size > 0 else 0.0
    pd = float(np.mean(pos_scores >= threshold)) if pos_scores.size > 0 else 0.0
    return {
        "threshold": float(threshold),
        "Pd": pd,
        "Pfa": pfa,
        "balanced_accuracy": 0.5 * (pd + (1.0 - pfa)),
        "youden": pd - pfa,
    }


def fit_binary_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold_mode: str,
    target_pfa: float = 0.1,
    calibration_source: str = "val",
) -> CalibratedThresholdResult:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)

    neg_scores = np.asarray(scores[labels == 0], dtype=np.float64)
    pos_scores = np.asarray(scores[labels == 1], dtype=np.float64)
    if neg_scores.size == 0 or pos_scores.size == 0:
        raise RuntimeError("Threshold calibration requires both positive and negative samples.")

    if threshold_mode == "target_pfa":
        best = None
        for threshold in candidate_thresholds(neg_scores, pos_scores):
            metrics = threshold_metrics(neg_scores, pos_scores, threshold)
            score = (abs(metrics["Pfa"] - float(target_pfa)), -metrics["Pd"])
            if best is None or score < best[0]:
                best = (score, metrics)
        chosen = best[1]
    elif threshold_mode in {"balanced_acc", "youden"}:
        metric_name = "balanced_accuracy" if threshold_mode == "balanced_acc" else "youden"
        chosen = None
        for threshold in candidate_thresholds(neg_scores, pos_scores):
            metrics = threshold_metrics(neg_scores, pos_scores, threshold)
            if chosen is None or metrics[metric_name] > chosen[metric_name]:
                chosen = metrics
    else:
        raise ValueError(
            f"Unknown threshold_mode '{threshold_mode}'. Expected one of: balanced_acc, youden, target_pfa."
        )

    return CalibratedThresholdResult(
        threshold=float(chosen["threshold"]),
        Pd=float(chosen["Pd"]),
        Pfa=float(chosen["Pfa"]),
        balanced_accuracy=float(chosen["balanced_accuracy"]),
        youden=float(chosen["youden"]),
        calibration_source=str(calibration_source),
        num_positive=int(pos_scores.size),
        num_negative=int(neg_scores.size),
    )


class BaseDetector(ABC):
    """Unified interface for all binary detection models in this project."""

    prefers_internal_threshold = False

    def __init__(self, **config: Any):
        self.config = dict(config)

    @abstractmethod
    def fit(self, train_dataset, val_dataset=None, **kwargs):
        """Train or calibrate the detector."""

    @abstractmethod
    def predict_scores(self, dataset):
        """Return real-valued detection scores. Larger means more likely to be signal."""

    def predict(self, dataset, threshold=0.0):
        scores = self.predict_scores(dataset)
        return (np.asarray(scores) >= threshold).astype(np.int64)

    def get_evaluation_threshold(self, threshold=None):
        return threshold

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict: dict):
        del state_dict
        return self

    def get_config(self):
        return dict(self.config)
