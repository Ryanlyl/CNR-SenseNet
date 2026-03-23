from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch

from project.models.base import BaseDetector


@dataclass(slots=True)
class ThresholdFitResult:
    threshold: float
    Pd: float
    Pfa: float
    balanced_accuracy: float
    youden: float
    calibration_source: str
    num_positive: int
    num_negative: int


class StatisticalThresholdDetector(BaseDetector):
    """
    Base class for classical spectrum sensing baselines with scalar test statistics.

    These detectors do not learn network weights. They compute a score from the
    received sample and calibrate a decision threshold on the train/validation
    split using labeled data.
    """

    prefers_internal_threshold = True

    def __init__(
        self,
        threshold_mode: str = "balanced_acc",
        target_pfa: float = 0.1,
        calibration_split: str = "val",
        score_batch_size: int = 16384,
        eps: float = 1e-12,
        **config,
    ):
        super().__init__(
            threshold_mode=threshold_mode,
            target_pfa=target_pfa,
            calibration_split=calibration_split,
            score_batch_size=score_batch_size,
            eps=eps,
            **config,
        )
        self.threshold_mode = str(threshold_mode)
        self.target_pfa = float(target_pfa)
        self.calibration_split = str(calibration_split)
        self.score_batch_size = int(score_batch_size)
        self.eps = float(eps)
        self.decision_threshold: float | None = None
        self.fit_result: ThresholdFitResult | None = None

    @staticmethod
    def _to_numpy(values) -> np.ndarray:
        if isinstance(values, torch.Tensor):
            return values.detach().cpu().numpy()
        return np.asarray(values)

    def _choose_calibration_dataset(self, train_dataset, val_dataset):
        if self.calibration_split == "val" and val_dataset is not None:
            return val_dataset, "val"
        return train_dataset, "train"

    def _dataset_labels(self, dataset) -> np.ndarray:
        if hasattr(dataset, "y"):
            labels = self._to_numpy(dataset.y)
        else:
            labels = [dataset[idx][1] for idx in range(len(dataset))]
        return self._to_numpy(labels).reshape(-1).astype(np.int64, copy=False)

    def _score_dataset(self, dataset) -> np.ndarray:
        if hasattr(dataset, "X"):
            samples = self._to_numpy(dataset.X).reshape(len(dataset), -1)
            score_chunks = []
            for start in range(0, len(samples), self.score_batch_size):
                batch = samples[start : start + self.score_batch_size]
                score_chunks.append(self._score_array(batch))
            if not score_chunks:
                return np.asarray([], dtype=np.float32)
            return np.concatenate(score_chunks, axis=0).astype(np.float32, copy=False)

        buffered_samples = []
        score_chunks = []
        for idx in range(len(dataset)):
            x = self._to_numpy(dataset[idx][0]).reshape(-1).astype(np.float32, copy=False)
            buffered_samples.append(x)
            if len(buffered_samples) >= self.score_batch_size:
                score_chunks.append(self._score_array(np.stack(buffered_samples, axis=0)))
                buffered_samples = []

        if buffered_samples:
            score_chunks.append(self._score_array(np.stack(buffered_samples, axis=0)))

        if not score_chunks:
            return np.asarray([], dtype=np.float32)
        return np.concatenate(score_chunks, axis=0).astype(np.float32, copy=False)

    @staticmethod
    def _candidate_thresholds(neg_scores: np.ndarray, pos_scores: np.ndarray) -> np.ndarray:
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

    @staticmethod
    def _metrics_from_threshold(
        neg_scores: np.ndarray, pos_scores: np.ndarray, threshold: float
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

    def _fit_threshold(self, scores: np.ndarray, labels: np.ndarray, calibration_source: str):
        neg_scores = np.asarray(scores[labels == 0], dtype=np.float64)
        pos_scores = np.asarray(scores[labels == 1], dtype=np.float64)
        if neg_scores.size == 0 or pos_scores.size == 0:
            raise RuntimeError("Threshold calibration requires both positive and negative samples.")

        if self.threshold_mode == "target_pfa":
            candidates = self._candidate_thresholds(neg_scores, pos_scores)
            best = None
            for threshold in candidates:
                metrics = self._metrics_from_threshold(neg_scores, pos_scores, threshold)
                score = (abs(metrics["Pfa"] - self.target_pfa), -metrics["Pd"])
                if best is None or score < best[0]:
                    best = (score, metrics)
            chosen = best[1]
        elif self.threshold_mode in {"balanced_acc", "youden"}:
            metric_name = "balanced_accuracy" if self.threshold_mode == "balanced_acc" else "youden"
            chosen = None
            for threshold in self._candidate_thresholds(neg_scores, pos_scores):
                metrics = self._metrics_from_threshold(neg_scores, pos_scores, threshold)
                if chosen is None or metrics[metric_name] > chosen[metric_name]:
                    chosen = metrics
        else:
            raise ValueError(
                f"Unknown threshold_mode '{self.threshold_mode}'. "
                "Expected one of: balanced_acc, youden, target_pfa."
            )

        self.decision_threshold = float(chosen["threshold"])
        self.fit_result = ThresholdFitResult(
            threshold=float(chosen["threshold"]),
            Pd=float(chosen["Pd"]),
            Pfa=float(chosen["Pfa"]),
            balanced_accuracy=float(chosen["balanced_accuracy"]),
            youden=float(chosen["youden"]),
            calibration_source=calibration_source,
            num_positive=int(pos_scores.size),
            num_negative=int(neg_scores.size),
        )

    def fit(self, train_dataset, val_dataset=None, **kwargs):
        threshold_mode = kwargs.get("threshold_mode")
        target_pfa = kwargs.get("target_pfa")
        calibration_split = kwargs.get("calibration_split")
        verbose = bool(kwargs.get("verbose", False))

        if threshold_mode is not None:
            self.threshold_mode = str(threshold_mode)
            self.config["threshold_mode"] = self.threshold_mode
        if target_pfa is not None:
            self.target_pfa = float(target_pfa)
            self.config["target_pfa"] = self.target_pfa
        if calibration_split is not None:
            self.calibration_split = str(calibration_split)
            self.config["calibration_split"] = self.calibration_split

        calibration_dataset, calibration_source = self._choose_calibration_dataset(
            train_dataset, val_dataset
        )
        calibration_scores = self._score_dataset(calibration_dataset)
        calibration_labels = self._dataset_labels(calibration_dataset)
        self._fit_threshold(calibration_scores, calibration_labels, calibration_source)

        if verbose and self.fit_result is not None:
            print(
                f"[{self.__class__.__name__}] threshold={self.fit_result.threshold:.6f} "
                f"Pd={self.fit_result.Pd:.4f} Pfa={self.fit_result.Pfa:.4f} "
                f"BA={self.fit_result.balanced_accuracy:.4f} "
                f"(calibrated on {self.fit_result.calibration_source})"
            )
        return self

    def predict_scores(self, dataset):
        return self._score_dataset(dataset)

    def get_evaluation_threshold(self, threshold=None):
        if threshold is not None:
            return float(threshold)
        if self.decision_threshold is None:
            raise RuntimeError("Detector threshold is not calibrated. Call fit() first.")
        return float(self.decision_threshold)

    def predict(self, dataset, threshold=None):
        threshold = self.get_evaluation_threshold(threshold)
        scores = self.predict_scores(dataset)
        return (np.asarray(scores) >= threshold).astype(np.int64)

    def state_dict(self):
        return {
            "config": self.get_config(),
            "decision_threshold": self.decision_threshold,
            "fit_result": asdict(self.fit_result) if self.fit_result is not None else None,
        }

    def load_state_dict(self, state_dict: dict):
        self.decision_threshold = (
            None
            if state_dict.get("decision_threshold") is None
            else float(state_dict["decision_threshold"])
        )
        fit_result = state_dict.get("fit_result")
        self.fit_result = ThresholdFitResult(**fit_result) if fit_result is not None else None
        return self

    def _score_array(self, samples: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class EnergyDetector(StatisticalThresholdDetector):
    """
    Classical energy detector for interleaved IQ samples.

    For your `.npz` dataset each sample is `[I0, Q0, I1, Q1, ...]`, so the
    detector computes the average complex-baseband power:

        T(x) = mean(I[n]^2 + Q[n]^2)

    This uses only the received sample, without leaking the ground-truth label
    or SNR into the score.
    """

    def __init__(self, statistic: str = "avg_power", **kwargs):
        super().__init__(statistic=statistic, **kwargs)
        self.statistic = str(statistic)

    def _score_array(self, samples: np.ndarray) -> np.ndarray:
        flattened = np.asarray(samples, dtype=np.float32).reshape(samples.shape[0], -1)
        if flattened.shape[1] % 2 == 0:
            i = flattened[:, 0::2]
            q = flattened[:, 1::2]
            point_power = i * i + q * q
        else:
            point_power = flattened * flattened

        if self.statistic in {"avg_power", "mean"}:
            return np.mean(point_power, axis=1, dtype=np.float64)
        if self.statistic in {"sum_energy", "sum"}:
            return np.sum(point_power, axis=1, dtype=np.float64)
        raise ValueError(f"Unknown statistic '{self.statistic}'. Expected avg_power or sum_energy.")


class AutocorrelationDetector(StatisticalThresholdDetector):
    """
    Non-coherent autocorrelation detector.

    AWGN has near-zero non-zero-lag correlation, while many modulated signals
    retain short-lag structure. The score is the normalized sum of the absolute
    autocorrelation magnitudes over the first few lags.
    """

    def __init__(self, max_lag: int = 4, score_mode: str = "sum_abs_lags", **kwargs):
        super().__init__(max_lag=max_lag, score_mode=score_mode, **kwargs)
        self.max_lag = int(max_lag)
        self.score_mode = str(score_mode)

    def _as_complex_samples(self, samples: np.ndarray) -> np.ndarray:
        flattened = np.asarray(samples, dtype=np.float32).reshape(samples.shape[0], -1)
        if flattened.shape[1] % 2 != 0:
            return flattened.astype(np.complex64, copy=False)
        i = flattened[:, 0::2]
        q = flattened[:, 1::2]
        return i.astype(np.complex64) + 1j * q.astype(np.complex64)

    def _score_array(self, samples: np.ndarray) -> np.ndarray:
        complex_samples = self._as_complex_samples(samples)
        if complex_samples.shape[1] <= 1:
            return np.zeros(complex_samples.shape[0], dtype=np.float32)

        max_lag = min(self.max_lag, complex_samples.shape[1] - 1)
        zero_lag_power = np.mean(np.abs(complex_samples) ** 2, axis=1, dtype=np.float64)
        lag_scores = []

        for lag in range(1, max_lag + 1):
            correlation = np.mean(
                complex_samples[:, lag:] * np.conjugate(complex_samples[:, :-lag]),
                axis=1,
                dtype=np.complex128,
            )
            lag_scores.append(np.abs(correlation))

        lag_scores = np.stack(lag_scores, axis=1)
        if self.score_mode == "sum_abs_lags":
            numerator = np.sum(lag_scores, axis=1, dtype=np.float64)
        elif self.score_mode == "max_abs_lag":
            numerator = np.max(lag_scores, axis=1)
        else:
            raise ValueError(
                f"Unknown score_mode '{self.score_mode}'. "
                "Expected sum_abs_lags or max_abs_lag."
            )

        return numerator / (zero_lag_power + self.eps)


__all__ = [
    "AutocorrelationDetector",
    "EnergyDetector",
    "StatisticalThresholdDetector",
    "ThresholdFitResult",
]
