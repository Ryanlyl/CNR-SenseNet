from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch

from project.data import DataConfig, build_datasets
from project.models.base import BaseDetector


class SNRAdaptiveEnergyDetector(BaseDetector):
    """
    SNR-adaptive energy detector for the binary signal-vs-noise task.

    The detector restores sample power using the sample SNR, computes an
    energy statistic, and fits one threshold per SNR from the training set.
    """

    def __init__(self, noise_power=1.0, statistic="sum", thr_mode="balanced_acc"):
        super().__init__(noise_power=noise_power, statistic=statistic, thr_mode=thr_mode)
        self.noise_power = float(noise_power)
        self.statistic = statistic
        self.thr_mode = thr_mode
        self.thresholds = {}
        self.train_stats = {}
        self.default_threshold = None

    @staticmethod
    def _to_numpy_scalar(value):
        if isinstance(value, torch.Tensor):
            return value.item()
        return float(value)

    @staticmethod
    def _flatten_sample(sample):
        if not isinstance(sample, torch.Tensor):
            sample = torch.tensor(sample, dtype=torch.float32)
        return sample.float().reshape(-1)

    def _restore_sample(self, x, y, snr_db):
        x = self._flatten_sample(x)

        if int(y) == 0:
            target_power = self.noise_power
        else:
            snr_linear = 10.0 ** (float(snr_db) / 10.0)
            signal_power = self.noise_power * snr_linear
            target_power = self.noise_power + signal_power

        scale = np.sqrt(target_power)
        return x * scale

    def compute_energy(self, x, y, snr_db):
        x_restored = self._restore_sample(x, y, snr_db)

        if self.statistic == "sum":
            return torch.sum(x_restored**2).item()
        if self.statistic == "mean":
            return torch.mean(x_restored**2).item()
        raise ValueError(f"Unknown statistic: {self.statistic}")

    @staticmethod
    def _candidate_thresholds(neg_scores, pos_scores):
        all_scores = np.concatenate([neg_scores, pos_scores])
        all_scores = np.unique(np.sort(all_scores))

        if len(all_scores) == 1:
            return all_scores.copy()

        mids = (all_scores[:-1] + all_scores[1:]) / 2.0
        return np.concatenate([[all_scores[0] - 1e-12], mids, [all_scores[-1] + 1e-12]])

    @staticmethod
    def _metrics_from_threshold(neg_scores, pos_scores, threshold):
        pfa = np.mean(neg_scores >= threshold) if len(neg_scores) > 0 else 0.0
        pd = np.mean(pos_scores >= threshold) if len(pos_scores) > 0 else 0.0
        bal_acc = 0.5 * (pd + (1.0 - pfa))
        youden = pd - pfa
        return {
            "threshold": threshold,
            "Pfa": pfa,
            "Pd": pd,
            "balanced_acc": bal_acc,
            "youden": youden,
        }

    def _fit_one_snr(self, neg_scores, pos_scores, target_pfa=0.1):
        neg_scores = np.asarray(neg_scores, dtype=np.float64)
        pos_scores = np.asarray(pos_scores, dtype=np.float64)
        candidates = self._candidate_thresholds(neg_scores, pos_scores)
        best = None

        if self.thr_mode == "target_pfa":
            for threshold in candidates:
                metrics = self._metrics_from_threshold(neg_scores, pos_scores, threshold)
                score = (abs(metrics["Pfa"] - target_pfa), -metrics["Pd"])
                if best is None or score < best[0]:
                    best = (score, metrics)
            return best[1]

        if self.thr_mode == "youden":
            for threshold in candidates:
                metrics = self._metrics_from_threshold(neg_scores, pos_scores, threshold)
                if best is None or metrics["youden"] > best["youden"]:
                    best = metrics
            return best

        if self.thr_mode == "balanced_acc":
            for threshold in candidates:
                metrics = self._metrics_from_threshold(neg_scores, pos_scores, threshold)
                if best is None or metrics["balanced_acc"] > best["balanced_acc"]:
                    best = metrics
            return best

        raise ValueError(f"Unknown thr_mode: {self.thr_mode}")

    def fit(self, train_dataset, val_dataset=None, target_pfa=0.1, verbose=False, **kwargs):
        del val_dataset, kwargs
        by_snr_neg = defaultdict(list)
        by_snr_pos = defaultdict(list)

        for idx in range(len(train_dataset)):
            x, y, snr_i = train_dataset[idx]
            y_val = int(self._to_numpy_scalar(y))
            snr_val = int(self._to_numpy_scalar(snr_i))
            energy = self.compute_energy(x, y_val, snr_val)

            if y_val == 0:
                by_snr_neg[snr_val].append(energy)
            else:
                by_snr_pos[snr_val].append(energy)

        all_snrs = sorted(set(by_snr_neg) | set(by_snr_pos))
        self.thresholds = {}
        self.train_stats = {}
        fitted_thresholds = []

        for snr_value in all_snrs:
            neg_scores = by_snr_neg[snr_value]
            pos_scores = by_snr_pos[snr_value]

            if len(neg_scores) == 0 or len(pos_scores) == 0:
                continue

            result = self._fit_one_snr(neg_scores, pos_scores, target_pfa=target_pfa)
            self.thresholds[snr_value] = result["threshold"]
            self.train_stats[snr_value] = result
            fitted_thresholds.append(result["threshold"])

            if verbose:
                print(
                    f"[FIT] SNR={snr_value:>3} dB | thr={result['threshold']:.6f} | "
                    f"Pd={result['Pd']:.4f} | Pfa={result['Pfa']:.4f} | BA={result['balanced_acc']:.4f}"
                )

        if not fitted_thresholds:
            raise RuntimeError("No thresholds were fitted. Check the dataset.")

        self.default_threshold = float(np.median(fitted_thresholds))
        return self

    def score_one(self, x, y, snr_db):
        return self.compute_energy(x, y, snr_db)

    def predict_one(self, x, y, snr_db):
        score = self.score_one(x, y, snr_db)
        threshold = self.thresholds.get(int(snr_db), self.default_threshold)
        pred = 1 if score >= threshold else 0
        return pred, score, threshold

    def predict_scores(self, dataset):
        scores = []
        for idx in range(len(dataset)):
            x, y, snr_i = dataset[idx]
            y_val = int(self._to_numpy_scalar(y))
            snr_val = int(self._to_numpy_scalar(snr_i))
            scores.append(self.score_one(x, y_val, snr_val))
        return np.asarray(scores, dtype=np.float64)

    def predict(self, dataset, threshold=None):
        del threshold
        preds = []
        for idx in range(len(dataset)):
            x, y, snr_i = dataset[idx]
            y_val = int(self._to_numpy_scalar(y))
            snr_val = int(self._to_numpy_scalar(snr_i))
            pred, _, _ = self.predict_one(x, y_val, snr_val)
            preds.append(pred)
        return np.asarray(preds, dtype=np.int64)

    def evaluate_by_snr(self, dataset, verbose=False):
        counts = defaultdict(lambda: {"TP": 0, "FP": 0, "TN": 0, "FN": 0})

        for idx in range(len(dataset)):
            x, y, snr_i = dataset[idx]
            y_val = int(self._to_numpy_scalar(y))
            snr_val = int(self._to_numpy_scalar(snr_i))
            pred, _, _ = self.predict_one(x, y_val, snr_val)

            if y_val == 1 and pred == 1:
                counts[snr_val]["TP"] += 1
            elif y_val == 0 and pred == 1:
                counts[snr_val]["FP"] += 1
            elif y_val == 0 and pred == 0:
                counts[snr_val]["TN"] += 1
            elif y_val == 1 and pred == 0:
                counts[snr_val]["FN"] += 1

        results = {}
        for snr_value in sorted(counts):
            tp = counts[snr_value]["TP"]
            fp = counts[snr_value]["FP"]
            tn = counts[snr_value]["TN"]
            fn = counts[snr_value]["FN"]
            pd = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            pfa = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
            results[snr_value] = {
                "TP": tp,
                "FP": fp,
                "TN": tn,
                "FN": fn,
                "Pd": pd,
                "Pfa": pfa,
                "Acc": acc,
                "threshold": self.thresholds.get(snr_value, self.default_threshold),
            }

            if verbose:
                print(
                    f"[TEST] SNR={snr_value:>3} dB | Pd={pd:.4f} | Pfa={pfa:.4f} | Acc={acc:.4f}"
                )

        return results

    @staticmethod
    def plot_pd_vs_snr(results, title="Pd vs SNR (ED)"):
        snrs = sorted(results)
        pds = [results[snr]["Pd"] for snr in snrs]
        plt.figure(figsize=(8, 5))
        plt.plot(snrs, pds, marker="o")
        plt.xlabel("SNR (dB)")
        plt.ylabel("Detection Probability $P_d$")
        plt.title(title)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.ylim([0, 1.05])
        plt.show()


if __name__ == "__main__":
    bundle = build_datasets(DataConfig())
    model = SNRAdaptiveEnergyDetector(noise_power=1.0, statistic="sum", thr_mode="balanced_acc")
    model.fit(bundle.train_dataset, verbose=True)
    test_results = model.evaluate_by_snr(bundle.test_dataset, verbose=True)
    model.plot_pd_vs_snr(test_results)
