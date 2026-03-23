import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict

from data.gen_dataset import (
    build_signal_vs_noise_dataset,
    stratified_split_binary,
    set_seed,
    SignalNoiseDataset
)



class SNRAdaptiveEnergyDetector:
    """
    SNR-adaptive Energy Detector for normalized RML-like datasets.

    Idea:
    1) Restore sample power using fixed noise power and sample SNR
    2) Compute energy statistic
    3) Fit one threshold per SNR from training set without gradient descent
    """

    def __init__(self, noise_power=1.0, statistic="sum", thr_mode="balanced_acc"):
        """
        Args:
            noise_power: fixed noise power sigma_n^2
            statistic: "sum" or "mean"
            thr_mode: "balanced_acc", "youden", or "target_pfa"
        """
        self.noise_power = float(noise_power)
        self.statistic = statistic
        self.thr_mode = thr_mode

        self.thresholds = {}          # {snr_value: threshold}
        self.train_stats = {}         # optional record
        self.default_threshold = None

    @staticmethod
    def _to_numpy_scalar(v):
        if isinstance(v, torch.Tensor):
            return v.item()
        return float(v)

    @staticmethod
    def _flatten_sample(x):
        """
        x can be torch tensor with shape [256] or [2,128] etc.
        Return flattened float tensor.
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        return x.float().reshape(-1)

    def _restore_sample(self, x, y, snr_db):
        """
        Restore normalized sample to physical power scale.

        H0: P = noise_power
        H1: P = noise_power * (1 + 10^(snr/10))
        """
        x = self._flatten_sample(x)

        if int(y) == 0:
            target_power = self.noise_power
        else:
            snr_lin = 10.0 ** (float(snr_db) / 10.0)
            signal_power = self.noise_power * snr_lin
            target_power = self.noise_power + signal_power

        scale = np.sqrt(target_power)
        x_restored = x * scale
        return x_restored

    def compute_energy(self, x, y, snr_db):
        """
        Compute restored energy statistic.
        """
        x_restored = self._restore_sample(x, y, snr_db)

        if self.statistic == "sum":
            return torch.sum(x_restored ** 2).item()
        elif self.statistic == "mean":
            return torch.mean(x_restored ** 2).item()
        else:
            raise ValueError(f"Unknown statistic: {self.statistic}")

    @staticmethod
    def _candidate_thresholds(neg_scores, pos_scores):
        all_scores = np.concatenate([neg_scores, pos_scores])
        all_scores = np.unique(np.sort(all_scores))

        if len(all_scores) == 1:
            return all_scores.copy()

        mids = (all_scores[:-1] + all_scores[1:]) / 2.0
        candidates = np.concatenate([
            [all_scores[0] - 1e-12],
            mids,
            [all_scores[-1] + 1e-12]
        ])
        return candidates

    @staticmethod
    def _metrics_from_threshold(neg_scores, pos_scores, thr):
        pfa = np.mean(neg_scores >= thr) if len(neg_scores) > 0 else 0.0
        pd  = np.mean(pos_scores >= thr) if len(pos_scores) > 0 else 0.0
        bal_acc = 0.5 * (pd + (1.0 - pfa))
        youden = pd - pfa
        return {
            "threshold": thr,
            "Pfa": pfa,
            "Pd": pd,
            "balanced_acc": bal_acc,
            "youden": youden
        }

    def _fit_one_snr(self, neg_scores, pos_scores, target_pfa=0.1):
        neg_scores = np.asarray(neg_scores, dtype=np.float64)
        pos_scores = np.asarray(pos_scores, dtype=np.float64)

        candidates = self._candidate_thresholds(neg_scores, pos_scores)
        best = None

        if self.thr_mode == "target_pfa":
            # Choose threshold whose Pfa is closest to target_pfa;
            # tie-breaker: higher Pd
            for thr in candidates:
                m = self._metrics_from_threshold(neg_scores, pos_scores, thr)
                cost = abs(m["Pfa"] - target_pfa)
                if best is None:
                    best = (cost, -m["Pd"], m)
                else:
                    if (cost < best[0]) or (cost == best[0] and -m["Pd"] < best[1]):
                        best = (cost, -m["Pd"], m)
            return best[2]

        elif self.thr_mode == "youden":
            for thr in candidates:
                m = self._metrics_from_threshold(neg_scores, pos_scores, thr)
                if best is None or m["youden"] > best["youden"]:
                    best = m
            return best

        elif self.thr_mode == "balanced_acc":
            for thr in candidates:
                m = self._metrics_from_threshold(neg_scores, pos_scores, thr)
                if best is None or m["balanced_acc"] > best["balanced_acc"]:
                    best = m
            return best

        else:
            raise ValueError(f"Unknown thr_mode: {self.thr_mode}")

    def fit(self, train_ds, target_pfa=0.1, verbose=True):
        """
        Fit one threshold per SNR using training dataset.
        """
        by_snr_neg = defaultdict(list)
        by_snr_pos = defaultdict(list)

        for i in range(len(train_ds)):
            x, y, snr_i = train_ds[i]
            y_val = int(self._to_numpy_scalar(y))
            snr_val = int(self._to_numpy_scalar(snr_i))

            e = self.compute_energy(x, y_val, snr_val)

            if y_val == 0:
                by_snr_neg[snr_val].append(e)
            else:
                by_snr_pos[snr_val].append(e)

        all_snrs = sorted(set(list(by_snr_neg.keys()) + list(by_snr_pos.keys())))
        self.thresholds = {}
        self.train_stats = {}

        fitted_thresholds = []

        for snr in all_snrs:
            neg_scores = by_snr_neg[snr]
            pos_scores = by_snr_pos[snr]

            if len(neg_scores) == 0 or len(pos_scores) == 0:
                if verbose:
                    print(f"[WARN] SNR={snr}: missing pos or neg samples, skipped.")
                continue

            result = self._fit_one_snr(neg_scores, pos_scores, target_pfa=target_pfa)
            self.thresholds[snr] = result["threshold"]
            self.train_stats[snr] = result
            fitted_thresholds.append(result["threshold"])

            if verbose:
                print(
                    f"[FIT] SNR={snr:>3} dB | thr={result['threshold']:.6f} | "
                    f"Pd={result['Pd']:.4f} | Pfa={result['Pfa']:.4f} | "
                    f"BA={result['balanced_acc']:.4f}"
                )

        if len(fitted_thresholds) == 0:
            raise RuntimeError("No thresholds were fitted. Check your dataset.")

        self.default_threshold = float(np.median(fitted_thresholds))
        if verbose:
            print(f"\nDefault threshold (median over SNRs) = {self.default_threshold:.6f}")

    def predict_one(self, x, y, snr_db):
        """
        Predict using threshold for the sample's SNR.
        """
        e = self.compute_energy(x, y, snr_db)
        thr = self.thresholds.get(int(snr_db), self.default_threshold)
        pred = 1 if e >= thr else 0
        return pred, e, thr

    def evaluate_by_snr(self, test_ds, verbose=True):
        """
        Evaluate Pd/Pfa/Acc for each SNR.
        """
        results = {}

        counts = defaultdict(lambda: {"TP": 0, "FP": 0, "TN": 0, "FN": 0})

        for i in range(len(test_ds)):
            x, y, snr_i = test_ds[i]
            y_val = int(self._to_numpy_scalar(y))
            snr_val = int(self._to_numpy_scalar(snr_i))

            pred, e, thr = self.predict_one(x, y_val, snr_val)

            if y_val == 1 and pred == 1:
                counts[snr_val]["TP"] += 1
            elif y_val == 0 and pred == 1:
                counts[snr_val]["FP"] += 1
            elif y_val == 0 and pred == 0:
                counts[snr_val]["TN"] += 1
            elif y_val == 1 and pred == 0:
                counts[snr_val]["FN"] += 1

        for snr in sorted(counts.keys()):
            TP = counts[snr]["TP"]
            FP = counts[snr]["FP"]
            TN = counts[snr]["TN"]
            FN = counts[snr]["FN"]

            Pd = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            Pfa = FP / (FP + TN) if (FP + TN) > 0 else 0.0
            Acc = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0

            results[snr] = {
                "TP": TP, "FP": FP, "TN": TN, "FN": FN,
                "Pd": Pd, "Pfa": Pfa, "Acc": Acc,
                "threshold": self.thresholds.get(snr, self.default_threshold)
            }

            if verbose:
                print(
                    f"[TEST] SNR={snr:>3} dB | thr={results[snr]['threshold']:.6f} | "
                    f"Pd={Pd:.4f} | Pfa={Pfa:.4f} | Acc={Acc:.4f} | "
                    f"TP={TP} FP={FP} TN={TN} FN={FN}"
                )

        return results

    def plot_pd_vs_snr(self, results, title="Pd vs SNR (SNR-adaptive ED)"):
        snrs = sorted(results.keys())
        pds = [results[s]["Pd"] for s in snrs]

        plt.figure(figsize=(8, 5))
        plt.plot(snrs, pds, marker='o')
        plt.xlabel("SNR (dB)")
        plt.ylabel("Detection Probability $P_d$")
        plt.title(title)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.ylim([0, 1.05])
        plt.show()

    def plot_pfa_vs_snr(self, results, title="Pfa vs SNR (SNR-adaptive ED)"):
        snrs = sorted(results.keys())
        pfas = [results[s]["Pfa"] for s in snrs]

        plt.figure(figsize=(8, 5))
        plt.plot(snrs, pfas, marker='s')
        plt.xlabel("SNR (dB)")
        plt.ylabel("False Alarm Probability $P_{fa}$")
        plt.title(title)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.ylim([0, 1.05])
        plt.show()

if __name__ == "__main__":
    set_seed(42)

    pkl_path = "data/RML2016.10a_dict.pkl"

    X, y, snr, meta = build_signal_vs_noise_dataset(
        pkl_path=pkl_path,
        snr_filter=None,
        selected_mods=None,
        seed=42
    )

    X_train, y_train, snr_train, meta_train, X_test, y_test, snr_test, meta_test = stratified_split_binary(
        X, y, snr, meta, test_ratio=0.2, seed=42
    )

    train_ds = SignalNoiseDataset(X_train, y_train, snr_train)
    test_ds = SignalNoiseDataset(X_test, y_test, snr_test)

    ed = SNRAdaptiveEnergyDetector(
        noise_power=1.0,          # 你定义一个固定噪声功率
        statistic="sum",          # "sum" or "mean"
        thr_mode="balanced_acc"   # "balanced_acc" / "youden" / "target_pfa"
    )

    ed.fit(train_ds, target_pfa=0.1, verbose=True)

    results = ed.evaluate_by_snr(test_ds, verbose=True)

    ed.plot_pd_vs_snr(results, title="Pd vs SNR (Restored-Power ED)")
    ed.plot_pfa_vs_snr(results, title="Pfa vs SNR (Restored-Power ED)")