from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib
import numpy as np
import torch
from torch.utils.data import Subset

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from project.data import DataConfig, SignalNoiseDataset, build_datasets, default_rml2016a_path
from project.models import create_model
from project.run_cnr_sensenet_eval import (
    compute_metrics,
    compute_metrics_by_snr,
    plot_training_history,
    seed_everything,
    stratified_index_split_binary,
    write_csv,
    write_json,
)
from project.utils import resolve_path


DEFAULT_OUTPUT_DIR = resolve_path("results", "cnr_sensenet_robustness")
SMOKE_OUTPUT_DIR = resolve_path("results", "cnr_sensenet_robustness_smoke")
DEFAULT_PERTURBATIONS = ["extra_awgn", "phase_rotation", "gain_imbalance", "impulsive_noise"]
PERTURBATION_LABELS = {
    "clean": "Clean",
    "extra_awgn": "Extra AWGN",
    "phase_rotation": "Phase Rotation",
    "gain_imbalance": "Gain Imbalance",
    "impulsive_noise": "Impulsive Noise",
}
PERTURBATION_AXIS_LABELS = {
    "extra_awgn": "Noise Std",
    "phase_rotation": "Max Rotation (deg)",
    "gain_imbalance": "Gain Imbalance",
    "impulsive_noise": "Impulse Probability",
}
SUMMARY_METRICS = [
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "pd",
    "pfa",
    "roc_auc",
    "average_precision",
]
PLOT_METRICS = [
    ("balanced_accuracy", "Balanced Acc", "#1c7ed6", "o"),
    ("f1", "F1", "#2b8a3e", "s"),
    ("pd", "Pd", "#e8590c", "^"),
    ("pfa", "Pfa", "#c92a2a", "D"),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train CNR-SenseNet once and evaluate robustness under controlled test-time perturbations."
    )
    parser.add_argument("--pkl-path", type=Path, default=default_rml2016a_path())
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--noise-power", type=float, default=1.0)
    parser.add_argument("--snr-filter", nargs="*", type=int, default=None)
    parser.add_argument("--selected-mods", nargs="*", default=None)
    parser.add_argument("--cache-path", type=Path, default=None)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--energy-window", type=int, default=8)
    parser.add_argument("--aux-branch-type", choices=["diff", "autocorr"], default="autocorr")
    parser.add_argument("--autocorr-max-lag", type=int, default=8)
    parser.add_argument(
        "--threshold-mode",
        choices=["fixed", "balanced_acc", "youden", "target_pfa"],
        default="balanced_acc",
    )
    parser.add_argument("--target-pfa", type=float, default=0.1)
    parser.add_argument("--calibration-split", choices=["train", "val"], default="val")
    parser.add_argument("--snr-loss-weighting", choices=["none", "two_band"], default="two_band")
    parser.add_argument("--low-snr-cutoff", type=int, default=-10)
    parser.add_argument("--low-snr-positive-weight", type=float, default=3.0)
    parser.add_argument("--mid-snr-cutoff", type=int, default=-6)
    parser.add_argument("--mid-snr-positive-weight", type=float, default=2.0)
    parser.add_argument("--decision-threshold", type=float, default=0.5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-threads", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument(
        "--perturbations",
        nargs="*",
        choices=DEFAULT_PERTURBATIONS,
        default=DEFAULT_PERTURBATIONS,
    )
    parser.add_argument("--awgn-std-values", nargs="*", type=float, default=[0.05, 0.1, 0.2, 0.35])
    parser.add_argument("--phase-max-deg-values", nargs="*", type=float, default=[5.0, 15.0, 30.0, 45.0])
    parser.add_argument("--gain-imbalance-values", nargs="*", type=float, default=[0.05, 0.1, 0.2, 0.3])
    parser.add_argument("--impulse-prob-values", nargs="*", type=float, default=[0.01, 0.03, 0.05, 0.1])
    parser.add_argument("--impulse-scale", type=float, default=3.0)
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-prefix", default="cnr_sensenet_robustness")
    return parser


def apply_smoke_overrides(args) -> None:
    if not args.smoke:
        return
    args.epochs = 1
    args.patience = 1
    args.batch_size = min(int(args.batch_size), 128)
    args.max_train_samples = args.max_train_samples or 512
    args.max_val_samples = args.max_val_samples or 256
    args.max_test_samples = args.max_test_samples or 512
    args.awgn_std_values = args.awgn_std_values[:2] or [0.05]
    args.phase_max_deg_values = args.phase_max_deg_values[:2] or [5.0]
    args.gain_imbalance_values = args.gain_imbalance_values[:2] or [0.05]
    args.impulse_prob_values = args.impulse_prob_values[:2] or [0.01]
    if Path(args.output_dir) == DEFAULT_OUTPUT_DIR:
        args.output_dir = SMOKE_OUTPUT_DIR
    if args.output_prefix == "cnr_sensenet_robustness":
        args.output_prefix = "cnr_sensenet_robustness_smoke"


def metric_to_float(value) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def normalize_positive_values(values: list[float], field_name: str) -> list[float]:
    normalized: list[float] = []
    for raw in values:
        value = float(raw)
        if value <= 0.0:
            raise ValueError(f"{field_name} values must be greater than 0.")
        if value not in normalized:
            normalized.append(value)
    if not normalized:
        raise ValueError(f"{field_name} requires at least one value.")
    return sorted(normalized)


def tensor_to_numpy(values) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def stratified_subsample_indices(
    y: np.ndarray,
    snr: np.ndarray,
    max_samples: int | None,
    seed: int,
) -> np.ndarray:
    total = len(y)
    if max_samples is None or max_samples <= 0 or total <= max_samples:
        return np.arange(total, dtype=np.int64)

    rng = np.random.default_rng(seed)
    buckets: list[list[int]] = []
    for label in sorted(np.unique(y)):
        for snr_value in sorted(np.unique(snr)):
            bucket = np.where((y == label) & (snr == snr_value))[0]
            if bucket.size == 0:
                continue
            bucket = bucket.copy()
            rng.shuffle(bucket)
            buckets.append(bucket.tolist())

    selected: list[int] = []
    cursor = 0
    active_buckets = [bucket for bucket in buckets if bucket]
    while len(selected) < max_samples and active_buckets:
        bucket_idx = cursor % len(active_buckets)
        selected.append(active_buckets[bucket_idx].pop())
        if not active_buckets[bucket_idx]:
            active_buckets.pop(bucket_idx)
            if active_buckets:
                cursor %= len(active_buckets)
        else:
            cursor += 1

    selected_idx = np.asarray(selected, dtype=np.int64)
    rng.shuffle(selected_idx)
    return selected_idx


def subset_arrays(arrays: dict[str, np.ndarray | None], indices: np.ndarray) -> dict[str, np.ndarray | None]:
    return {
        key: None if value is None else np.asarray(value)[indices]
        for key, value in arrays.items()
    }


def maybe_limit_dataset(dataset, arrays: dict[str, np.ndarray | None], max_samples: int | None, seed: int):
    indices = stratified_subsample_indices(
        y=np.asarray(arrays["y"]),
        snr=np.asarray(arrays["snr"]),
        max_samples=max_samples,
        seed=seed,
    )
    if len(indices) == len(arrays["y"]):
        return dataset, arrays
    return Subset(dataset, indices.tolist()), subset_arrays(arrays, indices)


def make_dataset_from_arrays(arrays: dict[str, np.ndarray | None]) -> SignalNoiseDataset:
    return SignalNoiseDataset(
        arrays["X"],
        arrays["y"],
        arrays["snr"],
        mod=arrays.get("mod"),
        label_snr=arrays.get("label_snr"),
        sample_type=arrays.get("sample_type"),
    )

def reshape_interleaved_iq(x: np.ndarray) -> np.ndarray:
    samples = np.asarray(x, dtype=np.float32)
    if samples.ndim != 2 or samples.shape[1] % 2 != 0:
        raise ValueError("Expected flattened interleaved IQ samples with shape [N, 2*T].")
    return samples.reshape(samples.shape[0], samples.shape[1] // 2, 2)


def flatten_interleaved_iq(x_iq: np.ndarray) -> np.ndarray:
    return np.asarray(x_iq, dtype=np.float32).reshape(x_iq.shape[0], -1)


def apply_extra_awgn(x: np.ndarray, severity: float, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(loc=0.0, scale=severity, size=x.shape).astype(np.float32)
    return (np.asarray(x, dtype=np.float32) + noise).astype(np.float32, copy=False)


def apply_phase_rotation(x: np.ndarray, severity: float, rng: np.random.Generator) -> np.ndarray:
    x_iq = reshape_interleaved_iq(x)
    theta = np.deg2rad(rng.uniform(-severity, severity, size=(x_iq.shape[0], 1))).astype(np.float32)
    cos_theta = np.cos(theta).astype(np.float32)
    sin_theta = np.sin(theta).astype(np.float32)

    i_values = x_iq[:, :, 0]
    q_values = x_iq[:, :, 1]
    rotated_i = i_values * cos_theta - q_values * sin_theta
    rotated_q = i_values * sin_theta + q_values * cos_theta
    rotated = np.stack([rotated_i, rotated_q], axis=-1)
    return flatten_interleaved_iq(rotated)


def apply_gain_imbalance(x: np.ndarray, severity: float, rng: np.random.Generator) -> np.ndarray:
    x_iq = reshape_interleaved_iq(x)
    imbalance = rng.uniform(-severity, severity, size=(x_iq.shape[0], 1)).astype(np.float32)
    i_scale = (1.0 + imbalance).astype(np.float32)
    q_scale = (1.0 - imbalance).astype(np.float32)

    adjusted = np.empty_like(x_iq, dtype=np.float32)
    adjusted[:, :, 0] = x_iq[:, :, 0] * i_scale
    adjusted[:, :, 1] = x_iq[:, :, 1] * q_scale
    return flatten_interleaved_iq(adjusted)


def apply_impulsive_noise(
    x: np.ndarray,
    severity: float,
    rng: np.random.Generator,
    impulse_scale: float,
) -> np.ndarray:
    mask = rng.random(size=x.shape) < severity
    impulses = rng.normal(loc=0.0, scale=impulse_scale, size=x.shape).astype(np.float32)
    return (np.asarray(x, dtype=np.float32) + mask.astype(np.float32) * impulses).astype(np.float32, copy=False)


def apply_perturbation(
    perturbation_name: str,
    x: np.ndarray,
    severity: float,
    rng: np.random.Generator,
    args,
) -> np.ndarray:
    if perturbation_name == "extra_awgn":
        return apply_extra_awgn(x, severity, rng)
    if perturbation_name == "phase_rotation":
        return apply_phase_rotation(x, severity, rng)
    if perturbation_name == "gain_imbalance":
        return apply_gain_imbalance(x, severity, rng)
    if perturbation_name == "impulsive_noise":
        return apply_impulsive_noise(x, severity, rng, impulse_scale=args.impulse_scale)
    raise ValueError(f"Unsupported perturbation: {perturbation_name}")


def build_perturbation_specs(args) -> list[dict[str, float | str]]:
    specs: list[dict[str, float | str]] = []
    perturbations = list(args.perturbations or [])

    if "extra_awgn" in perturbations:
        for severity in normalize_positive_values(list(args.awgn_std_values), "--awgn-std-values"):
            specs.append(
                {
                    "name": "extra_awgn",
                    "label": PERTURBATION_LABELS["extra_awgn"],
                    "axis_label": PERTURBATION_AXIS_LABELS["extra_awgn"],
                    "severity": float(severity),
                    "severity_label": f"std={severity:.3f}",
                }
            )
    if "phase_rotation" in perturbations:
        for severity in normalize_positive_values(list(args.phase_max_deg_values), "--phase-max-deg-values"):
            specs.append(
                {
                    "name": "phase_rotation",
                    "label": PERTURBATION_LABELS["phase_rotation"],
                    "axis_label": PERTURBATION_AXIS_LABELS["phase_rotation"],
                    "severity": float(severity),
                    "severity_label": f"max={severity:.1f} deg",
                }
            )
    if "gain_imbalance" in perturbations:
        for severity in normalize_positive_values(list(args.gain_imbalance_values), "--gain-imbalance-values"):
            specs.append(
                {
                    "name": "gain_imbalance",
                    "label": PERTURBATION_LABELS["gain_imbalance"],
                    "axis_label": PERTURBATION_AXIS_LABELS["gain_imbalance"],
                    "severity": float(severity),
                    "severity_label": f"imbalance={severity:.3f}",
                }
            )
    if "impulsive_noise" in perturbations:
        for severity in normalize_positive_values(list(args.impulse_prob_values), "--impulse-prob-values"):
            specs.append(
                {
                    "name": "impulsive_noise",
                    "label": PERTURBATION_LABELS["impulsive_noise"],
                    "axis_label": PERTURBATION_AXIS_LABELS["impulsive_noise"],
                    "severity": float(severity),
                    "severity_label": f"prob={severity:.3f}",
                }
            )
    return specs


def build_model(args, signal_length: int):
    return create_model(
        "cnr_sensenet",
        signal_length=signal_length,
        energy_window=args.energy_window,
        dropout=args.dropout,
        aux_branch_type=args.aux_branch_type,
        autocorr_max_lag=args.autocorr_max_lag,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        weight_decay=args.weight_decay,
        threshold=args.decision_threshold,
        threshold_mode=args.threshold_mode,
        target_pfa=args.target_pfa,
        calibration_split=args.calibration_split,
        snr_loss_weighting=args.snr_loss_weighting,
        low_snr_cutoff=args.low_snr_cutoff,
        low_snr_positive_weight=args.low_snr_positive_weight,
        mid_snr_cutoff=args.mid_snr_cutoff,
        mid_snr_positive_weight=args.mid_snr_positive_weight,
        device=args.device,
    )


def resolve_eval_threshold(model, default_threshold: float) -> float:
    requested_threshold = None if getattr(model, "prefers_internal_threshold", False) else default_threshold
    return float(model.get_evaluation_threshold(requested_threshold))


def extract_training_history(model) -> dict[str, float | int | list[float] | None | bool]:
    history = getattr(model, "history", None)
    train_loss = list(getattr(history, "train_loss", []))
    val_loss = list(getattr(history, "val_loss", []))
    return {
        "train_loss": train_loss,
        "val_loss": val_loss,
        "epochs_ran": len(train_loss),
        "best_epoch": getattr(history, "best_epoch", None),
        "best_val_loss": getattr(history, "best_val_loss", None),
        "stopped_early": bool(getattr(history, "stopped_early", False)),
    }


def extract_calibration(model):
    fit_result = getattr(model, "fit_result", None)
    if fit_result is None:
        return None
    try:
        return asdict(fit_result)
    except TypeError:
        return None


def count_parameters(model) -> int:
    network = getattr(model, "model", None)
    if network is None:
        return 0
    return int(sum(parameter.numel() for parameter in network.parameters()))


def build_clean_spec() -> dict[str, float | str]:
    return {
        "name": "clean",
        "label": PERTURBATION_LABELS["clean"],
        "axis_label": "Clean",
        "severity": 0.0,
        "severity_label": "clean",
    }


def build_overall_row(
    spec: dict[str, float | str],
    metrics: dict[str, float | int | None],
    clean_metrics: dict[str, float | int | None],
    infer_seconds: float,
) -> dict[str, float | int | str]:
    row: dict[str, float | int | str] = {
        "perturbation": str(spec["name"]),
        "perturbation_label": str(spec["label"]),
        "severity": float(spec["severity"]),
        "severity_label": str(spec["severity_label"]),
        "axis_label": str(spec["axis_label"]),
        "infer_seconds": float(infer_seconds),
        "threshold": float(metrics["threshold"]),
        "count": int(metrics["count"]),
        "signal_count": int(metrics["signal_count"]),
        "noise_count": int(metrics["noise_count"]),
        "tn": int(metrics["tn"]),
        "fp": int(metrics["fp"]),
        "fn": int(metrics["fn"]),
        "tp": int(metrics["tp"]),
    }
    for metric_key in SUMMARY_METRICS:
        metric_value = metric_to_float(metrics.get(metric_key))
        clean_value = metric_to_float(clean_metrics.get(metric_key))
        row[metric_key] = metric_value
        row[f"delta_{metric_key}"] = (
            metric_value - clean_value
            if np.isfinite(metric_value) and np.isfinite(clean_value)
            else float("nan")
        )
    return row


def build_snr_rows(
    spec: dict[str, float | str],
    rows: list[dict[str, float | int | None]],
    clean_rows: list[dict[str, float | int | None]],
) -> list[dict[str, float | int | str]]:
    clean_map = {int(item["snr"]): item for item in clean_rows}
    output_rows: list[dict[str, float | int | str]] = []

    for item in rows:
        snr_value = int(item["snr"])
        clean_item = clean_map.get(snr_value, {})
        output_row: dict[str, float | int | str] = {
            "perturbation": str(spec["name"]),
            "perturbation_label": str(spec["label"]),
            "severity": float(spec["severity"]),
            "severity_label": str(spec["severity_label"]),
            "snr": snr_value,
            "threshold": float(item["threshold"]),
            "count": int(item["count"]),
            "signal_count": int(item["signal_count"]),
            "noise_count": int(item["noise_count"]),
            "tn": int(item["tn"]),
            "fp": int(item["fp"]),
            "fn": int(item["fn"]),
            "tp": int(item["tp"]),
        }
        for metric_key in SUMMARY_METRICS:
            metric_value = metric_to_float(item.get(metric_key))
            clean_value = metric_to_float(clean_item.get(metric_key))
            output_row[metric_key] = metric_value
            output_row[f"delta_{metric_key}"] = (
                metric_value - clean_value
                if np.isfinite(metric_value) and np.isfinite(clean_value)
                else float("nan")
            )
        output_rows.append(output_row)
    return output_rows


def plot_overall_robustness(rows: list[dict[str, float | int | str]], output_path: Path) -> None:
    perturbed_rows = [row for row in rows if row["perturbation"] != "clean"]
    if not perturbed_rows:
        return

    clean_row = next((row for row in rows if row["perturbation"] == "clean"), None)
    perturbation_names: list[str] = []
    for row in perturbed_rows:
        name = str(row["perturbation"])
        if name not in perturbation_names:
            perturbation_names.append(name)

    nrows = len(perturbation_names)
    fig, axes = plt.subplots(nrows, 1, figsize=(10, max(4.5, 3.8 * nrows)), squeeze=False)
    axes_list = axes.flatten().tolist()

    for ax, perturbation_name in zip(axes_list, perturbation_names):
        series_rows = sorted(
            [row for row in perturbed_rows if row["perturbation"] == perturbation_name],
            key=lambda row: float(row["severity"]),
        )
        x_values = np.asarray([float(row["severity"]) for row in series_rows], dtype=np.float64)

        for metric_key, metric_label, color, marker in PLOT_METRICS:
            y_values = np.asarray([metric_to_float(row[metric_key]) for row in series_rows], dtype=np.float64)
            ax.plot(
                x_values,
                y_values,
                marker=marker,
                markersize=6,
                linewidth=2.0,
                color=color,
                label=metric_label,
            )
            if clean_row is not None:
                clean_value = metric_to_float(clean_row.get(metric_key))
                if np.isfinite(clean_value):
                    ax.axhline(clean_value, linestyle="--", linewidth=1.1, color=color, alpha=0.3)

        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel(str(series_rows[0]["axis_label"]))
        ax.set_ylabel("Score")
        ax.set_title(f"{series_rows[0]['perturbation_label']} Robustness")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(ncol=4, fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_worst_case_snr(rows: list[dict[str, float | int | str]], output_path: Path) -> None:
    perturbed_rows = [row for row in rows if row["perturbation"] != "clean"]
    if not perturbed_rows:
        return

    perturbation_names: list[str] = []
    for row in perturbed_rows:
        name = str(row["perturbation"])
        if name not in perturbation_names:
            perturbation_names.append(name)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for perturbation_name in perturbation_names:
        series_rows = [row for row in perturbed_rows if row["perturbation"] == perturbation_name]
        if not series_rows:
            continue
        max_severity = max(float(row["severity"]) for row in series_rows)
        worst_rows = sorted(
            [row for row in series_rows if float(row["severity"]) == max_severity],
            key=lambda row: int(row["snr"]),
        )
        if not worst_rows:
            continue

        snr_values = np.asarray([int(row["snr"]) for row in worst_rows], dtype=np.int64)
        label = f"{worst_rows[0]['perturbation_label']} ({worst_rows[0]['severity_label']})"
        bal_acc_delta = np.asarray(
            [metric_to_float(row["delta_balanced_accuracy"]) for row in worst_rows],
            dtype=np.float64,
        )
        pfa_delta = np.asarray(
            [metric_to_float(row["delta_pfa"]) for row in worst_rows],
            dtype=np.float64,
        )

        axes[0].plot(snr_values, bal_acc_delta, marker="o", linewidth=2.0, label=label)
        axes[1].plot(snr_values, pfa_delta, marker="s", linewidth=2.0, label=label)

    axes[0].axhline(0.0, linestyle="--", color="#868e96", linewidth=1.0)
    axes[1].axhline(0.0, linestyle="--", color="#868e96", linewidth=1.0)
    axes[0].set_ylabel("Delta Balanced Acc")
    axes[1].set_ylabel("Delta Pfa")
    axes[1].set_xlabel("Source SNR (dB)")
    axes[0].set_title("Worst-Severity Robustness Drop by SNR")
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[1].grid(True, linestyle="--", alpha=0.35)
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def maybe_save_checkpoint(model, output_path: Path, args, input_dim: int, clean_metrics: dict) -> None:
    if not args.save_checkpoint:
        return

    payload = {
        "model_name": "cnr_sensenet",
        "config": {
            "signal_length": int(input_dim),
            "energy_window": args.energy_window,
            "dropout": args.dropout,
            "aux_branch_type": args.aux_branch_type,
            "autocorr_max_lag": args.autocorr_max_lag,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "patience": args.patience,
            "weight_decay": args.weight_decay,
            "threshold": args.decision_threshold,
            "threshold_mode": args.threshold_mode,
            "target_pfa": args.target_pfa,
            "calibration_split": args.calibration_split,
            "snr_loss_weighting": args.snr_loss_weighting,
            "low_snr_cutoff": args.low_snr_cutoff,
            "low_snr_positive_weight": args.low_snr_positive_weight,
            "mid_snr_cutoff": args.mid_snr_cutoff,
            "mid_snr_positive_weight": args.mid_snr_positive_weight,
            "decision_threshold": clean_metrics["threshold"],
        },
        "metrics": clean_metrics,
        "state_dict": model.state_dict(),
    }
    torch.save(payload, output_path)


def main() -> None:
    args = build_parser().parse_args()
    apply_smoke_overrides(args)
    seed_everything(args.seed)
    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_start = time.perf_counter()
    data_config = DataConfig(
        pkl_path=args.pkl_path,
        test_ratio=args.test_ratio,
        seed=args.seed,
        noise_power=args.noise_power,
        snr_filter=args.snr_filter,
        selected_mods=args.selected_mods,
        cache_path=args.cache_path,
        use_cache=True,
        force_rebuild=args.force_rebuild,
    )
    bundle = build_datasets(data_config)

    if args.val_ratio > 0.0:
        train_idx, val_idx = stratified_index_split_binary(
            y=bundle.train_arrays["y"],
            snr=bundle.train_arrays["snr"],
            test_ratio=args.val_ratio,
            seed=args.seed,
        )
        train_dataset = Subset(bundle.train_dataset, train_idx.tolist())
        val_dataset = Subset(bundle.train_dataset, val_idx.tolist())
        train_arrays = subset_arrays(bundle.train_arrays, train_idx)
        val_arrays = subset_arrays(bundle.train_arrays, val_idx)
    else:
        train_dataset = bundle.train_dataset
        val_dataset = None
        train_arrays = dict(bundle.train_arrays)
        val_arrays = None

    test_dataset = bundle.test_dataset
    test_arrays = dict(bundle.test_arrays)

    train_dataset, train_arrays = maybe_limit_dataset(
        train_dataset,
        train_arrays,
        args.max_train_samples,
        seed=args.seed,
    )
    if val_dataset is not None and val_arrays is not None:
        val_dataset, val_arrays = maybe_limit_dataset(
            val_dataset,
            val_arrays,
            args.max_val_samples,
            seed=args.seed + 1,
        )
    test_dataset, test_arrays = maybe_limit_dataset(
        test_dataset,
        test_arrays,
        args.max_test_samples,
        seed=args.seed + 2,
    )

    model = build_model(args, signal_length=bundle.input_dim)

    train_start = time.perf_counter()
    model.fit(
        train_dataset,
        val_dataset=val_dataset,
        patience=args.patience,
        threshold_mode=args.threshold_mode,
        target_pfa=args.target_pfa,
        calibration_split=args.calibration_split,
        verbose=args.verbose,
    )
    train_seconds = time.perf_counter() - train_start

    eval_threshold = resolve_eval_threshold(model, args.decision_threshold)

    clean_spec = build_clean_spec()
    clean_eval_start = time.perf_counter()
    clean_scores = np.asarray(model.predict_scores(test_dataset), dtype=np.float64)
    clean_infer_seconds = time.perf_counter() - clean_eval_start
    clean_metrics, _ = compute_metrics(test_arrays["y"], clean_scores, threshold=eval_threshold)
    clean_snr_rows = compute_metrics_by_snr(
        test_arrays["y"],
        clean_scores,
        test_arrays["snr"],
        threshold=eval_threshold,
    )

    overall_rows = [build_overall_row(clean_spec, clean_metrics, clean_metrics, clean_infer_seconds)]
    by_snr_rows = build_snr_rows(clean_spec, clean_snr_rows, clean_snr_rows)

    perturbation_specs = build_perturbation_specs(args)
    for spec_index, spec in enumerate(perturbation_specs):
        rng = np.random.default_rng(args.seed + 1000 + spec_index)
        perturbed_arrays = dict(test_arrays)
        perturbed_arrays["X"] = apply_perturbation(
            str(spec["name"]),
            np.asarray(test_arrays["X"], dtype=np.float32),
            float(spec["severity"]),
            rng,
            args,
        )
        perturbed_dataset = make_dataset_from_arrays(perturbed_arrays)

        infer_start = time.perf_counter()
        perturbed_scores = np.asarray(model.predict_scores(perturbed_dataset), dtype=np.float64)
        infer_seconds = time.perf_counter() - infer_start
        perturbed_metrics, _ = compute_metrics(
            perturbed_arrays["y"],
            perturbed_scores,
            threshold=eval_threshold,
        )
        perturbed_snr_rows = compute_metrics_by_snr(
            perturbed_arrays["y"],
            perturbed_scores,
            perturbed_arrays["snr"],
            threshold=eval_threshold,
        )

        overall_rows.append(build_overall_row(spec, perturbed_metrics, clean_metrics, infer_seconds))
        by_snr_rows.extend(build_snr_rows(spec, perturbed_snr_rows, clean_snr_rows))

        if args.verbose:
            print(
                f"[{spec['label']:<16}] {spec['severity_label']:<18} "
                f"bal_acc={perturbed_metrics['balanced_accuracy']:.4f} "
                f"f1={perturbed_metrics['f1']:.4f} "
                f"pd={perturbed_metrics['pd']:.4f} "
                f"pfa={perturbed_metrics['pfa']:.4f}"
            )

    training_history = extract_training_history(model)
    calibration = extract_calibration(model)
    parameter_count = count_parameters(model)
    run_seconds = time.perf_counter() - run_start

    prefix = args.output_prefix
    table_robustness_1_path = output_dir / f"{prefix}_table_robustness_1_overall.csv"
    table_robustness_2_path = output_dir / f"{prefix}_table_robustness_2_by_snr.csv"
    summary_json_path = output_dir / f"{prefix}_summary.json"
    diagnostic_training_path = output_dir / f"{prefix}_diagnostic_training.png"
    figure_robustness_1_path = output_dir / f"{prefix}_figure_robustness_1_main.png"
    figure_robustness_2_path = output_dir / f"{prefix}_figure_robustness_2_worst_case_snr.png"
    checkpoint_path = output_dir / f"{prefix}_checkpoint.pt"

    write_csv(table_robustness_1_path, overall_rows)
    write_csv(table_robustness_2_path, by_snr_rows)
    plot_training_history(getattr(model, "history", None), diagnostic_training_path)
    plot_overall_robustness(overall_rows, figure_robustness_1_path)
    plot_worst_case_snr(by_snr_rows, figure_robustness_2_path)
    maybe_save_checkpoint(model, checkpoint_path, args, bundle.input_dim, clean_metrics)

    summary = {
        "dataset": {
            "pkl_path": str(Path(bundle.pkl_path).resolve()),
            "cache_path": str(Path(bundle.cache_path).resolve()),
            "input_dim": int(bundle.input_dim),
            "noise_power": float(bundle.noise_power),
            "train_size": int(len(train_dataset)),
            "val_size": int(len(val_dataset)) if val_dataset is not None else 0,
            "test_size": int(len(test_dataset)),
            "snrs": [int(value) for value in bundle.snrs],
            "mods": list(bundle.mods),
        },
        "model": {
            "name": "cnr_sensenet",
            "parameter_count": int(parameter_count),
            "aux_branch_type": args.aux_branch_type,
            "autocorr_max_lag": int(args.autocorr_max_lag),
            "decision_threshold": float(eval_threshold),
            "threshold_mode": args.threshold_mode,
            "calibration": calibration,
        },
        "training": {
            "seconds": float(train_seconds),
            "history": training_history,
        },
        "perturbations": [
            {
                "name": str(spec["name"]),
                "label": str(spec["label"]),
                "severity": float(spec["severity"]),
                "severity_label": str(spec["severity_label"]),
            }
            for spec in perturbation_specs
        ],
        "artifacts": {
            "table_robustness_1_overall": str(table_robustness_1_path.resolve()),
            "table_robustness_2_by_snr": str(table_robustness_2_path.resolve()),
            "diagnostic_training_plot": str(diagnostic_training_path.resolve()),
            "figure_robustness_1_main": str(figure_robustness_1_path.resolve()),
            "figure_robustness_2_worst_case_snr": str(figure_robustness_2_path.resolve()),
            "checkpoint": str(checkpoint_path.resolve()) if args.save_checkpoint else None,
        },
        "paper_reference_map": {
            "table_robustness_1": {
                "path": str(table_robustness_1_path.resolve()),
                "description": "Overall robustness summary across perturbations.",
            },
            "table_robustness_2": {
                "path": str(table_robustness_2_path.resolve()),
                "description": "Per-SNR robustness breakdown across perturbations.",
            },
            "figure_robustness_1": {
                "path": str(figure_robustness_1_path.resolve()),
                "description": "Main robustness curves across perturbation severity.",
            },
            "figure_robustness_2": {
                "path": str(figure_robustness_2_path.resolve()),
                "description": "Worst-severity robustness drop by source SNR.",
            },
        },
        "results": {
            "clean_metrics": clean_metrics,
            "clean_metrics_by_snr": clean_snr_rows,
            "overall_rows": overall_rows,
            "by_snr_rows": by_snr_rows,
        },
        "runtime": {
            "seed": int(args.seed),
            "smoke": bool(args.smoke),
            "total_seconds": float(run_seconds),
        },
    }
    write_json(summary_json_path, summary)

    print(f"Clean: bal_acc={clean_metrics['balanced_accuracy']:.4f} f1={clean_metrics['f1']:.4f} pd={clean_metrics['pd']:.4f} pfa={clean_metrics['pfa']:.4f}")
    print(f"Saved summary: {summary_json_path}")
    print(f"Saved Table R1: {table_robustness_1_path}")
    print(f"Saved Table R2: {table_robustness_2_path}")
    print(f"Saved Figure R1: {figure_robustness_1_path}")
    print(f"Saved Figure R2: {figure_robustness_2_path}")


if __name__ == "__main__":
    main()


