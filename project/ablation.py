from __future__ import annotations

import argparse
import csv
import json
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

from project.data import DataConfig, build_datasets, default_rml2016a_path
from project.models import create_model
from project.run_cnr_sensenet_eval import (
    compute_metrics,
    compute_metrics_by_snr,
    seed_everything,
    stratified_index_split_binary,
)
from project.utils import resolve_path


DEFAULT_OUTPUT_DIR = resolve_path("results", "cnr_sensenet_ablation")
SMOKE_OUTPUT_DIR = resolve_path("results", "cnr_sensenet_ablation_smoke")
DEFAULT_SEEDS = [42, 43, 44]

VARIANT_SPECS = [
    {
        "name": "full",
        "label": "Full",
        "color": "#0b7285",
        "linestyle": "-",
        "use_raw_branch": True,
        "use_energy_branch": True,
        "use_diff_branch": True,
    },
    {
        "name": "wo_raw",
        "label": "w/o Raw",
        "color": "#e8590c",
        "linestyle": "--",
        "use_raw_branch": False,
        "use_energy_branch": True,
        "use_diff_branch": True,
    },
    {
        "name": "wo_energy",
        "label": "w/o Energy",
        "color": "#2b8a3e",
        "linestyle": "-.",
        "use_raw_branch": True,
        "use_energy_branch": False,
        "use_diff_branch": True,
    },
    {
        "name": "wo_diff",
        "label": "w/o Aux",
        "color": "#c92a2a",
        "linestyle": ":",
        "use_raw_branch": True,
        "use_energy_branch": True,
        "use_diff_branch": False,
    },
]

OVERALL_CURVE_METRICS = [
    ("balanced_accuracy", "Balanced Acc", "#1c7ed6", "o"),
    ("f1", "F1", "#2b8a3e", "s"),
    ("roc_auc", "ROC-AUC", "#e8590c", "^"),
    ("pd", "Pd", "#9c36b5", "D"),
]
SNR_CURVE_METRICS = [
    ("balanced_accuracy", "Balanced Acc"),
    ("f1", "F1"),
    ("pd", "Pd"),
    ("pfa", "Pfa"),
]
DROP_METRICS = [
    ("balanced_accuracy", "ΔBal Acc", "#1c7ed6"),
    ("f1", "ΔF1", "#2b8a3e"),
    ("roc_auc", "ΔROC-AUC", "#e8590c"),
    ("pd", "ΔPd", "#9c36b5"),
    ("pfa", "ΔPfa", "#c92a2a"),
]
SUMMARY_METRICS = [
    "accuracy",
    "balanced_accuracy",
    "precision",
    "pd",
    "f1",
    "pfa",
    "roc_auc",
    "average_precision",
    "threshold",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run branch ablation studies for CNR-SenseNet.")
    parser.add_argument("--pkl-path", type=Path, default=default_rml2016a_path())
    parser.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--noise-power", type=float, default=1.0)
    parser.add_argument("--snr-filter", nargs="*", type=int, default=None)
    parser.add_argument("--selected-mods", nargs="*", default=None)
    parser.add_argument("--cache-path", type=Path, default=None)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--epochs", type=int, default=30)
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
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-prefix", default="cnr_sensenet_ablation")
    parser.add_argument("--verbose", action="store_true")
    return parser


def normalize_seeds(seeds: list[int]) -> list[int]:
    normalized: list[int] = []
    for seed in seeds:
        value = int(seed)
        if value not in normalized:
            normalized.append(value)
    if not normalized:
        raise ValueError("At least one seed must be provided.")
    return normalized


def apply_smoke_overrides(args) -> None:
    if not args.smoke:
        return

    args.epochs = 1
    args.patience = 1
    args.batch_size = min(int(args.batch_size), 128)
    if args.seeds == DEFAULT_SEEDS:
        args.seeds = [DEFAULT_SEEDS[0]]
    args.max_train_samples = args.max_train_samples or 512
    args.max_val_samples = args.max_val_samples or 256
    args.max_test_samples = args.max_test_samples or 512
    if Path(args.output_dir) == DEFAULT_OUTPUT_DIR:
        args.output_dir = SMOKE_OUTPUT_DIR
    if args.output_prefix == "cnr_sensenet_ablation":
        args.output_prefix = "cnr_sensenet_ablation_smoke"


def metric_to_float(value) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or np.all(np.isnan(array)):
        return float("nan"), float("nan")
    return float(np.nanmean(array)), float(np.nanstd(array))


def branch_signature(spec: dict, aux_branch_type: str = "diff") -> str:
    names: list[str] = []
    if spec["use_raw_branch"]:
        names.append("Raw")
    if spec["use_energy_branch"]:
        names.append("Energy")
    if spec["use_diff_branch"]:
        names.append("Autocorr" if aux_branch_type == "autocorr" else "Diff")
    return " + ".join(names)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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


def subset_arrays(arrays: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, np.ndarray]:
    return {key: np.asarray(value)[indices] for key, value in arrays.items()}


def maybe_limit_dataset(dataset, arrays: dict[str, np.ndarray], max_samples: int | None, seed: int):
    indices = stratified_subsample_indices(
        y=np.asarray(arrays["y"]),
        snr=np.asarray(arrays["snr"]),
        max_samples=max_samples,
        seed=seed,
    )
    if len(indices) == len(arrays["y"]):
        return dataset, arrays
    return Subset(dataset, indices.tolist()), subset_arrays(arrays, indices)


def build_variant_model(args, signal_length: int, spec: dict):
    return create_model(
        "cnr_sensenet",
        signal_length=signal_length,
        energy_window=args.energy_window,
        dropout=args.dropout,
        aux_branch_type=args.aux_branch_type,
        autocorr_max_lag=args.autocorr_max_lag,
        use_raw_branch=spec["use_raw_branch"],
        use_energy_branch=spec["use_energy_branch"],
        use_diff_branch=spec["use_diff_branch"],
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


def count_parameters(model) -> int:
    network = getattr(model, "model", None)
    if network is None:
        return 0
    return int(sum(parameter.numel() for parameter in network.parameters()))


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

def compute_metrics_by_band(
    y_true: np.ndarray,
    scores: np.ndarray,
    snr_values: np.ndarray,
    threshold: float,
    low_snr_cutoff: int,
    mid_snr_cutoff: int,
) -> list[dict[str, float | int | None | str]]:
    rows: list[dict[str, float | int | None | str]] = []
    band_specs = [
        ("low", f"<= {low_snr_cutoff} dB", snr_values <= low_snr_cutoff),
        (
            "mid",
            f"({low_snr_cutoff}, {mid_snr_cutoff}] dB",
            (snr_values > low_snr_cutoff) & (snr_values <= mid_snr_cutoff),
        ),
        ("high", f"> {mid_snr_cutoff} dB", snr_values > mid_snr_cutoff),
    ]
    for band_order, (band_name, band_label, mask) in enumerate(band_specs):
        if not np.any(mask):
            continue
        metrics, _ = compute_metrics(y_true[mask], scores[mask], threshold=threshold)
        metrics["snr_band"] = band_name
        metrics["snr_band_label"] = band_label
        metrics["snr_band_order"] = int(band_order)
        rows.append(metrics)
    return rows


def build_seed_rows(results: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for result in results:
        metrics = result["overall_metrics"]
        rows.append(
            {
                "seed": int(result["seed"]),
                "variant": result["variant"],
                "variant_label": result["variant_label"],
                "branches": result["branch_signature"],
                "parameter_count": int(result["parameter_count"]),
                "train_seconds": float(result["train_seconds"]),
                "infer_seconds": float(result["infer_seconds"]),
                "threshold": float(metrics["threshold"]),
                "accuracy": float(metrics["accuracy"]),
                "balanced_accuracy": float(metrics["balanced_accuracy"]),
                "precision": float(metrics["precision"]),
                "pd": float(metrics["pd"]),
                "f1": float(metrics["f1"]),
                "pfa": float(metrics["pfa"]),
                "roc_auc": metric_to_float(metrics["roc_auc"]),
                "average_precision": metric_to_float(metrics["average_precision"]),
            }
        )
    return rows


def aggregate_overall_results(results: list[dict], aux_branch_type: str = "diff") -> list[dict]:
    rows: list[dict] = []
    for spec in VARIANT_SPECS:
        variant_runs = [item for item in results if item["variant"] == spec["name"]]
        if not variant_runs:
            continue

        row = {
            "variant": spec["name"],
            "variant_label": spec["label"],
            "branches": branch_signature(spec, aux_branch_type),
            "seeds": int(len(variant_runs)),
        }
        for metric_key in SUMMARY_METRICS:
            values = [metric_to_float(item["overall_metrics"].get(metric_key)) for item in variant_runs]
            mean_value, std_value = mean_std(values)
            row[f"{metric_key}_mean"] = mean_value
            row[f"{metric_key}_std"] = std_value

        for field_name in ["parameter_count", "train_seconds", "infer_seconds"]:
            values = [metric_to_float(item[field_name]) for item in variant_runs]
            mean_value, std_value = mean_std(values)
            row[f"{field_name}_mean"] = mean_value
            row[f"{field_name}_std"] = std_value
        rows.append(row)
    return rows


def aggregate_snr_results(results: list[dict], aux_branch_type: str = "diff") -> list[dict]:
    rows: list[dict] = []
    metric_keys = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "pd",
        "f1",
        "pfa",
        "roc_auc",
        "average_precision",
    ]
    for spec in VARIANT_SPECS:
        variant_runs = [item for item in results if item["variant"] == spec["name"]]
        if not variant_runs:
            continue
        snr_values = sorted({int(row["snr"]) for item in variant_runs for row in item["metrics_by_snr"]})
        for snr_value in snr_values:
            grouped = [
                row
                for item in variant_runs
                for row in item["metrics_by_snr"]
                if int(row["snr"]) == int(snr_value)
            ]
            if not grouped:
                continue
            row = {
                "variant": spec["name"],
                "variant_label": spec["label"],
                "branches": branch_signature(spec, aux_branch_type),
                "snr": int(snr_value),
                "seeds": int(len(grouped)),
            }
            for metric_key in metric_keys:
                values = [metric_to_float(item.get(metric_key)) for item in grouped]
                mean_value, std_value = mean_std(values)
                row[f"{metric_key}_mean"] = mean_value
                row[f"{metric_key}_std"] = std_value
            rows.append(row)
    return rows


def aggregate_band_results(results: list[dict], aux_branch_type: str = "diff") -> list[dict]:
    rows: list[dict] = []
    metric_keys = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "pd",
        "f1",
        "pfa",
        "roc_auc",
        "average_precision",
    ]
    for spec in VARIANT_SPECS:
        variant_runs = [item for item in results if item["variant"] == spec["name"]]
        if not variant_runs:
            continue
        band_values = sorted(
            {int(row["snr_band_order"]) for item in variant_runs for row in item["metrics_by_band"]}
        )
        for band_order in band_values:
            grouped = [
                row
                for item in variant_runs
                for row in item["metrics_by_band"]
                if int(row["snr_band_order"]) == int(band_order)
            ]
            if not grouped:
                continue
            row = {
                "variant": spec["name"],
                "variant_label": spec["label"],
                "branches": branch_signature(spec, aux_branch_type),
                "snr_band": grouped[0]["snr_band"],
                "snr_band_label": grouped[0]["snr_band_label"],
                "snr_band_order": int(band_order),
                "seeds": int(len(grouped)),
            }
            for metric_key in metric_keys:
                values = [metric_to_float(item.get(metric_key)) for item in grouped]
                mean_value, std_value = mean_std(values)
                row[f"{metric_key}_mean"] = mean_value
                row[f"{metric_key}_std"] = std_value
            rows.append(row)
    return rows

def plot_overall_metrics(rows: list[dict], output_path: Path) -> None:
    if not rows:
        return

    labels = [row["variant_label"] for row in rows]
    x = np.arange(len(labels), dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={"width_ratios": [4.2, 1.4]})

    for metric_key, metric_label, color, marker in OVERALL_CURVE_METRICS:
        means = np.asarray([row[f"{metric_key}_mean"] for row in rows], dtype=np.float64)
        stds = np.asarray([row[f"{metric_key}_std"] for row in rows], dtype=np.float64)
        axes[0].errorbar(
            x,
            means,
            yerr=stds,
            marker=marker,
            markersize=7,
            linewidth=2.2,
            capsize=4,
            color=color,
            label=metric_label,
        )

    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Overall Metrics Across Ablation Variants")
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].legend(ncol=2)

    pfa_means = np.asarray([row["pfa_mean"] for row in rows], dtype=np.float64)
    pfa_stds = np.asarray([row["pfa_std"] for row in rows], dtype=np.float64)
    axes[1].errorbar(
        x,
        pfa_means,
        yerr=pfa_stds,
        marker="o",
        markersize=7,
        linewidth=2.2,
        capsize=4,
        color="#c92a2a",
    )
    axes[1].set_xticks(x, labels)
    upper = np.nanmax(pfa_means + pfa_stds) if pfa_means.size > 0 else float("nan")
    axes[1].set_ylim(0.0, max(0.05, 1.15 * upper if np.isfinite(upper) else 0.05))
    axes[1].set_ylabel("Pfa")
    axes[1].set_title("False Alarm Rate (Lower Better)")
    axes[1].grid(True, linestyle="--", alpha=0.35)

    for axis in axes:
        axis.tick_params(axis="x", rotation=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_metric_drop_vs_full(rows: list[dict], output_path: Path) -> None:
    full_row = next((row for row in rows if row["variant"] == "full"), None)
    comparison_rows = [row for row in rows if row["variant"] != "full"]
    if full_row is None or not comparison_rows:
        return

    x = np.arange(len(comparison_rows), dtype=np.float64)
    width = 0.15
    fig, ax = plt.subplots(figsize=(11, 5))

    for idx, (metric_key, metric_label, color) in enumerate(DROP_METRICS):
        deltas = [row[f"{metric_key}_mean"] - full_row[f"{metric_key}_mean"] for row in comparison_rows]
        offset = x + (idx - (len(DROP_METRICS) - 1) / 2.0) * width
        ax.bar(offset, deltas, width=width, label=metric_label, color=color)

    ax.axhline(0.0, color="#495057", linewidth=1)
    ax.set_xticks(x, [row["variant_label"] for row in comparison_rows])
    ax.set_ylabel("Mean delta vs Full")
    ax.set_title("Performance Change Relative to Full Model")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_snr_overlay_curves(rows: list[dict], output_path: Path) -> None:
    if not rows:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes = np.asarray(axes).reshape(-1)

    for axis, (metric_key, metric_label) in zip(axes, SNR_CURVE_METRICS):
        for spec in VARIANT_SPECS:
            variant_rows = sorted(
                [row for row in rows if row["variant"] == spec["name"]],
                key=lambda item: int(item["snr"]),
            )
            if not variant_rows:
                continue
            snrs = np.asarray([row["snr"] for row in variant_rows], dtype=np.float64)
            means = np.asarray([row[f"{metric_key}_mean"] for row in variant_rows], dtype=np.float64)
            stds = np.asarray([row[f"{metric_key}_std"] for row in variant_rows], dtype=np.float64)
            lower = np.clip(means - stds, 0.0, 1.0)
            upper = np.clip(means + stds, 0.0, 1.0)

            axis.plot(
                snrs,
                means,
                marker="o",
                markersize=6,
                linewidth=2.1,
                color=spec["color"],
                linestyle=spec["linestyle"],
                label=spec["label"],
            )
            axis.fill_between(snrs, lower, upper, color=spec["color"], alpha=0.12)

        axis.set_title(f"{metric_label} by Source SNR")
        axis.set_ylabel(metric_label)
        axis.grid(True, linestyle="--", alpha=0.35)
        if metric_key == "pfa":
            pfa_max = np.nanmax(
                [row[f"{metric_key}_mean"] + row[f"{metric_key}_std"] for row in rows],
                initial=np.nan,
            )
            axis.set_ylim(0.0, max(0.05, 1.15 * pfa_max if np.isfinite(pfa_max) else 0.05))
        else:
            axis.set_ylim(0.0, 1.05)

    axes[2].set_xlabel("Source SNR (dB)")
    axes[3].set_xlabel("Source SNR (dB)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.96])
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

def main() -> None:
    args = build_parser().parse_args()
    args.seeds = normalize_seeds(args.seeds)
    apply_smoke_overrides(args)
    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    dataset_summaries: list[dict] = []

    for seed in args.seeds:
        seed_everything(seed)
        data_config = DataConfig(
            pkl_path=args.pkl_path,
            test_ratio=args.test_ratio,
            seed=seed,
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
                seed=seed,
            )
            train_dataset = Subset(bundle.train_dataset, train_idx.tolist())
            val_dataset = Subset(bundle.train_dataset, val_idx.tolist())
            train_arrays = subset_arrays(bundle.train_arrays, train_idx)
            val_arrays = subset_arrays(bundle.train_arrays, val_idx)
        else:
            train_dataset = bundle.train_dataset
            val_dataset = None
            train_arrays = {key: np.asarray(value) for key, value in bundle.train_arrays.items()}
            val_arrays = None

        train_dataset, train_arrays = maybe_limit_dataset(
            train_dataset,
            train_arrays,
            max_samples=args.max_train_samples,
            seed=seed,
        )
        if val_dataset is not None and val_arrays is not None:
            val_dataset, val_arrays = maybe_limit_dataset(
                val_dataset,
                val_arrays,
                max_samples=args.max_val_samples,
                seed=seed + 1,
            )
        test_dataset, test_arrays = maybe_limit_dataset(
            bundle.test_dataset,
            {key: np.asarray(value) for key, value in bundle.test_arrays.items()},
            max_samples=args.max_test_samples,
            seed=seed + 2,
        )

        dataset_summaries.append(
            {
                "seed": int(seed),
                "cache_path": str(bundle.cache_path),
                "input_dim": int(bundle.input_dim),
                "train_size": int(len(train_dataset)),
                "val_size": int(len(val_dataset)) if val_dataset is not None else 0,
                "test_size": int(len(test_dataset)),
                "snrs": [int(value) for value in bundle.snrs],
                "mods": list(bundle.mods),
            }
        )

        run_dir = output_dir / "runs" / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)

        for spec in VARIANT_SPECS:
            print(f"\n=== Seed {seed} | {spec['label']} ({branch_signature(spec, args.aux_branch_type)}) ===")
            model = build_variant_model(args, signal_length=bundle.input_dim, spec=spec)

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

            infer_start = time.perf_counter()
            test_scores = np.asarray(model.predict_scores(test_dataset), dtype=np.float64)
            infer_seconds = time.perf_counter() - infer_start
            eval_threshold = resolve_eval_threshold(model, args.decision_threshold)

            overall_metrics, _ = compute_metrics(test_arrays["y"], test_scores, threshold=eval_threshold)
            metrics_by_snr = compute_metrics_by_snr(
                test_arrays["y"],
                test_scores,
                test_arrays["snr"],
                threshold=eval_threshold,
            )
            metrics_by_band = compute_metrics_by_band(
                test_arrays["y"],
                test_scores,
                test_arrays["snr"],
                threshold=eval_threshold,
                low_snr_cutoff=args.low_snr_cutoff,
                mid_snr_cutoff=args.mid_snr_cutoff,
            )

            result = {
                "seed": int(seed),
                "variant": spec["name"],
                "variant_label": spec["label"],
                "branch_signature": branch_signature(spec, args.aux_branch_type),
                "branches": {
                    "use_raw_branch": bool(spec["use_raw_branch"]),
                    "use_energy_branch": bool(spec["use_energy_branch"]),
                    "use_diff_branch": bool(spec["use_diff_branch"]),
                    "aux_branch_type": args.aux_branch_type,
                    "autocorr_max_lag": int(args.autocorr_max_lag),
                },
                "parameter_count": int(count_parameters(model)),
                "train_seconds": float(train_seconds),
                "infer_seconds": float(infer_seconds),
                "training_history": extract_training_history(model),
                "calibration": None if getattr(model, "fit_result", None) is None else asdict(model.fit_result),
                "overall_metrics": overall_metrics,
                "metrics_by_snr": metrics_by_snr,
                "metrics_by_band": metrics_by_band,
                "dataset": {
                    "train_size": int(len(train_dataset)),
                    "val_size": int(len(val_dataset)) if val_dataset is not None else 0,
                    "test_size": int(len(test_dataset)),
                    "input_dim": int(bundle.input_dim),
                    "cache_path": str(bundle.cache_path),
                },
            }
            results.append(result)
            write_json(run_dir / f"{spec['name']}_summary.json", result)

            print(
                f"{spec['label']} | "
                f"bal_acc={overall_metrics['balanced_accuracy']:.4f} "
                f"f1={overall_metrics['f1']:.4f} "
                f"roc_auc={metric_to_float(overall_metrics['roc_auc']):.4f} "
                f"pd={overall_metrics['pd']:.4f} "
                f"pfa={overall_metrics['pfa']:.4f} "
                f"thr={overall_metrics['threshold']:.4f} "
                f"train={train_seconds:.2f}s"
            )

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    seed_rows = build_seed_rows(results)
    summary_rows = aggregate_overall_results(results, args.aux_branch_type)
    snr_rows = aggregate_snr_results(results, args.aux_branch_type)
    band_rows = aggregate_band_results(results, args.aux_branch_type)

    artifact_paths = {
        "summary_json": output_dir / f"{args.output_prefix}_summary.json",
        "summary_csv": output_dir / f"{args.output_prefix}_summary.csv",
        "seed_csv": output_dir / f"{args.output_prefix}_seed_results.csv",
        "snr_csv": output_dir / f"{args.output_prefix}_metrics_by_snr.csv",
        "snr_band_csv": output_dir / f"{args.output_prefix}_snr_band_summary.csv",
        "overall_png": output_dir / f"{args.output_prefix}_overall_metrics.png",
        "drop_png": output_dir / f"{args.output_prefix}_metric_drop_vs_full.png",
        "snr_png": output_dir / f"{args.output_prefix}_snr_overlay_curves.png",
    }

    payload = {
        "config": {
            "seeds": [int(seed) for seed in args.seeds],
            "smoke": bool(args.smoke),
            "epochs": int(args.epochs),
            "patience": int(args.patience),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "dropout": float(args.dropout),
            "energy_window": int(args.energy_window),
            "aux_branch_type": args.aux_branch_type,
            "autocorr_max_lag": int(args.autocorr_max_lag),
            "decision_threshold": float(args.decision_threshold),
            "threshold_mode": args.threshold_mode,
            "target_pfa": float(args.target_pfa),
            "calibration_split": args.calibration_split,
            "snr_loss_weighting": args.snr_loss_weighting,
            "low_snr_cutoff": int(args.low_snr_cutoff),
            "low_snr_positive_weight": float(args.low_snr_positive_weight),
            "mid_snr_cutoff": int(args.mid_snr_cutoff),
            "mid_snr_positive_weight": float(args.mid_snr_positive_weight),
            "test_ratio": float(args.test_ratio),
            "val_ratio": float(args.val_ratio),
            "noise_power": float(args.noise_power),
            "device": args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
            "max_train_samples": args.max_train_samples,
            "max_val_samples": args.max_val_samples,
            "max_test_samples": args.max_test_samples,
        },
        "variants": [
            {
                "name": spec["name"],
                "label": spec["label"],
                "branches": branch_signature(spec, args.aux_branch_type),
                "use_raw_branch": bool(spec["use_raw_branch"]),
                "use_energy_branch": bool(spec["use_energy_branch"]),
                "use_diff_branch": bool(spec["use_diff_branch"]),
                "aux_branch_type": args.aux_branch_type,
                "autocorr_max_lag": int(args.autocorr_max_lag),
            }
            for spec in VARIANT_SPECS
        ],
        "dataset_by_seed": dataset_summaries,
        "results": results,
        "aggregated": {
            "overall": summary_rows,
            "by_snr": snr_rows,
            "by_snr_band": band_rows,
        },
        "artifacts": {key: str(value) for key, value in artifact_paths.items()},
    }

    write_json(artifact_paths["summary_json"], payload)
    write_csv(artifact_paths["summary_csv"], summary_rows)
    write_csv(artifact_paths["seed_csv"], seed_rows)
    write_csv(artifact_paths["snr_csv"], snr_rows)
    write_csv(artifact_paths["snr_band_csv"], band_rows)
    plot_overall_metrics(summary_rows, artifact_paths["overall_png"])
    plot_metric_drop_vs_full(summary_rows, artifact_paths["drop_png"])
    plot_snr_overlay_curves(snr_rows, artifact_paths["snr_png"])

    print(f"\nOutput directory: {output_dir}")
    for name, path in artifact_paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()

