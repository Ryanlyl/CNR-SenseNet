from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from project.data import DataConfig, SignalNoiseDataset, build_datasets, default_rml2016a_path
from project.models import create_model
from project.run_cnr_sensenet_eval import compute_metrics, seed_everything, stratified_index_split_binary
from project.utils import resolve_path

DEFAULT_OUTPUT_DIR = resolve_path("plots", "cnr_sensenet_explainability")
SMOKE_OUTPUT_DIR = resolve_path("plots", "cnr_sensenet_explainability_smoke")
BRANCH_NAMES = ("raw", "energy", "diff")
BRANCH_LABELS = {"raw": "Raw Branch", "energy": "Energy Branch", "diff": "Diff Branch"}
BRANCH_COLORS = {"raw": "#1c7ed6", "energy": "#e8590c", "diff": "#2b8a3e"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train CNR-SenseNet and generate three explainability figures.")
    parser.add_argument("--pkl-path", type=Path, default=default_rml2016a_path())
    parser.add_argument("--seed", type=int, default=42)
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
    parser.add_argument("--threshold-mode", choices=["fixed", "balanced_acc", "youden", "target_pfa"], default="balanced_acc")
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
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-prefix", default="cnr_sensenet_explainability")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--verbose", action="store_true")
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
    if Path(args.output_dir) == DEFAULT_OUTPUT_DIR:
        args.output_dir = SMOKE_OUTPUT_DIR
    if args.output_prefix == "cnr_sensenet_explainability":
        args.output_prefix = "cnr_sensenet_explainability_smoke"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def tensor_to_numpy(values) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def stratified_subsample_indices(y: np.ndarray, snr: np.ndarray, max_samples: int | None, seed: int) -> np.ndarray:
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


def dataset_arrays(dataset) -> dict[str, np.ndarray]:
    if isinstance(dataset, Subset):
        base = dataset.dataset
        if not isinstance(base, SignalNoiseDataset):
            raise TypeError("Expected Subset to wrap SignalNoiseDataset.")
        indices = np.asarray(dataset.indices, dtype=np.int64)
    elif isinstance(dataset, SignalNoiseDataset):
        base = dataset
        indices = np.arange(len(dataset), dtype=np.int64)
    else:
        raise TypeError("dataset_arrays only supports SignalNoiseDataset or Subset.")
    arrays = {
        "X": tensor_to_numpy(base.X)[indices].astype(np.float32, copy=False),
        "y": tensor_to_numpy(base.y)[indices].reshape(-1).astype(np.int64, copy=False),
        "snr": tensor_to_numpy(base.snr)[indices].reshape(-1).astype(np.int64, copy=False),
    }
    arrays["mod"] = None if base.mod is None else np.asarray(base.mod)[indices]
    arrays["sample_type"] = None if base.sample_type is None else np.asarray(base.sample_type)[indices]
    return arrays


def maybe_limit_dataset(dataset, max_samples: int | None, seed: int):
    if max_samples is None or max_samples <= 0 or len(dataset) <= max_samples:
        return dataset
    arrays = dataset_arrays(dataset)
    selected_idx = stratified_subsample_indices(arrays["y"], arrays["snr"], max_samples, seed)
    return Subset(dataset, selected_idx.tolist())


def build_branch_feature_dict(backbone, x: torch.Tensor) -> dict[str, torch.Tensor]:
    x_iq = backbone.reshape_iq(x)
    features: dict[str, torch.Tensor] = {}
    if backbone.use_raw_branch:
        features["raw"] = backbone.raw_branch(x_iq)
    if backbone.use_energy_branch:
        features["energy"] = backbone.energy_branch(backbone.compute_local_energy(x_iq))
    if backbone.use_diff_branch:
        features["diff"] = backbone.diff_branch(backbone.compute_diff(x_iq))
    return features


def classifier_from_branch_features(backbone, feature_dict: dict[str, torch.Tensor], masked_branch: str | None = None) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    for branch_name in BRANCH_NAMES:
        if branch_name not in feature_dict:
            continue
        feature = feature_dict[branch_name]
        if masked_branch == branch_name:
            feature = torch.zeros_like(feature)
        parts.append(feature)
    merged = parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)
    return backbone.classifier(merged).squeeze(-1)


def compute_branch_contribution_rows(model, arrays: dict[str, np.ndarray], threshold: float) -> tuple[list[dict], dict[str, np.ndarray]]:
    backbone = model.model
    if backbone is None:
        raise RuntimeError("Backbone is not initialized.")
    loader = DataLoader(TensorDataset(torch.from_numpy(arrays["X"]), torch.from_numpy(arrays["y"]), torch.from_numpy(arrays["snr"])), batch_size=model.batch_size, shuffle=False)
    branch_names = [name for name in BRANCH_NAMES if getattr(backbone, f"use_{name}_branch")]
    all_scores: list[np.ndarray] = []
    branch_drop_buffers = {name: [] for name in branch_names}
    backbone.eval()
    with torch.no_grad():
        for x_batch, _, _ in loader:
            x_batch = x_batch.float().to(model.device)
            feature_dict = build_branch_feature_dict(backbone, x_batch)
            full_scores = torch.sigmoid(classifier_from_branch_features(backbone, feature_dict))
            all_scores.append(full_scores.detach().cpu().numpy().astype(np.float32))
            for branch_name in branch_names:
                masked_scores = torch.sigmoid(classifier_from_branch_features(backbone, feature_dict, masked_branch=branch_name))
                branch_drop_buffers[branch_name].append((full_scores - masked_scores).detach().cpu().numpy().astype(np.float32))
    full_scores = np.concatenate(all_scores, axis=0).astype(np.float32)
    predictions = (full_scores >= threshold).astype(np.int64)
    y_true = arrays["y"]
    snr_values = arrays["snr"]
    positive_mask = y_true == 1
    tp_mask = positive_mask & (predictions == 1)
    rows: list[dict] = []
    branch_drop_arrays = {name: np.concatenate(buffers, axis=0).astype(np.float32) for name, buffers in branch_drop_buffers.items()}
    for branch_name, score_drop in branch_drop_arrays.items():
        for snr_value in sorted(np.unique(snr_values)):
            snr_mask = snr_values == snr_value
            selected_mask = tp_mask & snr_mask
            selection_source = "tp_signal"
            if not np.any(selected_mask):
                selected_mask = positive_mask & snr_mask
                selection_source = "signal"
            if not np.any(selected_mask):
                continue
            selected_values = score_drop[selected_mask]
            rows.append({
                "branch": branch_name,
                "branch_label": BRANCH_LABELS[branch_name],
                "snr": int(snr_value),
                "count": int(np.sum(selected_mask)),
                "selection": selection_source,
                "mean_score_drop": float(np.mean(selected_values)),
                "std_score_drop": float(np.std(selected_values)),
                "median_score_drop": float(np.median(selected_values)),
            })
    return rows, {
        "full_scores": full_scores,
        "predictions": predictions,
        "branch_drops": branch_drop_arrays,
        "selection_tp_count": int(np.sum(tp_mask)),
        "selection_signal_count": int(np.sum(positive_mask)),
    }


def plot_branch_contributions(rows: list[dict], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for branch_name in BRANCH_NAMES:
        branch_rows = [row for row in rows if row["branch"] == branch_name]
        if not branch_rows:
            continue
        branch_rows = sorted(branch_rows, key=lambda row: int(row["snr"]))
        snrs = [int(row["snr"]) for row in branch_rows]
        means = [float(row["mean_score_drop"]) for row in branch_rows]
        stds = [float(row["std_score_drop"]) for row in branch_rows]
        ax.plot(snrs, means, marker="o", linewidth=2.2, label=BRANCH_LABELS[branch_name], color=BRANCH_COLORS[branch_name])
        ax.fill_between(snrs, np.asarray(means) - np.asarray(stds), np.asarray(means) + np.asarray(stds), alpha=0.15, color=BRANCH_COLORS[branch_name])
    ax.axhline(0.0, color="#868e96", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Source SNR (dB)")
    ax.set_ylabel("Probability Drop After Branch Masking")
    ax.set_title("Branch Contribution by Source SNR")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

def predict_scores_for_array(model, X: np.ndarray, occlusion_window: int | None = None, energy_window: int | None = None) -> np.ndarray:
    backbone = model.model
    if backbone is None:
        raise RuntimeError("Backbone is not initialized.")
    batch_size = int(model.batch_size)
    scores: list[np.ndarray] = []
    backbone.eval()
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            end = min(start + batch_size, len(X))
            batch = torch.from_numpy(X[start:end]).float().to(model.device)
            if occlusion_window is not None:
                if energy_window is None:
                    raise ValueError("energy_window is required when occlusion_window is set.")
                iq_start = int(occlusion_window) * int(energy_window)
                iq_end = iq_start + int(energy_window)
                batch = batch.clone()
                batch[:, 2 * iq_start : 2 * iq_end] = 0.0
            probs = torch.sigmoid(backbone(batch))
            scores.append(probs.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(scores, axis=0).astype(np.float32) if scores else np.asarray([], dtype=np.float32)


def compute_occlusion_matrix(model, arrays: dict[str, np.ndarray], predictions: np.ndarray, energy_window: int) -> tuple[np.ndarray, list[int], list[dict]]:
    backbone = model.model
    if backbone is None:
        raise RuntimeError("Backbone is not initialized.")
    y_true = arrays["y"]
    snr_values = arrays["snr"]
    signal_mask = y_true == 1
    tp_mask = signal_mask & (predictions == 1)
    num_windows = int(backbone.num_windows)
    snr_levels = sorted(np.unique(snr_values).tolist())
    rows: list[dict] = []
    matrix = np.full((len(snr_levels), num_windows), np.nan, dtype=np.float32)
    base_scores = predict_scores_for_array(model, arrays["X"])
    for row_idx, snr_value in enumerate(snr_levels):
        selected_mask = tp_mask & (snr_values == snr_value)
        selection_source = "tp_signal"
        if not np.any(selected_mask):
            selected_mask = signal_mask & (snr_values == snr_value)
            selection_source = "signal"
        if not np.any(selected_mask):
            continue
        X_selected = arrays["X"][selected_mask]
        base_selected = base_scores[selected_mask]
        for window_idx in range(num_windows):
            occluded_scores = predict_scores_for_array(model, X_selected, occlusion_window=window_idx, energy_window=energy_window)
            score_drop = base_selected - occluded_scores
            matrix[row_idx, window_idx] = float(np.mean(score_drop))
            rows.append({
                "snr": int(snr_value),
                "window_idx": int(window_idx),
                "count": int(np.sum(selected_mask)),
                "selection": selection_source,
                "mean_score_drop": float(np.mean(score_drop)),
                "std_score_drop": float(np.std(score_drop)),
            })
    return matrix, snr_levels, rows


def plot_occlusion_heatmap(matrix: np.ndarray, snr_levels: list[int], energy_window: int, output_path: Path) -> None:
    if matrix.size == 0:
        return
    finite_values = matrix[np.isfinite(matrix)]
    vmax = float(np.max(np.abs(finite_values))) if finite_values.size else 1.0
    vmax = max(vmax, 1e-6)
    fig, ax = plt.subplots(figsize=(12, 7))
    image = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax, interpolation="nearest")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Mean Probability Drop")
    ax.set_xlabel(f"Occluded Window Index ({energy_window} IQ samples per window)")
    ax.set_ylabel("Source SNR (dB)")
    ax.set_title("Local Occlusion Heatmap on Signal Samples")
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_yticks(np.arange(len(snr_levels)), labels=[str(value) for value in snr_levels])
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def compute_window_importance_for_sample(model, sample_x: np.ndarray, energy_window: int) -> np.ndarray:
    sample_batch = np.asarray(sample_x, dtype=np.float32).reshape(1, -1)
    base_score = float(predict_scores_for_array(model, sample_batch)[0])
    backbone = model.model
    if backbone is None:
        raise RuntimeError("Backbone is not initialized.")
    values = np.zeros(int(backbone.num_windows), dtype=np.float32)
    for window_idx in range(int(backbone.num_windows)):
        occluded_score = float(predict_scores_for_array(model, sample_batch, occlusion_window=window_idx, energy_window=energy_window)[0])
        values[window_idx] = base_score - occluded_score
    return values


def pick_case_index(candidate_idx: np.ndarray, scores: np.ndarray, snr_values: np.ndarray, prefer: str) -> int | None:
    if candidate_idx.size == 0:
        return None
    snr_subset = snr_values[candidate_idx]
    score_subset = scores[candidate_idx]
    if prefer == "lowest_snr_high_score":
        target_snr = np.min(snr_subset)
        narrowed = candidate_idx[snr_subset == target_snr]
        return int(narrowed[np.argmax(scores[narrowed])])
    if prefer == "highest_snr_high_score":
        target_snr = np.max(snr_subset)
        narrowed = candidate_idx[snr_subset == target_snr]
        return int(narrowed[np.argmax(scores[narrowed])])
    if prefer == "highest_score":
        return int(candidate_idx[np.argmax(score_subset)])
    if prefer == "lowest_score":
        return int(candidate_idx[np.argmin(score_subset)])
    raise ValueError(f"Unknown preference: {prefer}")


def select_case_indices(arrays: dict[str, np.ndarray], scores: np.ndarray, predictions: np.ndarray) -> list[dict]:
    y_true = arrays["y"]
    snr_values = arrays["snr"]
    tp_idx = np.where((y_true == 1) & (predictions == 1))[0]
    fp_idx = np.where((y_true == 0) & (predictions == 1))[0]
    fn_idx = np.where((y_true == 1) & (predictions == 0))[0]
    noise_idx = np.where(y_true == 0)[0]
    pos_idx = np.where(y_true == 1)[0]
    case_specs = [
        {"case_id": "tp_low_snr", "title": "Correct Signal Detection (Low SNR)", "pool": tp_idx, "fallback": pos_idx, "prefer": "lowest_snr_high_score"},
        {"case_id": "tp_high_snr", "title": "Correct Signal Detection (High SNR)", "pool": tp_idx, "fallback": pos_idx, "prefer": "highest_snr_high_score"},
        {"case_id": "false_positive", "title": "False Positive Noise Sample", "pool": fp_idx, "fallback": noise_idx, "prefer": "highest_score"},
        {"case_id": "false_negative", "title": "False Negative Signal Sample", "pool": fn_idx, "fallback": pos_idx, "prefer": "lowest_score"},
    ]
    used_indices: set[int] = set()
    selected_cases: list[dict] = []
    for spec in case_specs:
        chosen_idx = pick_case_index(spec["pool"], scores, snr_values, spec["prefer"])
        if chosen_idx is None or chosen_idx in used_indices:
            fallback_candidates = np.asarray([idx for idx in spec["fallback"] if int(idx) not in used_indices], dtype=np.int64)
            chosen_idx = pick_case_index(fallback_candidates, scores, snr_values, spec["prefer"])
        if chosen_idx is None:
            continue
        used_indices.add(chosen_idx)
        selected_cases.append({"case_id": spec["case_id"], "title": spec["title"], "index": int(chosen_idx)})
    return selected_cases


def plot_case_examples(arrays: dict[str, np.ndarray], scores: np.ndarray, predictions: np.ndarray, selected_cases: list[dict], model, threshold: float, energy_window: int, output_path: Path) -> list[dict]:
    if not selected_cases:
        return []
    num_cases = len(selected_cases)
    fig, axes = plt.subplots(num_cases, 2, figsize=(14, 3.5 * num_cases))
    axes = np.asarray(axes).reshape(num_cases, 2)
    case_rows: list[dict] = []
    for row_idx, case in enumerate(selected_cases):
        sample_idx = int(case["index"])
        sample_x = arrays["X"][sample_idx]
        y_true = int(arrays["y"][sample_idx])
        y_pred = int(predictions[sample_idx])
        snr_value = int(arrays["snr"][sample_idx])
        score_value = float(scores[sample_idx])
        mod_value = None if arrays["mod"] is None else str(arrays["mod"][sample_idx])
        window_importance = compute_window_importance_for_sample(model, sample_x, energy_window)
        iq = sample_x.reshape(-1, 2)
        timeline = np.arange(len(iq))
        waveform_ax = axes[row_idx, 0]
        waveform_ax.plot(timeline, iq[:, 0], linewidth=1.4, color="#1c7ed6", label="I")
        waveform_ax.plot(timeline, iq[:, 1], linewidth=1.4, color="#e8590c", label="Q")
        top_windows = np.argsort(window_importance)[-3:]
        max_importance = float(np.max(np.abs(window_importance))) if window_importance.size else 0.0
        for window_idx in sorted(top_windows):
            alpha = 0.12 if max_importance <= 0.0 else 0.12 + 0.28 * float(abs(window_importance[window_idx]) / max_importance)
            waveform_ax.axvspan(window_idx * energy_window, (window_idx + 1) * energy_window, color="#74c0fc", alpha=min(alpha, 0.45))
        waveform_ax.set_xlim(0, len(iq) - 1)
        waveform_ax.set_xlabel("IQ Sample Index")
        waveform_ax.set_ylabel("Amplitude")
        waveform_ax.set_title(f"{case['title']}\ntrue={y_true} pred={y_pred} score={score_value:.3f} thr={threshold:.3f} SNR={snr_value} mod={mod_value}")
        waveform_ax.grid(True, linestyle="--", alpha=0.3)
        waveform_ax.legend(loc="upper right")
        bar_ax = axes[row_idx, 1]
        bar_ax.bar(np.arange(len(window_importance)), window_importance, color="#2b8a3e" if y_true == y_pred else "#c92a2a", alpha=0.85)
        bar_ax.axhline(0.0, color="#868e96", linewidth=1.0, linestyle="--")
        bar_ax.set_xlabel(f"Window Index ({energy_window} IQ samples/window)")
        bar_ax.set_ylabel("Probability Drop")
        bar_ax.set_title("Per-window Local Occlusion Importance")
        bar_ax.grid(True, axis="y", linestyle="--", alpha=0.3)
        case_rows.append({
            "case_id": case["case_id"],
            "title": case["title"],
            "index": sample_idx,
            "true_label": y_true,
            "pred_label": y_pred,
            "score": score_value,
            "threshold": float(threshold),
            "snr": snr_value,
            "mod": mod_value,
        })
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return case_rows


def maybe_save_checkpoint(model, output_path: Path, args, summary: dict) -> None:
    if not args.save_checkpoint:
        return
    payload = {
        "model_name": "cnr_sensenet",
        "config": {
            "signal_length": summary["dataset"]["input_dim"],
            "energy_window": args.energy_window,
            "dropout": args.dropout,
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
            "decision_threshold": summary["overall_metrics"]["threshold"],
        },
        "metrics": summary["overall_metrics"],
        "state_dict": model.state_dict(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)

def main() -> None:
    args = build_parser().parse_args()
    apply_smoke_overrides(args)
    seed_everything(args.seed)
    if args.num_threads is not None:
        torch.set_num_threads(int(args.num_threads))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()

    bundle = build_datasets(
        DataConfig(
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
    )

    if args.val_ratio > 0.0:
        train_idx, val_idx = stratified_index_split_binary(
            y=bundle.train_arrays["y"],
            snr=bundle.train_arrays["snr"],
            test_ratio=args.val_ratio,
            seed=args.seed,
        )
        train_dataset = Subset(bundle.train_dataset, train_idx.tolist())
        val_dataset = Subset(bundle.train_dataset, val_idx.tolist())
    else:
        train_dataset = bundle.train_dataset
        val_dataset = None

    train_dataset = maybe_limit_dataset(train_dataset, args.max_train_samples, seed=args.seed)
    if val_dataset is not None:
        val_dataset = maybe_limit_dataset(val_dataset, args.max_val_samples, seed=args.seed + 1)
    test_dataset = maybe_limit_dataset(bundle.test_dataset, args.max_test_samples, seed=args.seed + 2)

    model = create_model(
        "cnr_sensenet",
        signal_length=bundle.input_dim,
        energy_window=args.energy_window,
        dropout=args.dropout,
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

    train_start = time.perf_counter()
    model.fit(
        train_dataset,
        val_dataset=val_dataset,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        patience=args.patience,
        threshold_mode=args.threshold_mode,
        target_pfa=args.target_pfa,
        calibration_split=args.calibration_split,
        snr_loss_weighting=args.snr_loss_weighting,
        low_snr_cutoff=args.low_snr_cutoff,
        low_snr_positive_weight=args.low_snr_positive_weight,
        mid_snr_cutoff=args.mid_snr_cutoff,
        mid_snr_positive_weight=args.mid_snr_positive_weight,
        verbose=args.verbose,
    )
    train_seconds = time.perf_counter() - train_start

    test_arrays = dataset_arrays(test_dataset)
    scores = model.predict_scores(test_dataset)
    threshold = float(model.get_evaluation_threshold())
    metrics, predictions = compute_metrics(test_arrays["y"], scores, threshold=threshold)

    explain_start = time.perf_counter()
    branch_rows, branch_meta = compute_branch_contribution_rows(model, test_arrays, threshold=threshold)
    branch_fig_path = output_dir / f"{args.output_prefix}_branch_contribution_by_snr.png"
    branch_csv_path = output_dir / f"{args.output_prefix}_branch_contribution_by_snr.csv"
    plot_branch_contributions(branch_rows, branch_fig_path)
    write_csv(branch_csv_path, branch_rows)

    occlusion_matrix, snr_levels, occlusion_rows = compute_occlusion_matrix(model, test_arrays, predictions, args.energy_window)
    occlusion_fig_path = output_dir / f"{args.output_prefix}_local_occlusion_heatmap.png"
    occlusion_csv_path = output_dir / f"{args.output_prefix}_local_occlusion_heatmap.csv"
    plot_occlusion_heatmap(occlusion_matrix, snr_levels, args.energy_window, occlusion_fig_path)
    write_csv(occlusion_csv_path, occlusion_rows)

    selected_cases = select_case_indices(test_arrays, scores, predictions)
    case_fig_path = output_dir / f"{args.output_prefix}_example_cases.png"
    case_json_path = output_dir / f"{args.output_prefix}_example_cases.json"
    case_rows = plot_case_examples(test_arrays, scores, predictions, selected_cases, model, threshold, args.energy_window, case_fig_path)
    write_json(case_json_path, {"cases": case_rows})
    explain_seconds = time.perf_counter() - explain_start

    summary = {
        "model": "cnr_sensenet",
        "seed": int(args.seed),
        "device": str(model.device),
        "overall_metrics": metrics,
        "dataset": {
            "input_dim": int(bundle.input_dim),
            "train_size": int(len(train_dataset)),
            "val_size": int(len(val_dataset)) if val_dataset is not None else 0,
            "test_size": int(len(test_dataset)),
            "noise_power": float(bundle.noise_power),
            "cache_path": str(bundle.cache_path),
        },
        "analysis": {
            "branch_selection_tp_count": int(branch_meta["selection_tp_count"]),
            "branch_selection_signal_count": int(branch_meta["selection_signal_count"]),
            "snr_levels": [int(value) for value in snr_levels],
            "num_windows": int(model.model.num_windows if model.model is not None else 0),
        },
        "timing": {
            "train_seconds": float(train_seconds),
            "explain_seconds": float(explain_seconds),
            "total_seconds": float(time.perf_counter() - start_time),
        },
        "artifacts": {
            "branch_contribution_png": str(branch_fig_path),
            "branch_contribution_csv": str(branch_csv_path),
            "local_occlusion_png": str(occlusion_fig_path),
            "local_occlusion_csv": str(occlusion_csv_path),
            "example_cases_png": str(case_fig_path),
            "example_cases_json": str(case_json_path),
        },
    }
    summary_path = output_dir / f"{args.output_prefix}_summary.json"
    write_json(summary_path, summary)

    checkpoint_path = output_dir / f"{args.output_prefix}_checkpoint.pt"
    maybe_save_checkpoint(model, checkpoint_path, args, summary)

    if args.verbose:
        print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
