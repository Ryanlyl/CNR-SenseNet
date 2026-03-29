from __future__ import annotations

import argparse
import time
from itertools import product
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

from project.data import DataConfig, build_datasets, default_rml2016a_path
from project.models import create_model
from project.run_cnr_sensenet_eval import (
    compute_metrics,
    compute_metrics_by_snr,
    seed_everything,
    stratified_index_split_binary,
    write_csv,
    write_json,
)
from project.utils import resolve_path


DEFAULT_OUTPUT_DIR = resolve_path("results", "cnr_sensenet_search")
SELECTION_METRICS = (
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "pd",
    "roc_auc",
    "average_precision",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Grid-search training hyperparameters for CNR-SenseNet on the current dataset."
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
    parser.add_argument("--lr-values", nargs="+", type=float, default=[1e-3])
    parser.add_argument("--weight-decay-values", nargs="+", type=float, default=[0.0])
    parser.add_argument("--dropout-values", nargs="+", type=float, default=[0.2])
    parser.add_argument("--batch-size-values", nargs="+", type=int, default=[1024])
    parser.add_argument("--epochs-values", nargs="+", type=int, default=[5])
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--energy-window", type=int, default=8)
    parser.add_argument(
        "--threshold-mode",
        choices=["fixed", "balanced_acc", "youden", "target_pfa"],
        default="balanced_acc",
    )
    parser.add_argument("--target-pfa", type=float, default=0.1)
    parser.add_argument(
        "--calibration-split",
        choices=["train", "val"],
        default="val",
    )
    parser.add_argument(
        "--snr-loss-weighting",
        choices=["none", "two_band"],
        default="two_band",
    )
    parser.add_argument("--low-snr-cutoff", type=int, default=-10)
    parser.add_argument("--low-snr-positive-weight", type=float, default=3.0)
    parser.add_argument("--mid-snr-cutoff", type=int, default=-6)
    parser.add_argument("--mid-snr-positive-weight", type=float, default=2.0)
    parser.add_argument("--decision-threshold", type=float, default=0.5)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--selection-metric",
        choices=SELECTION_METRICS,
        default="balanced_accuracy",
        help="Metric used to choose the best trial.",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Number of top trials to keep in the summary.")
    parser.add_argument("--max-trials", type=int, default=None, help="Optional cap on the number of tried configs.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for search summaries and the best checkpoint.",
    )
    parser.add_argument("--output-prefix", default="cnr_sensenet_search")
    parser.add_argument("--save-best-checkpoint", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def ordered_unique(values):
    return list(dict.fromkeys(values))


def build_trials(args) -> list[dict[str, float | int]]:
    lr_values = ordered_unique(float(value) for value in args.lr_values)
    weight_decay_values = ordered_unique(float(value) for value in args.weight_decay_values)
    dropout_values = ordered_unique(float(value) for value in args.dropout_values)
    batch_size_values = ordered_unique(int(value) for value in args.batch_size_values)
    epochs_values = ordered_unique(int(value) for value in args.epochs_values)

    trials = [
        {
            "lr": lr,
            "weight_decay": weight_decay,
            "dropout": dropout,
            "batch_size": batch_size,
            "epochs": epochs,
        }
        for lr, weight_decay, dropout, batch_size, epochs in product(
            lr_values,
            weight_decay_values,
            dropout_values,
            batch_size_values,
            epochs_values,
        )
    ]
    if args.max_trials is not None:
        return trials[: max(int(args.max_trials), 0)]
    return trials


def extract_training_history(model) -> dict[str, object]:
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


def metric_value(row: dict, metric_name: str) -> float:
    value = row.get(metric_name)
    if value is None:
        return float("-inf")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


def clone_state_dict(model) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def maybe_save_best_checkpoint(
    model,
    output_path: Path,
    bundle,
    args,
    best_trial: dict,
) -> None:
    if not args.save_best_checkpoint:
        return

    payload = {
        "model_name": "cnr_sensenet",
        "config": {
            "signal_length": int(bundle.input_dim),
            "energy_window": int(args.energy_window),
            "dropout": float(best_trial["trial"]["dropout"]),
            "lr": float(best_trial["trial"]["lr"]),
            "batch_size": int(best_trial["trial"]["batch_size"]),
            "epochs": int(best_trial["trial"]["epochs"]),
            "weight_decay": float(best_trial["trial"]["weight_decay"]),
            "patience": int(args.patience),
            "threshold": float(args.decision_threshold),
            "threshold_mode": args.threshold_mode,
            "target_pfa": float(args.target_pfa),
            "calibration_split": args.calibration_split,
            "snr_loss_weighting": args.snr_loss_weighting,
            "low_snr_cutoff": int(args.low_snr_cutoff),
            "low_snr_positive_weight": float(args.low_snr_positive_weight),
            "mid_snr_cutoff": int(args.mid_snr_cutoff),
            "mid_snr_positive_weight": float(args.mid_snr_positive_weight),
            "decision_threshold": float(best_trial["overall_metrics"]["threshold"]),
        },
        "metrics": best_trial["overall_metrics"],
        "selection_metric": args.selection_metric,
        "selection_value": float(best_trial["overall_metrics"][args.selection_metric]),
        "state_dict": clone_state_dict(model),
    }
    torch.save(payload, output_path)


def main() -> None:
    args = build_parser().parse_args()
    seed_everything(args.seed)

    trials = build_trials(args)
    if not trials:
        raise ValueError("No search trials were generated from the provided hyperparameter values.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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
    else:
        train_dataset = bundle.train_dataset
        val_dataset = None

    artifact_paths = {
        "summary_json": output_dir / f"{args.output_prefix}_summary.json",
        "results_csv": output_dir / f"{args.output_prefix}_results.csv",
        "best_trial_json": output_dir / f"{args.output_prefix}_best_trial.json",
        "best_checkpoint_pt": output_dir / f"{args.output_prefix}_best_checkpoint.pt",
    }

    trial_rows: list[dict[str, object]] = []
    best_trial: dict[str, object] | None = None
    best_model = None

    for trial_id, trial in enumerate(trials, start=1):
        print(
            f"[Trial {trial_id}/{len(trials)}] "
            f"lr={trial['lr']} wd={trial['weight_decay']} dropout={trial['dropout']} "
            f"batch={trial['batch_size']} epochs={trial['epochs']}"
        )

        model = create_model(
            "cnr_sensenet",
            signal_length=bundle.input_dim,
            energy_window=args.energy_window,
            dropout=trial["dropout"],
            lr=trial["lr"],
            batch_size=trial["batch_size"],
            epochs=trial["epochs"],
            weight_decay=trial["weight_decay"],
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
            epochs=trial["epochs"],
            lr=trial["lr"],
            batch_size=trial["batch_size"],
            weight_decay=trial["weight_decay"],
            patience=args.patience,
            threshold_mode=args.threshold_mode,
            target_pfa=args.target_pfa,
            calibration_split=args.calibration_split,
            verbose=args.verbose,
        )
        train_seconds = time.perf_counter() - train_start

        eval_start = time.perf_counter()
        test_scores = np.asarray(model.predict_scores(bundle.test_dataset), dtype=np.float64)
        eval_seconds = time.perf_counter() - eval_start

        requested_threshold = None if getattr(model, "prefers_internal_threshold", False) else args.decision_threshold
        eval_threshold = float(model.get_evaluation_threshold(requested_threshold))
        overall_metrics, _ = compute_metrics(
            bundle.test_arrays["y"],
            test_scores,
            threshold=eval_threshold,
        )
        metrics_by_snr = compute_metrics_by_snr(
            bundle.test_arrays["y"],
            test_scores,
            bundle.test_arrays["snr"],
            threshold=eval_threshold,
        )
        training_history = extract_training_history(model)

        row = {
            "trial_id": trial_id,
            "lr": float(trial["lr"]),
            "weight_decay": float(trial["weight_decay"]),
            "dropout": float(trial["dropout"]),
            "batch_size": int(trial["batch_size"]),
            "epochs": int(trial["epochs"]),
            "epochs_ran": int(training_history["epochs_ran"]),
            "best_epoch": training_history["best_epoch"],
            "best_val_loss": training_history["best_val_loss"],
            "stopped_early": bool(training_history["stopped_early"]),
            "threshold": float(overall_metrics["threshold"]),
            "accuracy": float(overall_metrics["accuracy"]),
            "balanced_accuracy": float(overall_metrics["balanced_accuracy"]),
            "precision": float(overall_metrics["precision"]),
            "recall": float(overall_metrics["recall"]),
            "f1": float(overall_metrics["f1"]),
            "pd": float(overall_metrics["pd"]),
            "pfa": float(overall_metrics["pfa"]),
            "roc_auc": overall_metrics["roc_auc"],
            "average_precision": overall_metrics["average_precision"],
            "train_seconds": float(train_seconds),
            "eval_seconds": float(eval_seconds),
        }
        trial_rows.append(row)

        detailed_result = {
            "trial_id": trial_id,
            "trial": {
                "lr": float(trial["lr"]),
                "weight_decay": float(trial["weight_decay"]),
                "dropout": float(trial["dropout"]),
                "batch_size": int(trial["batch_size"]),
                "epochs": int(trial["epochs"]),
                "patience": int(args.patience),
            },
            "overall_metrics": overall_metrics,
            "metrics_by_snr": metrics_by_snr,
            "training_history": training_history,
            "timing_seconds": {
                "train": float(train_seconds),
                "evaluate": float(eval_seconds),
            },
        }

        if best_trial is None or metric_value(row, args.selection_metric) > metric_value(
            best_trial["overall_metrics"],
            args.selection_metric,
        ):
            best_trial = detailed_result
            best_model = model
            maybe_save_best_checkpoint(best_model, artifact_paths["best_checkpoint_pt"], bundle, args, best_trial)

    sorted_rows = sorted(
        trial_rows,
        key=lambda row: metric_value(row, args.selection_metric),
        reverse=True,
    )
    top_rows = sorted_rows[: max(int(args.top_k), 0)]

    summary = {
        "selection_metric": args.selection_metric,
        "num_trials": len(trials),
        "dataset": {
            "pkl_path": str(bundle.pkl_path),
            "cache_path": str(bundle.cache_path),
            "train_size": int(len(train_dataset)),
            "val_size": int(len(val_dataset)) if val_dataset is not None else 0,
            "test_size": int(len(bundle.test_dataset)),
            "input_dim": int(bundle.input_dim),
            "noise_power": float(bundle.noise_power),
            "snrs": [int(value) for value in bundle.snrs],
            "mods": list(bundle.mods),
        },
        "search_space": {
            "lr_values": ordered_unique(float(value) for value in args.lr_values),
            "weight_decay_values": ordered_unique(float(value) for value in args.weight_decay_values),
            "dropout_values": ordered_unique(float(value) for value in args.dropout_values),
            "batch_size_values": ordered_unique(int(value) for value in args.batch_size_values),
            "epochs_values": ordered_unique(int(value) for value in args.epochs_values),
            "patience": int(args.patience),
        },
        "fixed_config": {
            "energy_window": int(args.energy_window),
            "threshold_mode": args.threshold_mode,
            "target_pfa": float(args.target_pfa),
            "calibration_split": args.calibration_split,
            "snr_loss_weighting": args.snr_loss_weighting,
            "low_snr_cutoff": int(args.low_snr_cutoff),
            "low_snr_positive_weight": float(args.low_snr_positive_weight),
            "mid_snr_cutoff": int(args.mid_snr_cutoff),
            "mid_snr_positive_weight": float(args.mid_snr_positive_weight),
            "decision_threshold": float(args.decision_threshold),
            "device": args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
        },
        "best_trial": best_trial,
        "top_trials": top_rows,
        "artifacts": {
            name: str(path)
            for name, path in artifact_paths.items()
            if name != "best_checkpoint_pt" or args.save_best_checkpoint
        },
    }

    write_csv(artifact_paths["results_csv"], sorted_rows)
    write_json(artifact_paths["summary_json"], summary)
    if best_trial is not None:
        write_json(artifact_paths["best_trial_json"], best_trial)

    print(f"Search completed: {len(trials)} trials")
    if best_trial is not None:
        print(
            f"Best {args.selection_metric}={best_trial['overall_metrics'][args.selection_metric]:.4f} "
            f"at lr={best_trial['trial']['lr']}, wd={best_trial['trial']['weight_decay']}, "
            f"dropout={best_trial['trial']['dropout']}, batch={best_trial['trial']['batch_size']}, "
            f"epochs={best_trial['trial']['epochs']}"
        )
    print("Saved artifacts:")
    for name, path in artifact_paths.items():
        if name == "best_checkpoint_pt" and not args.save_best_checkpoint:
            continue
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
