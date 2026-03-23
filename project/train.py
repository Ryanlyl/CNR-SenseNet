from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from project.data.gen_dataset import SignalNoiseDataset, load_signal_vs_noise_archive
from project.models import MODEL_REGISTRY, create_model


DEFAULT_MODELS = ["energy_detector", "autocorr_detector", "mlp", "cnn1d", "lstm"]


def find_default_npz() -> Path:
    processed_dir = Path(__file__).resolve().parent / "data" / "processed"
    candidates = sorted(processed_dir.glob("*.npz"), key=lambda path: path.stat().st_size, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No .npz dataset found under {processed_dir}")
    return candidates[0]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train or calibrate baseline models on the prepared signal-vs-noise .npz dataset."
    )
    parser.add_argument("--dataset-npz", type=Path, default=None, help="Path to a prepared .npz file.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        choices=sorted(MODEL_REGISTRY),
        help="Models to train or calibrate.",
    )
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Held-out test ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation ratio over the full dataset.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs for each baseline.")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Adam weight decay.")
    parser.add_argument("--patience", type=int, default=2, help="Validation patience for early stopping.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for probability outputs.")
    parser.add_argument("--device", default=None, help="Training device. Defaults to cuda if available else cpu.")
    parser.add_argument("--num-threads", type=int, default=None, help="Torch CPU thread count.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("project/results/baselines"),
        help="Directory for checkpoints and metrics.",
    )
    return parser


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)



def build_strata(y: np.ndarray, snr: np.ndarray) -> np.ndarray:
    snr_min = int(snr.min())
    snr_span = int(snr.max() - snr_min + 1)
    return y.astype(np.int64) * snr_span + (snr.astype(np.int64) - snr_min)



def split_indices(
    y: np.ndarray,
    snr: np.ndarray,
    test_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not 0.0 <= test_ratio < 1.0:
        raise ValueError("test_ratio must be in [0, 1).")
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0, 1).")
    if test_ratio + val_ratio >= 1.0:
        raise ValueError("test_ratio + val_ratio must be smaller than 1.0.")

    indices = np.arange(len(y), dtype=np.int64)
    strata = build_strata(y, snr)

    if test_ratio > 0.0:
        train_val_idx, test_idx = train_test_split(
            indices,
            test_size=test_ratio,
            random_state=seed,
            stratify=strata,
        )
    else:
        train_val_idx = indices
        test_idx = np.asarray([], dtype=np.int64)

    if val_ratio <= 0.0:
        return np.sort(train_val_idx), np.asarray([], dtype=np.int64), np.sort(test_idx)

    val_ratio_within_train = val_ratio / (1.0 - test_ratio)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_ratio_within_train,
        random_state=seed,
        stratify=strata[train_val_idx],
    )
    return np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)



def make_dataset(arrays: dict[str, np.ndarray], indices: np.ndarray) -> SignalNoiseDataset:
    return SignalNoiseDataset(
        arrays["X"][indices],
        arrays["y"][indices],
        arrays["snr"][indices],
    )



def count_parameters(model) -> int:
    if getattr(model, "model", None) is None:
        return 0
    return int(sum(parameter.numel() for parameter in model.model.parameters()))



def evaluate_model(model, dataset: SignalNoiseDataset, threshold: float) -> dict:
    y_true = dataset.y.cpu().numpy().astype(np.int64)
    snr = dataset.snr.cpu().numpy().astype(np.int64)
    scores = model.predict_scores(dataset)
    requested_threshold = None if getattr(model, "prefers_internal_threshold", False) else threshold
    threshold = model.get_evaluation_threshold(requested_threshold)
    preds = model.predict(dataset, threshold=threshold)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        preds,
        average="binary",
        zero_division=0,
    )
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()

    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, preds)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "Pd": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        "Pfa": float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "num_samples": int(len(y_true)),
    }

    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
        metrics["average_precision"] = float(average_precision_score(y_true, scores))
    else:
        metrics["roc_auc"] = float("nan")
        metrics["average_precision"] = float("nan")

    by_snr = {}
    for snr_value in np.unique(snr):
        mask = snr == snr_value
        y_group = y_true[mask]
        pred_group = preds[mask]
        tp_group = int(np.sum((y_group == 1) & (pred_group == 1)))
        fp_group = int(np.sum((y_group == 0) & (pred_group == 1)))
        tn_group = int(np.sum((y_group == 0) & (pred_group == 0)))
        fn_group = int(np.sum((y_group == 1) & (pred_group == 0)))
        by_snr[int(snr_value)] = {
            "samples": int(mask.sum()),
            "accuracy": float(np.mean(pred_group == y_group)),
            "Pd": float(tp_group / (tp_group + fn_group)) if (tp_group + fn_group) > 0 else 0.0,
            "Pfa": float(fp_group / (fp_group + tn_group)) if (fp_group + tn_group) > 0 else 0.0,
        }

    return {
        "metrics": metrics,
        "by_snr": by_snr,
    }



def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")



def run_experiment(args) -> Path:
    set_seed(args.seed)
    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)

    dataset_path = args.dataset_npz or find_default_npz()
    dataset_path = dataset_path.resolve()
    archive = load_signal_vs_noise_archive(dataset_path)
    arrays = {
        "X": archive["X"],
        "y": archive["y"],
        "snr": archive["snr"],
    }
    train_idx, val_idx, test_idx = split_indices(
        y=arrays["y"],
        snr=arrays["snr"],
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    train_dataset = make_dataset(arrays, train_idx)
    val_dataset = make_dataset(arrays, val_idx) if len(val_idx) > 0 else None
    test_dataset = make_dataset(arrays, test_idx)
    del archive
    del arrays

    run_dir = (args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "dataset_npz": str(dataset_path),
        "seed": int(args.seed),
        "test_ratio": float(args.test_ratio),
        "val_ratio": float(args.val_ratio),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "threshold": float(args.threshold),
        "device": args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
        "split_sizes": {
            "train": int(len(train_dataset)),
            "val": int(len(val_dataset)) if val_dataset is not None else 0,
            "test": int(len(test_dataset)),
        },
    }
    save_json(run_dir / "manifest.json", manifest)

    print(f"Dataset: {dataset_path}")
    print(
        f"Split sizes | train={manifest['split_sizes']['train']} "
        f"val={manifest['split_sizes']['val']} test={manifest['split_sizes']['test']}"
    )

    rows = []
    for model_name in args.models:
        print(f"\n=== Training {model_name} ===")
        model = create_model(
            model_name,
            lr=args.lr,
            batch_size=args.batch_size,
            epochs=args.epochs,
            weight_decay=args.weight_decay,
            threshold=args.threshold,
            device=args.device,
        )
        model.fit(
            train_dataset,
            val_dataset=val_dataset,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            weight_decay=args.weight_decay,
            patience=args.patience,
            verbose=True,
        )

        result = {
            "model": model_name,
            "parameter_count": count_parameters(model),
            "history": {
                "train_loss": list(getattr(getattr(model, "history", None), "train_loss", [])),
                "val_loss": list(getattr(getattr(model, "history", None), "val_loss", [])),
            },
            "validation": evaluate_model(model, val_dataset, threshold=args.threshold)
            if val_dataset is not None
            else None,
            "test": evaluate_model(model, test_dataset, threshold=args.threshold),
        }

        state = {
            "model_name": model_name,
            "signal_length": int(getattr(model, "signal_length", 0) or 0),
            "state_dict": model.state_dict(),
            "config": model.get_config(),
        }
        torch.save(state, run_dir / f"{model_name}.pt")
        save_json(run_dir / f"{model_name}_metrics.json", result)

        summary = result["test"]["metrics"]
        print(
            f"{model_name}: acc={summary['accuracy']:.4f} "
            f"bal_acc={summary['balanced_accuracy']:.4f} "
            f"f1={summary['f1']:.4f} roc_auc={summary['roc_auc']:.4f}"
        )
        rows.append(
            {
                "model": model_name,
                "params": result["parameter_count"],
                "accuracy": summary["accuracy"],
                "balanced_accuracy": summary["balanced_accuracy"],
                "precision": summary["precision"],
                "recall": summary["recall"],
                "f1": summary["f1"],
                "Pd": summary["Pd"],
                "Pfa": summary["Pfa"],
                "roc_auc": summary["roc_auc"],
                "average_precision": summary["average_precision"],
            }
        )

    summary_path = run_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    save_json(run_dir / "summary.json", {"rows": rows, "manifest": manifest})
    return run_dir



def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_dir = run_experiment(args)
    print(f"\nSaved outputs to: {run_dir}")


if __name__ == "__main__":
    main()
