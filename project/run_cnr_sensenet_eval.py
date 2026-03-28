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
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import Subset

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from project.data import DataConfig, build_datasets, default_rml2016a_path
from project.models import create_model
from project.utils import resolve_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Train and evaluate CNR-SenseNet on the binary signal/noise dataset.'
    )
    parser.add_argument('--pkl-path', type=Path, default=default_rml2016a_path())
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test-ratio', type=float, default=0.2)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--noise-power', type=float, default=1.0)
    parser.add_argument('--snr-filter', nargs='*', type=int, default=None)
    parser.add_argument('--selected-mods', nargs='*', default=None)
    parser.add_argument('--cache-path', type=Path, default=None)
    parser.add_argument('--force-rebuild', action='store_true')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--energy-window', type=int, default=8)
    parser.add_argument(
        '--threshold-mode',
        choices=['fixed', 'balanced_acc', 'youden', 'target_pfa'],
        default='balanced_acc',
    )
    parser.add_argument('--target-pfa', type=float, default=0.1)
    parser.add_argument(
        '--calibration-split',
        choices=['train', 'val'],
        default='val',
    )
    parser.add_argument(
        '--snr-loss-weighting',
        choices=['none', 'two_band'],
        default='two_band',
    )
    parser.add_argument('--low-snr-cutoff', type=int, default=-10)
    parser.add_argument('--low-snr-positive-weight', type=float, default=3.0)
    parser.add_argument('--mid-snr-cutoff', type=int, default=-6)
    parser.add_argument('--mid-snr-positive-weight', type=float, default=2.0)
    parser.add_argument('--decision-threshold', type=float, default=0.5)
    parser.add_argument('--device', default=None)
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=resolve_path('plots', 'cnr_sensenet_eval'),
    )
    parser.add_argument('--output-prefix', default='cnr_sensenet')
    parser.add_argument('--save-checkpoint', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    return parser


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stratified_index_split_binary(
    y: np.ndarray,
    snr: np.ndarray,
    test_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 <= test_ratio < 1.0:
        raise ValueError('test_ratio must be in [0, 1).')

    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    test_idx: list[int] = []

    for label in sorted(np.unique(y)):
        for snr_value in sorted(np.unique(snr)):
            bucket = np.where((y == label) & (snr == snr_value))[0]
            if len(bucket) == 0:
                continue

            rng.shuffle(bucket)
            n_test = max(1, int(len(bucket) * test_ratio)) if len(bucket) > 1 else 0
            test_idx.extend(bucket[:n_test].tolist())
            train_idx.extend(bucket[n_test:].tolist())

    train_idx = np.asarray(train_idx, dtype=np.int64)
    test_idx = np.asarray(test_idx, dtype=np.int64)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_idx, test_idx


def safe_roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    try:
        return float(roc_auc_score(y_true, scores))
    except ValueError:
        return None


def safe_average_precision(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    try:
        return float(average_precision_score(y_true, scores))
    except ValueError:
        return None


def compute_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> tuple[dict[str, float | int | None], np.ndarray]:
    y_true = np.asarray(y_true, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    y_pred = (scores >= threshold).astype(np.int64)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    pd = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    pfa = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    metrics = {
        'threshold': float(threshold),
        'count': int(len(y_true)),
        'signal_count': int(np.sum(y_true == 1)),
        'noise_count': int(np.sum(y_true == 0)),
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'specificity': float(specificity),
        'pd': float(pd),
        'pfa': float(pfa),
        'miss_rate': float(1.0 - pd),
        'roc_auc': safe_roc_auc(y_true, scores),
        'average_precision': safe_average_precision(y_true, scores),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
    }
    return metrics, y_pred


def compute_metrics_by_snr(
    y_true: np.ndarray,
    scores: np.ndarray,
    snr_values: np.ndarray,
    threshold: float,
) -> list[dict[str, float | int | None]]:
    rows: list[dict[str, float | int | None]] = []
    for snr_value in sorted(np.unique(snr_values)):
        mask = snr_values == snr_value
        metrics, _ = compute_metrics(y_true[mask], scores[mask], threshold=threshold)
        metrics['snr'] = int(snr_value)
        rows.append(metrics)
    return rows


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')


def write_csv(path: Path, rows: list[dict[str, float | int | None]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return

    fieldnames = list(rows[0].keys())
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_training_history(history, output_path: Path) -> None:
    train_loss = list(getattr(history, 'train_loss', []))
    val_loss = list(getattr(history, 'val_loss', []))
    if not train_loss:
        return

    epochs = np.arange(1, len(train_loss) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_loss, marker='o', linewidth=2, label='Train Loss')
    if val_loss:
        val_epochs = np.arange(1, len(val_loss) + 1)
        ax.plot(val_epochs, val_loss, marker='s', linewidth=2, label='Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('BCE Loss')
    ax.set_title('CNR-SenseNet Training History')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def plot_confusion_matrix_figure(y_true: np.ndarray, y_pred: np.ndarray, output_path: Path) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap='Blues')
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 1], labels=['Noise', 'Signal'])
    ax.set_yticks([0, 1], labels=['Noise', 'Signal'])
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(
                col,
                row,
                f'{matrix[row, col]}',
                ha='center',
                va='center',
                color='white' if matrix[row, col] > matrix.max() / 2 else 'black',
                fontsize=12,
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def plot_roc_pr_curves(y_true: np.ndarray, scores: np.ndarray, output_path: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, scores)
    precision, recall, _ = precision_recall_curve(y_true, scores)
    auc_value = safe_roc_auc(y_true, scores)
    ap_value = safe_average_precision(y_true, scores)
    positive_rate = float(np.mean(y_true))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(fpr, tpr, linewidth=2, color='#1971c2', label=f'AUC={auc_value:.4f}')
    axes[0].plot([0, 1], [0, 1], linestyle='--', color='#868e96', linewidth=1)
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curve')
    axes[0].grid(True, linestyle='--', alpha=0.4)
    axes[0].legend()

    axes[1].plot(recall, precision, linewidth=2, color='#2b8a3e', label=f'AP={ap_value:.4f}')
    axes[1].axhline(positive_rate, linestyle='--', color='#868e96', linewidth=1)
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve')
    axes[1].grid(True, linestyle='--', alpha=0.4)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def plot_snr_metrics(rows: list[dict[str, float | int | None]], output_path: Path) -> None:
    snrs = [int(row['snr']) for row in rows]
    accuracy = [float(row['accuracy']) for row in rows]
    balanced_accuracy = [float(row['balanced_accuracy']) for row in rows]
    f1 = [float(row['f1']) for row in rows]
    pd = [float(row['pd']) for row in rows]
    pfa = [float(row['pfa']) for row in rows]
    precision = [float(row['precision']) for row in rows]

    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

    axes[0].plot(snrs, accuracy, marker='o', linewidth=2, label='Accuracy')
    axes[0].plot(snrs, balanced_accuracy, marker='s', linewidth=2, label='Balanced Acc')
    axes[0].plot(snrs, f1, marker='^', linewidth=2, label='F1')
    axes[0].set_ylabel('Score')
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_title('Overall Performance by Source SNR')
    axes[0].grid(True, linestyle='--', alpha=0.4)
    axes[0].legend(ncol=3)

    axes[1].plot(snrs, pd, marker='o', linewidth=2, label='Pd / Recall')
    axes[1].plot(snrs, pfa, marker='s', linewidth=2, label='Pfa')
    axes[1].plot(snrs, precision, marker='^', linewidth=2, label='Precision')
    axes[1].set_xlabel('Source SNR (dB)')
    axes[1].set_ylabel('Score')
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_title('Detection Metrics by Source SNR')
    axes[1].grid(True, linestyle='--', alpha=0.4)
    axes[1].legend(ncol=3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def maybe_save_checkpoint(model, output_path: Path, args, summary: dict) -> None:
    if not args.save_checkpoint:
        return

    payload = {
        'model_name': 'cnr_sensenet',
        'config': {
            'signal_length': summary['dataset']['input_dim'],
            'energy_window': args.energy_window,
            'dropout': args.dropout,
            'lr': args.lr,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'weight_decay': args.weight_decay,
            'threshold': args.decision_threshold,
            'threshold_mode': args.threshold_mode,
            'target_pfa': args.target_pfa,
            'calibration_split': args.calibration_split,
            'snr_loss_weighting': args.snr_loss_weighting,
            'low_snr_cutoff': args.low_snr_cutoff,
            'low_snr_positive_weight': args.low_snr_positive_weight,
            'mid_snr_cutoff': args.mid_snr_cutoff,
            'mid_snr_positive_weight': args.mid_snr_positive_weight,
            'decision_threshold': summary['overall_metrics']['threshold'],
        },
        'metrics': summary['overall_metrics'],
        'state_dict': model.state_dict(),
    }
    torch.save(payload, output_path)


def main() -> None:
    args = build_parser().parse_args()
    seed_everything(args.seed)

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
            y=bundle.train_arrays['y'],
            snr=bundle.train_arrays['snr'],
            test_ratio=args.val_ratio,
            seed=args.seed,
        )
        train_dataset = Subset(bundle.train_dataset, train_idx.tolist())
        val_dataset = Subset(bundle.train_dataset, val_idx.tolist())
    else:
        train_dataset = bundle.train_dataset
        val_dataset = None

    model = create_model(
        'cnr_sensenet',
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
        threshold_mode=args.threshold_mode,
        target_pfa=args.target_pfa,
        calibration_split=args.calibration_split,
        verbose=args.verbose,
    )
    train_seconds = time.perf_counter() - train_start

    eval_start = time.perf_counter()
    test_scores = np.asarray(model.predict_scores(bundle.test_dataset), dtype=np.float64)
    eval_seconds = time.perf_counter() - eval_start

    requested_threshold = None if getattr(model, 'prefers_internal_threshold', False) else args.decision_threshold
    eval_threshold = float(model.get_evaluation_threshold(requested_threshold))

    overall_metrics, test_pred = compute_metrics(
        bundle.test_arrays['y'],
        test_scores,
        threshold=eval_threshold,
    )
    snr_metrics = compute_metrics_by_snr(
        bundle.test_arrays['y'],
        test_scores,
        bundle.test_arrays['snr'],
        threshold=eval_threshold,
    )

    history = getattr(model, 'history', None)
    training_history = {
        'train_loss': list(getattr(history, 'train_loss', [])),
        'val_loss': list(getattr(history, 'val_loss', [])),
    }

    artifact_paths = {
        'summary_json': output_dir / f"{args.output_prefix}_summary.json",
        'metrics_by_snr_csv': output_dir / f"{args.output_prefix}_metrics_by_snr.csv",
        'training_history_png': output_dir / f"{args.output_prefix}_training_history.png",
        'roc_pr_png': output_dir / f"{args.output_prefix}_roc_pr.png",
        'confusion_matrix_png': output_dir / f"{args.output_prefix}_confusion_matrix.png",
        'snr_metrics_png': output_dir / f"{args.output_prefix}_metrics_by_snr.png",
        'checkpoint_pt': output_dir / f"{args.output_prefix}_checkpoint.pt",
    }

    summary = {
        'model': 'cnr_sensenet',
        'device': str(getattr(model, 'device', args.device or 'cpu')),
        'seed': int(args.seed),
        'dataset': {
            'pkl_path': str(bundle.pkl_path),
            'cache_path': str(bundle.cache_path),
            'train_size': int(len(train_dataset)),
            'val_size': int(len(val_dataset)) if val_dataset is not None else 0,
            'test_size': int(len(bundle.test_dataset)),
            'input_dim': int(bundle.input_dim),
            'num_classes': int(bundle.num_classes),
            'noise_power': float(bundle.noise_power),
            'snrs': [int(value) for value in bundle.snrs],
            'mods': list(bundle.mods),
        },
        'train_config': {
            'epochs': int(args.epochs),
            'batch_size': int(args.batch_size),
            'lr': float(args.lr),
            'weight_decay': float(args.weight_decay),
            'dropout': float(args.dropout),
            'energy_window': int(args.energy_window),
            'decision_threshold': float(args.decision_threshold),
            'threshold_mode': args.threshold_mode,
            'target_pfa': float(args.target_pfa),
            'calibration_split': args.calibration_split,
            'snr_loss_weighting': args.snr_loss_weighting,
            'low_snr_cutoff': int(args.low_snr_cutoff),
            'low_snr_positive_weight': float(args.low_snr_positive_weight),
            'mid_snr_cutoff': int(args.mid_snr_cutoff),
            'mid_snr_positive_weight': float(args.mid_snr_positive_weight),
        },
        'timing_seconds': {
            'train': float(train_seconds),
            'evaluate': float(eval_seconds),
        },
        'training_history': training_history,
        'overall_metrics': overall_metrics,
        'metrics_by_snr': snr_metrics,
        'artifacts': {name: str(path) for name, path in artifact_paths.items()},
    }

    write_json(artifact_paths['summary_json'], summary)
    write_csv(artifact_paths['metrics_by_snr_csv'], snr_metrics)
    plot_training_history(history, artifact_paths['training_history_png'])
    plot_roc_pr_curves(bundle.test_arrays['y'], test_scores, artifact_paths['roc_pr_png'])
    plot_confusion_matrix_figure(
        bundle.test_arrays['y'],
        test_pred,
        artifact_paths['confusion_matrix_png'],
    )
    plot_snr_metrics(snr_metrics, artifact_paths['snr_metrics_png'])
    maybe_save_checkpoint(model, artifact_paths['checkpoint_pt'], args, summary)

    print(f'Output directory: {output_dir}')
    print(
        'Overall metrics: '
        f"acc={overall_metrics['accuracy']:.4f}, "
        f"bal_acc={overall_metrics['balanced_accuracy']:.4f}, "
        f"precision={overall_metrics['precision']:.4f}, "
        f"recall={overall_metrics['recall']:.4f}, "
        f"f1={overall_metrics['f1']:.4f}, "
        f"roc_auc={overall_metrics['roc_auc']:.4f}, "
        f"ap={overall_metrics['average_precision']:.4f}, "
        f"pfa={overall_metrics['pfa']:.4f}"
    )
    print(
        f'Timing: train={train_seconds:.2f}s, evaluate={eval_seconds:.2f}s, '
        f'threshold={eval_threshold:.4f}'
    )
    print('Saved artifacts:')
    for name, path in artifact_paths.items():
        if name == 'checkpoint_pt' and not args.save_checkpoint:
            continue
        print(f'  {name}: {path}')


if __name__ == '__main__':
    main()
