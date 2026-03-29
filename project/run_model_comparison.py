from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib
import numpy as np
import torch
from torch.utils.data import Subset

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

matplotlib.use('Agg')
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


DEFAULT_OUTPUT_DIR = resolve_path('plots', 'model_comparison')
SMOKE_OUTPUT_DIR = resolve_path('plots', 'model_comparison_smoke')
DEFAULT_MODELS = [
    'energy_detector',
    'autocorr_detector',
    'mlp',
    'cnn1d',
    'lstm',
    'cnr_sensenet',
]
MODEL_ALIASES = {
    'ed': 'energy_detector',
}
MODEL_LABELS = {
    'energy_detector': 'Energy Detector',
    'autocorr_detector': 'Autocorr Detector',
    'mlp': 'MLP',
    'cnn1d': 'CNN1D',
    'lstm': 'LSTM',
    'cnr_sensenet': 'CNR-SenseNet',
}
MODEL_COLORS = {
    'energy_detector': '#c92a2a',
    'autocorr_detector': '#1c7ed6',
    'mlp': '#e8590c',
    'cnn1d': '#5f3dc4',
    'lstm': '#2b8a3e',
    'cnr_sensenet': '#0b7285',
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Train/evaluate all baselines and CNR-SenseNet on the same split.'
    )
    parser.add_argument('--models', nargs='*', default=DEFAULT_MODELS)
    parser.add_argument('--pkl-path', type=Path, default=default_rml2016a_path())
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test-ratio', type=float, default=0.2)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--noise-power', type=float, default=1.0)
    parser.add_argument('--snr-filter', nargs='*', type=int, default=None)
    parser.add_argument('--selected-mods', nargs='*', default=None)
    parser.add_argument('--cache-path', type=Path, default=None)
    parser.add_argument('--force-rebuild', action='store_true')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--energy-window', type=int, default=8)
    parser.add_argument('--aux-branch-type', choices=['diff', 'autocorr'], default='autocorr')
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
    parser.add_argument('--score-batch-size', type=int, default=16384)
    parser.add_argument(
        '--energy-statistic',
        choices=['avg_power', 'mean', 'sum_energy', 'sum'],
        default='avg_power',
    )
    parser.add_argument('--autocorr-max-lag', type=int, default=4)
    parser.add_argument(
        '--autocorr-score-mode',
        choices=['sum_abs_lags', 'max_abs_lag'],
        default='sum_abs_lags',
    )
    parser.add_argument('--patience', type=int, default=2)
    parser.add_argument('--decision-threshold', type=float, default=0.5)
    parser.add_argument('--device', default=None)
    parser.add_argument('--num-threads', type=int, default=None)
    parser.add_argument('--max-train-samples', type=int, default=None)
    parser.add_argument('--max-val-samples', type=int, default=None)
    parser.add_argument('--max-test-samples', type=int, default=None)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument('--output-prefix', default='model_comparison')
    return parser


def ensure_supported_models(models: list[str]) -> list[str]:
    normalized = [MODEL_ALIASES.get(name, name) for name in models]
    allowed = set(MODEL_LABELS)
    invalid = [name for name in normalized if name not in allowed]
    if invalid:
        raise ValueError(f'Unsupported models for this script: {invalid}')
    seen = []
    for name in normalized:
        if name not in seen:
            seen.append(name)
    return seen


def apply_smoke_overrides(args) -> None:
    if not args.smoke:
        return

    args.epochs = 1
    args.batch_size = min(int(args.batch_size), 128)
    args.max_train_samples = args.max_train_samples or 512
    args.max_val_samples = args.max_val_samples or 256
    args.max_test_samples = args.max_test_samples or 256
    if Path(args.output_dir) == DEFAULT_OUTPUT_DIR:
        args.output_dir = SMOKE_OUTPUT_DIR
    if args.output_prefix == 'model_comparison':
        args.output_prefix = 'model_comparison_smoke'


def build_model(model_name: str, args, signal_length: int):
    common_kwargs = {
        'signal_length': signal_length,
        'lr': args.lr,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'weight_decay': args.weight_decay,
        'device': args.device,
    }
    if model_name in {'cnr_sensenet', 'mlp', 'cnn1d', 'lstm'}:
        common_kwargs.update(
            dropout=args.dropout,
            threshold=args.decision_threshold,
            threshold_mode=args.threshold_mode,
            target_pfa=args.target_pfa,
            calibration_split=args.calibration_split,
        )
    if model_name in {'energy_detector', 'autocorr_detector'}:
        common_kwargs.update(
            threshold_mode=args.threshold_mode,
            target_pfa=args.target_pfa,
            calibration_split=args.calibration_split,
            score_batch_size=args.score_batch_size,
        )
    if model_name == 'energy_detector':
        common_kwargs['statistic'] = args.energy_statistic
    if model_name == 'autocorr_detector':
        common_kwargs['max_lag'] = args.autocorr_max_lag
        common_kwargs['score_mode'] = args.autocorr_score_mode
    if model_name == 'cnr_sensenet':
        common_kwargs['energy_window'] = args.energy_window
        common_kwargs['aux_branch_type'] = args.aux_branch_type
        common_kwargs['autocorr_max_lag'] = args.autocorr_max_lag
        common_kwargs['snr_loss_weighting'] = args.snr_loss_weighting
        common_kwargs['low_snr_cutoff'] = args.low_snr_cutoff
        common_kwargs['low_snr_positive_weight'] = args.low_snr_positive_weight
        common_kwargs['mid_snr_cutoff'] = args.mid_snr_cutoff
        common_kwargs['mid_snr_positive_weight'] = args.mid_snr_positive_weight
    return create_model(model_name, **common_kwargs)


def count_parameters(model) -> int:
    network = getattr(model, 'model', None)
    if network is None:
        return 0
    return int(sum(parameter.numel() for parameter in network.parameters()))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    with path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def metric_to_float(value) -> float:
    if value is None:
        return float('nan')
    try:
        return float(value)
    except (TypeError, ValueError):
        return float('nan')


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
    return {
        key: np.asarray(value)[indices]
        for key, value in arrays.items()
    }


def maybe_limit_dataset(dataset, arrays: dict[str, np.ndarray], max_samples: int | None, seed: int):
    indices = stratified_subsample_indices(
        y=np.asarray(arrays['y']),
        snr=np.asarray(arrays['snr']),
        max_samples=max_samples,
        seed=seed,
    )
    if len(indices) == len(arrays['y']):
        return dataset, arrays
    return Subset(dataset, indices.tolist()), subset_arrays(arrays, indices)


def resolve_eval_threshold(model, default_threshold: float) -> float:
    requested_threshold = None if getattr(model, 'prefers_internal_threshold', False) else default_threshold
    return float(model.get_evaluation_threshold(requested_threshold))


def extract_training_history(model) -> dict[str, list[float]]:
    history = getattr(model, 'history', None)
    return {
        'train_loss': list(getattr(history, 'train_loss', [])),
        'val_loss': list(getattr(history, 'val_loss', [])),
    }


def extract_calibration(model):
    fit_result = getattr(model, 'fit_result', None)
    return None if fit_result is None else asdict(fit_result)


def plot_overall_comparison(results: list[dict], output_path: Path) -> None:
    model_names = [item['model'] for item in results]
    labels = [MODEL_LABELS[name] for name in model_names]
    x = np.arange(len(labels), dtype=np.float64)
    width = 0.16

    performance_metrics = [
        ('accuracy', 'Accuracy'),
        ('balanced_accuracy', 'Balanced Acc'),
        ('f1', 'F1'),
        ('roc_auc', 'ROC-AUC'),
    ]
    detection_metrics = [
        ('precision', 'Precision'),
        ('pd', 'Pd / Recall'),
        ('pfa', 'Pfa'),
        ('average_precision', 'AP'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for idx, (metric_key, metric_label) in enumerate(performance_metrics):
        offsets = x + (idx - 1.5) * width
        values = [metric_to_float(item['overall_metrics'][metric_key]) for item in results]
        axes[0].bar(offsets, values, width=width, label=metric_label)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_title('Overall Performance Comparison')
    axes[0].grid(True, axis='y', linestyle='--', alpha=0.35)
    axes[0].legend(ncol=2)

    for idx, (metric_key, metric_label) in enumerate(detection_metrics):
        offsets = x + (idx - 1.5) * width
        values = [metric_to_float(item['overall_metrics'][metric_key]) for item in results]
        axes[1].bar(offsets, values, width=width, label=metric_label)
    axes[1].set_xticks(x, labels)
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_title('Detection Metric Comparison')
    axes[1].grid(True, axis='y', linestyle='--', alpha=0.35)
    axes[1].legend(ncol=2)

    for axis in axes:
        axis.tick_params(axis='x', rotation=15)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def plot_snr_comparison(results: list[dict], output_path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True)
    metric_specs = [
        ('accuracy', 'Accuracy by Source SNR'),
        ('pd', 'Pd by Source SNR'),
        ('pfa', 'Pfa by Source SNR'),
    ]

    for model_result in results:
        model_name = model_result['model']
        rows = model_result['metrics_by_snr']
        snrs = [row['snr'] for row in rows]
        for axis, (metric_key, title) in zip(axes, metric_specs):
            values = [row[metric_key] for row in rows]
            axis.plot(
                snrs,
                values,
                marker='o',
                linewidth=2,
                label=MODEL_LABELS[model_name],
                color=MODEL_COLORS[model_name],
            )
            axis.set_ylim(0.0, 1.05)
            axis.set_ylabel('Score')
            axis.set_title(title)
            axis.grid(True, linestyle='--', alpha=0.35)

    axes[-1].set_xlabel('Source SNR (dB)')
    axes[0].legend(ncol=min(3, max(1, len(results))))
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def plot_training_histories(results: list[dict], output_path: Path) -> None:
    if not results:
        return

    cols = 2 if len(results) > 1 else 1
    rows = math.ceil(len(results) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4 * rows))
    axes = np.atleast_1d(axes).flatten()

    for axis, model_result in zip(axes, results):
        train_loss = model_result['training_history']['train_loss']
        val_loss = model_result['training_history']['val_loss']
        axis.set_title(MODEL_LABELS[model_result['model']])

        if train_loss:
            epochs = np.arange(1, len(train_loss) + 1)
            axis.plot(
                epochs,
                train_loss,
                marker='o',
                linewidth=2,
                color=MODEL_COLORS[model_result['model']],
                label='Train',
            )
            if val_loss:
                axis.plot(
                    epochs[: len(val_loss)],
                    val_loss,
                    marker='s',
                    linewidth=2,
                    color='#495057',
                    label='Val',
                )
            axis.set_xlabel('Epoch')
            axis.set_ylabel('BCE Loss')
            axis.grid(True, linestyle='--', alpha=0.35)
            axis.legend()
            continue

        calibration = model_result.get('calibration')
        detail = 'No training curve'
        if calibration is not None:
            detail = (
                'Threshold calibration only\n'
                f"thr={calibration['threshold']:.4f} "
                f"({calibration['calibration_source']})"
            )
        axis.text(0.5, 0.5, detail, ha='center', va='center', fontsize=11)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.grid(False)

    for axis in axes[len(results) :]:
        axis.axis('off')

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    apply_smoke_overrides(args)
    args.models = ensure_supported_models(args.models)
    seed_everything(args.seed)
    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)

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
        seed=args.seed,
    )
    if val_dataset is not None and val_arrays is not None:
        val_dataset, val_arrays = maybe_limit_dataset(
            val_dataset,
            val_arrays,
            max_samples=args.max_val_samples,
            seed=args.seed + 1,
        )
    test_dataset, test_arrays = maybe_limit_dataset(
        bundle.test_dataset,
        {key: np.asarray(value) for key, value in bundle.test_arrays.items()},
        max_samples=args.max_test_samples,
        seed=args.seed + 2,
    )

    results = []
    overall_rows = []
    snr_rows = []
    for model_name in args.models:
        print(f'\n=== Training {MODEL_LABELS[model_name]} ===')
        model = build_model(model_name, args, signal_length=bundle.input_dim)

        train_start = time.perf_counter()
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
        train_seconds = time.perf_counter() - train_start

        infer_start = time.perf_counter()
        test_scores = np.asarray(model.predict_scores(test_dataset), dtype=np.float64)
        infer_seconds = time.perf_counter() - infer_start
        eval_threshold = resolve_eval_threshold(model, args.decision_threshold)

        overall_metrics, _ = compute_metrics(
            test_arrays['y'],
            test_scores,
            threshold=eval_threshold,
        )
        metrics_by_snr = compute_metrics_by_snr(
            test_arrays['y'],
            test_scores,
            test_arrays['snr'],
            threshold=eval_threshold,
        )
        result = {
            'model': model_name,
            'label': MODEL_LABELS[model_name],
            'config': model.get_config(),
            'parameter_count': count_parameters(model),
            'train_seconds': float(train_seconds),
            'infer_seconds': float(infer_seconds),
            'training_history': extract_training_history(model),
            'calibration': extract_calibration(model),
            'overall_metrics': overall_metrics,
            'metrics_by_snr': metrics_by_snr,
        }
        results.append(result)
        overall_rows.append(
            {
                'model': MODEL_LABELS[model_name],
                'params': result['parameter_count'],
                'train_seconds': result['train_seconds'],
                'infer_seconds': result['infer_seconds'],
                'threshold': overall_metrics['threshold'],
                'calibration_source': None
                if result['calibration'] is None
                else result['calibration']['calibration_source'],
                'accuracy': overall_metrics['accuracy'],
                'balanced_accuracy': overall_metrics['balanced_accuracy'],
                'precision': overall_metrics['precision'],
                'pd': overall_metrics['pd'],
                'f1': overall_metrics['f1'],
                'pfa': overall_metrics['pfa'],
                'roc_auc': overall_metrics['roc_auc'],
                'average_precision': overall_metrics['average_precision'],
            }
        )
        for row in metrics_by_snr:
            snr_rows.append(
                {
                    'model': MODEL_LABELS[model_name],
                    'snr': row['snr'],
                    'threshold': overall_metrics['threshold'],
                    'accuracy': row['accuracy'],
                    'balanced_accuracy': row['balanced_accuracy'],
                    'precision': row['precision'],
                    'pd': row['pd'],
                    'f1': row['f1'],
                    'pfa': row['pfa'],
                    'roc_auc': row['roc_auc'],
                    'average_precision': row['average_precision'],
                    'count': row['count'],
                    'signal_count': row['signal_count'],
                    'noise_count': row['noise_count'],
                }
            )
        print(
            f"{MODEL_LABELS[model_name]} | acc={overall_metrics['accuracy']:.4f} "
            f"f1={overall_metrics['f1']:.4f} roc_auc={metric_to_float(overall_metrics['roc_auc']):.4f} "
            f"pfa={overall_metrics['pfa']:.4f} thr={overall_metrics['threshold']:.4f} "
            f"time={train_seconds:.2f}s"
        )

    artifact_paths = {
        'summary_json': output_dir / f'{args.output_prefix}_summary.json',
        'summary_csv': output_dir / f'{args.output_prefix}_summary.csv',
        'snr_csv': output_dir / f'{args.output_prefix}_metrics_by_snr.csv',
        'overall_png': output_dir / f'{args.output_prefix}_overall_metrics.png',
        'snr_png': output_dir / f'{args.output_prefix}_snr_curves.png',
        'training_png': output_dir / f'{args.output_prefix}_training_curves.png',
    }

    payload = {
        'config': {
            'models': args.models,
            'smoke': bool(args.smoke),
            'seed': int(args.seed),
            'epochs': int(args.epochs),
            'batch_size': int(args.batch_size),
            'lr': float(args.lr),
            'weight_decay': float(args.weight_decay),
            'decision_threshold': float(args.decision_threshold),
            'threshold_mode': args.threshold_mode,
            'target_pfa': float(args.target_pfa),
            'calibration_split': args.calibration_split,
            'snr_loss_weighting': args.snr_loss_weighting,
            'low_snr_cutoff': int(args.low_snr_cutoff),
            'low_snr_positive_weight': float(args.low_snr_positive_weight),
            'mid_snr_cutoff': int(args.mid_snr_cutoff),
            'mid_snr_positive_weight': float(args.mid_snr_positive_weight),
            'score_batch_size': int(args.score_batch_size),
            'energy_statistic': args.energy_statistic,
            'aux_branch_type': args.aux_branch_type,
            'autocorr_max_lag': int(args.autocorr_max_lag),
            'autocorr_score_mode': args.autocorr_score_mode,
            'test_ratio': float(args.test_ratio),
            'val_ratio': float(args.val_ratio),
            'noise_power': float(args.noise_power),
            'device': args.device or ('cuda' if torch.cuda.is_available() else 'cpu'),
            'dataset_cache': str(bundle.cache_path),
            'max_train_samples': args.max_train_samples,
            'max_val_samples': args.max_val_samples,
            'max_test_samples': args.max_test_samples,
        },
        'dataset': {
            'pkl_path': str(bundle.pkl_path),
            'cache_path': str(bundle.cache_path),
            'input_dim': int(bundle.input_dim),
            'noise_power': float(bundle.noise_power),
            'mods': list(bundle.mods),
            'snrs': [int(value) for value in bundle.snrs],
            'split_sizes': {
                'train': int(len(train_dataset)),
                'val': int(len(val_dataset)) if val_dataset is not None else 0,
                'test': int(len(test_dataset)),
            },
        },
        'results': results,
        'artifacts': {key: str(value) for key, value in artifact_paths.items()},
    }
    write_json(artifact_paths['summary_json'], payload)
    write_csv(artifact_paths['summary_csv'], overall_rows)
    write_csv(artifact_paths['snr_csv'], snr_rows)
    plot_overall_comparison(results, artifact_paths['overall_png'])
    plot_snr_comparison(results, artifact_paths['snr_png'])
    plot_training_histories(results, artifact_paths['training_png'])

    print(f'\nOutput directory: {output_dir}')
    for name, path in artifact_paths.items():
        print(f'  {name}: {path}')


if __name__ == '__main__':
    main()
