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


DEFAULT_MODELS = ['cnr_sensenet', 'mlp', 'cnn1d', 'lstm']
MODEL_LABELS = {
    'cnr_sensenet': 'CNR-SenseNet',
    'mlp': 'MLP',
    'cnn1d': 'CNN1D',
    'lstm': 'LSTM',
}
MODEL_COLORS = {
    'cnr_sensenet': '#0b7285',
    'mlp': '#e8590c',
    'cnn1d': '#5f3dc4',
    'lstm': '#2b8a3e',
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Train multiple models on the same split and save comparison plots.'
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
    parser.add_argument('--patience', type=int, default=2)
    parser.add_argument('--decision-threshold', type=float, default=0.5)
    parser.add_argument('--device', default=None)
    parser.add_argument('--num-threads', type=int, default=None)
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=resolve_path('plots', 'model_comparison'),
    )
    parser.add_argument('--output-prefix', default='model_comparison')
    return parser


def ensure_supported_models(models: list[str]) -> list[str]:
    allowed = set(DEFAULT_MODELS)
    invalid = [name for name in models if name not in allowed]
    if invalid:
        raise ValueError(f'Unsupported models for this script: {invalid}')
    seen = []
    for name in models:
        if name not in seen:
            seen.append(name)
    return seen


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
        common_kwargs['dropout'] = args.dropout
    if model_name == 'cnr_sensenet':
        common_kwargs['energy_window'] = args.energy_window
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
        values = [item['overall_metrics'][metric_key] for item in results]
        axes[0].bar(offsets, values, width=width, label=metric_label)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_title('Overall Performance Comparison')
    axes[0].grid(True, axis='y', linestyle='--', alpha=0.35)
    axes[0].legend(ncol=2)

    for idx, (metric_key, metric_label) in enumerate(detection_metrics):
        offsets = x + (idx - 1.5) * width
        values = [item['overall_metrics'][metric_key] for item in results]
        axes[1].bar(offsets, values, width=width, label=metric_label)
    axes[1].set_xticks(x, labels)
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_title('Detection Metric Comparison')
    axes[1].grid(True, axis='y', linestyle='--', alpha=0.35)
    axes[1].legend(ncol=2)

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
    axes[0].legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def plot_training_histories(results: list[dict], output_path: Path) -> None:
    rows = 2
    cols = 2
    fig, axes = plt.subplots(rows, cols, figsize=(14, 10))
    axes = axes.flatten()

    for axis, model_result in zip(axes, results):
        train_loss = model_result['training_history']['train_loss']
        val_loss = model_result['training_history']['val_loss']
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
            axis.plot(epochs[: len(val_loss)], val_loss, marker='s', linewidth=2, color='#495057', label='Val')
        axis.set_title(MODEL_LABELS[model_result['model']])
        axis.set_xlabel('Epoch')
        axis.set_ylabel('BCE Loss')
        axis.grid(True, linestyle='--', alpha=0.35)
        axis.legend()

    for axis in axes[len(results) :]:
        axis.axis('off')

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
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
    else:
        train_dataset = bundle.train_dataset
        val_dataset = None

    results = []
    overall_rows = []
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
        test_scores = np.asarray(model.predict_scores(bundle.test_dataset), dtype=np.float64)
        infer_seconds = time.perf_counter() - infer_start

        overall_metrics, _ = compute_metrics(
            bundle.test_arrays['y'],
            test_scores,
            threshold=args.decision_threshold,
        )
        metrics_by_snr = compute_metrics_by_snr(
            bundle.test_arrays['y'],
            test_scores,
            bundle.test_arrays['snr'],
            threshold=args.decision_threshold,
        )
        history = getattr(model, 'history', None)
        result = {
            'model': model_name,
            'label': MODEL_LABELS[model_name],
            'parameter_count': count_parameters(model),
            'train_seconds': float(train_seconds),
            'infer_seconds': float(infer_seconds),
            'training_history': {
                'train_loss': list(getattr(history, 'train_loss', [])),
                'val_loss': list(getattr(history, 'val_loss', [])),
            },
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
        print(
            f"{MODEL_LABELS[model_name]} | acc={overall_metrics['accuracy']:.4f} "
            f"f1={overall_metrics['f1']:.4f} roc_auc={overall_metrics['roc_auc']:.4f} "
            f"pfa={overall_metrics['pfa']:.4f} time={train_seconds:.2f}s"
        )

    artifact_paths = {
        'summary_json': output_dir / f'{args.output_prefix}_summary.json',
        'summary_csv': output_dir / f'{args.output_prefix}_summary.csv',
        'overall_png': output_dir / f'{args.output_prefix}_overall_metrics.png',
        'snr_png': output_dir / f'{args.output_prefix}_snr_curves.png',
        'training_png': output_dir / f'{args.output_prefix}_training_curves.png',
    }

    payload = {
        'config': {
            'models': args.models,
            'seed': int(args.seed),
            'epochs': int(args.epochs),
            'batch_size': int(args.batch_size),
            'lr': float(args.lr),
            'weight_decay': float(args.weight_decay),
            'decision_threshold': float(args.decision_threshold),
            'test_ratio': float(args.test_ratio),
            'val_ratio': float(args.val_ratio),
            'noise_power': float(args.noise_power),
            'device': args.device or ('cuda' if torch.cuda.is_available() else 'cpu'),
            'dataset_cache': str(bundle.cache_path),
        },
        'results': results,
        'artifacts': {key: str(value) for key, value in artifact_paths.items()},
    }
    write_json(artifact_paths['summary_json'], payload)
    write_csv(artifact_paths['summary_csv'], overall_rows)
    plot_overall_comparison(results, artifact_paths['overall_png'])
    plot_snr_comparison(results, artifact_paths['snr_png'])
    plot_training_histories(results, artifact_paths['training_png'])

    print(f'\nOutput directory: {output_dir}')
    for name, path in artifact_paths.items():
        print(f'  {name}: {path}')


if __name__ == '__main__':
    main()
