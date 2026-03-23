from __future__ import annotations

import argparse
from pathlib import Path

import torch

from project.data import DataConfig, build_datasets, default_rml2016a_path
from project.models import create_model
from project.run_cnr_sensenet_eval import stratified_index_split_binary
from project.train import (
    evaluate_model,
    find_default_npz,
    load_signal_vs_noise_archive,
    make_dataset,
    save_json,
    set_seed,
    split_indices,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved checkpoint on the prepared signal-vs-noise dataset."
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to a saved .pt checkpoint.")
    parser.add_argument(
        "--dataset-mode",
        choices=["auto", "train", "bundle"],
        default="auto",
        help="How to rebuild the evaluation split.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="test",
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--dataset-npz",
        type=Path,
        default=None,
        help="Prepared .npz file used by checkpoints from project.train.",
    )
    parser.add_argument("--pkl-path", type=Path, default=default_rml2016a_path())
    parser.add_argument("--cache-path", type=Path, default=None)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-power", type=float, default=1.0)
    parser.add_argument("--snr-filter", nargs="*", type=int, default=None)
    parser.add_argument("--selected-mods", nargs="*", default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-threads", type=int, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser


def load_checkpoint(path: Path) -> dict:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")

    if not isinstance(payload, dict):
        raise TypeError(f"Expected checkpoint payload to be a dict, got {type(payload)!r}.")
    if "model_name" not in payload:
        raise KeyError(f"Checkpoint {path} does not contain 'model_name'.")
    if "state_dict" not in payload:
        raise KeyError(f"Checkpoint {path} does not contain 'state_dict'.")
    return payload


def infer_dataset_mode(args, checkpoint: dict) -> str:
    if args.dataset_mode != "auto":
        return args.dataset_mode
    return "bundle" if "metrics" in checkpoint else "train"


def build_model_from_checkpoint(checkpoint: dict, device: str | None):
    config = dict(checkpoint.get("config", {}))
    if checkpoint.get("signal_length") is not None and config.get("signal_length") is None:
        config["signal_length"] = int(checkpoint["signal_length"])
    if device is not None:
        config["device"] = device

    model = create_model(checkpoint["model_name"], **config)
    model.load_state_dict(checkpoint["state_dict"])
    return model, config


def select_train_mode_dataset(args):
    dataset_path = (args.dataset_npz or find_default_npz()).resolve()
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
    split_map = {
        "train": make_dataset(arrays, train_idx),
        "test": make_dataset(arrays, test_idx),
    }
    if len(val_idx) > 0:
        split_map["val"] = make_dataset(arrays, val_idx)
    if args.split not in split_map:
        raise ValueError(f"Split '{args.split}' is unavailable with val_ratio={args.val_ratio}.")

    return split_map[args.split], {
        "dataset_mode": "train",
        "dataset_npz": str(dataset_path),
        "split": args.split,
        "test_ratio": float(args.test_ratio),
        "val_ratio": float(args.val_ratio),
        "seed": int(args.seed),
    }


def select_bundle_mode_dataset(args):
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

    if args.split == "test":
        dataset = bundle.test_dataset
    elif args.split == "train":
        dataset = bundle.train_dataset
    else:
        if args.val_ratio <= 0.0:
            raise ValueError("Split 'val' requires val_ratio > 0.")
        _, val_idx = stratified_index_split_binary(
            y=bundle.train_arrays["y"],
            snr=bundle.train_arrays["snr"],
            test_ratio=args.val_ratio,
            seed=args.seed,
        )
        dataset = make_dataset(bundle.train_arrays, val_idx)

    return dataset, {
        "dataset_mode": "bundle",
        "pkl_path": str(bundle.pkl_path),
        "cache_path": str(bundle.cache_path),
        "split": args.split,
        "test_ratio": float(args.test_ratio),
        "val_ratio": float(args.val_ratio),
        "seed": int(args.seed),
        "noise_power": float(args.noise_power),
        "snr_filter": None if args.snr_filter is None else [int(value) for value in args.snr_filter],
        "selected_mods": None if args.selected_mods is None else list(args.selected_mods),
    }


def build_output_payload(
    checkpoint_path: Path,
    checkpoint: dict,
    config: dict,
    dataset_info: dict,
    result: dict,
) -> dict:
    return {
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "model_name": checkpoint["model_name"],
            "config": config,
        },
        "dataset": dataset_info,
        "evaluation": result,
    }


def print_summary(payload: dict) -> None:
    metrics = payload["evaluation"]["metrics"]
    print(f"Checkpoint: {payload['checkpoint']['path']}")
    print(f"Model: {payload['checkpoint']['model_name']}")
    print(f"Dataset mode: {payload['dataset']['dataset_mode']} | split: {payload['dataset']['split']}")
    print(
        f"Metrics: acc={metrics['accuracy']:.4f} "
        f"bal_acc={metrics['balanced_accuracy']:.4f} "
        f"precision={metrics['precision']:.4f} "
        f"recall={metrics['recall']:.4f} "
        f"f1={metrics['f1']:.4f} "
        f"roc_auc={metrics['roc_auc']:.4f} "
        f"Pfa={metrics['Pfa']:.4f}"
    )


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)

    checkpoint_path = args.checkpoint.resolve()
    checkpoint = load_checkpoint(checkpoint_path)
    dataset_mode = infer_dataset_mode(args, checkpoint)
    model, config = build_model_from_checkpoint(checkpoint, device=args.device)

    if dataset_mode == "train":
        dataset, dataset_info = select_train_mode_dataset(args)
    else:
        dataset, dataset_info = select_bundle_mode_dataset(args)

    result = evaluate_model(model, dataset, threshold=args.threshold)
    payload = build_output_payload(
        checkpoint_path=checkpoint_path,
        checkpoint=checkpoint,
        config=config,
        dataset_info=dataset_info,
        result=result,
    )
    print_summary(payload)

    if args.output_json is not None:
        save_json(args.output_json, payload)
        print(f"Saved JSON: {args.output_json}")


if __name__ == "__main__":
    main()
