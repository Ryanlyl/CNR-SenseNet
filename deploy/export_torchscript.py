from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from project.models import create_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a trained checkpoint to a TorchScript artifact for edge deployment."
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to a saved training checkpoint.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where model.torchscript.pt and metadata.json will be written.",
    )
    parser.add_argument("--device", default="cpu", help="Export device. CPU is recommended for edge deployment.")
    parser.add_argument("--num-threads", type=int, default=None, help="Optional Torch CPU thread count.")
    return parser


def load_checkpoint(path: Path) -> dict[str, Any]:
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


def build_model_from_checkpoint(checkpoint: dict[str, Any], device: str):
    config = dict(checkpoint.get("config", {}))
    if checkpoint.get("signal_length") is not None and config.get("signal_length") is None:
        config["signal_length"] = int(checkpoint["signal_length"])
    config["device"] = device

    model = create_model(checkpoint["model_name"], **config)
    model.load_state_dict(checkpoint["state_dict"])

    network = getattr(model, "model", None)
    if network is None:
        raise TypeError(
            f"Checkpoint model '{checkpoint['model_name']}' does not expose a torch.nn.Module runtime."
        )
    network = network.to(device)
    network.eval()
    return model, network, config


def resolve_threshold(checkpoint: dict[str, Any], config: dict[str, Any]) -> float:
    if config.get("decision_threshold") is not None:
        return float(config["decision_threshold"])
    metrics = checkpoint.get("metrics")
    if isinstance(metrics, dict) and metrics.get("threshold") is not None:
        return float(metrics["threshold"])
    if config.get("threshold") is not None:
        return float(config["threshold"])
    return 0.5


def build_metadata(
    checkpoint_path: Path,
    model_path: Path,
    checkpoint: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    signal_length = int(config.get("signal_length") or checkpoint.get("signal_length") or 0)
    if signal_length <= 0:
        raise ValueError("signal_length could not be inferred from the checkpoint.")

    return {
        "artifact_format": "cnr-sensenet-edge-v1",
        "model_name": str(checkpoint["model_name"]),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "exported_model_path": str(model_path.resolve()),
        "signal_length": signal_length,
        "num_iq_samples": signal_length // 2,
        "input_layout": "interleaved_iq",
        "score_activation": "sigmoid",
        "decision_threshold": resolve_threshold(checkpoint, config),
        "config": config,
    }


def main() -> None:
    args = build_parser().parse_args()
    if args.num_threads is not None:
        torch.set_num_threads(int(args.num_threads))

    checkpoint_path = args.checkpoint.resolve()
    checkpoint = load_checkpoint(checkpoint_path)
    _, network, config = build_model_from_checkpoint(checkpoint, device=args.device)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.torchscript.pt"
    metadata_path = output_dir / "metadata.json"

    scripted = torch.jit.script(network)
    frozen = torch.jit.freeze(scripted)
    frozen.save(str(model_path))

    metadata = build_metadata(
        checkpoint_path=checkpoint_path,
        model_path=model_path,
        checkpoint=checkpoint,
        config=config,
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Model name: {checkpoint['model_name']}")
    print(f"Exported model: {model_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Decision threshold: {metadata['decision_threshold']:.6f}")


if __name__ == "__main__":
    main()
