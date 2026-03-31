from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from deploy.runtime import DEFAULT_INPUT_KEY, EdgeModel


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run batch inference with an exported TorchScript edge artifact."
    )
    parser.add_argument("--model", type=Path, required=True, help="Path to model.torchscript.pt")
    parser.add_argument("--metadata", type=Path, default=None, help="Optional metadata.json path.")
    parser.add_argument("--input", type=Path, required=True, help="Input .npy or .npz file.")
    parser.add_argument("--input-key", default=DEFAULT_INPUT_KEY, help="Array key for .npz inputs.")
    parser.add_argument("--threshold", type=float, default=None, help="Override decision threshold.")
    parser.add_argument("--batch-size", type=int, default=256, help="Inference batch size.")
    parser.add_argument("--device", default="cpu", help="Runtime device. Default is cpu.")
    parser.add_argument("--num-threads", type=int, default=None, help="Optional Torch CPU thread count.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional output JSON path.")
    parser.add_argument(
        "--include-arrays",
        action="store_true",
        help="When saving JSON, include the full scores and labels arrays.",
    )
    return parser


def build_summary(result: dict) -> dict:
    scores = np.asarray(result["scores"], dtype=np.float32)
    labels = np.asarray(result["labels"], dtype=np.int64)
    return {
        "input_path": result["input_path"],
        "num_samples": int(result["num_samples"]),
        "signal_length": int(result["signal_length"]),
        "threshold": float(result["threshold"]),
        "positive_predictions": int(labels.sum()),
        "negative_predictions": int((labels == 0).sum()),
        "score_min": float(scores.min()) if scores.size else None,
        "score_mean": float(scores.mean()) if scores.size else None,
        "score_max": float(scores.max()) if scores.size else None,
    }


def main() -> None:
    args = build_parser().parse_args()
    runtime = EdgeModel(
        model_path=args.model,
        metadata_path=args.metadata,
        device=args.device,
        num_threads=args.num_threads,
    )
    result = runtime.predict_from_path(
        input_path=args.input,
        input_key=args.input_key,
        threshold=args.threshold,
        batch_size=args.batch_size,
    )
    summary = build_summary(result)

    print(f"Input: {summary['input_path']}")
    print(
        f"Samples: {summary['num_samples']} | signal_length={summary['signal_length']} "
        f"| threshold={summary['threshold']:.6f}"
    )
    print(
        f"Predictions: signal={summary['positive_predictions']} "
        f"noise={summary['negative_predictions']}"
    )
    if summary["score_mean"] is not None:
        print(
            f"Score stats: min={summary['score_min']:.6f} "
            f"mean={summary['score_mean']:.6f} max={summary['score_max']:.6f}"
        )

    if args.output_json is not None:
        payload = {"summary": summary}
        if args.include_arrays:
            payload["scores"] = np.asarray(result["scores"], dtype=np.float32).tolist()
            payload["labels"] = np.asarray(result["labels"], dtype=np.int64).tolist()
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved JSON: {args.output_json}")


if __name__ == "__main__":
    main()
