from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np

from deploy.runtime import DEFAULT_INPUT_KEY, EdgeModel, load_input_array


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark CPU inference latency for an exported TorchScript artifact."
    )
    parser.add_argument("--model", type=Path, required=True, help="Path to model.torchscript.pt")
    parser.add_argument("--metadata", type=Path, default=None, help="Optional metadata.json path.")
    parser.add_argument("--device", default="cpu", help="Runtime device. Default is cpu.")
    parser.add_argument("--num-threads", type=int, default=None, help="Optional Torch CPU thread count.")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations.")
    parser.add_argument("--iterations", type=int, default=100, help="Measured iterations.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for each benchmark step.")
    parser.add_argument("--input", type=Path, default=None, help="Optional .npy or .npz input file.")
    parser.add_argument("--input-key", default=DEFAULT_INPUT_KEY, help="Array key for .npz inputs.")
    return parser


def select_benchmark_input(runtime: EdgeModel, args) -> np.ndarray:
    if args.input is None:
        return np.random.randn(args.batch_size, runtime.signal_length).astype(np.float32)

    values = load_input_array(args.input, input_key=args.input_key)
    prepared = runtime.prepare_array(values)
    if prepared.shape[0] >= args.batch_size:
        return prepared[: args.batch_size]

    repeats = int(np.ceil(args.batch_size / prepared.shape[0]))
    tiled = np.tile(prepared, (repeats, 1))
    return tiled[: args.batch_size]


def percentile(values_ms: list[float], fraction: float) -> float:
    ordered = sorted(values_ms)
    if not ordered:
        return 0.0
    index = int(round((len(ordered) - 1) * fraction))
    return float(ordered[index])


def main() -> None:
    args = build_parser().parse_args()
    runtime = EdgeModel(
        model_path=args.model,
        metadata_path=args.metadata,
        device=args.device,
        num_threads=args.num_threads,
    )
    batch = select_benchmark_input(runtime, args)

    for _ in range(args.warmup):
        runtime.predict_scores_array(batch, batch_size=args.batch_size)

    latencies_ms: list[float] = []
    for _ in range(args.iterations):
        start = time.perf_counter()
        runtime.predict_scores_array(batch, batch_size=args.batch_size)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        latencies_ms.append(float(elapsed_ms))

    avg_ms = float(statistics.mean(latencies_ms))
    std_ms = float(statistics.pstdev(latencies_ms)) if len(latencies_ms) > 1 else 0.0
    p50_ms = percentile(latencies_ms, 0.50)
    p95_ms = percentile(latencies_ms, 0.95)
    throughput = (args.batch_size * 1000.0) / avg_ms if avg_ms > 0.0 else 0.0

    print(f"Model: {runtime.model_path}")
    print(
        f"Benchmark: iterations={args.iterations} warmup={args.warmup} "
        f"batch_size={args.batch_size} signal_length={runtime.signal_length}"
    )
    print(
        f"Latency (ms): avg={avg_ms:.3f} std={std_ms:.3f} "
        f"p50={p50_ms:.3f} p95={p95_ms:.3f}"
    )
    print(f"Throughput: {throughput:.2f} windows/s")


if __name__ == "__main__":
    main()
