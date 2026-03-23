from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from project.data import default_rml2016a_path
from project.data.gen_dataset import prepare_signal_vs_noise_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build or reuse the cached signal-vs-noise dataset archive."
    )
    parser.add_argument("--pkl-path", type=Path, default=default_rml2016a_path())
    parser.add_argument("--cache-path", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-power", type=float, default=1.0)
    parser.add_argument("--snr-filter", nargs="*", type=int, default=None)
    parser.add_argument("--selected-mods", nargs="*", default=None)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--summary-json", type=Path, default=None)
    return parser


def build_summary(
    cache_path: Path,
    X: np.ndarray,
    y: np.ndarray,
    snr: np.ndarray,
    meta: list[dict],
    noise_power: float,
) -> dict:
    signal_mask = y == 1
    mods = sorted({item["mod"] for item in meta if item["type"] == "signal"})
    return {
        "cache_path": str(cache_path),
        "num_samples": int(len(X)),
        "input_dim": int(X.shape[1]),
        "signal_samples": int(signal_mask.sum()),
        "noise_samples": int((~signal_mask).sum()),
        "noise_power": float(noise_power),
        "snr_values": [int(value) for value in sorted(np.unique(snr).tolist())],
        "mods": mods,
    }


def maybe_write_summary(path: Path | None, summary: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    X, y, snr, meta, cache_path = prepare_signal_vs_noise_dataset(
        pkl_path=args.pkl_path,
        snr_filter=args.snr_filter,
        selected_mods=args.selected_mods,
        seed=args.seed,
        noise_power=args.noise_power,
        cache_path=args.cache_path,
        use_cache=True,
        force_rebuild=args.force_rebuild,
    )
    summary = build_summary(
        cache_path=Path(cache_path),
        X=X,
        y=y,
        snr=snr,
        meta=meta,
        noise_power=args.noise_power,
    )
    maybe_write_summary(args.summary_json, summary)

    print(f"Cache path: {summary['cache_path']}")
    print(
        f"Samples: total={summary['num_samples']} signal={summary['signal_samples']} "
        f"noise={summary['noise_samples']} input_dim={summary['input_dim']}"
    )
    print(f"Noise power: {summary['noise_power']}")
    print(f"SNR values: {summary['snr_values']}")
    print(f"Signal mods ({len(summary['mods'])}): {', '.join(summary['mods'])}")
    if args.summary_json is not None:
        print(f"Summary JSON: {args.summary_json}")


if __name__ == "__main__":
    main()
