"""Visualize representative signal/noise samples from the rebuilt dataset."""

import argparse
import os
from pathlib import Path

import matplotlib
import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from project.data import default_rml2016a_path
from project.data.gen_dataset import compute_power, prepare_signal_vs_noise_dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_parser():
    parser = argparse.ArgumentParser(
        description="Visualize representative samples from the fused signal/noise dataset."
    )
    parser.add_argument(
        "--pkl-path",
        type=Path,
        default=default_rml2016a_path(),
        help="Path to the original RML2016.10a pickle file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "plots" / "dataset_sample_overview.png",
        help="Path to save the output figure.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sample selection.")
    parser.add_argument(
        "--noise-power",
        type=float,
        default=1.0,
        help="Fixed noise power used when preparing the fused dataset.",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Force rebuilding the cached fused dataset before visualization.",
    )
    return parser


def snr_bucket(snr_value):
    if snr_value <= -6:
        return "low"
    if snr_value >= 6:
        return "high"
    return "mid"


def select_representative_indices(meta, seed=42):
    rng = np.random.default_rng(seed)
    candidates = np.arange(len(meta))

    specs = [
        {"type": "signal", "bucket": "low", "mod": "BPSK"},
        {"type": "signal", "bucket": "low", "mod": "WBFM"},
        {"type": "signal", "bucket": "mid", "mod": "QPSK"},
        {"type": "signal", "bucket": "mid", "mod": "AM-DSB"},
        {"type": "signal", "bucket": "high", "mod": "8PSK"},
        {"type": "signal", "bucket": "high", "mod": "QAM16"},
        {"type": "noise", "bucket": "low", "mod": "noise"},
        {"type": "noise", "bucket": "mid", "mod": "noise"},
        {"type": "noise", "bucket": "high", "mod": "noise"},
        {"type": "signal", "bucket": "high", "mod": "CPFSK"},
    ]

    selected = []
    used = set()

    for spec in specs:
        idx = pick_one(meta, candidates, used, rng, spec)
        if idx is not None:
            selected.append(idx)
            used.add(idx)

    if len(selected) < 10:
        for idx in candidates:
            if idx in used:
                continue
            selected.append(int(idx))
            used.add(int(idx))
            if len(selected) == 10:
                break

    return selected[:10]


def pick_one(meta, candidates, used, rng, spec):
    matched = []
    relaxed = []

    for idx in candidates:
        idx = int(idx)
        if idx in used:
            continue

        item = meta[idx]
        if item["type"] != spec["type"]:
            continue

        if snr_bucket(item["source_snr"]) == spec["bucket"]:
            relaxed.append(idx)
            if item["mod"] == spec["mod"]:
                matched.append(idx)

    if matched:
        return int(rng.choice(matched))
    if relaxed:
        return int(rng.choice(relaxed))
    return None


def plot_samples(X, meta, indices, output_path, cache_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 5, figsize=(22, 9), constrained_layout=True)
    axes = axes.flatten()

    for axis, idx in zip(axes, indices):
        sample = X[idx]
        item = meta[idx]
        i_part = sample[0::2]
        q_part = sample[1::2]
        color = "#0b7285" if item["type"] == "signal" else "#c92a2a"

        axis.plot(i_part, color="#1c7ed6", linewidth=1.1, label="I")
        axis.plot(q_part, color="#f08c00", linewidth=1.1, label="Q")
        axis.axhline(0.0, color="#adb5bd", linewidth=0.8, linestyle="--")
        axis.set_xlim(0, len(i_part) - 1)
        axis.tick_params(labelsize=8)
        axis.set_title(
            (
                f"{item['type'].upper()} | {item['mod']}\n"
                f"label_snr={item['label_snr']} dB | source_snr={item['source_snr']} dB"
            ),
            fontsize=10,
            color=color,
        )
        axis.text(
            0.02,
            0.95,
            f"power={compute_power(sample):.3f}",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.85, "edgecolor": color},
        )
        for spine in axis.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(1.2)

    for axis in axes[10:]:
        axis.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    # fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    # fig.suptitle(
    #     "Representative Samples from Rebuilt RML2016.10a Signal/Noise Dataset\n"
    #     f"Waveform view (I/Q interleaved source), cache: {Path(cache_path).name}",
    #     fontsize=14,
    # )
    fig.supxlabel("Time index")
    fig.supylabel("Amplitude")
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    args = build_parser().parse_args()
    X, y, snr, meta, cache_path = prepare_signal_vs_noise_dataset(
        pkl_path=args.pkl_path,
        seed=args.seed,
        noise_power=args.noise_power,
        use_cache=True,
        force_rebuild=args.force_rebuild,
    )
    del y, snr

    indices = select_representative_indices(meta, seed=args.seed)
    plot_samples(X, meta, indices, args.output, cache_path)

    print(f"Saved visualization to: {args.output}")
    print(f"Using cache file: {cache_path}")
    print("Selected labels:")
    for idx in indices:
        item = meta[idx]
        print(
            f"  idx={idx:<7d} type={item['type']:<6} mod={item['mod']:<7} "
            f"label_snr={item['label_snr']:>3} source_snr={item['source_snr']:>3}"
        )


if __name__ == "__main__":
    main()
