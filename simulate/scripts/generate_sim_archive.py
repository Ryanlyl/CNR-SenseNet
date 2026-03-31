from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from simulate.core import (
    SimulationBatchConfig,
    add_complex_awgn,
    apply_flat_fading,
    apply_random_phase,
    compose_observation,
    generate_impulsive_interference,
    generate_single_tone,
    interleave_iq,
    sample_snr_values,
    scale_to_average_power,
    synthesize_qpsk,
    target_signal_power,
)
from simulate.schema import (
    default_archive_dir,
    default_manifest_path,
    save_archive,
    write_manifest,
)


SCENARIOS = ("qpsk_awgn", "qpsk_tone_hardneg")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a minimal simulation archive compatible with the current CNR-SenseNet .npz format."
    )
    parser.add_argument("--scenario", choices=SCENARIOS, default="qpsk_awgn")
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--num-samples", type=int, default=4096)
    parser.add_argument("--sample-length", type=int, default=128)
    parser.add_argument("--noise-power", type=float, default=1.0)
    parser.add_argument("--positive-ratio", type=float, default=0.5)
    parser.add_argument("--snr-values", nargs="*", type=int, default=[-20, -10, -4, 0, 6, 12])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tone-probability", type=float, default=0.35)
    parser.add_argument("--tone-amplitude-min", type=float, default=0.2)
    parser.add_argument("--tone-amplitude-max", type=float, default=0.8)
    parser.add_argument("--impulse-probability", type=float, default=0.04)
    parser.add_argument("--impulse-scale", type=float, default=3.0)
    parser.add_argument("--force", action="store_true")
    return parser


def validate_args(args) -> None:
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be greater than 0.")
    if args.sample_length <= 0:
        raise ValueError("--sample-length must be greater than 0.")
    if not 0.0 <= args.positive_ratio <= 1.0:
        raise ValueError("--positive-ratio must be in [0, 1].")
    if args.noise_power <= 0.0:
        raise ValueError("--noise-power must be greater than 0.")
    if not args.snr_values:
        raise ValueError("--snr-values requires at least one SNR label.")
    if args.tone_amplitude_min <= 0.0 or args.tone_amplitude_max <= 0.0:
        raise ValueError("Tone amplitudes must be greater than 0.")
    if args.tone_amplitude_min > args.tone_amplitude_max:
        raise ValueError("--tone-amplitude-min must be <= --tone-amplitude-max.")


def resolve_output_paths(args) -> tuple[Path, Path]:
    output_path = args.output_path or (default_archive_dir() / f"{args.scenario}.npz")
    manifest_path = args.manifest_path or default_manifest_path(output_path)
    return output_path, manifest_path


def build_labels(num_samples: int, positive_ratio: float, rng: np.random.Generator) -> np.ndarray:
    labels = np.zeros(num_samples, dtype=np.int64)
    positive_count = int(round(num_samples * positive_ratio))
    labels[:positive_count] = 1
    rng.shuffle(labels)
    return labels


def append_interference_tag(tags: np.ndarray, indices: np.ndarray, tag: str) -> None:
    for index in indices.tolist():
        current = str(tags[index])
        tags[index] = tag if current == "none" else f"{current}+{tag}"


def generate_archive_arrays(args) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(args.seed)
    batch = SimulationBatchConfig(
        num_samples=args.num_samples,
        sample_length=args.sample_length,
        noise_power=args.noise_power,
        positive_ratio=args.positive_ratio,
        seed=args.seed,
    )

    y_values = build_labels(batch.num_samples, batch.positive_ratio, rng)
    snr_values = sample_snr_values(batch.num_samples, list(args.snr_values), rng)

    signal_components = synthesize_qpsk(batch.num_samples, batch.sample_length, rng)
    signal_components = apply_random_phase(signal_components, rng)
    signal_components = apply_flat_fading(signal_components, rng)
    signal_components = scale_to_average_power(
        signal_components,
        target_signal_power(snr_values, batch.noise_power),
    )

    noise_components = add_complex_awgn(
        (batch.num_samples, batch.sample_length),
        noise_power=batch.noise_power,
        rng=rng,
    )
    observations = noise_components.copy()
    positive_idx = np.where(y_values == 1)[0]
    if positive_idx.size > 0:
        observations[positive_idx] = compose_observation(
            signal_components[positive_idx],
            noise_components[positive_idx],
        )

    interference_type = np.full(batch.num_samples, "none", dtype="<U32")
    mod_values = np.where(y_values == 1, "qpsk", "noise").astype("<U32")
    channel_type = np.where(y_values == 1, "flat_random_phase", "noise_only").astype("<U32")

    if args.scenario == "qpsk_tone_hardneg":
        negative_idx = np.where(y_values == 0)[0]

        tone_mask = rng.random(size=negative_idx.size) < args.tone_probability
        tone_idx = negative_idx[tone_mask]
        if tone_idx.size > 0:
            tone = generate_single_tone(
                tone_idx.size,
                batch.sample_length,
                rng,
                min_amplitude=args.tone_amplitude_min,
                max_amplitude=args.tone_amplitude_max,
            )
            observations[tone_idx] = compose_observation(
                np.zeros_like(tone),
                noise_components[tone_idx],
                interference=tone,
            )
            mod_values[tone_idx] = "tone_jammer"
            channel_type[tone_idx] = "jammed_noise"
            append_interference_tag(interference_type, tone_idx, "tone")

        impulse_mask = rng.random(size=negative_idx.size) < args.impulse_probability
        impulse_idx = negative_idx[impulse_mask]
        if impulse_idx.size > 0:
            impulses = generate_impulsive_interference(
                impulse_idx.size,
                batch.sample_length,
                rng,
                probability=0.05,
                scale=args.impulse_scale,
            )
            observations[impulse_idx] = observations[impulse_idx] + impulses
            channel_type[impulse_idx] = "jammed_noise"
            append_interference_tag(interference_type, impulse_idx, "impulsive")

    return {
        "X": interleave_iq(observations),
        "y": y_values,
        "snr": snr_values,
        "mod": mod_values,
        "label_snr": np.where(y_values == 1, snr_values, 0).astype(np.int64),
        "source_snr": snr_values.astype(np.int64),
        "sample_type": np.where(y_values == 1, "signal", "noise").astype("<U16"),
        "noise_power": np.asarray([batch.noise_power], dtype=np.float32),
        "domain": np.full(batch.num_samples, "simulated", dtype="<U16"),
        "scenario": np.full(batch.num_samples, args.scenario, dtype="<U64"),
        "generator": np.full(
            batch.num_samples,
            "simulate.scripts.generate_sim_archive",
            dtype="<U64",
        ),
        "channel_type": channel_type,
        "interference_type": interference_type,
        "sample_id": np.asarray(
            [f"{args.scenario}:{index:07d}" for index in range(batch.num_samples)],
            dtype="<U64",
        ),
        "sim_seed": (args.seed + np.arange(batch.num_samples, dtype=np.int64)),
    }


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    output_path, manifest_path = resolve_output_paths(args)

    if output_path.exists() and not args.force:
        raise FileExistsError(f"Output archive already exists: {output_path}")

    arrays = generate_archive_arrays(args)
    summary = save_archive(
        output_path,
        arrays,
        default_domain="simulated",
        default_scenario=args.scenario,
        default_generator="simulate.scripts.generate_sim_archive",
        noise_power=args.noise_power,
    )
    manifest = dict(summary)
    manifest["generation"] = {
        "scenario": args.scenario,
        "num_samples": int(args.num_samples),
        "sample_length": int(args.sample_length),
        "noise_power": float(args.noise_power),
        "positive_ratio": float(args.positive_ratio),
        "snr_values": [int(value) for value in args.snr_values],
        "seed": int(args.seed),
    }
    write_manifest(manifest_path, manifest)

    print(f"Archive path: {output_path}")
    print(f"Manifest path: {manifest_path}")
    print(
        f"Samples: total={summary['num_samples']} signal={summary['signal_samples']} "
        f"noise={summary['noise_samples']} input_dim={summary['input_dim']}"
    )
    print(f"SNR values: {summary['snr_values']}")
    print(f"Domains: {summary['domains']}")
    print(f"Interference types: {summary['interference_types']}")


if __name__ == "__main__":
    main()
