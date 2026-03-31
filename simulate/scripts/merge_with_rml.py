from __future__ import annotations

import argparse
from pathlib import Path

from simulate.schema import (
    build_archive_summary,
    concatenate_archives,
    default_archive_dir,
    default_manifest_path,
    load_archive,
    sample_count,
    save_archive,
    select_archive_rows,
    shuffle_archive_rows,
    stratified_subsample_indices,
    write_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge real cached archives with simulation archives into one mixed .npz dataset."
    )
    parser.add_argument("--base-archive", type=Path, required=True)
    parser.add_argument("--sim-archives", nargs="+", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--sim-ratio", type=float, default=None)
    parser.add_argument("--max-sim-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def resolve_output_paths(args) -> tuple[Path, Path]:
    if args.output_path is not None:
        output_path = args.output_path
    else:
        output_path = default_archive_dir() / f"{args.base_archive.stem}_with_sim.npz"
    manifest_path = args.manifest_path or default_manifest_path(output_path)
    return output_path, manifest_path


def resolve_sim_budget(args, base_count: int, sim_count: int) -> int:
    if args.max_sim_samples is not None:
        if args.max_sim_samples < 0:
            raise ValueError("--max-sim-samples must be >= 0.")
        return min(int(args.max_sim_samples), sim_count)

    if args.sim_ratio is not None:
        if args.sim_ratio < 0.0:
            raise ValueError("--sim-ratio must be >= 0.")
        requested = int(round(base_count * float(args.sim_ratio)))
        return min(requested, sim_count)

    return sim_count


def main() -> None:
    args = build_parser().parse_args()
    output_path, manifest_path = resolve_output_paths(args)

    if output_path.exists() and not args.force:
        raise FileExistsError(f"Output archive already exists: {output_path}")

    base_archive = load_archive(
        args.base_archive,
        default_domain="real",
        default_scenario=args.base_archive.stem,
        default_generator="project.data.gen_dataset",
    )
    sim_archives = [
        load_archive(
            path,
            default_domain="simulated",
            default_scenario=path.stem,
            default_generator="simulate",
        )
        for path in args.sim_archives
    ]
    sim_archive = concatenate_archives(sim_archives) if len(sim_archives) > 1 else sim_archives[0]

    target_sim_samples = resolve_sim_budget(
        args,
        base_count=sample_count(base_archive),
        sim_count=sample_count(sim_archive),
    )
    if target_sim_samples < sample_count(sim_archive):
        selected_idx = stratified_subsample_indices(
            sim_archive["y"],
            sim_archive["snr"],
            max_samples=target_sim_samples,
            seed=args.seed,
        )
        sim_archive = select_archive_rows(sim_archive, selected_idx)

    merged = concatenate_archives([base_archive, sim_archive])
    if args.shuffle:
        merged = shuffle_archive_rows(merged, seed=args.seed)

    summary = save_archive(
        output_path,
        merged,
        default_domain="merged",
        default_scenario="merged_real_sim",
        default_generator="simulate.scripts.merge_with_rml",
    )
    manifest = build_archive_summary(merged, path=output_path)
    manifest["merge"] = {
        "base_archive": str(args.base_archive),
        "sim_archives": [str(path) for path in args.sim_archives],
        "selected_sim_samples": int(sample_count(sim_archive)),
        "shuffle": bool(args.shuffle),
        "seed": int(args.seed),
    }
    write_manifest(manifest_path, manifest)

    print(f"Output archive: {output_path}")
    print(f"Manifest path: {manifest_path}")
    print(
        f"Merged samples: total={summary['num_samples']} signal={summary['signal_samples']} "
        f"noise={summary['noise_samples']}"
    )
    print(f"Domains: {summary['domains']}")
    print(f"Scenarios: {summary['scenarios']}")


if __name__ == "__main__":
    main()
