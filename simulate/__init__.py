"""Simulation helpers for future CNR-SenseNet dataset augmentation."""

from pathlib import Path

from .schema import (
    BASE_ARCHIVE_KEYS,
    REQUIRED_ARCHIVE_KEYS,
    SCHEMA_VERSION,
    SIMULATION_ARCHIVE_KEYS,
    build_archive_summary,
    concatenate_archives,
    default_archive_dir,
    default_manifest_dir,
    default_manifest_path,
    default_outputs_dir,
    default_preview_dir,
    load_archive,
    normalize_archive_arrays,
    save_archive,
    select_archive_rows,
    shuffle_archive_rows,
    stratified_subsample_indices,
    write_manifest,
)


SIMULATE_ROOT = Path(__file__).resolve().parent


def resolve_path(*parts: str) -> Path:
    """Resolve a path relative to the simulation package."""
    return SIMULATE_ROOT.joinpath(*parts)


__all__ = [
    "BASE_ARCHIVE_KEYS",
    "REQUIRED_ARCHIVE_KEYS",
    "SCHEMA_VERSION",
    "SIMULATE_ROOT",
    "SIMULATION_ARCHIVE_KEYS",
    "build_archive_summary",
    "concatenate_archives",
    "default_archive_dir",
    "default_manifest_dir",
    "default_manifest_path",
    "default_outputs_dir",
    "default_preview_dir",
    "load_archive",
    "normalize_archive_arrays",
    "resolve_path",
    "save_archive",
    "select_archive_rows",
    "shuffle_archive_rows",
    "stratified_subsample_indices",
    "write_manifest",
]
