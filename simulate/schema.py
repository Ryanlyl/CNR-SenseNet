from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SCHEMA_VERSION = "1.0"

REQUIRED_ARCHIVE_KEYS = ("X", "y", "snr")
BASE_ARCHIVE_KEYS = ("mod", "label_snr", "source_snr", "sample_type", "noise_power")
SIMULATION_ARCHIVE_KEYS = (
    "domain",
    "scenario",
    "generator",
    "channel_type",
    "interference_type",
    "sample_id",
    "sim_seed",
)
SCALAR_KEYS = ("noise_power", "schema_version")


def default_outputs_dir() -> Path:
    return Path(__file__).resolve().parent / "outputs"


def default_archive_dir() -> Path:
    return default_outputs_dir() / "archives"


def default_manifest_dir() -> Path:
    return default_outputs_dir() / "manifests"


def default_preview_dir() -> Path:
    return default_outputs_dir() / "previews"


def default_manifest_path(archive_path: str | Path) -> Path:
    archive_path = Path(archive_path)
    return default_manifest_dir() / f"{archive_path.stem}.json"


def sample_count(arrays: Mapping[str, Any]) -> int:
    x_values = np.asarray(arrays["X"])
    if x_values.ndim != 2:
        raise ValueError("Archive field 'X' must have shape [N, 2T].")
    return int(x_values.shape[0])


def _extract_scalar_float(value: Any, default: float) -> float:
    if value is None:
        return float(default)

    scalar = np.asarray(value).reshape(-1)
    if scalar.size != 1:
        raise ValueError("Scalar archive fields must contain exactly one value.")
    return float(scalar[0])


def _coerce_row_array_or_default(
    values: Any,
    default_values: np.ndarray,
    dtype: np.dtype | str,
) -> np.ndarray:
    default_values = np.asarray(default_values)
    length = int(default_values.shape[0])

    if values is None:
        array = default_values
    else:
        array = np.asarray(values)
        if array.ndim == 0:
            array = np.full(length, array.item())

    if array.shape[0] != length:
        raise ValueError(f"Expected row-aligned array with first dimension {length}, got {array.shape}.")

    return array.astype(dtype, copy=False)


def _string_counts(values: np.ndarray) -> dict[str, int]:
    unique, counts = np.unique(values.astype(str), return_counts=True)
    return {str(value): int(count) for value, count in zip(unique, counts, strict=False)}


def _default_sample_ids(length: int, prefix: str) -> np.ndarray:
    return np.asarray([f"{prefix}:{index:07d}" for index in range(length)], dtype="<U64")


def _normalize_passthrough_rows(
    arrays: Mapping[str, Any],
    normalized: dict[str, np.ndarray],
    length: int,
) -> None:
    for key, value in arrays.items():
        if key in normalized:
            continue

        array = np.asarray(value)
        if array.ndim == 0:
            normalized[key] = array.reshape(1).copy()
            continue

        if array.shape[0] != length:
            raise ValueError(
                f"Extra field '{key}' is neither scalar nor row-aligned with the archive length {length}."
            )
        normalized[key] = array.copy()


def normalize_archive_arrays(
    arrays: Mapping[str, Any],
    *,
    default_domain: str = "simulated",
    default_scenario: str = "unknown",
    default_generator: str = "simulate",
    noise_power: float | None = None,
) -> dict[str, np.ndarray]:
    if "X" not in arrays or "y" not in arrays or "snr" not in arrays:
        missing = [key for key in REQUIRED_ARCHIVE_KEYS if key not in arrays]
        raise KeyError(f"Archive is missing required keys: {missing}")

    x_values = np.asarray(arrays["X"], dtype=np.float32)
    if x_values.ndim != 2:
        raise ValueError("Archive field 'X' must have shape [N, 2T].")

    length = int(x_values.shape[0])
    y_values = _coerce_row_array_or_default(arrays.get("y"), np.zeros(length), np.int64)
    snr_values = _coerce_row_array_or_default(arrays.get("snr"), np.zeros(length), np.int64)

    mod_default = np.where(y_values == 1, "unknown_signal", "noise")
    label_snr_default = np.where(y_values == 1, snr_values, 0)
    source_snr_default = snr_values.copy()
    sample_type_default = np.where(y_values == 1, "signal", "noise")
    domain_default = np.full(length, default_domain)
    scenario_default = np.full(length, default_scenario)
    generator_default = np.full(length, default_generator)
    channel_default = np.full(length, "unspecified")
    interference_default = np.full(length, "none")
    sim_seed_default = np.full(length, -1, dtype=np.int64)
    sample_id_default = _default_sample_ids(length, prefix=default_scenario or default_domain)

    resolved_noise_power = _extract_scalar_float(
        arrays.get("noise_power"),
        default=1.0 if noise_power is None else noise_power,
    )

    normalized: dict[str, np.ndarray] = {
        "X": x_values,
        "y": y_values,
        "snr": snr_values,
        "mod": _coerce_row_array_or_default(arrays.get("mod"), mod_default, "<U32"),
        "label_snr": _coerce_row_array_or_default(arrays.get("label_snr"), label_snr_default, np.int64),
        "source_snr": _coerce_row_array_or_default(
            arrays.get("source_snr"),
            source_snr_default,
            np.int64,
        ),
        "sample_type": _coerce_row_array_or_default(
            arrays.get("sample_type"),
            sample_type_default,
            "<U16",
        ),
        "noise_power": np.asarray([resolved_noise_power], dtype=np.float32),
        "domain": _coerce_row_array_or_default(arrays.get("domain"), domain_default, "<U16"),
        "scenario": _coerce_row_array_or_default(arrays.get("scenario"), scenario_default, "<U64"),
        "generator": _coerce_row_array_or_default(arrays.get("generator"), generator_default, "<U64"),
        "channel_type": _coerce_row_array_or_default(
            arrays.get("channel_type"),
            channel_default,
            "<U32",
        ),
        "interference_type": _coerce_row_array_or_default(
            arrays.get("interference_type"),
            interference_default,
            "<U32",
        ),
        "sample_id": _coerce_row_array_or_default(arrays.get("sample_id"), sample_id_default, "<U64"),
        "sim_seed": _coerce_row_array_or_default(arrays.get("sim_seed"), sim_seed_default, np.int64),
        "schema_version": np.asarray([SCHEMA_VERSION], dtype="<U16"),
    }
    _normalize_passthrough_rows(arrays, normalized, length)
    return normalized


def build_archive_summary(
    arrays: Mapping[str, Any],
    *,
    path: str | Path | None = None,
) -> dict[str, Any]:
    normalized = normalize_archive_arrays(arrays)
    y_values = normalized["y"]
    snr_values = normalized["snr"]
    signal_mask = y_values == 1
    path_text = str(Path(path)) if path is not None else None

    return {
        "schema_version": SCHEMA_VERSION,
        "path": path_text,
        "num_samples": int(len(y_values)),
        "input_dim": int(normalized["X"].shape[1]),
        "num_iq_samples": int(normalized["X"].shape[1] // 2),
        "signal_samples": int(signal_mask.sum()),
        "noise_samples": int((~signal_mask).sum()),
        "noise_power": float(np.asarray(normalized["noise_power"]).reshape(-1)[0]),
        "snr_values": [int(value) for value in sorted(np.unique(snr_values).tolist())],
        "mods": sorted({str(value) for value in normalized["mod"][signal_mask]}),
        "domains": _string_counts(normalized["domain"]),
        "scenarios": _string_counts(normalized["scenario"]),
        "interference_types": _string_counts(normalized["interference_type"]),
        "fields": sorted(normalized.keys()),
    }


def write_manifest(path: str | Path, payload: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def save_archive(
    path: str | Path,
    arrays: Mapping[str, Any],
    *,
    default_domain: str = "simulated",
    default_scenario: str = "unknown",
    default_generator: str = "simulate",
    noise_power: float | None = None,
) -> dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    normalized = normalize_archive_arrays(
        arrays,
        default_domain=default_domain,
        default_scenario=default_scenario,
        default_generator=default_generator,
        noise_power=noise_power,
    )
    np.savez_compressed(path, **normalized)
    return build_archive_summary(normalized, path=path)


def load_archive(
    path: str | Path,
    *,
    default_domain: str = "simulated",
    default_scenario: str = "unknown",
    default_generator: str = "simulate",
) -> dict[str, np.ndarray]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}

    return normalize_archive_arrays(
        arrays,
        default_domain=default_domain,
        default_scenario=default_scenario,
        default_generator=default_generator,
    )


def stratified_subsample_indices(
    y: np.ndarray,
    snr: np.ndarray,
    max_samples: int | None,
    seed: int,
) -> np.ndarray:
    y = np.asarray(y, dtype=np.int64)
    snr = np.asarray(snr, dtype=np.int64)
    total = len(y)

    if max_samples is None or max_samples <= 0 or max_samples >= total:
        return np.arange(total, dtype=np.int64)

    rng = np.random.default_rng(seed)
    buckets: list[list[int]] = []
    for label in sorted(np.unique(y)):
        for snr_value in sorted(np.unique(snr)):
            bucket = np.where((y == label) & (snr == snr_value))[0]
            if bucket.size == 0:
                continue
            bucket = bucket.copy()
            rng.shuffle(bucket)
            buckets.append(bucket.tolist())

    selected: list[int] = []
    cursor = 0
    active_buckets = [bucket for bucket in buckets if bucket]
    while len(selected) < max_samples and active_buckets:
        bucket_idx = cursor % len(active_buckets)
        selected.append(active_buckets[bucket_idx].pop())
        if not active_buckets[bucket_idx]:
            active_buckets.pop(bucket_idx)
            if active_buckets:
                cursor %= len(active_buckets)
        else:
            cursor += 1

    selected_idx = np.asarray(selected, dtype=np.int64)
    rng.shuffle(selected_idx)
    return selected_idx


def select_archive_rows(arrays: Mapping[str, Any], indices: np.ndarray) -> dict[str, np.ndarray]:
    normalized = normalize_archive_arrays(arrays)
    indices = np.asarray(indices, dtype=np.int64)
    length = sample_count(normalized)

    subset: dict[str, np.ndarray] = {}
    for key, value in normalized.items():
        array = np.asarray(value)
        if key in SCALAR_KEYS or array.ndim == 0 or array.shape[0] != length:
            subset[key] = array.copy()
            continue
        subset[key] = array[indices]

    return normalize_archive_arrays(subset)


def shuffle_archive_rows(arrays: Mapping[str, Any], seed: int) -> dict[str, np.ndarray]:
    normalized = normalize_archive_arrays(arrays)
    rng = np.random.default_rng(seed)
    indices = np.arange(sample_count(normalized), dtype=np.int64)
    rng.shuffle(indices)
    return select_archive_rows(normalized, indices)


def _fill_missing_like(example: np.ndarray, length: int) -> np.ndarray:
    shape = (length, *example.shape[1:])
    if example.dtype.kind in {"U", "S", "O"}:
        return np.full(shape, "", dtype=example.dtype)
    if np.issubdtype(example.dtype, np.integer):
        return np.full(shape, -1, dtype=example.dtype)
    if np.issubdtype(example.dtype, np.floating):
        return np.full(shape, np.nan, dtype=example.dtype)
    if np.issubdtype(example.dtype, np.bool_):
        return np.zeros(shape, dtype=example.dtype)
    return np.zeros(shape, dtype=example.dtype)


def concatenate_archives(archives: list[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    if not archives:
        raise ValueError("At least one archive is required for concatenation.")

    normalized_archives = [normalize_archive_arrays(archive) for archive in archives]
    noise_powers = [
        float(np.asarray(archive["noise_power"]).reshape(-1)[0]) for archive in normalized_archives
    ]
    reference_noise_power = noise_powers[0]
    if not all(np.isclose(reference_noise_power, value) for value in noise_powers[1:]):
        raise ValueError("All archives must share the same noise_power before concatenation.")

    row_examples: dict[str, np.ndarray] = {}
    lengths = [sample_count(archive) for archive in normalized_archives]
    for archive, length in zip(normalized_archives, lengths, strict=False):
        for key, value in archive.items():
            array = np.asarray(value)
            if key in SCALAR_KEYS or array.ndim == 0 or array.shape[0] != length:
                continue
            row_examples.setdefault(key, array)

    merged: dict[str, np.ndarray] = {
        "noise_power": np.asarray([reference_noise_power], dtype=np.float32),
        "schema_version": np.asarray([SCHEMA_VERSION], dtype="<U16"),
    }
    for key, example in row_examples.items():
        parts: list[np.ndarray] = []
        for archive, length in zip(normalized_archives, lengths, strict=False):
            if key in archive:
                array = np.asarray(archive[key])
                if array.ndim > 0 and array.shape[0] == length:
                    parts.append(array)
                    continue
            parts.append(_fill_missing_like(example, length))
        merged[key] = np.concatenate(parts, axis=0)

    return normalize_archive_arrays(
        merged,
        default_domain="merged",
        default_scenario="merged",
        default_generator="simulate.schema.concatenate",
        noise_power=reference_noise_power,
    )


__all__ = [
    "BASE_ARCHIVE_KEYS",
    "REQUIRED_ARCHIVE_KEYS",
    "SCALAR_KEYS",
    "SCHEMA_VERSION",
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
    "sample_count",
    "save_archive",
    "select_archive_rows",
    "shuffle_archive_rows",
    "stratified_subsample_indices",
    "write_manifest",
]
