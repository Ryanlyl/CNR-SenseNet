from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from project.data.gen_dataset import (
    SignalNoiseDataset,
    build_cache_path,
    default_processed_dir,
    prepare_signal_vs_noise_dataset,
    stratified_split_binary,
)


@dataclass(slots=True)
class DataConfig:
    pkl_path: Optional[str | Path] = None
    test_ratio: float = 0.2
    seed: int = 42
    noise_power: float = 1.0
    snr_filter: Optional[Sequence[int]] = None
    selected_mods: Optional[Sequence[str]] = None
    cache_path: Optional[str | Path] = None
    use_cache: bool = True
    force_rebuild: bool = False


@dataclass(slots=True)
class DatasetBundle:
    train_dataset: SignalNoiseDataset
    test_dataset: SignalNoiseDataset
    train_arrays: dict
    test_arrays: dict
    train_meta: list
    test_meta: list
    input_dim: int
    num_classes: int
    snrs: list[int]
    mods: list[str]
    pkl_path: Path
    cache_path: Path
    noise_power: float


def default_rml2016a_path() -> Path:
    return Path(__file__).resolve().parent / "RML2016.10a_dict.pkl"


def build_datasets(config: Optional[DataConfig] = None) -> DatasetBundle:
    config = config or DataConfig()
    pkl_path = Path(config.pkl_path) if config.pkl_path is not None else default_rml2016a_path()

    X, y, snr, meta, cache_path = prepare_signal_vs_noise_dataset(
        pkl_path=pkl_path,
        snr_filter=config.snr_filter,
        selected_mods=config.selected_mods,
        seed=config.seed,
        noise_power=config.noise_power,
        cache_path=config.cache_path,
        use_cache=config.use_cache,
        force_rebuild=config.force_rebuild,
    )
    split = stratified_split_binary(X, y, snr, meta, test_ratio=config.test_ratio, seed=config.seed)
    X_train, y_train, snr_train, meta_train, X_test, y_test, snr_test, meta_test = split

    train_mod = np.asarray([item["mod"] for item in meta_train], dtype="<U32")
    test_mod = np.asarray([item["mod"] for item in meta_test], dtype="<U32")
    train_label_snr = np.asarray([item["label_snr"] for item in meta_train], dtype=np.int64)
    test_label_snr = np.asarray([item["label_snr"] for item in meta_test], dtype=np.int64)
    train_type = np.asarray([item["type"] for item in meta_train], dtype="<U16")
    test_type = np.asarray([item["type"] for item in meta_test], dtype="<U16")

    train_dataset = SignalNoiseDataset(
        X_train,
        y_train,
        snr_train,
        mod=train_mod,
        label_snr=train_label_snr,
        sample_type=train_type,
        meta=meta_train,
    )
    test_dataset = SignalNoiseDataset(
        X_test,
        y_test,
        snr_test,
        mod=test_mod,
        label_snr=test_label_snr,
        sample_type=test_type,
        meta=meta_test,
    )
    mods = sorted({item["mod"] for item in meta if item["type"] == "signal"})
    snrs = sorted(np.unique(snr).tolist())

    return DatasetBundle(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        train_arrays={
            "X": X_train,
            "y": y_train,
            "snr": snr_train,
            "label_snr": train_label_snr,
            "mod": train_mod,
            "sample_type": train_type,
        },
        test_arrays={
            "X": X_test,
            "y": y_test,
            "snr": snr_test,
            "label_snr": test_label_snr,
            "mod": test_mod,
            "sample_type": test_type,
        },
        train_meta=meta_train,
        test_meta=meta_test,
        input_dim=int(X.shape[1]),
        num_classes=2,
        snrs=snrs,
        mods=mods,
        pkl_path=pkl_path,
        cache_path=Path(cache_path),
        noise_power=float(config.noise_power),
    )


__all__ = [
    "DataConfig",
    "DatasetBundle",
    "SignalNoiseDataset",
    "build_cache_path",
    "build_datasets",
    "default_rml2016a_path",
    "default_processed_dir",
]
