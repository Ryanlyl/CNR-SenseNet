import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def interleave_iq(sample_2x128):
    """
    Convert an IQ sample with shape [2, 128] into [256] as:
    [I0, Q0, I1, Q1, ...]
    """
    i = sample_2x128[0]
    q = sample_2x128[1]
    out = np.empty((i.size + q.size,), dtype=np.float32)
    out[0::2] = i
    out[1::2] = q
    return out


def compute_power(x):
    i = x[0::2]
    q = x[1::2]
    return np.mean(i**2 + q**2)


def restore_signal_by_snr(x_norm, snr_db, noise_power=1.0):
    """
    Restore a normalized RML2016.10a sample to a target total power.

    Assumption:
        The normalized sample has unit-like power, and its original total
        power should be:
            P_total = P_noise * (1 + 10^(SNR/10))
        where P_noise is a fixed reference noise power.
    """
    current_power = compute_power(x_norm)
    target_power = noise_power * (1.0 + 10.0 ** (float(snr_db) / 10.0))
    scale = np.sqrt(target_power / (current_power + 1e-12))
    return (x_norm * scale).astype(np.float32)


def generate_fixed_power_awgn(shape, noise_power=1.0, rng=None):
    """
    Generate pure AWGN with fixed total power.

    For complex baseband noise:
        P_noise = E[I^2 + Q^2]
    so each real dimension uses std = sqrt(P_noise / 2).
    """
    if rng is None:
        rng = np.random.default_rng()

    std = np.sqrt(noise_power / 2.0)
    return rng.normal(loc=0.0, scale=std, size=shape).astype(np.float32)


def load_rml_dict(pkl_path):
    with open(pkl_path, "rb") as file:
        return pickle.load(file, encoding="latin1")


def _normalize_for_cache(values):
    if values is None:
        return None
    return list(values)


def default_processed_dir():
    return Path(__file__).resolve().parent / "processed"


def build_cache_path(
    pkl_path,
    snr_filter=None,
    selected_mods=None,
    seed=42,
    noise_power=1.0,
    processed_dir=None,
):
    processed_dir = Path(processed_dir) if processed_dir is not None else default_processed_dir()
    processed_dir.mkdir(parents=True, exist_ok=True)

    cache_key = {
        "dataset": Path(pkl_path).name,
        "snr_filter": _normalize_for_cache(snr_filter),
        "selected_mods": _normalize_for_cache(selected_mods),
        "seed": int(seed),
        "noise_power": float(noise_power),
    }
    digest = hashlib.md5(
        json.dumps(cache_key, sort_keys=True, ensure_ascii=True).encode("utf-8")
    ).hexdigest()[:12]
    return processed_dir / f"signal_noise_{digest}.npz"


def build_signal_vs_noise_dataset(
    pkl_path,
    snr_filter=None,
    selected_mods=None,
    seed=42,
    noise_power=1.0,
):
    """
    Build a binary detection dataset from RML2016.10a.

    Positive samples:
        Original modulation samples restored using their SNR labels.
    Negative samples:
        Pure AWGN with fixed noise power.

    Returns:
        X: [N, 256]
        y: [N], binary labels where 1=signal and 0=noise
        snr: [N], source/original SNR used for stratification and evaluation
        meta: list[dict] with binary label, mod label, label_snr, source_snr
    """
    rng = np.random.default_rng(seed)
    data = load_rml_dict(pkl_path)

    X_all = []
    y_all = []
    snr_all = []
    meta_all = []

    for (mod, snr), samples in data.items():
        if snr_filter is not None and snr not in snr_filter:
            continue
        if selected_mods is not None and mod not in selected_mods:
            continue

        for sample in samples:
            x_norm = interleave_iq(sample).astype(np.float32)
            x_signal = restore_signal_by_snr(x_norm, snr_db=snr, noise_power=noise_power)

            X_all.append(x_signal)
            y_all.append(1)
            snr_all.append(int(snr))
            meta_all.append(
                {
                    "type": "signal",
                    "binary_label": 1,
                    "mod": mod,
                    "label_snr": int(snr),
                    "source_snr": int(snr),
                }
            )

            x_noise = generate_fixed_power_awgn(x_signal.shape, noise_power=noise_power, rng=rng)
            X_all.append(x_noise)
            y_all.append(0)
            snr_all.append(int(snr))
            meta_all.append(
                {
                    "type": "noise",
                    "binary_label": 0,
                    "mod": "noise",
                    "label_snr": 0,
                    "source_snr": int(snr),
                }
            )

    X = np.stack(X_all, axis=0).astype(np.float32)
    y = np.asarray(y_all, dtype=np.int64)
    snr = np.asarray(snr_all, dtype=np.int64)

    idx = np.arange(len(X))
    rng.shuffle(idx)

    X = X[idx]
    y = y[idx]
    snr = snr[idx]
    meta = [meta_all[i] for i in idx]

    return X, y, snr, meta


def stratified_split_binary(X, y, snr, meta, test_ratio=0.2, seed=42):
    """
    Split jointly by binary label and source SNR.

    This keeps both the positive/negative balance and the per-SNR distribution
    aligned between train and test.
    """
    rng = np.random.default_rng(seed)
    train_idx = []
    test_idx = []

    unique_labels = sorted(np.unique(y))
    unique_snrs = sorted(np.unique(snr))

    for label in unique_labels:
        for snr_value in unique_snrs:
            bucket = np.where((y == label) & (snr == snr_value))[0]
            if len(bucket) == 0:
                continue

            rng.shuffle(bucket)
            n_test = max(1, int(len(bucket) * test_ratio)) if len(bucket) > 1 else 0
            test_idx.extend(bucket[:n_test].tolist())
            train_idx.extend(bucket[n_test:].tolist())

    train_idx = np.asarray(train_idx, dtype=np.int64)
    test_idx = np.asarray(test_idx, dtype=np.int64)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    X_train, y_train, snr_train = X[train_idx], y[train_idx], snr[train_idx]
    X_test, y_test, snr_test = X[test_idx], y[test_idx], snr[test_idx]
    meta_train = [meta[i] for i in train_idx]
    meta_test = [meta[i] for i in test_idx]

    return X_train, y_train, snr_train, meta_train, X_test, y_test, snr_test, meta_test


def save_signal_vs_noise_dataset(cache_path, X, y, snr, meta, noise_power):
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    mod = np.asarray([item["mod"] for item in meta], dtype="<U32")
    label_snr = np.asarray([item["label_snr"] for item in meta], dtype=np.int64)
    source_snr = np.asarray([item["source_snr"] for item in meta], dtype=np.int64)
    sample_type = np.asarray([item["type"] for item in meta], dtype="<U16")

    np.savez_compressed(
        cache_path,
        X=X.astype(np.float32),
        y=y.astype(np.int64),
        snr=snr.astype(np.int64),
        mod=mod,
        label_snr=label_snr,
        source_snr=source_snr,
        sample_type=sample_type,
        noise_power=np.asarray([noise_power], dtype=np.float32),
    )


def load_signal_vs_noise_archive(cache_path):
    cache_path = Path(cache_path)
    with np.load(cache_path, allow_pickle=False) as data:
        archive = {key: data[key] for key in data.files}

    archive["X"] = archive["X"].astype(np.float32, copy=False)
    archive["y"] = archive["y"].astype(np.int64, copy=False)
    archive["snr"] = archive["snr"].astype(np.int64, copy=False)
    archive["label_snr"] = archive["label_snr"].astype(np.int64, copy=False)
    archive["source_snr"] = archive["source_snr"].astype(np.int64, copy=False)
    archive["noise_power"] = float(np.asarray(archive["noise_power"]).reshape(-1)[0])
    return archive


def load_signal_vs_noise_dataset(cache_path):
    archive = load_signal_vs_noise_archive(cache_path)
    X = archive["X"]
    y = archive["y"]
    snr = archive["snr"]
    mod = archive["mod"]
    label_snr = archive["label_snr"]
    source_snr = archive["source_snr"]
    sample_type = archive["sample_type"]

    meta = []
    for idx in range(len(X)):
        meta.append(
            {
                "type": str(sample_type[idx]),
                "binary_label": int(y[idx]),
                "mod": str(mod[idx]),
                "label_snr": int(label_snr[idx]),
                "source_snr": int(source_snr[idx]),
            }
        )

    return X, y, snr, meta


def prepare_signal_vs_noise_dataset(
    pkl_path,
    snr_filter=None,
    selected_mods=None,
    seed=42,
    noise_power=1.0,
    cache_path=None,
    use_cache=True,
    force_rebuild=False,
):
    if cache_path is None:
        cache_path = build_cache_path(
            pkl_path=pkl_path,
            snr_filter=snr_filter,
            selected_mods=selected_mods,
            seed=seed,
            noise_power=noise_power,
        )
    else:
        cache_path = Path(cache_path)

    if use_cache and cache_path.exists() and not force_rebuild:
        X, y, snr, meta = load_signal_vs_noise_dataset(cache_path)
        return X, y, snr, meta, cache_path

    X, y, snr, meta = build_signal_vs_noise_dataset(
        pkl_path=pkl_path,
        snr_filter=snr_filter,
        selected_mods=selected_mods,
        seed=seed,
        noise_power=noise_power,
    )
    save_signal_vs_noise_dataset(cache_path, X, y, snr, meta, noise_power=noise_power)
    return X, y, snr, meta, cache_path


class SignalNoiseDataset(Dataset):
    def __init__(self, X, y, snr, mod=None, label_snr=None, sample_type=None, meta=None):
        self.X = self._as_tensor(X, dtype=torch.float32)
        self.y = self._as_tensor(y, dtype=torch.float32)
        self.snr = self._as_tensor(snr, dtype=torch.int64)
        self.mod = np.asarray(mod) if mod is not None else None
        self.label_snr = (
            self._as_tensor(label_snr, dtype=torch.int64) if label_snr is not None else None
        )
        self.sample_type = np.asarray(sample_type) if sample_type is not None else None
        self.meta = list(meta) if meta is not None else None

    @staticmethod
    def _as_tensor(values, dtype):
        if isinstance(values, torch.Tensor):
            return values.to(dtype=dtype)
        if isinstance(values, np.ndarray):
            tensor = torch.from_numpy(values)
            return tensor.to(dtype=dtype)
        return torch.tensor(values, dtype=dtype)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.snr[idx]

    def get_labels(self, idx):
        return {
            "binary_label": int(self.y[idx].item()),
            "source_snr": int(self.snr[idx].item()),
            "label_snr": int(self.label_snr[idx].item()) if self.label_snr is not None else None,
            "mod": str(self.mod[idx]) if self.mod is not None else None,
            "type": str(self.sample_type[idx]) if self.sample_type is not None else None,
        }


if __name__ == "__main__":
    pkl_path = Path(__file__).resolve().parent / "RML2016.10a_dict.pkl"
    X, y, snr, meta, cache_path = prepare_signal_vs_noise_dataset(
        pkl_path=pkl_path,
        seed=42,
        noise_power=1.0,
        use_cache=True,
    )
    split = stratified_split_binary(X, y, snr, meta, test_ratio=0.2, seed=42)
    X_train, y_train, snr_train, _, X_test, y_test, snr_test, _ = split

    print(f"cache: {cache_path}")
    print(X_train.shape, y_train.shape, snr_train.shape)
    print(X_test.shape, y_test.shape, snr_test.shape)
