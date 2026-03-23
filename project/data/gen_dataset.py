import pickle
import random
import numpy as np
import torch
from torch.utils.data import Dataset

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def interleave_iq(sample_2x128):
    """
    sample_2x128: shape [2, 128]
    return: shape [256]
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

def generate_awgn_like(x_ref, rng=None):
    """
    根据参考正样本 x_ref 的功率生成纯白噪声负样本
    """
    if rng is None:
        rng = np.random.default_rng()

    p_ref = compute_power(x_ref)
    std = np.sqrt(p_ref / 2.0 + 1e-12)
    noise = rng.normal(loc=0.0, scale=std, size=x_ref.shape).astype(np.float32)
    return noise

def build_signal_vs_noise_dataset(
    pkl_path,
    snr_filter=None,
    selected_mods=None,
    seed=42
):
    """
    从 RadioML 构造二分类数据集：
        正样本 = 原始调制信号样本 (label=1)
        负样本 = 匹配功率的 AWGN 纯噪声 (label=0)

    返回:
        X: [N, 256]
        y: [N]
        snr: [N]
        meta: list of dict
    """
    rng = np.random.default_rng(seed)

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    all_mods = sorted(list(set(k[0] for k in data.keys())))
    all_snrs = sorted(list(set(k[1] for k in data.keys())))

    print("Available mods:", all_mods)
    print("Available SNRs:", all_snrs)

    X_pos = []
    y_pos = []
    snr_pos = []
    meta_pos = []

    for (mod, snr), samples in data.items():
        if snr_filter is not None and snr not in snr_filter:
            continue
        if selected_mods is not None and mod not in selected_mods:
            continue

        for sample in samples:
            x = interleave_iq(sample)   # [256]
            X_pos.append(x.astype(np.float32))
            y_pos.append(1)
            snr_pos.append(snr)
            meta_pos.append({
                'type': 'signal',
                'mod': mod,
                'snr': snr
            })

    X_pos = np.stack(X_pos, axis=0)
    y_pos = np.array(y_pos, dtype=np.int64)
    snr_pos = np.array(snr_pos, dtype=np.int64)

    # 生成一一对应的负样本，数量平衡
    X_neg = []
    y_neg = []
    snr_neg = []
    meta_neg = []

    for i in range(len(X_pos)):
        x_ref = X_pos[i]
        noise = generate_awgn_like(x_ref, rng=rng)
        snr_i = snr_pos[i]

        X_neg.append(noise)
        y_neg.append(0)
        snr_neg.append(snr_i)
        meta_neg.append({
            'type': 'noise',
            'mod': 'noise',
            'snr': snr_i   # 继承对应正样本的 SNR 桶
        })

    X_neg = np.stack(X_neg, axis=0)
    y_neg = np.array(y_neg, dtype=np.int64)
    snr_neg = np.array(snr_neg, dtype=np.int64)

    # 合并
    X = np.concatenate([X_pos, X_neg], axis=0)
    y = np.concatenate([y_pos, y_neg], axis=0)
    snr = np.concatenate([snr_pos, snr_neg], axis=0)
    meta = meta_pos + meta_neg

    # 打乱
    idx = np.arange(len(X))
    rng.shuffle(idx)

    X = X[idx]
    y = y[idx]
    snr = snr[idx]
    meta = [meta[i] for i in idx]

    print("Final dataset:")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("snr shape:", snr.shape)
    print("num signal:", np.sum(y == 1))
    print("num noise :", np.sum(y == 0))

    return X, y, snr, meta

def stratified_split_binary(X, y, snr, meta, test_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)

    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    n_pos_test = int(len(pos_idx) * test_ratio)
    n_neg_test = int(len(neg_idx) * test_ratio)

    test_idx = np.concatenate([pos_idx[:n_pos_test], neg_idx[:n_neg_test]])
    train_idx = np.concatenate([pos_idx[n_pos_test:], neg_idx[n_neg_test:]])

    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    X_train, y_train, snr_train = X[train_idx], y[train_idx], snr[train_idx]
    X_test, y_test, snr_test = X[test_idx], y[test_idx], snr[test_idx]
    meta_train = [meta[i] for i in train_idx]
    meta_test = [meta[i] for i in test_idx]

    return X_train, y_train, snr_train, meta_train, X_test, y_test, snr_test, meta_test

class SignalNoiseDataset(Dataset):
    def __init__(self, X, y, snr):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.snr = torch.tensor(snr, dtype=torch.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.snr[idx]


if __name__ == "__main__":
    set_seed(42)

    pkl_path = "data/RML2016.10a_dict.pkl"

    X, y, snr, meta = build_signal_vs_noise_dataset(
        pkl_path=pkl_path,
        snr_filter=None,
        selected_mods=None,
        seed=42
    )

    X_train, y_train, snr_train, meta_train, X_test, y_test, snr_test, meta_test = stratified_split_binary(
        X, y, snr, meta, test_ratio=0.2, seed=42
    )

    print(X_train.shape, y_train.shape, snr_train.shape)
    print(X_test.shape, y_test.shape, snr_test.shape)

    train_ds = SignalNoiseDataset(X_train, y_train, snr_train)
    test_ds = SignalNoiseDataset(X_test, y_test, snr_test)

    print("Sample from train_ds:")
    for i in range(20):
        x, label, snr_i = train_ds[i]
        p = compute_power(x.numpy())
        print(f"Sample {i}: label={label.item()}, snr={snr_i.item()}, power={p:.8e}")

    print("\nSample from test_ds:")
    for i in range(20):
        x, label, snr_i = test_ds[i]
        p = compute_power(x.numpy())
        print(f"Sample {i}: label={label.item()}, snr={snr_i.item()}, power={p:.8e}")