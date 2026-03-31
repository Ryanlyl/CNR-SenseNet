from __future__ import annotations

import numpy as np


def add_complex_awgn(
    shape: tuple[int, int],
    noise_power: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate circular complex AWGN with a target total noise power."""
    std = np.sqrt(noise_power / 2.0)
    real = rng.normal(loc=0.0, scale=std, size=shape).astype(np.float32)
    imag = rng.normal(loc=0.0, scale=std, size=shape).astype(np.float32)
    return (real + 1j * imag).astype(np.complex64)


def apply_iq_gain_imbalance(signal: np.ndarray, imbalance: np.ndarray | float) -> np.ndarray:
    """Apply a simple IQ gain imbalance to a complex baseband signal."""
    signal = np.asarray(signal, dtype=np.complex64)
    imbalance = np.asarray(imbalance, dtype=np.float32)
    if imbalance.ndim == 0:
        imbalance = np.full((signal.shape[0], 1), imbalance.item(), dtype=np.float32)
    elif imbalance.ndim == 1:
        imbalance = imbalance.reshape(-1, 1)

    i_values = signal.real * (1.0 + imbalance)
    q_values = signal.imag * (1.0 - imbalance)
    return (i_values + 1j * q_values).astype(np.complex64)


def apply_dc_offset(signal: np.ndarray, offset: complex | float) -> np.ndarray:
    """Apply a complex DC offset."""
    signal = np.asarray(signal, dtype=np.complex64)
    return (signal + np.complex64(offset)).astype(np.complex64)
