from __future__ import annotations

import numpy as np


def apply_random_phase(
    signal: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply one random phase offset per sample."""
    signal = np.asarray(signal, dtype=np.complex64)
    theta = rng.uniform(-np.pi, np.pi, size=(signal.shape[0], 1)).astype(np.float32)
    return (signal * np.exp(1j * theta)).astype(np.complex64)


def apply_flat_fading(
    signal: np.ndarray,
    rng: np.random.Generator,
    *,
    min_gain: float = 0.85,
    max_gain: float = 1.15,
) -> np.ndarray:
    """Apply a simple sample-wise flat fading gain."""
    signal = np.asarray(signal, dtype=np.complex64)
    gain = rng.uniform(min_gain, max_gain, size=(signal.shape[0], 1)).astype(np.float32)
    return (signal * gain).astype(np.complex64)
