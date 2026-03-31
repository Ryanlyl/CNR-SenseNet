from __future__ import annotations

import numpy as np


def generate_single_tone(
    num_samples: int,
    sample_length: int,
    rng: np.random.Generator,
    *,
    min_amplitude: float = 0.2,
    max_amplitude: float = 0.8,
    min_frequency: float = 0.02,
    max_frequency: float = 0.25,
) -> np.ndarray:
    """Generate a simple narrowband tone interferer."""
    time_axis = np.arange(sample_length, dtype=np.float32).reshape(1, -1)
    amplitude = rng.uniform(min_amplitude, max_amplitude, size=(num_samples, 1)).astype(np.float32)
    frequency = rng.uniform(min_frequency, max_frequency, size=(num_samples, 1)).astype(np.float32)
    phase = rng.uniform(-np.pi, np.pi, size=(num_samples, 1)).astype(np.float32)
    tone = amplitude * np.exp(1j * (2.0 * np.pi * frequency * time_axis + phase))
    return np.asarray(tone, dtype=np.complex64)


def generate_impulsive_interference(
    num_samples: int,
    sample_length: int,
    rng: np.random.Generator,
    *,
    probability: float = 0.02,
    scale: float = 3.0,
) -> np.ndarray:
    """Generate sparse impulsive interference in complex baseband."""
    mask = rng.random(size=(num_samples, sample_length)) < probability
    std = scale / np.sqrt(2.0)
    real = rng.normal(loc=0.0, scale=std, size=(num_samples, sample_length)).astype(np.float32)
    imag = rng.normal(loc=0.0, scale=std, size=(num_samples, sample_length)).astype(np.float32)
    impulses = (real + 1j * imag).astype(np.complex64)
    return (mask.astype(np.float32) * impulses).astype(np.complex64)
