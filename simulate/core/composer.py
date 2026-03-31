from __future__ import annotations

import numpy as np


def synthesize_qpsk(
    num_samples: int,
    sample_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate unit-power QPSK samples as complex baseband sequences."""
    i_values = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=(num_samples, sample_length))
    q_values = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=(num_samples, sample_length))
    return ((i_values + 1j * q_values) / np.sqrt(2.0)).astype(np.complex64)


def sample_snr_values(
    num_samples: int,
    snr_values: list[int],
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample per-example SNR values from a user-provided grid."""
    return rng.choice(np.asarray(snr_values, dtype=np.int64), size=num_samples, replace=True).astype(np.int64)


def target_signal_power(snr_db: np.ndarray, noise_power: float) -> np.ndarray:
    """Convert SNR labels into signal-component power targets."""
    snr_db = np.asarray(snr_db, dtype=np.float32)
    return (noise_power * np.power(10.0, snr_db / 10.0)).astype(np.float32)


def scale_to_average_power(signal: np.ndarray, target_power: np.ndarray) -> np.ndarray:
    """Rescale each sample to a desired average signal power."""
    signal = np.asarray(signal, dtype=np.complex64)
    target_power = np.asarray(target_power, dtype=np.float32).reshape(-1, 1)
    current_power = np.mean(np.abs(signal) ** 2, axis=1, keepdims=True).astype(np.float32)
    scale = np.sqrt(target_power / (current_power + 1e-12)).astype(np.float32)
    return (signal * scale).astype(np.complex64)


def compose_observation(
    signal: np.ndarray,
    noise: np.ndarray,
    *,
    interference: np.ndarray | None = None,
) -> np.ndarray:
    """Combine signal, noise, and optional interference into one observation."""
    signal = np.asarray(signal, dtype=np.complex64)
    noise = np.asarray(noise, dtype=np.complex64)
    observation = signal + noise
    if interference is not None:
        observation = observation + np.asarray(interference, dtype=np.complex64)
    return np.asarray(observation, dtype=np.complex64)


def interleave_iq(complex_samples: np.ndarray) -> np.ndarray:
    """Flatten complex baseband into [I0, Q0, I1, Q1, ...] rows."""
    complex_samples = np.asarray(complex_samples, dtype=np.complex64)
    output = np.empty((complex_samples.shape[0], complex_samples.shape[1] * 2), dtype=np.float32)
    output[:, 0::2] = complex_samples.real
    output[:, 1::2] = complex_samples.imag
    return output
