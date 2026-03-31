"""Core building blocks for simulation scenarios."""

from .base import ScenarioMetadata, SimulationBatchConfig
from .channels import apply_flat_fading, apply_random_phase
from .composer import (
    compose_observation,
    interleave_iq,
    sample_snr_values,
    scale_to_average_power,
    synthesize_qpsk,
    target_signal_power,
)
from .impairments import add_complex_awgn, apply_dc_offset, apply_iq_gain_imbalance
from .interferers import generate_impulsive_interference, generate_single_tone


__all__ = [
    "ScenarioMetadata",
    "SimulationBatchConfig",
    "add_complex_awgn",
    "apply_dc_offset",
    "apply_flat_fading",
    "apply_iq_gain_imbalance",
    "apply_random_phase",
    "compose_observation",
    "generate_impulsive_interference",
    "generate_single_tone",
    "interleave_iq",
    "sample_snr_values",
    "scale_to_average_power",
    "synthesize_qpsk",
    "target_signal_power",
]
