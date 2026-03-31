from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class SimulationBatchConfig:
    """Shared batch-level controls for simulation runs."""

    num_samples: int = 4096
    sample_length: int = 128
    noise_power: float = 1.0
    positive_ratio: float = 0.5
    seed: int = 42


@dataclass(slots=True)
class ScenarioMetadata:
    """Common scenario metadata recorded into the archive."""

    name: str = "qpsk_awgn"
    domain: str = "simulated"
    generator: str = "simulate"
    channel_type: str = "flat_random_phase"
    interference_type: str = "none"
