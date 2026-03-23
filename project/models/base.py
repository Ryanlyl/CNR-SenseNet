from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseDetector(ABC):
    """Unified interface for all binary detection models in this project."""

    prefers_internal_threshold = False

    def __init__(self, **config: Any):
        self.config = dict(config)

    @abstractmethod
    def fit(self, train_dataset, val_dataset=None, **kwargs):
        """Train or calibrate the detector."""

    @abstractmethod
    def predict_scores(self, dataset):
        """Return real-valued detection scores. Larger means more likely to be signal."""

    def predict(self, dataset, threshold=0.0):
        scores = self.predict_scores(dataset)
        return (np.asarray(scores) >= threshold).astype(np.int64)

    def get_evaluation_threshold(self, threshold=None):
        return threshold

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict: dict):
        del state_dict
        return self

    def get_config(self):
        return dict(self.config)
