"""Model registry helpers for experiment entrypoints."""

from importlib import import_module


MODEL_MODULES = {
    "ed": "project.models.ed",
    "mlp": "project.models.mlp",
    "cnn1d": "project.models.cnn1d",
    "lstm": "project.models.lstm",
    "cn_lssnet": "project.models.cn_lssnet",
}


def load_model_module(name: str):
    """Load a model module by registry name."""
    if name not in MODEL_MODULES:
        available = ", ".join(sorted(MODEL_MODULES))
        raise ValueError(f"Unknown model '{name}'. Available models: {available}")
    return import_module(MODEL_MODULES[name])
