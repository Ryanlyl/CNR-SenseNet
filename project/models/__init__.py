"""Model registry helpers for experiment entrypoints."""

from importlib import import_module
from inspect import Parameter, signature


MODEL_REGISTRY = {
    "cnr_sensenet": ("project.CNR_SenseNet", "CNRSenseNetModel"),
    "ed": ("project.models.classical_detectors", "EnergyDetector"),
    "energy_detector": ("project.models.classical_detectors", "EnergyDetector"),
    "autocorr_detector": ("project.models.classical_detectors", "AutocorrelationDetector"),
    "mlp": ("project.models.mlp", "MLPModel"),
    "cnn1d": ("project.models.cnn1d", "CNN1DModel"),
    "lstm": ("project.models.lstm", "LSTMModel"),
}


def get_model_class(name: str):
    if name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model '{name}'. Available models: {available}")

    module_name, class_name = MODEL_REGISTRY[name]
    module = import_module(module_name)
    return getattr(module, class_name)


def _supports_var_kwargs(model_class) -> bool:
    return any(
        parameter.kind == Parameter.VAR_KEYWORD
        for parameter in signature(model_class).parameters.values()
    )


def _filter_supported_kwargs(model_class, kwargs):
    if not kwargs or _supports_var_kwargs(model_class):
        return kwargs

    supported = {
        name
        for name, parameter in signature(model_class).parameters.items()
        if parameter.kind in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY)
    }
    return {name: value for name, value in kwargs.items() if name in supported}


def create_model(name: str, **kwargs):
    model_class = get_model_class(name)
    return model_class(**_filter_supported_kwargs(model_class, kwargs))


__all__ = ["MODEL_REGISTRY", "get_model_class", "create_model"]
