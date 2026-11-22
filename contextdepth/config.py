"""Declarative backends and experiments; model names carry no runtime behavior."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import os
from pathlib import Path

import yaml


RUNTIME_CONFIG = Path(__file__).resolve().parents[1] / "models.yaml"
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "evaluation" / "models.yaml"
PATH_KEYS = frozenset(("source", "weights", "base_model", "flex_gemm_source",
                       "utils3d_source", "checkpoint", "ptxas", "cache_dir", "config"))


@dataclass(frozen=True)
class Experiment:
    name: str
    baseline: str
    camera: str
    context_fraction: float
    reference: str
    options: dict = field(default_factory=dict)
    reference_only: bool = False
    requires_rng: bool = False
    # Unknown-camera experiments only. "native": every input estimates its own
    # camera. "query": the camera estimated on the unexpanded query is reused
    # for the outpainted candidates; selection still observes native cameras.
    camera_source: str = "native"


def _read(path):
    path = Path(path).resolve()
    with path.open() as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a mapping")
    sections = [*config.get("models", {}).values(), config.get("outpainting", {}), config.get("retrieval", {})]
    for section in sections:
        for key, value in list(section.items()):
            if key in PATH_KEYS and value:
                expanded = Path(os.path.expandvars(str(value))).expanduser()
                section[key] = str(expanded if expanded.is_absolute() else path.parent / expanded)
            elif key == "cache_dependencies":
                section[key] = [str((path.parent / Path(os.path.expandvars(str(item))).expanduser()).resolve())
                                for item in value]
            elif key == "implementation" and value:
                module, separator, symbol = value.partition(":")
                if module.endswith(".py"):
                    source = Path(os.path.expandvars(module)).expanduser()
                    source = source if source.is_absolute() else path.parent / source
                    section[key] = str(source.resolve()) + (separator + symbol if separator else "")
    return config


def _merge(base, override):
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path=DEFAULT_CONFIG):
    config = _read(path)
    # Existing local files contain only machine-specific resource paths. Treat
    # those as overrides of the shipped, explicitly named protocol. A file
    # defining experiments is complete and can use entirely new plugins.
    if "experiments" not in config:
        config = _merge(_read(DEFAULT_CONFIG), config)
    if not isinstance(config.get("experiments"), dict) or not config["experiments"]:
        raise ValueError("Configure at least one experiment")
    for name in config["experiments"]:
        spec = experiment(config, name)
        if spec.baseline not in config.get("models", {}):
            raise ValueError(f"Unknown backend {spec.baseline!r} for {name}")
        if not config["models"][spec.baseline].get("implementation"):
            raise ValueError(f"Backend {spec.baseline!r} requires an implementation entry point")
        if not spec.reference_only:
            reference = experiment(config, spec.reference)
            if reference.camera != "known":
                raise ValueError(f"Reference calibration must use known camera: {spec.reference}")
    return config


def load_runtime_config(path=None):
    """Load model resources and normal-use defaults, without a paper protocol."""
    defaults = _read(RUNTIME_CONFIG)
    if path is None or Path(path).resolve() == RUNTIME_CONFIG.resolve():
        return defaults
    config = _merge(defaults, _read(path))
    config.pop("experiments", None)
    return config


def experiment(config, name):
    if not isinstance(name, str) or not name or Path(name).name != name or name in (".", ".."):
        raise ValueError(f"Invalid experiment name: {name!r}")
    try:
        value = config["experiments"][name]
    except KeyError as exc:
        raise ValueError(f"Unknown experiment {name!r}; choose from {list(config.get('experiments', {}))}") from exc
    spec = Experiment(name=name, **value)
    if spec.camera not in ("known", "unknown"):
        raise ValueError(f"Invalid camera setting for {name}: {spec.camera}")
    if not 0 <= spec.context_fraction <= .25:
        raise ValueError(f"Context fraction must be between 0 and .25: {name}")
    if spec.camera_source not in ("native", "query"):
        raise ValueError(f"camera_source must be native or query: {name}")
    if spec.camera_source == "query" and spec.camera != "unknown":
        raise ValueError(f"camera_source: query requires camera: unknown: {name}")
    return spec


def configuration_names(config):
    return [name for name in config["experiments"] if not experiment(config, name).reference_only]


def model_paths(config, name):
    spec = experiment(config, name)
    result = deepcopy(config["models"][spec.baseline])
    result["setting"] = spec.camera
    result["options"] = {**result.get("options", {}), **spec.options}
    result["batch_size"] = int(config.get("depth_batch_size", result.get("batch_size", 1)))
    if result["batch_size"] < 1:
        raise ValueError("depth_batch_size must be positive")
    return result


def model_load_config(paths):
    """Conservative identity for a resident model, excluding its camera mode.

    Resources, implementation, options, precision, batching and RNG policy
    must all match before a model instance can be reused between experiments.
    """
    return {key: deepcopy(value) for key, value in paths.items() if key != "setting"}
