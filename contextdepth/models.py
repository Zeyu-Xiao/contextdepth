"""Load configured implementations without a model registry or model branches."""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
from pathlib import Path
import sys

from .cache import digest
from .config import model_load_config


def load_implementation(entrypoint: str):
    """Load package.module:Class or /path/baseline.py:Class in the worker.

    Importing an implementation must not load weights or initialize a device.
    Pass this string to spawned workers instead of initialized GPU models.
    """
    module_name, separator, attribute = entrypoint.rpartition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("implementation must be module:Class or file.py:Class")
    if module_name.endswith(".py"):
        path = Path(module_name).expanduser().resolve(strict=True)
        identity = hashlib.sha256(str(path).encode()).hexdigest()[:20]
        module_key = f"_contextdepth_baseline_{identity}"
        module = sys.modules.get(module_key)
        if module is None:
            spec = importlib.util.spec_from_file_location(module_key, path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load implementation from {path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_key] = module
            try:
                spec.loader.exec_module(module)
            except BaseException:
                sys.modules.pop(module_key, None)
                raise
    else:
        module = importlib.import_module(module_name)
    implementation = getattr(module, attribute)
    if not callable(implementation):
        raise TypeError(f"Implementation {entrypoint} is not callable")
    return implementation


def create_model(name: str, paths: dict, device: str = "cuda:0"):
    """Instantiate a depth adapter; name is only a result label."""
    paths = dict(paths)
    implementation = load_implementation(paths["implementation"])
    model = implementation(name, paths, device)
    if not callable(getattr(model, "predict", None)):
        raise TypeError("Depth implementation must provide predict(requests)")
    return model


def model_reuse_key(name: str, paths: dict):
    """Group only adapters that explicitly support switching camera settings."""
    implementation = load_implementation(paths["implementation"])
    if (getattr(implementation, "supports_setting_reuse", False)
            and callable(getattr(implementation, "configure", None))):
        return digest({"resident": model_load_config(paths)})
    return digest({"experiment": name, "paths": paths})
