"""The nine component registries.

    Models (1)  Optimizers (2)  Metrics (3)  Callbacks (4)  DataLoaders (5)
    Visualizations (6)  Layers (7)  Preprocessors (8)  Losses (9)

Each registry module is populated when imported. ``load_all(config)`` imports all nine,
loads plugins from ``Config.PLUGINS_DIR`` and checks that every component the config names
exists - call it once at a program's entry point (Trainer, Predictor, CLI), never at import
time. Importing this package itself does NOT import TensorFlow.
"""
from __future__ import annotations

import importlib
from typing import Dict, List, Optional

REGISTRY_MODULES = {
    "Models": "neural_trade.registries.models",
    "Optimizers": "neural_trade.registries.optimizers",
    "Metrics": "neural_trade.registries.metrics",
    "Callbacks": "neural_trade.registries.callbacks",
    "DataLoaders": "neural_trade.registries.data_loaders",
    "Visualizations": "neural_trade.registries.visualizations",
    "Layers": "neural_trade.registries.layers",
    "Preprocessors": "neural_trade.registries.preprocessors",
    "Losses": "neural_trade.registries.losses",
}


def all_registries() -> Dict[str, type]:
    """Import and return every registry class by name."""
    return {name: getattr(importlib.import_module(mod), name) for name, mod in REGISTRY_MODULES.items()}


def __getattr__(name):  # lazy: `from neural_trade.registries import Models`, `ALL_REGISTRIES`
    if name in REGISTRY_MODULES:
        return getattr(importlib.import_module(REGISTRY_MODULES[name]), name)
    if name == "ALL_REGISTRIES":
        return all_registries()
    raise AttributeError(name)


def validate_config_components(config) -> List[str]:
    """Every component name ``config`` selects that is NOT registered (empty = all resolvable)."""
    regs = all_registries()
    wanted = [
        ("Models", [config.MODEL_NAME]),
        ("Optimizers", [config.OPTIMIZER_NAME, config.INDICATOR_OPTIMIZER_NAME]),
        ("Losses", [config.LOSS_NAME]),
        ("DataLoaders", [config.DATA_LOADER]),
        ("Preprocessors", list(config.PREPROCESSORS)),
        ("Callbacks", list(config.CALLBACKS)),
        ("Visualizations", [config.VISUALIZATION]),
        ("Layers", list(dict(config.LAYERS).values())),
        ("Metrics", list(config.METRICS) + list(config.STEP_METRICS)),
    ]
    return [f"{reg}:{n}" for reg, names in wanted for n in names if not regs[reg].has(n)]


def load_all(config=None, *, plugins_dir: Optional[str] = None, strict: bool = True) -> Dict[str, type]:
    """Import all registries, load plugins, and (with a config) check its component names.

    Raises InvalidConfigurationError listing every missing component when ``strict``.
    """
    from neural_trade.core.exceptions import InvalidConfigurationError
    from neural_trade.core.plugin_loader import load_plugins

    regs = all_registries()
    for reg in regs.values():
        reg.auto_discover()
    directory = plugins_dir or (getattr(config, "PLUGINS_DIR", None) if config is not None else None)
    if directory:
        load_plugins(directory, strict=False)
    if config is not None:
        missing = validate_config_components(config)
        if missing and strict:
            raise InvalidConfigurationError(f"config names components that are not registered: {missing}")
    return regs


def registry_summary() -> str:
    """One line per registry: name, component count, components."""
    rows = []
    for name, reg in all_registries().items():
        rows.append(f"{name:15s} {reg.count():3d}  {', '.join(reg.list_names())}")
    total = sum(r.count() for r in all_registries().values())
    return "\n".join(rows + [f"{'total':15s} {total:3d}"])


def print_registry_summary() -> None:
    print(registry_summary())


__all__ = ["REGISTRY_MODULES", "all_registries", "validate_config_components", "load_all",
           "registry_summary", "print_registry_summary", *REGISTRY_MODULES]
