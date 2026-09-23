"""Helpers for the temporary root-level compatibility shims (removed in Phase B18).

During the migration the old import paths (``core``, ``losses``, ``model`` ...) keep
working by aliasing the new modules in ``sys.modules``. A package shim must alias its
submodules too: otherwise ``import core.registry`` would load a second copy of
``neural_trade/core/registry.py`` under a different name, and every class in it
(BaseRegistry, Config, ...) would exist twice with isinstance checks failing across
the copies.
"""
from __future__ import annotations

import importlib
import sys
from types import ModuleType
from typing import Iterable


def alias_module(old: str, new: str) -> ModuleType:
    """Make ``import old`` return the already-importable module ``new``."""
    mod = importlib.import_module(new)
    sys.modules[old] = mod
    return mod


def alias_package(old: str, new: str, submodules: Iterable[str]) -> ModuleType:
    """Alias a package and the listed submodules (eagerly, so no duplicate copies)."""
    pkg = importlib.import_module(new)
    for sub in submodules:
        sys.modules[f"{old}.{sub}"] = importlib.import_module(f"{new}.{sub}")
    sys.modules[old] = pkg
    return pkg
