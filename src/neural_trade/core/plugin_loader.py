"""Load plugin modules from a directory so their decorators register extra components.

    from neural_trade.core.plugin_loader import load_plugins
    load_plugins("plugins")            # every *.py below it, except templates/ and _private.py

A plugin is an ordinary module that imports a registry and registers into it, e.g.
``@Metrics.register(name="my_metric")``. Plugins are loaded only when a program asks for it
(``registries.load_all(config)`` reads ``Config.PLUGINS_DIR``), never at import time.
"""
from __future__ import annotations

import importlib.util
import logging
import sys
import warnings
from pathlib import Path
from typing import List

from .exceptions import RegistryError

_log = logging.getLogger(__name__)
SKIP_DIRS = {"templates", "__pycache__"}


def load_plugins(directory, strict: bool = False) -> List[str]:
    """Import every plugin module under ``directory``; return the module names loaded.

    A plugin that fails to import is skipped with a warning, or raises RegistryError when
    ``strict``. Loading the same file twice is a no-op.
    """
    root = Path(directory)
    if not root.is_dir():
        msg = f"plugin directory {root} does not exist"
        if strict:
            raise RegistryError(msg)
        warnings.warn(msg, UserWarning, stacklevel=2)
        return []
    loaded = []
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root)
        if any(part in SKIP_DIRS for part in rel.parts[:-1]) or path.name.startswith("_"):
            continue
        mod_name = "neural_trade_plugins." + ".".join(rel.with_suffix("").parts)
        if mod_name in sys.modules:
            loaded.append(mod_name)
            continue
        try:
            spec = importlib.util.spec_from_file_location(mod_name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = module
            spec.loader.exec_module(module)
            loaded.append(mod_name)
            _log.info("loaded plugin %s", mod_name)
        except Exception as exc:
            sys.modules.pop(mod_name, None)
            msg = f"plugin {path} failed to load: {exc}"
            if strict:
                raise RegistryError(msg) from exc
            warnings.warn(msg, UserWarning, stacklevel=2)
    return loaded
