"""DESK-0.2: importing the desk package is cheap and pulls in no heavy or optional dependency."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

HEAVY = ("tensorflow", "fastapi", "nicegui", "ccxt")


def test_import_desk_loads_no_heavy_dependency():
    code = (
        "import sys, neural_trade.desk\n"
        f"bad = [m for m in {HEAVY!r} if m in sys.modules]\n"
        "print('LOADED:' + ','.join(bad))\n"
    )
    src = str(Path(__file__).resolve().parents[2] / "src")  # test this checkout, not the editable install's
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([src, os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep)}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env).stdout
    assert out.strip().splitlines()[-1] == "LOADED:", out


def test_desk_exports_version_and_settings():
    import neural_trade
    import neural_trade.desk as desk

    assert desk.__version__ == neural_trade.__version__
    assert desk.DeskSettings is not None and desk.DESK_HOME.parts[-2:] == ("var", "desk")


# modules that are allowed to import the web or exchange stacks once their items land (tests there must skip without them)
_HEAVY_ALLOWED = ("neural_trade.desk.api", "neural_trade.desk.ui", "neural_trade.desk.broker.ccxt_broker",
                  "neural_trade.desk.data.exchange")


def test_every_desk_module_outside_api_ui_exchange_loads_no_heavy_dependency():
    """Lead's guard after QA of NT-265: a desk module that reaches TensorFlow through a reused package
    (for example neural_trade.notebook before NT-291) must fail here, not in the server at runtime."""
    code = (
        "import importlib, pkgutil, sys\n"
        "import neural_trade.desk as d\n"
        f"skip = {_HEAVY_ALLOWED!r}\n"
        "for m in pkgutil.walk_packages(d.__path__, 'neural_trade.desk.'):\n"
        "    if m.name.startswith(skip) or m.name.endswith('__main__'):\n"
        "        continue\n"
        "    importlib.import_module(m.name)\n"
        f"bad = [m for m in {HEAVY!r} if m in sys.modules]\n"
        "print('LOADED:' + ','.join(bad))\n"
    )
    src = str(Path(__file__).resolve().parents[2] / "src")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([src, os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep)}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env).stdout
    assert out.strip().splitlines()[-1] == "LOADED:", out
