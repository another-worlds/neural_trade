"""DESK-0.2: importing the desk package is cheap and pulls in no heavy or optional dependency."""
from __future__ import annotations

import subprocess
import sys

HEAVY = ("tensorflow", "fastapi", "nicegui", "ccxt")


def test_import_desk_loads_no_heavy_dependency():
    code = (
        "import sys, neural_trade.desk\n"
        f"bad = [m for m in {HEAVY!r} if m in sys.modules]\n"
        "print('LOADED:' + ','.join(bad))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip().splitlines()[-1] == "LOADED:", out


def test_desk_exports_version_and_settings():
    import neural_trade
    import neural_trade.desk as desk

    assert desk.__version__ == neural_trade.__version__
    assert desk.DeskSettings is not None and desk.DESK_HOME.parts[-2:] == ("var", "desk")
