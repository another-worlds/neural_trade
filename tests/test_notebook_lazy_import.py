"""NT-291: importing a notebook submodule must not load the interactive classes' heavy dependencies."""
import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
HEAVY = ["tensorflow", "keras", "ipywidgets", "pandas"]

_PROBE = """
import json, sys
import {mod}
print(json.dumps({{m: m in sys.modules for m in {heavy!r}}}))
"""


def _loaded(mod: str) -> dict:
    import os
    env = {**os.environ, "PYTHONPATH": str(SRC), "CUDA_VISIBLE_DEVICES": "-1"}
    out = subprocess.run([sys.executable, "-c", _PROBE.format(mod=mod, heavy=HEAVY)], env=env,
                         capture_output=True, text=True, timeout=300, check=True)
    return json.loads(out.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("mod", ["neural_trade.notebook.longrun", "neural_trade.notebook"])
def test_import_loads_no_heavy_dependency(mod):
    loaded = _loaded(mod)
    assert not any(loaded.values()), f"{mod} loaded {[m for m, v in loaded.items() if v]}"


def test_exports_resolve_to_submodule_objects():
    pytest.importorskip("ipywidgets")
    nb = importlib.import_module("neural_trade.notebook")
    for name, sub in nb._EXPORTS.items():
        assert getattr(nb, name) is getattr(importlib.import_module(f"neural_trade.notebook.{sub}"), name)
        assert name in vars(nb)  # cached after first access
    assert set(nb.__all__) == set(nb._EXPORTS)
    assert set(nb.__all__) <= set(dir(nb))


def test_unknown_name_raises_attribute_error():
    nb = importlib.import_module("neural_trade.notebook")
    with pytest.raises(AttributeError, match="has no attribute 'nope'"):
        nb.nope  # noqa: B018


def test_submodule_import_forms_still_work():
    from neural_trade.notebook import panel_data
    assert panel_data.__name__ == "neural_trade.notebook.panel_data"
