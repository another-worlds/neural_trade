"""DESK-5.4: Lab figures (plotly JSON of the saved-run figures) and the theme CSS."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from neural_trade.desk.lab import figures as F
from neural_trade.desk.lab.theme_css import DESK_TOKENS, theme_css, theme_tokens

ROOT = Path(__file__).resolve().parents[2]
REAL_RUN = ROOT / "runs" / "20261003T225052Z-91fa363-11993eec"      # light files are tracked in git (NT-010)

pytestmark = pytest.mark.skipif(not REAL_RUN.is_dir(), reason="the light run directory is not in this checkout")


def _tree_state(root: Path):
    return sorted((str(p.relative_to(root)), p.stat().st_size, p.stat().st_mtime_ns)
                  for p in root.rglob("*") if p.is_file())


def _valid(doc):
    assert set(doc) >= {"data", "layout", "config"} and doc["data"]
    assert doc["config"]["responsive"] is True
    json.dumps(doc, allow_nan=False)


# ------------------------------------------------------------------ the list
def test_the_lab_list_is_registered_and_documented():
    from neural_trade.registries.visualizations import Visualizations

    keys = F.lab_figure_keys()
    assert len(set(keys)) == len(keys) == 13
    for k in keys:
        assert k in Visualizations.registry
        assert F.required_files(k)
    with pytest.raises(F.UnknownFigure):
        F.required_files("matplotlib_splits")
    doc = F.__doc__
    assert all(k in doc for k in keys)


# ------------------------------------------------------------------ light run: JSON
@pytest.mark.parametrize("key", F.LIGHT_KEYS)
def test_light_figures_on_a_real_run_are_valid_json_and_the_run_is_untouched(key):
    before = _tree_state(REAL_RUN)
    doc = F.figure_json(REAL_RUN, key)
    _valid(doc)
    assert _tree_state(REAL_RUN) == before
    assert F.figure_json(REAL_RUN, key) == doc                    # deterministic


def test_figure_json_is_what_the_builder_draws_not_a_simplified_copy():
    import plotly.io as pio

    fig = F.build_figure(REAL_RUN, "training_dashboard")
    doc = F.figure_json(REAL_RUN, "training_dashboard")
    assert len(doc["data"]) == len(fig.data)
    assert doc["layout"] == json.loads(pio.to_json(fig, validate=False))["layout"]


# ------------------------------------------------------------------ missing artifacts
def test_missing_artifacts_raise_needs_artifacts_naming_the_files(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    with pytest.raises(F.NeedsArtifacts) as e:
        F.figure_json(run, "training_dashboard")
    assert e.value.missing == ("config.yaml", "metrics.jsonl") and e.value.key == "training_dashboard"
    with pytest.raises(F.NeedsArtifacts) as e:
        F.figure_json(REAL_RUN, "direction_analytics")            # light files only: no prediction blocks
    assert e.value.missing == ("predictions_oos.npz", "predictions_cal.npz")
    assert "predictions_oos.npz" in str(e.value)


# ------------------------------------------------------------------ block figures on a synthetic run
def _synthetic_run(tmp_path: Path, n: int = 600) -> Path:
    from neural_trade.evaluation.frame import PredictionFrame
    from neural_trade.experiments.scorer import PREDICTION_FILES

    run = tmp_path / "block_run"
    run.mkdir()
    shutil.copy(REAL_RUN / "config.yaml", run / "config.yaml")
    shutil.copy(REAL_RUN / "meta.json", run / "meta.json")
    steps = (10, 15, 20)
    hs = [f"h{i}" for i in range(3)]

    def block(seed):
        r = np.random.default_rng(seed)
        close = 100.0 + np.cumsum(r.normal(0, 0.2, n + 30))
        last = close[:n]
        y = np.stack([close[s:s + n] - last for s in steps], axis=1)
        delta = {h: 0.3 * y[:, i] + r.normal(0, 0.1, n) for i, h in enumerate(hs)}
        p = {h: 1 / (1 + np.exp(-(delta[h] * 3))) for h in hs}
        sd = {h: np.full(n, 0.5) for h in hs}
        frame = PredictionFrame(
            y=y, last_close=last, delta=delta, direction_prob=p, variance_scaled=sd, pred_scale=1.0,
            horizon_steps=steps, split="test", direction_prob_calibrated={h: np.clip(p[h], .05, .95) for h in hs},
            intervals={h: (delta[h] - 1.0, delta[h] + 1.0) for h in hs},
            meta={"delta_raw": {h: delta[h] * 1.5 for h in hs}, "delta_scale": {h: 0.66 for h in hs}})
        o = last + r.normal(0, 0.05, n)
        extra = {"bar_open": o, "bar_high": last + 0.3, "bar_low": last - 0.3, "bar_close": last,
                 "anchor_timestamp": np.arange(n, dtype=np.int64) * 60_000_000_000 + 1_700_000_000_000_000_000,
                 "anchor_bar": np.arange(n, dtype=np.int64), "sequence_index": np.arange(n, dtype=np.int64),
                 "bar_minutes": 1.0}
        return frame, extra

    for key, seed in (("oos", 1), ("cal", 2)):
        frame, extra = block(seed)
        frame.save_npz(run / PREDICTION_FILES[key], extra=extra)
    return run


@pytest.fixture(scope="module")
def block_run(tmp_path_factory):
    return _synthetic_run(tmp_path_factory.mktemp("lab"))


@pytest.mark.parametrize("key", F.BLOCK_KEYS)
def test_block_figures_return_json_from_the_saved_prediction_blocks(block_run, key):
    before = _tree_state(block_run)
    _valid(F.figure_json(block_run, key))
    assert _tree_state(block_run) == before


def test_one_missing_block_file_is_named(block_run, tmp_path):
    run = tmp_path / "half"
    shutil.copytree(block_run, run)
    (run / "predictions_cal.npz").unlink()
    with pytest.raises(F.NeedsArtifacts) as e:
        F.figure_json(run, "eval_report")
    assert e.value.missing == ("predictions_cal.npz",)


def test_importing_the_modules_and_the_css_load_no_tensorflow():
    code = ("import sys, neural_trade.desk.lab.figures, neural_trade.desk.lab.theme_css as t\n"
            "t.theme_css()\n"
            "print('TF:' + str('tensorflow' in sys.modules))\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                         env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")}).stdout
    assert out.strip().splitlines()[-1] == "TF:False"


# ------------------------------------------------------------------ plotly.js
def test_served_plotlyjs_version_equals_the_python_packages():
    from plotly.offline import get_plotlyjs, get_plotlyjs_version

    version, text = F.plotlyjs_bundle()
    assert version == get_plotlyjs_version()
    assert text == get_plotlyjs()
    assert int(version.split(".")[0]) >= 3                       # typed arrays (bdata) need plotly.js 3.x


# ------------------------------------------------------------------ theme css
def test_theme_css_has_every_token_exactly_and_follows_theme_py():
    from neural_trade.visualization import theme as T

    css = theme_css()
    vars_ = dict(re.findall(r"--nt-([a-z0-9-]+): (#[0-9a-fA-F]{6});", css))
    expected = {
        "paper": T.PAPER, "surface": T.SURFACE, "ink": T.INK, "ink-2": T.INK_2, "muted": T.MUTED, "grid": T.GRID,
        "axis": T.AXIS, "horizon-h0": T.SERIES[0], "horizon-h1": T.SERIES[1], "horizon-h2": T.SERIES[2],
        "good": T.GOOD, "warning": T.WARNING, "critical": T.CRITICAL, "long": T.LONG_COLOR, "up": T.UP_COLOR,
        "short": T.SHORT_COLOR, "down": T.DOWN_COLOR}
    for name, value in expected.items():
        assert vars_[name] == value, name
    assert vars_["paper"] == "#121211" and vars_["horizon-h1"] == "#d95926"      # the pinned literals of PLAN A8
    assert css.startswith(":root {") and css.endswith("}\n") and "color-scheme: dark" in css


def test_candles_have_their_own_tokens_not_the_status_colours():
    t = theme_tokens()
    status = {t["good"], t["critical"], t["warning"], t["serious"]}
    assert set(DESK_TOKENS) == {"candle-up", "candle-down"}
    assert t["candle-up"] not in status and t["candle-down"] not in status
    assert t["candle-up"] != t["candle-down"]


def test_theme_css_is_deterministic_and_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert theme_css() == theme_css()
    assert list(tmp_path.iterdir()) == []
