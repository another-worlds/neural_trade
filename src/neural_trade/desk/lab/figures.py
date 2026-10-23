"""DESK-5.4: Lab figures, the saved-run figures of the Visualizations registry as plotly JSON (CONTRACTS C5
``GET /lab/runs/{id}/figure/{key}``; the HTTP router belongs to the API items and calls these functions).

Read-only: inputs are a run directory's saved files (``config.yaml``, ``metrics.jsonl``, ``predictions_oos.npz``,
``predictions_cal.npz``); nothing is written. A figure whose files are missing raises :class:`NeedsArtifacts`
(the router maps it to 409). The figures are the existing builders, called through
``Visualizations.build(key, data, config)``; nothing is simplified (D-014).

The Lab list (PLAN A8 "Run detail figures", chosen from the registry's entries; the rest are data-only, HTML,
matplotlib, cross-run or need a live model):

* from the light files (``config.yaml`` + ``metrics.jsonl``): ``training_dashboard``, ``training_direction_detail``,
  ``training_loss_terms``, ``indicator_evolution``, ``indicator_family_periods``;
* from the saved prediction blocks (``predictions_oos.npz`` + ``predictions_cal.npz``, heavy, machine-local):
  ``eval_report``, ``direction_analytics``, ``delta_analytics``, ``variance_analytics``, ``confidence_analytics``,
  ``coherence_analytics``, ``trading_dashboard``, ``trade_analytics``.

Not served (gaps): ``discovered_indicators`` and ``indicator_applied_periods`` (need the served model's applied
periods, i.e. TensorFlow inference) and ``permutation_importance`` (needs a computed groups result).

TensorFlow (found): building a figure DOES load TensorFlow (the registries and the notebook helpers reach it;
measured on the light run). So every visualization, registry and experiment import is inside a function, the module
import stays light (``tests/desk/test_import_guard.py``), and the HTTP layer should build figures in a worker, not
at import time.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Tuple

LIGHT_FILES = ("config.yaml", "metrics.jsonl")
BLOCK_FILES = ("config.yaml", "predictions_oos.npz", "predictions_cal.npz")

LIGHT_KEYS = ("training_dashboard", "training_direction_detail", "training_loss_terms", "indicator_evolution",
              "indicator_family_periods")
BLOCK_KEYS = ("eval_report", "direction_analytics", "delta_analytics", "variance_analytics",
              "confidence_analytics", "coherence_analytics", "trading_dashboard", "trade_analytics")

#: the config the browser passes to Plotly.newPlot: figures follow their container
PLOTLY_CONFIG: Dict[str, Any] = {"responsive": True}


class NeedsArtifacts(Exception):
    """The figure's input files are not in the run directory; ``missing`` lists them, ``key`` is the figure."""

    def __init__(self, key: str, missing, run_dir=None):
        self.key = key
        self.missing = tuple(missing)
        self.run_dir = None if run_dir is None else str(run_dir)
        super().__init__(f"figure {key!r} needs {', '.join(self.missing)}"
                         + (f" in {self.run_dir}" if self.run_dir else ""))


class UnknownFigure(KeyError):
    """A key that is not in the Lab list."""

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else ""


def lab_figure_keys() -> Tuple[str, ...]:
    return LIGHT_KEYS + BLOCK_KEYS


def required_files(key: str) -> Tuple[str, ...]:
    if key in LIGHT_KEYS:
        return LIGHT_FILES
    if key in BLOCK_KEYS:
        return BLOCK_FILES
    raise UnknownFigure(f"unknown figure {key!r}; the Lab serves {', '.join(lab_figure_keys())}")


# ------------------------------------------------------------------ plotly.js
_JS_VERSION = re.compile(r"plotly\.js v(\d+\.\d+\.\d+)")


def plotlyjs_bundle() -> Tuple[str, str]:
    """(version, text) of the plotly.js bundle the python package ships (``plotly.offline.get_plotlyjs``).
    Typed-array figure JSON needs plotly.js 3.x, so the page serves this bundle, never a CDN copy. The version
    is read from the bundle's own header."""
    from plotly.offline import get_plotlyjs

    text = get_plotlyjs()
    found = _JS_VERSION.search(text[:500])
    if found is None:
        raise RuntimeError("plotly.js bundle without a version header")
    return found.group(1), text


# ------------------------------------------------------------------ builders
def _meta(run_dir: Path) -> dict:
    try:
        doc = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return doc if isinstance(doc, dict) else {}


def _build_block(key: str, run_dir: Path, config):
    from neural_trade.experiments.scorer import load_block
    from neural_trade.registries.visualizations import Visualizations

    test, bars, extra = load_block(run_dir / "predictions_oos.npz")
    cal, _, _ = load_block(run_dir / "predictions_cal.npz")
    if key in ("trading_dashboard", "trade_analytics"):
        from neural_trade.experiments.scorer import BlockSignals, fit_and_backtest

        strat = _meta(run_dir).get("engine", {}).get("strategy") or {}
        sig = BlockSignals.build(cal, test)
        result, strategy = fit_and_backtest(sig, bars, bar_minutes=float(extra.get("bar_minutes", config.RESAMPLE_MINUTES)),
                                            strategy=strat.get("name"), strategy_params=strat.get("params"))
        if key == "trading_dashboard":
            return Visualizations.build(key, result, config, bars=bars, signals=sig.oos, strategy=strategy,
                                        times=extra.get("anchor_timestamp"))
        return Visualizations.build(key, result, config, bars=bars, signals=sig.oos)
    kw: Dict[str, Any] = {}
    if key == "confidence_analytics":
        from neural_trade.strategy import var_scale_from

        kw["var_scale"] = float(var_scale_from(cal))
    return Visualizations.build(key, test, config, **kw)


def build_figure(run_dir, key: str):
    """The plotly figure object for ``key`` from ``run_dir``'s saved files (read-only)."""
    from neural_trade.notebook.run_report import load_config
    from neural_trade.registries.visualizations import Visualizations

    run_dir = Path(run_dir)
    missing = [name for name in required_files(key) if not (run_dir / name).is_file()]
    if missing:
        raise NeedsArtifacts(key, missing, run_dir)
    config = load_config(run_dir)
    if key in LIGHT_KEYS:
        return Visualizations.build(key, run_dir / "metrics.jsonl", config)
    return _build_block(key, run_dir, config)


def figure_json(run_dir, key: str) -> Dict[str, Any]:
    """``{"data", "layout", "config"}`` for the browser: ``pio.to_json(fig, validate=False)`` parsed back into a
    dict (typed arrays stay ``{"dtype", "bdata"}``, which plotly.js 3.x reads; NaN becomes null) with
    ``config.responsive`` true. Deterministic for the same files."""
    import plotly.io as pio

    doc = json.loads(pio.to_json(build_figure(run_dir, key), validate=False))
    doc["config"] = dict(PLOTLY_CONFIG)
    json.dumps(doc, allow_nan=False)            # fail here, not in the browser, if a NaN slipped through
    return doc


__all__ = ["NeedsArtifacts", "UnknownFigure", "build_figure", "figure_json", "lab_figure_keys", "plotlyjs_bundle",
           "required_files"]
