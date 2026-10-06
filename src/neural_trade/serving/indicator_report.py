"""Self-contained indicator report for one run (NT-048).

This lives in serving, next to the artifact bundle it reads. ``evaluation`` cannot import
it back: the figures already import evaluation, and serving already imports evaluation.

``<run_dir>/indicator_report.html`` embeds plotly. It holds three figures: the learned
indicators against their textbook periods, every logged family period, and the grouped
permutation importance on the validation block. The training package does not import this
module. A notebook session writes the file when it has a run directory; the experiment
engine writes it only when the scenario sets ``run.indicator_report`` (and
``run.save_artifacts``, because the figures are read from ``artifacts/``).

The importance score is per window: the mean squared error of the three scaled price heads,
plus the direction log loss on moves outside the deadband. The band is the block bootstrap
of that per-window loss. Direction importance is the per-horizon drop of the ROC AUC on the
labelled windows; its band recomputes the AUC on every block-bootstrap resample (block at least
the longest horizon in bars). The drop of the per-window hit rate is kept beside it as ``hit_drop``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from neural_trade.core.outputs import PredictiveOutputs
from neural_trade.core.postprocess import HORIZONS
from neural_trade.metrics.direction_labels import direction_labels_np
from neural_trade.metrics.statistics import BLOCK


def _metrics_path(run_dir: Path) -> Path:
    path = run_dir / "metrics.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"{run_dir} has no metrics.jsonl")
    return path


def _price_and_direction(heads, y_scaled, y_raw, last_close, deadband):
    """Per-window loss and per-horizon hits for one forward of the ten model outputs."""
    n = len(y_scaled)
    flat = [np.asarray(head, dtype=float).reshape(n, -1)[:, 0] for head in heads]
    named = PredictiveOutputs(*flat)
    prices = (named.price_h0, named.price_h1, named.price_h2)
    probs = (named.direction_h0, named.direction_h1, named.direction_h2)
    sq = np.mean([(prices[i] - y_scaled[:, i]) ** 2 for i in range(3)], axis=0)
    labels = direction_labels_np(y_raw, last_close, deadband)
    bce = np.zeros(n, dtype=float)
    labels_out = {}
    scores_out = {}
    hit = {}
    for i, h in enumerate(HORIZONS):
        lab, mask = labels[h]
        p = np.clip(probs[i], 1e-6, 1.0 - 1e-6)
        if np.any(mask):
            bce[mask] += -(lab[mask] * np.log(p[mask]) + (1.0 - lab[mask]) * np.log(1.0 - p[mask]))
        correct = np.zeros(n, dtype=float)
        correct[mask] = (p[mask] >= 0.5) == (lab[mask] >= 0.5)
        hit[h] = correct
        labels_out[h] = np.where(mask, lab, np.nan)
        scores_out[h] = np.asarray(p, dtype=float)
    bce /= len(HORIZONS)
    loss_i = sq + bce
    return {"loss": float(np.mean(loss_i)), "loss_i": loss_i, "hit_i": hit,
            "labels": labels_out, "scores": scores_out}


def indicator_importance(predictor, block):
    """Grouped permutation importance of every family instance on a data block (the report uses ``val``)."""
    from neural_trade.evaluation.permutation_importance import importance_from_model
    from neural_trade.utils.seeding import set_arithmetic_rewrite

    cfg = predictor.config
    y = np.asarray(block["y"], dtype=float)
    if len(y) < 1:
        raise ValueError("the validation block has no windows")
    last_close = np.asarray(block["last_close"], dtype=np.float32)
    windows = predictor.bundle.normalizer.transform(np.asarray(block["X_model"], dtype=np.float32), last_close)
    scale = float(predictor.bundle.pred_scale)
    mean = float(predictor.bundle.pred_mean)
    y_scaled = (y - mean) / scale if scale else y * 0.0
    deadband = float(getattr(cfg, "DIR_DEADBAND_BPS", 0.0))
    set_arithmetic_rewrite(cfg)

    def score_fn(heads):
        return _price_and_direction(heads, y_scaled, y, last_close, deadband)

    steps = [int(h) for h in getattr(cfg, "HORIZON_STEPS", [1])]
    return importance_from_model(predictor.model, windows, score_fn, block=max(BLOCK, max(steps)),
                                 horizon_bars=max(steps))


def indicator_figures(run_dir):
    """The three report figures, in write order. Raises when a run file is missing."""
    run_dir = Path(run_dir)
    artifacts = run_dir / "artifacts"
    if not artifacts.is_dir():
        raise FileNotFoundError(f"{run_dir} has no artifacts/ bundle to draw")
    metrics = _metrics_path(run_dir)

    from neural_trade.core.config import Config
    from neural_trade.data.processor import split_arrays
    from neural_trade.registries.visualizations import Visualizations
    from neural_trade.serving.predictor import Predictor
    from neural_trade.visualization.indicator_evolution import applied_periods

    cfg_path = run_dir / "config.yaml"
    predictor = Predictor.from_artifacts(artifacts)
    cfg = Config.from_yaml(cfg_path) if cfg_path.is_file() else predictor.config
    val = split_arrays(cfg)["val"]
    applied = applied_periods(predictor, val["X_model"], block="val")
    discovered = Visualizations.build(
        "discovered_indicators", val["X"], cfg, applied=applied, metrics=metrics, ohlcv=val["X_model"])
    periods = Visualizations.build("indicator_family_periods", metrics, cfg, applied=applied)
    importance = Visualizations.build("permutation_importance", indicator_importance(predictor, val), cfg)
    return discovered, periods, importance


def write_indicator_report(run_dir, *, figures=None) -> Path:
    """Write ``<run_dir>/indicator_report.html``. The first figure embeds plotly; the later ones do not.

    An empty panel is refused before the file is written.
    """
    from neural_trade.visualization import theme as T

    run_dir = Path(run_dir)
    figs = tuple(figures) if figures is not None else indicator_figures(run_dir)
    if len(figs) != 3:
        raise ValueError(f"the indicator report is three figures, got {len(figs)}")
    empty = [(i, T.empty_panels(fig)) for i, fig in enumerate(figs) if T.empty_panels(fig)]
    if empty:
        raise RuntimeError(f"indicator report has an empty panel and was not written: {empty}")
    parts = [fig.to_html(full_html=False, include_plotlyjs=True if i == 0 else False)
             for i, fig in enumerate(figs)]
    html = (
        "<!DOCTYPE html><html><head><meta charset=\"utf-8\">"
        "<title>Indicator report</title></head><body>\n"
        + "\n".join(parts)
        + "\n</body></html>\n"
    )
    path = run_dir / "indicator_report.html"
    path.write_text(html, encoding="utf-8", newline="\n")
    return path
