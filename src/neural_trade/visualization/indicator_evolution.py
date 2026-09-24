"""The 18 learned indicator periods: how training moved them, and what the model applies.

The indicator layer (``models/layers/learnable_indicators.py``) learns 18 EWMA periods: three
copies (#0 / #1 / #2) each of a moving average, a MACD (fast / slow / signal), an RSI and a
Bollinger band. Every copy starts from its own configured period (``MA_SPANS``, ``MACD_SETTINGS``,
``RSI_PERIODS``, ``BB_PERIODS``). What training learns is one *base* logit per period, and
metrics.jsonl logs its period (``period/*``). The model does not apply the base period as it is:
each window shifts every logit by ``0.5 * tanh(Dense(mean, max of the normalised window))``, so
the period applied to a window differs from the base (about x0.6 to x1.6 on long periods).

* :func:`indicator_evolution` (registry entry) - base periods per family over the epochs on a log
  scale, starting at the recorded or configured start; the change of every period from its start;
  and each period's correlation with the validation loss (epoch-to-epoch changes against a noise
  band, with the level correlation the old notebook printed kept as a hollow marker).
* :func:`indicator_summary` - the table behind it: start, after epoch 1, last, min, max, change %,
  recent slope, distance to the clip bounds, CV and the three correlations (plus the applied
  percentiles when ``applied`` is given).
* :func:`applied_periods` - the periods a model applies to a block of windows. It is cheap: one
  2x18 Dense on the mean and max of each window.
* :func:`indicator_applied_periods` - base against applied periods, one row per period.

``data``: a metrics.jsonl / indicator_params_history.csv path, a list of metrics rows, or a
DataFrame with the period columns (ma_period_*, macd_*_{fast,slow,signal}, rsi_period_*, bb_period_*).
The start of each period comes from, in order: ``start=``; ``period_init.json`` next to the
metrics file (written by the JSONL epoch logger when training begins, so it is right after a warm
start too); the config's periods (the start when training from scratch) - ``config=``, or the
run's ``config.yaml`` next to the metrics file when no config is passed, so the figure and the
table agree however they are called. Without any of these, changes are measured from the end of
epoch 1 and the figure says so.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from neural_trade.core.indicator_periods import MACD_ROLES, PERIOD_INIT_FILE, configured_periods
from neural_trade.visualization import stats as S
from neural_trade.visualization import theme as T

__all__ = ["applied_periods", "clip_bounds", "configured_periods", "indicator_applied_periods",
           "indicator_evolution", "indicator_summary", "label"]

PREFIXES = ("ma_period_", "macd_", "rsi_period_", "bb_period_")
FAMILY = {"ma_period_": "MA", "macd_": "MACD", "rsi_period_": "RSI", "bb_period_": "BB"}
PANEL_TITLES = {"ma_period_": "Moving-average periods", "macd_": "MACD periods",
                "rsi_period_": "RSI periods", "bb_period_": "Bollinger-band periods"}
# Copy #0 / #1 / #2 inside a family: the non-horizon categorical slots, in order.
COPY_COLORS = tuple(T.OTHER_SERIES[:3])
INDEX_COLORS = COPY_COLORS                      # the old name
# Dotted means "training" in every figure, so the MACD signal line is dash-dot.
MACD_DASH = {"fast": "solid", "slow": "dash", "signal": "dashdot"}
# The applied-median diamond of each MACD role (keyed with the role's line in the top legend).
ROLE_SYMBOL = {"fast": "diamond", "slow": "diamond-wide", "signal": "diamond-tall", None: "diamond"}
CHANGE_COLOR = T.OTHER_SERIES[5]                # change bars: one plain colour (the label names the period)
BEYOND_NOISE_COLOR = T.OTHER_SERIES[3]          # correlation bars outside the noise band
WITHIN_NOISE_COLOR = T.rgba(T.NEUTRAL, 0.55)
SERVED_LINE = dict(color=T.rgba(T.INK_2, 0.55), width=1.5)     # solid: the clip bounds are dashed
STRIP_FILL = T.rgba(T.NEUTRAL, 0.08)            # the applied-median strip right of the last epoch
RUN_CONFIG_FILE = "config.yaml"                 # the run's config, next to metrics.jsonl
MIN_EPOCHS_FOR_CORR = 8                         # fewer epochs: every correlation is NaN (n=2 always gives +/-1)
NEAR_BOUND = 0.05                               # "near a clip bound": within 5% of it
MIN_PANEL_PX = 430                              # a panel's width in a 1100 px wide output: text must fit it

_NAME = re.compile(r"^(ma_period_|macd_|rsi_period_|bb_period_)(\d+)(?:_(fast|slow|signal))?$")
_START_TEXT = {"config": "configured start",
               "logged": "start (recorded when training began)",
               "warm": "start (recorded: warm start, not the config)",
               "given": "start (as given)"}
_TICKS_1 = (1, 2, 5, 10, 20, 50, 100, 200, 500)
_TICKS_2 = (1.5, 3, 4, 6, 8, 15, 30, 40, 60, 80, 150, 300)


# ---------------------------------------------------------------------------------- data
def _path_of(data) -> Optional[Path]:
    if isinstance(data, (str, Path)) or hasattr(data, "read_text"):
        return Path(str(data))
    return None


def _frame(data) -> pd.DataFrame:
    if isinstance(data, (str, Path)) or hasattr(data, "read_text"):
        path = str(data)
        if path.endswith(".jsonl"):
            from neural_trade.telemetry.epoch_logger import read_metrics

            data = read_metrics(path)
        else:
            return pd.read_csv(path)
    if isinstance(data, list):
        rows = []
        for r in data:
            row = {k.split("/", 1)[1]: v for k, v in r.items() if k.startswith("period/")}
            row.update({k: r.get(k) for k in ("epoch", "val_loss", "loss") if k in r})
            rows.append(row)
        return pd.DataFrame(rows)
    df = pd.DataFrame(data).copy()
    return df.rename(columns={c: c.split("/", 1)[1] for c in df.columns if c.startswith("period/")})


def _parse(col: str) -> Tuple[Optional[str], int, Optional[str]]:
    m = _NAME.match(col)
    return (m.group(1), int(m.group(2)), m.group(3)) if m else (None, 0, None)


def _period_cols(df):
    return [c for c in df.columns if _NAME.match(str(c))]


def _order_key(col):
    prefix, idx, role = _parse(col)
    return (PREFIXES.index(prefix) if prefix in PREFIXES else 9, idx,
            MACD_ROLES.index(role) if role in MACD_ROLES else 0)


def label(col: str) -> str:
    """Readable name of a period column: 'MA #0', 'MACD #1 slow', 'RSI #2', 'BB #0'."""
    prefix, idx, role = _parse(col)
    if prefix is None:
        return str(col)
    return f"{FAMILY[prefix]} #{idx}" + (f" {role}" if role else "")


def _num(s) -> np.ndarray:
    return pd.to_numeric(pd.Series(s), errors="coerce").to_numpy(float)


def _val_loss(df):
    for k in ("val_loss", "log_val_loss"):
        if k in df.columns:
            return _num(df[k])
    return None


def _epochs(df) -> np.ndarray:
    """1-based epoch numbers (metrics.jsonl and the CSV log 0-based epochs)."""
    if "epoch" in df.columns:
        e = _num(df["epoch"]) + 1
        if np.all(np.isfinite(e)):
            return e
    return np.arange(1, len(df) + 1, dtype=float)


def _last_finite(v) -> float:
    f = np.asarray(v, float)
    f = f[np.isfinite(f)]
    return float(f[-1]) if len(f) else np.nan


def _corr(a, b, min_n: int = MIN_EPOCHS_FOR_CORR) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < min_n or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def _slope_pct(v, k: int = 5) -> float:
    """Least-squares slope over the last ``k`` logged epochs, in % of the last value per epoch."""
    f = np.asarray(v, float)
    f = f[np.isfinite(f)][-k:]
    if len(f) < 3 or f[-1] == 0:
        return np.nan
    return float(100 * np.polyfit(np.arange(len(f)), f, 1)[0] / f[-1])


def _fmt(v) -> str:
    """A start in a key: 5, 12, 5.5 (3 significant digits, no trailing zeros)."""
    return f"{v:.3g}" if v is not None and np.isfinite(v) else "?"


def _bars(v) -> str:
    """A period in the change-bar labels: always one decimal, so start and end read alike."""
    return f"{v:.1f}" if v is not None and np.isfinite(v) else "?"


# ---------------------------------------------------------------------------------- config / start
def _run_config(path: Optional[Path]):
    """The run's own config.yaml next to the metrics file (what it trained with), or None."""
    if path is None:
        return None
    f = path.with_name(RUN_CONFIG_FILE)
    if not f.exists():
        return None
    try:
        from neural_trade.core.config import Config

        return Config.from_yaml(f)
    except Exception:  # noqa: BLE001 - a config from another version must not break a figure
        return None


def _config_for(data, config):
    return config if config is not None else _run_config(_path_of(data))


def clip_bounds(config) -> Tuple[Optional[float], Optional[float]]:
    """(floor, ceiling) the base periods are clipped to after every step (MOMENTUM_CLIP_MIN / _MAX)."""
    if config is None:
        return None, None
    lo = getattr(config, "MOMENTUM_CLIP_MIN", None)
    hi = getattr(config, "MOMENTUM_CLIP_MAX", None) or getattr(config, "LOOKBACK", None)
    return (float(lo) if lo else None), (float(hi) if hi else None)


def _find_start(path: Optional[Path], config, start) -> Tuple[Dict[str, float], Optional[str]]:
    """(start period per column, source): ``start=``, else period_init.json, else the config."""
    if start:
        return {k: float(v) for k, v in dict(start).items()}, "given"
    if path is not None:
        f = path.with_name(PERIOD_INIT_FILE)
        if f.exists():
            try:
                rec = json.loads(f.read_text(encoding="utf-8"))
                per = {k: float(v) for k, v in rec["periods"].items() if v is not None}
                if per:
                    return per, ("warm" if rec.get("matches_config") is False else "logged")
            except (OSError, ValueError, KeyError, TypeError, AttributeError):
                pass
    per = configured_periods(config)
    return (per, "config") if per else ({}, None)


def _applied_frame(applied) -> Optional[pd.DataFrame]:
    if applied is None:
        return None
    if isinstance(applied, pd.DataFrame):
        return applied
    return pd.DataFrame({k: np.asarray(v, float).reshape(-1) for k, v in dict(applied).items()})


# ---------------------------------------------------------------------------------- table
def _summary(df, config, init, source, app) -> pd.DataFrame:
    cols = _period_cols(df)
    vl = _val_loss(df)
    ep = np.arange(len(df), dtype=float)
    lo, hi = clip_bounds(config)
    base = (app.attrs.get("base") or {}) if app is not None else {}
    rows, n_changes = {}, 0
    for c in cols:
        v = _num(df[c])
        s0 = float(init.get(c, np.nan))
        first = float(v[0]) if len(v) else np.nan
        last = _last_finite(v)
        ref = s0 if np.isfinite(s0) and s0 > 0 else first
        both = np.concatenate([[s0], v])
        row = {"indicator": label(c)}
        if init:
            row["start"] = s0
        row.update({"after epoch 1": first, "last": last,
                    "min": float(np.nanmin(both)) if np.isfinite(both).any() else np.nan,
                    "max": float(np.nanmax(both)) if np.isfinite(both).any() else np.nan,
                    "change %": 100 * (last - ref) / ref if np.isfinite(ref) and ref else np.nan,
                    "slope %/epoch (last 5)": _slope_pct(v)})
        if lo and hi:
            ok = np.isfinite(last) and last > 0
            row["headroom %"] = 100 * min(last / lo - 1, hi / last - 1) if ok else np.nan
            row["near bound"] = ("ceiling" if ok and hi / last - 1 <= NEAR_BOUND
                                 else "floor" if ok and last / lo - 1 <= NEAR_BOUND else "")
        mean = np.nanmean(v) if np.isfinite(v).any() else np.nan
        row["CV"] = float(np.nanstd(v) / mean) if np.isfinite(mean) and mean else np.nan
        row["corr with val loss"] = _corr(v, vl) if vl is not None else np.nan
        row["r with epoch"] = _corr(v, ep)
        if vl is not None and len(v) > 1:
            dv, dl = np.diff(v), np.diff(vl)
            n_changes = max(n_changes, int((np.isfinite(dv) & np.isfinite(dl)).sum()))
            row["r of changes"] = _corr(dv, dl, MIN_EPOCHS_FOR_CORR - 1)
        else:
            row["r of changes"] = np.nan
        if app is not None and c in app.columns:
            a = app[c].to_numpy(float)
            a = a[np.isfinite(a)]
            if len(a):
                p5, p50, p95 = np.percentile(a, [5, 50, 95])
                b = float(base.get(c, np.nan))
                row.update({"served base": b, "applied p5": p5, "applied p50": p50, "applied p95": p95,
                            "applied p50 vs base %": 100 * (p50 / b - 1) if np.isfinite(b) and b else np.nan})
        rows[c] = row
    out = pd.DataFrame.from_dict(rows, orient="index")
    if len(out):
        out = out.sort_values("change %", key=np.abs, ascending=False, na_position="last")
    out.attrs.update(change_from=source or "epoch 1", n_changes=n_changes)
    return out


def indicator_summary(data, config=None, *, start=None, applied=None) -> pd.DataFrame:
    """One row per learned period (index = the metrics name), sorted by |change %|.

    Columns: readable ``indicator`` name; ``start`` (see the module docstring; absent when unknown);
    ``after epoch 1``, ``last``, ``min`` / ``max`` (start included); ``change %`` from the start
    (from the end of epoch 1 when no start is known - ``df.attrs["change_from"]`` says which);
    ``slope %/epoch (last 5)``; with a config, ``headroom %`` to the nearer clip bound and
    ``near bound`` (within 5%); ``CV``; ``corr with val loss`` (levels, the old notebook's statistic,
    mostly a shared trend with training time), ``r with epoch`` and ``r of changes`` (epoch-to-epoch
    changes of the period against those of val loss: the one not driven by the shared trend).
    Correlations are NaN below 8 epochs. With ``applied`` (see :func:`applied_periods`): the served
    base period and the 5 / 50 / 95th percentiles of the applied periods.

    Without ``config``, a metrics path uses the run's config.yaml next to it (the same start the
    figure uses when the notebook passes the config).
    """
    df = _frame(data)
    config = _config_for(data, config)
    init, source = _find_start(_path_of(data), config, start)
    return _summary(df, config, init, source, _applied_frame(applied))


# ---------------------------------------------------------------------------------- applied periods
def _resolve_model(obj, normalizer):
    """(functional model, window normaliser or None) from a Predictor, a TrainResult or a Keras model."""
    net, norm = obj, normalizer
    if hasattr(obj, "bundle") and hasattr(obj, "model"):                     # serving.Predictor
        net, norm = obj.model, normalizer or obj.bundle.normalizer
    elif hasattr(obj, "target_scaler") and hasattr(obj, "model"):            # training.TrainResult
        net, norm = obj.model, normalizer or getattr(obj, "normalizer", None)
        if norm is None:
            from neural_trade.data.scaling import WindowNormalizer

            s = float(obj.target_scaler.scale_[0])
            norm = WindowNormalizer("window_relative", s if s > 0 else 1.0)  # as ArtifactBundle.from_result
    return getattr(net, "base_model", None) or net, norm


def _indicator_layer(net):
    try:
        return net.get_layer("learnable_indicators")
    except (ValueError, AttributeError):
        pass
    for layer in getattr(net, "layers", ()):
        if hasattr(layer, "get_learned_parameters") and hasattr(layer, "get_indicator_trainable_variables"):
            return layer
    raise ValueError("the model has no learnable-indicators layer")


def applied_periods(model, windows, *, normalizer=None, block: Optional[str] = None,
                    batch_size: int = 4096) -> pd.DataFrame:
    """The period each window actually gets, per learned period: ``[N windows, 18]``.

    ``model``: a :class:`~neural_trade.serving.Predictor` (the served weights), a ``TrainResult``
    or a Keras model holding the ``learnable_indicators`` layer. ``windows``: raw close windows
    ``[N, LOOKBACK]``; they are normalised as the model saw them (the predictor's / result's window
    normaliser, or ``normalizer=``). A bare Keras model without ``normalizer=`` takes the windows as
    already normalised. Runs only the meta-adjust Dense on the window mean and max, so it is cheap
    on the CPU.

    Per window and period: ``period = 2 / sigmoid(logit + meta_scale * meta_adjust) - 1``, the same
    transform the layer applies. ``attrs``: ``base`` (the base periods, meta_adjust = 0, what
    metrics.jsonl logs), ``n`` windows, ``block`` name, ``meta_scale``.
    """
    import tensorflow as tf

    net, norm = _resolve_model(model, normalizer)
    layer = _indicator_layer(net)
    X = np.asarray(windows, dtype="float32")
    if X.ndim == 1:
        X = X[None, :]
    Xn = norm.transform(X, X[:, -1]) if norm is not None else X
    meta = tf.keras.Model(net.inputs, layer.input[1])
    adj = np.asarray(meta.predict(Xn, batch_size=int(batch_size), verbose=0), dtype="float64")
    base = {k: float(v) for k, v in layer.get_learned_parameters().items()}
    logits = np.array([float(tf.keras.backend.get_value(v)) for v in layer.get_indicator_trainable_variables()])
    if adj.ndim != 2 or adj.shape[1] != len(base) or len(logits) != len(base):
        raise ValueError(f"meta_adjust has {adj.shape[-1]} columns for {len(base)} learned periods")
    scale = float(getattr(layer, "meta_scale", 0.5))
    alpha = 1.0 / (1.0 + np.exp(-(logits[None, :] + scale * adj)))
    per = np.maximum(2.0 / (alpha + float(getattr(layer, "epsilon", 1e-8))) - 1.0, 0.0)
    out = pd.DataFrame(per.astype("float32"), columns=list(base))
    out.attrs.update(base=base, n=int(len(out)), block=block or "input", meta_scale=scale)
    return out


def _served_epoch(df, cols, x, app) -> Optional[float]:
    """The logged epoch whose base periods equal the applied frame's base (the served weights)."""
    base = (app.attrs.get("base") or {}) if app is not None else {}
    common = [c for c in cols if c in base and base[c]]
    if not common:
        return None
    b = np.array([base[c] for c in common], float)
    vals = np.column_stack([_num(df[c]) for c in common])
    rel = np.nanmax(np.abs(vals - b) / np.abs(b), axis=1)
    hit = np.where(rel < 1e-3)[0]
    return float(x[hit[-1]]) if len(hit) else None


# ---------------------------------------------------------------------------------- figure helpers
def _log_ticks(lo: float, hi: float, px: float, min_px: float = 20.0):
    """Tick values for a log axis spanning [lo, hi] over ``px`` pixels, at least ``min_px`` apart
    (explicit ticks: Plotly's own minor labels read '2' for 20 on short ranges)."""
    per_dec = px / max(math.log10(hi) - math.log10(lo), 1e-9)
    chosen = []
    for t in list(_TICKS_1) + list(_TICKS_2):
        if lo <= t <= hi and all(abs(math.log10(t) - math.log10(c)) * per_dec >= min_px for c in chosen):
            chosen.append(t)
    if len(chosen) < 2:
        chosen = sorted(set(chosen) | {float(f"{v:.2g}") for v in np.geomspace(lo * 1.02, hi / 1.02, 3)})
    return sorted(chosen)


def _log_range(lo: float, hi: float, px: float, above=(), below=(), pad_px: float = 8.0) -> Tuple[float, float]:
    """log10 range of a log axis ``px`` high that shows [lo, hi] with ``pad_px`` to spare (markers are
    not cut at the edges) and keeps ``(value, px)`` free above each value of ``above`` / below each
    of ``below`` (room for a reference line's label)."""
    top = [(math.log10(hi), pad_px)] + [(math.log10(v), p) for v, p in above]
    bot = [(math.log10(lo), pad_px)] + [(math.log10(v), p) for v, p in below]
    a, b = min(t for t, _ in bot), max(t for t, _ in top)
    if b - a < 1e-3:                            # a flat panel: show about a factor of 1.3 around it
        a, b = a - 0.06, b + 0.06
    for _ in range(30):                         # fixed point: the pads are a share of the range they widen
        per_dec = max(px, 60.0) / (b - a)
        a = min(t - p / per_dec for t, p in bot)
        b = max(t + p / per_dec for t, p in top)
    return a, b


def _epoch_ticks(last: int, with_start: bool):
    """Epoch ticks 1, step, 2*step ... (plus the last epoch) and 'start' at 0."""
    vals = []
    if last >= 1:
        step = next((s for s in (1, 2, 5, 10, 20, 25, 50, 100, 200, 500) if last / s <= 7), 1000)
        vals = sorted(({1} if step == 1 or not with_start else set()) | set(range(step, last + 1, step))) or [last]
        if last - vals[-1] >= step / 2:
            vals.append(last)
    return ([0] if with_start else []) + vals, (["start"] if with_start else []) + [str(v) for v in vals]


def _panel_px(fig, row, col, height, margin_tb):
    yd = fig.get_subplot(row, col).yaxis.domain
    return (yd[1] - yd[0]) * max(height - margin_tb, 100)


# ---------------------------------------------------------------------------------- the figure
_GRID = {"ma_period_": (1, 1), "macd_": (1, 2), "rsi_period_": (2, 1), "bb_period_": (2, 2)}
_LEGEND = {"ma_period_": "legend2", "macd_": "legend3", "rsi_period_": "legend4", "bb_period_": "legend5"}
_CORR_TITLE = "r with val loss"


def _strip(last_x: float) -> Tuple[float, Dict[Optional[str], float], float]:
    """The applied-median strip right of the last epoch: (left edge, x of each MACD role's column,
    right end of the axis). It scales with the epoch count, so the columns stay apart on long runs."""
    u = max(1.0, last_x / 20.0)
    xs = {"fast": last_x + 0.9 * u, "slow": last_x + 1.7 * u, "signal": last_x + 2.5 * u}
    xs[None] = xs["slow"]                       # MA / RSI / BB: one column, in the middle
    return last_x + 0.45 * u, xs, last_x + 3.0 * u


def _near_bound_flags(summary, fcols) -> Dict[str, list]:
    """{'ceiling' | 'floor': [(label, % of the bound)]} for the base periods within 5% of a clip bound."""
    out: Dict[str, list] = {}
    if "near bound" not in summary.columns:
        return out
    for c in fcols:
        where = summary.at[c, "near bound"] if c in summary.index else ""
        if where:
            out.setdefault(where, []).append((summary.at[c, "indicator"], float(summary.at[c, "last"])))
    return out


def indicator_evolution(data, config=None, *, height: Optional[int] = None, start=None, applied=None,
                        title: Optional[str] = None, **_):
    """Base periods over training (log scale), change from the start, correlation with val loss.

    ``config`` gives the configured start, the clip bounds and the lookback (without it, a metrics
    path uses the run's config.yaml next to it). ``start``: explicit start periods ({name: period}).
    ``applied``: the :func:`applied_periods` of the served model on some windows; adds each period's
    median applied period as a diamond in a strip right of the last epoch (MACD: one column and one
    diamond shape per role).
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    df = _frame(data)
    config = _config_for(data, config)
    cols = sorted(_period_cols(df), key=_order_key)
    init, source = _find_start(_path_of(data), config, start)
    app = _applied_frame(applied)
    summary = _summary(df, config, init, source, app)
    x = _epochs(df)
    n = len(x)
    last_x = float(x[-1]) if n else 0.0
    height = int(height or 1200)
    margin_t, margin_b = 172, 70
    lo_b, hi_b = clip_bounds(config)
    lookback = getattr(config, "LOOKBACK", None) if config is not None else None
    served = _served_epoch(df, cols, x, app)
    diamonds = app is not None and served is not None
    strip_lo, strip_x, strip_hi = _strip(last_x)
    u = max(1.0, last_x / 20.0)
    x_lo = -0.6 * u if init else 1 - 0.6 * u
    x_hi = strip_hi if diamonds else last_x + 0.6 * u
    start_word = {"config": "configured start", "logged": "recorded start", "warm": "recorded start",
                  "given": "start"}.get(source, "end of epoch 1")
    change_title = f"Change of each base period, {start_word} to epoch {int(last_x) if n else '?'} (%)"
    titles = (PANEL_TITLES["ma_period_"], PANEL_TITLES["macd_"], PANEL_TITLES["rsi_period_"],
              PANEL_TITLES["bb_period_"], change_title, _CORR_TITLE)
    fig = make_subplots(rows=3, cols=2, subplot_titles=titles, row_heights=[0.28, 0.28, 0.44],
                        vertical_spacing=0.085, horizontal_spacing=0.1)

    used_bounds = False
    for prefix, (r, k) in _GRID.items():
        fcols = [c for c in cols if _parse(c)[0] == prefix]
        if not fcols:
            continue
        legend_id = _LEGEND[prefix]
        span, base_span = [], []
        for c in fcols:
            _, idx, role = _parse(c)
            color = COPY_COLORS[idx % 3]
            v = T.positive(_num(df[c]))
            s0 = init.get(c)
            has0 = s0 is not None and np.isfinite(s0) and s0 > 0
            xs = ([0.0] if has0 else []) + [float(e) for e in x]
            ys = np.concatenate([[s0] if has0 else [], v]).astype("float32")
            span.extend(ys[np.isfinite(ys)].tolist())
            base_span.extend(ys[np.isfinite(ys)].tolist())
            text = ([start_word] if has0 else []) + [f"epoch {int(e)}" for e in x]
            symbols = (["circle-open"] if has0 else []) + ["circle"] * n
            sizes = ([10] if has0 else []) + [5 if n <= 40 else 0] * n
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines+markers", name=_copy_name(prefix, idx, init), legend=legend_id,
                legendgroup=f"{prefix}{idx}", showlegend=role in (None, "fast"),
                line=dict(color=color, width=2, dash=MACD_DASH.get(role, "solid")),
                marker=dict(symbol=symbols, size=sizes, color=color, line=dict(color=color, width=1.5)),
                text=text, hovertemplate=f"{label(c)} base period<br>%{{text}}: %{{y:.2f}} bars<extra></extra>"),
                r, k)
        if diamonds:
            dc = [c for c in fcols if c in app.columns]
            vals = [app[c].to_numpy(float) for c in dc]
            p50 = [float(np.nanmedian(v)) for v in vals]
            q = [np.nanpercentile(v, [5, 95]) for v in vals]
            roles = [_parse(c)[2] for c in dc]
            span.extend(p50)
            n_app, blk = app.attrs.get("n", len(app)), app.attrs.get("block", "input")
            fig.add_vrect(x0=strip_lo, x1=x_hi, fillcolor=STRIP_FILL, line_width=0, layer="below", row=r, col=k)
            fig.add_trace(go.Scatter(
                x=[strip_x[ro] for ro in roles], y=np.asarray(p50, "float32"), mode="markers", showlegend=False,
                legendgroup="applied", name="applied median",
                marker=dict(symbol=[ROLE_SYMBOL[ro] for ro in roles], size=11, color=[COPY_COLORS[_parse(c)[1] % 3]
                                                                                        for c in dc],
                            line=dict(color=T.PAPER, width=1)),
                text=[f"{label(c)}: median applied period {m:.2f} bars (5-95%: {a:.1f}-{b:.1f}) over "
                      f"{n_app:,} {blk} windows<br>served weights (epoch {int(served)}): base "
                      f"{app.attrs['base'].get(c, np.nan):.2f} bars" for c, m, (a, b) in zip(dc, p50, q)],
                hovertemplate="%{text}<extra></extra>"), r, k)
        # clip bounds of the BASE period (the applied periods are not clipped): drawn only when the
        # panel's base periods come near one, over the epochs only; the label names any base period
        # that ends within 5% of it
        pmin, pmax = (min(base_span), max(base_span)) if base_span else (1.0, 10.0)
        flags = _near_bound_flags(summary, fcols)
        room = {"ceiling": [], "floor": []}         # (bound, px) kept free above / below a bound for its label
        for bound, near, word in ((hi_b, pmax >= 0.8 * hi_b if hi_b else False, "ceiling"),
                                  (lo_b, pmin <= 1.5 * lo_b if lo_b else False, "floor")):
            if not near:
                continue
            used_bounds = True
            room[word].append((bound, 19.0))
            fig.add_trace(go.Scatter(x=[x_lo, last_x + 0.3 * u], y=[bound, bound], mode="lines", showlegend=False,
                                     legendgroup="bound", name=f"clip {word}", hoverinfo="skip",
                                     line=dict(color=T.NEUTRAL, width=1, dash="dash")), r, k)
            what = f"clip {word} {bound:g} bars" + (" = lookback" if lookback and bound == lookback else "")
            hits = flags.get(word, [])
            if hits:
                what += " · " + ", ".join(f"{lab} ends at {100 * v / bound:.0f}%" for lab, v in hits) + " of it"
            xr, yr = _axis_refs(fig, r, k)
            # in the free band beyond the bound (no base period passes a bound; the applied diamonds sit
            # in the strip, right of the label's end), on the panel colour so a served-epoch line
            # crossing it does not strike through the text
            fig.add_annotation(x=last_x, y=math.log10(bound), xref=xr, yref=yr, text=what, showarrow=False,
                               xanchor="right", yanchor="bottom" if word == "ceiling" else "top",
                               yshift=2 if word == "ceiling" else -2, bgcolor=T.SURFACE, borderpad=1,
                               font=dict(size=10, color=T.INK if hits else T.MUTED))
        px = _panel_px(fig, r, k, height, margin_t + margin_b)
        lo_v, hi_v = (min(span), max(span)) if span else (1.0, 10.0)
        lo_l, hi_l = _log_range(lo_v, hi_v, px, room["ceiling"], room["floor"])
        ticks = _log_ticks(10 ** lo_l, 10 ** hi_l, px)
        fig.update_yaxes(type="log", range=[lo_l, hi_l], tickvals=ticks,
                         ticktext=[f"{t:g}" for t in ticks], title_text="base period, bars (log)", row=r, col=k)
        tv, tt = _epoch_ticks(int(last_x), bool(init))
        fig.update_xaxes(range=[x_lo, x_hi], tickvals=tv, ticktext=tt, title_text="epoch", title_standoff=4,
                         row=r, col=k)
        if served is not None and served != last_x:
            fig.add_vline(x=served, line=SERVED_LINE, row=r, col=k)

    order = summary.dropna(subset=["change %"]).sort_values("change %") if len(summary) else summary
    labels = order["indicator"].tolist() if len(order) else []
    _change_panel(fig, go, order, labels, init)
    _corr_panel(fig, go, summary, order, labels)

    # figure-wide keys (the per-copy keys head each family panel); legend-only traces, parked on the
    # change panel's axes so they never make an empty family panel look drawn
    keys = []
    if init and cols:
        keys.append(go.Scatter(mode="markers", name=_START_TEXT[source], legendgroup="start",
                               marker=dict(symbol="circle-open", size=10, color=T.INK_2, line=dict(width=1.5))))
    if any(_parse(c)[0] == "macd_" for c in cols):
        for role in MACD_ROLES:                 # the line dash and the applied diamond of each MACD role
            keys.append(go.Scatter(mode="lines+markers" if diamonds else "lines", name=f"MACD {role}",
                                   legendgroup=f"role-{role}", line=dict(color=T.INK_2, width=2, dash=MACD_DASH[role]),
                                   marker=dict(symbol=ROLE_SYMBOL[role], size=11, color=T.INK_2)))
    if diamonds:
        keys.append(go.Scatter(mode="markers", legendgroup="applied",
                               marker=dict(symbol="diamond", size=11, color=T.INK_2),
                               name=f"median applied period, epoch {int(served)} weights, "
                                    f"{app.attrs.get('n', len(app)):,} {app.attrs.get('block', 'input')} windows"))
    if served is not None and served != last_x:
        keys.append(go.Scatter(mode="lines", name=f"served weights (epoch {int(served)})", legendgroup="served",
                               line=SERVED_LINE))
    if used_bounds:
        keys.append(go.Scatter(mode="lines", name="clip bound (base periods)", legendgroup="bound",
                               line=dict(color=T.NEUTRAL, width=1, dash="dash")))
    for key in keys:
        key.update(x=[None], y=[None], hoverinfo="skip")
        fig.add_trace(key, 3, 1)

    _notes(fig, {(r, k): "not in this log" for (r, k) in _GRID.values()} | {
        (3, 1): "no learned periods in this log",
        (3, 2): f"needs {MIN_EPOCHS_FOR_CORR}+ epochs with val loss" if cols else "no learned periods in this log"})
    T.apply(fig, title=title or "Learned indicator periods (base values)", height=height,
            subtitle=_subtitle(n, served, app, last_x))
    for prefix, (r, k) in _GRID.items():
        if any(_parse(c)[0] == prefix for c in cols):
            T.panel_legend(fig, _LEGEND[prefix], r, k, PANEL_TITLES[prefix])
    if any(getattr(t, "legend", None) == "legend6" for t in fig.data):
        T.panel_legend(fig, "legend6", 3, 2, _CORR_TITLE)
    xd = fig.get_subplot(3, 1).xaxis.domain                  # the change panel has no keys: same heading style
    for a in fig.layout.annotations or ():
        if a.text == change_title:
            a.update(text=f"<b>{change_title}</b>", x=xd[0], xanchor="left", font=dict(size=12, color=T.INK_2))
    plot_px = max(height - margin_t - margin_b, 100)
    # the figure-wide legend draws keys at their trace sizes, with room for the MACD dash patterns
    # either side of the role's diamond
    fig.update_layout(margin=dict(t=margin_t, b=margin_b, r=24), barmode="overlay",
                      legend=dict(y=1 + 34 / plot_px, yanchor="bottom", x=0, xanchor="left", itemsizing="trace",
                                  itemwidth=44))
    return fig


def _notes(fig, texts):
    """A panel-specific note in the middle of every panel that has no trace."""
    empty = set(T.empty_panels(fig))
    for (r, k), text in texts.items():
        sp = fig.get_subplot(r, k)
        if "y" + sp.yaxis.plotly_name[5:] in empty:
            xd, yd = sp.xaxis.domain, sp.yaxis.domain
            fig.add_annotation(x=(xd[0] + xd[1]) / 2, y=(yd[0] + yd[1]) / 2, xref="paper", yref="paper",
                               text=text, showarrow=False, font=dict(color=T.MUTED, size=12))


def _copy_name(prefix, idx, init) -> str:
    """A copy's key in its panel heading: '#0: 5', MACD '#0: 12/26/9' (copy, then its start)."""
    if prefix == "macd_":
        vals = [init.get(f"macd_{idx}_{r}") for r in MACD_ROLES]
        if all(v is not None and np.isfinite(v) for v in vals):
            return f"#{idx}: {'/'.join(_fmt(v) for v in vals)}"
        return f"#{idx}"
    s0 = init.get(f"{prefix}{idx}")
    return f"#{idx}: {_fmt(s0)}" if s0 is not None and np.isfinite(s0) else f"#{idx}"


def _axis_refs(fig, r, k):
    sp = fig.get_subplot(r, k)
    return sp.xaxis.plotly_name.replace("axis", ""), sp.yaxis.plotly_name.replace("axis", "")


def _text_px(s: str, size: float = 10.0) -> float:
    """Rough rendered width of a short label (digits and letters ~0.56 em, an arrow ~1 em)."""
    return size * (0.56 * len(s) + 0.45 * s.count("→"))


def _outside_text_room(texts, lo0, hi0, panel_px=MIN_PANEL_PX):
    """(left pad, right pad) in data units so every outside bar label fits inside the panel at
    ``panel_px`` wide: a label of w px needs w / panel_px of the axis span past its bar's end."""
    span = max(hi0 - lo0, 1.0)
    a = b = 0.04 * span
    for _ in range(40):                          # fixed point: the pads widen the span they are a share of
        w = span + a + b
        a = max([0.04 * span] + [min(0.4, (_text_px(t) + 10) / panel_px) * w - (hi0 - v) for v, t in texts if v >= 0])
        b = max([0.04 * span] + [min(0.4, (_text_px(t) + 10) / panel_px) * w - (v - lo0) for v, t in texts if v < 0])
    return b, a


def _change_panel(fig, go, order, labels, init):
    if not len(order):
        return
    ch = order["change %"].to_numpy(float)
    texts = []
    for (c, row), v in zip(order.iterrows(), ch):
        s0 = row.get("start", np.nan) if init else np.nan
        frm = s0 if np.isfinite(s0) else row["after epoch 1"]
        txt = f"{_bars(frm)} → {_bars(row['last'])}"
        where = row.get("near bound") or ""
        if where:
            txt += f" at {where}"
        texts.append((float(v), txt))
        fig.add_trace(go.Bar(
            x=[float(v)], y=[row["indicator"]], orientation="h", name=c, showlegend=False,
            marker=dict(color=CHANGE_COLOR), text=[txt], textposition="outside", cliponaxis=False,
            textfont=dict(size=10, color=T.INK_2),
            hovertemplate=f"{row['indicator']} ({c})<br>{frm:.2f} → {row['last']:.2f} bars"
                          + (f" (within 5% of the clip {where})" if where else "")
                          + "<br>change %{x:>+.1f}%<extra></extra>"), 3, 1)
    lo0, hi0 = min(0.0, float(np.nanmin(ch))), max(0.0, float(np.nanmax(ch)))
    pad_lo, pad_hi = _outside_text_room(texts, lo0, hi0)
    fig.add_vline(x=0, line=dict(color=T.NEUTRAL, width=1), row=3, col=1)
    fig.update_xaxes(range=[lo0 - pad_lo, hi0 + pad_hi], title_text="% change of the base period", ticksuffix="%",
                     row=3, col=1)
    fig.update_yaxes(categoryorder="array", categoryarray=labels, row=3, col=1)


def _corr_panel(fig, go, summary, order, labels):
    """Bars: r of epoch-to-epoch changes (period vs val loss) against the no-relation band.
    Hollow circles: r of the levels, which mostly repeats the shared trend with training time."""
    if not len(order) or "r of changes" not in order.columns:
        return None, 0
    m = int(summary.attrs.get("n_changes", 0))
    rc = order["r of changes"].to_numpy(float)
    lev = order["corr with val loss"].to_numpy(float)
    labs = np.asarray(labels, dtype=object)
    if not np.isfinite(rc).any() and not np.isfinite(lev).any():
        return None, m
    band = S.corr_null(m)
    inside = np.isfinite(rc) & (np.abs(rc) <= band)
    outside = np.isfinite(rc) & ~inside
    for sel, name, color in ((inside, "Δ in noise", WITHIN_NOISE_COLOR),
                             (outside, "Δ beyond noise", BEYOND_NOISE_COLOR)):
        if sel.any():
            fig.add_trace(go.Bar(x=rc[sel].astype("float32"), y=labs[sel].tolist(), orientation="h", name=name,
                                 legend="legend6", legendgroup=name, marker=dict(color=color),
                                 hovertemplate="%{y}: r of epoch-to-epoch changes = %{x:>+.2f}<extra></extra>"),
                          3, 2)
    if np.isfinite(lev).any():
        ok = np.isfinite(lev)
        fig.add_trace(go.Scatter(x=lev[ok].astype("float32"), y=labs[ok].tolist(), mode="markers",
                                 name="levels (shared trend)", legend="legend6",
                                 marker=dict(symbol="circle-open", size=9, color=T.INK, line=dict(width=1.5)),
                                 hovertemplate="%{y}: r of levels = %{x:>+.2f} (mostly the trend both share with "
                                               "training time)<extra></extra>"), 3, 2)
    fig.add_vrect(x0=-band, x1=band, fillcolor=T.rgba(T.NEUTRAL, 0.16), line_width=0, layer="below", row=3, col=2)
    fig.add_vline(x=0, line=dict(color=T.NEUTRAL, width=1), row=3, col=2)
    fig.update_xaxes(range=[-1.05, 1.05], row=3, col=2,
                     title_text=f"Pearson r across epochs · bars: epoch-to-epoch changes (Δ) · ○: levels<br>"
                                f"shaded ±{band:.2f} = no relation at 95% ({m} epoch-to-epoch changes)")
    fig.update_yaxes(categoryorder="array", categoryarray=labels, row=3, col=2)
    return band, m


def _subtitle(n, served, app, last_x) -> str:
    line1 = ("Base period = the trainable logit alone (meta_adjust = 0); the model shifts each window's logit "
             "by 0.5·tanh(Dense(mean, max of the window)),")
    shift = "so the period it applies differs per window (about ×0.6 to ×1.6)"
    if app is not None and served is not None:
        line2 = (f"{shift}: ◆ = median applied period, served weights = epoch {int(served)}"
                 + (" (the last)" if served == last_x else "") + ", in the strip right of the last epoch")
    elif app is not None:
        line2 = f"{shift}; the applied periods' base matches no logged epoch, so they are not drawn"
    else:
        line2 = f"{shift}: see the applied-periods figure"
    line3 = (f"{n} epochs · colour = copy #0 / #1 / #2 inside a family (not linked across families); "
             "key = copy: start · log scale: equal slope = equal % change")
    return "<br>".join((line1, line2, line3))


# ---------------------------------------------------------------------------------- applied figure
def indicator_applied_periods(data, config=None, *, metrics=None, start=None, height: Optional[int] = None,
                              title: Optional[str] = None):
    """Base period against the periods the model applies, one row per learned period.

    ``data``: :func:`applied_periods` output (its ``attrs["base"]`` gives the served base periods).
    ``config``: the configured start and the lookback / clip reference lines (without it, the run's
    config.yaml next to ``metrics``). ``metrics``: the run's metrics.jsonl path, so a recorded
    ``period_init.json`` start is used; ``start``: explicit starts.
    """
    import plotly.graph_objects as go

    app = _applied_frame(data)
    names = sorted([c for c in app.columns if _NAME.match(str(c))], key=_order_key)
    base = app.attrs.get("base") or {}
    config = _config_for(metrics, config)
    init, source = _find_start(_path_of(metrics), config, start)
    labs = [label(c) for c in names]
    q = np.nanpercentile(app[names].to_numpy(float), [5, 25, 50, 75, 95], axis=0) if names else np.zeros((5, 0))
    fig = go.Figure()
    for idx in range(3):
        sel = [j for j, c in enumerate(names) if _parse(c)[1] % 3 == idx]
        if not sel:
            continue
        color = COPY_COLORS[idx]
        for lo_q, hi_q, width in ((0, 4, 2), (1, 3, 8)):
            xs, ys = [], []
            for j in sel:
                xs += [q[lo_q, j], q[hi_q, j], None]
                ys += [labs[j], labs[j], None]
            fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", showlegend=False, hoverinfo="skip",
                                     line=dict(color=color, width=width)))
        fig.add_trace(go.Scatter(
            x=q[2, sel].astype("float32"), y=[labs[j] for j in sel], mode="markers", showlegend=False,
            marker=dict(symbol="diamond", size=11, color=color, line=dict(color=T.PAPER, width=1)),
            text=[f"{labs[j]}: applied median {q[2, j]:.2f} bars, middle 50% {q[1, j]:.1f}-{q[3, j]:.1f}, "
                  f"5-95% {q[0, j]:.1f}-{q[4, j]:.1f}; base {base.get(names[j], np.nan):.2f}" for j in sel],
            hovertemplate="%{text}<extra></extra>"))
    bx = [base.get(c, np.nan) for c in names]
    if np.isfinite(bx).any():
        fig.add_trace(go.Scatter(x=np.asarray(bx, "float32"), y=labs, mode="markers",
                                 name="base period (served weights)",
                                 marker=dict(symbol="circle-open", size=11, color=T.INK, line=dict(width=2)),
                                 hovertemplate="%{y}: base %{x:.2f} bars<extra></extra>"))
    sx = [init.get(c, np.nan) for c in names]
    if np.isfinite(sx).any():
        fig.add_trace(go.Scatter(x=np.asarray(sx, "float32"), y=labs, mode="markers", name=_START_TEXT[source],
                                 marker=dict(symbol="line-ns-open", size=12, color=T.NEUTRAL, line=dict(width=2)),
                                 hovertemplate="%{y}: start %{x:.2f} bars<extra></extra>"))
    # keys drawn at their real widths: this legend sizes items by trace (the theme's constant item
    # sizing draws every line key 5 px wide, so 8 px and 2 px bars would look the same)
    for key, nm, sym, width in (("med", "applied median", "diamond", 0), ("iqr", "middle 50% of windows", None, 8),
                                ("p90", "5-95% of windows", None, 2)):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers" if sym else "lines", name=nm, legendgroup=key,
                                 marker=dict(symbol=sym or "circle", size=11, color=T.INK_2),
                                 line=dict(color=T.INK_2, width=width or 2)))
    vals = np.concatenate([q[[0, 4]].reshape(-1), np.asarray(bx, float), np.asarray(sx, float)])
    vals = vals[np.isfinite(vals) & (vals > 0)]
    lookback = getattr(config, "LOOKBACK", None) if config is not None else None
    lo_b, hi_b = clip_bounds(config)
    refs = [(lookback, f"lookback {lookback:g} bars: a longer EWMA never warms up in the window " if lookback else "",
             "right")]
    if lo_b:
        refs.append((lo_b, f" clip floor {lo_b:g} (base periods only)", "left"))
    lo_v, hi_v = (vals.min() / 1.12, vals.max() * 1.12) if len(vals) else (1.0, 100.0)
    for ref, text, anchor in refs:
        if ref and lo_v <= ref * 1.3 and ref / 1.3 <= hi_v:
            lo_v, hi_v = min(lo_v, ref / 1.12), max(hi_v, ref * 1.12)
            # shapes take data units on a log axis, annotations take log10 units
            fig.add_shape(type="line", x0=ref, x1=ref, xref="x", yref="y domain", y0=0,
                          y1=1, line=dict(color=T.NEUTRAL, width=1, dash="dash"))
            fig.add_annotation(x=math.log10(ref), y=1, xref="x", yref="y domain", text=text, showarrow=False,
                               xanchor=anchor, yanchor="bottom", font=dict(size=10, color=T.MUTED))
    for j in range(1, len(names)):                              # thin rules between the families
        if _parse(names[j])[0] != _parse(names[j - 1])[0]:
            fig.add_shape(type="line", xref="x domain", x0=0, x1=1, yref="y", y0=len(names) - j - 0.5,
                          y1=len(names) - j - 0.5, line=dict(color=T.AXIS, width=1))
    margin_t, margin_b = 132, 56
    height = int(height or max(440, 26 * len(names) + margin_t + margin_b + 40))
    ticks = _log_ticks(lo_v, hi_v, 800.0, min_px=34)
    fig.update_xaxes(type="log", range=[math.log10(lo_v), math.log10(hi_v)], tickvals=ticks,
                     ticktext=[f"{t:g}" for t in ticks], title_text="period, bars (log)")
    fig.update_yaxes(categoryorder="array", categoryarray=labs[::-1], title_text=None)
    n = app.attrs.get("n", len(app))
    blk = app.attrs.get("block", "input")
    T.note_on_empty(fig, "no applied periods given")
    T.apply(fig, title=title or "Applied indicator periods (served model)", height=height,
            subtitle=f"{n:,} {blk} windows · each window's period = base logit + "
                     f"{app.attrs.get('meta_scale', 0.5):g}·tanh(Dense(mean, max of the window)), turned into bars"
                     "<br>base period = the same logit with meta_adjust = 0 (what the training figure plots) "
                     "· colour = copy #0 / #1 / #2 inside a family")
    plot_px = max(height - margin_t - margin_b, 100)
    fig.update_layout(margin=dict(t=margin_t, b=margin_b, l=110, r=24),
                      legend=dict(y=1 + 22 / plot_px, itemsizing="trace"))
    return fig
