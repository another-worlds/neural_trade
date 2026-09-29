"""The discovered indicators on price: each learned indicator next to the same indicator at its textbook period.

VISION "What every run delivers": the indicators the network learned are the product, drawn on the price
chart next to the textbook defaults. This is the first comprehension view (D-027); D-014 applies in full.

:func:`discovered_indicators` (registry entry) draws, for one window of a block (the ``LOOKBACK`` bars the
model reads, ending at its decision bar):

* the block's close (one point per window) with the chosen window marked;
* a grid, one row per family (moving average, Bollinger bands, RSI, MACD) and one column per copy
  (#0 / #1 / #2): the learned indicator (solid) against the same indicator at its configured textbook period
  (dashed; ``MA_SPANS``, ``MACD_SETTINGS``, ``RSI_PERIODS``, ``BB_PERIODS``), on price for the moving averages
  and Bollinger bands, in their own panels for RSI and MACD;
* how each base period moved over training (metrics.jsonl, from the recorded start), against its textbook
  value, and, in a strip right of the last epoch, how the period the model applies varies per window: 5-95%
  and the middle 50% of the block's windows, their median, this window's period and the served base period;
* a table of learned against textbook periods and their change (:func:`discovered_table`, which the notebook
  also shows as a DataFrame).

*Learned* is the period the served model applied to this window: the base logit plus
``0.5 * tanh(meta_adjust)`` of the window, turned into bars
(:func:`~neural_trade.visualization.indicator_evolution.applied_periods`; ``learned="base"`` draws the base
periods instead, meta_adjust = 0, what metrics.jsonl logs). *Textbook* is the configured period the copy
starts from. Colour = copy, as in the period figures of :mod:`indicator_evolution` (horizon colours are for
horizons only); solid = learned, dashed = textbook; dotted is not used (it means training everywhere).

The indicator maths is the model's own (``models/layers/learnable_indicators.py``), in :func:`indicator_lines`:

* EWMA (``utils/math.ewma_sequence``; ``ewma_sequence_matrix`` is the same recurrence unrolled, alpha clamped
  to [1e-6, 1 - 1e-6]): ``alpha = 2 / (period + 1)``, ``ema[0] = x[0]``,
  ``ema[t] = alpha * x[t] + (1 - alpha) * ema[t - 1]``. It starts at the window's first bar, because the layer
  sees only the window: at the window's last bar the seed close still weighs ``(1 - alpha) ** (LOOKBACK - 1)``
  (14% for a 60-bar period in a 60-bar window).
* moving average: ``ema(close, p)``.
* MACD: ``line = ema(close, fast) - ema(close, slow)``, ``signal = ema(line, signal)``,
  ``histogram = line - signal`` (the layer also reads ``tanh(10 * histogram)`` of its input).
* RSI: gains and losses are ``max(+-diff(close), 0)`` with a leading 0;
  ``RSI = 100 - 100 / (1 + ema(gains, p) / (ema(losses, p) + 1e-8))``. The smoothing is the same EWMA
  (alpha = 2 / (p + 1)), not Wilder's 1 / p, for the learned and the textbook line alike.
* Bollinger: ``mid = ema(close, p)``, ``var = ema((close - mid) ** 2, p)``, ``std = sqrt(var + 1e-8)``,
  ``upper / lower = mid +- 2 std``, ``%B = (close - lower) / (4 std + 1e-8)``.

The lines are drawn on the raw close. The layer computes the same formulas on the window-relative input
``(close - last close) / scale`` (``data/scaling.WindowNormalizer``); the weights of an EWMA sum to 1, so the
moving average and the Bollinger lines map back exactly (``value * scale + last close``), MACD by ``* scale``,
and RSI and %B are unchanged, apart from the 1e-8 guards. :func:`layer_channels` gives the layer's 31
channels in its own order; tests/test_viz_indicators.py pins both against the layer. (The legacy
``per_lag_standard`` normaliser has no such mapping: for such a run the lines are the model's formulas on
the raw close, not its channels.)
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from neural_trade.core.indicator_periods import MACD_ROLES, configured_periods
from neural_trade.visualization import theme as T
from neural_trade.visualization.indicator_evolution import (
    COPY_COLORS, SERVED_LINE, STRIP_FILL, _applied_frame, _config_for, _epoch_ticks, _epochs, _find_start, _frame,
    _log_range, _log_ticks, _num, _order_key, _parse, _path_of, _period_cols, _served_epoch, clip_bounds, label,
)

__all__ = ["discovered_indicators", "discovered_table", "ewma", "indicator_lines", "layer_channels", "pick_window"]

LEARNED_DASH = T.VAL_DASH          # solid: the learned indicator
# The same indicator at its textbook period is a reference: dashed, with an explicit pattern (plotly's "dash"
# draws as one block in a 30 px legend key, analytics_confidence.REF_DASH). Dotted means training everywhere.
TEXTBOOK_DASH = "6px,4px"
FAMILIES = ("ma", "bb", "rsi", "macd")          # the rows of the price grid, top to bottom
PREFIX = {"ma": "ma_period_", "macd": "macd_", "rsi": "rsi_period_", "bb": "bb_period_"}
FAMILY_OF = {v: k for k, v in PREFIX.items()}
FAMILY_NAME = {"ma": "Moving average", "bb": "Bollinger", "rsi": "RSI", "macd": "MACD"}
# the period panels (family, MACD role), in reading order: one panel per learned quantity
PERIOD_GROUPS = (("ma", None), ("rsi", None), ("bb", None), ("macd", "fast"), ("macd", "slow"), ("macd", "signal"))
WINDOW_PICKS = ("last", "typical", "longest", "shortest")
RSI_LEVELS = (30.0, 70.0)
SIGNAL_COLOR = T.INK_2                          # the MACD signal line (the MACD line takes the copy colour)
CLOSE_LINE = dict(color=T.INK_2, width=1.2)
THIS_WINDOW = dict(symbol="x", size=9, color=T.INK, line=dict(color=T.PAPER, width=1))
BASE_MARK = dict(symbol="circle-open", size=9, color=T.INK_2, line=dict(width=1.5))
MEDIAN_MARK = dict(symbol="diamond", size=10, line=dict(color=T.PAPER, width=1))
BOUND_LINE = dict(color=T.NEUTRAL, width=1, dash="5px,4px")    # the clip bounds of the base periods
TABLE_HEADING = ("<b>Learned against textbook periods</b> · bars; the change against the textbook period in "
                 "brackets")

_ALPHA_CLAMP = 1e-6                             # = utils.math._EWMA_ALPHA_CLAMP (that module imports TensorFlow)
_EPS = 1e-8                                     # the layer's guards (RSI, Bollinger)
_ROW_PX = {"overview": 96, "ma": 190, "bb": 210, "rsi": 140, "macd": 165, "period": 172}
# between rows: the tick labels and x title of the panel above, then the next panel's two-line heading (its
# title above its keys, so a heading fits one panel of a 1100 px output)
_GAP_PX = 86
_TABLE_HEADER_PX, _TABLE_ROW_PX = 40, 21        # the header has room for two lines (it wraps at 1100 px)
_MARGIN_T, _MARGIN_B = 232, 40
_TOP_LEGEND_PX = 46                             # the figure-wide legend ends this far above the first heading's panel
_MACD_LINES = ("line", "signal", "hist")        # the MACD channels, in the layer's order


# ---------------------------------------------------------------------------------- the model's indicator maths
def _alpha(period) -> np.ndarray:
    return np.clip(2.0 / (np.asarray(period, dtype=float) + 1.0), _ALPHA_CLAMP, 1.0 - _ALPHA_CLAMP)


def ewma(x, period) -> np.ndarray:
    """The layer's EWMA along the last axis: ``alpha = 2 / (period + 1)``, ``ema[0] = x[0]``,
    ``ema[t] = alpha * x[t] + (1 - alpha) * ema[t - 1]``. ``period``: a number, or one per series
    (broadcast to ``x.shape[:-1]``, e.g. the period each window was given)."""
    x = np.asarray(x, dtype=float)
    a = np.broadcast_to(_alpha(period), x.shape[:-1])
    out = np.empty_like(x)
    out[..., 0] = x[..., 0]
    for t in range(1, x.shape[-1]):
        out[..., t] = a * x[..., t] + (1.0 - a) * out[..., t - 1]
    return out


def _instances(names) -> Dict[str, List[int]]:
    """{family: [copy index, ...]} for the period names present (MACD counts once per copy)."""
    out: Dict[str, List[int]] = {f: [] for f in FAMILIES}
    for name in names:
        prefix, idx, _ = _parse(str(name))
        fam = FAMILY_OF.get(prefix)
        if fam is not None and idx not in out[fam]:
            out[fam].append(idx)
    return {f: sorted(v) for f, v in out.items()}


def _names_of(fam: str, idx: int) -> List[str]:
    return [f"macd_{idx}_{r}" for r in MACD_ROLES] if fam == "macd" else [f"{PREFIX[fam]}{idx}"]


def indicator_lines(close, periods) -> Dict[str, Dict[str, np.ndarray]]:
    """Every indicator of ``periods`` on the close window(s) ``close`` (``[..., T]``), by the layer's formulas.

    ``periods``: {metrics name: period in bars} (``ma_period_0``, ``macd_0_fast`` / ``_slow`` / ``_signal``,
    ``rsi_period_0``, ``bb_period_0`` ...; a Series row of :func:`applied_periods` works), each a number or one
    value per window. Returns ``{"ma_0": {"ma"}, "macd_0": {"line", "signal", "hist"}, "rsi_0": {"rsi"},
    "bb_0": {"mid", "upper", "lower", "std", "pct_b"}, ...}``; a MACD copy needs all three of its periods."""
    x = np.asarray(close, dtype=float)
    per = {str(k): np.asarray(v, dtype=float) for k, v in dict(periods).items() if v is not None}
    diffs = np.diff(x, axis=-1)
    zero = np.zeros(x.shape[:-1] + (1,))
    gains = np.concatenate([zero, np.where(diffs > 0, diffs, 0.0)], axis=-1)
    losses = np.concatenate([zero, np.where(diffs < 0, -diffs, 0.0)], axis=-1)
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for fam, idxs in _instances(per).items():
        for i in idxs:
            names = _names_of(fam, i)
            if not all(n in per and np.all(np.isfinite(per[n])) for n in names):
                continue
            p = [per[n] for n in names]
            if fam == "ma":
                lines = {"ma": ewma(x, p[0])}
            elif fam == "macd":
                line = ewma(x, p[0]) - ewma(x, p[1])
                sig = ewma(line, p[2])
                lines = {"line": line, "signal": sig, "hist": line - sig}
            elif fam == "rsi":
                lines = {"rsi": 100.0 - 100.0 / (1.0 + ewma(gains, p[0]) / (ewma(losses, p[0]) + _EPS))}
            else:
                mid = ewma(x, p[0])
                std = np.sqrt(ewma((x - mid) ** 2, p[0]) + _EPS)
                lines = {"mid": mid, "upper": mid + 2.0 * std, "lower": mid - 2.0 * std, "std": std,
                         "pct_b": (x - (mid - 2.0 * std)) / (4.0 * std + _EPS)}
            out[f"{fam}_{i}"] = lines
    return out


def layer_channels(close, periods) -> np.ndarray:
    """The indicator layer's output channels for ``close`` (``[..., T]``, in the units the layer is given) at
    ``periods``: ``[..., T, C]`` in the layer's order - every MA copy; per MACD copy line, signal, histogram,
    ``tanh(10 * histogram)``; every RSI copy; per Bollinger copy mid, upper, lower, %B; the input itself
    (31 channels with the default 3 copies per family)."""
    x = np.asarray(close, dtype=float)
    lines = indicator_lines(x, periods)
    inst = _instances(periods)
    chans = [lines[f"ma_{i}"]["ma"] for i in inst["ma"]]
    for i in inst["macd"]:
        m = lines[f"macd_{i}"]
        chans += [m[k] for k in _MACD_LINES] + [np.tanh(10.0 * m["hist"])]
    chans += [lines[f"rsi_{i}"]["rsi"] for i in inst["rsi"]]
    for i in inst["bb"]:
        b = lines[f"bb_{i}"]
        chans += [b["mid"], b["upper"], b["lower"], b["pct_b"]]
    return np.stack(chans + [x], axis=-1)


# ---------------------------------------------------------------------------------- which window
def pick_window(window, n: int, applied=None) -> int:
    """Index of the window to draw among ``n``: an int (negative counts from the end), ``"last"`` (or None),
    or, with the applied periods, ``"typical"`` (the window whose applied periods are closest to the block's
    medians, in log space), ``"longest"`` / ``"shortest"`` (the largest / smallest mean applied / base ratio:
    where meta_adjust stretches or shrinks the periods most)."""
    if n < 1:
        raise ValueError("no windows to pick from")
    if window is None or (isinstance(window, str) and window == "last"):
        return n - 1
    if isinstance(window, (int, np.integer)) and not isinstance(window, bool):
        w = int(window) + (n if int(window) < 0 else 0)
        if not 0 <= w < n:
            raise ValueError(f"window {window} is outside the block's {n:,} windows")
        return w
    if window in WINDOW_PICKS:
        app = _applied_frame(applied)
        if app is None:
            raise ValueError(f"window={window!r} needs the applied periods (applied=)")
        cols = [c for c in app.columns if _parse(str(c))[0]]
        logp = np.log(np.clip(app[cols].to_numpy(float), 1e-6, None))
        if window == "typical":
            return int(np.nanargmin(((logp - np.nanmedian(logp, axis=0)) ** 2).sum(axis=1)))
        base = app.attrs.get("base") or {}
        ref = (np.log([float(base.get(c, np.nan)) for c in cols]) if base
               else np.nanmedian(logp, axis=0))
        shift = np.nanmean(logp - ref, axis=1)
        return int(np.nanargmax(shift) if window == "longest" else np.nanargmin(shift))
    raise ValueError(f"window must be an index or one of {WINDOW_PICKS}, got {window!r}")


# ---------------------------------------------------------------------------------- what is drawn
@dataclass
class _View:
    """Everything the figure and the table read, gathered once so the two always agree."""

    config: object
    names: List[str]
    textbook: Dict[str, float]
    base: Dict[str, float]
    base_source: str
    app: Optional[pd.DataFrame]
    w: Optional[int]
    drawn: Dict[str, float]
    drawn_kind: str
    df: Optional[pd.DataFrame]
    epochs: np.ndarray
    init: Dict[str, float]
    source: Optional[str]
    served: Optional[float]
    block: str
    n: int
    lookback: Optional[int]
    notes: List[str] = field(default_factory=list)


def _status_epoch(path: Optional[Path]) -> Optional[float]:
    """The served epoch (1-based) from the run's status.json next to metrics.jsonl, or None."""
    if path is None:
        return None
    f = path.with_name("status.json")
    try:
        e = json.loads(f.read_text(encoding="utf-8")).get("weights_epoch")
        return float(e) if e is not None else None
    except (OSError, ValueError, AttributeError, TypeError):
        return None


def _gather(applied, config, metrics, window, start, learned, n_windows, block) -> _View:
    config = _config_for(metrics, config)
    textbook = configured_periods(config)
    app = _applied_frame(applied)
    df = _frame(metrics) if metrics is not None else None
    cols = sorted(_period_cols(df), key=_order_key) if df is not None else []
    x = _epochs(df) if df is not None else np.zeros(0)
    init, source = _find_start(_path_of(metrics), config, start)
    notes: List[str] = []
    served = None
    if df is not None and len(cols) and len(x):
        if app is not None and app.attrs.get("base"):
            served = _served_epoch(df, cols, x, app)       # the logged epoch whose periods are the served base
            if served is None:
                notes.append("the served base periods match no epoch of metrics.jsonl")
        else:
            served = _status_epoch(_path_of(metrics))
    base, base_source = {}, ""
    if app is not None and app.attrs.get("base"):
        base, base_source = {k: float(v) for k, v in app.attrs["base"].items()}, "served weights"
    elif df is not None and len(cols) and len(x):
        hit = np.where(x == served)[0] if served is not None else []
        row = int(hit[-1]) if len(hit) else len(x) - 1
        base = {c: float(_num(df[c])[row]) for c in cols}
        base_source = f"epoch {int(x[row])} of metrics.jsonl" + ("" if len(hit) else " (the last)")
    names = sorted({str(k) for k in (*textbook, *base, *(app.columns if app is not None else ()))
                    if _parse(str(k))[0] in FAMILY_OF}, key=_order_key)
    n = len(app) if app is not None else int(n_windows or 0)
    w = pick_window(window, n, app) if n else None
    if learned == "applied" and app is not None and w is not None:
        drawn = {c: float(app[c].iloc[w]) for c in app.columns if c in names}
        kind = "applied"
    else:
        drawn, kind = dict(base), "base"
        if learned == "applied" and base:
            notes.append("no applied periods given: the learned lines use the base periods")
    if not base:
        notes.append("no learned periods given (pass applied= or metrics=)")
    lookback = getattr(config, "LOOKBACK", None) if config is not None else None
    blk = block or (app.attrs.get("block") if app is not None else None) or "input"
    return _View(config, names, textbook, base, base_source, app, w, drawn, kind, df, x, init, source, served,
                 blk, n, lookback, notes)


def _pct(a, b) -> float:
    return 100.0 * (a / b - 1.0) if np.isfinite(a) and np.isfinite(b) and b else np.nan


def _table(v: _View) -> pd.DataFrame:
    lo_b, hi_b = clip_bounds(v.config)
    rows = {}
    for c in v.names:
        t, b = float(v.textbook.get(c, np.nan)), float(v.base.get(c, np.nan))
        row = {"indicator": label(c), "textbook": t, "learned (served base)": b, "base vs textbook %": _pct(b, t)}
        if v.app is not None and c in v.app.columns:
            a = v.app[c].to_numpy(float)
            a = a[np.isfinite(a)]
            p5, p50, p95 = np.percentile(a, [5, 50, 95]) if len(a) else (np.nan,) * 3
            row.update({"applied median": p50, "median vs textbook %": _pct(p50, t), "applied p5": p5,
                        "applied p95": p95, "this window": float(v.app[c].iloc[v.w]) if v.w is not None else np.nan})
            if v.lookback:
                row["applied > lookback, % of windows"] = 100.0 * float(np.mean(a > v.lookback)) if len(a) else np.nan
        if v.df is not None and c in v.df.columns:
            traj = _num(v.df[c])
            s0 = v.init.get(c)
            both = np.concatenate([[s0] if s0 is not None else [], traj]).astype(float)
            ok = both[np.isfinite(both)]
            row.update({"training min": float(ok.min()) if len(ok) else np.nan,
                        "training max": float(ok.max()) if len(ok) else np.nan})
        if lo_b and hi_b and np.isfinite(b) and b > 0:
            row["base near a clip bound"] = ("ceiling" if hi_b / b - 1 <= 0.05 else "floor" if b / lo_b - 1 <= 0.05
                                             else "")
        rows[c] = row
    out = pd.DataFrame.from_dict(rows, orient="index")
    if len(out):
        out.index.name = "period"
    what = (f"applied = the period each of the {v.n:,} {v.block} windows gets (served weights); "
            f"this window = #{v.w:,}" if v.app is not None and v.w is not None else "no applied periods given")
    out.attrs.update(block=v.block, n=v.n, window=v.w, served_epoch=v.served, base_source=v.base_source,
                     caption=f"Periods in bars. textbook = the configured start; learned (served base) = the "
                             f"trained logit with meta_adjust = 0 ({v.base_source or 'unknown'}); {what}.")
    return out


def discovered_table(applied=None, config=None, *, metrics=None, window="last", start=None) -> pd.DataFrame:
    """Learned against textbook periods, one row per learned period (index = the metrics name, model order).

    Columns: ``indicator``; ``textbook`` (the configured period); ``learned (served base)`` and
    ``base vs textbook %``; with ``applied`` (:func:`applied_periods` on a block's windows): ``applied median``
    and ``median vs textbook %``, ``applied p5`` / ``applied p95``, ``this window`` (the ``window`` of
    :func:`pick_window`) and ``applied > lookback, % of windows`` (an EWMA longer than the window has not
    warmed up at its end); with ``metrics``: ``training min`` / ``training max`` (start included); with the
    clip bounds of the config: ``base near a clip bound``. ``attrs``: block, n, window, served_epoch, caption.
    The same numbers as the table in :func:`discovered_indicators`."""
    return _table(_gather(applied, config, metrics, window, start, "applied", None, None))


# ---------------------------------------------------------------------------------- figure helpers
def _f32(a):
    return np.asarray(a, dtype=np.float32)


def _p(v) -> str:
    """A period in a key: a whole number as is (a textbook 10), else 3 significant digits with their zeros (a
    learned 10.0 is not the textbook 10; 4.07, 58.9), and no decimals from 100 bars up."""
    if v is None or not np.isfinite(v):
        return "?"
    v = float(v)
    if v == round(v) or v >= 100:
        return f"{v:.0f}"
    return f"{v:#.3g}"


def _periods_text(per: Dict[str, float], fam: str, idx: int) -> str:
    """'4.07' or, for MACD, fast/slow/signal '6.67/32.3/7.29'."""
    return "/".join(_p(per.get(n, np.nan)) for n in _names_of(fam, idx))


def _copy_color(idx: int) -> str:
    """Copy #0 / #1 / #2 take the period figures' colours; more copies take the next non-horizon slots."""
    return COPY_COLORS[idx] if idx < len(COPY_COLORS) else T.OTHER_SERIES[idx % len(T.OTHER_SERIES)]


def _key(fig, row, col, legend, name, group, **style):
    """A legend-only entry (drawn nowhere) on the panel at (row, col)."""
    import plotly.graph_objects as go

    fig.add_trace(go.Scatter(x=[None], y=[None], name=name, legend=legend, legendgroup=group, hoverinfo="skip",
                             **style), row, col)


def _time_text(t) -> str:
    try:
        return pd.Timestamp(t).strftime("%Y-%m-%d %H:%M")
    except (ValueError, TypeError):
        return str(t)


# ---------------------------------------------------------------------------------- the figure
def discovered_indicators(data, config=None, *, applied=None, metrics=None, window="last", learned="applied",
                          times=None, block: Optional[str] = None, start=None, height: Optional[int] = None,
                          title: Optional[str] = None, **_):
    """The learned indicators on the price of one window, each next to its textbook default (module docstring).

    ``data``: the block's raw close windows ``[N, LOOKBACK]`` as the model reads them (e.g.
    ``split_arrays(cfg)["test"]["X"]``), or one window. ``config``: the run's config (textbook periods,
    lookback, clip bounds; without it, the run's config.yaml next to ``metrics``). ``applied``:
    :func:`applied_periods` of the served model on the same windows (its ``attrs["base"]`` holds the served base
    periods). ``metrics``: the run's metrics.jsonl (the periods over training; its period_init.json and
    status.json are read when present). ``window``: see :func:`pick_window`. ``learned``: ``"applied"`` (the
    period the model applied to this window) or ``"base"``. ``times``: the decision-bar time of every window
    (axis labels and the subtitle only). ``block``: the block's name (default: the applied frame's).
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    X = np.asarray(data, dtype=float)
    if X.ndim == 1:
        X = X[None, :]
    n, L = X.shape
    app = _applied_frame(applied)
    if app is not None and len(app) != n:
        raise ValueError(f"applied has {len(app)} rows for {n} windows: compute it on the same windows")
    if learned not in ("applied", "base"):
        raise ValueError(f"learned must be 'applied' or 'base', got {learned!r}")
    v = _gather(app, config, metrics, window, start, learned, n, block)
    w = int(v.w if v.w is not None else n - 1)
    table = _table(v)
    inst = _instances(v.names)
    fams = [f for f in FAMILIES if inst[f]]
    groups = [g for g in PERIOD_GROUPS if inst[g[0]]]
    ncols = max([3] + [len(inst[f]) for f in fams])
    n_period_rows = math.ceil(len(groups) / ncols) if groups else 0

    rows = [("overview", None)] + [("price", f) for f in fams] + [("period", k) for k in range(n_period_rows)]
    rows.append(("table", None))
    table_px = _TABLE_HEADER_PX + _TABLE_ROW_PX * max(len(table), 1) + 24    # slack: a table taller than its
    # domain scrolls, which hides its last rows in a saved figure
    px = [(_ROW_PX["overview"] if kind == "overview" else _ROW_PX[arg] if kind == "price"
           else _ROW_PX["period"] if kind == "period" else table_px) for kind, arg in rows]
    plot_px = float(sum(px) + _GAP_PX * (len(rows) - 1))
    specs = []
    for kind, arg in rows:
        if kind in ("overview", "table"):
            specs.append([{"type": "table", "colspan": ncols} if kind == "table" else {"colspan": ncols}]
                         + [None] * (ncols - 1))
        elif kind == "price":
            k = len(inst[arg])
            specs.append([{}] * k + [None] * (ncols - k))
        else:
            k = len(groups[arg * ncols:(arg + 1) * ncols])
            specs.append([{}] * k + [None] * (ncols - k))
    fig = make_subplots(rows=len(rows), cols=ncols, specs=specs, row_heights=px, vertical_spacing=_GAP_PX / plot_px,
                        horizontal_spacing=0.055)
    legends: List[Tuple[str, int, int, str]] = []          # (legend id, row, col, heading)

    def new_legend(row, col, heading) -> str:
        lid = f"legend{len(legends) + 2}"
        legends.append((lid, row, col, heading))
        return lid

    # ---- row 1: the block's close with the window marked
    step = max(1, int(getattr(v.config, "WINDOW_STEP", 1) or 1)) if v.config is not None else 1
    close_blk = X[:, -1]
    lg = new_legend(1, 1, f"The {v.block} block: close at each window's decision bar ({n:,} windows)")
    fig.add_trace(go.Scatter(y=_f32(close_blk), x0=0, dx=1, mode="lines", name="close", legend=lg,
                             line=CLOSE_LINE, hovertemplate="window %{x:,}<br>close %{y:,.2f}<extra></extra>"), 1, 1)
    fig.add_vrect(x0=max(w - (L - 1) / step, -0.5), x1=w + 0.5, fillcolor=T.rgba(T.INK, 0.16), line_width=0,
                  layer="below", row=1, col=1)
    fig.add_trace(go.Scatter(x=[w], y=_f32([close_blk[w]]), mode="markers", name=f"this window (#{w:,}, shaded)",
                             legend=lg, marker=THIS_WINDOW,
                             hovertemplate=f"window {w:,}: its {L} bars end here<br>close %{{y:,.2f}}<extra></extra>"),
                  1, 1)
    xa = dict(range=[-0.5, n - 0.5], title_text=None)
    t_arr = None
    if times is not None and len(times) == n:
        t_arr = list(pd.to_datetime(pd.Series(list(times))))
        ticks = sorted(set(np.linspace(0, n - 1, min(n, 6)).round().astype(int).tolist()))
        xa.update(tickvals=ticks, ticktext=[_time_text(t_arr[i]) for i in ticks])
    else:
        xa.update(title_text="window in the block")
    fig.update_xaxes(row=1, col=1, **xa)
    fig.update_yaxes(title_text="close", row=1, col=1)

    # ---- the price grid: family rows, copy columns; learned solid against textbook dashed
    win = X[w]
    offs = np.arange(-(L - 1), 1, dtype=float)
    xs = dict(x0=float(-(L - 1)), dx=1.0)
    learned_lines = indicator_lines(win, v.drawn)
    textbook_lines = indicator_lines(win, v.textbook)
    base_txt = "base" if v.drawn_kind == "base" else "applied to this window"
    for r_off, fam in enumerate(fams):
        r = 2 + r_off
        for c_off, idx in enumerate(inst[fam]):
            c = 1 + c_off
            key, color = f"{fam}_{idx}", _copy_color(idx)
            lid = new_legend(r, c, f"{FAMILY_NAME[fam]} #{idx}")
            lp, tp = _periods_text(v.drawn, fam, idx), _periods_text(v.textbook, fam, idx)
            bp = _periods_text(v.base, fam, idx)
            what = f"{FAMILY_NAME[fam]} #{idx}" + (" (fast/slow/signal)" if fam == "macd" else "")
            l_hover = f"{what} learned: {lp} bars {base_txt} (served base {bp}; textbook {tp})"
            t_hover = f"{what} textbook: {tp} bars (the configured start)"
            L_ = learned_lines.get(key)
            T_ = textbook_lines.get(key)
            if fam in ("ma", "bb"):
                fig.add_trace(go.Scatter(y=_f32(win), **xs, mode="lines", name="close", showlegend=False,
                                         legend=lid, line=CLOSE_LINE,
                                         hovertemplate="close<br>bar %{x}: %{y:,.2f}<extra></extra>"), r, c)
            if fam == "ma":
                for lines, name, dash, width, hover in ((L_, f"learned {lp}", LEARNED_DASH, 2.0, l_hover),
                                                        (T_, f"textbook {tp}", TEXTBOOK_DASH, 1.6, t_hover)):
                    if lines is None:
                        continue
                    fig.add_trace(go.Scatter(y=_f32(lines["ma"]), **xs, mode="lines", name=name, legend=lid,
                                             line=dict(color=color, width=width, dash=dash),
                                             hovertemplate=f"{hover}<br>bar %{{x}}: %{{y:,.2f}}<extra></extra>"), r, c)
            elif fam == "bb":
                if L_ is not None:
                    fig.add_trace(go.Scatter(x=_f32(np.r_[offs, offs[::-1]]),
                                             y=_f32(np.r_[L_["upper"], L_["lower"][::-1]]), mode="lines",
                                             fill="toself", fillcolor=T.rgba(color, 0.12), line=dict(width=0),
                                             showlegend=False, legend=lid, legendgroup=f"{key}-l", hoverinfo="skip",
                                             name="learned band"), r, c)
                for lines, name, dash, grp, hover in ((L_, f"learned {lp}", LEARNED_DASH, f"{key}-l", l_hover),
                                                      (T_, f"textbook {tp}", TEXTBOOK_DASH, f"{key}-t", t_hover)):
                    if lines is None:
                        continue
                    for part in ("upper", "lower"):
                        fig.add_trace(go.Scatter(y=_f32(lines[part]), **xs, mode="lines", showlegend=False,
                                                 legend=lid, legendgroup=grp, name=f"{name} {part}",
                                                 line=dict(color=color, width=1.0, dash=dash),
                                                 hovertemplate=f"{hover}<br>bar %{{x}}: {part} band %{{y:,.2f}}"
                                                               f"<extra></extra>"), r, c)
                    fig.add_trace(go.Scatter(
                        y=_f32(lines["mid"]), **xs, mode="lines", name=name, legend=lid, legendgroup=grp,
                        line=dict(color=color, width=1.8, dash=dash),
                        customdata=np.stack([lines["upper"], lines["lower"], lines["pct_b"]], axis=-1).astype("float32"),
                        hovertemplate=f"{hover}<br>bar %{{x}}: mid %{{y:,.2f}} · upper %{{customdata[0]:,.2f}} · "
                                      f"lower %{{customdata[1]:,.2f}} · %B %{{customdata[2]:.2f}}<extra></extra>"), r, c)
            elif fam == "rsi":
                # exclude_empty_subplots=False: plotly skips a subplot that has no trace yet
                fig.add_hrect(y0=RSI_LEVELS[0], y1=RSI_LEVELS[1], fillcolor=T.rgba(T.NEUTRAL, 0.07), line_width=0,
                              layer="below", row=r, col=c, exclude_empty_subplots=False)
                for level in RSI_LEVELS:
                    fig.add_hline(y=level, line=dict(color=T.NEUTRAL, width=1), row=r, col=c,
                                  exclude_empty_subplots=False)
                for lines, name, dash, width, hover in ((L_, f"learned {lp}", LEARNED_DASH, 2.0, l_hover),
                                                        (T_, f"textbook {tp}", TEXTBOOK_DASH, 1.6, t_hover)):
                    if lines is None:
                        continue
                    fig.add_trace(go.Scatter(y=_f32(lines["rsi"]), **xs, mode="lines", name=name, legend=lid,
                                             line=dict(color=color, width=width, dash=dash),
                                             hovertemplate=f"{hover}<br>bar %{{x}}: RSI %{{y:.1f}}<extra></extra>"), r, c)
                fig.update_yaxes(range=[-3, 103], tickvals=[0, 30, 50, 70, 100], row=r, col=c)
            else:
                fig.add_hline(y=0, line=dict(color=T.NEUTRAL, width=1), row=r, col=c, exclude_empty_subplots=False)
                for lines, grp, hollow in ((L_, f"{key}-l", False), (T_, f"{key}-t", True)):
                    if lines is None:
                        continue
                    marker = (dict(color="rgba(0,0,0,0)", line=dict(color=T.rgba(T.INK_2, 0.7), width=1)) if hollow
                              else dict(color=T.rgba(color, 0.38), line=dict(width=0)))
                    fig.add_trace(go.Bar(y=_f32(lines["hist"]), **xs, marker=marker, showlegend=False, legend=lid,
                                         legendgroup=grp, name="histogram",
                                         hovertemplate=(t_hover if hollow else l_hover)
                                         + "<br>bar %{x}: histogram %{y:,.2f}<extra></extra>"), r, c)
                for lines, name, dash, grp, hover in ((L_, f"learned {lp}", LEARNED_DASH, f"{key}-l", l_hover),
                                                      (T_, f"textbook {tp}", TEXTBOOK_DASH, f"{key}-t", t_hover)):
                    if lines is None:
                        continue
                    fig.add_trace(go.Scatter(y=_f32(lines["line"]), **xs, mode="lines", name=name, legend=lid,
                                             legendgroup=grp, line=dict(color=color, width=2.0 if grp.endswith("l")
                                                                        else 1.6, dash=dash),
                                             hovertemplate=f"{hover}<br>bar %{{x}}: MACD %{{y:,.2f}}<extra></extra>"),
                                  r, c)
                    fig.add_trace(go.Scatter(y=_f32(lines["signal"]), **xs, mode="lines", name=f"{name} signal",
                                             showlegend=False, legend=lid, legendgroup=grp,
                                             line=dict(color=SIGNAL_COLOR, width=1.2, dash=dash),
                                             hovertemplate=f"{hover}<br>bar %{{x}}: signal %{{y:,.2f}}<extra></extra>"),
                                  r, c)
            fig.update_xaxes(range=[-(L - 1) - 0.8, 0.8], row=r, col=c,
                             title_text=("bars before the decision bar (0 = the window's last bar)"
                                         if fam == fams[-1] else None), title_standoff=4)
            if c == 1:
                fig.update_yaxes(title_text={"ma": "close", "bb": "close", "rsi": "RSI",
                                             "macd": "MACD (price units)"}[fam], row=r, col=c)

    # ---- the periods over training, and per window (the strip right of the last epoch)
    epochs = v.epochs
    last_x = float(epochs[-1]) if len(epochs) else 0.0
    u = max(1.0, last_x / 20.0)
    has0 = bool(v.init)
    x_lo = -0.6 * u if has0 or not len(epochs) else 1 - 0.6 * u
    max_copies = max([len(inst[g[0]]) for g in groups] + [1])
    strip_x = {k: last_x + u * (1.0 + 0.75 * k) for k in range(max_copies)}
    strip_lo, strip_hi = last_x + 0.45 * u, last_x + u * (1.0 + 0.75 * (max_copies - 1) + 0.55)
    x_hi = strip_hi if v.app is not None else last_x + 0.6 * u
    lo_b, hi_b = clip_bounds(v.config)
    period_row0 = 2 + len(fams)
    used_bound = False
    for gi, (fam, role) in enumerate(groups):
        r, c = period_row0 + gi // ncols, 1 + gi % ncols
        heading = (f"MACD {role} periods" if fam == "macd" else f"{FAMILY_NAME[fam]} periods") + " over training"
        lid = new_legend(r, c, heading)
        span = []
        for idx in inst[fam]:
            name = _names_of(fam, idx)[MACD_ROLES.index(role)] if fam == "macd" else _names_of(fam, idx)[0]
            color = _copy_color(idx)
            tb = v.textbook.get(name)
            has_traj = v.df is not None and name in v.df.columns and len(epochs) > 0
            if has_traj:
                vals = _num(v.df[name])
                s0 = v.init.get(name)
                ok0 = s0 is not None and np.isfinite(s0) and s0 > 0
                xs_e = ([0.0] if ok0 else []) + [float(e) for e in epochs]
                ys_e = np.concatenate([[s0] if ok0 else [], vals]).astype("float32")
                span += ys_e[np.isfinite(ys_e) & (ys_e > 0)].tolist()
                text = (["start"] if ok0 else []) + [f"epoch {int(e)}" for e in epochs]
                fig.add_trace(go.Scatter(
                    x=xs_e, y=T.positive(ys_e), mode="lines+markers", name=f"#{idx}", legend=lid,
                    legendgroup=f"{name}-traj", line=dict(color=color, width=2, dash=LEARNED_DASH),
                    marker=dict(symbol=(["circle-open"] if ok0 else []) + ["circle"] * len(epochs),
                                size=([9] if ok0 else []) + [4 if len(epochs) <= 40 else 0] * len(epochs), color=color),
                    text=text, hovertemplate=f"{label(name)} base period<br>%{{text}}: %{{y:.2f}} bars<extra></extra>"),
                    r, c)
            if tb is not None and np.isfinite(tb):
                span.append(float(tb))
                # the copy's key is its trajectory; without one (no metrics) the textbook line carries it, so the
                # panel heading (a legend) is still drawn
                fig.add_trace(go.Scatter(x=[x_lo, x_hi], y=[tb, tb], mode="lines", name=f"#{idx} textbook {tb:g}",
                                         showlegend=not has_traj, legend=lid, legendgroup=f"{name}-traj",
                                         line=dict(color=color, width=1.2, dash=TEXTBOOK_DASH),
                                         hovertemplate=f"{label(name)} textbook period {tb:g} bars<extra></extra>"),
                              r, c)
            if v.app is not None and name in v.app.columns:
                a = v.app[name].to_numpy(float)
                a = a[np.isfinite(a) & (a > 0)]
                if len(a):
                    q5, q25, q50, q75, q95 = np.percentile(a, [5, 25, 50, 75, 95])
                    xk = strip_x[inst[fam].index(idx)]
                    span += [q5, q95]
                    for lo_q, hi_q, width in ((q5, q95, 2), (q25, q75, 7)):
                        fig.add_trace(go.Scatter(x=[xk, xk], y=_f32([lo_q, hi_q]), mode="lines", showlegend=False,
                                                 legend=lid, legendgroup=f"{name}-traj", name="applied range",
                                                 line=dict(color=color, width=width), hoverinfo="skip"), r, c)
                    this = float(v.app[name].iloc[w])
                    b = v.base.get(name, np.nan)
                    span += [this] + ([b] if np.isfinite(b) and b > 0 else [])
                    txt = (f"{label(name)} over {len(a):,} {v.block} windows: median {q50:.2f} bars, middle 50% "
                           f"{q25:.1f}-{q75:.1f}, 5-95% {q5:.1f}-{q95:.1f}<br>this window (#{w:,}) {this:.2f} · "
                           f"served base {b:.2f} · textbook {_p(tb)}")
                    fig.add_trace(go.Scatter(x=[xk], y=_f32([q50]), mode="markers", showlegend=False, legend=lid,
                                             legendgroup=f"{name}-traj", name="applied median",
                                             marker=dict(color=color, **MEDIAN_MARK), text=[txt],
                                             hovertemplate="%{text}<extra></extra>"), r, c)
                    if np.isfinite(b) and b > 0:
                        fig.add_trace(go.Scatter(x=[xk], y=_f32([b]), mode="markers", showlegend=False, legend=lid,
                                                 legendgroup=f"{name}-traj", name="served base", marker=BASE_MARK,
                                                 text=[txt], hovertemplate="%{text}<extra></extra>"), r, c)
                    fig.add_trace(go.Scatter(x=[xk], y=_f32([this]), mode="markers", showlegend=False, legend=lid,
                                             legendgroup=f"{name}-traj", name="this window", marker=THIS_WINDOW,
                                             text=[txt], hovertemplate="%{text}<extra></extra>"), r, c)
        if v.app is not None:
            fig.add_vrect(x0=strip_lo, x1=strip_hi, fillcolor=STRIP_FILL, line_width=0, layer="below", row=r, col=c)
        if v.served is not None and len(epochs):
            fig.add_vline(x=v.served, line=SERVED_LINE, row=r, col=c)
        # the clip bounds of the BASE periods (applied periods are not clipped): drawn when a base period nears one
        room = {"ceiling": [], "floor": []}
        base_vals = [v.base.get(nm) for nm in v.names if _parse(nm)[0] == PREFIX[fam]
                     and (role is None or _parse(nm)[2] == role)]
        base_vals = [b for b in base_vals if b is not None and np.isfinite(b) and b > 0]
        for bound, near, word in ((hi_b, bool(hi_b) and bool(base_vals) and max(base_vals) >= 0.8 * hi_b, "ceiling"),
                                  (lo_b, bool(lo_b) and bool(base_vals) and min(base_vals) <= 1.5 * lo_b, "floor")):
            if not near:
                continue
            used_bound = True
            room[word].append((bound, 18.0))
            span.append(bound)
            fig.add_trace(go.Scatter(x=[x_lo, last_x + 0.3 * u], y=[bound, bound], mode="lines", showlegend=False,
                                     legend=lid, name=f"clip {word}", hoverinfo="skip", line=BOUND_LINE), r, c)
            sp = fig.get_subplot(r, c)
            fig.add_annotation(x=last_x, y=math.log10(bound), xref=sp.xaxis.plotly_name.replace("axis", ""),
                               yref=sp.yaxis.plotly_name.replace("axis", ""), showarrow=False, xanchor="right",
                               yanchor="bottom" if word == "ceiling" else "top", yshift=2 if word == "ceiling" else -2,
                               text=f"clip {word} {bound:g}" + (" = lookback" if v.lookback and bound == v.lookback
                                                                else ""),
                               font=dict(size=10, color=T.MUTED), bgcolor=T.SURFACE, borderpad=1)
        panel_px = _ROW_PX["period"]
        lo_v, hi_v = (min(span), max(span)) if span else (1.0, 10.0)
        lo_l, hi_l = _log_range(lo_v, hi_v, panel_px, room["ceiling"], room["floor"])
        ticks = _log_ticks(10 ** lo_l, 10 ** hi_l, panel_px)
        fig.update_yaxes(type="log", range=[lo_l, hi_l], tickvals=ticks, ticktext=[f"{t:g}" for t in ticks],
                         title_text="period, bars (log)" if c == 1 else None, row=r, col=c)
        tv, tt = _epoch_ticks(int(last_x), has0) if len(epochs) else ([], [])
        if v.app is not None:        # the strip's label takes the place of the epoch ticks next to it
            keep = [(a, b) for a, b in zip(tv, tt) if a <= last_x - 1.2 * u or a == 0]
            tv, tt = [a for a, _ in keep] + [(strip_lo + strip_hi) / 2], [b for _, b in keep] + ["windows"]
        fig.update_xaxes(range=[x_lo, x_hi], tickvals=tv, ticktext=tt, title_text="epoch", title_standoff=4,
                         row=r, col=c)

    # ---- the table: learned against textbook
    fig.add_trace(_table_trace(go, table, v), len(rows), 1)

    # ---- figure-wide keys (legend-only, parked on the block panel's axes)
    learned_key = ("learned: the period the served model applied to this window" if v.drawn_kind == "applied"
                   else "learned: the base period (meta_adjust = 0)")
    _key(fig, 1, 1, "legend", learned_key, "k-learned", mode="lines", line=dict(color=T.INK_2, width=2,
                                                                                  dash=LEARNED_DASH))
    _key(fig, 1, 1, "legend", "textbook: the configured period", "k-textbook", mode="lines",
         line=dict(color=T.INK_2, width=2, dash=TEXTBOOK_DASH))
    if "macd" in fams:
        _key(fig, 1, 1, "legend", "MACD signal line", "k-signal", mode="lines", line=dict(color=SIGNAL_COLOR, width=1.2))
        _key(fig, 1, 1, "legend", "MACD histogram (filled = learned, hollow = textbook)", "k-hist", mode="markers",
             marker=dict(symbol="square", size=11, color=T.rgba(T.INK_2, 0.38), line=dict(color=T.INK_2, width=1)))
    if "rsi" in fams:
        _key(fig, 1, 1, "legend", "RSI 30 / 70", "k-rsi", mode="lines", line=dict(color=T.NEUTRAL, width=1))
    if v.served is not None and len(epochs):
        _key(fig, 1, 1, "legend", f"served weights (epoch {int(v.served)})", "k-served", mode="lines", line=SERVED_LINE)
    if v.app is not None:
        _key(fig, 1, 1, "legend", f"per window ({n:,} {v.block} windows): 5-95%", "k-p90", mode="lines",
             line=dict(color=T.INK_2, width=2))
        _key(fig, 1, 1, "legend", "middle 50%", "k-iqr", mode="lines", line=dict(color=T.INK_2, width=7))
        _key(fig, 1, 1, "legend", "median", "k-med", mode="markers", marker=dict(color=T.INK_2, **MEDIAN_MARK))
        _key(fig, 1, 1, "legend", "this window", "k-this", mode="markers", marker=THIS_WINDOW)
        _key(fig, 1, 1, "legend", "served base period", "k-base", mode="markers", marker=BASE_MARK)
    if used_bound:
        _key(fig, 1, 1, "legend", "clip bound (base periods)", "k-bound", mode="lines", line=BOUND_LINE)

    _empty_notes(fig, v)
    fig_height = int(height or plot_px + _MARGIN_T + _MARGIN_B)
    T.apply(fig, title=title or "Discovered indicators: learned against the textbook periods", height=fig_height,
            subtitle=_subtitle(v, n, L, w, t_arr))
    for lid, r, c, heading in legends:
        T.panel_legend(fig, lid, r, c, heading)
        fig.update_layout({lid: dict(title=dict(side="top"))})            # the title above its keys
    sp = fig.get_subplot(len(rows), 1)                        # the table's domain (a SubplotDomain: x, y)
    fig.add_annotation(x=sp.x[0], y=sp.y[1] + 0.002, xref="paper", yref="paper", xanchor="left",
                       yanchor="bottom", showarrow=False, font=dict(size=12, color=T.INK_2), text=TABLE_HEADING)
    plot_h = max(fig_height - _MARGIN_T - _MARGIN_B, 100)
    fig.update_layout(margin=dict(t=_MARGIN_T, b=_MARGIN_B, l=70, r=24), barmode="overlay", bargap=0.1,
                      legend=dict(y=1 + _TOP_LEGEND_PX / plot_h, yanchor="bottom", x=0, xanchor="left",
                                  itemsizing="trace", itemwidth=40))
    return fig


def _table_trace(go, table: pd.DataFrame, v: _View):
    def num(x, fmt=".2f"):
        return format(x, fmt) if x is not None and np.isfinite(x) else "n/a"

    def chg(x):
        return f"{x:+.1f}%" if x is not None and np.isfinite(x) else "n/a"

    def col(name):
        return table[name] if name in table.columns else pd.Series(np.nan, index=table.index)

    # short labels: none wraps to more than two lines in a 1100 px output (the header row is two lines high)
    header = ["indicator", "textbook", "learned: served base", "median applied", "applied 5-95%",
              f"this window #{v.w:,}" if v.w is not None else "this window", "training min-max",
              "applied > lookback", "base near a clip bound"]
    cells = [
        table["indicator"].tolist() if "indicator" in table.columns else [],
        [num(x, "g") for x in col("textbook")],
        [f"{num(b)} ({chg(d)})" for b, d in zip(col("learned (served base)"), col("base vs textbook %"))],
        [f"{num(m)} ({chg(d)})" for m, d in zip(col("applied median"), col("median vs textbook %"))],
        [f"{num(a, '.1f')}-{num(b, '.1f')}" for a, b in zip(col("applied p5"), col("applied p95"))],
        [num(x) for x in col("this window")],
        [f"{num(a, '.1f')}-{num(b, '.1f')}" for a, b in zip(col("training min"), col("training max"))],
        [f"{x:.0f}%" if np.isfinite(x) else "n/a" for x in col("applied > lookback, % of windows")],
        [str(x) if isinstance(x, str) and x else "" for x in (table["base near a clip bound"] if "base near a clip bound"
                                                              in table.columns else [""] * len(table))],
    ]
    colors = [[_copy_color(_parse(c)[1]) for c in table.index]] + [[T.INK] * len(table)] * (len(header) - 1)
    return go.Table(
        columnwidth=[1.3, 0.8, 1.5, 1.5, 1.1, 1.1, 1.2, 1.3, 1.2],
        header=dict(values=header, fill_color=T.SURFACE, line_color=T.GRID, height=_TABLE_HEADER_PX,
                    align=["left"] + ["right"] * (len(header) - 1), font=dict(size=12, weight="bold", color=T.INK)),
        cells=dict(values=cells, fill_color=T.PAPER, line_color=T.GRID, height=_TABLE_ROW_PX,
                   align=["left"] + ["right"] * (len(header) - 1), font=dict(size=11.5, color=colors)))


def _empty_notes(fig, v: _View):
    """A note in the middle of any panel left without a trace (a family missing from this run)."""
    for yref in T.empty_panels(fig):
        ax = fig.layout["yaxis" + yref[1:]]
        xd = fig.layout["xaxis" + (ax.anchor or "x")[1:]].domain or (0, 1)
        yd = ax.domain or (0, 1)
        fig.add_annotation(x=(xd[0] + xd[1]) / 2, y=(yd[0] + yd[1]) / 2, xref="paper", yref="paper",
                           showarrow=False, text="; ".join(v.notes) or "not in this run",
                           font=dict(color=T.MUTED, size=12))


def _subtitle(v: _View, n: int, L: int, w: int, t_arr) -> str:
    """Four lines, each short enough for a 1100 px output."""
    end = f" ending {_time_text(t_arr[w])}" if t_arr is not None else ""
    served = f"served weights = epoch {int(v.served)}" if v.served is not None else "served epoch unknown"
    line1 = f"{v.block} block, window #{w:,} of {n:,}: its {L} bars{end} · {served}"
    line2 = ("solid = learned: " + ("the period the served model applied to this window (base logit + "
                                    "0.5·tanh(meta_adjust))" if v.drawn_kind == "applied"
                                    else "the base period (meta_adjust = 0)")
             + " · dashed = textbook: the configured period")
    line3 = ("colour = copy #0 / #1 / #2 · every line starts at the window's first bar, as the model computes it "
             "(the EWMA is seeded with the first close)")
    line4 = ("periods in bars, log scale · the strip right of the last epoch: the periods the model applies to each "
             "of the block's windows" if v.app is not None else "periods in bars, log scale")
    if v.notes:
        line4 += " · " + "; ".join(v.notes)
    return "<br>".join((line1, line2, line3, line4))
