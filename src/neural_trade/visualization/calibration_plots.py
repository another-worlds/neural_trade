"""Calibration figures: direction reliability and conformal-interval coverage over time.

Both figures show how much of what they draw is noise. Consecutive 1-minute samples are not
independent: the outcome of a horizon of h bars overlaps its neighbour's by h - 1 bars. So:

* a reliability bin's up-rate gets a Bartlett (HAC) 95% band with lag = the horizon in bars
  (``horizon_steps``). The members of a bin are spread out in time (a bin is a band of P(up),
  not a stretch of the series), so the band is only ~1.3-2x wider than the binomial one on the
  BTC test block - far less than the sqrt(h) that N / h effective samples would give;
* the rolling coverage gets the 95% range that one window shows by chance when coverage is exactly
  on target, from the long-run variance of the inside/outside indicator (autocorrelations up to
  twice the horizon). That variance is estimated on the block being plotted (the test block in the
  calibration explorer) unless ``inflation=`` supplies one from another block; the subtitle names
  the block it came from (``inflation_from``). The band is centred on the target, not on the data,
  so the estimate sets only its width.

Inputs must be in time order (the deadband mask keeps the order).
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from neural_trade.visualization import stats as S
from neural_trade.visualization import theme as T
from neural_trade.visualization.theme import apply

RAW_DASH = "dash"           # raw P(up): the horizon colour, dashed, open markers (dotted means training)
REF_DASH = "5px,4px"        # reference lines (NEUTRAL): target, this block's up-rate (as in the direction figure)
REF2_DASH = "dashdot"       # a second reference (NEUTRAL): the calibration block's up-rate. Its swatch
                            # must differ from solid ("perfect") at legend length, which longdashdot does not


# ------------------------------------------------------------------ statistics
def _hac_sum_var(z: np.ndarray, lag: int) -> float:
    """Bartlett long-run variance of ``sum(z)`` for a centred series ``z`` (lags 1..``lag``)."""
    v = float(z @ z)
    for k in range(1, min(int(lag), len(z) - 1) + 1):
        v += 2.0 * (1.0 - k / (lag + 1.0)) * float(z[:-k] @ z[k:])
    return v


def _binned_rate(x, y, n_bins: int, lag: int = 0) -> np.ndarray:
    """Equal-count bins of ``x``: (mean x, mean y, count, 95% low, 95% high) of the 0/1 series ``y``.

    ``lag`` = 0: binomial band (samples independent). ``lag`` > 0: Bartlett HAC band of the bin
    mean over the time-ordered series (members of a bin that sit close in time share outcomes);
    never narrower than the binomial band.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    order = np.argsort(x, kind="mergesort")
    rows = []
    for idx in np.array_split(order, n_bins):
        if len(idx) == 0:
            continue
        p, m = x[idx].mean(), y[idx].mean()
        if lag and lag > 0:
            z = np.zeros(len(y))
            z[idx] = y[idx] - m
            v = max(_hac_sum_var(z, lag), float(z @ z), 1e-12)
            half = 1.96 * np.sqrt(v) / len(idx)
        else:
            half = 1.96 * np.sqrt(max(m * (1 - m), 1e-12) / len(idx))
        rows.append((p, m, len(idx), m - half, m + half))
    return np.array(rows)


def reliability_table(labels, probs, n_bins: int = 10, *, lag: int = 0):
    """Equal-count bins of predicted P(up): mean prediction, observed up-rate, count, 95% band.

    ``lag``: the horizon in bars for time-ordered labels whose outcomes overlap (HAC band);
    0 treats the samples as independent (the binomial band).
    """
    return _binned_rate(probs, labels, n_bins, lag)


def coverage_inflation(inside, horizon_steps: Optional[int] = None, *, window: Optional[int] = None) -> float:
    """How many times larger the variance of a ``window``-sample mean of the inside/outside
    indicator is than for independent samples: 1 + 2 sum_k (1 - k/window) rho_k over lags
    k = 1 .. 2 x ``horizon_steps`` (30 when unknown); at least 1. ``window / inflation`` is the
    effective number of samples in one window."""
    x = np.asarray(inside, float)
    x = x[np.isfinite(x)]
    lags = int(2 * horizon_steps) if horizon_steps else 30
    x = x - x.mean() if len(x) else x
    den = float(x @ x)
    if den <= 0 or len(x) <= lags + 1:
        return 1.0
    w = float(window) if window else float(len(x))
    s = sum(max(0.0, 1.0 - k / w) * float(x[:-k] @ x[k:]) / den for k in range(1, lags + 1))
    return float(max(1.0, 1.0 + 2.0 * s))


def _auc(labels, scores) -> float:
    from neural_trade.visualization.analytics_common import roc_curve

    labels = np.asarray(labels, float)
    if labels.min() == labels.max():
        return float("nan")
    return roc_curve(labels, scores)[2]


# ------------------------------------------------------------------ reliability
def reliability_figure(labels, p_raw, p_cal=None, *, n_bins: int = 10, title: Optional[str] = None,
                       horizon: Optional[str] = None, horizon_steps: Optional[int] = None,
                       ref_rate: Optional[float] = None, ref_label: str = "cal-block up-rate",
                       p_saved=None, saved_label: str = "saved", subtitle: Optional[str] = None,
                       note: Optional[str] = None, block: str = "test", height: Optional[int] = None):
    """Reliability diagram (equal-count bins) of the raw and, if given, calibrated P(up), with the
    distribution of the predictions underneath.

    ``labels``/``p_*`` are the samples outside the deadband, in time order. ``horizon`` colours the
    curves (raw dashed, calibrated solid); ``horizon_steps`` widens the bands for overlapping outcomes
    (HAC, lag = the horizon in bars). Grey reference lines: the diagonal (solid), the observed up-rate
    of this block (dashed) and, with ``ref_rate``, of the block the calibration was fitted on
    (dash-dot). ``p_saved`` adds a third curve
    (e.g. the run's saved pipeline next to a refit). Axes are zoomed to the bins and their bands;
    the subtitle says how many predictions lie outside.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from neural_trade.metrics.numpy_metrics import ece_pos

    labels = np.asarray(labels, float)
    lag = int(horizon_steps or 0)
    c = T.HORIZON_COLORS.get(horizon, T.INK_2)
    series = [(name, np.asarray(p, float), dash, sym) for name, p, dash, sym in (
        ("raw P(up)", p_raw, RAW_DASH, "circle-open"),
        ("calibrated P(up)", p_cal, T.VAL_DASH, "circle"),
        (f"{saved_label} P(up)", p_saved, T.ALT_DASH, "diamond-open")) if p is not None]
    tables = [reliability_table(labels, p, n_bins, lag=lag) for _, p, _, _ in series]
    base = float(labels.mean()) if len(labels) else float("nan")

    # one shared range on both axes: the bins, their bands and the reference rates (not the extreme
    # single predictions, which would squeeze every bin into a corner)
    pts = np.concatenate([np.r_[t[:, 0], t[:, 3], t[:, 4]] for t in tables] +
                         [np.array([base] + ([ref_rate] if ref_rate is not None else []))])
    pts = pts[np.isfinite(pts)]
    lo, hi = max(0.0, pts.min() - 0.03), min(1.0, pts.max() + 0.03)
    if hi - lo < 0.12:
        mid = (lo + hi) / 2
        lo, hi = max(0.0, mid - 0.06), min(1.0, mid + 0.06)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.74, 0.26], vertical_spacing=0.05)
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="perfect",
                             line=dict(color=T.NEUTRAL, width=1), hoverinfo="skip"), 1, 1)
    # raw and calibrated share their bins when calibration keeps the order of P(up) (temperature
    # scaling does): then the bands are identical and are drawn once, on the calibrated curve
    same_bins = len(series) > 1 and all(np.array_equal(tables[0][:, 1:], t[:, 1:]) for t in tables[1:])
    for k, ((name, _p, dash, sym), t) in enumerate(zip(series, tables)):
        bands = not same_bins or name.startswith("calibrated") or (k == 0 and p_cal is None)
        fig.add_trace(go.Scatter(
            x=t[:, 0], y=t[:, 1], mode="lines+markers", name=name, legendgroup=name,
            line=dict(color=c, dash=dash, width=2 if dash == T.VAL_DASH else 1.5),
            marker=dict(size=8, color=c, symbol=sym, line=dict(color=c, width=1.5)),
            error_y=dict(type="data", symmetric=False, array=t[:, 4] - t[:, 1], arrayminus=t[:, 1] - t[:, 3],
                         thickness=1.2, width=4, color=c, visible=bool(bands)),
            customdata=np.c_[t[:, 2], t[:, 3], t[:, 4]],
            hovertemplate=name + "<br>predicted %{x:.3f}<br>observed %{y:.3f} (95% band %{customdata[1]:.3f}-"
                                 "%{customdata[2]:.3f})<br>n %{customdata[0]:,.0f}<extra></extra>"), 1, 1)
    # base rates as legend entries (not in-plot labels, which would sit on the curves)
    rates = [(f"{block} up-rate {base:.3f}", base, REF_DASH)]
    if ref_rate is not None and np.isfinite(ref_rate):
        rates.append((f"{ref_label} {ref_rate:.3f}", float(ref_rate), REF2_DASH))
    for name, r, dash in rates:
        fig.add_trace(go.Scatter(x=[lo, hi], y=[r, r], mode="lines", name=name, hoverinfo="name",
                                 line=dict(color=T.NEUTRAL, dash=dash, width=1)), 1, 1)

    # the predictions under the curves: where the bins sit and how much lies outside the zoom
    edges = np.linspace(lo, hi, 41)
    mids = (edges[:-1] + edges[1:]) / 2
    outside = []
    for name, p, dash, _ in series:
        counts, _ = np.histogram(p, bins=edges)
        outside.append((name.split(" P(up)")[0], float(np.mean((p < lo) | (p > hi))), float(p.min()), float(p.max())))
        fig.add_trace(go.Scatter(x=mids.astype(np.float32), y=counts, mode="lines", line_shape="hvh", name=name,
                                 legendgroup=name, showlegend=False,
                                 line=dict(color=c, dash=dash, width=1.5 if dash == T.VAL_DASH else 1),
                                 fill="tozeroy" if dash == T.VAL_DASH else None, fillcolor=T.rgba(c, 0.15),
                                 hovertemplate=f"{name}<br>P(up) %{{x:.3f}}<br>%{{y:,}} samples<extra></extra>"), 2, 1)
    fig.update_xaxes(range=[lo, hi], row=1, col=1)
    fig.update_xaxes(range=[lo, hi], title_text="predicted P(up)", row=2, col=1)
    fig.update_yaxes(range=[lo, hi], title_text="observed up-rate", row=1, col=1)
    fig.update_yaxes(title_text="samples", row=2, col=1)

    n = len(labels)
    if subtitle is None:
        band = (f"bars: 95% band adjusted for overlapping {lag}-bar outcomes (HAC, lag {lag})" if lag
                else "bars: 95% binomial band (samples treated as independent)")
        lines = [f"{block} block: {n:,} samples outside the deadband, {n_bins} equal-count bins of ~{n // max(n_bins, 1):,}",
                 band + ("; raw and calibrated P(up) share their bins, so the bars are drawn once" if same_bins else "")]
        short = [nm.split(" P(up)")[0] for nm, _, _, _ in series]
        means = [f"{s} {float(np.mean(p)):.3f}" for s, (_, p, _, _) in zip(short, series)]
        eces = [f"{s} {ece_pos(labels, p):.3f}" for s, (_, p, _, _) in zip(short, series)]
        lines.append(f"mean P(up) {', '.join(means)} vs observed up-rate {base:.3f} · ECE {', '.join(eces)} · "
                     f"AUC {_auc(labels, series[0][1]):.3f} (0.5 = no skill)")
        if any(o[1] > 0 for o in outside):
            lines.append("axes zoomed to the bins; outside them: " + ", ".join(
                f"{o[1]:.1%} of {o[0]} (range {o[2]:.2f}-{o[3]:.2f})" for o in outside))
        if note:
            lines.append(note)
        subtitle = "<br>".join(lines)
    n_lines = subtitle.count("<br>") + 1
    apply(fig, title=title or ("Direction reliability" + (f", {horizon}" if horizon else "")), subtitle=subtitle,
          height=height or 560 + 18 * n_lines)
    fig.update_layout(margin=dict(t=78 + 17 * n_lines), legend=dict(y=1.01))
    return fig


# ------------------------------------------------------------------ coverage
def coverage_over_time_figure(y, lo, hi, *, window: int = 500, target: float = 0.9, title: Optional[str] = None,
                              horizon: Optional[str] = None, horizon_steps: Optional[int] = None,
                              inflation: Optional[float] = None, inflation_from: str = "this block",
                              saved: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                              saved_target: Optional[float] = None, subtitle: Optional[str] = None,
                              block: str = "test", n_width_bins: int = 5, max_points: int = 1500,
                              height: Optional[int] = None):
    """Rolling share of outcomes inside the interval with the range it shows by chance, the interval
    width along the block, and coverage by interval width (equal-count bins).

    ``target``: the level the intervals were built for (1 - alpha). ``inflation``: the variance
    inflation of a window mean (``coverage_inflation``) from a reference block; estimated on this
    block when omitted. ``saved``: (lo, hi) of a second pipeline, drawn dash-dotted for comparison.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    y, lo, hi = (np.asarray(a, float) for a in (y, lo, hi))
    n = len(y)
    window = int(max(2, min(window, n)))
    c = T.HORIZON_COLORS.get(horizon, T.INK_2)
    inside = ((y >= lo) & (y <= hi)).astype(float)
    width = hi - lo
    k = np.ones(window) / window
    roll = np.convolve(inside, k, mode="valid")          # trailing windows, plotted at their end
    roll_w = np.convolve(width, k, mode="valid")
    infl = float(inflation) if inflation is not None else coverage_inflation(inside, horizon_steps, window=window)
    half = S.Z95 * np.sqrt(target * (1 - target) * infl / window)
    b_lo, b_hi = max(0.0, target - half), min(1.0, target + half)
    n_eff_w = window / infl
    steps_txt = f"{horizon_steps}-bar" if horizon_steps else "overlapping"

    fig = make_subplots(rows=2, cols=2, specs=[[{}, {"rowspan": 2}], [{}, None]], column_widths=[0.7, 0.3],
                        row_heights=[0.58, 0.42], horizontal_spacing=0.09, vertical_spacing=0.16)
    # the right panel starts lower, so the time panel's legend row can run above it without touching it
    right = fig.get_subplot(1, 2).yaxis
    right.domain = (right.domain[0], right.domain[1] * 0.86)
    idx = S.thin(len(roll), max_points)
    x_roll = (idx + window - 1).astype(np.int32)
    x0, x1 = float(window - 1), float(n - 1)
    fig.add_trace(go.Scatter(x=[x0, x1, x1, x0, x0], y=[b_lo, b_lo, b_hi, b_hi, b_lo], fill="toself", mode="lines",
                             fillcolor=T.rgba(T.NEUTRAL, 0.16), line=dict(width=0),
                             name="95% chance range if on target", legend="legend", hoverinfo="skip"), 1, 1)
    fig.add_trace(go.Scatter(x=[x0, x1], y=[target, target], mode="lines", name=f"target {target:.2f}",
                             line=dict(color=T.NEUTRAL, dash=REF_DASH, width=1), legend="legend", hoverinfo="skip"),
                  1, 1)
    fig.add_trace(go.Scatter(x=x_roll, y=roll[idx].astype(np.float32), mode="lines", name="coverage",
                             line=dict(color=c, width=1.6), legend="legend",
                             hovertemplate="sample %{x:,}<br>coverage %{y:.3f}<extra></extra>"), 1, 1)
    ys = [roll]
    if saved is not None:
        s_lo, s_hi = (np.asarray(a, float) for a in saved)
        s_in = ((y >= s_lo) & (y <= s_hi)).astype(float)
        s_roll = np.convolve(s_in, k, mode="valid")
        ys.append(s_roll)
        fig.add_trace(go.Scatter(x=x_roll, y=s_roll[idx].astype(np.float32), mode="lines", name="saved pipeline",
                                 line=dict(color=c, width=1.2, dash=T.ALT_DASH), legend="legend",
                                 hovertemplate="saved pipeline<br>sample %{x:,}<br>coverage %{y:.3f}<extra></extra>"),
                      1, 1)
    y_min = min(float(np.min([a.min() for a in ys])), b_lo, target) - 0.01
    y_max = min(1.005, max(float(np.max([a.max() for a in ys])), b_hi, target) + 0.01)
    fig.update_yaxes(range=[y_min, y_max], title_text="share inside", row=1, col=1)

    # width along the block
    iw = S.thin(n, max_points)
    fig.add_trace(go.Scatter(x=iw.astype(np.int32), y=width[iw].astype(np.float32), mode="lines", name="per sample",
                             line=dict(color=T.rgba(c, 0.35), width=0.8), legend="legend2",
                             hovertemplate="sample %{x:,}<br>width $%{y:,.0f}<extra></extra>"), 2, 1)
    fig.add_trace(go.Scatter(x=x_roll, y=roll_w[idx].astype(np.float32), mode="lines", name=f"trailing {window} mean",
                             line=dict(color=c, width=1.6), legend="legend2",
                             hovertemplate="sample %{x:,}<br>mean width $%{y:,.0f}<extra></extra>"), 2, 1)
    if saved is not None:
        s_w = np.convolve(s_hi - s_lo, k, mode="valid")
        fig.add_trace(go.Scatter(x=x_roll, y=s_w[idx].astype(np.float32), mode="lines", name="saved pipeline",
                                 line=dict(color=c, width=1.2, dash=T.ALT_DASH), legend="legend2",
                                 hovertemplate="saved pipeline<br>sample %{x:,}<br>mean width $%{y:,.0f}<extra></extra>"),
                      2, 1)
    bottom = fig.get_subplot(2, 1).xaxis.plotly_name.replace("axis", "")        # "x3"
    fig.update_xaxes(matches=bottom, showticklabels=False, row=1, col=1)
    fig.update_xaxes(title_text=f"{block} sample (1-minute bars, time order)", row=2, col=1)
    fig.update_yaxes(title_text="width ($)", tickprefix="$", row=2, col=1)

    # coverage by width: do narrow intervals under-cover?
    t = _binned_rate(width, inside, n_width_bins, lag=int(horizon_steps or 0))
    fig.add_trace(go.Scatter(x=[t[0, 0], t[-1, 0]], y=[target, target], mode="lines", name=f"target {target:.2f}",
                             line=dict(color=T.NEUTRAL, dash=REF_DASH, width=1), legend="legend3", hoverinfo="skip"),
                  1, 2)
    fig.add_trace(go.Scatter(x=t[:, 0], y=t[:, 1], mode="lines+markers", name="coverage", legend="legend3",
                             line=dict(color=c, width=2), marker=dict(size=8, color=c),
                             error_y=dict(type="data", symmetric=False, array=t[:, 4] - t[:, 1],
                                          arrayminus=t[:, 1] - t[:, 3], thickness=1.2, width=4, color=c),
                             customdata=np.c_[t[:, 2], t[:, 3], t[:, 4]],
                             hovertemplate="mean width $%{x:,.0f}<br>coverage %{y:.3f} (95% band "
                                           "%{customdata[1]:.3f}-%{customdata[2]:.3f})<br>n %{customdata[0]:,.0f}"
                                           "<extra></extra>"), 1, 2)
    fig.update_xaxes(title_text=f"mean interval width, {n_width_bins} equal-count bins", tickprefix="$",
                     row=1, col=2)
    fig.update_yaxes(title_text="share inside", row=1, col=2)

    cov_all = float(inside.mean()) if n else float("nan")
    if subtitle is None:
        lines = [f"{block} block, {n:,} samples: overall coverage {cov_all:.3f} vs target {target:.2f}, "
                 f"mean width ${np.mean(width):,.0f}"]
        if saved is not None:
            st = f" (built for {saved_target:.2f})" if saved_target is not None else ""
            lines.append(f"dash-dot: the saved pipeline{st}, coverage {float(s_in.mean()):.3f}, "
                         f"mean width ${np.mean(s_hi - s_lo):,.0f}")
        lines += [f"shaded: the 95% range one {window}-sample window shows by chance when coverage is exactly on target",
                  f"({steps_txt} outcomes overlap: ~{n_eff_w:.0f} effective samples per window, autocorrelation from "
                  f"{inflation_from})"
                  + (f"; bars (right): 95% band adjusted for the overlap (HAC, lag {horizon_steps})"
                     if horizon_steps else "")]
        subtitle = "<br>".join(lines)
    n_lines = subtitle.count("<br>") + 1
    apply(fig, title=title or ("Interval coverage" + (f", {horizon}" if horizon else "")), subtitle=subtitle,
          height=height or 540 + 18 * n_lines, legend_top=False)
    T.panel_legend(fig, "legend", 1, 1, f"Coverage, trailing {window}")
    T.panel_legend(fig, "legend2", 2, 1, "Interval width")
    T.panel_legend(fig, "legend3", 1, 2, "Coverage by width")
    fig.update_layout(margin=dict(t=88 + 17 * n_lines))
    return fig


# ------------------------------------------------------------------ registry entries (data, config)
def _steps_from(config, kw):
    if config is not None and kw.get("horizon") and kw.get("horizon_steps") is None:
        kw["horizon_steps"] = S.horizon_steps(config, kw["horizon"])
    return kw


def reliability(data, config=None, **kw):
    """``data``: {"labels", "p_raw", "p_cal"?, "p_saved"?} (time order; outside the deadband)."""
    kw = _steps_from(config, kw)
    if data.get("p_saved") is not None:
        kw.setdefault("p_saved", data["p_saved"])
    return reliability_figure(data["labels"], data["p_raw"], data.get("p_cal"), **kw)


def interval_coverage(data, config=None, **kw):
    """``data``: {"y", "lo", "hi"} (time order)."""
    return coverage_over_time_figure(data["y"], data["lo"], data["hi"], **_steps_from(config, kw))
