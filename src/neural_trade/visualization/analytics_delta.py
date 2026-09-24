"""Price heads (delta): what the raw heads predict, what is served, and how much of it is noise.

The served delta is ``beta_h x raw``: the calibration pipeline fits ``beta_h`` as the least-squares
slope through 0 of the realised move on the raw head, on the calibration block, clipped to [0, 1].
The figure takes the served frame plus the raw heads (``raw_delta``) and derives beta exactly as
``dot(served, raw) / dot(raw, raw)``.

Layout, one column per horizon:

* table - metrics as rows, horizons as columns: served beta, RMSE and MAE of the raw head, the
  served delta and a zero prediction, skill vs predicting 0 (raw and served, with 95% intervals),
  mean / sd / share > 0 of the prediction vs the outcome, sign agreement with the direction head;
  under it the magnitude ordering |d0| <= |d1| <= |d2| of the raw heads vs the served deltas.
* row 2 - every sample, realised vs predicted (raw head). The plotted range spans the middle 99% of
  the predictions (fenced at 6 IQR) and the middle 99% of the outcomes; a point beyond it is drawn
  as a diamond in the shaded margin on that side (the shading's edge is the clip level), and every
  sample enters every statistic. The title gives Pearson and rank
  correlation with the 95% no-skill band; the in-panel note gives the top 0.5% |prediction|: how
  many episodes they form and the correlation without them.
* row 3 - mean realised move per predicted decile (95% CI, overlap-adjusted), with the served
  mapping y = beta x, the slope through 0 on this block (with and without the top 0.5% |prediction|),
  y = x and the block mean.
* row 4 - rolling correlation of the head with the outcome and its 95% no-skill band. Shrinkage
  does not change a correlation, so this row is the same for the raw and the served delta.
* row 5 - rolling skill of the served delta vs predicting 0 and its 95% no-skill band (per window).

Uncertainty. Consecutive 1-minute samples have overlapping targets (h - 1 shared bars), so N samples
carry far fewer independent outcomes. Correlation bands use the effective sample count N / deff,
with deff = 1 + 2 sum_k acf_pred(k) acf_outcome(k) (Bartlett); the binned means use a Newey-West
(Bartlett kernel, lag 2 x horizon bars) standard error per bin; the skill intervals are Newey-West
on the linearised loss ratio (``skill_half_width``), so a heavy-tailed head's own spread counts.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from neural_trade.visualization import stats as S
from neural_trade.visualization import theme as T
from neural_trade.visualization.analytics_common import _rolling_corr, _rolling_mean

SERVED_COLOR = T.OTHER_SERIES[1]     # the served delta (beta x raw): one colour in every row
TOP_PCT = 99.5                        # "top 0.5% |prediction|": the leverage points
N_BINS = 10
MAX_LINE_POINTS = 1500
BAND_POINTS = 400                     # a band of rolling means is smooth: fewer points keep the figure small
TABLE_PX = 230                        # header + nine metric rows (plotly pads each row a little)
NOTE_HEADROOM = 0.30                  # scatter y axis: room above the data for the in-panel note
IQR_FENCE = 6.0                       # scatter x range: never beyond 6 IQR from the quartiles
EDGE_X, EDGE_Y = 0.03, 0.045          # scatter: shaded margins beyond the plotted range (share of its span)
OUT_NAME = "beyond the plotted range (in the shaded margin; in every statistic)"
BINNED_NAME = "decile mean ± 95% CI (overlap-adjusted)"
DIAG_NAME = "y = x (face value)"
DIAG_DASH = "6px,3px"                 # short dashes: tells y = x apart from the solid block-mean line
MEAN_NAME = "block mean (no skill)"
BAND_NAME = "95% band if no skill"
EMPTY_NOTE = "constant prediction or too few samples: nothing to draw"


# ------------------------------------------------------------------ statistics
def _acf(v: np.ndarray, lags: int) -> np.ndarray:
    v = np.asarray(v, float) - np.mean(v)
    den = float(v @ v)
    if den <= 0:
        return np.r_[1.0, np.zeros(lags)]
    return np.r_[1.0, [float(v[:-k] @ v[k:]) / den for k in range(1, lags + 1)]]


def design_effect(a, b, steps: int) -> float:
    """Variance inflation of a correlation (or of a mean of a*b) between two autocorrelated series.

    Bartlett: var(r) ~ (1 + 2 sum_k acf_a(k) acf_b(k)) / n under no relationship, so the effective
    sample count is n / deff. For a prediction that stays fully persistent over the target's span
    this equals the report's n / steps; a less persistent prediction gives a smaller deff.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    lags = int(min(10 * max(1, steps), max(1, len(a) // 4)))
    if len(a) < 8 or np.std(a) == 0 or np.std(b) == 0:
        return float(max(1, steps))
    return float(max(1.0, 1.0 + 2.0 * np.sum(_acf(a, lags)[1:] * _acf(b, lags)[1:])))


def corr_band(n: int, deff: float) -> float:
    """95% half-width of a correlation of ``n`` samples with no relationship (n / deff effective)."""
    return S.corr_null(n / max(deff, 1.0))


def newey_west_var_of_mean(z, lag: int) -> float:
    """Newey-West (Bartlett kernel) variance of the mean of the time-ordered series ``z``."""
    z = np.asarray(z, float)
    n = len(z)
    if n < 2:
        return np.nan
    z = z - z.mean()
    lag = int(min(max(lag, 0), n - 1))
    v = float(z @ z)
    for k in range(1, lag + 1):
        v += 2.0 * (1.0 - k / (lag + 1)) * float(z[:-k] @ z[k:])
    return max(v, 0.0) / (n * n)


def binned_means(x, y, n_bins: int = N_BINS, *, steps: int = 1) -> np.ndarray:
    """Equal-count bins of x: rows (mean x, mean y, lo, hi, n, n_eff) with a 95% CI of mean y.

    The standard error is Newey-West (Bartlett kernel, lag 2 x steps) on the time-ordered residuals
    of the bin (zero outside it), so neighbours in time that fall in the same bin share information.
    With ``steps=1`` it is the independent-sample interval. ``n_eff`` = n x iid variance / HAC variance.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    lag = 2 * max(1, int(steps)) if steps > 1 else 0
    order = np.argsort(x, kind="mergesort")
    rows = []
    for idx in np.array_split(order, n_bins):
        m = len(idx)
        if m < 2:
            continue
        mu = float(y[idx].mean())
        e = np.zeros(len(y))
        e[idx] = y[idx] - mu                   # sums to 0, so its mean over the block is 0 too
        v0 = float(e @ e)
        v = max(newey_west_var_of_mean(e, lag) * len(e) ** 2, v0 / m)   # never below one sample's worth
        half = S.Z95 * np.sqrt(v) / m
        n_eff = m * v0 / v if v > 0 else float(m)
        rows.append((float(x[idx].mean()), mu, mu - half, mu + half, m, n_eff))
    return np.array(rows).reshape(-1, 6)


def slope0(d, y) -> float:
    """Least-squares slope through 0 of y on d (the calibration pipeline's shrink fit)."""
    dd = float(np.dot(d, d))
    return float(np.dot(y, d) / dd) if dd > 0 else np.nan


def skill_vs_zero(y, pred) -> float:
    """1 - MSE(pred) / MSE(0): > 0 means the prediction beats predicting no move."""
    den = float(np.mean(np.square(y)))
    return 1.0 - float(np.mean(np.square(y - pred))) / den if den > 0 else np.nan


def skill_half_width(y, pred, *, steps: int = 1) -> float:
    """95% half-width of ``skill_vs_zero``: a HAC interval on the loss ratio.

    skill = -R with R = mean(L) / mean(y^2) and L_t = pred_t^2 - 2 y_t pred_t. Linearising the ratio,
    R_hat - R ~ mean(z) with z_t = (L_t - R y_t^2) / mean(y^2), so every source of noise enters: the
    cross term E[y pred], the prediction's own spread E[pred^2] (dominated by a few extreme predictions
    when the head is heavy-tailed) and the denominator E[y^2]. The variance of mean(z) is Newey-West
    (Bartlett kernel, lag 2 x steps) because the h-bar targets overlap. It matches a moving-block
    bootstrap of the skill within about 5-15% on the real blocks.
    """
    y, p = np.asarray(y, float), np.asarray(pred, float)
    ey2 = float(np.mean(np.square(y))) if len(y) else 0.0
    if len(y) < 2 or ey2 <= 0:
        return np.nan
    loss = p * p - 2.0 * y * p
    r = float(np.mean(loss)) / ey2
    z = (loss - r * y * y) / ey2
    lag = 2 * int(steps) if steps > 1 else 0
    return float(S.Z95 * np.sqrt(newey_west_var_of_mean(z, lag)))


def _spearman(a, b) -> float:
    from scipy.stats import spearmanr

    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return np.nan
    return float(spearmanr(a, b).correlation)


def _pearson(a, b) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _episodes(idx: np.ndarray, gap: int) -> List[Tuple[int, int]]:
    """Runs of sorted sample indices whose neighbours are at most ``gap`` apart: [(first, last)]."""
    idx = np.sort(np.asarray(idx, int))
    if len(idx) == 0:
        return []
    cut = np.where(np.diff(idx) > gap)[0]
    starts, ends = np.r_[idx[0], idx[cut + 1]], np.r_[idx[cut], idx[-1]]
    return list(zip(starts.tolist(), ends.tolist()))


def _served_scale(served, raw) -> Tuple[float, bool]:
    """(beta, exact): served = beta x raw by least squares; exact when that reproduces served."""
    b = slope0(raw, served)
    if not np.isfinite(b):
        return np.nan, False
    tol = 1e-6 * max(float(np.max(np.abs(served))), 1e-9)
    return b, bool(np.max(np.abs(served - b * raw)) <= tol)


# ------------------------------------------------------------------ formatting
def _f(v, fmt: str, na: str = "n/a") -> str:
    return na if v is None or not np.isfinite(v) else format(v, fmt)


def _skill_cell(v: float, half: float) -> Tuple[str, str]:
    """Text and colour: ▲ good when the whole interval is above 0, ▼ bad when below, else neutral."""
    if not np.isfinite(v):
        return "n/a", T.MUTED
    text = f"{v:+.4f} ± {half:.4f}" if np.isfinite(half) else f"{v:+.4f}"
    if np.isfinite(half) and v - half > 0:
        return text + " ▲", T.GOOD
    if np.isfinite(half) and v + half < 0:
        return text + " ▼", T.CRITICAL
    return text + " ≈0", T.INK_2


def _pct(v: float) -> str:
    return _f(100 * v, ".1f") + "%"


def _axref(fig, row: int, col: int) -> Tuple[str, str]:
    sp = fig.get_subplot(row, col)
    return sp.xaxis.plotly_name.replace("axis", ""), sp.yaxis.plotly_name.replace("axis", "")


def _f32(v) -> np.ndarray:
    return np.asarray(v, dtype=np.float32)


# ------------------------------------------------------------------ per-horizon numbers
def horizon_stats(frame, raw: Dict[str, np.ndarray], served_known: bool) -> Dict[str, dict]:
    """Every number the figure shows, per horizon."""
    n = len(frame)
    out = {}
    for i, h in enumerate(T.HORIZONS):
        d = np.asarray(raw[h], float).reshape(-1)[:n]
        s = np.asarray(frame.delta[h], float).reshape(-1)[:n]
        y = np.asarray(frame.y[:, i], float)
        steps = S.horizon_steps(frame, h)
        deff = design_effect(d, y, steps)
        thr = float(np.percentile(np.abs(d), TOP_PCT)) if n else 0.0
        top = np.abs(d) > thr
        keep = ~top
        beta, exact = _served_scale(s, d) if served_known else (np.nan, False)
        p_up = np.asarray(frame.prob(h, True), float) > 0.5
        a_up = d > 0
        eps = _episodes(np.where(top)[0], steps)
        big = None
        if top.any():
            k = int(np.argmax(np.abs(d)))
            big = next(((a, b) for a, b in eps if a <= k <= b), (k, k)) + (float(d[k]),)
        out[h] = dict(
            d=d, s=s, y=y, steps=steps, deff=deff, top=top, thr=thr, beta=beta, exact=exact,
            corr=_pearson(d, y), rank=_spearman(d, y), band=corr_band(n, deff),
            slope=slope0(d, y), slope_trim=slope0(d[keep], y[keep]) if keep.any() else np.nan,
            corr_trim=_pearson(d[keep], y[keep]) if keep.sum() > 2 else np.nan,
            rmse=[float(np.sqrt(np.mean((y - p) ** 2))) for p in (d, s, np.zeros(n))],
            mae=[float(np.mean(np.abs(y - p))) for p in (d, s, np.zeros(n))],
            skill_raw=skill_vs_zero(y, d), skill_raw_hw=skill_half_width(y, d, steps=steps),
            skill_srv=skill_vs_zero(y, s), skill_srv_hw=skill_half_width(y, s, steps=steps),
            mean_ci=S.mean_ci(y, steps=steps), p_up=p_up, agree=float(np.mean(a_up == p_up)),
            agree_indep=float(np.mean(a_up) * np.mean(p_up) + (1 - np.mean(a_up)) * (1 - np.mean(p_up))),
            episodes=eps, biggest=big,
        )
    return out


def magnitude_ordering(delta: Dict[str, np.ndarray]) -> Tuple[float, float, float]:
    """Shares of samples with |d0| <= |d1|, |d1| <= |d2| and both."""
    A = np.abs(np.stack([np.asarray(delta[h], float) for h in T.HORIZONS], 1))
    a, b = A[:, 0] <= A[:, 1], A[:, 1] <= A[:, 2]
    return float(np.mean(a)), float(np.mean(b)), float(np.mean(a & b))


# ------------------------------------------------------------------ figure
def delta_analytics_figure(frame, config=None, *, raw_delta: Optional[Dict[str, np.ndarray]] = None,
                           window: int = 500, height: int = 1820, n_bins: int = N_BINS):
    """Price heads on one block: raw head vs outcome, the served (shrunk) delta, and their noise.

    ``frame``: the served PredictionFrame (delta = beta x raw). ``raw_delta``: the price heads before
    shrinkage; without it the x axis is the served delta and no served-vs-raw comparison is drawn.
    ``window``: rolling window in samples (bars); ``n_bins``: bins of the calibration row.
    """
    from plotly.subplots import make_subplots

    served_known = raw_delta is not None
    raw = raw_delta if served_known else frame.delta
    n = len(frame)
    w = int(min(window, max(20, n // 3)))
    minutes = None
    if config is not None:
        minutes = float(getattr(config, "RESAMPLE_MINUTES", 1) or 1) * float(getattr(config, "WINDOW_STEP", 1) or 1)
    ctx = dict(
        served_known=served_known, n=n, w=w, n_bins=n_bins,
        per_x=minutes / 60.0 if minutes else 1.0, x_unit="hours into the block" if minutes else "sample (time order)",
        x_suffix=" h" if minutes else "", head=("raw price head" if served_known else "served delta"),
    )
    block = getattr(frame, "split", "") or "this"
    st = horizon_stats(frame, raw, served_known)

    titles = [f"Per-horizon errors on the {block} block: n = {n:,} samples"
              + (f" ({n * minutes / 1440:.1f} days of {minutes:g}-min bars)" if minutes else "")]
    titles += [f"{T.horizon_label(h, config)}: corr {_f(st[h]['corr'], '+.3f')}, rank {_f(st[h]['rank'], '+.3f')}"
               f" (noise ±{st[h]['band']:.3f})" for h in T.HORIZONS]
    titles += [f"fit {_f(st[h]['slope'], '+.2f')} · w/o top 0.5% {_f(st[h]['slope_trim'], '+.2f')}"
               + (f" · served β {_f(st[h]['beta'], '.3f')}" if served_known else "") for h in T.HORIZONS]
    titles += [f"noise ±{corr_band(w, st[h]['deff']):.2f} (n_eff ≈ {w / st[h]['deff']:.0f} / window)"
               f" · block {_f(st[h]['corr'], '+.3f')}" for h in T.HORIZONS]
    titles += [f"block skill {_f(st[h]['skill_srv'], '+.4f')} ± {_f(st[h]['skill_srv_hw'], '.4f')}"
               for h in T.HORIZONS]

    top_m, bottom_m, vspace = 132, 44, 0.058
    plot_px = height - top_m - bottom_m
    table_frac = min(0.35, TABLE_PX / (plot_px * (1 - 4 * vspace)))   # the table keeps its pixel height
    specs = [[{"type": "table", "colspan": 3}, None, None]] + [[{}, {}, {}] for _ in range(4)]
    fig = make_subplots(rows=5, cols=3, specs=specs, subplot_titles=titles,
                        row_heights=[table_frac] + [(1 - table_frac) * r for r in (0.29, 0.25, 0.23, 0.23)],
                        vertical_spacing=vspace, horizontal_spacing=0.055)
    for a in fig.layout.annotations:
        a.update(font=dict(size=12, color=T.INK_2))

    _add_table(fig, st, config, served_known, 4.0 / plot_px)
    first = T.legend_once()
    for j, h in enumerate(T.HORIZONS, start=1):
        _scatter_panel(fig, j, h, st[h], ctx, first)
        _binned_panel(fig, j, h, st[h], ctx, first)
        _rolling_panels(fig, j, h, st[h], ctx, first)
        fig.update_xaxes(title_text=f"predicted move ($), {ctx['head']}", row=2, col=j)
        fig.update_xaxes(title_text="predicted move ($), decile mean", row=3, col=j)
        fig.update_xaxes(title_text=ctx["x_unit"], row=4, col=j)
        fig.update_xaxes(title_text=ctx["x_unit"], row=5, col=j)
    for row, title in ((2, "realised move ($)"), (3, "mean realised ($)"), (4, "Pearson r"), (5, "skill vs 0")):
        fig.update_yaxes(title_text=title, row=row, col=1)
    fig.update_xaxes(title_standoff=4)
    fig.update_yaxes(title_standoff=4)

    T.note_on_empty(fig, EMPTY_NOTE)
    T.apply(fig, title="Price heads (delta)", height=height, subtitle=_subtitle(st, served_known, block),
            legend_top=False)
    fig.update_layout(margin=dict(t=top_m, b=bottom_m, l=64, r=24), showlegend=True)
    span = (f", {w} bars ≈ {w * ctx['per_x']:.1f} h" if minutes else f", {w} samples")
    heads = {
        "legend2": f"Every sample ({ctx['head']})",
        "legend3": "Binned by predicted decile",
        "legend4": "Rolling correlation" + span + (" (same for raw and served)" if served_known else ""),
        "legend5": "Served delta: rolling skill vs predicting 0" + span,
    }
    for k, row in (("legend2", 2), ("legend3", 3), ("legend4", 4), ("legend5", 5)):
        T.panel_legend(fig, k, row, 1, heads[k])
        fig.layout[k].y = fig.layout[k].y + 21.0 / plot_px     # above the subplot titles
    return fig


def scatter_x_range(d) -> Tuple[float, float]:
    """Plotted prediction range (before padding): the middle 99% of the predictions, fenced at
    6 IQR beyond the quartiles (a single burst of extreme predictions can fill the top 0.5% of a short
    block; the fence never binds on a Gaussian), and always including 0."""
    d = np.asarray(d, float)
    if len(d) == 0:
        return 0.0, 0.0
    lo, hi = np.percentile(d, [100 - TOP_PCT, TOP_PCT])
    q1, q3 = np.percentile(d, [25, 75])
    iqr = q3 - q1
    if iqr > 0:
        lo, hi = max(lo, q1 - IQR_FENCE * iqr), min(hi, q3 + IQR_FENCE * iqr)
    return min(float(lo), 0.0), max(float(hi), 0.0)


def _proxy(fig, row, legend, group, name, **kw):
    """A legend key in neutral ink for a series drawn in every horizon's colour."""
    import plotly.graph_objects as go

    fig.add_trace(go.Scatter(x=[None], y=[None], name=name, legend=legend, legendgroup=group, showlegend=True,
                             hoverinfo="skip", **kw), row, 1)


def scatter_box(d, y) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """The scatter's plotted range (x, y): x = the middle 99% of the predictions (fenced, see
    ``scatter_x_range``) padded by 6%; y = +/- the 99.5th percentile of |outcome|, plus 2%."""
    lo, hi = scatter_x_range(d)
    pad = 0.06 * (hi - lo) if hi > lo else max(1.0, abs(hi))
    y = np.asarray(y, float)
    ylim = float(np.percentile(np.abs(y), TOP_PCT)) * 1.02 if len(y) else 1.0
    return (lo - pad, hi + pad), ((-ylim, ylim) if ylim > 0 else (-1.0, 1.0))


def _scatter_panel(fig, j, h, q, ctx, first):
    """Row 2: every sample. Inside the plotted range (the clear box) each sample is a dot; a sample
    beyond it is a diamond in the shaded margin on that side, so every point stays visible and the
    clip level is the edge of the shading."""
    import plotly.graph_objects as go

    d, y, c = q["d"], q["y"], T.HORIZON_COLORS[h]
    xr, yr = scatter_box(d, y)
    mx, my = EDGE_X * (xr[1] - xr[0]), EDGE_Y * (yr[1] - yr[0])      # margin widths (data units)
    ax_x = (xr[0] - mx, xr[1] + mx)
    ax_y = (yr[0] - my, yr[1] + my)
    out = (d < xr[0]) | (d > xr[1]) | (y < yr[0]) | (y > yr[1])
    if first("samples"):
        _proxy(fig, 2, "legend2", "samples", "sample", mode="markers", marker=dict(size=6, color=T.rgba(T.INK_2, 0.6)))
        _proxy(fig, 2, "legend2", "beyond", OUT_NAME, mode="markers",
               marker=dict(size=8, symbol="diamond-open", color=T.INK_2, line=dict(width=1.5)))
    fig.add_trace(go.Scattergl(
        x=_f32(d[~out]), y=_f32(y[~out]), mode="markers", name="sample", legend="legend2", legendgroup="samples",
        showlegend=False, marker=dict(size=3, color=T.rgba(c, 0.3)),
        hovertemplate="predicted %{x:$,.1f}<br>realised %{y:$,.1f}<extra>" + h + "</extra>"), 2, j)
    if out.any():
        def to_margin(v, lo, hi, m):             # inside: unchanged; beyond: the middle of that side's margin
            return np.where(v < lo, lo - m / 2, np.where(v > hi, hi + m / 2, v))

        fig.add_trace(go.Scattergl(
            x=_f32(to_margin(d[out], *xr, mx)), y=_f32(to_margin(y[out], *yr, my)),
            mode="markers", name=OUT_NAME, legend="legend2", legendgroup="beyond",
            showlegend=False, customdata=np.c_[d[out], y[out], np.where(out)[0]].astype(np.float32),
            marker=dict(size=7, symbol="diamond-open", color=c, line=dict(width=1.5)),
            hovertemplate="sample %{customdata[2]:.0f}<br>predicted %{customdata[0]:$,.1f}"
                          "<br>realised %{customdata[1]:$,.1f}<extra>beyond the plotted range</extra>"), 2, j)
    # the margins: four strips around the plotted range (the note's headroom above stays clear)
    xref, yref = _axref(fig, 2, j)
    for x0, x1, y0, y1 in ((ax_x[0], xr[0], ax_y[0], ax_y[1]), (xr[1], ax_x[1], ax_y[0], ax_y[1]),
                           (xr[0], xr[1], ax_y[0], yr[0]), (xr[0], xr[1], yr[1], ax_y[1])):
        fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1, xref=xref, yref=yref, layer="below",
                      line=dict(width=0), fillcolor=T.rgba(T.NEUTRAL, 0.13))
    fig.add_hline(y=0, line=dict(color=T.NEUTRAL, width=1), row=2, col=j, exclude_empty_subplots=False)
    fig.update_xaxes(range=list(ax_x), row=2, col=j)
    # headroom above the top margin holds the note, so it never covers a point
    fig.update_yaxes(range=[ax_y[0], ax_y[1] + NOTE_HEADROOM * (yr[1] - yr[0])], row=2, col=j)
    eps, big = q["episodes"], q["biggest"]
    lines = [f"{int(out.sum())} points beyond the plotted range (◇ in the shaded margin)",
             f"top 0.5% |pred| (> ${q['thr']:,.0f}): {int(q['top'].sum())} in {len(eps)} "
             f"episode{'s' if len(eps) != 1 else ''}"]
    if big is not None:
        lines.append(f"largest {big[2]:+,.0f} $, samples {big[0]}-{big[1]}")
    lines.append(f"without them: corr {_f(q['corr_trim'], '+.3f')}, fit {_f(q['slope_trim'], '+.2f')}")
    fig.add_annotation(x=0.01, y=0.99, xref=f"{xref} domain", yref=f"{yref} domain", text="<br>".join(lines),
                       showarrow=False, xanchor="left", yanchor="top", align="left",
                       font=dict(size=10, color=T.INK_2), bgcolor=T.rgba(T.PAPER, 0.75))


def _binned_panel(fig, j, h, q, ctx, first):
    """Row 3: mean realised per predicted decile with overlap-adjusted CIs, and the slopes as lines."""
    import plotly.graph_objects as go

    d, y, c = q["d"], q["y"], T.HORIZON_COLORS[h]
    if not (np.std(d) > 0 and ctx["n"] >= 2 * ctx["n_bins"]):
        return
    t = binned_means(d, y, ctx["n_bins"], steps=q["steps"])
    if first("binned"):
        _proxy(fig, 3, "legend3", "binned", BINNED_NAME, mode="lines+markers",
               line=dict(color=T.INK_2, width=2), marker=dict(size=6, color=T.INK_2))
    fig.add_trace(go.Scatter(
        x=_f32(t[:, 0]), y=_f32(t[:, 1]), mode="lines+markers", name=BINNED_NAME, legend="legend3",
        legendgroup="binned", showlegend=False, line=dict(color=c, width=2), marker=dict(size=6, color=c),
        error_y=dict(type="data", symmetric=False, array=_f32(t[:, 3] - t[:, 1]), arrayminus=_f32(t[:, 1] - t[:, 2]),
                     thickness=1.2, width=4, color=c),
        customdata=np.c_[t[:, 4], t[:, 5], t[:, 2], t[:, 3]].astype(np.float32),
        hovertemplate="predicted (decile mean) %{x:$,.1f}<br>realised mean %{y:$,.1f}"
                      "<br>95% CI %{customdata[2]:$,.1f} to %{customdata[3]:$,.1f} (overlap-adjusted)"
                      "<br>n %{customdata[0]:,.0f} (n_eff %{customdata[1]:,.0f})<extra>" + h + "</extra>"), 3, j)
    xs = np.array([t[:, 0].min(), t[:, 0].max()])
    lines = [("fit", "fit through 0", "least-squares slope through 0 on this block", q["slope"],
              dict(color=T.INK, width=1.5)),
             ("trim", "fit w/o top 0.5%", "the same fit without the top 0.5% |prediction|", q["slope_trim"],
              dict(color=T.INK, width=1.5, dash=T.ALT_DASH))]
    if ctx["served_known"]:
        lines.insert(0, ("served", "served β × raw", "served delta = β × raw, β fitted on the calibration block",
                         q["beta"], dict(color=SERVED_COLOR, width=2.2)))
    for key, name, tip, b, line in lines:
        if np.isfinite(b):
            fig.add_trace(go.Scatter(x=_f32(xs), y=_f32(b * xs), mode="lines", name=name, legend="legend3",
                                     legendgroup=key, showlegend=first(key), line=line,
                                     hovertemplate=f"{tip}: slope {b:+.3f}<extra>{h}</extra>"), 3, j)
    fig.add_trace(go.Scatter(x=_f32(xs), y=_f32(xs), mode="lines", name=DIAG_NAME, legend="legend3",
                             legendgroup="diag", showlegend=first("diag"),
                             line=dict(color=T.NEUTRAL, width=1.5, dash=DIAG_DASH),
                             hovertemplate="y = x: the prediction taken at face value<extra></extra>"), 3, j)
    ym = float(np.mean(y))
    fig.add_trace(go.Scatter(x=_f32(xs), y=_f32([ym, ym]), mode="lines", name=MEAN_NAME,
                             legend="legend3", legendgroup="mean", showlegend=first("mean"),
                             line=dict(color=T.NEUTRAL, width=1),
                             hovertemplate=f"block mean realised {ym:+,.1f} $ (no skill)<extra>{h}</extra>"), 3, j)
    # the y range follows the CIs: y = x leaves the panel rather than squashing the decile means
    ylo, yhi = min(float(t[:, 2].min()), ym, 0.0), max(float(t[:, 3].max()), ym, 0.0)
    ypad = 0.08 * (yhi - ylo) if yhi > ylo else 1.0
    fig.update_yaxes(range=[ylo - ypad, yhi + ypad], row=3, col=j)


def served_skill_band(s, y, w: int, deff: float):
    """Pointwise 95% band of the rolling served skill if the served delta had no skill.

    In a window, skill = (2 E[y s] - E[s^2]) / E[y^2]. With s independent of y it is centred on
    -E_w[s^2] / E_w[y^2] (the cost of the served spread in that window) with half-width
    2 z sqrt(E_w[s^2] / E_w[y^2]) sqrt(deff / w). Conditioning on the window keeps the coverage at 95%
    when volatility changes along the block (a constant band misses 6-10% of the no-skill windows).
    """
    _, my2 = _rolling_mean(y * y, w)
    _, ms2 = _rolling_mean(s * s, w)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(my2 > 0, ms2 / my2, np.nan)
    hw = 2.0 * S.Z95 * np.sqrt(r) * np.sqrt(max(deff, 1.0) / w)
    return -r - hw, -r + hw


def _rolling_panels(fig, j, h, q, ctx, first):
    """Rows 4-5: rolling correlation (raw = served) and rolling skill of the served delta, with bands."""
    import plotly.graph_objects as go

    d, s, y, c, w = q["d"], q["s"], q["y"], T.HORIZON_COLORS[h], ctx["w"]
    xw, rc = _rolling_corr(d, y, w)
    if len(xw):
        _, my2 = _rolling_mean(y * y, w)
        _, mse = _rolling_mean((y - s) ** 2, w)
        idx = S.thin(len(xw), MAX_LINE_POINTS)
        xx = _f32(xw[idx] * ctx["per_x"])
        ends = np.array([xx[0], xx[-1]])
        hx = "%{x:,.1f}" + ctx["x_suffix"]
        band = corr_band(w, q["deff"])
        _band(fig, 4, j, ends, np.full(2, -band), np.full(2, band), first("band4"), "legend4", "band4")
        if first("rc"):
            _proxy(fig, 4, "legend4", "rc", f"{ctx['head']}, trailing window", mode="lines",
                   line=dict(color=T.INK_2, width=1.5))
            _proxy(fig, 4, "legend4", "blk4", "whole block", mode="lines",
                   line=dict(color=T.INK_2, width=1.2, dash=T.ALT_DASH))
        fig.add_trace(go.Scatter(x=xx, y=_f32(rc[idx]), mode="lines", name=f"{ctx['head']}, trailing window",
                                 legend="legend4", legendgroup="rc", showlegend=False, line=dict(color=c, width=1.5),
                                 hovertemplate=hx + "<br>corr %{y:+.3f}<extra>" + h + "</extra>"), 4, j)
        _block_line(fig, 4, j, ends, q["corr"], c, False, "legend4", "blk4", "whole block")
        with np.errstate(invalid="ignore", divide="ignore"):
            rs = np.where(my2 > 0, 1.0 - mse / my2, np.nan)
        lo, hi = served_skill_band(s, y, w, q["deff"])
        bi = S.thin(len(xw), BAND_POINTS)
        _band(fig, 5, j, _f32(xw[bi] * ctx["per_x"]), lo[bi], hi[bi], first("band5"), "legend5", "band5")
        fig.add_trace(go.Scatter(x=xx, y=_f32(rs[idx]), mode="lines", name="served delta, trailing window",
                                 legend="legend5", legendgroup="rs", showlegend=first("rs"),
                                 line=dict(color=SERVED_COLOR, width=1.5),
                                 hovertemplate=hx + "<br>skill %{y:+.4f}<extra>" + h + "</extra>"), 5, j)
        _block_line(fig, 5, j, ends, q["skill_srv"], SERVED_COLOR, first("blk5"), "legend5", "blk5", "whole block")
    for r in (4, 5):
        fig.add_hline(y=0, line=dict(color=T.NEUTRAL, width=1), row=r, col=j, exclude_empty_subplots=False)


def _band(fig, row, col, x, lo, hi, show, legend, group):
    """A filled 95% no-skill band between ``lo`` and ``hi`` (constant or pointwise along ``x``)."""
    import plotly.graph_objects as go

    const = bool(np.ptp(lo) == 0 and np.ptp(hi) == 0)
    tip = (f"95% band if no skill: {lo[0]:+.4f} to {hi[0]:+.4f}<extra></extra>" if const
           else "95% band if no skill (this window)<extra></extra>")
    fig.add_trace(go.Scatter(x=x, y=_f32(lo), mode="lines", line=dict(width=0), legend=legend,
                             legendgroup=group, showlegend=False, hoverinfo="skip"), row, col)
    fig.add_trace(go.Scatter(x=x, y=_f32(hi), mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor=T.rgba(T.NEUTRAL, 0.22), name=BAND_NAME,
                             legend=legend, legendgroup=group, showlegend=show, hovertemplate=tip), row, col)


def _block_line(fig, row, col, x_ends, v, color, show, legend, group, name):
    import plotly.graph_objects as go

    if np.isfinite(v):
        fig.add_trace(go.Scatter(x=x_ends, y=[v, v], mode="lines", name=name, legend=legend, legendgroup=group,
                                 showlegend=show, line=dict(color=color, width=1.2, dash=T.ALT_DASH),
                                 hovertemplate=f"{name}: {v:+.4f}<extra></extra>"), row, col)


def _subtitle(st, served_known: bool, block: str) -> str:
    """Three short lines (the figure autosizes to the notebook width)."""
    if served_known:
        lines = ["x = raw price head. Served delta = β × raw; β = least-squares slope through 0 fitted on the "
                 "calibration block, clipped to [0, 1].",
                 ("On this calibration block the fit through 0 is β itself. " if block == "cal" else
                  f"Fits measured on this {block} block do not set β. ")
                 + "Skill = 1 - MSE / MSE(predict 0): below 0 is worse than predicting no move."]
    else:
        lines = ["x = served delta (raw heads not supplied: pass raw_delta= to compare the heads with what is served).",
                 f"Fits are least-squares slopes through 0 on the {block} block. "
                 "Skill = 1 - MSE / MSE(predict 0): below 0 is worse than predicting no move."]
    lines.append("Targets overlap (h-bar moves, 1 bar apart): noise bands use N / deff effective samples, deff = "
                 + " / ".join(f"{st[h]['deff']:.1f}" for h in T.HORIZONS)
                 + " (Bartlett); binned CIs and skill ± are Newey-West (lag 2 × horizon).")
    return "<br>".join(lines)


def _add_table(fig, st, config, served_known: bool, px: float):
    """Metrics as rows, horizons as columns (in the order of the panel columns below)."""
    import plotly.graph_objects as go

    na = "n/a"
    rows: List[Tuple[str, List[Tuple[str, str]]]] = []

    def add(label, cells):
        rows.append((label, cells))

    add("served β (served = β × raw, β fit on cal)",
        [((_f(st[h]["beta"], ".3f") + ("" if st[h]["exact"] else " (≈)")
           + (" (clipped: served = 0)" if st[h]["beta"] == 0 else "")) if served_known else na, T.INK)
         for h in T.HORIZONS])
    for label, key in (("RMSE $: raw / served / predict 0", "rmse"), ("MAE $: raw / served / predict 0", "mae")):
        add(label, [(f"{_f(st[h][key][0], ',.1f') if served_known else na} / {st[h][key][1]:,.1f} / "
                     f"{st[h][key][2]:,.1f}", T.INK_2) for h in T.HORIZONS])
    add("skill vs predicting 0: raw head ± 95%",
        [_skill_cell(st[h]["skill_raw"], st[h]["skill_raw_hw"]) if served_known else (na, T.MUTED)
         for h in T.HORIZONS])
    add("skill vs predicting 0: served delta ± 95%",
        [_skill_cell(st[h]["skill_srv"], st[h]["skill_srv_hw"]) for h in T.HORIZONS])
    mean_cells = []
    for h in T.HORIZONS:
        m, mlo, mhi = st[h]["mean_ci"]
        mean_cells.append((f"{np.mean(st[h]['d']):+,.1f} / {m:+,.1f} ± {(mhi - mlo) / 2:,.1f}", T.INK_2))
    add("mean $: prediction / realised ± 95%", mean_cells)
    add("sd $: prediction / realised", [(f"{np.std(st[h]['d']):,.1f} / {np.std(st[h]['y']):,.1f}", T.INK_2)
                                        for h in T.HORIZONS])
    add("share above 0: prediction / realised",
        [(f"{_pct(np.mean(st[h]['d'] > 0))} / {_pct(np.mean(st[h]['y'] > 0))}", T.INK_2) for h in T.HORIZONS])
    add("sign(Δ) matches the side of P(up) vs ½ (if independent)",  # no ">": plotly tables mis-size it
        [(f"{_pct(st[h]['agree'])} ({_pct(st[h]['agree_indep'])})", T.INK_2) for h in T.HORIZONS])

    values = [[r[0] for r in rows]] + [[r[1][k][0] for r in rows] for k in range(3)]
    colors = [[T.INK_2] * len(rows)] + [[r[1][k][1] for r in rows] for k in range(3)]
    header = ["<b>per horizon</b>" + ("" if served_known else " (x = served delta)")] + [
        f"<b>{T.horizon_label(h, config)}</b>" for h in T.HORIZONS]
    fig.add_trace(go.Table(
        columnwidth=[1.35, 1, 1, 1],
        header=dict(values=header, fill_color=T.SURFACE, line_color=T.GRID, height=24, align=["left"] + ["right"] * 3,
                    font=dict(color=[T.INK] + [T.HORIZON_COLORS[h] for h in T.HORIZONS], size=11)),
        cells=dict(values=values, fill_color=T.PAPER, line_color=T.GRID, font=dict(color=colors, size=11),
                   align=["left"] + ["right"] * 3, height=21)), 1, 1)

    # cross-horizon: magnitude ordering (raw vs served) and sign agreement on all three horizons
    labels = ("|Δh0| ≤ |Δh1|", "|Δh1| ≤ |Δh2|", "full chain")
    so = magnitude_ordering({h: st[h]["s"] for h in T.HORIZONS})
    if served_known:
        ro = magnitude_ordering({h: st[h]["d"] for h in T.HORIZONS})
        first = "Magnitude ordering, share of samples: " + " · ".join(
            f"{lab}: raw {_pct(r)} → served {_pct(s)}" for lab, r, s in zip(labels, ro, so))
        second = "The served ordering reflects the per-horizon β, not the model. "
    else:
        first = "Magnitude ordering of the served deltas, share of samples: " + " · ".join(
            f"{lab}: {_pct(s)}" for lab, s in zip(labels, so))
        second = ("Served deltas: the ordering reflects the per-horizon β, not the model; "
                  "pass raw_delta= for the heads. ")
    agree_all = np.all(np.stack([(st[h]["d"] > 0) == st[h]["p_up"] for h in T.HORIZONS], 1), axis=1)
    second += f"sign(Δ) agrees with P(up) > ½ on all 3 horizons: {_pct(float(np.mean(agree_all)))} of samples."
    y0 = fig.data[-1].domain.y[0]           # the footer sits just under the table, in the gap above row 2
    fig.add_annotation(x=0.0, y=y0 - px, xref="paper", yref="paper", xanchor="left", yanchor="top",
                       text=first + "<br>" + second, showarrow=False, align="left",
                       font=dict(size=11, color=T.INK_2))
