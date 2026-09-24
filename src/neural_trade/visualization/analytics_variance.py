"""Variance heads: does the predicted sigma size and rank the error, is the error shape Gaussian,
and do the 90% intervals hold their coverage and width over time.

One column per horizon, five rows:

1. **Size and rank.** RMS error per equal-count bin of the predicted sigma, with 95% block-bootstrap
   intervals, against the diagonal (a calibrated sigma has RMS error = sigma) and against a free
   baseline: the trailing realised volatility of the raw input window times sqrt(horizon bars), the
   same scale the conformal intervals use (``interval_scale("realized_vol")``). The panel title gives
   both rank correlations with the squared error and a block-bootstrap interval of their difference.
2. **Shape.** The PIT histogram with the range a calibrated sigma would still show from noise (two
   thin grey edges over the bars). The titles separate scale (``std z``) from shape
   (``mean |z| / std z``, 0.80 for any Gaussian; excess kurtosis).
3. **Tails.** How often ``|error| > k sigma``, as a multiple of the Gaussian rate, on a log axis that
   always reaches below 1x. A PIT hump with ``std(z)`` near 1 comes from heavy tails, which this row
   shows directly.
4. **Coverage over time.** Trailing coverage of the conformal interval and of the Gaussian
   +/-1.645 sigma interval, with the range expected from noise.
5. **Width over time.** The trailing mean width of both intervals (sharpness).

Errors are measured around the SERVED delta (``frame.delta``: beta x raw head), as ``eval_report``
does, so the numbers match the report. The variance head was trained on the raw head's errors; pass
``raw_delta`` to add that view to row 1.

Uncertainty: consecutive 1-minute samples have overlapping targets (a horizon of h bars shares h - 1
bars with its neighbour), so nothing here treats the N samples as independent.

* Row 1: circular block bootstrap (blocks of :data:`BOOT_BLOCK_HORIZONS` horizons; sigma bins held
  fixed). The rank-correlation difference: circular block bootstrap with :data:`RANK_BOOT_BLOCK`-sample
  blocks, since the ranking follows volatility regimes that last hours.
* Rows 2-4: the variance of a mean of 0/1 indicators is inflated by a Bartlett long-run-variance
  factor estimated from the indicator's own autocorrelation over :data:`LRV_LAG_HORIZONS` horizons
  of lags (a HAC estimate). The rule of thumb ``N / h`` effective samples is about twice too
  conservative for these indicators: it would hide real departures. Row 4 uses the wider of the two
  indicators' factors for its one band, and the band is pointwise: over a long block the trailing
  line leaves it now and then by chance alone.

Layout: the gap between rows is fixed in pixels (row heading, keys and two-line panel titles above
each panel), so any ``height`` keeps the headings clear of the previous row's axis titles; the text
fits an output area of 1000 px or wider.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from neural_trade.visualization import stats as S
from neural_trade.visualization import theme as T
from neural_trade.visualization.analytics_common import _grid, _rolling_mean

Z90 = 1.6448536269514722            # two-sided 90% Gaussian quantile
BOOT_BLOCK_HORIZONS = 3              # block length of the per-bin RMS bootstrap, in horizons
RANK_BOOT_BLOCK = 200                # block length (samples) of the rank-correlation bootstrap: the
                                     # ranking depends on volatility regimes, which last hours
LRV_LAG_HORIZONS = 3                 # lags of the long-run-variance factor, in horizons
BOOT_REPS = 300
K_GRID = np.round(np.arange(0.5, 3.51, 0.25), 2)   # |z| thresholds of the tail row
MAX_LINE_POINTS = 1500               # rolling lines are thinned to at most this many points

_BAND = T.rgba(T.NEUTRAL, 0.28)      # "range expected from noise" fill (row 4, drawn below the lines)
_BAND_EDGE_DASH = "dash"             # row 2: the noise range as two thin grey edges over the bars
_LOG_TICKS = (0.1, 0.25, 0.5, 0.75, 1, 2, 5, 10, 20, 50, 100)
_ROWS = 5

# Vertical layout in pixels, so the row headings never depend on the figure height: under each
# panel its tick labels and x title; above it the row heading (legend title + one line of keys) and
# the two-line panel titles.
_AX_BELOW_PX = 46
_HEAD_PX = 40
_TITLES_PX = 32
_AIR_PX = 12
_GAP_PX = _AX_BELOW_PX + _AIR_PX + _HEAD_PX + _TITLES_PX
_TITLE_PX = 40                       # figure title
_SUB_LINE_PX = 16                    # each subtitle line
_MARGIN_T = _TITLE_PX + 4 * _SUB_LINE_PX + _HEAD_PX + _TITLES_PX + 8   # with the usual four lines
_MARGIN_B = 50
_ROW_HEIGHTS = (1.15, 1.0, 1.0, 1.0, 0.85)
DEFAULT_HEIGHT = 1840
MIN_HEIGHT = 1500                    # below this the panels get too short; smaller heights are raised


# ------------------------------------------------------------------ statistics
def lrv_factor(indicator, lags: int) -> float:
    """How many times larger the variance of a mean of ``indicator`` is than for independent samples:
    ``1 + 2 * sum_k (1 - k / (L + 1)) * acf(k)`` over ``L = lags`` lags (Bartlett weights), floored at 1."""
    x = np.asarray(indicator, float)
    x = x[np.isfinite(x)]
    x = x - x.mean()
    n, v = len(x), float(x @ x)
    if n < 3 or v <= 0:
        return 1.0
    L = int(min(max(lags, 0), n - 2))
    s = sum((1.0 - k / (L + 1.0)) * float(x[:-k] @ x[k:]) / v for k in range(1, L + 1))
    return max(1.0, 1.0 + 2.0 * s)


def _circular_blocks(n: int, block: int, reps: int, rng) -> np.ndarray:
    """[reps, n] indices of a circular block bootstrap."""
    block = int(max(1, min(block, n)))
    nb = -(-n // block)
    starts = rng.integers(0, n, size=(reps, nb))
    return ((starts[:, :, None] + np.arange(block)) % n).reshape(reps, -1)[:, :n]


def binned_rms(x, e, n_bins: int = 12, *, block: int = 1, reps: int = BOOT_REPS, seed: int = 0) -> np.ndarray:
    """Equal-count bins of ``x``: rows of (mean x, min x, max x, RMS of e, CI lo, CI hi, n).

    The 95% interval of each bin's RMS comes from a circular block bootstrap of the samples in time
    order (blocks of ``block`` samples), with the bin of every sample held fixed; ``reps=0`` skips it
    (CI columns NaN).
    """
    x, e = np.asarray(x, float), np.asarray(e, float)
    order = np.argsort(x, kind="mergesort")
    parts = [p for p in np.array_split(order, n_bins) if len(p)]
    k = len(parts)
    lab = np.empty(len(x), dtype=np.int64)
    for b, p in enumerate(parts):
        lab[p] = b
    e2 = e ** 2
    cnt = np.bincount(lab, minlength=k)
    rms = np.sqrt(np.bincount(lab, e2, k) / np.maximum(cnt, 1))
    lo = hi = np.full(k, np.nan)
    if reps > 0:
        idx = _circular_blocks(len(x), block, reps, np.random.default_rng(seed))
        key = (np.arange(reps)[:, None] * k + lab[idx]).ravel()
        s = np.bincount(key, e2[idx].ravel(), reps * k).reshape(reps, k)
        c = np.bincount(key, minlength=reps * k).reshape(reps, k)
        with np.errstate(invalid="ignore", divide="ignore"):
            boot = np.sqrt(np.where(c > 0, s / np.maximum(c, 1), np.nan))
        lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)
    return np.column_stack([[x[p].mean() for p in parts], [x[p].min() for p in parts],
                            [x[p].max() for p in parts], rms, lo, hi, cnt])


def spearman_diff_ci(a, b, target, *, block: int, reps: int = 200, seed: int = 0):
    """95% circular-block-bootstrap interval of ``spearman(b, target) - spearman(a, target)``.

    Ranks are recomputed inside every resample (average ranks for the repeated samples)."""
    from scipy.stats import rankdata

    target = np.asarray(target, float)
    n = len(target)
    ga, gb, gt = (rankdata(np.asarray(v, float), method="dense").astype(np.int64) - 1 for v in (a, b, target))
    idx = _circular_blocks(n, block, reps, np.random.default_rng(seed))
    off = np.arange(reps)[:, None] * n

    def ranks(g):
        # average rank inside the resample = cum - (cnt - 1) / 2, whose mean is always (n + 1) / 2;
        # returned doubled and centred (a correlation does not see the scale)
        flat = (off + g[idx]).ravel()
        cnt = np.bincount(flat, minlength=reps * n)
        cum = np.cumsum(cnt.reshape(reps, n), axis=1).ravel()
        return (cum[flat] * 2.0 - cnt[flat] - n).reshape(reps, n)

    def corr(p, q):
        den = np.sqrt(np.einsum("ij,ij->i", p, p) * np.einsum("ij,ij->i", q, q))
        return np.where(den > 0, np.einsum("ij,ij->i", p, q) / np.where(den > 0, den, 1), np.nan)

    rt = ranks(gt)
    d = corr(ranks(gb), rt) - corr(ranks(ga), rt)
    lo, hi = np.nanpercentile(d, [2.5, 97.5])
    return float(lo), float(hi)


def realised_vol_scale(frame, windows=None, rv_bars: Optional[int] = None):
    """(u per horizon, closes per window) of the trailing realised-vol baseline, or (None, 0).

    ``u = std(1-bar close changes over the window) * sqrt(horizon bars)`` from the raw input windows
    (``windows``, default ``frame.X_raw``); only bars up to the sample's own close, so it is causal.
    ``rv_bars``: use only the last ``rv_bars`` closes of each window (default: the whole window).
    """
    explicit = windows is not None
    W = windows if explicit else getattr(frame, "X_raw", None)
    if W is None:
        return None, 0
    W = np.asarray(W, float)
    if W.ndim == 3 and W.shape[2] == 1:
        W = W[..., 0]
    if W.ndim != 2 or len(W) != len(frame) or W.shape[1] < 3:
        if explicit:
            raise ValueError(f"windows must be [N={len(frame)}, bars] raw close windows, got shape {W.shape}")
        return None, 0
    if rv_bars:
        if int(rv_bars) < 3:
            raise ValueError(f"rv_bars must be at least 3, got {rv_bars}")
        W = W[:, -int(rv_bars):]
    from neural_trade.calibration.conformal import interval_scale

    return interval_scale("realized_vol", windows=W, horizon_steps=frame.horizon_steps), int(W.shape[1])


def variance_summary(frame, *, raw_delta: Optional[Dict[str, np.ndarray]] = None, windows=None,
                     rv_bars: Optional[int] = None, seed: int = 0) -> Dict[str, dict]:
    """The numbers the variance figure prints, per horizon.

    ``z = (y - served delta) / sigma``. Coverage and width are over the whole block and its two
    halves (first, second). ``spearman_*`` are rank correlations with the squared error, as in
    ``eval_report`` (``corr_var_err2_spearman``).
    """
    from scipy.stats import kurtosis, norm, spearmanr

    n = len(frame)
    u, rv_len = realised_vol_scale(frame, windows, rv_bars)
    half = n // 2
    out = {}
    for i, h in enumerate(T.HORIZONS):
        steps = S.horizon_steps(frame, h)
        sig, mu, y = frame.sigma(h), np.asarray(frame.delta[h], float), frame.y[:, i]
        e = y - mu
        z = e / np.maximum(sig, 1e-12)
        std_z = float(z.std())
        r = dict(steps=steps, n=n, std_z=std_z, mean_abs_z=float(np.abs(z).mean()),
                 # shape alone (scale removed): sqrt(2 / pi) = 0.798 for any Gaussian
                 mean_abs_z_over_std=float(np.abs(z).mean() / std_z) if std_z > 0 else np.nan,
                 excess_kurtosis=float(kurtosis(z)), p_z3=float(np.mean(np.abs(z) > 3)),
                 p_z258=float(np.mean(np.abs(z) > 2.5758293)),
                 gauss_p_z3=float(2 * norm.sf(3)), gauss_p_z258=0.01)
        r["spearman_head"] = float(spearmanr(sig, e ** 2).correlation) if np.ptp(sig) > 0 else np.nan
        if raw_delta is not None:
            er = y - np.asarray(raw_delta[h], float)
            r["spearman_raw"] = float(spearmanr(sig, er ** 2).correlation) if np.ptp(sig) > 0 else np.nan
        if u is not None:
            r["rv_bars"] = rv_len
            r["spearman_rv"] = float(spearmanr(u[h], e ** 2).correlation) if np.ptp(u[h]) > 0 else np.nan
            if np.ptp(sig) > 0 and np.ptp(u[h]) > 0:
                r["rv_minus_head_ci"] = spearman_diff_ci(sig, u[h], e ** 2, seed=seed,
                                                         block=max(RANK_BOOT_BLOCK, BOOT_BLOCK_HORIZONS * steps))
        g_in = np.abs(e) <= Z90 * sig
        g_w = 2 * Z90 * sig
        r.update(cov_gauss=float(g_in.mean()), cov_gauss_halves=(float(g_in[:half].mean()), float(g_in[half:].mean())),
                 width_gauss=float(g_w.mean()), width_gauss_halves=(float(g_w[:half].mean()), float(g_w[half:].mean())))
        if frame.intervals is not None and h in frame.intervals:
            lo, hi = (np.asarray(a, float) for a in frame.intervals[h])
            c_in, c_w = (y >= lo) & (y <= hi), hi - lo
            r.update(cov_conf=float(c_in.mean()), cov_conf_halves=(float(c_in[:half].mean()), float(c_in[half:].mean())),
                     width_conf=float(c_w.mean()),
                     width_conf_halves=(float(c_w[:half].mean()), float(c_w[half:].mean())))
        out[h] = r
    return out


# ------------------------------------------------------------------ drawing helpers
def _thin_line(x, y):
    """x0 / dx / float32 y of an evenly spaced series, keeping at most MAX_LINE_POINTS points."""
    x, y = np.asarray(x), np.asarray(y, float)
    step = max(1, -(-len(y) // MAX_LINE_POINTS))
    return dict(x0=int(x[0]) if len(x) else 0, dx=step, y=y[::step].astype(np.float32))


def _key(fig, row, legend, name, group, *, line=None, marker=None, mode="lines"):
    """A legend-only entry in neutral ink: the key shows the style, not one horizon's colour."""
    import plotly.graph_objects as go

    fig.add_trace(go.Scatter(x=[None], y=[None], mode=mode, name=name, legendgroup=group, legend=legend,
                             line=line, marker=marker, hoverinfo="skip"), row, 1)


def _signed(v, fmt="+.3f"):
    """Signed number with a real minus sign; a value that rounds to zero carries no sign."""
    if v is None or not np.isfinite(v):
        return "n/a"
    s = format(v, fmt)
    if set(ch for ch in s if ch.isdigit()) <= {"0"}:
        s = s.lstrip("+-")
    return s.replace("-", "−")


def _pct(v):
    return "n/a" if not np.isfinite(v) else f"{100 * v:.2f}%"


def _log_ticks(lo: float, hi: float, panel_px: float, min_px: float = 14.0) -> list:
    """Tick values of :data:`_LOG_TICKS` inside [lo, hi] on a log axis ``panel_px`` tall, at least
    ``min_px`` apart (so the labels never overprint on a short panel); 1 is always kept."""
    per_decade = panel_px / max(float(np.log10(hi / lo)), 1e-9)
    cand = [v for v in _LOG_TICKS if lo <= v <= hi]
    keep = [1.0] if lo <= 1 <= hi else []
    for side in (sorted(v for v in cand if v > 1), sorted((v for v in cand if v < 1), reverse=True)):
        last = 1.0
        for v in side:
            if abs(np.log10(v / last)) * per_decade >= min_px:
                keep.append(v)
                last = v
    return sorted(keep)


# ------------------------------------------------------------------ figure
def variance_analytics_figure(frame, config=None, *, window: int = 500, height: Optional[int] = None,
                              raw_delta: Optional[Dict[str, np.ndarray]] = None, windows=None,
                              rv_bars: Optional[int] = None, n_bins: int = 12, seed: int = 0):
    """Variance heads on one block (see the module docstring for the five rows).

    ``raw_delta``: the price heads before delta shrinkage (the variance head's training target); adds
    the RMS error around them to row 1. ``windows``: raw close windows [N, bars] for the trailing
    realised-vol baseline (default ``frame.X_raw``; no baseline when neither is present).
    ``rv_bars``: closes of each window the baseline uses (default: all). ``window``: trailing samples of
    rows 4-5. ``height``: figure height in px (default :data:`DEFAULT_HEIGHT`); values below
    :data:`MIN_HEIGHT` are raised, because the row headings keep a fixed pixel gap between the rows.
    The text is laid out to fit an output area of 1000 px or wider.
    """
    import plotly.graph_objects as go
    from scipy.stats import norm

    n = len(frame)
    summ = variance_summary(frame, raw_delta=raw_delta, windows=windows, rv_bars=rv_bars, seed=seed)
    u, rv_len = realised_vol_scale(frame, windows, rv_bars)
    has_iv = frame.intervals is not None and all(h in frame.intervals for h in T.HORIZONS)
    w = int(min(window, max(2, n // 2)))
    bar_min = int(getattr(config, "RESAMPLE_MINUTES", 1) or 1) if config is not None else 1
    hours = w * bar_min / 60.0
    span = f"~{hours:.0f} h" if hours >= 1.5 else f"~{w * bar_min:.0f} min"

    # ---- panel titles: two short lines each (they must fit a 1000 px output area), horizon in colour
    def head(h, text, sub):
        c = T.HORIZON_COLORS[h]
        return (f"<span style='color:{c}'><b>{h}</b></span> {text}"
                f"<br><span style='font-size:11px;color:{T.MUTED}'>{sub}</span>")

    titles = []
    for h in T.HORIZONS:
        r = summ[h]
        lab = f"<span style='color:{T.HORIZON_COLORS[h]}'>({r['steps']} bars)</span>"
        if "spearman_rv" in r:
            ci = r.get("rv_minus_head_ci", (np.nan, np.nan))
            titles.append(head(h, f"{lab} ρ: head {_signed(r['spearman_head'])} · "
                                  f"{r['rv_bars']}-bar vol {_signed(r['spearman_rv'])}",
                               f"vol − head {_signed(r['spearman_rv'] - r['spearman_head'])}, "
                               f"95% CI {_signed(ci[0])} to {_signed(ci[1])}"))
        else:
            titles.append(head(h, f"{lab} ρ: head {_signed(r['spearman_head'])}",
                               "no raw input windows: no realised-vol baseline"))
    for h in T.HORIZONS:
        r = summ[h]
        titles.append(head(h, f"std z {r['std_z']:.2f} · excess kurtosis {_signed(r['excess_kurtosis'], '+.2f')}",
                           f"shape: mean |z| ÷ std z {r['mean_abs_z_over_std']:.2f} (Gaussian 0.80)"))
    for h in T.HORIZONS:
        r = summ[h]
        ratio = r["p_z3"] / r["gauss_p_z3"]
        titles.append(head(h, f"|z| > 3: {_pct(r['p_z3'])} vs {_pct(r['gauss_p_z3'])} Gaussian ({ratio:.1f}×)",
                           f"|z| > 2.58: {_pct(r['p_z258'])} vs 1.00%"))
    for h in T.HORIZONS:
        r = summ[h]
        g0, g1 = r["cov_gauss_halves"]
        if "cov_conf" in r:
            c0, c1 = r["cov_conf_halves"]
            titles.append(head(h, f"coverage: conformal {r['cov_conf']:.3f} · Gaussian {r['cov_gauss']:.3f}",
                               f"halves: conformal {c0:.3f}→{c1:.3f} · Gaussian {g0:.3f}→{g1:.3f}"))
        else:
            titles.append(head(h, f"coverage: Gaussian {r['cov_gauss']:.3f}",
                               f"halves: {g0:.3f}→{g1:.3f} (no conformal intervals)"))
    for h in T.HORIZONS:
        r = summ[h]
        g0, g1 = r["width_gauss_halves"]
        if "width_conf" in r:
            c0, c1 = r["width_conf_halves"]
            titles.append(head(h, f"mean width: conformal ${r['width_conf']:,.0f} · Gaussian "
                                  f"${r['width_gauss']:,.0f}",
                               f"2nd half vs 1st: conformal {_signed(c1 / c0 - 1, '+.0%')} · Gaussian "
                               f"{_signed(g1 / g0 - 1, '+.0%')}"))
        else:
            titles.append(head(h, f"mean width: Gaussian ${r['width_gauss']:,.0f}",
                               f"2nd half vs 1st: {_signed(g1 / g0 - 1, '+.0%')}"))

    # ---- figure subtitle: short lines (each fits a 1000 px output area)
    sub_lines = [
        f"{frame.split} block, {n:,} samples · error = y − served delta (β × raw head), as in eval_report "
        "· z = error / σ",
        "σ was trained on the raw head's errors · ρ = rank correlation (Spearman) with the squared error",
        "RMS on the diagonal + PIT hump + tail ratio above 1 = tails heavier than a Gaussian, not a too-wide σ.",
        "σ too wide: RMS below the diagonal and std z < 1 · too narrow: U-shaped PIT and std z > 1"]
    if frame.split == "cal":
        sub_lines.insert(1, "conformal coverage is 90% by construction on this block (the conformal scale "
                            "was fitted on it)")

    # ---- vertical layout in pixels: the gap between rows is fixed, whatever the height
    margin_t = _TITLE_PX + _SUB_LINE_PX * len(sub_lines) + _HEAD_PX + _TITLES_PX + 8
    extra = margin_t - _MARGIN_T                   # a fifth subtitle line on the cal block
    height = max(int(height) if height else DEFAULT_HEIGHT + extra, MIN_HEIGHT + extra)
    plot_h = height - margin_t - _MARGIN_B
    fig = _grid(_ROWS, titles, vspace=_GAP_PX / plot_h, row_heights=list(_ROW_HEIGHTS))
    for a in fig.layout.annotations:
        a.update(font=dict(color=T.INK_2, size=12))

    rv_name = f"baseline: {rv_len}-bar realised vol × √horizon"
    tail_lo, tail_hi = [], []
    for j, h in enumerate(T.HORIZONS, start=1):
        i, c = j - 1, T.HORIZON_COLORS[h]
        r = summ[h]
        steps = r["steps"]
        sig, mu, y = frame.sigma(h), np.asarray(frame.delta[h], float), frame.y[:, i]
        e = y - mu
        z = e / np.maximum(sig, 1e-12)
        block = BOOT_BLOCK_HORIZONS * steps
        lags = LRV_LAG_HORIZONS * steps

        # 1: RMS error per sigma bin (+ baseline, + raw-head view)
        t = binned_rms(sig, e, n_bins, block=block, seed=seed + j)
        xs = [t[:, 0]]
        fig.add_trace(go.Scatter(
            x=t[:, 0], y=t[:, 3], mode="lines+markers", name="RMS error per σ bin", legendgroup="rms",
            showlegend=False, line=dict(color=c, width=2), marker=dict(size=6, color=c),
            error_y=dict(type="data", symmetric=False, array=t[:, 5] - t[:, 3], arrayminus=t[:, 3] - t[:, 4],
                         thickness=1, width=3, color=c),
            customdata=np.column_stack([t[:, 1], t[:, 2], t[:, 4], t[:, 5], t[:, 6]]),
            hovertemplate="predicted σ bin %{customdata[0]:$,.0f} to %{customdata[1]:$,.0f} "
                          "(mean %{x:$,.0f})<br>RMS error %{y:$,.1f} (95% CI %{customdata[2]:$,.0f} to "
                          "%{customdata[3]:$,.0f}, block bootstrap)<br>n %{customdata[4]:,.0f}<extra></extra>"),
            1, j)
        if raw_delta is not None:
            tr = binned_rms(sig, y - np.asarray(raw_delta[h], float), n_bins, reps=0)
            fig.add_trace(go.Scatter(
                x=tr[:, 0], y=tr[:, 3], mode="lines+markers", name="RMS error around the raw price head",
                legendgroup="raw", showlegend=False, line=dict(color=c, width=1.5, dash=T.ALT_DASH),
                marker=dict(size=7, symbol="diamond-open", color=c),
                hovertemplate="predicted σ %{x:$,.0f}<br>RMS of y − raw head %{y:$,.1f}<br>"
                              f"rank corr(σ, err²) around the raw head {_signed(r['spearman_raw'])}"
                              "<extra></extra>"), 1, j)
        if u is not None:
            tb = binned_rms(u[h], e, n_bins, reps=0)
            xs.append(tb[:, 0])
            fig.add_trace(go.Scatter(
                x=tb[:, 0], y=tb[:, 3], mode="lines+markers", name=rv_name, legendgroup="rv", showlegend=False,
                line=dict(color=T.NEUTRAL, width=1.25), marker=dict(size=5, symbol="square", color=T.NEUTRAL),
                customdata=np.column_stack([tb[:, 1], tb[:, 2]]),
                hovertemplate=f"trailing {rv_len}-bar realised vol × √{steps} bars<br>"
                              "forecast bin %{customdata[0]:$,.0f} to %{customdata[1]:$,.0f} "
                              "(mean %{x:$,.0f})<br>RMS error %{y:$,.1f}<extra></extra>"), 1, j)
        lo_x = float(min(a.min() for a in xs))
        hi_x = float(max(a.max() for a in xs))
        fig.add_trace(go.Scatter(x=[lo_x, hi_x], y=[lo_x, hi_x], mode="lines", name="calibrated: RMS error = σ",
                                 legendgroup="diag", showlegend=False,
                                 line=dict(color=T.NEUTRAL, dash="dot", width=1), hoverinfo="skip"), 1, j)

        # 2: PIT, pre-binned, with the range a calibrated sigma shows from noise as two thin grey edges
        # over the bars (a filled band over opaque bars would tint them two-tone)
        pit = norm.cdf(z)
        counts, edges = np.histogram(pit, bins=20, range=(0.0, 1.0))
        width = edges[1] - edges[0]
        dens = counts / max(n, 1) / width
        vif = float(np.mean([lrv_factor((pit >= a) & (pit < b), lags) for a, b in zip(edges[:-1], edges[1:])]))
        hw = 1.96 * np.sqrt(width * (1 - width) * vif / max(n, 1)) / width
        fig.add_trace(go.Bar(x=(edges[:-1] + width / 2).astype(np.float32), y=dens.astype(np.float32), width=width,
                             name="PIT density", legendgroup="pit", showlegend=False,
                             marker=dict(color=c, line=dict(width=0)),
                             customdata=np.column_stack([edges[:-1], edges[1:], counts]),
                             hovertemplate="PIT %{customdata[0]:.2f} to %{customdata[1]:.2f}<br>density %{y:.2f}"
                                           "<br>n %{customdata[2]:,.0f}<extra></extra>"), 2, j)
        fig.add_trace(go.Scatter(x=[0, 1], y=[1, 1], mode="lines", name="uniform", legendgroup="uniform",
                                 showlegend=False, line=dict(color=T.NEUTRAL, dash="dot", width=1),
                                 hovertemplate="uniform density 1<extra></extra>"), 2, j)
        fig.add_trace(go.Scatter(
            x=[0, 1, None, 0, 1], y=[max(0.0, 1 - hw), max(0.0, 1 - hw), None, 1 + hw, 1 + hw], mode="lines",
            name="noise range if calibrated", legendgroup="pit-band", showlegend=False,
            line=dict(color=T.NEUTRAL, dash=_BAND_EDGE_DASH, width=1),
            hovertemplate=f"a calibrated σ keeps each bin inside 1 ± {hw:.2f} (95%, overlap factor {vif:.1f})"
                          "<extra></extra>"), 2, j)

        # 3: tails - observed rate of |z| > k over the Gaussian rate, with 95% CIs on the effective count
        p0 = 2 * norm.sf(K_GRID)
        obs = np.array([np.mean(np.abs(z) > k) for k in K_GRID])
        ne = np.array([n / lrv_factor(np.abs(z) > k, lags) for k in K_GRID])
        _, wlo, whi = S.wilson(obs * ne, ne)
        ratio = T.positive(obs / p0)
        fig.add_trace(go.Scatter(
            x=K_GRID, y=ratio, mode="lines+markers", name="observed ÷ Gaussian", legendgroup="tail",
            showlegend=False, line=dict(color=c, width=2), marker=dict(size=6, color=c),
            error_y=dict(type="data", symmetric=False, array=np.maximum(whi / p0 - obs / p0, 0),
                         arrayminus=np.maximum(obs / p0 - np.maximum(wlo, 1e-12) / p0, 0),
                         thickness=1, width=3, color=c),
            customdata=np.column_stack([100 * obs, 100 * p0, wlo / p0, whi / p0]),
            hovertemplate="|z| > %{x:.2f}: %{customdata[0]:.2f}% of samples vs %{customdata[1]:.2f}% Gaussian"
                          "<br>ratio %{y:.2f} (95% CI %{customdata[2]:.2f} to %{customdata[3]:.2f})"
                          "<extra></extra>"), 3, j)
        tail_lo += [float(v) for v in (wlo / p0) if v > 0]
        tail_hi += [float(v) for v in (whi / p0) if v > 0]
        fig.add_trace(go.Scatter(x=[float(K_GRID[0]), float(K_GRID[-1])], y=[1, 1], mode="lines",
                                 name="Gaussian rate (ratio 1)", legendgroup="gauss1", showlegend=False,
                                 line=dict(color=T.NEUTRAL, dash="dot", width=1), hoverinfo="skip"), 3, j)
        if "cov_conf" in r:
            miss = 1 - r["cov_conf"]
            tail_lo.append(miss / 0.10)
            tail_hi.append(miss / 0.10)
            fig.add_trace(go.Scatter(
                x=[Z90], y=T.positive([miss / 0.10]), mode="markers", name="conformal 90% interval", legendgroup="conf",
                showlegend=False, marker=dict(size=10, symbol="diamond", color=c, line=dict(color=T.PAPER, width=1)),
                hovertemplate=f"conformal 90% interval misses {100 * miss:.2f}% (nominal 10%)<br>"
                              "ratio %{y:.2f}<extra></extra>"), 3, j)

        # 4: trailing coverage with the range expected from noise around 0.90
        g_in = (np.abs(e) <= Z90 * sig).astype(float)
        c_in = None
        if has_iv:
            lo, hi = (np.asarray(a, float) for a in frame.intervals[h])
            c_in = ((y >= lo) & (y <= hi)).astype(float)
        # one band for both lines: the wider of the two indicators' overlap factors
        vif_c = max(lrv_factor(v, lags) for v in ((g_in,) if c_in is None else (c_in, g_in)))
        hw_c = 1.96 * np.sqrt(0.09 * vif_c / w)
        fig.add_hrect(y0=0.9 - hw_c, y1=min(1.0, 0.9 + hw_c), fillcolor=_BAND, line_width=0, layer="below",
                      exclude_empty_subplots=False, row=4, col=j)
        fig.add_trace(go.Scatter(x=[w - 1, n - 1], y=[0.9, 0.9], mode="lines", name="90% target", legendgroup="target",
                                 showlegend=False, line=dict(color=T.NEUTRAL, dash="dot", width=1),
                                 hovertemplate=f"90% target; noise range ±{hw_c:.3f} (overlap factor {vif_c:.1f}). "
                                               "Pointwise 95%: over a long block the<br>trailing line leaves "
                                               "it now and then by chance alone<extra></extra>"), 4, j)
        if c_in is not None:
            x, cov = _rolling_mean(c_in, w)
            fig.add_trace(go.Scatter(**_thin_line(x, cov), mode="lines", name="conformal interval", legendgroup="conf-cov",
                                     showlegend=False, line=dict(color=c, width=1.5),
                                     hovertemplate="sample %{x}<br>conformal coverage %{y:.3f}<extra></extra>"), 4, j)
        x, covg = _rolling_mean(g_in, w)
        fig.add_trace(go.Scatter(**_thin_line(x, covg), mode="lines", name="Gaussian ±1.645σ",
                                 legendgroup="gauss-cov", showlegend=False,
                                 line=dict(color=c, width=1.5, dash=T.ALT_DASH),
                                 hovertemplate="sample %{x}<br>Gaussian coverage %{y:.3f}<extra></extra>"), 4, j)

        # 5: trailing mean interval width
        if has_iv:
            x, wc = _rolling_mean(hi - lo, w)
            fig.add_trace(go.Scatter(**_thin_line(x, wc), mode="lines", name="conformal width", legendgroup="conf-cov",
                                     showlegend=False, line=dict(color=c, width=1.5),
                                     hovertemplate="sample %{x}<br>conformal width %{y:$,.0f}<extra></extra>"), 5, j)
        x, wg = _rolling_mean(2 * Z90 * sig, w)
        fig.add_trace(go.Scatter(**_thin_line(x, wg), mode="lines", name="Gaussian width", legendgroup="gauss-cov",
                                 showlegend=False, line=dict(color=c, width=1.5, dash=T.ALT_DASH),
                                 hovertemplate="sample %{x}<br>Gaussian width 2×1.645σ %{y:$,.0f}"
                                               "<extra></extra>"), 5, j)
        fig.update_xaxes(matches=f"x{12 + j}", row=4, col=j)

        fig.update_xaxes(title_text="predicted σ or baseline forecast ($), bin mean" if u is not None
                         else "predicted σ ($), bin mean", row=1, col=j)
        fig.update_xaxes(title_text="PIT = Φ((y − served delta) / σ)", range=[0, 1], row=2, col=j)
        fig.update_xaxes(title_text="k (multiples of σ)", row=3, col=j)
        fig.update_xaxes(title_text="sample (time order)", row=4, col=j)
        fig.update_xaxes(title_text="sample (time order)", row=5, col=j)

    # ---- axes: y titles in column 1, one shared scale per row where horizons are comparable
    fig.update_yaxes(title_text="RMS of y − served delta ($)", row=1, col=1)
    fig.update_yaxes(title_text="density", row=2, col=1)
    fig.update_yaxes(title_text="observed ÷ Gaussian", row=3, col=1)
    fig.update_yaxes(title_text="share inside interval", row=4, col=1)
    fig.update_yaxes(title_text="mean width ($)", row=5, col=1)
    for row in (2, 4):
        for col in (2, 3):
            fig.update_yaxes(matches=f"y{(row - 1) * 3 + 1}", row=row, col=col)
    # row 3: one fixed log range for all three columns (the shared scale), always reaching below 1x so
    # the sub-Gaussian rates at k ~ 1 (a peaked centre) read off a tick. No `matches` here: in
    # plotly.js 3.5 a fixed range on a matched group switches off the autorange of the row-4 group.
    lo_r = min([0.5] + [0.85 * v for v in tail_lo])
    hi_r = max([2.0] + [1.2 * v for v in tail_hi])
    yd3 = fig.get_subplot(3, 1).yaxis.domain
    ticks = _log_ticks(lo_r, hi_r, (yd3[1] - yd3[0]) * plot_h)
    for col in (1, 2, 3):
        fig.update_yaxes(type="log", range=[float(np.log10(lo_r)), float(np.log10(hi_r))], autorange=False,
                         tickvals=ticks, ticktext=[f"{v:g}×" for v in ticks], row=3, col=col)
    fig.update_xaxes(title_standoff=6)

    # ---- per-row legends (neutral swatches, the key names the style); the legend title is the row
    # heading, on its own line above the keys (a heading beside the keys is not counted when plotly
    # wraps the keys, which cut them off in narrow outputs)
    ink = T.INK_2
    _key(fig, 1, "legend", "RMS error per σ bin, 95% CI", "rms", mode="lines+markers",
         line=dict(color=ink, width=2), marker=dict(size=6, color=ink))
    if u is not None:
        _key(fig, 1, "legend", rv_name, "rv", mode="lines+markers", line=dict(color=T.NEUTRAL, width=1.25),
             marker=dict(size=5, symbol="square", color=T.NEUTRAL))
    if raw_delta is not None:
        _key(fig, 1, "legend", "RMS around raw head (training target)", "raw", mode="lines+markers",
             line=dict(color=ink, width=1.5, dash=T.ALT_DASH), marker=dict(size=7, symbol="diamond-open", color=ink))
    _key(fig, 1, "legend", "calibrated: RMS = σ", "diag", line=dict(color=T.NEUTRAL, dash="dot", width=1))
    _key(fig, 2, "legend2", "PIT density (20 bins)", "pit", mode="markers",
         marker=dict(symbol="square", size=11, color=ink))
    _key(fig, 2, "legend2", "noise range if calibrated (95% per bin)", "pit-band",
         line=dict(color=T.NEUTRAL, dash=_BAND_EDGE_DASH, width=1))
    _key(fig, 2, "legend2", "uniform", "uniform", line=dict(color=T.NEUTRAL, dash="dot", width=1))
    _key(fig, 3, "legend3", "observed ÷ Gaussian, 95% CI", "tail", mode="lines+markers",
         line=dict(color=ink, width=2), marker=dict(size=6, color=ink))
    if has_iv:
        _key(fig, 3, "legend3", "conformal 90% interval: miss rate ÷ 10%", "conf", mode="markers",
             marker=dict(size=10, symbol="diamond", color=ink))
    _key(fig, 3, "legend3", "Gaussian rate (ratio 1)", "gauss1", line=dict(color=T.NEUTRAL, dash="dot", width=1))
    if has_iv:
        _key(fig, 4, "legend4", "conformal interval", "conf-cov", line=dict(color=ink, width=1.5))
    _key(fig, 4, "legend4", "Gaussian ±1.645σ around served delta", "gauss-cov",
         line=dict(color=ink, width=1.5, dash=T.ALT_DASH))
    _key(fig, 4, "legend4", "noise range (pointwise 95%)", "cov-band", mode="markers",
         marker=dict(symbol="square", size=11, color=_BAND))
    _key(fig, 4, "legend4", "90% target", "target", line=dict(color=T.NEUTRAL, dash="dot", width=1))
    # every trace joins its row's legend, so a key toggles that series in all three columns (row 5
    # shares row 4's keys)
    for tr in fig.data:
        row = (int((tr.yaxis or "y")[1:] or 1) - 1) // 3 + 1
        tr.legend = ("legend", "legend2", "legend3", "legend4", "legend4")[row - 1]

    T.apply(fig, title="Variance heads", subtitle="<br>".join(sub_lines), height=height)
    fig.update_layout(bargap=0, margin=dict(t=margin_t, b=_MARGIN_B))

    # row headings: each row's legend (heading on top, keys below) sits just above its panel titles
    headings = {
        "legend": "1 · Does σ size and rank the error? (95% CIs: block bootstrap)",
        "legend2": "2 · Error shape: PIT histogram (flat = calibrated; Gaussian z: std 1, excess kurtosis 0)",
        "legend3": "3 · Tails: rate of |z| > k relative to a Gaussian (log scale)",
        "legend4": f"4 · 90% interval coverage, trailing {w} samples ({bar_min}-min bars, {span})",
    }
    lift = (_TITLES_PX + 2) / plot_h
    for row, lid in enumerate(("legend", "legend2", "legend3", "legend4"), start=1):
        yd = fig.get_subplot(row, 1).yaxis.domain
        fig.update_layout({lid: dict(
            title=dict(text=f"<b>{headings[lid]}</b>", side="top", font=dict(color=T.INK, size=12)),
            orientation="h", x=0.0, xanchor="left", y=yd[1] + lift, yanchor="bottom",
            # itemsizing "trace": each key shows the drawn line width and dash (with "constant" plotly
            # draws every key line 5 px wide, which turns a thin dashed edge into a solid bar)
            font=dict(size=11, color=T.INK_2), bgcolor="rgba(0,0,0,0)", itemsizing="trace", itemwidth=30,
            groupclick="togglegroup", tracegroupgap=0)})
    yd = fig.get_subplot(5, 1).yaxis.domain
    styles = ("solid = conformal (hi − lo) · dash-dot = Gaussian 2 × 1.645σ" if has_iv
              else "dash-dot = Gaussian 2 × 1.645σ (no conformal intervals)")
    fig.add_annotation(x=0.0, xref="paper", xanchor="left", y=yd[1] + lift, yref="paper", yanchor="bottom",
                       showarrow=False, align="left",
                       text=f"<b>5 · Interval width, trailing {w} samples</b><br>"
                            f"<span style='font-size:11px;color:{T.INK_2}'>line styles as in row 4: {styles}</span>",
                       font=dict(color=T.INK, size=12))
    return fig
