"""Direction heads, one column per horizon (h0 / h1 / h2).

Rows:

1. the served (calibrated) P(up), split by the realised move: does it separate up moves from down moves?
2. the ROC drawn as lift over chance, TPR - FPR against FPR, for the direction head and the price head's
   Gaussian readout, inside the pointwise band a no-skill head stays in 95% of the time. The area under
   this curve is AUC - 0.5, so deviations of a few hundredths are visible (on a unit-square ROC they are not);
3. reliability of the raw and the calibrated P(up) on 10 equal-count bins, with the realised up-rate;
4. a scorecard: labelled and effective n, up-rates, accuracy, precision / recall, MCC, the head-vs-Gaussian
   AUC difference, Brier and ECE against a constant 0.5 (the ECE with the evaluation report's bins, so the
   numbers match eval_report_*.md), and how often the delta head agrees with the direction head on the sign.

Every direction number uses only the moves outside the deadband (``DIR_DEADBAND_BPS``), as the report does.

A frame without a fitted calibration (``direction_prob_calibrated`` is None, or equal to the raw head, as the
Predictor serves it with ``calibrated=False``) is labelled as such: P(up) is the raw head, the reliability row
draws one series and no raw -> calibrated shift, and nothing claims to be calibrated.

Uncertainty: consecutive 1-minute samples overlap (a horizon of h bars shares h-1 bars with its neighbour).

* AUC, accuracy, the head-vs-Gaussian AUC difference and the chance band use n / h effective samples
  (:mod:`neural_trade.visualization.stats`, the evaluation report's convention; conservative for AUC).
* The reliability bars are block-clustered: residuals are summed within contiguous blocks of ``ci_block``
  bars of the original time order. A decile's members are scattered in time, so treating each bin as
  n_bin / h samples would overstate the overlap about twofold; the clustered error matches a moving-block
  bootstrap with the report's 80-bar block.
"""
from __future__ import annotations

import numpy as np

from neural_trade.visualization import stats as S
from neural_trade.visualization import theme as T
from neural_trade.visualization.analytics_common import DOWN_COLOR, UP_COLOR, _labels

CI_BLOCK = 80            # bars per cluster for the reliability bars (= evaluation.report.BLOCK)
N_REL_BINS = 10          # equal-count reliability bins
# vertical layout (pixels): figure title + three subtitle lines, the gap between panel rows (tick labels,
# x title, the next row's heading and its two-line column titles), the heading's lift above its row
_TOP_PX, _BOTTOM_PX, _GAP_PX, _HEADING_PX = 160, 20, 128, 40
# the scorecard: header, one line per row, and room for two labels or values that wrap in a narrow container
_HEADER_PX, _ROW_PX, _WRAP_PX = 26, 22, 40
_SCORECARD_TITLE = "<b>Scorecard</b> · labelled samples · definitions as in the evaluation report"
_REF_DASH = "5px,4px"    # the realised up-rate line (dotted is reserved for training curves)
_SHIFT_WIDTH, _SHIFT_ALPHA = 3.5, 0.4   # raw -> calibrated connector: a thick translucent bar (dumbbell)
_NO_CAL = "no calibration applied"
_trapz = getattr(np, "trapezoid", None) or np.trapz


# ------------------------------------------------------------------ statistics
def roc_points(labels, scores, max_points: int = 400):
    """(fpr, tpr, threshold, auc): the ROC with ties handled, thinned to ``max_points`` for plotting.

    ``threshold[k]`` is the score at or above which a sample is called up at point k (inf at the origin).
    The AUC is computed on the full curve before thinning (equal to the Mann-Whitney AUC); it is NaN when
    one class is missing.
    """
    labels = np.asarray(labels, float)
    scores = np.asarray(scores, float)
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.zeros(1), np.zeros(1), np.full(1, np.inf), float("nan")
    order = np.argsort(-scores, kind="mergesort")
    s, y = scores[order], labels[order]
    distinct = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    tps = np.cumsum(y)[distinct]
    fps = (distinct + 1) - tps
    tpr, fpr = np.r_[0.0, tps / n_pos], np.r_[0.0, fps / n_neg]
    thr = np.r_[np.inf, s[distinct]]
    auc = float(_trapz(tpr, fpr))
    keep = S.thin(len(fpr), max_points)
    return fpr[keep], tpr[keep], thr[keep], auc


def _placements(labels, scores):
    """DeLong placement values: per positive, the share of negatives it outranks; per negative, the
    share of positives that outrank it (ties count one half). Both average to the AUC."""
    from scipy.stats import rankdata

    pos = np.asarray(labels) > 0.5
    n1, n0 = int(pos.sum()), int((~pos).sum())
    r = rankdata(scores)
    v10 = (r[pos] - rankdata(scores[pos])) / n0
    v01 = 1.0 - (r[~pos] - rankdata(scores[~pos])) / n1
    return v10, v01


def auc_difference(labels, scores_a, scores_b, *, steps: int = 1):
    """(AUC_a - AUC_b, lo, hi): paired DeLong 95% interval on n / steps effective samples per class."""
    labels = np.asarray(labels, float)
    n1 = int((labels > 0.5).sum())
    n0 = len(labels) - n1
    if n1 < 2 or n0 < 2:
        return float("nan"), float("nan"), float("nan")
    a10, a01 = _placements(labels, np.asarray(scores_a, float))
    b10, b01 = _placements(labels, np.asarray(scores_b, float))
    d = float(a10.mean() - b10.mean())
    var = np.var(a10 - b10, ddof=1) / S.n_eff(n1, steps) + np.var(a01 - b01, ddof=1) / S.n_eff(n0, steps)
    half = S.Z95 * float(np.sqrt(max(var, 0.0)))
    return d, d - half, d + half


def reliability_rows(labels, probs, t_index=None, *, block: int = CI_BLOCK, n_bins: int = N_REL_BINS,
                     steps: int = 1):
    """Equal-count bins of P(up): rows of (mean P(up), observed up-rate, n, lo, hi, se).

    ``t_index`` is each sample's position in the original time order (``np.flatnonzero(mask)`` after the
    deadband mask). The 95% interval uses a block-clustered standard error: the residuals y - ybar of the
    bin are summed within contiguous blocks of ``block`` bars, SE = sqrt(sum of squared block sums) / n.
    It is never narrower than the i.i.d. binomial error. A bin whose labels are all equal falls back to a
    Wilson interval on n / steps effective samples.
    """
    labels, probs = np.asarray(labels, float), np.asarray(probs, float)
    t_index = np.arange(len(labels)) if t_index is None else np.asarray(t_index)
    order = np.argsort(probs, kind="mergesort")
    rows = []
    for idx in np.array_split(order, n_bins):
        n = len(idx)
        if n == 0:
            continue
        y = labels[idx]
        m = float(y.mean())
        if 0.0 < m < 1.0:
            sums = np.bincount((t_index[idx] // max(1, int(block))).astype(np.int64), weights=y - m)
            se = max(float(np.sqrt(np.sum(sums * sums))) / n, float(np.sqrt(m * (1 - m) / n)))
            lo, hi = max(0.0, m - S.Z95 * se), min(1.0, m + S.Z95 * se)
        else:
            _, lo, hi = S.wilson(y.sum(), n, steps=steps)
            lo, hi = float(lo), float(hi)
            se = (hi - lo) / (2 * S.Z95)
        rows.append((float(probs[idx].mean()), m, n, lo, hi, se))
    return np.array(rows, float).reshape(-1, 6)


def table_ece(rows) -> float:
    """ECE of a reliability table: count-weighted mean |observed - predicted| over its bins."""
    rows = np.asarray(rows, float)
    if len(rows) == 0 or rows[:, 2].sum() <= 0:
        return float("nan")
    return float(np.average(np.abs(rows[:, 1] - rows[:, 0]), weights=rows[:, 2]))


def ece_noise_level(rows) -> float:
    """The table ECE a perfectly calibrated head would score from sampling noise alone:
    count-weighted E|N(0, se)| = se * sqrt(2 / pi) over the bins."""
    rows = np.asarray(rows, float)
    if len(rows) == 0 or rows[:, 2].sum() <= 0:
        return float("nan")
    return float(np.average(rows[:, 5] * np.sqrt(2 / np.pi), weights=rows[:, 2]))


def calibration_applied(frame) -> bool:
    """True when the frame serves a fitted calibration: a calibrated P(up) that differs from the raw head.

    ``direction_prob_calibrated`` is None when no calibration was fitted, and equal to the raw head when the
    Predictor serves raw probabilities (``calibrated=False``); ``frame.prob(h)`` is the raw head in both cases.
    """
    cal = getattr(frame, "direction_prob_calibrated", None)
    if cal is None:
        return False
    return any(h in cal and not np.array_equal(np.asarray(cal[h], float), np.asarray(frame.direction_prob[h], float))
               for h in T.HORIZONS)


def _rate(num, den):
    return float(num) / float(den) if den else float("nan")


def _hist_range(samples, *, fence: float = 3.0):
    """One axis for several P(up) samples: per sample the 0.5-99.5 percentile range, cut at the quartiles
    -/+ ``fence`` x IQR so a small heavy tail cannot squash the core; the union over the samples."""
    if not samples:
        return 0.4, 0.6
    lows, highs = [], []
    for p in samples:
        q005, q25, q75, q995 = np.percentile(p, [0.5, 25, 75, 99.5])
        iqr = q75 - q25
        lows.append(max(q005, q25 - fence * iqr))
        highs.append(min(q995, q75 + fence * iqr))
    return float(min(lows)), float(max(highs))


def _horizon_stats(frame, h, lab_mask, deadband, *, block, max_points):
    from neural_trade.metrics import numpy_metrics as npm

    lab_all, mask = lab_mask
    steps = S.horizon_steps(frame, h)
    t = np.flatnonzero(mask)
    lab = lab_all[mask]
    p_cal = np.asarray(frame.prob(h, True), float)[mask]
    p_raw = np.asarray(frame.direction_prob[h], float)[mask]
    g = np.asarray(frame.gauss_prob(h, deadband), float)[mask]
    n = int(len(lab))
    n_pos = int(lab.sum())
    n_neg = n - n_pos
    s = dict(lab=lab, t=t, p_cal=p_cal, p_raw=p_raw, g=g, n=n, n_pos=n_pos, n_neg=n_neg, steps=steps,
             n_eff=int(n // max(1, steps)), both=n_pos > 0 and n_neg > 0,
             # the calibration keeps the order of the raw head: its bins hold the same samples
             monotone=bool(n < 2 or np.all(np.diff(p_cal[np.argsort(p_raw, kind="mergesort")]) >= -1e-12)))
    # discrimination: ROC of the served P(up) (the report's AUC) and of the Gaussian readout
    for key, score in (("head", p_cal), ("gauss", g)):
        s[f"roc_{key}"] = roc_points(lab, score, max_points)
        auc = s[f"roc_{key}"][3]
        s[f"auc_{key}"] = auc
        s[f"ci_{key}"] = S.auc_ci(auc, n_pos, n_neg, steps=steps) if s["both"] else (np.nan, np.nan)
    s["auc_diff"] = auc_difference(lab, p_cal, g, steps=steps)
    # calibration: the drawn equal-count tables, their ECE and its noise level
    s["block"] = max(int(block), 2 * steps)
    s["rel_raw"] = reliability_rows(lab, p_raw, t, block=s["block"], steps=steps)
    s["rel_cal"] = reliability_rows(lab, p_cal, t, block=s["block"], steps=steps)
    s["ece_eq_raw"], s["ece_eq_cal"] = table_ece(s["rel_raw"]), table_ece(s["rel_cal"])
    s["ece_noise"] = ece_noise_level(s["rel_cal"])
    # scorecard (the evaluation report's definitions: P(up) > 0.5 calls up)
    called, up = p_cal > 0.5, lab > 0.5
    tp, fp, fn, tn = (int(np.sum(called & up)), int(np.sum(called & ~up)), int(np.sum(~called & up)),
                      int(np.sum(~called & ~up)))
    sens, spec = _rate(tp, tp + fn), _rate(tn, tn + fp)
    finite = [v for v in (sens, spec) if np.isfinite(v)]
    nan = float("nan")
    s.update(
        base=_rate(n_pos, n), called_up=_rate(tp + fp, n),
        acc=S.wilson(tp + tn, max(n, 1), steps=steps), bal_acc=float(np.mean(finite)) if finite else nan,
        precision=_rate(tp, tp + fp), recall=sens,
        mcc=npm.mcc(lab, p_cal) if n else nan,
        brier_cal=float(np.mean((p_cal - lab) ** 2)) if n else nan,
        brier_const=float(np.mean((0.5 - lab) ** 2)) if n else nan,
        brier_gauss=float(np.mean((g - lab) ** 2)) if n else nan,
        ece_rep_raw=npm.ece_pos(lab, p_raw), ece_rep_cal=npm.ece_pos(lab, p_cal),
        ece_rep_const=npm.ece_pos(lab, np.full(n, 0.5)), ece_rep_gauss=npm.ece_pos(lab, g),
        sign_agree=float(np.mean((np.asarray(frame.delta[h]) > 0) == (np.asarray(frame.prob(h, True)) > 0.5))),
    )
    return s


# ------------------------------------------------------------------ formatting
def _f(v, fmt):
    return "n/a" if v is None or not np.isfinite(v) else format(v, fmt)


def _verdict(lo, hi, what):
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return f"{what}: n/a"
    if lo > 0.5:
        return f"{what} above chance"
    if hi < 0.5:
        return f"{what} below chance"
    return f"{what} at chance"


def _small(text):
    return f"<span style='font-size:10.5px;color:{T.MUTED}'>{text}</span>"


# ------------------------------------------------------------------ figure
def direction_analytics_figure(frame, config=None, *, bins: int = 25, height: int = 1650,
                               ci_block: int = CI_BLOCK, max_points: int = 400):
    """Direction heads: P(up) by realised move, ROC lift over chance, reliability, scorecard.

    ``frame``: a PredictionFrame (served P(up) = ``frame.prob(h)``; raw head = ``frame.direction_prob``);
    ``config``: supplies ``DIR_DEADBAND_BPS`` and the horizon labels. ``bins``: histogram bins of row 1;
    ``ci_block``: contiguous bars per cluster for the reliability bars; ``max_points``: ROC points drawn.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    deadband = float(getattr(config, "DIR_DEADBAND_BPS", 0.0) or 0.0) if config is not None else 0.0
    labels = _labels(frame, config)
    fitted = calibration_applied(frame)
    served = "calibrated P(up)" if fitted else "raw P(up)"          # what frame.prob(h) returns
    st = {h: _horizon_stats(frame, h, labels[h], deadband, block=ci_block, max_points=max_points)
          for h in T.HORIZONS}

    # row 1: one P(up) axis for all three columns; the tails beyond it are folded into the end bins
    p_lo, p_hi = _hist_range([st[h]["p_cal"] for h in T.HORIZONS if st[h]["n"]])
    if not p_hi - p_lo > 1e-6:
        p_lo, p_hi = p_lo - 0.01, p_hi + 0.01
    edges = np.linspace(p_lo, p_hi, int(bins) + 1)
    mids = (edges[:-1] + edges[1:]) / 2

    # row 2: one lift range for all three columns (the chance band and both curves fit)
    f_grid = np.linspace(0, 1, 101)
    lift_max = 0.02
    for h in T.HORIZONS:
        s = st[h]
        s["band"] = np.zeros_like(f_grid)
        if s["both"]:
            s["band"] = S.Z95 * np.sqrt(f_grid * (1 - f_grid) * (1 / S.n_eff(s["n_pos"], s["steps"])
                                                                + 1 / S.n_eff(s["n_neg"], s["steps"])))
            lift_max = max(lift_max, float(s["band"].max()),
                           *(float(np.max(np.abs(s[k][1] - s[k][0]))) for k in ("roc_head", "roc_gauss")))
    lift_lim = 1.15 * lift_max

    # row 3: one reliability range for all three columns, symmetric around 0.5, from what is drawn
    # (bin means, observed rates and their bars, the realised up-rate), not from every single prediction
    vals = [0.5]
    for h in T.HORIZONS:
        s = st[h]
        for key in ("rel_raw", "rel_cal"):
            vals += list(s[key][:, 0]) + list(s[key][:, 1])
        vals += list(s["rel_cal"][:, 3]) + list(s["rel_cal"][:, 4])
        if np.isfinite(s["base"]):
            vals.append(s["base"])
    r = min(max(abs(v - 0.5) for v in vals) + 0.02, 0.52)
    rel_lo, rel_hi = 0.5 - r, 0.5 + r

    # titles: row 1 names the horizon and its sample sizes; rows 2-3 carry the numbers of the panel
    titles = []
    for h in T.HORIZONS:
        s, c = st[h], T.HORIZON_COLORS[h]
        out = float(np.mean((s["p_cal"] < p_lo) | (s["p_cal"] > p_hi))) if s["n"] else 0.0
        full = f"{s['p_cal'].min():.2f}-{s['p_cal'].max():.2f}" if s["n"] else "n/a"
        titles.append(f"<span style='color:{c}'>■</span> <b>{T.horizon_label(h, config)}</b>"
                      f" · n {s['n']:,} labelled (≈{s['n_eff']:,} effective)<br>"
                      + _small(f"{out:.1%} beyond the axis, in end bins (range {full})"))
    for h in T.HORIZONS:
        s = st[h]
        (lo_h, hi_h), (lo_g, hi_g) = s["ci_head"], s["ci_gauss"]
        both_chance = (np.isfinite(lo_h) and lo_h <= 0.5 <= hi_h and np.isfinite(lo_g) and lo_g <= 0.5 <= hi_g)
        verdict = ("head and Gaussian at chance: both 95% CIs include 0.5" if both_chance else
                   f"{_verdict(lo_h, hi_h, 'head')} · {_verdict(lo_g, hi_g, 'Gaussian')} (95% CI vs 0.5)")
        titles.append(f"AUC head {_f(s['auc_head'], '.3f')} [{_f(lo_h, '.2f')}, {_f(hi_h, '.2f')}]"
                      f" · Gaussian {_f(s['auc_gauss'], '.3f')} [{_f(lo_g, '.2f')}, {_f(hi_g, '.2f')}]<br>"
                      + _small(verdict))
    for h in T.HORIZONS:
        s = st[h]
        ece = (f"raw {_f(s['ece_eq_raw'], '.2f')} → calibrated {_f(s['ece_eq_cal'], '.2f')}" if fitted
               else f"{_f(s['ece_eq_cal'], '.2f')} ({_NO_CAL})")
        titles.append(f"ECE, {N_REL_BINS} equal-count bins: {ece}<br>"
                      + _small(f"a perfectly calibrated head scores ≈{_f(s['ece_noise'], '.2f')} from noise alone"))
    titles.append(_SCORECARD_TITLE)

    # vertical layout in pixels: under each panel its tick labels and x title, then the next row's heading
    # (a legend) and its two-line column titles; the scorecard table (one line per row) at the bottom
    score_rows = _scorecard_rows(deadband, fitted)
    top, bottom, gap_px = _TOP_PX, _BOTTOM_PX, _GAP_PX
    table_px = _HEADER_PX + len(score_rows) * _ROW_PX + _WRAP_PX
    plot_px = float(max(height - top - bottom, 3 * gap_px + table_px + 300))
    panel_px = (plot_px - 3 * gap_px - table_px) / 3
    fig = make_subplots(rows=4, cols=3, subplot_titles=titles, vertical_spacing=gap_px / plot_px,
                        horizontal_spacing=0.06, row_heights=[panel_px, panel_px, panel_px, table_px],
                        specs=[[{}, {}, {}], [{}, {}, {}], [{}, {}, {}], [{"type": "table", "colspan": 3}, None, None]])

    def f32(a):
        return np.asarray(a, dtype=np.float32)

    # row-1 hover: each bin's P(up) interval; the end bins also hold the folded tails
    bin_text = [f"{a:.3f}-{b:.3f}" for a, b in zip(edges[:-1], edges[1:])]
    bin_text[0] = f"≤ {edges[1]:.3f} (with the tail below the axis)"
    bin_text[-1] = f"≥ {edges[-2]:.3f} (with the tail above the axis)"
    bin_text = [bin_text[0], *bin_text, bin_text[-1]]

    for j, h in enumerate(T.HORIZONS, start=1):
        s, c = st[h], T.HORIZON_COLORS[h]
        first = j == 1
        # ---- row 1: served P(up) by realised class (full-width end bins: steps drawn edge to edge)
        x_step = np.r_[edges[0], mids, edges[-1]]
        for cls, name, color, sym in ((1, "realised up", UP_COLOR, "triangle-up"),
                                      (0, "realised down", DOWN_COLOR, "triangle-down")):
            x = np.clip(s["p_cal"][s["lab"] == cls], p_lo, p_hi)
            dens = np.histogram(x, bins=edges, density=True)[0] if len(x) else np.zeros(len(mids))
            fig.add_trace(go.Scatter(
                x=f32(x_step), y=f32(np.r_[dens[0], dens, dens[-1]]), mode="lines+markers", line_shape="hvh",
                name=name, legend="legend", legendgroup=f"cls{cls}", showlegend=first,
                line=dict(color=color, width=1.8),
                marker=dict(symbol=sym, size=np.r_[0, np.full(len(mids), 10), 0], color=color,
                            line=dict(width=1.5, color=T.PAPER)),
                fill="tozeroy", fillcolor=T.rgba(color, 0.10), customdata=bin_text,
                hovertemplate=f"{h} {name}<br>{served} %{{customdata}}<br>density %{{y:.2f}}<extra></extra>"),
                1, j)
        fig.add_vline(x=0.5, line=dict(color=T.NEUTRAL, dash=_REF_DASH, width=1), row=1, col=j)
        fig.update_xaxes(title_text="calibrated P(up) (served)" if fitted else "raw P(up) (served, uncalibrated)",
                         range=[p_lo, p_hi], row=1, col=j)
        fig.update_yaxes(title_text="density" if first else None, rangemode="tozero", row=1, col=j)

        # ---- row 2: ROC as lift over chance
        band = s["band"]
        fig.add_trace(go.Scatter(
            x=f32(np.r_[f_grid, f_grid[::-1]]), y=f32(np.r_[band, -band[::-1]]), mode="lines", fill="toself",
            fillcolor=T.rgba(T.NEUTRAL, 0.2), line=dict(width=0.8, color=T.rgba(T.NEUTRAL, 0.55)),
            hoverinfo="skip", name="no-skill head, pointwise 95% band", legend="legend2", legendgroup="chance",
            showlegend=first), 2, j)
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 0], mode="lines", line=dict(color=T.NEUTRAL, width=1),
                                 hoverinfo="skip", showlegend=False, legend="legend2", legendgroup="chance"), 2, j)
        for key, name, dash, width in (("roc_head", "direction head", T.VAL_DASH, 2.2),
                                       ("roc_gauss", "Gaussian readout (price head)", T.ALT_DASH, 1.6)):
            if not s["both"]:
                continue
            fpr, tpr, thr, auc = s[key]
            lo, hi = s["ci_" + key[4:]]
            fig.add_trace(go.Scatter(
                x=f32(fpr), y=f32(tpr - fpr), mode="lines", name=name, legend="legend2", legendgroup=key,
                showlegend=False, line=dict(color=c, dash=dash, width=width),
                customdata=np.c_[f32(np.where(np.isfinite(thr), thr, np.nan)), f32(tpr)],
                hovertemplate=(f"{h} {name}, AUC {auc:.3f} (95% CI {lo:.2f}-{hi:.2f})"
                               "<br>up when the score ≥ %{customdata[0]:.3f}"
                               "<br>FPR %{x:.3f} · TPR %{customdata[1]:.3f}<br>TPR − FPR %{y:>+.3f}<extra></extra>")),
                2, j)
        fig.update_xaxes(title_text="false positive rate (down moves called up)", range=[0, 1], row=2, col=j)
        fig.update_yaxes(title_text="TPR − FPR (lift over chance)" if first else None,
                         range=[-lift_lim, lift_lim], zeroline=False, row=2, col=j)

        # ---- row 3: reliability, raw (open) vs calibrated (filled, with its 95% CI); one series if no calibration
        fig.add_trace(go.Scatter(x=[rel_lo, rel_hi], y=[rel_lo, rel_hi], mode="lines", name="perfect",
                                 legend="legend3", legendgroup="perfect", showlegend=first, hoverinfo="skip",
                                 line=dict(color=T.NEUTRAL, width=1)), 3, j)
        if np.isfinite(s["base"]):
            fig.add_trace(go.Scatter(
                x=[rel_lo, rel_hi], y=[s["base"], s["base"]], mode="lines",
                name="realised up-rate", legend="legend3", legendgroup="base", showlegend=first,
                line=dict(color=T.NEUTRAL, width=1.2, dash=_REF_DASH),
                hovertemplate=f"{h} realised up-rate {s['base']:.3f} (a flat curve here has no resolution)"
                              "<extra></extra>"), 3, j)
        tr, tc = s["rel_raw"], s["rel_cal"]
        paired = fitted and s["monotone"] and len(tr) == len(tc) and len(tc) > 0
        if paired:
            nan_col = np.full(len(tc), np.nan)
            fig.add_trace(go.Scatter(
                x=f32(np.c_[tr[:, 0], tc[:, 0], nan_col].ravel()), y=f32(np.c_[tr[:, 1], tc[:, 1], nan_col].ravel()),
                mode="lines", hoverinfo="skip", showlegend=False, legend="legend3", legendgroup="shift",
                line=dict(color=T.rgba(c, _SHIFT_ALPHA), width=_SHIFT_WIDTH)), 3, j)
        if fitted and len(tr):
            moved = " → calibrated %{customdata[2]:.3f}" if paired else ""
            fig.add_trace(go.Scatter(
                x=f32(tr[:, 0]), y=f32(tr[:, 1]), mode="markers", name="raw P(up)", legend="legend3",
                legendgroup="raw", showlegend=False,
                marker=dict(symbol="circle-open", size=9, color=c, line=dict(width=1.6, color=c)),
                customdata=np.c_[np.arange(1, len(tr) + 1), tr[:, 2], tc[:, 0] if paired else tr[:, 0]],
                hovertemplate=(f"{h} raw P(up), bin %{{customdata[0]:.0f}}<br>mean raw P(up) %{{x:.3f}}{moved}"
                               "<br>observed up-rate %{y:.3f}<br>n %{customdata[1]:,.0f}<extra></extra>")), 3, j)
        if len(tc):
            what = served if fitted else f"{served} (served, uncalibrated)"
            fig.add_trace(go.Scatter(
                x=f32(tc[:, 0]), y=f32(tc[:, 1]), mode="markers", name=served, legend="legend3",
                legendgroup="cal", showlegend=False, marker=dict(symbol="circle", size=8, color=c),
                error_y=dict(type="data", symmetric=False, array=f32(tc[:, 4] - tc[:, 1]),
                             arrayminus=f32(tc[:, 1] - tc[:, 3]), thickness=1.2, width=3, color=c),
                customdata=np.c_[np.arange(1, len(tc) + 1), tc[:, 2], tc[:, 3], tc[:, 4]],
                hovertemplate=(f"{h} {what}, bin %{{customdata[0]:.0f}}<br>mean P(up) %{{x:.3f}}"
                               "<br>observed up-rate %{y:.3f}<br>95% CI %{customdata[2]:.3f}-%{customdata[3]:.3f}"
                               f" (clustered in {s['block']}-bar blocks)"
                               "<br>n %{customdata[1]:,.0f}<extra></extra>")), 3, j)
        fig.update_xaxes(title_text="mean predicted P(up) in the bin", range=[rel_lo, rel_hi], row=3, col=j)
        fig.update_yaxes(title_text="observed up-rate" if first else None, range=[rel_lo, rel_hi], row=3, col=j)

    # legend-only keys for the horizon-coloured series, in neutral ink (the colour is the column's horizon)
    keys = [("direction head", "roc_head", 2, dict(mode="lines", line=dict(color=T.INK_2, width=2.2))),
            ("Gaussian readout (price head)", "roc_gauss", 2,
             dict(mode="lines", line=dict(color=T.INK_2, width=1.6, dash=T.ALT_DASH))),
            ("calibrated (served), 95% CI" if fitted else "raw P(up) (served), 95% CI", "cal", 3,
             dict(mode="markers", marker=dict(symbol="circle", size=8, color=T.INK_2)))]
    if fitted:
        keys.append(("raw", "raw", 3,
                     dict(mode="markers", marker=dict(symbol="circle-open", size=9, color=T.INK_2, line=dict(width=1.6)))))
        if any(st[h]["monotone"] for h in T.HORIZONS):
            # a thick translucent bar, as drawn between each raw (open) and calibrated (filled) marker
            keys.append(("○→● calibration shift", "shift", 3,
                         dict(mode="lines", line=dict(color=T.rgba(T.INK_2, _SHIFT_ALPHA + 0.15), width=_SHIFT_WIDTH))))
    for name, group, row, kw in keys:
        fig.add_trace(go.Scatter(x=[None], y=[None], name=name, legend=f"legend{row}", legendgroup=group,
                                 showlegend=True, hoverinfo="skip", **kw), row, 1)

    # ---- row 4: scorecard
    fig.add_trace(_scorecard(st, config, score_rows), 4, 1)

    blk = max(int(ci_block), 2 * max(st[h]["steps"] for h in T.HORIZONS))
    if fitted:
        ranking = ("ranks like the raw head" if all(st[h]["monotone"] for h in T.HORIZONS)
                   else "ranks differently from the raw head")
        uses = f"ROC and scorecard use the served (calibrated) P(up), which {ranking}"
    else:
        uses = f"ROC and scorecard use the raw P(up): {_NO_CAL}"
    T.apply(fig, title="Direction heads", height=height, subtitle=(
        f"{frame.split} block, {len(frame):,} samples · direction numbers use only moves beyond ±{deadband:g} bps"
        " (labelled n per column) · line colour = horizon<br>"
        "95% intervals: AUC, accuracy and the chance band on effective samples (n / horizon bars; outcomes overlap)"
        f"<br>reliability bars clustered in {blk}-bar blocks · {uses}"))
    fig.update_layout(height=top + bottom + plot_px, margin=dict(t=top, b=bottom, l=70, r=24))
    for a in fig.layout.annotations or ():
        if a.text and a.text.startswith(("<span", "AUC", "ECE")):
            a.update(font=dict(color=T.INK, size=12.5))
        elif a.text == _SCORECARD_TITLE:
            a.update(x=0.0, xanchor="left", font=dict(color=T.INK, size=13))
    heading1 = "Calibrated P(up) by realised move" if fitted else f"Raw P(up) by realised move ({_NO_CAL})"
    for legend_id, row, heading in (("legend", 1, heading1),
                                    ("legend2", 2, "ROC as lift over chance (area = AUC − 0.5)"),
                                    ("legend3", 3, f"Reliability, {N_REL_BINS} equal-count bins")):
        T.panel_legend(fig, legend_id, row, 1, heading)
        yd = fig.get_subplot(row, 1).yaxis.domain
        fig.layout[legend_id].update(y=yd[1] + _HEADING_PX / plot_px, font=dict(size=11, color=T.INK_2),
                                     title=dict(font=dict(size=13, color=T.INK)))
    # reliability keys at their drawn widths: the thin 'perfect' line and the thick raw -> calibrated bar must
    # not share one constant swatch width (the keys are proxies with scalar marker sizes, so this is safe)
    fig.layout.legend3.itemsizing = "trace"
    return fig


def _scorecard_rows(deadband, fitted):
    """(label, formatter) per scorecard row; labels short enough to stay on one line in a ~1000 px notebook.

    No '<' / '>' and no HTML tags in table cells: plotly then measures the cell as rich text and gives the
    row extra height, so the table would no longer fit its reserved height.
    """
    def pct_ci(s):
        p, lo, hi = s["acc"]
        return f"{_f(p, '.1%')} [{_f(100 * lo, '.1f')}, {_f(100 * hi, '.1f')}]"

    if fitted:
        ece = ("ECE, report bins (0.1 wide): raw → cal · const 0.5 · Gaussian",
               lambda s: (f"{_f(s['ece_rep_raw'], '.3f')} → {_f(s['ece_rep_cal'], '.3f')} · "
                          f"{_f(s['ece_rep_const'], '.3f')} · {_f(s['ece_rep_gauss'], '.3f')}"))
    else:
        ece = ("ECE, report bins (0.1 wide): head · const 0.5 · Gaussian",
               lambda s: (f"{_f(s['ece_rep_cal'], '.3f')} · {_f(s['ece_rep_const'], '.3f')} · "
                          f"{_f(s['ece_rep_gauss'], '.3f')}"))
    return [
        (f"labelled n (moves beyond ±{deadband:g} bps) · effective n", lambda s: f"{s['n']:,} · {s['n_eff']:,}"),
        ("realised up-rate · called up (P(up) above 0.5)",
         lambda s: f"{_f(s['base'], '.1%')} · {_f(s['called_up'], '.1%')}"),
        ("accuracy [95% CI, effective n] · balanced accuracy",
         lambda s: f"{pct_ci(s)} · {_f(s['bal_acc'], '.1%')}"),
        ("precision of up calls · recall of up moves",
         lambda s: f"{_f(s['precision'], '.1%')} · {_f(s['recall'], '.1%')}"),
        ("MCC", lambda s: _f(s["mcc"], "+.3f")),
        ("AUC head − Gaussian (paired DeLong) [95% CI, effective n]",
         lambda s: (f"{_f(s['auc_diff'][0], '+.3f')} [{_f(s['auc_diff'][1], '+.2f')}, "
                    f"{_f(s['auc_diff'][2], '+.2f')}]")),
        (f"Brier: {'calibrated head' if fitted else 'head (uncalibrated)'} · const 0.5 · Gaussian",
         lambda s: f"{_f(s['brier_cal'], '.4f')} · {_f(s['brier_const'], '.4f')} · {_f(s['brier_gauss'], '.4f')}"),
        ece,
        ("delta head's sign agrees with P(up) above 0.5 (all samples)", lambda s: _f(s["sign_agree"], ".1%")),
    ]


def _scorecard(st, config, rows):
    import plotly.graph_objects as go

    header = ["metric"] + [T.horizon_label(h, config) for h in T.HORIZONS]
    cells = [[label for label, _ in rows]] + [[fn(st[h]) for _, fn in rows] for h in T.HORIZONS]
    return go.Table(
        columnwidth=[5, 2, 2, 2],
        header=dict(values=header, fill_color=T.SURFACE, line_color=T.GRID, height=_HEADER_PX,
                    align=["left", "right", "right", "right"],
                    font=dict(size=12, weight="bold", color=[T.INK] + [T.HORIZON_COLORS[h] for h in T.HORIZONS])),
        cells=dict(values=cells, fill_color=T.PAPER, line_color=T.GRID, height=_ROW_PX,
                   align=["left", "right", "right", "right"],
                   font=dict(size=11.5, color=[T.INK_2, T.INK, T.INK, T.INK])))
