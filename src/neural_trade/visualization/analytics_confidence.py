"""Direction confidence and cross-horizon coherence (one column per horizon).

* :func:`confidence_analytics_figure` - accuracy by decile of |P(up) - 0.5|, selective accuracy
  (keep only the most decided x%), accuracy by decile of the strategies' variance confidence
  exp(-var / var_scale), and the confusion matrices with recall / precision / MCC.
* :func:`coherence_analytics_figure` - how the horizons agree: P(up) correlation, the eight
  vote patterns, the realised up-rate by number of horizons voting up, direction head vs price
  head sign agreement (raw heads, as the eval report), |delta| ordering (raw heads), and the
  strategies' vote agreement. Where beta = 0 the served delta is identically 0: its sign and
  ordering are drawn as n/a, never as a measured share.

Uncertainty. Consecutive 1-minute samples have overlapping targets (a 20-bar outcome shares 19
bars with its neighbour), so the correctness of neighbouring calls is strongly correlated. Every
interval here is a 95% moving-block bootstrap over the time-ordered block (80-bar blocks: the
method, block length and draw the evaluation report uses for its confidence gap), applied to each
subset through prefix sums over the resampled blocks, so a decile scattered in time and a
contiguous stretch are both treated honestly, and memory stays small at any block length. A
block too short for it falls back to a Wilson interval on N / steps effective samples
(``stats.wilson``).

Chance references. Accuracy is read against what no skill would score on the same samples:
random calls with the model's own up/down mix, q * b + (1 - q) * (1 - b) (q = share of up calls,
b = share of up moves), and against the test block's majority class ("always up").
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from neural_trade.evaluation.report import _frame_extra, _resolve_raw
from neural_trade.visualization import stats as S
from neural_trade.visualization import theme as T
from neural_trade.visualization.analytics_common import _grid, _labels

H = T.HORIZONS
PATTERNS = ("UUU", "UUD", "UDU", "UDD", "DUU", "DUD", "DDU", "DDD")
CI_NOTE = "95% CI: 80-bar moving-block bootstrap (neighbouring outcomes overlap)"
REF_DASH = "6px,4px"            # reference lines (an explicit pattern: the legend draws "dash" as one block)
TICK = dict(symbol="line-ew-open", size=18, line=dict(color=T.NEUTRAL, width=2), color=T.NEUTRAL)
NOSKILL_W = 1.25                # the no-skill curve: thin grey, against the 2.5 px horizon-coloured accuracy line
NOSKILL_KEY = "no skill: the same up/down mix at random"
SWATCH = dict(symbol="square", size=11)     # legend-only key for a bar colour (a bar trace's key takes its 1st bar)
KEY_ROW_PX = 23                 # the row of legend keys under a legend-title heading


# ------------------------------------------------------------------ uncertainty
def _boot_plan(n: int, *, n_boot: int = 1000, seed: int = 0, block: Optional[int] = None):
    """A moving-block bootstrap over ``n`` time-ordered bars, or None when ``n`` is shorter than two blocks.

    The evaluation report's draw (the same generator call: nb = ceil(n / block) blocks of ``block``
    bars, the last one cut so a resample holds n bars). Resample r concatenates the bars
    [starts[r, j], ends[r, j]) over the blocks j. Returned as the sparse [n_boot, n + 1] operator
    ``B`` with +1 at every block end and -1 at every block start: for prefix sums c (c[0] = 0,
    c[i] = sum of the first i bars), ``B @ c`` is each resample's sum. Memory O(n_boot * n / block),
    never an [n_boot, n] matrix.
    """
    from scipy import sparse

    from neural_trade.evaluation.report import BLOCK

    block = int(BLOCK if block is None else block)
    if n_boot <= 0 or n < 2 * block:
        return None
    rng = np.random.default_rng(seed)
    nb = int(math.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, nb))
    lens = np.full(nb, block)
    lens[-1] = n - (nb - 1) * block
    rows = np.repeat(np.arange(n_boot), nb)
    vals = np.r_[np.ones(starts.size), -np.ones(starts.size)]
    return sparse.csr_matrix((vals, (np.r_[rows, rows], np.r_[(starts + lens).ravel(), starts.ravel()])),
                             shape=(n_boot, n + 1))


def _resample_sums(sub, x, plan, *, budget: int = 2_000_000):
    """[k, n_boot]: in every bootstrap resample, the sum of ``x`` over each row of the boolean [k, N]
    ``sub`` (``x`` None: the number of the subset's bars in the resample).

    Prefix sums over the bars, one column per subset, in chunks of about ``budget`` elements, so peak
    memory does not grow with n_boot * N; a block [s, e) adds c[e] - c[s] (``plan @ c``).
    """
    sub = np.atleast_2d(sub)
    k, n = sub.shape
    out = np.empty((k, plan.shape[0]))
    step = max(1, int(budget) // (n + 1))
    for a in range(0, k, step):
        v = sub[a:a + step].astype(float)
        if x is not None:
            v *= x
        c = np.zeros((n + 1, len(v)))
        np.cumsum(v.T, axis=0, out=c[1:])
        out[a:a + step] = (plan @ c).T
    return out


def _subset_rates(x, subsets, plan, steps: int = 1):
    """Share of ``x`` (0/1 over all N bars) inside each subset (rows of a boolean [k, N] matrix).

    Returns (rate, lo, hi, n): the 95% interval is the percentile interval of the block-bootstrap
    rate when ``plan`` (``_boot_plan``) is given, else a Wilson interval on n / steps effective samples.
    """
    sub = np.atleast_2d(np.asarray(subsets, bool))
    x = np.asarray(x, float)
    n = sub.sum(1).astype(float)
    k = np.array([x[row].sum() for row in sub])
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where(n > 0, k / np.maximum(n, 1.0), np.nan)
    if plan is None:
        _, lo, hi = S.wilson(k, n, steps=steps)
    else:
        num, den = _resample_sums(sub, x, plan), _resample_sums(sub, None, plan)
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.where(den > 0, num / np.maximum(den, 1e-9), np.nan)
        ok = np.isfinite(r).any(1)
        lo, hi = np.full(len(n), np.nan), np.full(len(n), np.nan)
        if ok.any():
            lo[ok], hi[ok] = np.nanpercentile(r[ok], [2.5, 97.5], axis=1)
    lo, hi = np.where(n > 0, lo, np.nan), np.where(n > 0, hi, np.nan)
    return rate, np.minimum(lo, rate), np.maximum(hi, rate), n.astype(int)


def _confidence_gap(correct, confidence, threshold, *, n_boot: int = 1000, seed: int = 0):
    """``evaluation.report.confidence_gap`` with the same draw and the same numbers (tested), without
    its [n_boot, n] resample matrices (about 1 GB at a month of 1-minute bars)."""
    from neural_trade.evaluation.report import confidence_gap

    correct = np.asarray(correct, float)
    hi = np.asarray(confidence) > threshold
    plan = _boot_plan(len(correct), n_boot=n_boot, seed=seed)
    if plan is None or hi.all() or not hi.any():            # the report's INSUFFICIENT row (it returns early)
        return confidence_gap(correct, confidence, threshold, n_boot=n_boot, seed=seed)
    sub = np.stack([hi, ~hi])
    num, den = _resample_sums(sub, correct, plan), _resample_sums(sub, None, plan)
    with np.errstate(invalid="ignore", divide="ignore"):
        gaps = num[0] / den[0] - num[1] / den[1]
    lo, up = np.nanpercentile(gaps, [2.5, 97.5])
    gap = correct[hi].mean() - correct[~hi].mean()
    if lo > 0 and gap >= 0.01:
        verdict = "WORKS"
    elif up < 0 and gap <= -0.01:
        verdict = "INVERTED"
    else:
        verdict = "NOISE"
    return {"gap": float(gap), "ci": [float(lo), float(up)], "verdict": verdict, "threshold": float(threshold),
            "acc_high": float(correct[hi].mean()), "acc_low": float(correct[~hi].mean()), "n_high": int(hi.sum())}


def _noskill(q, b):
    """Accuracy of random calls that say up with probability q when a share b of the moves is up."""
    return q * b + (1 - q) * (1 - b)


# ------------------------------------------------------------------ small helpers
def _gap_text(g) -> str:
    if not g or not np.isfinite(g.get("gap", np.nan)):
        return "gap n/a"
    lo, hi = g["ci"]
    return f"gap {g['gap']:+.3f} [{lo:+.3f}, {hi:+.3f}] {g['verdict']}"


def _report_gaps(report, n: int):
    """{h: confidence_gap row} from an EvalReport, its dict, or an eval_report_*.json path; ({}, reason)
    when absent or when the report scored a different number of samples than this frame."""
    if report is None:
        return {}, None
    if isinstance(report, (str, Path)):
        if not Path(report).is_file():
            return {}, f"no eval report at {Path(report).name}"
        report = json.loads(Path(report).read_text(encoding="utf-8"))
    if hasattr(report, "model") and hasattr(report, "n"):
        d = {"n": report.n, "model": report.model}
    else:
        d = dict(report)
    if d.get("n") is not None and int(d["n"]) != int(n):
        return {}, f"the report given scored {int(d['n']):,} samples, this block has {int(n):,}"
    horizons = (d.get("model") or d).get("horizons", {})
    return {h: horizons[h]["confidence_gap"] for h in H if "confidence_gap" in horizons.get(h, {})}, None


def _dec(v: float, digits: int) -> str:
    """A share in [0, 1] without its leading zero ('.504'), so ten decile edges fit under a narrow panel."""
    return f"{v:.{digits}f}".replace("0.", ".", 1) if 0 <= v < 1 else f"{v:.{digits}f}"


def _tick(values, fmt, prefix="&#8805;"):
    return [f"{k + 1}<br>{prefix}{fmt(v)}" for k, v in enumerate(values)]


def _legend(fig, legend_id: str, row: int, col: int, title: str, *, y_px: float, plot_h: float, lead: str = ""):
    """A legend drawn as a heading above the subplot titles of ``row``: the bold title, then ``lead``
    (plain text introducing the first keys), then the keys."""
    sp = fig.get_subplot(row, col)
    fig.update_layout({legend_id: dict(
        title=dict(text=f"<b>{title}</b>   {lead}", side="left", font=dict(color=T.INK_2, size=12)),
        orientation="h", x=sp.xaxis.domain[0], xanchor="left", y=sp.yaxis.domain[1] + y_px / plot_h,
        yanchor="bottom", font=dict(size=11, color=T.INK_2), bgcolor="rgba(0,0,0,0)", itemsizing="constant",
        itemwidth=30, traceorder="normal", tracegroupgap=0)})


def _heading(fig, legend_id: Optional[str], row: int, col: int, title: str, *, plot_h: float, note: str = "",
             lift: float = 0.0):
    """A panel heading at the panel's top-left: the bold title (plus a note), then that panel's legend keys
    on the line below. Replaces the subplot title with the same text. Without a legend the heading is an
    annotation, ``lift`` px higher than its default 4 px (``KEY_ROW_PX``: level with its neighbours' titles)."""
    sp = fig.get_subplot(row, col)
    x0, y0 = sp.xaxis.domain[0], sp.yaxis.domain[1]
    fig.layout.annotations = [a for a in (fig.layout.annotations or ()) if a.text != title]
    text = f"<b>{title}</b>" + (f"<br><span style='font-size:11px;color:{T.MUTED}'>{note}</span>" if note else "")
    if legend_id is None:
        fig.add_annotation(x=x0, y=y0 + (4 + lift) / plot_h, xref="paper", yref="paper", xanchor="left",
                           yanchor="bottom", align="left", showarrow=False, text=text,
                           font=dict(size=12, color=T.INK_2))
        return
    fig.update_layout({legend_id: dict(
        title=dict(text=text, side="top", font=dict(color=T.INK_2, size=12)),
        orientation="h", x=x0, xanchor="left", y=y0 + 2 / plot_h, yanchor="bottom",
        font=dict(size=11, color=T.INK_2), bgcolor="rgba(0,0,0,0)", itemsizing="constant", itemwidth=30,
        traceorder="normal", tracegroupgap=0)})


def _key(fig, row, col, legend, name, **kw):
    """A legend-only key (no data) for a mark drawn in several panels, or for a colour inside one trace."""
    import plotly.graph_objects as go

    fig.add_trace(go.Scatter(x=[None], y=[None], name=name, legend=legend, hoverinfo="skip", **kw), row, col)


def _majority(b):
    return ("always up", b) if b >= 0.5 else ("always down", 1 - b)


def _vote_lines():
    """(up, down): SignalFrame counts a horizon's vote when P(up) > up or P(up) < down.

    The module constants when ``strategy.signals`` defines them (VOTE_UP / VOTE_DOWN), else read off
    ``SignalFrame.build`` itself on an exact 0.001 grid of P(up) for h0 (h1 and h2 at 0.5 cast no vote),
    so the figure states the lines the strategies really apply. (None, None) if that fails.
    """
    from neural_trade.evaluation.frame import PredictionFrame
    from neural_trade.strategy import signals

    up, dn = getattr(signals, "VOTE_UP", None), getattr(signals, "VOTE_DOWN", None)
    if up is not None and dn is not None:
        return float(up), float(dn)
    try:
        g = np.arange(1001) / 1000.0                 # exact decimals: 550 / 1000 == 0.55
        n, half, zero = len(g), np.full(len(g), 0.5), np.zeros(len(g))
        fr = PredictionFrame(y=np.zeros((n, 3)), last_close=np.ones(n), delta={h: zero for h in H},
                             direction_prob={"h0": g, "h1": half, "h2": half},
                             variance_scaled={h: np.ones(n) for h in H}, pred_scale=1.0)
        cons = signals.SignalFrame.build(fr, 1.0).consensus
        if not ((cons > 0).any() and (cons < 0).any()):
            return None, None
        return float(g[cons < 1].max()), float(g[cons > -1].min())
    except Exception:                                # a changed SignalFrame must not break the figure
        return None, None


# ------------------------------------------------------------------ confidence
def _horizon_stats(frame, config, h, labels, plan, var_scale, gap_from_report):
    """Everything the confidence figure draws for one horizon."""
    from scipy.stats import spearmanr

    N = len(frame)
    lab, mask = labels[h]
    p = np.asarray(frame.prob(h, True), float)
    steps = S.horizon_steps(frame, h)
    up = (np.asarray(lab) > 0.5)
    pred = p > 0.5
    correct = (pred == up).astype(np.float32)
    conf = np.abs(p - 0.5)
    midx = np.flatnonzero(mask)
    n_m = len(midx)
    out: Dict[str, Any] = dict(h=h, n=n_m, N=N, steps=steps, ok=n_m >= 20)
    if not out["ok"]:
        return out
    b = float(up[midx].mean())
    out["b"], out["acc"] = b, float(correct[midx].mean())
    # -- deciles of |P(up) - 0.5| over the scored samples (the eval report's confidence)
    order = midx[np.argsort(conf[midx], kind="mergesort")]
    chunks = [c for c in np.array_split(order, 10) if len(c)]
    sub = np.zeros((len(chunks), N), bool)
    for r, c in enumerate(chunks):
        sub[r, c] = True
    acc, lo, hi, n = _subset_rates(correct, sub, plan, steps)
    q = np.array([pred[c].mean() for c in chunks])
    bu = np.array([up[c].mean() for c in chunks])
    out["dec"] = dict(acc=acc, lo=lo, hi=hi, n=n, q=q, b=bu, noskill=_noskill(q, bu),
                      expected=np.array([0.5 + conf[c].mean() for c in chunks]),
                      edge_lo=np.array([0.5 + conf[c].min() for c in chunks]),
                      edge_hi=np.array([0.5 + conf[c].max() for c in chunks]))
    # -- selective accuracy: the most decided k samples, on a log grid of k
    desc = order[::-1]
    ks = np.unique(np.geomspace(max(1, math.ceil(0.02 * n_m)), n_m, 200).astype(int))
    rank = np.full(N, np.iinfo(np.int64).max)
    rank[desc] = np.arange(n_m)
    top = rank[None, :] < ks[:, None]
    s_acc, s_lo, s_hi, _ = _subset_rates(correct, top, plan, steps)
    cq, cb = np.cumsum(pred[desc])[ks - 1] / ks, np.cumsum(up[desc])[ks - 1] / ks
    out["sel"] = dict(share=ks / n_m, k=ks, acc=s_acc, lo=s_lo, hi=s_hi, q=cq, b=cb, noskill=_noskill(cq, cb),
                      expected=np.cumsum(0.5 + conf[desc])[ks - 1] / ks)
    # -- the eval report's confidence gap (or the same function, split at this block's median)
    if gap_from_report is not None:
        out["gap"], out["gap_src"] = gap_from_report, "report"
    else:
        out["gap"] = _confidence_gap(correct[midx], conf[midx], float(np.median(conf[midx])))
        out["gap_src"] = "self"
    # -- the strategies' confidence exp(-var / var_scale), deciles over ALL bars (what a strategy sees)
    v = np.clip(np.asarray(frame.variance_scaled[h], float), 0, 1e4)
    sig = frame.sigma(h)
    cv = np.exp(-v / var_scale) if var_scale else -v          # ranking only when var_scale is unknown
    vorder = np.argsort(cv, kind="mergesort")                 # decile 1 = least confident = highest variance
    vchunks = [c for c in np.array_split(vorder, 10) if len(c)]
    vsub = np.zeros((len(vchunks), N), bool)
    for r, c in enumerate(vchunks):
        vsub[r, c[mask[c]]] = True
    vacc, vlo, vhi, vn = _subset_rates(correct, vsub, plan, steps)
    vq = np.array([pred[c[mask[c]]].mean() if mask[c].any() else np.nan for c in vchunks])
    vb = np.array([up[c[mask[c]]].mean() if mask[c].any() else np.nan for c in vchunks])
    out["var"] = dict(acc=vacc, lo=vlo, hi=vhi, n=vn, q=vq, b=vb, noskill=_noskill(vq, vb),
                      deadband=np.array([1 - mask[c].mean() for c in vchunks]),
                      conf_lo=np.array([cv[c].min() for c in vchunks]) if var_scale else None,
                      conf_mean=np.array([cv[c].mean() for c in vchunks]) if var_scale else None,
                      sig_lo=np.array([sig[c].min() for c in vchunks]),
                      sig_hi=np.array([sig[c].max() for c in vchunks]))
    out["var_gap"] = _confidence_gap(correct[midx], cv[midx], float(np.median(cv[midx])))
    rho = spearmanr(cv[midx], conf[midx]).correlation if np.ptp(cv[midx]) > 0 and np.ptp(conf[midx]) > 0 else np.nan
    out["rho"] = float(rho)
    # -- confusion matrix (float counts: the int64 MCC product overflows)
    t, pr = up[midx], pred[midx]
    tp, tn = float(np.sum(pr & t)), float(np.sum(~pr & ~t))
    fp, fn = float(np.sum(pr & ~t)), float(np.sum(~pr & t))
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    rec_up, rec_dn = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
    out["cm"] = dict(counts=np.array([[tn, fp], [fn, tp]]), mcc=(tp * tn - fp * fn) / den if den > 0 else 0.0,
                     bal=(rec_up + rec_dn) / 2, rec_up=rec_up, rec_dn=rec_dn,
                     prec_up=tp / max(tp + fp, 1), prec_dn=tn / max(tn + fn, 1), pred_up=(tp + fp) / n_m)
    return out


def confidence_analytics_figure(frame, config=None, *, height: Optional[int] = None,
                                var_scale: Optional[float] = None, report=None, n_boot: int = 1000,
                                seed: int = 0):
    """Is a more decided call a more accurate one? Four rows, one column per horizon.

    ``var_scale``: the strategies' confidence scale from the CALIBRATION block (the run bundle's
    ``meta['var_scale']`` or ``strategy.var_scale_from(cal_frame)``); without it row 3 ranks by
    predicted sigma (the same deciles) and labels them in dollars. ``report``: the run's
    EvalReport (or its dict / ``eval_report_test.json`` path) so the confidence gap in row 1 is
    the report's own (threshold from the calibration block); without it the same function is
    applied with a median split on this block. ``n_boot``: bootstrap resamples (0 = Wilson on
    N / steps effective samples).
    """
    import plotly.graph_objects as go

    labels = _labels(frame, config)
    N = len(frame)
    plan = _boot_plan(N, n_boot=n_boot, seed=seed)
    gaps, gap_note = _report_gaps(report, N)
    st = {h: _horizon_stats(frame, config, h, labels, plan, var_scale, gaps.get(h)) for h in H}
    hl = {h: T.horizon_label(h, config) for h in H}
    deadband = float(getattr(config, "DIR_DEADBAND_BPS", 0.0) or 0.0) if config is not None else 0.0

    def _t(h, text):
        return f"{hl[h]}: {text}" if st[h]["ok"] else f"{hl[h]}: fewer than 20 scored samples"

    def _vs(h):
        """'accuracy 51.6% vs always up 51.4%' (the test block's majority class)."""
        name, base = _majority(st[h]["b"])
        return f"accuracy {st[h]['acc']:.1%} vs {name} {base:.1%}"

    titles = [_t(h, _gap_text(st[h].get("gap"))) for h in H]
    titles += [_t(h, f"{st[h]['n']:,} scored · {_vs(h)}" if st[h]["ok"] else "") for h in H]
    titles += [_t(h, f"{_gap_text(st[h].get('var_gap'))} · &#961; vs |P&#8722;0.5| {st[h].get('rho', np.nan):+.2f}")
               for h in H]
    titles += [_t(h, _vs(h) if st[h]["ok"] else "") for h in H]
    # rows 2-4: the short horizon name (row 1 carries the horizon length)
    titles = titles[:3] + [t.replace(hl[h] + ":", h + ":", 1) for t, h in zip(titles[3:], H * 3)]
    height = int(height or 1960)
    margin = dict(t=170, b=70, l=64, r=24)
    plot_h = height - margin["t"] - margin["b"]
    fig = _grid(4, titles, row_heights=[0.27, 0.27, 0.27, 0.19], vspace=0.078)
    for a in fig.layout.annotations:
        a.font = dict(size=13, color=T.INK_2)

    rng = {1: [], 2: []}            # y extents: rows 1 + 3 share one range, row 2 its own
    for j, h in enumerate(H, start=1):
        s, c = st[h], T.HORIZON_COLORS[h]
        if not s["ok"]:
            continue
        base = _majority(s["b"])[1]
        for row, key in ((1, "dec"), (3, "var")):
            d = s[key]
            cats = [str(k + 1) for k in range(len(d["acc"]))]
            legend = "legend" if row == 1 else "legend3"
            if row == 1:
                ticks = _tick(d["edge_lo"], lambda v: _dec(v, 3))
                extra = [f"max(P, 1&#8722;P) {a:.3f}&#8211;{b_:.3f}" for a, b_ in zip(d["edge_lo"], d["edge_hi"])]
                ex = d["expected"]
            else:
                if d["conf_lo"] is not None:
                    ticks = _tick(d["conf_lo"], lambda v: _dec(v, 2))
                    extra = [f"confidence &#8805; {a:.3f} (mean {m:.3f}), &#963; ${lo_:,.0f}&#8211;${hi_:,.0f}"
                             for a, m, lo_, hi_ in zip(d["conf_lo"], d["conf_mean"], d["sig_lo"], d["sig_hi"])]
                else:
                    ticks = _tick(d["sig_lo"], lambda v: f"${v:,.0f}")
                    extra = [f"&#963; ${lo_:,.0f}&#8211;${hi_:,.0f}" for lo_, hi_ in zip(d["sig_lo"], d["sig_hi"])]
                extra = [f"{e}<br>{100 * db:.0f}% of these bars move inside the deadband (not scored)"
                         for e, db in zip(extra, d["deadband"])]
                ticks = [f"{t}<br><span style='color:{T.MUTED}'>{100 * db:.0f}%</span>"
                         for t, db in zip(ticks, d["deadband"])]
                ex = None
            cd = np.column_stack([d["n"], d["lo"], d["hi"], d["q"], d["b"], d["noskill"]])
            fig.add_trace(go.Scatter(
                x=cats, y=d["acc"], mode="markers", name=h, legend=legend, showlegend=False,
                marker=dict(color=c, size=9, symbol="circle", line=dict(color=T.PAPER, width=1)),
                error_y=dict(type="data", symmetric=False, array=d["hi"] - d["acc"], arrayminus=d["acc"] - d["lo"],
                             color=c, thickness=1.5, width=4),
                customdata=cd, text=extra,
                hovertemplate=f"{h} decile %{{x}}: %{{text}}<br>accuracy %{{y:.3f}} "
                              "[%{customdata[1]:.3f}, %{customdata[2]:.3f}] (n %{customdata[0]:,})<br>"
                              "called up %{customdata[3]:.0%} · moves up %{customdata[4]:.0%} · "
                              "no skill %{customdata[5]:.3f}<extra></extra>"), row, j)
            fig.add_trace(go.Scatter(x=cats, y=d["noskill"], mode="markers", name="no skill", legend=legend,
                                     showlegend=False, marker=TICK,
                                     hovertemplate="decile %{x}: random calls with this up/down mix score "
                                                   "%{y:.3f}<extra></extra>"), row, j)
            if ex is not None:
                fig.add_trace(go.Scatter(x=cats, y=ex, mode="markers", name="expected if calibrated", legend=legend,
                                         showlegend=False,
                                         marker=dict(symbol="diamond-open", size=9, color=T.INK,
                                                     line=dict(color=T.INK, width=1.5)),
                                         hovertemplate="decile %{x}: a calibrated P(up) would be right "
                                                       "%{y:.3f}<extra></extra>"), row, j)
            fig.add_hline(y=base, line=dict(color=T.NEUTRAL, dash=REF_DASH, width=1), row=row, col=j)
            fig.update_xaxes(type="category", tickmode="array", tickvals=cats, ticktext=ticks, tickangle=0,
                             tickfont=dict(size=10), row=row, col=j)
            rng[1] += [np.nanmin(d["lo"]), np.nanmax(d["hi"]), np.nanmin(d["noskill"]), np.nanmax(d["noskill"]),
                       base]
            if ex is not None:
                rng[1] += [ex.min(), ex.max()]
        # -- selective accuracy: the horizon-coloured line (its own legend key) and band; no skill thin grey
        d = s["sel"]
        x = d["share"].astype(np.float32)
        fig.add_trace(go.Scatter(x=np.r_[x, x[::-1]], y=np.r_[d["hi"], d["lo"][::-1]].astype(np.float32),
                                 fill="toself", fillcolor=T.rgba(c, 0.18), line=dict(width=0), name="95% CI",
                                 legend="legend2", legendgroup=f"sel-{h}", showlegend=False, hoverinfo="skip"), 2, j)
        fig.add_trace(go.Scatter(x=x, y=d["noskill"].astype(np.float32), mode="lines", name="no skill",
                                 legend="legend2", showlegend=False, line=dict(color=T.NEUTRAL, width=NOSKILL_W),
                                 customdata=np.column_stack([d["q"], d["b"]]).astype(np.float32),
                                 hovertemplate="top %{x:.1%}: random calls with the same mix score %{y:.3f}"
                                               "<br>(called up %{customdata[0]:.0%}, moves up "
                                               "%{customdata[1]:.0%})<extra></extra>"), 2, j)
        fig.add_trace(go.Scatter(x=x, y=d["acc"].astype(np.float32), mode="lines", name=h, legend="legend2",
                                 legendgroup=f"sel-{h}", showlegend=True, line=dict(color=c, width=2.5),
                                 customdata=np.column_stack([d["k"], d["lo"], d["hi"], d["expected"]]).astype(
                                     np.float32),
                                 hovertemplate="most decided %{x:.1%} (n %{customdata[0]:,.0f})<br>accuracy %{y:.3f} "
                                               "[%{customdata[1]:.3f}, %{customdata[2]:.3f}]<br>a calibrated P(up) "
                                               "would be right %{customdata[3]:.3f}<extra></extra>"), 2, j)
        fig.add_hline(y=base, line=dict(color=T.NEUTRAL, dash=REF_DASH, width=1), row=2, col=j)
        fig.update_xaxes(type="log", range=[math.log10(0.02), 0], tickvals=[0.02, 0.05, 0.1, 0.2, 0.5, 1.0],
                         tickformat=".0%", row=2, col=j)
        rng[2] += [np.nanmin(d["lo"]), np.nanmax(d["hi"]), np.nanmin(d["noskill"]), np.nanmax(d["noskill"]), base]
        # -- confusion matrix: rows = real class (down on top), colour = share of the real class
        m = s["cm"]
        cnt = m["counts"]
        rows_share = cnt / np.maximum(cnt.sum(1, keepdims=True), 1)
        tot = cnt / max(cnt.sum(), 1)
        ylab = ["real down", "real up"]
        xlab = [f"pred down · {1 - m['pred_up']:.0%} of calls<br>precision {m['prec_dn']:.1%}",
                f"pred up · {m['pred_up']:.0%} of calls<br>precision {m['prec_up']:.1%}"]
        text = [[f"{cnt[r, k]:,.0f}<br>{100 * rows_share[r, k]:.1f}% of {ylab[r]}" for k in range(2)]
                for r in range(2)]
        fig.add_trace(go.Heatmap(z=rows_share, x=xlab, y=ylab, zmin=0, zmax=1, showscale=False,
                                 colorscale=[[0, T.SURFACE], [1, c]], text=text, texttemplate="%{text}",
                                 textfont=dict(color=T.INK, size=12), customdata=tot,
                                 hovertemplate="%{y}, %{x}<br>%{text}<br>%{customdata:.1%} of all scored samples"
                                               "<extra></extra>"), 4, j)
        for k in range(2):     # outline the correct calls (the main diagonal)
            fig.add_shape(type="rect", x0=k - 0.5, x1=k + 0.5, y0=k - 0.5, y1=k + 0.5, row=4, col=j,
                          line=dict(color=T.INK, width=2), fillcolor="rgba(0,0,0,0)")
        fig.update_yaxes(autorange="reversed", showgrid=False, row=4, col=j)
        fig.update_xaxes(showgrid=False, tickfont=dict(size=10), title_standoff=6, row=4, col=j,
                         title_text=f"balanced accuracy {m['bal']:.1%} · MCC {m['mcc']:+.3f}")

    # shared y ranges (one scale per row, so the columns compare by eye)
    for row, key in ((1, 1), (3, 1), (2, 2)):
        v = np.asarray(rng[key], float)
        v = v[np.isfinite(v)]
        if len(v):
            pad = 0.1 * (v.max() - v.min() + 1e-3)
            fig.update_yaxes(range=[max(0.0, v.min() - pad), min(1.0, v.max() + pad)], row=row)
    for row in (1, 2, 3):
        fig.update_yaxes(tickformat=".0%", row=row)
    fig.update_yaxes(title_text="accuracy", title_standoff=4, row=1, col=1)
    fig.update_yaxes(title_text="accuracy of the kept", title_standoff=4, row=2, col=1)
    fig.update_yaxes(title_text="accuracy", title_standoff=4, row=3, col=1)
    fig.update_xaxes(title_text="decile of max(P, 1&#8722;P), labelled by its lower edge (1 = closest to 0.5)",
                     title_standoff=4, row=1)
    fig.update_xaxes(title_text="share of scored samples kept, most decided first (log)", title_standoff=4, row=2)
    vs_txt = (f"decile of exp(&#8722;var / {var_scale:.3g}) over all bars, labelled by its lower edge" if var_scale
              else "decile of predicted &#963; over all bars, highest &#963; first, by its lower &#963; edge")
    fig.update_xaxes(title_text=f"{vs_txt}<br><span style='color:{T.MUTED}'>grey: share of the decile's bars "
                                "inside the deadband (not scored)</span>", title_standoff=4, row=3)

    # legends as row headings (keys drawn once, on the first panel of the row)
    ok = next((h for h in H if st[h]["ok"]), None)
    if ok is not None:
        maj = [(h, *_majority(st[h]["b"])) for h in H if st[h]["ok"]]
        names = {m[1] for m in maj}
        maj_name = f"{names.pop()} (majority class)" if len(names) == 1 else "majority class"
        maj_key = maj_name + ": " + " · ".join(f"{h} {v:.1%}" for h, _, v in maj)
        acc_key = dict(mode="markers", marker=dict(color=T.INK_2, size=9),
                       error_y=dict(type="constant", value=0, color=T.INK_2))
        for row, legend in ((1, "legend"), (3, "legend3")):
            _key(fig, row, 1, legend, "accuracy &#177; 95% CI", **acc_key)
            if row == 1:
                _key(fig, row, 1, legend, "expected if calibrated",
                     mode="markers", marker=dict(symbol="diamond-open", size=9, color=T.INK,
                                                 line=dict(color=T.INK, width=1.5)))
            _key(fig, row, 1, legend, NOSKILL_KEY, mode="markers",
                 marker=TICK)
            _key(fig, row, 1, legend, maj_key, mode="lines", line=dict(color=T.NEUTRAL, dash=REF_DASH, width=1))
        # row 2: the three horizon-coloured accuracy lines are their own keys (after the heading's lead text)
        _key(fig, 2, 1, "legend2", NOSKILL_KEY, mode="lines",
             line=dict(color=T.NEUTRAL, width=NOSKILL_W))
        _key(fig, 2, 1, "legend2", maj_name, mode="lines",           # its values are in the row's panel titles
             line=dict(color=T.NEUTRAL, dash=REF_DASH, width=1))
    src = {"report": "row 1 gap = the eval report's: accuracy of the more decided half minus the less decided half, "
                     "split at the calibration-block median",
           "self": "row 1 gap: accuracy of the more decided half minus the less decided half, split at this block's "
                   "median (pass report= for the eval report's own)"}[st[ok]["gap_src"] if ok else "self"]
    if gap_note:
        src += f" ({gap_note})"
    heads = {1: "Accuracy by decile of |P(up) &#8722; 0.5|",
             2: "Selective accuracy",
             3: "Accuracy by decile of the strategies' confidence exp(&#8722;var / var_scale)"}
    leads = {2: "of the most decided x% (shaded: 95% CI):"}
    for row, legend in ((1, "legend"), (2, "legend2"), (3, "legend3")):
        _legend(fig, legend, row, 1, heads[row], y_px=26, plot_h=plot_h, lead=leads.get(row, ""))
    sp4 = fig.get_subplot(4, 1)
    fig.add_annotation(x=sp4.xaxis.domain[0], y=sp4.yaxis.domain[1] + 26 / plot_h, xref="paper", yref="paper",
                       xanchor="left", yanchor="bottom", showarrow=False, align="left",
                       text="<b>Confusion matrix</b>   scored samples only · colour and % = share of the real class "
                            "(the outlined diagonal is the recall) · precision under each column",
                       font=dict(size=12, color=T.INK_2))
    neff = " / ".join(f"{st[h]['N'] // max(1, st[h]['steps']):,}" for h in H)
    scored = " / ".join(f"{st[h]['n']:,}" for h in H)
    sub = (f"{frame.split} block, N = {N:,} bars (n_eff = N / steps: {neff}) · calibrated P(up) · moves inside the "
           f"{deadband:g} bps deadband are not scored (scored: {scored})<br>"
           + (CI_NOTE if plan is not None else "95% CI: Wilson on N / steps effective samples")
           + " · 10 deciles x 3 horizons at 95%: 1-2 outside a reference by chance is expected<br>"
           + src + "; row 3 gap: the same for the strategies' confidence, split at this block's median")
    T.note_on_empty(fig, "fewer than 20 samples outside the deadband")
    T.apply(fig, title="Direction confidence: is a more decided call a more accurate one?", subtitle=sub,
            height=height, legend_top=False)
    fig.update_layout(margin=margin)
    return fig


# ------------------------------------------------------------------ coherence
NA_BETA0 = "n/a (beta = 0: served delta is 0)"


def _served_betas(D, raw, delta_scale) -> Dict[str, float]:
    """{h: beta} for served = beta x raw: as given (``delta_scale`` / the frame's), else read off the frame
    when served is exactly a multiple of the raw head, else 0 where the served column is identically 0."""
    if delta_scale:
        return {h: float(delta_scale[h]) for h in H if delta_scale.get(h) is not None}
    out = {}
    for i, h in enumerate(H):
        d = D[:, i]
        if not np.any(d):
            out[h] = 0.0
        elif raw is not None and np.any(raw[:, i]):
            r = raw[:, i]
            b = float(d @ r / (r @ r))
            if np.max(np.abs(d - b * r)) <= 1e-5 * np.max(np.abs(d)):
                out[h] = b
    return out


def coherence_analytics_figure(frame, config=None, *, height: Optional[int] = None,
                               raw_delta: Optional[Dict[str, np.ndarray]] = None, n_boot: int = 1000,
                               seed: int = 0, delta_scale: Optional[Dict[str, float]] = None):
    """How the three horizons agree with each other and with the price heads.

    ``raw_delta``: the price heads before the per-horizon delta shrinkage ({h: array} in dollars,
    e.g. ``result.predictions["delta"]``, or the raw PredictionFrame); default: what the frame carries
    (``frame.meta["delta_raw"]``), else served / beta when every beta > 0, the way ``evaluate`` resolves
    them. The served delta is beta x raw with a beta per horizon, which reorders |delta|, and beta = 0
    makes it identically 0. The sign check compares P(up) with the raw heads (the eval report's
    ``delta_dir_align``); without them it uses the served deltas, and a horizon whose served delta is 0
    reads n/a. ``delta_scale``: the per-horizon betas (default: the frame's ``meta["delta_scale"]``, else
    read off served / raw), printed in the notes. Every served-delta statistic is n/a where the served
    delta is 0, and a series with nothing measured leaves the legend. ``n_boot``: bootstrap resamples
    (0 = Wilson on N / steps).
    """
    import plotly.graph_objects as go

    from neural_trade.strategy.signals import SignalFrame

    labels = _labels(frame, config)
    N = len(frame)
    plan = _boot_plan(N, n_boot=n_boot, seed=seed)
    P = np.column_stack([np.asarray(frame.prob(h, True), float) for h in H])
    U = P > 0.5
    votes = U.sum(1)
    q = U.mean(0)
    steps = max(S.horizon_steps(frame, h) for h in H)
    # 1. correlation
    corr = np.corrcoef(P.T) if np.all(P.std(0) > 0) else np.eye(3)
    band = S.corr_null_r(N, steps=steps)                     # r units, not Fisher z
    # 2. vote patterns
    code = (~U[:, 0]) * 4 + (~U[:, 1]) * 2 + (~U[:, 2]) * 1        # UUU = 0 ... DDD = 7, the PATTERNS order
    share = np.bincount(code, minlength=8) / max(N, 1)
    indep = np.array([np.prod([q[k] if pat[k] == "U" else 1 - q[k] for k in range(3)]) for pat in PATTERNS])
    unanimous = share[0] + share[7]
    # the price heads: served (beta x raw) and, when known, raw (resolved as evaluate() does)
    if raw_delta is None:
        raw_delta = _frame_extra(frame, "delta_raw")
    raw_delta = getattr(raw_delta, "delta", raw_delta)       # a raw PredictionFrame works too
    if delta_scale is None:
        delta_scale = _frame_extra(frame, "delta_scale")
    raw_d = _resolve_raw(frame, raw_delta, delta_scale)
    raw = None if raw_d is None else np.column_stack([raw_d[h] for h in H])
    D = np.column_stack([np.asarray(frame.delta[h], float) for h in H])
    zero = ~np.any(D, axis=0)                  # served delta identically 0 (beta = 0): its statistics are n/a
    betas = _served_betas(D, raw, delta_scale)
    zero_h = [h for h, z in zip(H, zero) if z]
    # 4. direction head vs price head: the sign of the raw head (a positive beta keeps it, beta = 0 does not),
    #    as evaluation.report.coherence_block and analytics_tables.alignment_table
    DU = (raw if raw is not None else D) > 0
    sign_na = np.zeros(3, bool) if raw is not None else zero
    align = np.where(sign_na, np.nan, (U == DU).mean(0))
    qd = DU.mean(0)
    align_indep = np.where(sign_na, np.nan, _noskill(q, qd))
    dcode = (~DU[:, 0]) * 4 + (~DU[:, 1]) * 2 + (~DU[:, 2]) * 1
    all3 = np.nan if sign_na.any() else float((U == DU).all(1).mean())
    all3_indep = np.nan if sign_na.any() else float(np.sum(share * (np.bincount(dcode, minlength=8) / max(N, 1))))
    # 5. |delta| ordering
    def order_rates(A):
        a = np.abs(np.asarray(A, float))
        o01, o12 = a[:, 0] <= a[:, 1], a[:, 1] <= a[:, 2]
        return np.array([o01.mean(), o12.mean(), (o01 & o12).mean()])

    order_na = np.array([zero[0] | zero[1], zero[1] | zero[2], zero.any()])   # 0 <= 0 holds trivially
    served_order = np.where(order_na, np.nan, order_rates(D))
    realised = order_rates(frame.y)
    # 6. the strategies' votes (SignalFrame's own rule)
    sf = SignalFrame.build(frame, 1.0)          # agreement / consensus do not depend on var_scale
    agr = np.round(sf.agreement * 3).astype(int)
    agr_share = np.array([(agr == a).mean() for a in (1, 2, 3)])
    agr_up = np.array([((agr == a) & (sf.consensus > 0)).mean() for a in (1, 2, 3)])
    agr_dn = np.array([((agr == a) & (sf.consensus < 0)).mean() for a in (1, 2, 3)])
    gate = float(sf.direction_aligned.mean())               # the strategies' gates, on the served deltas
    mag_gate = float(sf.magnitude_coherent.mean())
    vote_up, vote_dn = _vote_lines()

    zero_txt = ", ".join(zero_h)
    beta_txt = " / ".join(f"{h} {betas[h]:.3g}" for h in H if h in betas)
    if zero.any():
        # beta = 0 on h0 or h2 alone leaves one comparison measured; on h1 (or on two) none is
        if zero.all():
            order_note = f"beta = 0 on {zero_txt}: served delta = 0, ordering n/a"
        else:
            order_note = (f"beta {beta_txt}" if len(betas) == 3 else f"beta = 0 on {zero_txt}") + ": served ordering n/a"
            if not order_na.all():
                order_note += f" where it involves {zero_txt}"
        if raw is None:
            order_note += " · pass raw_delta"
    elif raw is None:
        order_note = "served deltas are shrunk per horizon: pass raw_delta for the raw heads"
    elif len(betas) == 3:
        order_note = f"served = beta x raw · beta {beta_txt}"
    else:
        order_note = "served = beta x raw, with a beta per horizon"
    if raw is not None:
        sign_note = "P(up) &gt; 0.5 vs raw price head &gt; 0 per bar (eval report)"
    elif zero.all():
        sign_note = f"beta = 0 on {zero_txt}: served delta = 0, sign n/a · pass raw_delta"
    elif zero.any():
        sign_note = f"P(up) &gt; 0.5 vs served delta &gt; 0 · n/a on {zero_txt} (beta = 0) · pass raw_delta"
    else:
        sign_note = "P(up) &gt; 0.5 vs served delta &gt; 0, per bar"
    heads = ["P(up) correlation between horizons",
             "Vote pattern h0 h1 h2 (U = P(up) &gt; 0.5)",
             "Realised up-rate by number of horizons voting up",
             "Direction head and price head give the same sign",
             "Does |delta| grow with the horizon?",
             "Strategy vote agreement (SignalFrame.agreement)"]
    notes = [f"shade = |r| · |r| &lt; {band:.2f} is within chance (n_eff = N / {steps} = {N // steps:,})",
             f"light = unanimous UUU / DDD: {unanimous:.1%} of bars · grey = split",
             "skill = the up-rate rising from left to right",
             sign_note,
             order_note,
             (f"a horizon votes only when P(up) &gt; {vote_up:g} or &lt; {vote_dn:g}" if vote_up is not None
              else "a horizon votes only beyond the strategies' P(up) lines")]
    height = int(height or 976)
    margin = dict(t=176, b=64, l=64, r=24)          # four subtitle lines
    plot_h = height - margin["t"] - margin["b"]
    fig = _grid(2, heads, vspace=0.2)

    # (1, 1) correlation: shade by |r| on a neutral ramp, signed value in the cell, diagonal blank
    z = np.where(np.eye(3, dtype=bool), np.nan, np.abs(corr))
    text = [["" if r == k else f"{corr[r, k]:+.2f}" for k in range(3)] for r in range(3)]
    fig.add_trace(go.Heatmap(z=z, x=list(H), y=list(H), zmin=0, zmax=1, showscale=False,
                             colorscale=[[0, T.SURFACE], [1, T.MUTED]], text=text, texttemplate="%{text}",
                             textfont=dict(color=T.INK, size=13), hoverongaps=False,
                             hovertemplate="%{y} vs %{x}: r = %{text}<extra></extra>"), 1, 1)
    fig.update_yaxes(autorange="reversed", showgrid=False, row=1, col=1)
    fig.update_xaxes(showgrid=False, row=1, col=1)

    # (1, 2) the eight vote patterns vs what independent votes would give
    diss = {"UUD": "h2 dissents", "UDU": "h1 dissents", "DUU": "h0 dissents", "DDU": "h2 dissents",
            "DUD": "h1 dissents", "UDD": "h0 dissents", "UUU": "unanimous up", "DDD": "unanimous down"}
    counts = np.bincount(code, minlength=8)
    fig.add_trace(go.Bar(x=list(PATTERNS), y=share, name="share of bars", legend="legend", showlegend=False,
                         marker=dict(color=[T.INK_2 if p in ("UUU", "DDD") else T.MUTED for p in PATTERNS]),
                         text=[f"{v:.0%}" for v in share], textposition="outside", textfont=dict(size=10),
                         customdata=np.column_stack([counts, indep]), hovertext=[diss[p] for p in PATTERNS],
                         hovertemplate="%{x} (%{hovertext}): %{y:.1%} of bars (n %{customdata[0]:,})<br>"
                                       "independent votes would give %{customdata[1]:.1%}<extra></extra>"), 1, 2)
    fig.add_trace(go.Scatter(x=list(PATTERNS), y=indep, mode="markers", name="if the horizons voted independently",
                             legend="legend", marker=TICK,
                             hovertemplate="%{x}: %{y:.1%} if independent<extra></extra>"), 1, 2)
    fig.update_xaxes(type="category", row=1, col=2)
    fig.update_yaxes(tickformat=".0%", range=[0, max(share.max(), indep.max()) * 1.2 + 0.01],
                     title_text="share of bars", title_standoff=4, row=1, col=2)

    # (1, 3) realised up-rate by number of horizons voting up: skill = rising left to right
    base_all = []
    for i, h in enumerate(H):
        lab, mask = labels[h]
        up = (np.asarray(lab) > 0.5).astype(np.float32)
        sub = np.array([mask & (votes == v) for v in range(4)])
        rate, lo, hi, n = _subset_rates(up, sub, plan, S.horizon_steps(frame, h))
        base_all.append(float(up[mask].mean()) if mask.any() else np.nan)
        xs = np.arange(4) + (i - 1) * 0.16
        fig.add_trace(go.Scatter(x=xs, y=rate, mode="markers", name=h, legend="legend2",
                                 marker=dict(color=T.HORIZON_COLORS[h], size=9, line=dict(color=T.PAPER, width=1)),
                                 error_y=dict(type="data", symmetric=False, array=hi - rate, arrayminus=rate - lo,
                                              color=T.HORIZON_COLORS[h], thickness=1.5, width=3),
                                 customdata=np.column_stack([np.arange(4), n, lo, hi]),
                                 hovertemplate=f"{h}: %{{customdata[0]}} horizons vote up<br>realised up-rate "
                                               "%{y:.3f} [%{customdata[2]:.3f}, %{customdata[3]:.3f}] "
                                               "(n %{customdata[1]:,})<extra></extra>"), 1, 3)
    base = float(np.nanmean(base_all))
    fig.add_trace(go.Scatter(x=[-0.4, 3.4], y=[base, base], mode="lines", name="test-block up-rate",
                             legend="legend2", line=dict(color=T.NEUTRAL, dash=REF_DASH, width=1),
                             hovertemplate="up-rate of scored test moves: " +
                                           " · ".join(f"{h} {v:.1%}" for h, v in zip(H, base_all)) +
                                           "<extra></extra>"), 1, 3)
    fig.update_xaxes(tickmode="array", tickvals=[0, 1, 2, 3], ticktext=["0 (all down)", "1", "2", "3 (all up)"],
                     range=[-0.5, 3.5], title_text="horizons with P(up) > 0.5", title_standoff=4, row=1, col=3)
    fig.update_yaxes(tickformat=".0%", title_text="realised up-rate", title_standoff=4, row=1, col=3)

    def _pct(v):
        return "" if not np.isfinite(v) else f"{v:.0%}"

    def _na_marks(row, col, xs):
        """'n/a' on the zero line where a served-delta statistic is undefined (beta = 0), never a measured 0%."""
        sp = fig.get_subplot(row, col)
        for x in xs:
            fig.add_annotation(x=x, y=0, xref=sp.xaxis.plotly_name.replace("axis", ""),
                               yref=sp.yaxis.plotly_name.replace("axis", ""), yanchor="bottom", yshift=2,
                               showarrow=False, text="n/a", font=dict(size=10, color=T.MUTED),
                               hovertext=NA_BETA0)

    # (2, 1) direction head vs price head sign agreement, against independent heads
    cats = list(H) + ["all 3"]
    vals = np.r_[align, all3]
    ind = np.r_[align_indep, all3_indep]
    head_name = "raw price head" if raw is not None else "served delta"
    sign_keys = bool(np.isfinite(vals).any())      # nothing measured (beta = 0, no raw heads): no keys to explain
    fig.add_trace(go.Bar(x=cats, y=vals, name="share of bars", legend="legend3", showlegend=False,
                         marker=dict(color=[T.HORIZON_COLORS[h] for h in H] + [T.INK_2]),
                         text=[_pct(v) for v in vals], textposition="outside", textfont=dict(size=10),
                         customdata=ind, hovertemplate=f"%{{x}}: P(up) > 0.5 and {head_name} > 0 give the "
                                                       "same sign on %{y:.1%} of bars<br>independent heads would "
                                                       "agree on %{customdata:.1%}<extra></extra>"), 2, 1)
    _key(fig, 2, 1, "legend3", "same sign", mode="markers", showlegend=sign_keys,
         marker=dict(color=T.INK_2, **SWATCH))
    fig.add_trace(go.Scatter(x=cats, y=ind, mode="markers", name="if independent",
                             legend="legend3", marker=TICK, showlegend=sign_keys,
                             hovertemplate="%{x}: independent heads would agree on %{y:.1%}<extra></extra>"), 2, 1)
    _na_marks(2, 1, [k for k, v in enumerate(vals) if not np.isfinite(v)])
    fig.update_xaxes(type="category", row=2, col=1)
    fig.update_yaxes(tickformat=".0%", range=[0, 1.08], title_text="share of bars", title_standoff=4, row=2, col=1)

    # (2, 2) |delta| ordering on the raw heads (served deltas are shrunk per horizon; n/a where beta = 0)
    ocats = ["|h0| &#8804; |h1|", "|h1| &#8804; |h2|", "both"]
    series = [("raw price heads", order_rates(raw), T.INK_2)] if raw is not None else []
    series.append(("served deltas (shrunk)" if raw is not None else "deltas as given", served_order, T.MUTED))
    width = 0.8 / len(series)                  # explicit geometry: the 'n/a' marks sit on the served slots
    for i, (name, vals_, color) in enumerate(series):
        fig.add_trace(go.Bar(x=ocats, y=vals_, name=name, legend="legend4", marker=dict(color=color),
                             showlegend=bool(np.isfinite(vals_).any()),     # no swatch for bars never drawn
                             width=width, offset=-0.4 + i * width,
                             text=[_pct(v) for v in vals_], textposition="outside", textfont=dict(size=10),
                             hovertemplate=f"{name}<br>%{{x}} on %{{y:.1%}} of bars<extra></extra>"), 2, 2)
    _na_marks(2, 2, [k + 0.4 - width / 2 for k in np.flatnonzero(order_na)])
    fig.add_trace(go.Scatter(x=ocats, y=realised, mode="markers", name="realised |y|", legend="legend4",
                             marker=TICK, hovertemplate="realised moves: %{x} on %{y:.1%} of bars"
                                                        "<br>(unordered magnitudes: 50% / 50% / 17%)"
                                                        "<extra></extra>"), 2, 2)
    fig.update_xaxes(type="category", row=2, col=2)
    fig.update_yaxes(tickformat=".0%", range=[0, 1.08], title_text="share of bars", title_standoff=4, row=2, col=2)
    fig.update_layout(barmode="group")

    # (2, 3) the strategies' votes: SignalFrame.agreement (a horizon votes only beyond its thresholds)
    acats = ["1 of 3 or none", "2 of 3", "3 of 3"]
    fig.add_trace(go.Bar(x=acats, y=agr_share, name="share of bars", legend="legend5", marker=dict(color=T.MUTED),
                         text=[f"{v:.1%}" for v in agr_share], textposition="outside", textfont=dict(size=10),
                         customdata=np.column_stack([agr_up, agr_dn]),
                         hovertemplate="agreement %{x}: %{y:.1%} of bars<br>consensus up %{customdata[0]:.1%}, "
                                       "down %{customdata[1]:.1%}<extra></extra>"), 2, 3)
    fig.update_xaxes(type="category", title_text="largest group of horizons voting the same side", title_standoff=4,
                     row=2, col=3)
    fig.update_yaxes(tickformat=".0%", range=[0, 1.08], title_text="share of bars", title_standoff=4, row=2, col=3)

    for k, (legend, (r, c_)) in enumerate(((None, (1, 1)), ("legend", (1, 2)), ("legend2", (1, 3)),
                                           ("legend3", (2, 1)), ("legend4", (2, 2)), ("legend5", (2, 3)))):
        if legend == "legend3" and not sign_keys:
            # a legend with no visible key is not drawn, title included: an annotation, level with the row's others
            _heading(fig, None, r, c_, heads[k], plot_h=plot_h, note=notes[k], lift=KEY_ROW_PX)
        else:
            _heading(fig, legend, r, c_, heads[k], plot_h=plot_h, note=notes[k])
    chain = "|h0| &#8804; |h1| &#8804; |h2|"
    if np.isfinite(all3):
        heads_line = (f"P(up) and the {head_name} give the same sign on all 3 horizons: {all3:.1%} "
                      f"({all3_indep:.1%} if independent)")
    else:
        heads_line = f"P(up) and the served delta give the same sign on all 3 horizons: {NA_BETA0}"
    if raw is not None:
        heads_line += f" · {chain} on the raw price heads: {order_rates(raw)[2]:.1%}"
    if zero.any():
        # SignalFrame.direction_aligned then asks P(up) <= 0.5 of those horizons (and the sign match of the
        # others): a pass rate, not an alignment. At beta = 0 on all three it is the DDD share.
        live = ", ".join(h for h, z in zip(H, zero) if not z)
        passes = ("P(up) &#8804; 0.5 on all 3: the DDD share" if not live
                  else f"P(up) &#8804; 0.5 on {zero_txt} and sign match on {live}")
        gate_line = (f"SignalFrame direction_aligned, magnitude_coherent: n/a (beta = 0 on {zero_txt}: served delta is "
                     f"0) · direction_aligned then passes {gate:.1%} of bars ({passes})")
    else:
        gate_line = (f"served deltas{f' (beta {beta_txt})' if betas else ''}: SignalFrame.direction_aligned "
                     f"{gate:.1%} · SignalFrame.magnitude_coherent ({chain}) {mag_gate:.1%}")
    subtitle = (f"{frame.split} block, N = {N:,} bars · calibrated P(up) · a horizon votes up when P(up) &gt; 0.5 "
                f"(the eval report's unanimity)<br>{heads_line}<br>{gate_line}<br>"
                + (CI_NOTE if plan is not None else "95% CI: Wilson on N / steps effective samples")
                + " · grey ticks: what independent votes or heads would give; in the |delta| panel, "
                  "how often the realised moves |y| are so ordered")
    T.apply(fig, title="Cross-horizon coherence", height=height, subtitle=subtitle, legend_top=False)
    fig.update_layout(margin=margin)
    return fig
