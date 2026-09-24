"""evaluate(): the evaluation protocol (plan section C2).

Every direction metric uses the neutral mask (|return| <= deadband is excluded). Reported per
horizon:

    direction        (head; temperature-calibrated when a CalibrationPipeline was fit)
                     mcc auc brier ece_pos acc bal_acc pred_up_rate true_up_rate n_masked, and the
                     confusion matrix of P(up) > 0.5 with UP as the positive class: tp fp tn fn
                     precision recall (= sensitivity) specificity f1
    gauss_direction  the same for the Gaussian readout P(up | move leaves the deadband)
    delta            rmse mae ev corr skill_vs_zero      (never price-level EV: it is ~0.999
                                                          for "no change" and says nothing)
                     rmse_zero mae_zero (predicting 0 on the same block), corr_spearman,
                     mean_pred mean_true (dollars), share_pred_up share_true_up (share of deltas > 0)
    delta_raw        the same for the raw price heads, before the calibration's delta shrink (only
                     when evaluate() is given them: ``raw_delta=``, or ``frame.meta["delta_raw"]``)
    variance         crps nll (dollars) pit_ks corr_var_err2_spearman coverage90 width90 crpss
    confidence_gap   accuracy(high confidence) - accuracy(low confidence) with a moving-block
                     bootstrap CI (block = 80 bars); WORKS only when the CI excludes 0 and
                     the gap is at least 1 pp
    n_eff            N // horizon_steps: non-overlapping outcomes, the honest sample size

Across horizons (``coherence``):

    mag_h0_le_h1 mag_h1_le_h2 mag_order_full   share of samples with |d_h0| <= |d_h1| (<= |d_h2|) on the
                                               deltas in the frame: the SERVED deltas, whose ordering
                                               after a per-horizon shrink beta mostly reflects the betas
    ..._raw                                    the same on the raw price heads, the ordering the loss
                                               trains (when the raw heads are known)
    delta_dir_align_h0/h1/h2, _all             sign(delta) agrees with calibrated P(up) > 0.5 (raw heads
                                               when known; the same as the served deltas while beta > 0)
    delta_dir_align_indep_h0/h1/h2, _all       the agreement two independent signs would show
    coherence_primary                          = delta_dir_align_h1 (kept for older readers)
    unanimity                                  all three P(up) on the same side of 0.5

Baselines (fit on train, see evaluation.baselines) are scored with the same code.
``beats_baseline`` records, per baseline and relevant metric, whether the model is better;
``baseline_margins`` keeps both values, the margin (positive = the model is better) and a noise test
of the margin (|z| < 1.96: the win, or the loss, is inside the noise):

    dm_z        metrics that are means of a per-sample loss (rmse and skill_vs_zero: squared error,
                mae, brier, acc: error rate, crps, nll): the Diebold-Mariano z of the paired loss
                difference, its long-run variance from a Bartlett (Newey-West) kernel with
                lag = DM_LAG_PER_STEP x bars ahead (consecutive targets overlap for bars-ahead - 1
                bars, and volatility clusters beyond that)
    boot_z      the other ranked metrics (mcc auc bal_acc ece_pos, ev corr, pit_ks
    boot_se     corr_var_err2_spearman): the margin over its standard error from a paired moving-block
                bootstrap (BLOCK bars, BOOT_N resamples; the model and the baseline are rescored on the
                same resampled bars). A z, not a percentile interval: ECE and PIT KS are biased upward in
                resamples, which shifts a percentile interval but hardly changes the spread.

The markdown verdict: "beats" / "does not beat" (the point values), then "noise" when |z| < 1.96, or
"significantly worse" when the model loses with z <= -1.96.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from scipy.special import ndtr
from scipy.stats import spearmanr

from neural_trade.evaluation.frame import HORIZONS, PredictionFrame
from neural_trade.metrics import numpy_metrics as npm
from neural_trade.metrics.direction_labels import direction_labels_np

# precision / recall / specificity / f1 are deliberately NOT here: a constant baseline (class_prior
# calls one side only) has recall 0 or 1 and an undefined precision, so a "beats" on them is empty.
HIGHER_IS_BETTER = {"mcc", "auc", "acc", "bal_acc", "ev", "corr", "skill_vs_zero", "corr_var_err2_spearman", "crpss"}
LOWER_IS_BETTER = {"brier", "ece_pos", "rmse", "mae", "crps", "nll", "pit_ks"}
DOLLAR_METRICS = {"rmse", "mae", "crps", "rmse_zero", "mae_zero", "mean_pred", "mean_true", "width90"}
BLOCK = 80
Z95 = 1.959964
DM_LAG_PER_STEP = 2      # Bartlett lag of the Diebold-Mariano long-run variance, in multiples of bars ahead
BOOT_N = 500             # resamples of the paired block bootstrap of the ranked baseline metrics
NOISE_TESTS = {"dm_z": f"Diebold-Mariano, Bartlett (Newey-West) long-run variance, lag {DM_LAG_PER_STEP} x bars "
                       "ahead",
               "boot_z": f"paired moving-block bootstrap, block {BLOCK} bars, {BOOT_N} resamples: margin / its "
                         "bootstrap standard error"}

# Baseline rows that only restate the RMSE verdict for a constant prediction (its EV and corr are 0 and
# its skill is a monotone function of its RMSE): left out of the markdown, kept in beats_baseline.
RESTATED_BASELINE_ROWS = {("zero_delta", "delta/ev"), ("zero_delta", "delta/corr"), ("zero_delta", "delta/skill_vs_zero"),
                          ("mean_delta", "delta/ev"), ("mean_delta", "delta/corr"),
                          ("mean_delta", "delta/skill_vs_zero")}
BASELINE_ORDER = ("logreg_lags", "class_prior", "zero_delta", "mean_delta", "const_var")
# a served-delta statistic where the calibration's shrink beta is 0 (the served delta is 0 on every sample)
SERVED_ZERO_NA = "n/a (beta = 0: served delta is 0)"


def _auc(labels, scores):
    from sklearn.metrics import roc_auc_score

    labels = np.asarray(labels)
    if len(labels) < 2 or labels.min() == labels.max():
        return float("nan")
    return float(roc_auc_score(labels, scores))


def _ratio(num, den) -> float:
    return float(num) / float(den) if den else float("nan")


def direction_block(labels, mask, prob) -> Dict[str, float]:
    t, p = labels[mask], np.asarray(prob)[mask]
    pb, up = p > 0.5, t > 0.5
    tp, tn = int(np.sum(pb & up)), int(np.sum(~pb & ~up))
    fp, fn = int(np.sum(pb & ~up)), int(np.sum(~pb & up))
    sens, spec = _ratio(tp, tp + fn), _ratio(tn, tn + fp)
    rates = [v for v in (sens, spec) if math.isfinite(v)]
    return {
        "mcc": npm.mcc(t, p), "auc": _auc(t, p), "brier": npm.brier(t, p), "ece_pos": npm.ece_pos(t, p),
        "acc": npm.direction_accuracy(t, p), "bal_acc": float(np.mean(rates)) if rates else float("nan"),
        "pred_up_rate": float(pb.mean()) if len(pb) else float("nan"),
        "true_up_rate": float(t.mean()) if len(t) else float("nan"), "n_masked": int(mask.sum()),
        "precision": _ratio(tp, tp + fp), "recall": sens, "specificity": spec,
        "f1": _ratio(2 * tp, 2 * tp + fp + fn), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def _spearman(a, b) -> float:
    """Rank correlation; 0.0 when either side is constant (like numpy_metrics.corr)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return 0.0
    return float(spearmanr(a, b).correlation)


def delta_block(y, delta) -> Dict[str, float]:
    y, delta = np.asarray(y, float), np.asarray(delta, float)
    mse = float(np.mean((y - delta) ** 2))
    mse0 = float(np.mean(y ** 2))
    return {"rmse": math.sqrt(mse), "mae": float(np.mean(np.abs(y - delta))),
            "ev": npm.explained_variance(y, delta), "corr": npm.corr(y, delta),
            "skill_vs_zero": 1.0 - mse / mse0 if mse0 > 0 else float("nan"),
            "rmse_zero": math.sqrt(mse0), "mae_zero": float(np.mean(np.abs(y))),
            "corr_spearman": _spearman(y, delta),
            "mean_pred": float(np.mean(delta)), "mean_true": float(np.mean(y)),
            "share_pred_up": float(np.mean(delta > 0)), "share_true_up": float(np.mean(y > 0))}


def gaussian_crps(y, mu, sigma):
    """Closed-form CRPS of N(mu, sigma^2) at y (same units as y)."""
    s = np.maximum(sigma, 1e-12)
    z = (y - mu) / s
    return s * (z * (2 * ndtr(z) - 1) + 2 * np.exp(-0.5 * z ** 2) / math.sqrt(2 * math.pi) - 1 / math.sqrt(math.pi))


def gaussian_nll(y, mu, sigma):
    """Per-sample negative log-likelihood of N(mu, sigma^2) at y (sigma in the units of y)."""
    s = np.maximum(sigma, 1e-12)
    return 0.5 * np.log(2 * math.pi * s ** 2) + (y - mu) ** 2 / (2 * s ** 2)


def variance_block(y, mu, sigma, interval=None) -> Dict[str, float]:
    s = np.maximum(sigma, 1e-12)
    err2 = (y - mu) ** 2
    out = {
        "crps": float(np.mean(gaussian_crps(y, mu, s))),
        "nll": float(np.mean(gaussian_nll(y, mu, s))),
        "pit_ks": npm.pit_ks(y, mu, variance=s ** 2),
        # (0.0 when either side is constant: e.g. a constant-variance baseline, or zero served deltas)
        "corr_var_err2_spearman": (float(spearmanr(s ** 2, err2).correlation)
                                   if np.ptp(s ** 2) > 0 and np.ptp(err2) > 0 else 0.0),
        "sigma_dispersion": float(np.std(s) / np.mean(s)) if np.mean(s) > 0 else 0.0,
    }
    if interval is not None:
        lo, hi = (np.asarray(interval[0], float), np.asarray(interval[1], float))
        out["coverage90"] = float(np.mean((y >= lo) & (y <= hi)))
        out["width90"] = float(np.mean(hi - lo))
    return out


def confidence_gap(correct, confidence, threshold, *, block=BLOCK, n_boot=1000, seed=0) -> Dict[str, Any]:
    """accuracy(conf > threshold) - accuracy(conf <= threshold), moving-block bootstrap 95% CI."""
    correct = np.asarray(correct, dtype=float)
    hi = np.asarray(confidence) > threshold
    n = len(correct)
    if n < 2 * block or hi.all() or (~hi).all():
        return {"gap": float("nan"), "ci": [float("nan"), float("nan")], "verdict": "INSUFFICIENT",
                "threshold": float(threshold)}
    gap = correct[hi].mean() - correct[~hi].mean()
    rng = np.random.default_rng(seed)
    nb = int(math.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, nb))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :n]
    c, h = correct[idx], hi[idx]
    with np.errstate(invalid="ignore", divide="ignore"):
        gaps = (c * h).sum(1) / h.sum(1) - (c * ~h).sum(1) / (~h).sum(1)
    lo, up = np.nanpercentile(gaps, [2.5, 97.5])
    if lo > 0 and gap >= 0.01:
        verdict = "WORKS"
    elif up < 0 and gap <= -0.01:
        verdict = "INVERTED"
    else:
        verdict = "NOISE"
    return {"gap": float(gap), "ci": [float(lo), float(up)], "verdict": verdict, "threshold": float(threshold),
            "acc_high": float(correct[hi].mean()), "acc_low": float(correct[~hi].mean()), "n_high": int(hi.sum())}


def _delta_dict(delta, n: int) -> Dict[str, np.ndarray]:
    return {h: np.asarray(delta[h], dtype=float).reshape(-1)[:n] for h in HORIZONS}


def magnitude_ordering(d: np.ndarray):
    """(share |d0| <= |d1|, share |d1| <= |d2|, share of the full chain) for an [N, 3] array."""
    a = np.abs(np.asarray(d, float))
    le01, le12 = a[:, 0] <= a[:, 1], a[:, 1] <= a[:, 2]
    return float(le01.mean()), float(le12.mean()), float((le01 & le12).mean())


def independent_agreement(a: np.ndarray, b: np.ndarray) -> float:
    """Share of rows where the boolean [N, k] arrays ``a`` and ``b`` agree on every column, expected if the
    rows of ``a`` and ``b`` were paired at random: the sum over the 2^k patterns of P_a(pattern) P_b(pattern)."""
    from itertools import product

    a, b = np.asarray(a, bool), np.asarray(b, bool)
    total = 0.0
    for pattern in product((False, True), repeat=a.shape[1]):
        pat = np.array(pattern)
        total += float((a == pat).all(1).mean()) * float((b == pat).all(1).mean())
    return total


def coherence_block(frame: PredictionFrame, raw_delta: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, float]:
    """Cross-horizon checks. ``raw_delta``: the price heads before the calibration's delta shrink."""
    served = np.stack([frame.delta[h] for h in HORIZONS], 1)
    raw = None if raw_delta is None else np.stack(list(_delta_dict(raw_delta, len(frame)).values()), 1)
    p = np.stack([frame.prob(h) for h in HORIZONS], 1)
    head_up = p > 0.5
    delta_up = (raw if raw is not None else served) > 0  # a positive shrink keeps the sign; beta = 0 would not
    agree = head_up == delta_up
    s01, s12, sfull = magnitude_ordering(served)
    out = {
        "mag_order_full": sfull,
        "unanimity": float(np.mean(head_up.all(1) | (~head_up).all(1))),
        "delta_dir_align_all": float(np.mean(agree.all(1))),
        "coherence_primary": float(np.mean(agree[:, 1])),
        "mag_h0_le_h1": s01, "mag_h1_le_h2": s12,
    }
    for i, h in enumerate(HORIZONS):
        out[f"delta_dir_align_{h}"] = float(np.mean(agree[:, i]))
    for i, h in enumerate(HORIZONS):
        pd_, ph = float(delta_up[:, i].mean()), float(head_up[:, i].mean())
        out[f"delta_dir_align_indep_{h}"] = pd_ * ph + (1 - pd_) * (1 - ph)
    out["delta_dir_align_indep_all"] = independent_agreement(delta_up, head_up)
    if raw is not None:
        r01, r12, rfull = magnitude_ordering(raw)
        out.update(mag_h0_le_h1_raw=r01, mag_h1_le_h2_raw=r12, mag_order_full_raw=rfull)
    return out


def score_frame(frame: PredictionFrame, deadband_bps: float, conf_threshold=None, *,
                raw_delta: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, Any]:
    labels = direction_labels_np(frame.y, frame.last_close, deadband_bps)
    raw = None if raw_delta is None else _delta_dict(raw_delta, len(frame))
    out: Dict[str, Any] = {"horizons": {}, "coherence": coherence_block(frame, raw)}
    for i, h in enumerate(HORIZONS):
        lab, mask = labels[h]
        yt = frame.y[:, i]
        prob = frame.prob(h)
        gauss = frame.gauss_prob(h, deadband_bps)
        row = {
            "direction": direction_block(lab, mask, prob),
            "gauss_direction": direction_block(lab, mask, gauss),
            "delta": delta_block(yt, frame.delta[h]),
            "variance": variance_block(yt, frame.delta[h], frame.sigma(h),
                                       frame.intervals.get(h) if frame.intervals else None),
            "n": len(frame), "n_eff": len(frame) // max(1, int(frame.horizon_steps[i])),
        }
        if raw is not None:
            row["delta_raw"] = delta_block(yt, raw[h])
        conf = np.abs(prob - 0.5)
        thr = (conf_threshold or {}).get(h, float(np.median(conf[mask]))) if mask.any() else 0.0
        correct = ((prob > 0.5) == (lab > 0.5)).astype(float)
        row["confidence_gap"] = confidence_gap(correct[mask], conf[mask], thr)
        row["confidence_gap"]["threshold_source"] = "cal" if conf_threshold else "self"
        out["horizons"][h] = row
    return out


# ------------------------------------------------------------------ paired loss differences (baselines)
def _sample_loss(frame: PredictionFrame, i: int, group: str, metric: str, labels) -> Optional[np.ndarray]:
    """Per-sample loss whose mean orders the model and a baseline like ``metric`` does (None: no such loss).

    One value per bar; direction losses are NaN inside the deadband (unscored), so the array keeps time."""
    h = HORIZONS[i]
    y = frame.y[:, i]
    if group == "delta" and metric in ("rmse", "skill_vs_zero"):
        return (y - frame.delta[h]) ** 2
    if group == "delta" and metric == "mae":
        return np.abs(y - frame.delta[h])
    if group == "direction" and metric in ("brier", "acc"):
        lab, mask = labels[h]
        p = np.asarray(frame.prob(h), float)
        loss = (p - lab) ** 2 if metric == "brier" else ((p > 0.5) != (lab > 0.5)).astype(float)
        return np.where(mask, loss, np.nan)
    if group == "variance" and metric in ("crps", "nll"):
        fn = gaussian_crps if metric == "crps" else gaussian_nll
        return fn(y, frame.delta[h], np.maximum(frame.sigma(h), 1e-12))
    return None


def long_run_variance(x: np.ndarray, lag: int) -> float:
    """Bartlett (Newey-West) long-run variance of a time-ordered series; NaN entries (unscored bars) keep their
    place in time and add nothing. Divide by the number of finite entries for the variance of their mean."""
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    n = int(ok.sum())
    if n < 2:
        return float("nan")
    z = np.where(ok, x - x[ok].mean(), 0.0)
    lrv = float(z @ z) / n
    for k in range(1, min(int(lag), len(z) - 1) + 1):
        lrv += 2.0 * (1.0 - k / (lag + 1.0)) * float(z[k:] @ z[:-k]) / n
    return lrv


def dm_z(loss_model: np.ndarray, loss_base: np.ndarray, steps: int, *, lag: Optional[int] = None) -> Optional[float]:
    """Diebold-Mariano z of mean(loss_base - loss_model) (> 0: the model is better).

    Both arrays are per bar, time-ordered (NaN = unscored). The variance of the mean difference is the
    Bartlett (Newey-West) long-run variance with ``lag`` bars (default DM_LAG_PER_STEP x ``steps``): h-bar
    targets overlap for h - 1 bars, so the differences are autocorrelated at least that far. None when fewer
    than 2 non-overlapping outcomes are scored or the difference is constant.
    """
    diff = np.asarray(loss_base, float) - np.asarray(loss_model, float)
    steps = max(1, int(steps))
    n = int(np.isfinite(diff).sum())
    if n // steps < 2:
        return None
    lrv = long_run_variance(diff, DM_LAG_PER_STEP * steps if lag is None else int(lag))
    if not lrv > 0:
        return None
    return float(np.nanmean(diff) / math.sqrt(lrv / n))


# ------------------------------------------------------------------ paired block bootstrap (ranked metrics)
def block_bootstrap_counts(n: int, *, block: int = BLOCK, n_boot: int = BOOT_N, seed: int = 0) -> np.ndarray:
    """[n_boot, n] multiplicities of each bar in moving-block bootstrap resamples of a length-n series."""
    block = max(1, min(int(block), n))
    rng = np.random.default_rng(seed)
    nb = int(math.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, nb))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_boot, -1)[:, :n]
    flat = (idx + (np.arange(n_boot) * n)[:, None]).ravel()
    return np.bincount(flat, minlength=n_boot * n).reshape(n_boot, n).astype(float)


def _ties(x: np.ndarray):
    """(order, starts, group): the sort order of x, where each tie group starts in it, and each sorted item's group."""
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    starts = np.flatnonzero(np.r_[True, xs[1:] != xs[:-1]])
    return order, starts, np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, len(x)]))


def _w_group_midranks(W: np.ndarray, starts: np.ndarray, order: np.ndarray):
    """(G, mid): per resample (rows of W: multiplicities), the weight of each tie group and its mid-rank."""
    G = np.add.reduceat(np.take(W, order, axis=1), starts, axis=1)
    return G, np.cumsum(G, axis=1) - G + (G + 1.0) / 2.0


def _w_pearson(W: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Weighted Pearson correlation of two [n] series per row of W; 0 where either side is constant."""
    a, b = a - a.mean(), b - b.mean()
    sw = W.sum(1)
    ma, mb = W @ a / sw, W @ b / sw
    cov, va, vb = W @ (a * b) / sw - ma * mb, W @ (a * a) / sw - ma * ma, W @ (b * b) / sw - mb * mb
    with np.errstate(invalid="ignore", divide="ignore"):
        r = cov / np.sqrt(va * vb)
    return np.where((va > 0) & (vb > 0), r, 0.0)


def _w_spearman(W: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Spearman correlation (Pearson on tie-averaged ranks) per weighted resample; 0 where a side is constant."""
    ox, sx, gx = _ties(x)
    oy, sy, gy = _ties(y)
    Gx, mx = _w_group_midranks(W, sx, ox)
    Gy, my = _w_group_midranks(W, sy, oy)
    group_x = np.empty(len(x), int)
    group_x[ox] = gx
    sw = W.sum(1)
    m = (sw + 1.0) / 2.0                                       # the mean rank of every resample
    cross = (np.take(W, oy, axis=1) * my[:, gy] * np.take(mx, group_x[oy], axis=1)).sum(1) / sw - m * m
    vx, vy = (Gx * mx * mx).sum(1) / sw - m * m, (Gy * my * my).sum(1) / sw - m * m
    with np.errstate(invalid="ignore", divide="ignore"):
        r = cross / np.sqrt(vx * vy)
    return np.where((vx > 1e-9) & (vy > 1e-9), r, 0.0)


def _boot_metric(frame: PredictionFrame, i: int, group: str, metric: str, labels, W: np.ndarray):
    """``metric`` of ``frame`` on every bootstrap resample (rows of W), computed like the report computes it
    (with W = 1 it equals the report's value); None for metrics without a bootstrap here."""
    h = HORIZONS[i]
    y = frame.y[:, i]
    if group == "direction" and metric in ("mcc", "auc", "bal_acc", "ece_pos"):
        lab, mask = labels[h]
        p = np.asarray(frame.prob(h), float)
        Wm = W * mask
        up = lab > 0.5
        if metric == "ece_pos":
            pc = np.clip(p, 0.0, 1.0)
            idx = np.clip(np.floor(pc * 10).astype(int), 0, 9)
            onehot = (idx[:, None] == np.arange(10)) * (up.astype(float) - pc)[:, None]
            return np.abs(Wm @ onehot).sum(1) / Wm.sum(1)
        if metric == "auc":  # Mann-Whitney U on the mid-ranks of P(up) among the scored bars of each resample
            npos, nneg = Wm @ up, Wm @ ~up
            order, starts, _ = _ties(p)
            _, mid = _w_group_midranks(Wm, starts, order)
            gpos = np.add.reduceat(np.take(Wm * up, order, axis=1), starts, axis=1)
            u = (gpos * mid).sum(1) - npos * (npos + 1.0) / 2.0
            with np.errstate(invalid="ignore", divide="ignore"):
                return np.where((npos > 0) & (nneg > 0), u / (npos * nneg), np.nan)
        pb = p > 0.5
        tp, tn = Wm @ (pb & up), Wm @ (~pb & ~up)
        fp, fn = Wm @ (pb & ~up), Wm @ (~pb & up)
        if metric == "mcc":
            den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
            with np.errstate(invalid="ignore", divide="ignore"):
                return np.where(den > 0, (tp * tn - fp * fn) / np.sqrt(np.where(den > 0, den, 1.0)), 0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            sens, spec = tp / (tp + fn), tn / (tn + fp)
        return np.where(np.isfinite(sens) & np.isfinite(spec), (sens + spec) / 2.0,
                        np.where(np.isfinite(sens), sens, spec))
    if group == "delta" and metric in ("corr", "ev"):
        d = np.asarray(frame.delta[h], float)
        if metric == "corr":
            return np.zeros(len(W)) if np.ptp(d) == 0 or np.ptp(y) == 0 else _w_pearson(W, y, d)
        sw = W.sum(1)
        e, yc = y - d, y - y.mean()
        e = e - e.mean()
        ve = (W @ (e * e)) / sw - ((W @ e) / sw) ** 2
        vy = (W @ (yc * yc)) / sw - ((W @ yc) / sw) ** 2
        with np.errstate(invalid="ignore", divide="ignore"):
            return 1.0 - ve / vy
    if group == "variance" and metric in ("pit_ks", "corr_var_err2_spearman"):
        s = np.maximum(frame.sigma(h), 1e-12)
        mu = frame.delta[h]
        if metric == "pit_ks":
            u = ndtr((y - mu) / np.sqrt(np.maximum(s ** 2, 1e-12)))
            order = np.argsort(u, kind="mergesort")
            us, Ws = u[order], np.take(W, order, axis=1)
            sw = Ws.sum(1)[:, None]
            after = np.cumsum(Ws, axis=1) / sw
            return np.maximum((after - us).max(1), (us - after + Ws / sw).max(1))
        v, err2 = s ** 2, (y - mu) ** 2
        return np.zeros(len(W)) if np.ptp(v) == 0 or np.ptp(err2) == 0 else _w_spearman(W, v, err2)
    return None


def _boot_margin(metric: str, bm: np.ndarray, bb: np.ndarray, margin) -> Dict[str, Any]:
    """The margin's bootstrap standard error and z (margin / se) from paired resampled metric values."""
    diffs = (bm - bb) if metric in HIGHER_IS_BETTER else (bb - bm)
    diffs = diffs[np.isfinite(diffs)]
    if len(diffs) < 20 or margin is None or not math.isfinite(margin):
        return {}
    se = float(diffs.std(ddof=1))
    return {"boot_se": se, "boot_z": float(margin / se) if se > 0 else None}


def baseline_margin(metric: str, m, b) -> Dict[str, Any]:
    finite = m is not None and b is not None and math.isfinite(m) and math.isfinite(b)
    out: Dict[str, Any] = {"model": m, "baseline": b,
                           "margin": ((m - b) if metric in HIGHER_IS_BETTER else (b - m)) if finite else float("nan")}
    if metric in DOLLAR_METRICS and finite and b != 0:
        out["margin_pct"] = 100.0 * out["margin"] / abs(b)
    return out


@dataclass
class EvalReport:
    run_id: Optional[str]
    split: str
    deadband_bps: float
    n: int
    model: Dict[str, Any]
    baselines: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    beats_baseline: Dict[str, Dict[str, Dict[str, bool]]] = field(default_factory=dict)
    backtest: Optional[Dict[str, Any]] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    baseline_margins: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = field(default_factory=dict)

    # ------------------------------------------------------------------ views
    def metric(self, h: str, group: str, name: str) -> float:
        return self.model["horizons"][h][group][name]

    def flat(self) -> Dict[str, float]:
        out = {}
        for h, row in self.model["horizons"].items():
            for group in ("direction", "gauss_direction", "delta", "delta_raw", "variance"):
                for k, v in (row.get(group) or {}).items():
                    out[f"{h}/{group}/{k}"] = v
            out[f"{h}/confidence_gap/gap"] = row["confidence_gap"]["gap"]
            out[f"{h}/n_eff"] = row["n_eff"]
        for k, v in self.model["coherence"].items():
            out[f"coherence/{k}"] = v
        if self.backtest:
            for k, v in (self.backtest.get("summary") or {}).items():
                out[f"backtest/{k}"] = v
        return out

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, path=None) -> str:
        text = json.dumps(self.to_dict(), indent=2, default=_jsonable)
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    def margin(self, name: str, key: str, h: str) -> Dict[str, Any]:
        """Model value, baseline value, margin (> 0: the model is better) and noise test (dm_z or boot_z) for one cell."""
        got = ((self.baseline_margins.get(name) or {}).get(key) or {}).get(h)
        if got is not None:
            return got
        group, metric = key.split("/", 1)
        m = ((self.model["horizons"].get(h) or {}).get(group) or {}).get(metric)
        b = ((((self.baselines.get(name) or {}).get("horizons") or {}).get(h) or {}).get(group) or {}).get(metric)
        return baseline_margin(metric, m, b)

    # ------------------------------------------------------------------ markdown
    def to_markdown(self, path=None) -> str:
        steps = list(self.meta.get("horizon_steps") or [])
        head = [f"{h} ({steps[i]} bars)" if i < len(steps) else h for i, h in enumerate(HORIZONS)]
        L = [f"# Evaluation report - {self.split} split" + (f" - run `{self.run_id}`" if self.run_id else ""),
             "", f"n = {self.n} samples, one per 1-minute bar. Direction metrics count only moves beyond "
             f"{self.deadband_bps:g} bps (the neutral mask). Consecutive samples share most of their target "
             "window, so n_eff = n // bars ahead counts the non-overlapping outcomes.", ""]
        L += self._md_direction(head) + self._md_delta(head) + self._md_variance(head)
        L += ["## Confidence gap (accuracy of the more confident half minus the less confident half)", "",
              "| horizon | gap | 95% CI | verdict |", "|---|---|---|---|"]
        for h in HORIZONS:
            g = self.model["horizons"][h]["confidence_gap"]
            L.append(f"| {h} | {_fmt(g['gap'])} | [{_fmt(g['ci'][0])}, {_fmt(g['ci'][1])}] | {g['verdict']} |")
        L += [""] + self._md_coherence()
        if self.beats_baseline:
            L += self._md_baselines(head)
        if self.backtest:
            L += ["", "## Backtest (costs included)", ""]
            L += [f"- {k}: {_fmt(v)}" for k, v in (self.backtest.get("summary") or {}).items()]
        text = "\n".join(L) + "\n"
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text

    def _get(self, h, group, key):
        row = self.model["horizons"].get(h) or {}
        return row.get(key) if group is None else (row.get(group) or {}).get(key)

    def _table(self, head, rows):
        L = ["| metric | " + " | ".join(head) + " |", "|---|" + "---|" * len(head)]
        for label, fn in rows:
            L.append(f"| {label} | " + " | ".join(fn(h) for h in HORIZONS) + " |")
        return L + [""]

    def _cell(self, group, key, fmt=None):
        fmt = fmt or _fmt
        return lambda h: fmt(self._get(h, group, key))

    def _md_direction(self, head):
        d, g = "direction", "gauss_direction"

        def counts(h):
            r = self.model["horizons"][h].get(d) or {}
            vals = [r.get(k) for k in ("tp", "fp", "tn", "fn")]
            return "n/a" if any(v is None for v in vals) else " / ".join(str(int(v)) for v in vals)

        def const_ece(h):
            u = self._get(h, d, "true_up_rate")
            return _fmt(abs(u - 0.5)) if u is not None and math.isfinite(u) else "n/a"

        steps = list(self.meta.get("horizon_steps") or [])

        def n_eff_scored(h):
            i, n = HORIZONS.index(h), self._get(h, d, "n_masked")
            return "n/a" if n is None or i >= len(steps) else str(int(n) // max(1, int(steps[i])))

        rows = [("n scored (outside the deadband)", self._cell(d, "n_masked")),
                ("n_eff of the scored moves (n scored // bars ahead)", n_eff_scored),
                ("true up-rate", self._cell(d, "true_up_rate")),
                ("calls up (predicted up-rate)", self._cell(d, "pred_up_rate")),
                ("accuracy", self._cell(d, "acc")), ("balanced accuracy", self._cell(d, "bal_acc")),
                ("precision (up)", self._cell(d, "precision")),
                ("recall / sensitivity (up)", self._cell(d, "recall")),
                ("specificity (down)", self._cell(d, "specificity")), ("F1 (up)", self._cell(d, "f1")),
                ("MCC", self._cell(d, "mcc")), ("AUC", self._cell(d, "auc")), ("Brier", self._cell(d, "brier")),
                ("ECE (positive class)", self._cell(d, "ece_pos")),
                ("ECE of a constant 0.5 (= distance of the up-rate from 0.5)", const_ece),
                ("TP / FP / TN / FN", counts),
                ("Gaussian readout: calls up", self._cell(g, "pred_up_rate")),
                ("Gaussian readout: MCC", self._cell(g, "mcc")), ("Gaussian readout: AUC", self._cell(g, "auc")),
                ("Gaussian readout: Brier", self._cell(g, "brier")),
                ("Gaussian readout: ECE", self._cell(g, "ece_pos"))]
        return [f"## Direction heads: P(up) > 0.5 on moves beyond {self.deadband_bps:g} bps", "",
                "Up is the positive class. Temperature scaling does not move P(up) across 0.5, so the "
                "counts and rates are the same before and after calibration; Brier and ECE are not.", ""] \
            + self._table(head, rows)

    def _md_delta(self, head):
        has_raw = any(self.model["horizons"][h].get("delta_raw") for h in HORIZONS)
        betas = self.meta.get("delta_scale") or {}
        s, r = "delta", "delta_raw"
        sv = "served"  # the frame's deltas: PredictionFrame.from_result holds what the predictor serves
        zero_beta = {h for h in HORIZONS if betas.get(h) is not None and float(betas[h]) <= 0}

        def served(key):
            """A served-delta statistic that is fixed by construction where beta = 0 (the served delta is
            0 there: skill and EV 0, no correlation): n/a, not a measured 0.0000."""
            cell = self._cell(s, key)
            return lambda h: SERVED_ZERO_NA if h in zero_beta else cell(h)

        rows = [(f"RMSE ($), {sv}", self._cell(s, "rmse", _usd))]
        rows += [("RMSE ($), raw heads", self._cell(r, "rmse", _usd))] if has_raw else []
        rows += [("RMSE ($), zero prediction", self._cell(s, "rmse_zero", _usd)),
                 (f"MAE ($), {sv}", self._cell(s, "mae", _usd))]
        rows += [("MAE ($), raw heads", self._cell(r, "mae", _usd))] if has_raw else []
        rows += [("MAE ($), zero prediction", self._cell(s, "mae_zero", _usd)),
                 (f"skill vs zero (1 - MSE / MSE of 0), {sv}", served("skill_vs_zero"))]
        rows += [("skill vs zero, raw heads", self._cell(r, "skill_vs_zero"))] if has_raw else []
        rows += [(f"EV, {sv}", served("ev"))]
        rows += [("EV, raw heads", self._cell(r, "ev"))] if has_raw else []
        # corr(beta x raw, y) = corr(raw, y) only while beta > 0: with the raw heads, print theirs
        rows += ([("corr, Pearson, raw heads (the same for served while beta > 0)", self._cell(r, "corr")),
                  ("corr, Spearman, raw heads", self._cell(r, "corr_spearman"))] if has_raw else
                 [("corr, Pearson", served("corr")), ("corr, Spearman", served("corr_spearman"))])
        rows += [(f"mean predicted ($), {sv}", self._cell(s, "mean_pred", _usd))]
        rows += [("mean predicted ($), raw heads", self._cell(r, "mean_pred", _usd))] if has_raw else []
        rows += [("mean realised ($)", self._cell(s, "mean_true", _usd)),
                 ("share predicted up" + (", raw heads" if has_raw else ""),
                  self._cell(r if has_raw else s, "share_pred_up")),
                 ("share realised up", self._cell(s, "share_true_up"))]
        if betas:
            rows.append(("shrink beta (served = beta x raw, fit on cal)",
                         lambda h: _fmt(betas.get(h)) if betas.get(h) is not None else "n/a"))
        note = [] if has_raw else [
            "Scored on the deltas in the frame, as served: beta x the raw price head when the calibration fits a "
            "delta shrink, the raw head otherwise. The raw heads were not passed to evaluate() (raw_delta=), so "
            "their errors are not shown.", ""]
        return ["## Price heads (dollars)", ""] + note + self._table(head, rows)

    def _md_variance(self, head):
        v = "variance"
        rows = [("CRPS ($)", self._cell(v, "crps", _usd)), ("CRPSS vs constant variance", self._cell(v, "crpss")),
                ("NLL", self._cell(v, "nll")), ("PIT KS", self._cell(v, "pit_ks")),
                ("var / err^2 Spearman", self._cell(v, "corr_var_err2_spearman")),
                ("coverage of the 90% interval", self._cell(v, "coverage90")),
                ("width of the 90% interval ($)", self._cell(v, "width90", _usd))]
        return ["## Variance heads", ""] + self._table(head, rows)

    def _md_coherence(self):
        c = self.model["coherence"]
        betas = self.meta.get("delta_scale") or {}
        has_raw = "mag_order_full_raw" in c
        served_head = "served deltas" + (" (beta-shrunk: " + " / ".join(
            f"{h} {betas[h]:.3f}" for h in HORIZONS if betas.get(h) is not None) + ")" if betas else "")
        cols = (["raw price heads (the trained ordering)"] if has_raw else []) + [served_head, "realised moves"]
        L = ["## Coherence across horizons", "",
             "Magnitude ordering: the share of samples whose predicted move grows with the horizon, as the loss "
             "asks. Magnitudes in random order would give 0.5, 0.5 and 0.1667.", "",
             "| check | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
        realised = self.meta.get("realised_ordering") or [None, None, None]
        for j, (label, key) in enumerate((("abs(d h0) <= abs(d h1)", "mag_h0_le_h1"),
                                          ("abs(d h1) <= abs(d h2)", "mag_h1_le_h2"),
                                          ("full chain h0 <= h1 <= h2", "mag_order_full"))):
            vals = ([_fmt(c.get(f"{key}_raw"))] if has_raw else []) + [_fmt(c.get(key)), _fmt(realised[j])]
            L.append(f"| {label} | " + " | ".join(vals) + " |")
        zero_beta = [h for h in HORIZONS if betas.get(h) is not None and float(betas[h]) <= 0]
        if zero_beta:
            L += ["", f"beta = 0 for {', '.join(zero_beta)}: the served delta is 0 there, so every sign and "
                  "magnitude check on the served deltas is empty" + ("; the sign checks below use the raw heads."
                                                                    if has_raw else ".")]
        if has_raw:
            L += ["", "The served ordering mostly reflects the per-horizon shrink beta, not the model: judge the "
                  "trained constraint on the raw heads."]
        else:
            L += ["", "Scored on the deltas in the frame (the served deltas when the calibration shrinks them); "
                  "pass raw_delta to evaluate() to score the raw heads."]
        L += ["", "Sign agreement: sign(delta) against calibrated P(up) > 0.5 (the same for raw and served deltas "
              "while beta > 0).", "", "| | " + " | ".join(HORIZONS) + " | all 3 |", "|---|---|---|---|---|",
              "| agree | " + " | ".join(_fmt(c.get(f"delta_dir_align_{h}", c.get("coherence_primary") if h == "h1"
                                                   else None)) for h in HORIZONS)
              + f" | {_fmt(c.get('delta_dir_align_all'))} |",
              "| expected if the two signs were independent | "
              + " | ".join(_fmt(c.get(f"delta_dir_align_indep_{h}")) for h in HORIZONS)
              + f" | {_fmt(c.get('delta_dir_align_indep_all'))} |", "",
              f"- P(up) unanimity (all three horizons call the same side): {_fmt(c.get('unanimity'))}"]
        return L

    def _md_baselines(self, head):
        names = [n for n in BASELINE_ORDER if n in self.beats_baseline] + \
                [n for n in self.beats_baseline if n not in BASELINE_ORDER]
        tests = self.meta.get("noise_tests")
        if tests:
            how = (f"\"DM z\": Diebold-Mariano test of the per-sample loss difference (RMSE and skill, MAE, Brier, "
                   f"accuracy, CRPS, NLL) with a Bartlett (Newey-West) long-run variance, lag {DM_LAG_PER_STEP} x "
                   f"bars ahead. \"boot z\": the margin over its standard error in a paired moving-block bootstrap "
                   f"({BLOCK}-bar blocks, {BOOT_N} resamples; MCC, AUC, balanced accuracy, ECE, EV, corr, PIT KS, "
                   f"var / err^2 Spearman). ")
        elif any(mg.get("dm_z") is not None for per_key in self.baseline_margins.values()
                 for per_h in per_key.values() for mg in per_h.values()):
            how = ("This report predates the current noise tests: its z divides the loss difference's sd by "
                   "sqrt(n // bars ahead), which is conservative (it calls some real differences noise), and the "
                   "ranked metrics have none; re-score it with evaluate(). ")
        else:
            how = "This report has no noise test (written before it existed); re-score it with evaluate(). "
        L = ["", "## Against baselines (fit on the training block)", "",
             "Each cell: model vs baseline, the margin (positive = the model is better) and the verdict. "
             "\"noise\": |z| < 1.96, so the ordering is not established; \"significantly worse\": the model "
             "loses with z <= -1.96. " + how
             + "Rows that only restate the RMSE verdict for a constant prediction (EV, corr and "
             "skill against zero_delta / mean_delta) are left out; the JSON keeps every verdict.", "",
             "| baseline | metric | " + " | ".join(head) + " |", "|---|---|" + "---|" * len(head)]
        for name in names:
            for key, per_h in self.beats_baseline[name].items():
                if (name, key) in RESTATED_BASELINE_ROWS:
                    continue
                metric = key.split("/", 1)[1]
                cells = [_margin_cell(metric, self.margin(name, key, h), bool(per_h.get(h))) for h in HORIZONS]
                L.append(f"| {name} | {key} | " + " | ".join(cells) + " |")
        return L


def margin_z(mg: Dict[str, Any]):
    """(z, test) of one baseline margin: the Diebold-Mariano z ("DM") when the metric is a mean per-sample loss,
    else the block-bootstrap z ("boot"); (None, "") when neither was computed (e.g. an older report)."""
    for key, test in (("dm_z", "DM"), ("boot_z", "boot")):
        z = (mg or {}).get(key)
        if z is not None and math.isfinite(float(z)):
            return float(z), test
    return None, ""


def _margin_cell(metric: str, mg: Dict[str, Any], beats: bool) -> str:
    fmt = _usd if metric in DOLLAR_METRICS else _fmt
    m, b, margin = mg.get("model"), mg.get("baseline"), mg.get("margin")
    verdict = "beats" if beats else "does not beat"
    if margin is None or not math.isfinite(margin):
        return f"{fmt(m)} vs {fmt(b)}: {verdict}"
    pct = mg.get("margin_pct")
    size = f"{margin:+.2f}" if metric in DOLLAR_METRICS else f"{margin:+.4f}"
    if pct is not None and math.isfinite(pct):
        size += f", {pct:+.2f}%"
    return f"{fmt(m)} vs {fmt(b)} ({size}): {verdict}{noise_suffix(*margin_z(mg))}"


def noise_suffix(z: Optional[float], test: str) -> str:
    """', noise (DM z -1.23)' inside the noise, ', significantly worse (DM z -2.07)' for a real loss, else
    ' (DM z +5.76)'; '' without a test."""
    if z is None:
        return ""
    tag = f"({test} z {z:+.2f})"
    if abs(z) < Z95:
        return f", noise {tag}"
    return f", significantly worse {tag}" if z < 0 else f" {tag}"


def _fmt(v):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "n/a"
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return f"{v:.4f}"


def _usd(v):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "n/a"
    return f"{float(v):.2f}"


def _jsonable(o):
    if isinstance(o, (np.generic,)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _metric_groups(name):
    from neural_trade.evaluation.baselines import RELEVANT

    return RELEVANT.get(name, ())


def _frame_extra(frame: PredictionFrame, name: str):
    """``frame.<name>`` or ``frame.meta[name]``: how a frame can carry its raw heads / betas to evaluate()."""
    got = getattr(frame, name, None)
    if got is None:
        got = (getattr(frame, "meta", None) or {}).get(name)
    return got


def _resolve_raw(frame: PredictionFrame, raw_delta, delta_scale) -> Optional[Dict[str, np.ndarray]]:
    """The raw price heads: given, carried by the frame (``delta_raw``), or served / beta when every beta > 0."""
    if raw_delta is None:
        raw_delta = _frame_extra(frame, "delta_raw")
    if raw_delta is not None:
        return _delta_dict(raw_delta, len(frame))
    if delta_scale and all(float(delta_scale.get(h, 0.0)) > 0 for h in HORIZONS):
        return {h: frame.delta[h] / float(delta_scale[h]) for h in HORIZONS}
    return None


def _betas(frame: PredictionFrame, raw, delta_scale) -> Dict[str, float]:
    if delta_scale:
        return {h: float(delta_scale[h]) for h in HORIZONS if h in delta_scale}
    if raw is None:
        return {}
    out = {}
    for h in HORIZONS:
        dd = float(np.dot(raw[h], raw[h]))
        out[h] = float(np.dot(frame.delta[h], raw[h]) / dd) if dd > 0 else float("nan")
    return out


def evaluate(frame: PredictionFrame, config, *, baselines=None, cal_frame: Optional[PredictionFrame] = None,
             run_id: Optional[str] = None, backtest: Optional[Dict[str, Any]] = None,
             raw_delta: Optional[Dict[str, np.ndarray]] = None,
             delta_scale: Optional[Dict[str, float]] = None) -> EvalReport:
    """Score ``frame`` (and every fitted baseline) under the protocol above.

    ``raw_delta``: the price heads before the calibration's delta shrink (e.g. ``result.predictions["delta"]``);
    with it the report also scores the raw heads (``delta_raw``) and the magnitude ordering the loss trains.
    ``delta_scale``: the calibration's per-horizon beta (served = beta x raw); recorded in ``meta`` and, when
    every beta > 0 and no raw heads are given, used to recover them. Both default to what the frame carries
    (``frame.delta_raw`` / ``frame.delta_scale`` or the same keys of ``frame.meta``).
    """
    deadband = float(getattr(config, "DIR_DEADBAND_BPS", 0.0))
    if delta_scale is None:
        delta_scale = _frame_extra(frame, "delta_scale")
    conf_thr = None
    if cal_frame is not None:  # confidence threshold from the CALIBRATION split, not the test split
        labels = direction_labels_np(cal_frame.y, cal_frame.last_close, deadband)
        conf_thr = {h: float(np.median(np.abs(cal_frame.prob(h) - 0.5)[labels[h][1]])) for h in HORIZONS
                    if labels[h][1].any()}
    raw = _resolve_raw(frame, raw_delta, delta_scale)
    model = score_frame(frame, deadband, conf_thr, raw_delta=raw)
    meta: Dict[str, Any] = {"horizon_steps": list(frame.horizon_steps), "pred_scale": frame.pred_scale,
                            "realised_ordering": list(magnitude_ordering(frame.y[:, :3]))}
    betas = _betas(frame, raw, delta_scale)
    if betas:
        meta["delta_scale"] = betas
    report = EvalReport(run_id, frame.split, deadband, len(frame), model, backtest=backtest, meta=meta)
    if baselines is not None:
        labels = direction_labels_np(frame.y, frame.last_close, deadband)
        W = block_bootstrap_counts(len(frame))       # the same resampled bars for the model and every baseline
        boot_model: Dict[Any, Any] = {}
        meta["noise_tests"] = dict(NOISE_TESTS)
        for name, bframe in baselines.predict(frame).items():
            scored = score_frame(bframe, deadband)
            report.baselines[name] = scored
            beats: Dict[str, Dict[str, bool]] = {}
            margins: Dict[str, Dict[str, Dict[str, Any]]] = {}
            for group in _metric_groups(name):
                for metric in (model["horizons"]["h1"][group] or {}):
                    if metric not in HIGHER_IS_BETTER | LOWER_IS_BETTER:
                        continue
                    key = f"{group}/{metric}"
                    beats[key], margins[key] = {}, {}
                    for i, h in enumerate(HORIZONS):
                        m, b = model["horizons"][h][group].get(metric), scored["horizons"][h][group].get(metric)
                        ok = (m is not None and b is not None and math.isfinite(m) and math.isfinite(b)
                              and ((m > b) if metric in HIGHER_IS_BETTER else (m < b)))
                        beats[key][h] = bool(ok)
                        mg = baseline_margin(metric, m, b)
                        lm = _sample_loss(frame, i, group, metric, labels)
                        lb = _sample_loss(bframe, i, group, metric, labels)
                        mg["dm_z"] = (dm_z(lm, lb, int(frame.horizon_steps[i]))
                                      if lm is not None and lb is not None else None)
                        if lm is None:
                            if (i, group, metric) not in boot_model:
                                boot_model[(i, group, metric)] = _boot_metric(frame, i, group, metric, labels, W)
                            bm = boot_model[(i, group, metric)]
                            bb = _boot_metric(bframe, i, group, metric, labels, W) if bm is not None else None
                            if bb is not None:
                                mg.update(_boot_margin(metric, bm, bb, mg["margin"]))
                        margins[key][h] = mg
            report.beats_baseline[name] = beats
            report.baseline_margins[name] = margins
        if "const_var" in report.baselines:
            for h in HORIZONS:
                c_m = model["horizons"][h]["variance"]["crps"]
                c_b = report.baselines["const_var"]["horizons"][h]["variance"]["crps"]
                model["horizons"][h]["variance"]["crpss"] = 1.0 - c_m / c_b if c_b > 0 else float("nan")
    return report
