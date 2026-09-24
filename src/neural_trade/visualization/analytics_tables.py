"""The numbers behind the analytics figures, as tables: what the old notebooks printed, plus references.

Pure functions returning pandas DataFrames (``run_settings_table`` also reads the run directory):

* :func:`classification_table` - per horizon, the confusion matrix of P(up) > 0.5 outside the deadband
  and every rate built from it (accuracy, balanced accuracy, precision, recall, specificity, F1, MCC),
  AUC, Brier, ECE, predicted vs true up-rate, with the chance references next to them.
* :func:`delta_quality_table` - the price heads in dollars: RMSE / MAE of the raw heads, the served
  (shrunk) deltas and a zero prediction, skill, EV, correlations, mean and spread of the predictions.
* :func:`magnitude_ordering_table` - does |delta| grow with the horizon, on the raw heads, the served
  deltas and the realised moves, with Wilson intervals on n_eff.
* :func:`alignment_table` - does sign(delta) agree with P(up) > 0.5, per horizon and on all three.
* :func:`baseline_table` - the report's baseline comparison with both values, the margin and a noise
  test (HAC Diebold-Mariano z for losses, paired block-bootstrap z for ranked metrics), not only
  "beats / does not beat".
* :func:`trailing_move_table` - the extended-trend feature (the trailing move over each horizon,
  also the momentum prior of the loss) against the realised move, per block.
* :func:`run_settings_table` - a saved run's configuration next to the values it actually used
  (auto-calibrated loss weights, final learning rates, fitted calibration).
* :func:`scored_counts_text`, :func:`confusion_title`, :func:`direction_stats_line`,
  :func:`delta_summary_text` - the same numbers as one-line captions for figure titles.

Consecutive 1-minute samples have overlapping targets, so every interval uses n_eff = N / bars ahead
(:mod:`neural_trade.visualization.stats`), like the evaluation report; the price-head correlation band
uses the delta figure's N / deff (Bartlett design effect), so both print the same +/-. A served delta
with beta = 0 is the constant 0: its correlation, sign and magnitude ordering print ``ZERO_BETA_NA``
(or the raw heads' values, labelled so), never the 0.0 / 1.0 fallback, and its Gaussian readout (the
constant 0.5) ``GAUSS_CONST_NA``. Tables with metrics as rows and
horizons as columns mix counts and rates: their cells are rounded to ``digits`` (dollars to
``usd_digits``) and kept as numbers; intervals are "[lo, hi]" strings. What a table cannot say in its
labels is in ``df.attrs["caption"]``, which :func:`styled` shows under it.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import numpy as np
import pandas as pd

from neural_trade.evaluation.report import (
    BASELINE_ORDER, BLOCK, BOOT_N, DM_LAG_PER_STEP, DOLLAR_METRICS, GAUSS_CONST_NA, HORIZONS, RESTATED_BASELINE_ROWS,
    Z95, ZERO_BETA_NA, baseline_margin, coherence_block, delta_block, direction_block, independent_agreement,
    magnitude_ordering, margin_z,
)
from neural_trade.metrics.direction_labels import direction_labels_np
from neural_trade.visualization import stats as S
from neural_trade.visualization.analytics_delta import corr_band, design_effect

__all__ = ["classification_table", "delta_quality_table", "magnitude_ordering_table", "alignment_table",
           "baseline_table", "trailing_move_table", "run_settings_table", "styled", "scored_counts_text",
           "confusion_title", "direction_stats_line", "delta_summary_text"]

MARGIN = "model better by"
OUTER_LAMBDAS = ("LAMBDA_TREND_OUTER", "LAMBDA_DIR_OUTER", "LAMBDA_NLL_OUTER")  # rescaled only when CALIB_OUTER
# LAMBDA_* fields that are not loss weights (losses/functions.py reads them as thresholds)
THRESHOLD_LAMBDAS = {"LAMBDA_VAC": "a threshold on the cross-horizon spread of the price heads, not a weight (the "
                                   "vacuum term enters the loss unweighted); 0 = off"}
# terms the total loss scales by a fixed factor on top of their LAMBDA_* weight (losses/functions.py)
FIXED_LOSS_SCALE = {"LAMBDA_INTER": 0.1, "LAMBDA_VOL": 0.1}


# ------------------------------------------------------------------ helpers
def _deadband(config, deadband_bps) -> float:
    if deadband_bps is not None:
        return float(deadband_bps)
    return float(getattr(config, "DIR_DEADBAND_BPS", 0.0) or 0.0) if config is not None else 0.0


def _num(v, digits: Optional[int]):
    """A plain Python number (rounded), a string, or NaN."""
    if v is None:
        return float("nan")
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        v = float(v) + 0.0                      # + 0.0: -0.0 (0 x a negative raw head) prints as 0.00
        return round(v, digits) + 0.0 if digits is not None and math.isfinite(v) else v
    return v


def _interval(lo, hi, digits: Optional[int]) -> str:
    if lo is None or hi is None or not (math.isfinite(lo) and math.isfinite(hi)):
        return "n/a"
    d = 4 if digits is None else digits
    return f"[{lo:.{d}f}, {hi:.{d}f}]"


def _object_frame(cols: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(cols, dtype=object)
    df.index.name = "metric"
    df.columns.name = "horizon"
    return df


def _corr(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _raw_or(frame, raw_delta):
    return {h: np.asarray(raw_delta[h], float).reshape(-1)[:len(frame)] for h in HORIZONS} if raw_delta else None


def _zero_columns(arr) -> list:
    """Indices of the columns of an [N, 3] array that are 0 on every row (a served delta with beta = 0)."""
    arr = np.asarray(arr, float)
    return [i for i in range(arr.shape[1]) if len(arr) and not np.any(arr[:, i])]


def _constant_na(pred, served: bool) -> Optional[str]:
    """None for a prediction that varies; else why its correlation and sign share are not shown."""
    pred = np.asarray(pred, float)
    if len(pred) and np.ptp(pred) > 0:
        return None
    return ZERO_BETA_NA if served and not np.any(pred) else "n/a (constant prediction)"


# ------------------------------------------------------------------ direction
def classification_table(frame, config=None, *, calibrated: bool = True, readout: str = "head",
                         deadband_bps: Optional[float] = None, digits: Optional[int] = 4) -> pd.DataFrame:
    """Per-horizon classification numbers of P(up) > 0.5 on moves beyond the deadband (up = positive).

    ``readout``: "head" (the direction head; ``calibrated`` picks the temperature-scaled P(up)) or
    "gaussian" (the price head's Gaussian readout). Rows are metrics, columns h0 / h1 / h2. The
    references: "majority-class accuracy" (always calling the block's more common side, known only in
    hindsight), the Brier of a constant equal to the block's up-rate, and the ECE of a constant 0.5.
    Where the served delta is 0 on every bar (beta = 0) the Gaussian readout is the constant 0.5: its
    calls, rates, AUC, Brier and ECE are fixed by construction, so they read GAUSS_CONST_NA (the
    block's own numbers and the references stay) and ``df.attrs["caption"]`` names the horizons.
    """
    if readout not in ("head", "gaussian"):
        raise ValueError(f"readout must be 'head' or 'gaussian', got {readout!r}")
    deadband = _deadband(config, deadband_bps)
    labels = direction_labels_np(frame.y, frame.last_close, deadband)
    cols: Dict[str, Dict[str, Any]] = {}
    const_h = []                    # horizons whose Gaussian readout is the constant 0.5 (served delta 0)
    for h in HORIZONS:
        lab, mask = labels[h]
        p = np.asarray(frame.prob(h, calibrated) if readout == "head" else frame.gauss_prob(h, deadband), float)
        d = direction_block(lab, mask, p)
        steps = S.horizon_steps(frame, h)
        n_scored, u = d["n_masked"], d["true_up_rate"]
        _, lo, hi = S.wilson(d["tp"] + d["tn"], max(n_scored, 1), steps=steps)
        n_pos, n_neg = d["tp"] + d["fn"], d["tn"] + d["fp"]
        auc_lo, auc_hi = (S.auc_ci(d["auc"], n_pos, n_neg, steps=steps)
                          if math.isfinite(d["auc"]) and n_pos and n_neg else (float("nan"), float("nan")))
        c = {
            "bars ahead": steps,
            "n samples": len(frame),
            f"n scored (moves beyond {deadband:g} bps)": n_scored,
            "n_eff of the scored moves (n scored // bars ahead)": n_scored // max(1, steps),
            "true up-rate": u,
            "calls up (predicted up-rate)": d["pred_up_rate"],
            "accuracy": d["acc"],
            "accuracy 95% CI (n_eff)": _interval(float(lo), float(hi), digits) if n_scored else "n/a",
            "majority-class accuracy (hindsight)": max(u, 1 - u) if math.isfinite(u) else float("nan"),
            "balanced accuracy": d["bal_acc"],
            "precision (up)": d["precision"],
            "recall / sensitivity (up)": d["recall"],
            "specificity (down)": d["specificity"],
            "F1 (up)": d["f1"],
            "MCC": d["mcc"],
            "AUC": d["auc"],
            "AUC 95% CI (n_eff)": _interval(auc_lo, auc_hi, digits),
            "Brier": d["brier"],
            "Brier of a constant = the up-rate (hindsight)": u * (1 - u) if math.isfinite(u) else float("nan"),
            "ECE (positive class)": d["ece_pos"],
            "ECE of a constant 0.5": abs(u - 0.5) if math.isfinite(u) else float("nan"),
            "TP (called up, went up)": d["tp"],
            "FP (called up, went down)": d["fp"],
            "TN (called down, went down)": d["tn"],
            "FN (called down, went up)": d["fn"],
            "mean P(up), all samples": float(p.mean()) if len(p) else float("nan"),
            "std P(up), all samples": float(p.std()) if len(p) else float("nan"),
        }
        if readout == "gaussian" and len(p) and not np.any(np.asarray(frame.delta[h], float)):
            const_h.append(h)       # never calls up, AUC 0.5, Brier 0.25 by construction: not measured
            keep = {"bars ahead", "n samples", f"n scored (moves beyond {deadband:g} bps)",
                    "n_eff of the scored moves (n scored // bars ahead)", "true up-rate",
                    "majority-class accuracy (hindsight)", "Brier of a constant = the up-rate (hindsight)",
                    "ECE of a constant 0.5", "mean P(up), all samples", "std P(up), all samples"}
            c = {k: (v if k in keep else GAUSS_CONST_NA) for k, v in c.items()}
        cols[h] = {k: _num(v, digits) for k, v in c.items()}
    df = _object_frame(cols)
    if const_h:
        df.attrs["caption"] = (f"beta = 0 for {', '.join(const_h)}: the served delta is 0 on every bar there, so the "
                               "Gaussian readout is the constant 0.5 (it never calls up; AUC 0.5 and Brier 0.25 by "
                               "construction): its calls and rates are n/a.")
    return df


# ------------------------------------------------------------------ price heads
def delta_quality_table(frame, config=None, *, raw_delta: Optional[Dict[str, np.ndarray]] = None,
                        delta_scale: Optional[Dict[str, float]] = None, digits: Optional[int] = 4,
                        usd_digits: Optional[int] = 2, tail: float = 0.995) -> pd.DataFrame:
    """The price heads in dollars, per horizon: raw heads vs served (shrunk) deltas vs a zero prediction.

    ``raw_delta``: the heads before the calibration's delta shrink. served = beta x raw has the same
    correlations and signs as the raw heads while beta > 0 (errors, skill and magnitudes differ); at
    beta = 0 the served delta is the constant 0, which has no correlation and no sign. So with
    ``raw_delta`` every correlation, sign and min / max row is the raw heads' (labelled ", raw heads"),
    and the caption names the horizons whose served delta is 0. Without it those rows are the frame's
    deltas', and a delta that is 0 on every bar shows ``ZERO_BETA_NA`` instead of the 0.0 fallback.
    ``delta_scale``: the calibration's beta per horizon (default: the least-squares ratio served / raw).
    Without either, the frame's deltas are labelled "model" (a served frame holds beta x raw, a raw
    frame the heads; ``df.attrs["caption"]`` says so). ``tail``: the correlation "without the largest
    predictions" drops |prediction| above this quantile, to show how much a few outliers carry. The
    correlation noise band is the delta figure's: 95% on N / deff effective samples, deff the Bartlett
    design effect of the head and the outcome (analytics_delta.design_effect). "LS slope" is the
    unconstrained least-squares slope of the realised move on the head; the calibration fits
    clip(slope, 0, 1) on the calibration block, shown next to it.
    """
    raw = _raw_or(frame, raw_delta)
    sv = "served" if raw is not None or delta_scale else "model"
    rw = ", raw heads" if raw is not None else f", {sv}"
    cols: Dict[str, Dict[str, Any]] = {}
    zero = []                       # horizons whose served delta is 0 on every bar (beta = 0)
    for i, h in enumerate(HORIZONS):
        y = frame.y[:, i]
        s = np.asarray(frame.delta[h], float)
        d = raw[h] if raw is not None else s
        steps = S.horizon_steps(frame, h)
        bs, br = delta_block(y, s), (delta_block(y, raw[h]) if raw is not None else None)
        dd = float(np.dot(d, d))
        if delta_scale and h in delta_scale:
            beta = float(delta_scale[h])
        else:
            beta = float(np.dot(s, d) / dd) if raw is not None and dd > 0 else None
        keep = np.abs(d) <= np.quantile(np.abs(d), tail) if len(d) else np.ones(0, bool)
        m, mlo, mhi = S.mean_ci(y, steps=steps)
        c: list = [("bars ahead", steps), ("n samples", len(frame)),
                   ("n_eff (non-overlapping outcomes)", len(frame) // max(1, steps))]
        if beta is not None:
            c.append(("shrink beta (served = beta x raw)", beta))
        for metric, name in (("rmse", "RMSE ($)"), ("mae", "MAE ($)")):
            c += [(f"{name}, raw heads", br[metric])] if br is not None else []
            c += [(f"{name}, {sv}", bs[metric]), (f"{name}, zero prediction", bs[f"{metric}_zero"])]
        for metric, name in (("skill_vs_zero", "skill vs zero (1 - MSE / MSE of 0)"), ("ev", "explained variance")):
            c += [(f"{name}, raw heads", br[metric])] if br is not None else []
            c += [(f"{name}, {sv}", bs[metric])]
        slope = float(np.dot(y, d)) / dd if dd > 0 else None
        on = "raw head" if br is not None else ("served delta" if sv == "served" else "model delta")
        if len(s) and not np.any(s):
            zero.append(h)
        # correlations and signs of d: the raw heads when known, else the frame's deltas (none if constant)
        na = _constant_na(d, sv == "served") if br is None else None
        cs = {k: na for k in ("corr", "corr_spearman", "share_pred_up")} if na is not None else (br or bs)
        c += [(f"corr, Pearson{rw}", cs["corr"]),
              ("corr noise band +/- (95%, N / deff, Bartlett)",
               na if na is not None else corr_band(len(y), design_effect(d, y, steps))),
              (f"corr, Spearman{rw}", cs["corr_spearman"]),
              (f"corr without the {100 * (1 - tail):g}% largest |prediction|{rw}",
               na if na is not None else _corr(d[keep], y[keep])),
              (f"LS slope of realised on the {on} (unclipped)", slope)]
        if br is not None:
            c.append(("beta this block would fit: clip(LS slope, 0, 1)",
                      min(max(slope, 0.0), 1.0) if slope is not None else None))
        c += [("mean predicted ($), raw heads", br["mean_pred"])] if br is not None else []
        c += [(f"mean predicted ($), {sv}", bs["mean_pred"]), ("mean realised ($)", m),
              ("mean realised 95% CI ($, n_eff)", _interval(mlo, mhi, usd_digits)),
              (f"share predicted up (delta > 0){rw}", cs["share_pred_up"]),
              ("share realised up (move > 0)", bs["share_true_up"])]
        c += [("std predicted ($), raw heads", float(np.std(d)))] if br is not None else []
        c += [(f"std predicted ($), {sv}", float(np.std(s))), ("std realised ($)", float(np.std(y)))]
        c += [("median |predicted| ($), raw heads", float(np.median(np.abs(d))))] if br is not None else []
        c += [(f"median |predicted| ($), {sv}", float(np.median(np.abs(s)))),
              (f"min predicted ($){rw}", float(np.min(d))), (f"max predicted ($){rw}", float(np.max(d)))]
        cols[h] = {k: _num(v, usd_digits if "($" in k else digits) for k, v in c}
    df = _object_frame(cols)
    if raw is None and not delta_scale:
        df.attrs["caption"] = ("\"model\" = the deltas in the frame: beta x the raw price head in a served frame, the "
                               "raw head in a raw one. Pass raw_delta= to compare raw heads, served deltas and zero.")
    notes = []
    if raw is not None:
        notes.append("Correlations, signs and min / max are the raw heads': the served delta, beta x raw, has the same "
                     "correlations and signs while beta > 0.")
    if zero and sv == "served":
        notes.append(f"beta = 0 for {', '.join(zero)}: the served delta is 0 there, the zero prediction (its errors "
                     "are the zero prediction's), so it has no correlation and no sign"
                     + (" of its own." if raw is not None else "; those rows are n/a. Pass raw_delta= for the raw "
                                                               "heads."))
    if notes:
        df.attrs["caption"] = " ".join(notes)
    return df


# ------------------------------------------------------------------ across horizons
def magnitude_ordering_table(frame, *, raw_delta: Optional[Dict[str, np.ndarray]] = None,
                             digits: Optional[int] = 4) -> pd.DataFrame:
    """Share of samples with |delta_h0| <= |delta_h1| (<= |delta_h2|).

    Columns: the raw heads (the ordering the loss trains; only with ``raw_delta``), the served deltas
    (after a per-horizon shrink beta this mostly reflects the betas), the realised moves, each with its
    95% Wilson interval on n / bars-ahead effective samples (the longer horizon of the check), and the
    value for magnitudes in random order (1/2, 1/2, 1/6).

    A served delta with beta = 0 is 0 on every bar: |0| <= |0| holds by ties, so a check that involves
    such a horizon measures nothing. Its share is NaN, its interval cell says ``ZERO_BETA_NA`` and
    ``df.attrs["caption"]`` names the horizons (the same for any series with a column of zeros).
    """
    raw = _raw_or(frame, raw_delta)
    served = np.stack([frame.delta[h] for h in HORIZONS], 1)
    n = len(frame)
    steps = [max(S.horizon_steps(frame, a), S.horizon_steps(frame, b)) for a, b in (("h0", "h1"), ("h1", "h2"))]
    steps.append(max(S.horizon_steps(frame, h) for h in HORIZONS))
    involved = ((0, 1), (1, 2), (0, 1, 2))
    series = {}
    if raw is not None:
        series["raw heads (trained ordering)"] = np.stack([raw[h] for h in HORIZONS], 1)
    series["served deltas"] = served
    series["realised moves"] = frame.y[:, :3]
    cols: Dict[str, list] = {}
    zero_served = []
    for name, arr in series.items():
        zero = set(_zero_columns(arr))
        if name == "served deltas":
            zero_served = [HORIZONS[i] for i in sorted(zero)]
        void = [bool(zero & set(ix)) for ix in involved]
        shares = magnitude_ordering(arr)
        cols[name] = [float("nan") if v else (round(s, digits) if digits is not None else s)
                      for s, v in zip(shares, void)]
        cis = [S.wilson(round(s * n), n, steps=st)[1:] for s, st in zip(shares, steps)]
        why = ZERO_BETA_NA if name == "served deltas" else "n/a (0 on every bar)"
        cols[f"{name.split(' (')[0]} 95% CI (n_eff)"] = [why if v else _interval(float(lo), float(hi), digits)
                                                         for (lo, hi), v in zip(cis, void)]
    cols["magnitudes in random order"] = [round(v, digits) if digits is not None else v for v in (0.5, 0.5, 1 / 6)]
    cols["n_eff"] = [n // max(1, st) for st in steps]
    df = pd.DataFrame(cols, index=["|d h0| <= |d h1|", "|d h1| <= |d h2|", "full chain |d h0| <= |d h1| <= |d h2|"])
    df.index.name = "check"
    if zero_served:
        df.attrs["caption"] = (f"beta = 0 for {', '.join(zero_served)}: the served delta is 0 on every bar there, so "
                               "|0| <= |0| holds by ties and the served checks that involve it are n/a"
                               + ("; the raw heads carry the ordering the loss trains." if raw is not None
                                  else "; pass raw_delta= for the raw heads."))
    return df


def alignment_table(frame, config=None, *, raw_delta: Optional[Dict[str, np.ndarray]] = None,
                    calibrated: bool = True, digits: Optional[int] = 4) -> pd.DataFrame:
    """Does sign(delta) agree with P(up) > 0.5? Per horizon and on all three at once, on every sample.

    The sign of delta is the same raw or served while beta > 0 (``raw_delta`` protects against beta = 0).
    Without ``raw_delta``, a horizon whose delta is 0 on every bar (beta = 0) has no sign: its row, and
    "all 3", are NaN, and ``df.attrs["caption"]`` says so.
    "expected if independent": the agreement two unrelated signs with these up-shares would show
    (for "all 3", summed over the eight up/down patterns of the three horizons). The 95% interval is a
    Wilson interval on n / bars-ahead effective samples (the longest horizon for "all 3").
    """
    raw = _raw_or(frame, raw_delta)
    D = np.stack([np.asarray((raw if raw is not None else frame.delta)[h], float) for h in HORIZONS], 1)
    zero = set() if raw is not None else set(_zero_columns(D))
    d = D > 0
    p = np.stack([np.asarray(frame.prob(h, calibrated), float) for h in HORIZONS], 1) > 0.5
    n = len(frame)
    nan = float("nan")
    rows = {}
    for i, h in enumerate(HORIZONS):
        steps = S.horizon_steps(frame, h)
        agree = d[:, i] == p[:, i]
        k = int(agree.sum())
        _, lo, hi = S.wilson(k, n, steps=steps)
        sd, sp = float(d[:, i].mean()), float(p[:, i].mean())
        rows[h] = {"agree": k / n, "95% CI low": float(lo), "95% CI high": float(hi),
                   "expected if independent": sd * sp + (1 - sd) * (1 - sp),
                   "share delta > 0": sd, "share P(up) > 0.5": sp, "n": n, "n_eff": n // max(1, steps)}
        if i in zero:
            rows[h].update({k_: nan for k_ in ("agree", "95% CI low", "95% CI high", "expected if independent",
                                               "share delta > 0")})
    steps = max(S.horizon_steps(frame, h) for h in HORIZONS)
    k = int((d == p).all(1).sum())
    _, lo, hi = S.wilson(k, n, steps=steps)
    indep = independent_agreement(d, p)
    rows["all 3"] = {"agree": nan if zero else k / n, "95% CI low": nan if zero else float(lo),
                     "95% CI high": nan if zero else float(hi), "expected if independent": nan if zero else indep,
                     "share delta > 0": nan, "share P(up) > 0.5": nan, "n": n, "n_eff": n // max(1, steps)}
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "horizon"
    df[["n", "n_eff"]] = df[["n", "n_eff"]].astype(int)
    if digits is not None:
        num = [c for c in df.columns if c not in ("n", "n_eff")]
        df[num] = df[num].round(digits)
    if zero:
        df.attrs["caption"] = (f"The delta is 0 on every bar for {', '.join(HORIZONS[i] for i in sorted(zero))} "
                               "(a served delta with beta = 0): it has no sign, so those rows and \"all 3\" are n/a. "
                               "Pass raw_delta= to score the raw heads.")
    return df


# ------------------------------------------------------------------ baselines
def _load_report(report) -> Dict[str, Any]:
    if isinstance(report, (str, Path)):
        return json.loads(Path(report).read_text(encoding="utf-8"))
    if hasattr(report, "to_dict"):
        return report.to_dict()
    return dict(report)


def baseline_table(report, *, compact: bool = True, digits: Optional[int] = 4) -> pd.DataFrame:
    """The report's comparison with every baseline, with the numbers.

    ``report``: an EvalReport, its dict, or the path of an eval_report_*.json. One row per (baseline,
    metric); per horizon: the model value, the baseline value, "model better by" (the margin, oriented
    so that positive = the model is better), "z" and the verdict ("noise" when |z| < 1.96). z is the
    Diebold-Mariano statistic of the paired per-sample loss difference with a Bartlett (Newey-West)
    long-run variance for RMSE / skill, MAE, Brier, accuracy, CRPS and NLL ("DM" in the verdict), and
    the margin over its paired moving-block bootstrap standard error for MCC, AUC, balanced accuracy,
    ECE, EV, corr, PIT KS and the var / err^2 Spearman ("boot"); blank when the report has neither.
    A report written before these tests carries a conservative z (sd / sqrt(n_eff)); ``df.attrs["caption"]``
    says so. ``compact`` drops the rows that only restate the RMSE verdict for a constant prediction (EV,
    corr, skill against zero_delta / mean_delta).
    """
    rep = _load_report(report)
    beats = rep.get("beats_baseline") or {}
    margins = rep.get("baseline_margins") or {}
    names = [n for n in BASELINE_ORDER if n in beats] + [n for n in beats if n not in BASELINE_ORDER]
    current = bool((rep.get("meta") or {}).get("noise_tests"))
    rows, index = [], []
    for name in names:
        for key, per_h in beats[name].items():
            if compact and (name, key) in RESTATED_BASELINE_ROWS:
                continue
            group, metric = key.split("/", 1)
            row: Dict[Any, Any] = {}
            for h in HORIZONS:
                mg = (margins.get(name) or {}).get(key, {}).get(h)
                if mg is None:
                    m = (((rep.get("model") or {}).get("horizons") or {}).get(h) or {}).get(group, {}).get(metric)
                    b = ((((rep.get("baselines") or {}).get(name) or {}).get("horizons") or {}).get(h) or {}) \
                        .get(group, {}).get(metric)
                    mg = baseline_margin(metric, m, b)
                z, test = margin_z(mg)
                verdict = "beats" if per_h.get(h) else "does not beat"
                if z is not None:
                    kind = "noise, " if abs(z) < Z95 else ("significantly worse, " if z < 0 else "")
                    verdict += f" ({kind}{test})"
                d = 2 if metric in DOLLAR_METRICS else digits
                row[(h, "model")] = _num(mg.get("model"), d)
                row[(h, "baseline")] = _num(mg.get("baseline"), d)
                row[(h, MARGIN)] = _num(mg.get("margin"), d)
                row[(h, "z")] = _num(z if z is not None else float("nan"), 2)
                row[(h, "verdict")] = verdict
            rows.append(row)
            index.append((name, key))
    cols = pd.MultiIndex.from_tuples([(h, f) for h in HORIZONS for f in ("model", "baseline", MARGIN, "z",
                                                                             "verdict")], names=["horizon", ""])
    df = pd.DataFrame(rows, index=pd.MultiIndex.from_tuples(index or [], names=["baseline", "metric"]), columns=cols)
    for h in HORIZONS:
        for f in ("model", "baseline", MARGIN, "z"):
            df[(h, f)] = df[(h, f)].astype(float)
    if current:
        df.attrs["caption"] = ("noise = |z| < 1.96; significantly worse = the model loses with z <= -1.96. "
                               "DM: Diebold-Mariano z of the per-sample loss difference, "
                               f"Bartlett long-run variance with lag {DM_LAG_PER_STEP} x bars ahead. boot: margin / "
                               f"its paired moving-block bootstrap standard error ({BLOCK}-bar blocks, {BOOT_N} "
                               "resamples).")
    elif margins:
        df.attrs["caption"] = ("This report predates the current noise tests: its DM z uses sd / sqrt(n // bars "
                               "ahead), which is conservative, and the ranked metrics have no test. Re-score it "
                               "with evaluate() before reading a 'noise' verdict.")
    else:
        df.attrs["caption"] = "This report has no noise test (written before it existed); re-score it with evaluate()."
    return df


# ------------------------------------------------------------------ extended trends
def trailing_move_table(frames: Mapping[str, Any], config, *,
                        raw_deltas: Optional[Mapping[str, Dict[str, np.ndarray]]] = None,
                        trends: Optional[Mapping[str, np.ndarray]] = None, deadband_bps: Optional[float] = None,
                        digits: Optional[int] = 4) -> pd.DataFrame:
    """The trailing move (the extended-trend feature, also the loss's momentum prior) vs the realised move.

    ``frames``: {block name: PredictionFrame}, e.g. {"cal": ..., "test": ...}; each needs ``X_raw`` (the
    close windows) unless ``trends`` gives the [N, 3] trailing moves ("extended_trends" of split_arrays).
    The trailing move of horizon k is close[t] - close[t - p_k], p = config.EXTENDED_TREND_PERIODS.
    ``raw_deltas``: {block: raw heads}; correlations are the same for the served deltas while beta > 0.
    Rows (block, horizon). The sign agreement counts moves beyond the deadband with a non-zero
    trailing move; its 95% interval and the correlation noise band use n_eff = n / bars ahead. A
    correlation that changes sign between blocks is not a stable edge.
    """
    periods = [int(p) for p in getattr(config, "EXTENDED_TREND_PERIODS", (10, 15, 20))]
    lam = getattr(config, "LAMBDA_EXTENDED_TREND", None)
    head_col = "corr(head, trailing move)" + (f": momentum prior, lambda {float(lam):g}" if lam is not None else "")
    deadband = _deadband(config, deadband_bps)
    rows, index = [], []
    for block, frame in frames.items():
        if trends is not None and block in trends:
            ext_all = np.asarray(trends[block], float)
        else:
            X = getattr(frame, "X_raw", None)
            if X is None:
                raise ValueError(f"block {block!r}: the frame has no X_raw windows; pass trends={{{block!r}: ...}}")
            X = np.asarray(X, float)
            if max(periods) >= X.shape[1]:
                raise ValueError(f"block {block!r}: a trailing window of {max(periods)} bars needs more than the "
                                 f"{X.shape[1]}-bar input windows; pass trends=")
            ext_all = np.stack([X[:, -1] - X[:, -1 - p] for p in periods], 1)
        labels = direction_labels_np(frame.y, frame.last_close, deadband)
        heads = (raw_deltas or {}).get(block) or frame.delta
        n = len(frame)
        for i, h in enumerate(HORIZONS):
            steps = S.horizon_steps(frame, h)
            y, ext = frame.y[:, i], ext_all[:, i]
            head = np.asarray(heads[h], float).reshape(-1)[:n]
            mask = labels[h][1] & (ext != 0)
            k, m = int(((ext > 0) == (y > 0))[mask].sum()), int(mask.sum())
            share, lo, hi = S.wilson(k, max(m, 1), steps=steps)
            rows.append({"bars ahead": steps, "trailing window (bars)": periods[i], "n": n,
                         "n_eff": n // max(1, steps),
                         "corr(trailing move, realised)": _corr(ext, y),
                         "corr noise band +/- (95%, n_eff)": S.corr_null_r(n, steps=steps),
                         "trailing sign = realised sign (beyond the deadband)": float(share) if m else float("nan"),
                         "95% CI low": float(lo) if m else float("nan"),
                         "95% CI high": float(hi) if m else float("nan"),
                         "corr(head, realised)": _corr(head, y), head_col: _corr(head, ext)})
            index.append((block, h))
    df = pd.DataFrame(rows, index=pd.MultiIndex.from_tuples(index, names=["block", "horizon"]))
    for c in ("bars ahead", "trailing window (bars)", "n", "n_eff"):
        df[c] = df[c].astype(int)
    if digits is not None:
        num = [c for c in df.columns if c not in ("bars ahead", "trailing window (bars)", "n", "n_eff")]
        df[num] = df[num].round(digits)
    return df


# ------------------------------------------------------------------ run settings
def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _show(v) -> str:
    if v is None:
        return ""
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.4g}"
    if isinstance(v, (list, tuple)):
        return ", ".join(_show(x) for x in v)
    if isinstance(v, dict):
        return json.dumps(v)
    return str(v)


def _reductions(metrics: list, key: str) -> int:
    """How many times the logged learning rate ``key`` dropped between consecutive epochs."""
    vals = [r.get(key) for r in metrics if r.get(key) is not None]
    return sum(1 for a, b in zip(vals, vals[1:]) if b < a * (1 - 1e-6))


def _per_h(d: Optional[Mapping[str, Any]], fmt="{:.3f}") -> str:
    if not d:
        return ""
    return " / ".join(fmt.format(float(d[h])) if d.get(h) is not None else "n/a" for h in HORIZONS)


def run_settings_table(run_dir: Union[str, Path, None] = None, config=None, *,
                       metrics: Optional[list] = None) -> pd.DataFrame:
    """A saved run's settings: the config value next to the value the run actually used.

    ``run_dir``: the run directory (config.yaml, metrics.jsonl, meta.json, status.json, artifacts/);
    ``config``: the run's Config (default: run_dir/config.yaml); ``metrics``: epoch rows (default:
    run_dir/metrics.jsonl). The effective loss weights are the logged ``lambda_*`` of the last epoch:
    the pre-training calibration pass rescales most of them, so config.yaml alone misleads. Retired and
    unused LAMBDA_* fields are left out. Index (group, setting); columns config, effective, note. A blank
    "effective" means the run used the config value as it is.
    """
    from dataclasses import fields

    run = Path(run_dir) if run_dir is not None else None
    if config is None:
        if run is None:
            raise ValueError("pass run_dir or config")
        from neural_trade.core.config import Config

        config = Config.from_yaml(run / "config.yaml")
    if metrics is None and run is not None and (run / "metrics.jsonl").exists():
        from neural_trade.telemetry.epoch_logger import read_metrics

        metrics = read_metrics(run / "metrics.jsonl")
    metrics = list(metrics or [])
    last = metrics[-1] if metrics else {}
    art = _read_json(run / "artifacts" / "meta.json") if run is not None else {}
    meta = _read_json(run / "meta.json") if run is not None else {}
    status = _read_json(run / "status.json") if run is not None else {}
    pipe = _read_json(run / "artifacts" / "calibration" / "pipeline_meta.json") if run is not None else {}
    temps = (_read_json(run / "artifacts" / "calibration" / "temperature_params.json").get("temperatures")
             if run is not None else None)
    lam_final = art.get("lambda_values_final") or {}
    rows = []

    def add(group, setting, cfg_value="", effective="", note=""):
        rows.append({"group": group, "setting": setting, "config": _show(cfg_value) if cfg_value != "" else "",
                     "effective": _show(effective) if effective != "" else "", "note": note})

    cfgv = lambda name: getattr(config, name, None)  # noqa: E731
    # run
    add("run", "run id", "", meta.get("run_id") or status.get("run_id") or (run.name if run is not None else ""))
    add("run", "SEED", cfgv("SEED"), meta.get("seed", ""))
    if meta.get("tags"):
        add("run", "tags", "", meta.get("tags"))
    if (art.get("env") or {}).get("git"):
        add("run", "git commit", "", art["env"]["git"])
    epochs_run = art.get("epochs_run", status.get("epochs_completed", len(metrics) or ""))
    stopped = isinstance(epochs_run, int) and cfgv("EPOCHS") is not None and epochs_run < int(cfgv("EPOCHS"))
    add("run", "EPOCHS", cfgv("EPOCHS"), epochs_run, "stopped before the planned epochs" if stopped else "")
    if status.get("elapsed_seconds") is not None:
        add("run", "wall time", "", f"{float(status['elapsed_seconds']) / 60:.1f} min")
    # data / split
    for name in ("LOOKBACK", "HORIZON_STEPS", "EXTENDED_TREND_PERIODS", "MAX_SEQUENCE_COUNT", "N_FOLDS",
                 "FOLD_INDEX", "VAL_FRACTION", "CAL_FRACTION", "RESAMPLE_MINUTES"):
        add("data / split", name, cfgv(name))
    fold = art.get("fold") or {}
    if fold:
        add("data / split", "block sizes: train / val / cal / test (purge gap)", "",
            " / ".join(str(fold.get(k, "?")) for k in ("train", "val", "cal", "test"))
            + (f" (gap {fold['gap']})" if "gap" in fold else ""))
    # training
    lr_final, lri_final = last.get("lr"), last.get("lr_indicator")
    n_red, n_red_ind = _reductions(metrics, "lr"), _reductions(metrics, "lr_indicator")
    add("training", "BATCH_SIZE", cfgv("BATCH_SIZE"))
    add("training", "LR", cfgv("LR"), lr_final if lr_final is not None else "",
        f"final main learning rate after {n_red} reduction(s)" if lr_final is not None else "")
    mult = cfgv("INDICATOR_LR_MULT")
    add("training", "INDICATOR_LR_MULT", mult, "", "indicator learning rate = LR x this, at the start")
    ind_cfg = float(cfgv("LR")) * float(mult) if cfgv("LR") is not None and mult is not None else ""
    if lri_final is None:
        ind_note = "not logged"
    elif n_red and not n_red_ind:
        ind_note = (f"final indicator learning rate: NOT reduced with the main LR ({_show(cfgv('LR'))} -> "
                    f"{_show(lr_final)}), so the indicator / main ratio grew from {_show(mult)} to "
                    f"{_show(float(lri_final) / float(lr_final)) if lr_final else 'n/a'}")
    else:
        ind_note = f"final indicator learning rate after {n_red_ind} reduction(s)"
    add("training", "indicator learning rate (LR x INDICATOR_LR_MULT)", ind_cfg,
        lri_final if lri_final is not None else "", ind_note)
    for name in ("PATIENCE", "EARLY", "GRAD_CLIP_NORM", "INDICATOR_GRAD_MULT", "MODEL_NAME", "OPTIMIZER_NAME",
                 "LOSS_NAME", "TRAIN_METRICS_EVERY"):
        add("training", name, cfgv(name))
    # direction
    for name in ("DIRECTION_LOSS", "FOCAL_ALPHA", "FOCAL_GAMMA", "DIR_DEADBAND_BPS", "DIRECTION_SKIP"):
        add("direction", name, cfgv(name))
    # loss weights: config vs the weights the run used
    ablated = set(getattr(config, "ABLATE_LAMBDAS", None) or ())
    align_on = float(cfgv("LAMBDA_DIR_ALIGN_OUTER") or 0.0) > 0 and "LAMBDA_DIR_ALIGN_OUTER" not in ablated
    for f in fields(config):
        if not f.name.startswith("LAMBDA_"):
            continue
        doc = str(f.metadata.get("doc", ""))
        if "retired" in doc or doc.startswith("unused"):
            continue
        key = f.name.lower()
        eff = last.get(key, lam_final.get(key))
        c = float(getattr(config, f.name))
        if f.name in ablated:
            add("loss weights", f.name, c, 0.0 if eff is None else float(eff), "ablated (ABLATE_LAMBDAS)")
            continue
        if f.name in THRESHOLD_LAMBDAS:
            add("loss weights", f.name, c, "off" if c <= 0 else "", THRESHOLD_LAMBDAS[f.name])
            continue
        if f.name == "LAMBDA_DIR_ALIGN_OUTER" and not align_on:
            add("loss weights", f.name, c, "off", "0 switches the direction-head / Gaussian-readout alignment term off "
                                                  "(LAMBDA_DIR_ALIGN is then unused)")
            continue
        if f.name == "LAMBDA_DIR_ALIGN" and not align_on:
            add("loss weights", f.name, c, "off", "inactive: the alignment term runs only when "
                                                  "LAMBDA_DIR_ALIGN_OUTER > 0")
            continue
        fixed = FIXED_LOSS_SCALE.get(f.name)
        tail = f"; the loss multiplies this term by a fixed {fixed:g}" if fixed is not None else ""
        if eff is None:
            if f.name in OUTER_LAMBDAS and getattr(config, "CALIB_OUTER", False):
                add("loss weights", f.name, c, "?", "rescaled by the calibration pass (CALIB_OUTER) but not logged"
                    + tail)
            else:
                add("loss weights", f.name, c, "", "not logged; the calibration pass leaves it as configured" + tail)
            continue
        eff = float(eff)
        changed = abs(eff - c) > 1e-4 * max(abs(c), 1e-12)
        note = (f"auto-calibrated: x{eff / c:.3g}" if c else "auto-calibrated") if changed else ""
        add("loss weights", f.name, c, eff, (note + tail).lstrip("; "))
    for name in ("CALIB_DAMPING", "CALIB_LAMBDA_MIN", "CALIB_LAMBDA_MAX", "CALIB_OUTER", "ABLATE_LAMBDAS"):
        add("loss-weight calibration", name, cfgv(name))
    # serving calibration
    add("serving calibration", "CONFORMAL_SCALE", cfgv("CONFORMAL_SCALE"), pipe.get("conformal_scale", ""))
    add("serving calibration", "DELTA_SHRINKAGE", cfgv("DELTA_SHRINKAGE"), pipe.get("shrink_delta", ""))
    if pipe.get("delta_scale"):
        add("serving calibration", "delta shrink beta h0 / h1 / h2", "", _per_h(pipe["delta_scale"]),
            "served delta = beta x raw price head")
    if temps:
        add("serving calibration", "temperature h0 / h1 / h2", "", _per_h(temps),
            "P(up) = sigmoid(logit / T); T > 1 pulls P(up) towards 0.5")
    if art.get("var_scale") is not None:
        add("serving calibration", "var_scale", "", art["var_scale"], "the strategies' confidence scale")
    return pd.DataFrame(rows, columns=["group", "setting", "config", "effective", "note"]).set_index(["group", "setting"])


# ------------------------------------------------------------------ captions for the figures
# One-line summaries for figure titles and annotations, computed by the same code as the tables above, so a
# number printed on a figure equals the one in the table and in the evaluation report.
SEP = " · "


def _direction(frame, config, h, calibrated, deadband_bps):
    deadband = _deadband(config, deadband_bps)
    lab, mask = direction_labels_np(frame.y, frame.last_close, deadband)[h]
    return direction_block(lab, mask, np.asarray(frame.prob(h, calibrated), float)), deadband


def scored_counts_text(frame, config=None, *, block: Optional[str] = None,
                       deadband_bps: Optional[float] = None) -> str:
    """'test block 7,236 samples; scored beyond the 5 bps deadband: h0 5,345 / h1 5,697 / h2 5,874
    (n_eff 534 / 379 / 293)': the per-horizon sample sizes of every direction number."""
    deadband = _deadband(config, deadband_bps)
    labels = direction_labels_np(frame.y, frame.last_close, deadband)
    n_sc = [int(labels[h][1].sum()) for h in HORIZONS]
    n_eff = [n // max(1, S.horizon_steps(frame, h)) for n, h in zip(n_sc, HORIZONS)]
    name = block if block is not None else getattr(frame, "split", "")
    return (f"{name} block {len(frame):,} samples; scored beyond the {deadband:g} bps deadband: "
            + " / ".join(f"{h} {n:,}" for h, n in zip(HORIZONS, n_sc))
            + " (n_eff " + " / ".join(f"{n:,}" for n in n_eff) + ")").strip()


def confusion_title(frame, config=None, h: str = "h1", *, calibrated: bool = True,
                    deadband_bps: Optional[float] = None, sep: str = "<br>") -> str:
    """Two-line confusion-matrix title: 'h1 confusion matrix, n 5,697: calls up 35%, really up 51%' and
    'acc 0.488 · prec 0.504 · rec 0.345 · spec 0.640 · F1 0.410 · MCC -0.017' (P(up) > 0.5, up = positive)."""
    d, _ = _direction(frame, config, h, calibrated, deadband_bps)
    pct = lambda v: f"{100 * v:.0f}%" if math.isfinite(v) else "n/a"  # noqa: E731
    f = lambda v, k=3: f"{v:.{k}f}" if v is not None and math.isfinite(v) else "n/a"  # noqa: E731
    return (f"{h} confusion matrix, n {d['n_masked']:,}: calls up {pct(d['pred_up_rate'])}, really up "
            f"{pct(d['true_up_rate'])}{sep}"
            + SEP.join(f"{k} {f(d[m])}" for k, m in (("acc", "acc"), ("prec", "precision"), ("rec", "recall"),
                                                      ("spec", "specificity"), ("F1", "f1"), ("MCC", "mcc"))))


def direction_stats_line(frame, config=None, h: str = "h1", *, raw_delta: Optional[Dict[str, np.ndarray]] = None,
                         calibrated: bool = True, deadband_bps: Optional[float] = None) -> str:
    """'up-rate 0.515 · calls up 0.353 · acc 0.488 · bal acc 0.493 · Brier 0.2517 (constant up-rate 0.2498) ·
    sign(delta) = call 0.612 (0.503 if independent)'. The sign agreement uses the raw heads when given (the
    same as the served deltas while beta > 0) and the calibrated P(up), like the report's coherence block."""
    d, _ = _direction(frame, config, h, calibrated, deadband_bps)
    u = d["true_up_rate"]
    raw = _raw_or(frame, raw_delta)
    coh = coherence_block(frame, raw)
    f = lambda v, k=3: f"{v:.{k}f}" if v is not None and math.isfinite(v) else "n/a"  # noqa: E731
    if raw is None and not np.any(np.asarray(frame.delta[h], float)):
        sign = f"sign(delta) = call {ZERO_BETA_NA}"          # a served delta of 0 (beta = 0) has no sign
    else:
        sign = (f"sign(delta) = call {f(coh[f'delta_dir_align_{h}'])} "
                f"({f(coh[f'delta_dir_align_indep_{h}'])} if independent)")
    return SEP.join([f"up-rate {f(u)}", f"calls up {f(d['pred_up_rate'])}", f"acc {f(d['acc'])}",
                     f"bal acc {f(d['bal_acc'])}",
                     f"Brier {f(d['brier'], 4)} (constant up-rate {f(u * (1 - u), 4)})", sign])


def delta_summary_text(frame, h: str = "h1", *, raw_delta: Optional[Dict[str, np.ndarray]] = None,
                       delta_scale: Optional[Dict[str, float]] = None, sep: str = "<br>") -> str:
    """'RMSE $ raw 250.12 / served 240.55 / zero 240.61' and 'mean $ predicted raw -3.10 / served -0.07 vs
    realised 1.20 · beta 0.023' for one horizon (raw only with ``raw_delta``), from report.delta_block."""
    i = HORIZONS.index(h)
    y = frame.y[:, i]
    bs = delta_block(y, frame.delta[h])
    raw = _raw_or(frame, raw_delta)
    br = delta_block(y, raw[h]) if raw is not None else None
    sv = "served" if raw is not None or delta_scale else "model"
    rmse = ([f"raw {br['rmse']:.2f}"] if br else []) + [f"{sv} {bs['rmse']:.2f}", f"zero {bs['rmse_zero']:.2f}"]
    mean = ([f"raw {br['mean_pred']:.2f}"] if br else []) + [f"{sv} {bs['mean_pred']:.2f}"]
    beta = None
    if delta_scale and delta_scale.get(h) is not None:
        beta = float(delta_scale[h])
    elif raw is not None and float(np.dot(raw[h], raw[h])) > 0:
        beta = float(np.dot(frame.delta[h], raw[h]) / np.dot(raw[h], raw[h]))
    return ("RMSE $ " + " / ".join(rmse) + sep + "mean $ predicted " + " / ".join(mean)
            + f" vs realised {bs['mean_true']:.2f}" + (f"{SEP}beta {beta:.3f}" if beta is not None else ""))


# ------------------------------------------------------------------ display
def _is_usd(label) -> bool:
    parts = label if isinstance(label, tuple) else (label,)
    return any("($" in str(p) or str(p).rsplit("/", 1)[-1] in DOLLAR_METRICS for p in parts)


def _cell_formatter(decimals: int):
    def fmt(v):
        if v is None:
            return "n/a"
        if isinstance(v, (bool, np.bool_)):
            return str(bool(v))
        if isinstance(v, (int, np.integer)):
            return f"{int(v):d}"
        if isinstance(v, (float, np.floating)):
            v = float(v)
            if not math.isfinite(v):
                return "n/a"
            return f"{int(round(v)):d}" if decimals == 0 else f"{v:.{decimals}f}"
        return str(v)
    return fmt


def styled(df: pd.DataFrame, *, digits: int = 4, usd_digits: int = 2):
    """The table for display (a pandas Styler): the same decimals down a whole row or column, dollars with
    ``usd_digits`` (rows or columns labelled "($" or a dollar metric such as delta/rmse), counts without
    decimals, "z" with 2, missing values as "n/a", and ``df.attrs["caption"]`` (what the table cannot say
    in its labels) as the caption. The DataFrame itself keeps the numbers."""
    st = df.style
    for label in df.index:
        d = usd_digits if _is_usd(label) else digits
        st = st.format(_cell_formatter(d), subset=pd.IndexSlice[[label], :])
    for col in df.columns:
        parts = col if isinstance(col, tuple) else (col,)
        if "z" in parts:
            st = st.format(_cell_formatter(2), subset=pd.IndexSlice[:, [col]])
        elif any(str(p) in ("n", "n_eff", "bars ahead", "trailing window (bars)") for p in parts):
            st = st.format(_cell_formatter(0), subset=pd.IndexSlice[:, [col]])
    if df.attrs.get("caption"):
        st = st.set_caption(str(df.attrs["caption"])).set_table_styles(
            [{"selector": "caption", "props": [("caption-side", "bottom"), ("text-align", "left"),
                                               ("font-size", "0.85em"), ("max-width", "70em")]}], overwrite=False)
    return st
