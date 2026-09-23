"""evaluate(): the evaluation protocol (plan section C2).

Every direction metric uses the neutral mask (|return| <= deadband is excluded). Reported per
horizon:

    direction        (head; temperature-calibrated when a CalibrationPipeline was fit)
                     mcc auc brier ece_pos acc bal_acc pred_up_rate true_up_rate n_masked
    gauss_direction  the same for the Gaussian readout P(up | move leaves the deadband)
    delta            rmse mae ev corr skill_vs_zero      (never price-level EV: it is ~0.999
                                                          for "no change" and says nothing)
    variance         crps nll (dollars) pit_ks corr_var_err2_spearman coverage90 width90 crpss
    confidence_gap   accuracy(high confidence) - accuracy(low confidence) with a moving-block
                     bootstrap CI (block = 80 bars); WORKS only when the CI excludes 0 and
                     the gap is at least 1 pp
    n_eff            N // horizon_steps: non-overlapping outcomes, the honest sample size

Across horizons: coherence (magnitude ordering, unanimity, head/delta sign agreement).
Baselines (fit on train, see evaluation.baselines) are scored with the same code, and
``beats_baseline`` records, per baseline and relevant metric, whether the model is better.
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

HIGHER_IS_BETTER = {"mcc", "auc", "acc", "bal_acc", "ev", "corr", "skill_vs_zero", "corr_var_err2_spearman", "crpss"}
LOWER_IS_BETTER = {"brier", "ece_pos", "rmse", "mae", "crps", "nll", "pit_ks"}
BLOCK = 80


def _auc(labels, scores):
    from sklearn.metrics import roc_auc_score

    labels = np.asarray(labels)
    if len(labels) < 2 or labels.min() == labels.max():
        return float("nan")
    return float(roc_auc_score(labels, scores))


def direction_block(labels, mask, prob) -> Dict[str, float]:
    t, p = labels[mask], np.asarray(prob)[mask]
    pb = p > 0.5
    tp, tn = float(np.sum(pb & (t > 0.5))), float(np.sum(~pb & (t < 0.5)))
    fp, fn = float(np.sum(pb & (t < 0.5))), float(np.sum(~pb & (t > 0.5)))
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    return {
        "mcc": npm.mcc(t, p), "auc": _auc(t, p), "brier": npm.brier(t, p), "ece_pos": npm.ece_pos(t, p),
        "acc": npm.direction_accuracy(t, p), "bal_acc": float(np.nanmean([sens, spec])),
        "pred_up_rate": float(pb.mean()) if len(pb) else float("nan"),
        "true_up_rate": float(t.mean()) if len(t) else float("nan"), "n_masked": int(mask.sum()),
    }


def delta_block(y, delta) -> Dict[str, float]:
    mse = float(np.mean((y - delta) ** 2))
    mse0 = float(np.mean(y ** 2))
    return {"rmse": math.sqrt(mse), "mae": float(np.mean(np.abs(y - delta))),
            "ev": npm.explained_variance(y, delta), "corr": npm.corr(y, delta),
            "skill_vs_zero": 1.0 - mse / mse0 if mse0 > 0 else float("nan")}


def gaussian_crps(y, mu, sigma):
    """Closed-form CRPS of N(mu, sigma^2) at y (same units as y)."""
    s = np.maximum(sigma, 1e-12)
    z = (y - mu) / s
    return s * (z * (2 * ndtr(z) - 1) + 2 * np.exp(-0.5 * z ** 2) / math.sqrt(2 * math.pi) - 1 / math.sqrt(math.pi))


def variance_block(y, mu, sigma, interval=None) -> Dict[str, float]:
    s = np.maximum(sigma, 1e-12)
    err2 = (y - mu) ** 2
    out = {
        "crps": float(np.mean(gaussian_crps(y, mu, s))),
        "nll": float(np.mean(0.5 * np.log(2 * math.pi * s ** 2) + err2 / (2 * s ** 2))),
        "pit_ks": npm.pit_ks(y, mu, variance=s ** 2),
        "corr_var_err2_spearman": (float(spearmanr(s ** 2, err2).correlation) if np.std(s) > 0 else 0.0),
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


def coherence_block(frame: PredictionFrame) -> Dict[str, float]:
    d = np.stack([frame.delta[h] for h in HORIZONS], 1)
    p = np.stack([frame.prob(h) for h in HORIZONS], 1)
    head_up = p > 0.5
    delta_up = d > 0
    return {
        "mag_order_full": float(np.mean((np.abs(d[:, 0]) <= np.abs(d[:, 1])) & (np.abs(d[:, 1]) <= np.abs(d[:, 2])))),
        "unanimity": float(np.mean(head_up.all(1) | (~head_up).all(1))),
        "delta_dir_align_all": float(np.mean((head_up == delta_up).all(1))),
        "coherence_primary": float(np.mean(head_up[:, 1] == delta_up[:, 1])),
    }


def score_frame(frame: PredictionFrame, deadband_bps: float, conf_threshold=None) -> Dict[str, Any]:
    labels = direction_labels_np(frame.y, frame.last_close, deadband_bps)
    out: Dict[str, Any] = {"horizons": {}, "coherence": coherence_block(frame)}
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
        conf = np.abs(prob - 0.5)
        thr = (conf_threshold or {}).get(h, float(np.median(conf[mask]))) if mask.any() else 0.0
        correct = ((prob > 0.5) == (lab > 0.5)).astype(float)
        row["confidence_gap"] = confidence_gap(correct[mask], conf[mask], thr)
        row["confidence_gap"]["threshold_source"] = "cal" if conf_threshold else "self"
        out["horizons"][h] = row
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

    # ------------------------------------------------------------------ views
    def metric(self, h: str, group: str, name: str) -> float:
        return self.model["horizons"][h][group][name]

    def flat(self) -> Dict[str, float]:
        out = {}
        for h, row in self.model["horizons"].items():
            for group in ("direction", "gauss_direction", "delta", "variance"):
                for k, v in row[group].items():
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

    def to_markdown(self, path=None) -> str:
        L = [f"# Evaluation report - {self.split} split" + (f" - run `{self.run_id}`" if self.run_id else ""),
             "", f"n = {self.n} samples; direction metrics exclude moves within {self.deadband_bps:g} bps "
             "(neutral mask); n_eff counts non-overlapping outcomes.", ""]
        L += ["| metric | " + " | ".join(HORIZONS) + " |", "|---|" + "---|" * len(HORIZONS)]
        rows = [("n_eff", None, "n_eff"), ("direction MCC", "direction", "mcc"), ("direction AUC", "direction", "auc"),
                ("direction ECE", "direction", "ece_pos"), ("Gaussian MCC", "gauss_direction", "mcc"),
                ("Gaussian AUC", "gauss_direction", "auc"), ("delta EV", "delta", "ev"), ("delta corr", "delta", "corr"),
                ("skill vs zero", "delta", "skill_vs_zero"), ("CRPS ($)", "variance", "crps"),
                ("PIT KS", "variance", "pit_ks"), ("var/err2 Spearman", "variance", "corr_var_err2_spearman"),
                ("coverage 90%", "variance", "coverage90")]
        for label, group, key in rows:
            vals = []
            for h in HORIZONS:
                row = self.model["horizons"][h]
                v = row[key] if group is None else row[group].get(key)
                vals.append(_fmt(v))
            L.append(f"| {label} | " + " | ".join(vals) + " |")
        L += ["", "## Confidence gap (accuracy of the more confident half minus the less confident half)", "",
              "| horizon | gap | 95% CI | verdict |", "|---|---|---|---|"]
        for h in HORIZONS:
            g = self.model["horizons"][h]["confidence_gap"]
            L.append(f"| {h} | {_fmt(g['gap'])} | [{_fmt(g['ci'][0])}, {_fmt(g['ci'][1])}] | {g['verdict']} |")
        L += ["", "## Coherence across horizons", ""]
        L += [f"- {k}: {_fmt(v)}" for k, v in self.model["coherence"].items()]
        if self.beats_baseline:
            L += ["", "## Against baselines (fit on the training block)", "",
                  "| baseline | metric | " + " | ".join(HORIZONS) + " |", "|---|---|" + "---|" * len(HORIZONS)]
            for name, per_metric in self.beats_baseline.items():
                for metric, per_h in per_metric.items():
                    L.append(f"| {name} | {metric} | " + " | ".join(
                        ("beats" if per_h.get(h) else "does not beat") for h in HORIZONS) + " |")
        if self.backtest:
            L += ["", "## Backtest (costs included)", ""]
            L += [f"- {k}: {_fmt(v)}" for k, v in (self.backtest.get("summary") or {}).items()]
        text = "\n".join(L) + "\n"
        if path is not None:
            Path(path).write_text(text, encoding="utf-8")
        return text


def _fmt(v):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "n/a"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return f"{v:.4f}"


def _jsonable(o):
    if isinstance(o, (np.generic,)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _metric_groups(name):
    from neural_trade.evaluation.baselines import RELEVANT

    return RELEVANT.get(name, ())


def evaluate(frame: PredictionFrame, config, *, baselines=None, cal_frame: Optional[PredictionFrame] = None,
             run_id: Optional[str] = None, backtest: Optional[Dict[str, Any]] = None) -> EvalReport:
    """Score ``frame`` (and every fitted baseline) under the protocol above."""
    deadband = float(getattr(config, "DIR_DEADBAND_BPS", 0.0))
    conf_thr = None
    if cal_frame is not None:  # confidence threshold from the CALIBRATION split, not the test split
        labels = direction_labels_np(cal_frame.y, cal_frame.last_close, deadband)
        conf_thr = {h: float(np.median(np.abs(cal_frame.prob(h) - 0.5)[labels[h][1]])) for h in HORIZONS
                    if labels[h][1].any()}
    model = score_frame(frame, deadband, conf_thr)
    report = EvalReport(run_id, frame.split, deadband, len(frame), model, backtest=backtest,
                        meta={"horizon_steps": list(frame.horizon_steps), "pred_scale": frame.pred_scale})
    if baselines is not None:
        for name, bframe in baselines.predict(frame).items():
            scored = score_frame(bframe, deadband)
            report.baselines[name] = scored
            beats: Dict[str, Dict[str, bool]] = {}
            for group in _metric_groups(name):
                for metric in (model["horizons"]["h1"][group] or {}):
                    if metric not in HIGHER_IS_BETTER | LOWER_IS_BETTER:
                        continue
                    beats[f"{group}/{metric}"] = {}
                    for h in HORIZONS:
                        m, b = model["horizons"][h][group].get(metric), scored["horizons"][h][group].get(metric)
                        ok = (m is not None and b is not None and math.isfinite(m) and math.isfinite(b)
                              and ((m > b) if metric in HIGHER_IS_BETTER else (m < b)))
                        beats[f"{group}/{metric}"][h] = bool(ok)
            report.beats_baseline[name] = beats
        if "const_var" in report.baselines:
            for h in HORIZONS:
                c_m = model["horizons"][h]["variance"]["crps"]
                c_b = report.baselines["const_var"]["horizons"][h]["variance"]["crps"]
                model["horizons"][h]["variance"]["crpss"] = 1.0 - c_m / c_b if c_b > 0 else float("nan")
    return report
