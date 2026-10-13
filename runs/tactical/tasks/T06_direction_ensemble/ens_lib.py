"""T06: the ensemble pieces. Pure numpy / sklearn; nothing here reads a data file.

Every ensemble is a function of the MEMBERS' OUTPUTS only (P(up) arrays [N, 3], one column per horizon):
  E1  the equal-weight mean of P(up)
  E2  sigmoid of the equal-weight mean of the logits
  E3  a logistic regression per horizon on the members' logits, fitted ONLY on the tail rows (the last 15% of the training block,
      `tail_rows`), on the members' predictions for those rows; applied to the validation rows afterwards
Thresholds for the top-10% read come from the tail rows (a training-block quantity) and are applied to the validation rows.
"""
import math

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

INNER = 0.15
EPS = 1e-6
STACK_C = 1.0


def logit(p):
    p = np.clip(np.asarray(p, np.float64), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, np.float64)))


def tail_rows(n, inner=INNER):
    """Positions of the training block (n windows) in its last `inner` share: the same rows ladder.load_slice calls `iva`."""
    cut = int(round((1 - inner) * n))
    return np.arange(cut, n)


def e1(Ps):
    """Equal-weight mean of the members' P(up). Ps: [M, N, 3]."""
    return np.mean(np.asarray(Ps, np.float64), 0)


def e2(Ps):
    """Equal-weight mean of the members' logits, mapped back to P(up)."""
    return sigmoid(np.mean(logit(Ps), 0))


def fit_stacker(tail_Ps, tail_rtr, deadband, C=STACK_C):
    """E3: one logistic regression per horizon on the members' tail logits [M, n, 3] -> list of 3 fitted models.
    Labels and the deadband mask come from the tail rows' own training returns tail_rtr [n, 3]; no other row is read."""
    Z = logit(tail_Ps); out = []
    for h in range(3):
        m = np.abs(tail_rtr[:, h]) > deadband
        out.append(LogisticRegression(C=C, max_iter=1000).fit(Z[:, m, h].T, (tail_rtr[m, h] > 0).astype(int)))
    return out


def apply_stacker(models, Ps):
    """P(up) [N, 3] from the members' predictions Ps [M, N, 3] and the fitted stackers."""
    Z = logit(Ps)
    return np.stack([models[h].predict_proba(Z[:, :, h].T)[:, 1] for h in range(3)], 1)


def conf_threshold(P_tail, q=0.9):
    """The |mean-P - 0.5| quantile of the tail rows (training block only): rows above it form the top (1-q) share."""
    return float(np.quantile(np.abs(np.asarray(P_tail, np.float64).mean(1) - 0.5), q))


def top_read(P, rva, deadband, thr):
    """Hit rate and gross bps of the h1 return on rows whose confidence reaches thr (the threshold is given, never fitted here)."""
    s = np.asarray(P, np.float64).mean(1); r = np.asarray(rva)[:, 1]
    k = (np.abs(s - 0.5) >= thr) & (np.abs(r) > deadband)
    if not k.any():
        return dict(hit=float("nan"), bps=float("nan"), n=0)
    mv = np.sign(s[k] - 0.5) * r[k] * 1e4
    return dict(hit=float(np.mean(mv > 0)), bps=float(np.mean(mv)), n=int(k.sum()))


def score(P, rva, rtr, deadband):
    """auc_h0..2, auc3 (mean of the three), ll_h1, ll_const_h1 (the training block's base rate), as the lab's evaluate_slice."""
    out = {}
    for i in range(3):
        mva = np.abs(rva[:, i]) > deadband; y = (rva[mva, i] > 0).astype(int)
        out[f"auc_h{i}"] = float(roc_auc_score(y, P[mva, i]))
        if i == 1:
            q = np.clip(P[mva, 1], 1e-7, 1 - 1e-7); mtr = np.abs(rtr[:, 1]) > deadband; base = float((rtr[mtr, 1] > 0).mean())
            out["ll_h1"] = float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))
            out["ll_const_h1"] = float(-np.mean(y * np.log(base) + (1 - y) * np.log(1 - base)))
    out["auc3"] = float(np.mean([out[f"auc_h{i}"] for i in range(3)]))
    return out


def tci(v, level=0.95):
    """mean and t-interval of a vector over slices (the unit of inference)."""
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) < 2:
        return [float(v.mean()) if len(v) else float("nan")] * 3
    m = v.mean(); h = stats.t.ppf(0.5 + level / 2, len(v) - 1) * v.std(ddof=1) / math.sqrt(len(v))
    return [float(m), float(m - h), float(m + h)]


def mean_pair_corr(Z):
    """Pairwise Pearson correlation of the members' logits Z [M, N] (all horizons stacked by the caller)."""
    return np.corrcoef(np.asarray(Z, np.float64))
