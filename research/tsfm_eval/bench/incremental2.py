"""Feature-role test: do TSFM outputs add direction information beyond simple causal features?

Expanding walk-forward inside the scored origins: 5 equal blocks; fold k trains on blocks < k and tests
on block k, with a 15-minute purge before each test block. Logistic regression, standardised features.
Base features: past 1/5/15/60-min log returns, 60-min realised vol, last 1-min range.
TSFM features: median move (bps), P(up) - 0.5, 80% interval width (bps).
Reports pooled out-of-fold AUC for base, tsfm-only, base+tsfm, and the paired delta (base+tsfm - base).
Usage: python incremental2.py <tag> [<tag> ...]   (price-framing files only)
"""
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from common import HORIZONS, QLEVELS, RESULTS, load  # noqa: E402

LV = np.array(QLEVELS)
df, _ = load()
lc = np.log(df["close"].to_numpy(np.float64))
r1 = np.concatenate([[0.0], np.diff(lc)]) * 1e4
rng_bps = np.log(df["high"].to_numpy() / df["low"].to_numpy()) * 1e4
rs = np.random.default_rng(1)
PURGE_MIN = 15


def p_up(qb):
    return np.array([1 - np.interp(0.0, np.maximum.accumulate(row), LV, left=0, right=1) for row in qb])


def oof_scores(X, y, org, keep):
    n = len(org)
    edges = np.linspace(0, n, 6).astype(int)
    s = np.full(n, np.nan)
    for k in range(1, 5 + 0):
        te = np.zeros(n, bool)
        te[edges[k]:edges[k + 1]] = True
        tr = np.zeros(n, bool)
        tr[:edges[k]] = True
        tr &= org < org[edges[k]] - PURGE_MIN - max(HORIZONS)
        tr &= keep
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        m = LogisticRegression(C=1.0, max_iter=1000).fit((X[tr] - mu) / sd, y[tr])
        s[te] = m.predict_proba((X[te] - mu) / sd)[:, 1]
    return s


def paired_auc_ci(y, a, b, block, nboot=1000):
    n = len(y)
    out = []
    for _ in range(nboot):
        st = rs.integers(0, n - block + 1, int(np.ceil(n / block)))
        idx = (st[:, None] + np.arange(block)).ravel()[:n]
        if len(set(y[idx])) == 2:
            out.append(roc_auc_score(y[idx], b[idx]) - roc_auc_score(y[idx], a[idx]))
    return [round(float(v), 4) for v in np.percentile(out, [2.5, 97.5])]


res_all = []
for tag in sys.argv[1:]:
    z = np.load(f"{RESULTS}/{tag}.npz", allow_pickle=True)
    if str(z["framing"]) not in ("price", "price_cov"):
        print(f"skip {tag}: framing {z['framing']}")
        continue
    org = z["origins"]
    block = max(5, int(round(60 / np.median(np.diff(org)))))
    base = np.column_stack([(lc[org] - lc[org - k]) * 1e4 for k in (1, 5, 15, 60)]
                           + [np.array([r1[t - 59:t + 1].std() for t in org]), rng_bps[org]])
    for hi, h in enumerate(HORIZONS):
        y_ret = (lc[org + h] - lc[org]) * 1e4
        keep = y_ret != 0
        y = (y_ret > 0).astype(int)
        qb = (np.log(z["q"][:, hi, :].astype(np.float64)) - lc[org][:, None]) * 1e4
        tsfm = np.column_stack([qb[:, 10], p_up(qb) - 0.5, qb[:, 18] - qb[:, 2]])
        sc = {name: oof_scores(X, y, org, keep) for name, X in
              [("base", base), ("tsfm_only", tsfm), ("base+tsfm", np.column_stack([base, tsfm]))]}
        ev = keep & ~np.isnan(sc["base"])
        r = {"tag": tag, "h": h, "n_test": int(ev.sum())}
        for name, s in sc.items():
            r[f"auc_{name}"] = round(float(roc_auc_score(y[ev], s[ev])), 4)
        r["delta_auc_base+tsfm_minus_base"] = round(r["auc_base+tsfm"] - r["auc_base"], 4)
        r["delta_auc_ci95"] = paired_auc_ci(y[ev], sc["base"][ev], sc["base+tsfm"][ev], block)
        res_all.append(r)
        print(json.dumps(r), flush=True)
json.dump(res_all, open(f"{RESULTS}/incremental2.json", "w"), indent=1)
