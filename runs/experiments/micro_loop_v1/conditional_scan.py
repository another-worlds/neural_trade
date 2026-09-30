"""Micro loop H6: is directional predictability concentrated in conditions (hour of day, volatility regime,
after a large move, volume regime) rather than spread over all bars?

    python runs/experiments/micro_loop_v1/conditional_scan.py     # writes conditional_scan.json next to this file

Data: Bitcoin_BTCUSDT.csv; FIT 2023-01-01 .. 2024-12-31, SCORE 2025-01-01 .. 2025-07-20 (before the micro dev
block and the protected last 64 days). Anchors every 5 bars. Target: sign of the H-bar forward close change,
H in {15, 60, 240}. Model: logistic regression on log returns over the last 1, 5, 15, 60, 240 bars, log EWMA
volatility (half-life 60), log volume ratio (last 60 vs last 1440 bars), and hour-of-day sin/cos. For each
condition bucket on the SCORE block: n, n_eff = n * 5 / H, AUC with its approximate s.e., hit rate of the
model's calls, and the hit rate on the bucket's top-10% most confident calls. Pre-registered bar for a
"concentration" finding: a bucket with AUC >= 0.56 and z >= 3 at n_eff >= 200 (it must also hold in the FIT block
to count). Descriptive; no choice is made from the SCORE block beyond this bar.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

CSV = Path("D:/neural_trade/Bitcoin_BTCUSDT.csv")
OUT = Path(__file__).with_name("conditional_scan.json")
STEP = 5
HS = (15, 60, 240)
LAGS = (1, 5, 15, 60, 240)


def auc_stats(y, p, h):
    n = len(y)
    if n < 50 or len(np.unique(y)) < 2:
        return None
    auc = float(roc_auc_score(y, p))
    n_eff = n * STEP / h
    se = float(np.sqrt(1.0 / (3.0 * max(n_eff, 1.0))))
    conf = np.abs(p - 0.5)
    top = conf >= np.quantile(conf, 0.9)
    return {"n": n, "n_eff": round(n_eff, 1), "auc": round(auc, 4), "z": round((auc - 0.5) / se, 2),
            "hit": round(float(np.mean((p > 0.5) == (y == 1))), 4),
            "hit_top10": round(float(np.mean((p[top] > 0.5) == (y[top] == 1))), 4)}


def main() -> None:
    df = pd.read_csv(CSV, usecols=["timestamp", "close", "volume"], parse_dates=["timestamp"])
    df = df[(df.timestamp >= "2022-12-30") & (df.timestamp < "2025-07-21")].reset_index(drop=True)
    c = df.close.to_numpy(float)
    v = df.volume.to_numpy(float)
    lr = np.diff(np.log(c), prepend=np.log(c[0]))
    sig = np.sqrt(pd.Series(lr ** 2).ewm(halflife=60, adjust=False).mean().to_numpy())
    vs = pd.Series(v)
    vol_ratio = np.log((vs.rolling(60).mean() + 1e-9) / (vs.rolling(1440).mean() + 1e-9)).to_numpy()
    ts = df.timestamp
    hour = ts.dt.hour.to_numpy()
    idx = np.arange(1500, len(c) - max(HS) - 1, STEP)
    ts_i = ts.to_numpy()[idx]
    fit = (ts_i >= np.datetime64("2023-01-01")) & (ts_i < np.datetime64("2025-01-01"))
    score = ts_i >= np.datetime64("2025-01-01")
    X = np.column_stack([np.log(c[idx] / c[idx - L]) for L in LAGS]
                        + [np.log(sig[idx] + 1e-12), np.nan_to_num(vol_ratio[idx]),
                           np.sin(2 * np.pi * hour[idx] / 24), np.cos(2 * np.pi * hour[idx] / 24)])
    ret60 = np.log(c[idx] / c[idx - 60])
    shock = np.abs(ret60) / (sig[idx] * np.sqrt(60) + 1e-12)
    # regime buckets use FIT-block quantiles only
    sig_q = np.quantile(sig[idx][fit], [1 / 3, 2 / 3])
    vol_q = np.quantile(np.nan_to_num(vol_ratio[idx])[fit], [1 / 3, 2 / 3])
    buckets = {
        "all": np.ones(len(idx), bool),
        **{f"hour_{h0:02d}-{h0 + 3:02d}": (hour[idx] >= h0) & (hour[idx] < h0 + 4) for h0 in range(0, 24, 4)},
        "vol_low": sig[idx] <= sig_q[0], "vol_mid": (sig[idx] > sig_q[0]) & (sig[idx] <= sig_q[1]),
        "vol_high": sig[idx] > sig_q[1],
        "volume_low": np.nan_to_num(vol_ratio[idx]) <= vol_q[0], "volume_high": np.nan_to_num(vol_ratio[idx]) > vol_q[1],
        "after_shock_2sd": shock >= 2.0, "after_shock_3sd": shock >= 3.0,
    }
    res = {"fit": "2023-01-01..2024-12-31", "score": "2025-01-01..2025-07-20", "step": STEP,
           "bar": "AUC >= 0.56 and z >= 3 at n_eff >= 200, holding in FIT too", "rows": []}
    for H in HS:
        y = (c[idx + H] > c[idx]).astype(int)
        model = LogisticRegression(max_iter=2000).fit(X[fit], y[fit])
        p_all = model.predict_proba(X)[:, 1]
        for name, m in buckets.items():
            s = auc_stats(y[score & m], p_all[score & m], H)
            f = auc_stats(y[fit & m], p_all[fit & m], H)
            if s is None:
                continue
            row = {"H": H, "bucket": name, "score": s, "fit_in_sample": f}
            row["meets_bar"] = bool(s["auc"] >= 0.56 and s["z"] >= 3 and s["n_eff"] >= 200
                                    and f is not None and f["auc"] >= 0.56)
            res["rows"].append(row)
            print(json.dumps({"H": H, "bucket": name, **s, "fit_auc": f and f["auc"], "bar": row["meets_bar"]}))
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
