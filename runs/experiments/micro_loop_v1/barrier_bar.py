"""E3 model-free bar (P&L plan section 3 "E3"): can a logistic regression on lagged returns predict which
volatility-scaled barrier is hit first better than it predicts the close-to-close sign?

    python runs/experiments/micro_loop_v1/barrier_bar.py      # writes barrier_bar.json next to this file

Data: Bitcoin_BTCUSDT.csv 2025-01-01 .. 2025-07-20 (before the micro dev block and far from fold -1);
fit on 2025-01-01 .. 2025-04-30, score on 2025-05-01 .. 2025-07-20. Anchors every 5 bars. Barriers
+-k * sigma * sqrt(H), sigma a causal EWMA (half-life 60 bars) of 1-bar log returns. Label 1 = upper barrier
first, 0 = lower first; untouched paths are masked out (the "no trade" class). Same-bar ties count as lower
first (sl_first). Features: log returns over the last 1, 5, 15, 60, 240 bars and log(sigma).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

CSV = Path("D:/neural_trade/Bitcoin_BTCUSDT.csv")
OUT = Path(__file__).with_name("barrier_bar.json")
STEP = 5
HORIZONS = (60, 120, 240)
KS = (1.0, 1.5)
LAGS = (1, 5, 15, 60, 240)


def main() -> None:
    df = pd.read_csv(CSV, usecols=["timestamp", "high", "low", "close"], parse_dates=["timestamp"])
    df = df[(df.timestamp >= "2024-12-31") & (df.timestamp < "2025-07-21")].reset_index(drop=True)
    c = df.close.to_numpy(float)
    hi = df.high.to_numpy(float)
    lo = df.low.to_numpy(float)
    lr = np.diff(np.log(c), prepend=np.log(c[0]))
    sig = np.sqrt(pd.Series(lr**2).ewm(halflife=60, adjust=False).mean().to_numpy())
    ts = df.timestamp.to_numpy()
    maxh = max(HORIZONS)
    idx = np.arange(max(LAGS) + 300, len(c) - maxh - 1, STEP)
    feats = np.column_stack([np.log(c[idx] / c[idx - L]) for L in LAGS] + [np.log(sig[idx] + 1e-12)])
    is_train = ts[idx] < np.datetime64("2025-05-01")
    is_test = ts[idx] >= np.datetime64("2025-05-01")
    # forward paths relative to the decision close (entry at the close, as a label)
    fwd = np.arange(1, maxh + 1)
    up_path = np.log(hi[idx[:, None] + fwd[None, :]] / c[idx][:, None])
    dn_path = np.log(lo[idx[:, None] + fwd[None, :]] / c[idx][:, None])
    close_path = np.log(c[idx[:, None] + fwd[None, :]] / c[idx][:, None])
    res = {"data": "2025-01-01..2025-07-20, fit < 2025-05-01, score >= 2025-05-01", "step": STEP, "rows": []}
    for H in HORIZONS:
        # the sign label (control)
        y_sign = (close_path[:, H - 1] > 0).astype(int)
        for label, k in [("sign", None)] + [(f"barrier_k{k}", k) for k in KS]:
            if k is None:
                y, touched = y_sign, np.ones(len(idx), bool)
                b = np.nan
            else:
                b = k * sig[idx] * np.sqrt(H)
                up_hit = up_path[:, :H] >= b[:, None]
                dn_hit = dn_path[:, :H] <= -b[:, None]
                first_up = np.where(up_hit.any(1), up_hit.argmax(1), H + 1)
                first_dn = np.where(dn_hit.any(1), dn_hit.argmax(1), H + 1)
                touched = (first_up <= H) | (first_dn <= H)
                y = (first_up < first_dn).astype(int)  # ties -> lower first (sl_first)
            tr, te = is_train & touched, is_test & touched
            model = LogisticRegression(max_iter=1000).fit(feats[tr], y[tr])
            p = model.predict_proba(feats[te])[:, 1]
            auc = roc_auc_score(y[te], p)
            n = int(te.sum())
            n_eff = n * STEP / H
            se = float(np.sqrt(1.0 / (3.0 * max(n_eff, 1.0))))  # AUC s.e. near 0.5 for balanced classes
            conf = np.abs(p - 0.5)
            top = conf >= np.quantile(conf, 0.9)
            hit_top = float(np.mean((p[top] > 0.5) == (y[te][top] == 1)))
            row = {"H": H, "label": label, "touched_share_test": float(touched[is_test].mean()),
                   "n_test": n, "n_eff": round(n_eff, 1), "auc": round(float(auc), 4),
                   "auc_se_approx": round(se, 4), "z_vs_0.5": round((auc - 0.5) / se, 2),
                   "hit_top10pct": round(hit_top, 4), "n_top10pct": int(top.sum()),
                   "median_barrier_bps": None if k is None else round(float(1e4 * np.median(b[is_test])), 1)}
            res["rows"].append(row)
            print(json.dumps(row))
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
