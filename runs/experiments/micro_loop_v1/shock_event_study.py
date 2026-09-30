"""Micro loop H6b: after a large move (|60-bar log return| >= k * EWMA sigma * sqrt(60)), is the next H-bar
direction predictable (continuation or reversal), out of sample on years of data?

    python runs/experiments/micro_loop_v1/shock_event_study.py   # writes shock_event_study.json next to this file

Data: Bitcoin_BTCUSDT.csv; FIT 2017-06-01 .. 2022-12-31, SCORE 2023-01-01 .. 2025-07-20 (before the protected
last 64 days). Events: every 5th bar where the shock condition holds, k in {2, 3}; horizons 15 and 60 bars.
Two predictors, both fitted on FIT only:
  (a) the sign rule: predict continuation (or reversal, whichever FIT prefers) of the 60-bar move;
  (b) a logistic regression on the shock's signed size, lagged returns (1, 5, 15, 240), log sigma, hour sin/cos.
Pre-registered bar (the same as H6): SCORE AUC >= 0.56 (or hit >= 0.56 for the rule) with z >= 3 at n_eff >= 200,
and the FIT block pointing the same way. Also reported: the median |H-bar move| in bps after the event and the
gross edge per trade in bps of the rule on SCORE (entry at the next bar's close proxy), against the 26 bps cost.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

CSV = Path("D:/neural_trade/Bitcoin_BTCUSDT.csv")
OUT = Path(__file__).with_name("shock_event_study.json")
STEP = 5


def main() -> None:
    df = pd.read_csv(CSV, usecols=["timestamp", "close"], parse_dates=["timestamp"])
    df = df[(df.timestamp >= "2017-05-01") & (df.timestamp < "2025-07-21")].reset_index(drop=True)
    c = df.close.to_numpy(float)
    lr = np.diff(np.log(c), prepend=np.log(c[0]))
    sig = np.sqrt(pd.Series(lr ** 2).ewm(halflife=60, adjust=False).mean().to_numpy())
    hour = df.timestamp.dt.hour.to_numpy()
    ts = df.timestamp.to_numpy()
    idx = np.arange(1500, len(c) - 62, STEP)
    ret60 = np.log(c[idx] / c[idx - 60])
    z60 = ret60 / (sig[idx] * np.sqrt(60) + 1e-12)
    fit_t = (ts[idx] >= np.datetime64("2017-06-01")) & (ts[idx] < np.datetime64("2023-01-01"))
    score_t = ts[idx] >= np.datetime64("2023-01-01")
    res = {"fit": "2017-06..2022-12", "score": "2023-01..2025-07-20", "rows": []}
    for k in (2.0, 3.0):
        ev = np.abs(z60) >= k
        for H in (15, 60):
            fwd = np.log(c[idx + 1 + H] / c[idx + 1])       # entry one bar after the decision bar
            y = (fwd > 0).astype(int)
            cont = (np.sign(ret60) == np.sign(fwd)).astype(int)
            fit, score = ev & fit_t, ev & score_t
            fit_cont = float(cont[fit].mean())
            direction = 1 if fit_cont >= 0.5 else -1              # continuation or reversal, chosen on FIT
            pred_up = (np.sign(ret60) * direction > 0).astype(int)
            hit_score = float((pred_up[score] == y[score]).mean())
            n = int(score.sum())
            n_eff = n * STEP / H
            se_hit = float(np.sqrt(0.25 / max(n_eff, 1)))
            edge_bps = float(1e4 * np.mean(np.where(pred_up[score] == 1, fwd[score], -fwd[score])))
            X = np.column_stack([z60, np.log(c[idx] / c[idx - 1]), np.log(c[idx] / c[idx - 5]),
                                 np.log(c[idx] / c[idx - 15]), np.log(c[idx] / c[idx - 240]),
                                 np.log(sig[idx] + 1e-12), np.sin(2 * np.pi * hour[idx] / 24),
                                 np.cos(2 * np.pi * hour[idx] / 24)])
            model = LogisticRegression(max_iter=2000).fit(X[fit], y[fit])
            p = model.predict_proba(X[score])[:, 1]
            auc = float(roc_auc_score(y[score], p))
            se_auc = float(np.sqrt(1 / (3 * max(n_eff, 1))))
            row = {"k": k, "H": H, "n_fit_events": int(fit.sum()), "n_score_events": n, "n_eff": round(n_eff, 1),
                   "fit_continuation_share": round(fit_cont, 4), "rule": "continuation" if direction == 1 else "reversal",
                   "rule_hit_score": round(hit_score, 4), "rule_z": round((hit_score - 0.5) / se_hit, 2),
                   "rule_gross_edge_bps": round(edge_bps, 2),
                   "median_abs_move_bps": round(float(1e4 * np.median(np.abs(fwd[score]))), 1),
                   "logreg_auc_score": round(auc, 4), "logreg_z": round((auc - 0.5) / se_auc, 2)}
            row["meets_bar"] = bool(n_eff >= 200 and ((row["rule_hit_score"] >= 0.56 and row["rule_z"] >= 3)
                                                      or (auc >= 0.56 and row["logreg_z"] >= 3)))
            res["rows"].append(row)
            print(json.dumps(row))
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
