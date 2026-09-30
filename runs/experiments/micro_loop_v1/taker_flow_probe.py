"""Evidence for owner question 8 (a new data source): does Binance taker-buy volume (order flow) add directional
signal over price-only features? A measurement, not an integration: no product code, no data in the repo.

    python runs/experiments/micro_loop_v1/taker_flow_probe.py    # downloads to D:/nt_data_probe (scratch),
                                                                  # writes taker_flow_probe.json next to this file

Data: data.binance.vision spot BTCUSDT 1m monthly klines 2024-01 .. 2025-07 (public; columns include the taker buy
base volume). FIT 2024-01-01 .. 2024-12-31, SCORE 2025-01-01 .. 2025-07-20 (before the protected last 64 days of the
local file). Anchors every 5 bars; horizons 15, 60, 240 bars.
Models (logistic regression, fitted on FIT only): price-only = log returns over 1, 5, 15, 60, 240 bars + log EWMA
sigma; flow = price-only + taker-buy imbalance (2 * taker_buy / volume - 1) summed over the last 1, 5, 15, 60 bars
(volume-weighted) + log volume ratio (60 vs 1440 bars) + trade-count ratio.
Pre-registered bar: SCORE AUC(flow) - AUC(price-only) >= +0.02 with a paired bootstrap z >= 3 (80-bar blocks) on at
least one horizon, and AUC(flow) >= 0.55 there.
"""
from __future__ import annotations

import io
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

CACHE = Path("D:/nt_data_probe")
OUT = Path(__file__).with_name("taker_flow_probe.json")
MONTHS = [f"{y}-{m:02d}" for y in (2024, 2025) for m in range(1, 13) if (y, m) <= (2025, 7)]
COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore"]
STEP = 5
HS = (15, 60, 240)


def load() -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    frames = []
    for m in MONTHS:
        f = CACHE / f"BTCUSDT-1m-{m}.zip"
        if not f.exists():
            url = f"https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-{m}.zip"
            f.write_bytes(urllib.request.urlopen(url, timeout=120).read())
        with zipfile.ZipFile(f) as z:
            frames.append(pd.read_csv(io.BytesIO(z.read(z.namelist()[0])), header=None, names=COLS))
    df = pd.concat(frames, ignore_index=True)
    t = df.open_time.astype("int64")
    t = np.where(t > 1e15, t // 1000, t)  # 2025 files use microseconds
    df["ts"] = pd.to_datetime(t, unit="ms")
    return df.sort_values("ts").drop_duplicates("ts").reset_index(drop=True)


def block_boot_z(y, p1, p0, h, reps=300, rng=np.random.default_rng(0)):
    n = len(y)
    b = max(1, 80 // STEP * 1)
    starts = np.arange(0, n - b, b)
    diffs = []
    for _ in range(reps):
        s = rng.choice(starts, size=len(starts))
        ix = (s[:, None] + np.arange(b)[None, :]).ravel()
        if len(np.unique(y[ix])) < 2:
            continue
        diffs.append(roc_auc_score(y[ix], p1[ix]) - roc_auc_score(y[ix], p0[ix]))
    d = roc_auc_score(y, p1) - roc_auc_score(y, p0)
    sd = float(np.std(diffs, ddof=1))
    return float(d), (float(d / sd) if sd > 0 else float("nan"))


def main() -> None:
    df = load()
    c = df.close.to_numpy(float)
    v = df.volume.to_numpy(float)
    tb = df.taker_buy_base.to_numpy(float)
    tr = df.trades.to_numpy(float)
    lr = np.diff(np.log(c), prepend=np.log(c[0]))
    sig = np.sqrt(pd.Series(lr ** 2).ewm(halflife=60, adjust=False).mean().to_numpy())
    imb_bar = (2 * tb - v)  # signed taker volume (buy minus sell), base units
    cs_imb = np.cumsum(imb_bar)
    cs_v = np.cumsum(v)
    vs, ts_ = pd.Series(v), pd.Series(tr)
    vol_ratio = np.log((vs.rolling(60).mean() + 1e-9) / (vs.rolling(1440).mean() + 1e-9)).to_numpy()
    tr_ratio = np.log((ts_.rolling(60).mean() + 1e-9) / (ts_.rolling(1440).mean() + 1e-9)).to_numpy()
    idx = np.arange(1500, len(c) - max(HS) - 2, STEP)
    ts = df.ts.to_numpy()[idx]
    fit = ts < np.datetime64("2025-01-01")
    score = (ts >= np.datetime64("2025-01-01")) & (ts < np.datetime64("2025-07-21"))
    price = np.column_stack([np.log(c[idx] / c[idx - L]) for L in (1, 5, 15, 60, 240)] + [np.log(sig[idx] + 1e-12)])
    flow_feats = [(cs_imb[idx] - cs_imb[idx - L]) / (cs_v[idx] - cs_v[idx - L] + 1e-9) for L in (1, 5, 15, 60)]
    flow = np.column_stack([price] + flow_feats + [np.nan_to_num(vol_ratio[idx]), np.nan_to_num(tr_ratio[idx])])
    res = {"months": f"{MONTHS[0]}..{MONTHS[-1]}", "fit": "2024", "score": "2025-01..2025-07-20", "rows": []}
    for H in HS:
        y = (c[idx + 1 + H] > c[idx + 1]).astype(int)
        m0 = LogisticRegression(max_iter=3000).fit(price[fit], y[fit])
        m1 = LogisticRegression(max_iter=3000).fit(flow[fit], y[fit])
        p0 = m0.predict_proba(price[score])[:, 1]
        p1 = m1.predict_proba(flow[score])[:, 1]
        d, z = block_boot_z(y[score], p1, p0, H)
        a0, a1 = roc_auc_score(y[score], p0), roc_auc_score(y[score], p1)
        conf = np.abs(p1 - 0.5)
        top = conf >= np.quantile(conf, 0.9)
        row = {"H": H, "n_score": int(score.sum()), "n_eff": round(score.sum() * STEP / H, 1),
               "auc_price": round(float(a0), 4), "auc_flow": round(float(a1), 4), "lift": round(d, 4),
               "lift_z": round(z, 2), "hit_flow": round(float(np.mean((p1 > 0.5) == (y[score] == 1))), 4),
               "hit_flow_top10": round(float(np.mean((p1[top] > 0.5) == (y[score][top] == 1))), 4)}
        row["meets_bar"] = bool(d >= 0.02 and z >= 3 and a1 >= 0.55)
        res["rows"].append(row)
        print(json.dumps(row))
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
