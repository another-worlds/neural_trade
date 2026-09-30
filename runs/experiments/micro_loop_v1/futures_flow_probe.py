"""Evidence for owner question 8, second probe: do USD-M futures signals (basis = premium index, futures taker
order flow, futures-minus-spot recent return) add directional signal for SPOT BTCUSDT over price + spot flow?

    python runs/experiments/micro_loop_v1/futures_flow_probe.py   # data cached in D:/nt_data_probe (scratch)

Public data.binance.vision: spot 1m klines (from the first probe), futures/um 1m klines and 1m premiumIndexKlines,
2024-01 .. 2025-07. FIT 2024, SCORE 2025-01-01 .. 2025-07-20. Anchors every 5 bars; target: spot close H bars after
the next bar, H in {15, 60, 240}. Models (logistic, fit on FIT only):
  base = spot price features + spot taker imbalance (the first probe's flow model);
  full = base + premium index close and its change over 5/15/60 bars + futures taker imbalance (1/5/15/60) +
         (futures log return - spot log return) over 1/5/15 bars.
Pre-registered bar (as the first probe): SCORE AUC(full) - AUC(base) >= +0.02 with paired block-bootstrap z >= 3,
and AUC(full) >= 0.55, on at least one horizon.
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
OUT = Path(__file__).with_name("futures_flow_probe.json")
MONTHS = [f"{y}-{m:02d}" for y in (2024, 2025) for m in range(1, 13) if (y, m) <= (2025, 7)]
KCOLS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades",
         "taker_buy_base", "taker_buy_quote", "ignore"]
STEP = 5
HS = (15, 60, 240)
SOURCES = {
    "spot": ("spot/monthly/klines", "BTCUSDT-1m-{m}.zip"),
    "fut": ("futures/um/monthly/klines", "fut-BTCUSDT-1m-{m}.zip"),
    "prem": ("futures/um/monthly/premiumIndexKlines", "prem-BTCUSDT-1m-{m}.zip"),
}


def load(kind: str) -> pd.DataFrame:
    path, local = SOURCES[kind]
    frames = []
    for m in MONTHS:
        f = CACHE / local.format(m=m)
        if not f.exists():
            url = f"https://data.binance.vision/data/{path}/BTCUSDT/1m/BTCUSDT-1m-{m}.zip"
            f.write_bytes(urllib.request.urlopen(url, timeout=120).read())
        with zipfile.ZipFile(f) as z:
            raw = z.read(z.namelist()[0])
        first = raw.split(b"\n", 1)[0]
        header = 0 if first[:1].isalpha() else None  # futures files carry a header row
        d = pd.read_csv(io.BytesIO(raw), header=header)
        d = d.iloc[:, :12]
        d.columns = KCOLS
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    t = pd.to_numeric(df.open_time, errors="coerce").astype("int64")
    t = np.where(t > 1e15, t // 1000, t)
    df["ts"] = pd.to_datetime(t, unit="ms")
    for col in ("close", "volume", "taker_buy_base", "trades"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values("ts").drop_duplicates("ts").set_index("ts")


def imbalance(df, L_list, index):
    imb = np.cumsum(np.nan_to_num(2 * df.taker_buy_base.to_numpy() - df.volume.to_numpy()))
    cv = np.cumsum(np.nan_to_num(df.volume.to_numpy()))
    return [(imb[index] - imb[index - L]) / (cv[index] - cv[index - L] + 1e-9) for L in L_list]


def block_boot_z(y, p1, p0, reps=300, rng=np.random.default_rng(0)):
    n = len(y)
    b = 16
    starts = np.arange(0, n - b, b)
    diffs = []
    for _ in range(reps):
        ix = (rng.choice(starts, size=len(starts))[:, None] + np.arange(b)[None, :]).ravel()
        if len(np.unique(y[ix])) > 1:
            diffs.append(roc_auc_score(y[ix], p1[ix]) - roc_auc_score(y[ix], p0[ix]))
    d = roc_auc_score(y, p1) - roc_auc_score(y, p0)
    sd = float(np.std(diffs, ddof=1))
    return float(d), (float(d / sd) if sd > 0 else float("nan"))


def main() -> None:
    spot, fut, prem = load("spot"), load("fut"), load("prem")
    common = spot.index.intersection(fut.index).intersection(prem.index)
    spot, fut, prem = spot.loc[common], fut.loc[common], prem.loc[common]
    c = spot.close.to_numpy(float)
    fc = fut.close.to_numpy(float)
    pc = prem.close.to_numpy(float)
    lr = np.diff(np.log(c), prepend=np.log(c[0]))
    sig = np.sqrt(pd.Series(lr ** 2).ewm(halflife=60, adjust=False).mean().to_numpy())
    idx = np.arange(1500, len(c) - max(HS) - 2, STEP)
    ts = common.to_numpy()[idx]
    fit = ts < np.datetime64("2025-01-01")
    score = (ts >= np.datetime64("2025-01-01")) & (ts < np.datetime64("2025-07-21"))
    price = [np.log(c[idx] / c[idx - L]) for L in (1, 5, 15, 60, 240)] + [np.log(sig[idx] + 1e-12)]
    base = np.column_stack(price + imbalance(spot, (1, 5, 15, 60), idx))
    extra = ([pc[idx], pc[idx] - pc[idx - 5], pc[idx] - pc[idx - 15], pc[idx] - pc[idx - 60]]
             + imbalance(fut, (1, 5, 15, 60), idx)
             + [np.log(fc[idx] / fc[idx - L]) - np.log(c[idx] / c[idx - L]) for L in (1, 5, 15)])
    full = np.column_stack([base] + extra)
    full = np.nan_to_num(full)
    base = np.nan_to_num(base)
    res = {"months": f"{MONTHS[0]}..{MONTHS[-1]}", "n_common_bars": int(len(common)), "rows": []}
    for H in HS:
        y = (c[idx + 1 + H] > c[idx + 1]).astype(int)
        m0 = LogisticRegression(max_iter=3000).fit(base[fit], y[fit])
        m1 = LogisticRegression(max_iter=3000).fit(full[fit], y[fit])
        p0, p1 = m0.predict_proba(base[score])[:, 1], m1.predict_proba(full[score])[:, 1]
        d, z = block_boot_z(y[score], p1, p0)
        a0, a1 = roc_auc_score(y[score], p0), roc_auc_score(y[score], p1)
        top = np.abs(p1 - 0.5) >= np.quantile(np.abs(p1 - 0.5), 0.9)
        row = {"H": H, "n_score": int(score.sum()), "auc_base_spotflow": round(float(a0), 4),
               "auc_full": round(float(a1), 4), "lift": round(d, 4), "lift_z": round(z, 2),
               "hit_full": round(float(np.mean((p1 > 0.5) == (y[score] == 1))), 4),
               "hit_full_top10": round(float(np.mean((p1[top] > 0.5) == (y[score][top] == 1))), 4)}
        row["meets_bar"] = bool(d >= 0.02 and z >= 3 and a1 >= 0.55)
        res["rows"].append(row)
        print(json.dumps(row))
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
