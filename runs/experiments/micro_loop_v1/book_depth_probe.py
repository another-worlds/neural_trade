"""Evidence for owner question 8, third probe: does ORDER-BOOK DEPTH (Binance USD-M futures bookDepth: cumulative
depth at +-1..5% of price, sampled about every minute) add directional signal over price + spot order flow?

    python runs/experiments/micro_loop_v1/book_depth_probe.py   # data cached in D:/nt_data_probe (scratch)

Public data.binance.vision futures/um/daily/bookDepth 2025-01-01 .. 2025-07-20 plus the spot 1m klines of the
first probe. FIT 2025-01-01 .. 2025-04-30, SCORE 2025-05-01 .. 2025-07-20. Anchors every minute where depth
exists (then every 5th); target: spot close H bars after the next bar, H in {1, 5, 15, 60}.
Depth features at the decision minute (latest snapshot at or before it): for each level L in {1, 2, 5} %:
log(bid depth / ask depth) (imbalance), its change over 5 and 15 minutes, and log total depth.
Models (logistic, FIT only): base = price + spot taker imbalance (the first probe's flow model);
full = base + depth features.
Pre-registered bars: (a) as the other probes: AUC(full) - AUC(base) >= +0.02, paired block-bootstrap z >= 3,
AUC(full) >= 0.55; (b) the owner's target read directly: hit rate of the full model's calls >= 0.60 on all SCORE
bars at some horizon (and the top-10% hit reported, not counted, since it is selected).
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

CACHE = Path("D:/nt_data_probe/bookDepth")
SPOT = Path("D:/nt_data_probe")
OUT = Path(__file__).with_name("book_depth_probe.json")
DAYS = pd.date_range("2025-01-01", "2025-07-20", freq="D")
HS = (1, 5, 15, 60)
STEP = 5


def load_depth() -> pd.DataFrame:
    CACHE.mkdir(parents=True, exist_ok=True)
    frames = []
    for d in DAYS:
        s = d.strftime("%Y-%m-%d")
        f = CACHE / f"BTCUSDT-bookDepth-{s}.zip"
        if not f.exists():
            url = f"https://data.binance.vision/data/futures/um/daily/bookDepth/BTCUSDT/BTCUSDT-bookDepth-{s}.zip"
            try:
                f.write_bytes(urllib.request.urlopen(url, timeout=120).read())
            except Exception:
                continue
        with zipfile.ZipFile(f) as z:
            frames.append(pd.read_csv(io.BytesIO(z.read(z.namelist()[0]))))
    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["minute"] = df["timestamp"].dt.floor("min")
    wide = df.pivot_table(index="minute", columns="percentage", values="depth", aggfunc="last")
    return wide.sort_index()


def load_spot() -> pd.DataFrame:
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades",
            "taker_buy_base", "taker_buy_quote", "ignore"]
    frames = []
    for m in ("2025-01", "2025-02", "2025-03", "2025-04", "2025-05", "2025-06", "2025-07"):
        with zipfile.ZipFile(SPOT / f"BTCUSDT-1m-{m}.zip") as z:
            frames.append(pd.read_csv(io.BytesIO(z.read(z.namelist()[0])), header=None, names=cols))
    df = pd.concat(frames, ignore_index=True)
    t = df.open_time.astype("int64")
    t = np.where(t > 1e15, t // 1000, t)
    df["ts"] = pd.to_datetime(t, unit="ms")
    return df.drop_duplicates("ts").set_index("ts").sort_index()


def block_boot_z(y, p1, p0, reps=300, rng=np.random.default_rng(0)):
    b = 16
    starts = np.arange(0, len(y) - b, b)
    diffs = []
    for _ in range(reps):
        ix = (rng.choice(starts, size=len(starts))[:, None] + np.arange(b)[None, :]).ravel()
        if len(np.unique(y[ix])) > 1:
            diffs.append(roc_auc_score(y[ix], p1[ix]) - roc_auc_score(y[ix], p0[ix]))
    d = roc_auc_score(y, p1) - roc_auc_score(y, p0)
    sd = float(np.std(diffs, ddof=1))
    return float(d), (float(d / sd) if sd > 0 else float("nan"))


def main() -> None:
    depth = load_depth()
    spot = load_spot()
    depth = depth.reindex(spot.index, method="ffill", limit=2)
    c = spot.close.to_numpy(float)
    v = spot.volume.to_numpy(float)
    tb = spot.taker_buy_base.to_numpy(float)
    lr = np.diff(np.log(c), prepend=np.log(c[0]))
    sig = np.sqrt(pd.Series(lr ** 2).ewm(halflife=60, adjust=False).mean().to_numpy())
    imb = np.cumsum(2 * tb - v)
    cv = np.cumsum(v)
    feats = {}
    for L in (1, 2, 5):
        if L in depth.columns and -L in depth.columns:
            x = np.log(depth[L].to_numpy(float) / depth[-L].to_numpy(float))  # positive % = asks side? sign-agnostic
            feats[f"imb{L}"] = x
            feats[f"tot{L}"] = np.log(depth[L].to_numpy(float) + depth[-L].to_numpy(float))
    idx = np.arange(1500, len(c) - max(HS) - 2, STEP)
    ok = np.all(np.isfinite(np.column_stack([feats[k][idx] for k in feats] +
                                            [feats[k][idx - 15] for k in feats if k.startswith("imb")])), axis=1)
    idx = idx[ok]
    ts = spot.index.to_numpy()[idx]
    fit = ts < np.datetime64("2025-05-01")
    score = (ts >= np.datetime64("2025-05-01")) & (ts < np.datetime64("2025-07-21"))
    base = np.column_stack([np.log(c[idx] / c[idx - L]) for L in (1, 5, 15, 60, 240)] + [np.log(sig[idx] + 1e-12)]
                           + [(imb[idx] - imb[idx - L]) / (cv[idx] - cv[idx - L] + 1e-9) for L in (1, 5, 15, 60)])
    extra = []
    for k, x in feats.items():
        extra.append(x[idx])
        if k.startswith("imb"):
            extra += [x[idx] - x[idx - 5], x[idx] - x[idx - 15]]
    full = np.column_stack([base] + extra)
    keep = np.isfinite(full).all(axis=1)           # the same rows for both models
    idx, fit, score, base, full = idx[keep], fit[keep], score[keep], base[keep], full[keep]
    res = {"days": f"{DAYS[0].date()}..{DAYS[-1].date()}", "depth_levels": sorted(int(k) for k in depth.columns),
           "n_anchors": int(len(idx)), "fit": "2025-01..04", "score": "2025-05..07-20", "rows": []}
    for H in HS:
        y = (c[idx + 1 + H] > c[idx + 1]).astype(int)
        m0 = LogisticRegression(max_iter=3000).fit(base[fit], y[fit])
        m1 = LogisticRegression(max_iter=3000).fit(full[fit], y[fit])
        p0, p1 = m0.predict_proba(base[score])[:, 1], m1.predict_proba(full[score])[:, 1]
        d, z = block_boot_z(y[score], p1, p0)
        a0, a1 = roc_auc_score(y[score], p0), roc_auc_score(y[score], p1)
        hit = float(np.mean((p1 > 0.5) == (y[score] == 1)))
        top = np.abs(p1 - 0.5) >= np.quantile(np.abs(p1 - 0.5), 0.9)
        move = float(1e4 * np.median(np.abs(np.log(c[idx + 1 + H] / c[idx + 1])[score])))
        row = {"H": H, "n_score": int(score.sum()), "auc_base": round(float(a0), 4), "auc_full": round(float(a1), 4),
               "lift": round(d, 4), "lift_z": round(z, 2), "hit_full": round(hit, 4),
               "hit_full_top10": round(float(np.mean((p1[top] > 0.5) == (y[score][top] == 1))), 4),
               "median_abs_move_bps": round(move, 1)}
        row["meets_bar_a"] = bool(d >= 0.02 and z >= 3 and a1 >= 0.55)
        row["meets_target_60"] = bool(hit >= 0.60)
        res["rows"].append(row)
        print(json.dumps(row))
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
