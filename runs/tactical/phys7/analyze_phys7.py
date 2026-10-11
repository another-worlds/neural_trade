"""phys7 analysis (SPEC.md in this folder): the variance head's Spearman with |r| and QLIKE after re-scaling, per horizon,
network against HAR+tod, on the OOS block of each slice's fold -2.  No GPU.
usage: python analyze_phys7.py score <screen> [<screens_root>]   -> phys7/scores_<screen>.json   (one row per slice x seed x forecast)
       python analyze_phys7.py report                            -> prints the tables, writes phys7/report_tables.json
Definitions (fixed in SPEC.md before any run):
  r_h        close[t+h]/close[t]-1 for the window ending at bar t, h = 10, 15, 20 bars (checked against the saved y / last_close)
  Spearman   rank correlation of the head's variance with |r_h| over the whole OOS block (n = 12,600 windows)
  QLIKE      mean(x - log x - 1), x = r_h^2 / (c * s2), over the SECOND half B of the OOS block; the scale c = mean(r_h^2 / s2)
             is fitted on the FIRST half A (the cal block is not saved by the screen).  Same A/B rule for the net and HAR.
  HAR+tod    volbench's regression: log future variance on the log trailing variances over 15/60/240/1440 bars and the first two
             time-of-day harmonics, OLS per horizon, fitted on the bars of the same data segment that precede the OOS block by at
             least 81 bars (the net's own train/val/cal span), every 5th bar.  s2 = exp(fit) * h.
Hard cutoff: asserts that no bar used is after 2025-07-20 (owner rule after H66)."""
import glob
import json
import math
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
TAC = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(TAC, "vol"))
import volbench as vb  # noqa: E402  (its pure helpers only; its CSV is not read)

CSV = "D:/nt/neural_trade/Bitcoin_BTCUSDT_tactical.csv"
CUTOFF = pd.Timestamp("2025-07-20 23:59:59")
HZ = (10, 15, 20)
GAP = 81


def load_csv():
    d = pd.read_csv(CSV, usecols=["timestamp", "close"])
    t = pd.to_datetime(d["timestamp"])
    close = d["close"].to_numpy(np.float64)
    tod = (t.dt.hour * 60 + t.dt.minute).to_numpy(np.float64)
    return t, close, tod


def locate(close32, last_close):
    n = len(last_close)
    for i in np.where(close32 == last_close[0])[0]:
        if i + n <= len(close32) and np.array_equal(close32[i:i + n], last_close):
            return int(i)
    raise RuntimeError("OOS block not found in the excerpt")


def score_slice(P, t, close, tod, lr, cs, close32):
    lc = P["last_close"].astype(np.float32)
    n = len(lc)
    i0 = locate(close32, lc)
    idx = i0 + np.arange(n)
    assert t.iloc[idx[-1] + max(HZ)] <= CUTOFF, "bar after the cutoff"
    r = np.stack([close[idx + h] / close[idx] - 1.0 for h in HZ], 1)
    # the saved y is the price change in quote currency: check it against r (loose: float32)
    assert np.allclose(P["y"] / lc[:, None], r, rtol=2e-3, atol=2e-6), "saved y differs from the csv returns"
    # HAR training span: the same contiguous data segment, bars at least GAP before the OOS block
    gaps = np.where(np.diff(t.values[:i0 + 1]).astype("timedelta64[s]").astype(np.int64) != 60)[0]
    seg0 = int(gaps[-1]) + 1 if len(gaps) else 0
    tmin = seg0 + 1440
    tmax = i0 - GAP
    assert tmax - tmin > 20000, "too little HAR training data"
    sub = np.arange(tmin, tmax, 5)
    logv = [np.log(vb.trailing_var(cs, k) + 1e-12) for k in vb.HAR_LAGS]
    s2_har = np.zeros((n, 3))
    for hi, h in enumerate(HZ):
        X = vb.har_design([lv[sub] for lv in logv], tod[sub] + h / 2.0)
        coef = vb.ols(X, np.log(vb.future_var(cs, sub, h) + 1e-12))
        Xo = vb.har_design([lv[idx] for lv in logv], tod[idx] + h / 2.0)
        s2_har[:, hi] = np.exp(Xo @ coef) * h
    A, B = slice(0, n // 2), slice(n // 2, n)
    out = {}
    for name in ("net", "har_tod"):
        sp, ql = [], []
        for hi in range(3):
            s2 = P[f"var_h{hi}"].astype(np.float64) if name == "net" else s2_har[:, hi]
            s2 = np.maximum(s2, 1e-14)
            r2 = np.maximum(r[:, hi] ** 2, vb.R2_FLOOR)
            sp.append(float(stats.spearmanr(s2, np.abs(r[:, hi]))[0]))
            c = float(np.mean(r2[A] / s2[A]))
            ql.append(vb.qlike(r2[B], s2[B] * c))
        out[name] = dict(spearman_h=sp, qlike_h=ql, spearman=float(np.mean(sp)), qlike=float(np.mean(ql)))
    out["info"] = dict(i0=i0, start=str(t.iloc[i0]), end=str(t.iloc[idx[-1] + 20]), n=n, har_train_bars=len(sub),
                       const_var=[bool(np.std(P[f"var_h{h}"]) < 1e-9) for h in range(3)])
    return out


def score(screen, root):
    t, close, tod = load_csv()
    lr = np.concatenate([[0.0], np.diff(np.log(np.maximum(close, 1e-9)))])
    cs = vb.cum_sq(lr)
    close32 = close.astype(np.float32)
    rows = []
    for f in sorted(glob.glob(os.path.join(root, "screens", screen, "results*.jsonl"))):
        for l in open(f, encoding="utf-8"):
            x = json.loads(l)
            if not x.get("head_metrics"):
                continue
            p = os.path.join(root, "screens", screen, "preds", x["trial_key"] + ".npz")
            P = np.load(p)
            o = score_slice(P, t, close, tod, lr, cs, close32)
            rows.append(dict(screen=screen, slice=x["data_end"], seed=x["seed"], trial_key=x["trial_key"],
                             wall_s=x.get("wall_s"), **o))
            print(screen, x["data_end"], x["seed"], "net", round(o["net"]["spearman"], 4), "har", round(o["har_tod"]["spearman"], 4),
                  flush=True)
    json.dump(rows, open(os.path.join(HERE, f"scores_{screen}.json"), "w"), indent=1)


def tci(v):
    v = np.asarray(v, float)
    n = len(v)
    m = v.mean()
    hw = stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / math.sqrt(n)
    return [float(m), float(m - hw), float(m + hw), float(v.std(ddof=1)), n]


def per_slice(rows, who, key):
    """slice -> value averaged over seeds (and horizons for the scalar keys)."""
    d = {}
    for r in rows:
        d.setdefault(r["slice"], []).append(r[who][key])
    return {s: np.mean(v, axis=0) for s, v in d.items()}


def paired(a, b):
    s = sorted(set(a) & set(b))
    return np.array([a[k] - b[k] for k in s]), s


def report():
    cells = [("none", "off"), ("none", "on"), ("price", "off"), ("price", "on")]
    R = {}
    for ph, phys in cells:
        f = os.path.join(HERE, f"scores_phys7_{ph}_{phys}.json")
        if os.path.exists(f):
            R[(ph, phys)] = json.load(open(f))
    out = {}
    print("cell (price head, physics): n runs; mean over slices of net / HAR+tod Spearman, QLIKE (second half B)")
    for k, rows in R.items():
        ns = per_slice(rows, "net", "spearman"); hs = per_slice(rows, "har_tod", "spearman")
        nq = per_slice(rows, "net", "qlike"); hq = per_slice(rows, "har_tod", "qlike")
        print(k, len(rows), "net sp %.4f  har sp %.4f  net ql %.4f  har ql %.4f" % (np.mean(list(ns.values())), np.mean(list(hs.values())),
              np.mean(list(nq.values())), np.mean(list(hq.values()))))
    for ph in ("none", "price"):
        if (ph, "on") in R and (ph, "off") in R:
            for key, lab in (("spearman", "Spearman"), ("qlike", "QLIKE")):
                on = per_slice(R[(ph, "on")], "net", key); off = per_slice(R[(ph, "off")], "net", key)
                d, s = paired(on, off)
                out[f"{ph}_{key}_on_minus_off"] = tci(d)
                print(f"PHYSICS on-off, price head {ph}, {lab} (mean of h): mean/lo/hi/sd/n", [round(x, 4) for x in tci(d)], "slices up:", int((d > 0).sum()))
                for hi, h in enumerate(HZ):
                    on_h = {k: v[hi] for k, v in per_slice(R[(ph, "on")], "net", key + "_h").items()}
                    off_h = {k: v[hi] for k, v in per_slice(R[(ph, "off")], "net", key + "_h").items()}
                    d2, _ = paired(on_h, off_h)
                    out[f"{ph}_{key}_h{hi}_on_minus_off"] = tci(d2)
                    print(f"   h{hi} ({h} bars)", [round(x, 4) for x in tci(d2)])
        for phys in ("on", "off"):
            if (ph, phys) in R:
                for key, lab in (("spearman", "Spearman"), ("qlike", "QLIKE")):
                    nn = per_slice(R[(ph, phys)], "net", key); hh = per_slice(R[(ph, phys)], "har_tod", key)
                    d, _ = paired(nn, hh)
                    out[f"{ph}_{phys}_{key}_net_minus_har"] = tci(d)
                    print(f"NET - HAR+tod, price head {ph}, physics {phys}, {lab}:", [round(x, 4) for x in tci(d)])
    json.dump(out, open(os.path.join(HERE, "report_tables.json"), "w"), indent=1)


def seed_noise(screen, root):
    """sd of the seed-0 minus seed-1 paired difference of the variance Spearman, per slice (existing runs)."""
    score(screen, root)
    rows = json.load(open(os.path.join(HERE, f"scores_{screen}.json")))
    by = {}
    for r in rows:
        by.setdefault(r["slice"], {})[r["seed"]] = r["net"]["spearman"]
    d = np.array([v[0] - v[1] for v in by.values() if 0 in v and 1 in v])
    print("seed0-seed1 diffs", np.round(d, 4), "sd", d.std(ddof=1), "n", len(d))
    s_net = np.array([r["net"]["spearman"] for r in rows]); s_har = np.array([r["har_tod"]["spearman"] for r in rows])
    print("mean net", s_net.mean(), "mean har", s_har.mean())


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "score":
        score(a[1], a[2] if len(a) > 2 else os.path.dirname(HERE))
    elif a[0] == "noise":
        seed_noise(a[1], a[2] if len(a) > 2 else os.path.dirname(HERE))
    else:
        report()
