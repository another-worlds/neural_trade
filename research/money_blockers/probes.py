"""Probes for plans/money_blockers.md (descriptive; thresholds fixed on pre-Test data, then applied to Test).

1. Volatility regimes: zero-shot Timestral-28M predicted sigma (IQR / 1.349) vs realised |move|, and the
   break-even accuracy at 4 bps round trip per regime (wins and losses equal to the regime's mean |move|).
2. Volatility-gated reversal: fade the last H-minute move only when predicted sigma_H is in its pre-Test top 10%;
   hold H minutes (time exit) or use a +-1 sigma bracket (stop checked first inside a bar); no overlap.
3. Implied vs realised volatility: Deribit DVOL (30-day implied, hourly) against forward 24 h and 7 d realised
   volatility from 1-min closes, over each data window.
Needs the Timestral-LIG work cache (research/timestral_lig, `train.py cache`) and network access for DVOL.
"""
import json
import os
import sys
import urllib.request

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "timestral_lig"))
import train  # noqa: E402


def regimes(X):
    D = X.D
    lc = D.lc
    cal = np.unique(np.concatenate([D.train[::2], D.v1, D.v2]))
    te = D.test
    for H, hi in ((5, 4), (15, 14)):
        qc, qt = X.q0(cal), X.q0(te)
        sc = (qc[:, hi, 15] - qc[:, hi, 5]) / 1.349
        st = (qt[:, hi, 15] - qt[:, hi, 5]) / 1.349
        edges = np.quantile(sc, [0, .5, .8, .9, .95, 1.0])
        edges[0], edges[-1] = -np.inf, np.inf
        yc = np.abs((lc[cal + H] - lc[cal]) * 1e4)
        yt = np.abs((lc[te + H] - lc[te]) * 1e4)
        print(f" H={H}: corr(pred sigma, |move|) pre-Test {np.corrcoef(sc, yc)[0, 1]:.2f}, Test {np.corrcoef(st, yt)[0, 1]:.2f}")
        for a, b, lab in ((0, 1, "bottom 50%"), (1, 2, "50-80%"), (2, 3, "80-90%"), (3, 4, "90-95%"), (4, 5, "top 5%")):
            mc = (sc >= edges[a]) & (sc < edges[b])
            mt = (st >= edges[a]) & (st < edges[b])
            print(f"   {lab:10s} mean|move| {yc[mc].mean():5.1f} -> {yt[mt].mean():5.1f} bps; "
                  f"break-even accuracy at 4 bps {0.5 + 2 / yc[mc].mean():.0%} -> {0.5 + 2 / yt[mt].mean():.0%} (n Test {mt.sum()})")


def sim(org, H, d, lc, high, low, close, sig_bps, bracket):
    out, nxt = [], -1
    for i in np.flatnonzero(d != 0):
        t = org[i]
        if t < nxt:
            continue
        if not bracket:
            r = (lc[t + H] - lc[t]) * 1e4 * d[i]
        else:
            b = sig_bps[i] / 1e4
            tp, sl = close[t] * np.exp(d[i] * b), close[t] * np.exp(-d[i] * b)
            r = None
            for k in range(1, H + 1):
                hit_sl = (low[t + k] <= sl) if d[i] > 0 else (high[t + k] >= sl)
                hit_tp = (high[t + k] >= tp) if d[i] > 0 else (low[t + k] <= tp)
                if hit_sl:
                    r = -b * 1e4
                    break
                if hit_tp:
                    r = b * 1e4
                    break
            if r is None:
                r = (lc[t + H] - lc[t]) * 1e4 * d[i]
        out.append(r)
        nxt = t + H
    return np.array(out)


def gated_fade(X):
    D = X.D
    lc = D.lc
    cal = np.unique(np.concatenate([D.train[::2], D.v1, D.v2]))
    for H, hi in ((5, 4), (15, 14)):
        thr = None
        for lab, org in (("pre-Test", cal), ("Test", D.test)):
            qq = X.q0(org)
            s = (qq[:, hi, 15] - qq[:, hi, 5]) / 1.349
            if thr is None:
                thr = np.quantile(s, 0.9)
            d = np.where(s >= thr, -np.sign((lc[org] - lc[org - H]) * 1e4), 0.0)
            days = (org[-1] - org[0]) / 1440
            for bracket in (False, True):
                p = sim(org, H, d, lc, D.high, D.low, D.close, s, bracket)
                dec = p[p != 0]
                t = p.mean() / (p.std(ddof=1) / np.sqrt(len(p)))
                print(f"  H={H:2d} {lab:8s} {'bracket +-1 sigma' if bracket else 'time exit'}: trades {len(p)} "
                      f"({len(p) / days:.1f}/day), accuracy {(dec > 0).mean():.1%}, mean gross {p.mean():+.2f} bps (t {t:+.2f}), "
                      f"net at 4 bps {(p - 4).sum():+.0f} bps, avg win {p[p > 0].mean():.1f}, avg loss {-p[p < 0].mean():.1f}")


def vrp(ds, csv, start_ms, end_ms):
    url = ("https://www.deribit.com/api/v2/public/get_volatility_index_data?currency=BTC"
           f"&start_timestamp={start_ms}&end_timestamp={end_ms}&resolution=3600")
    data = json.load(urllib.request.urlopen(url, timeout=30))["result"]["data"]
    dv = pd.Series([x[4] for x in data], index=pd.to_datetime([x[0] for x in data], unit="ms", utc=True))
    df = pd.read_csv(csv, parse_dates=["datetime"]).set_index("datetime")
    r = np.log(df["close"]).diff().dropna()
    for hz, mins in (("24h", 1440), ("7d", 7 * 1440)):
        rv = np.sqrt((r ** 2).rolling(mins).sum().shift(-mins) * 525600 / mins) * 100
        j = pd.concat([dv.rename("iv"), rv.rename("rv")], axis=1, join="inner").dropna()
        gap = j["iv"] - j["rv"]
        print(f"  {ds} {hz}: hours {len(j)}, implied {j['iv'].mean():.1f}, forward realised {j['rv'].mean():.1f}, "
              f"gap {gap.mean():+.1f} vol points, implied > realised in {(gap > 0).mean():.0%} of hours")


if __name__ == "__main__":
    for ds in ("repo", "post"):
        X = train.Ctx(ds)
        print(f"\n=== {ds}: 1. volatility regimes (deciles fixed on pre-Test data)")
        regimes(X)
        print(f"=== {ds}: 2. volatility-gated reversal (threshold fixed on pre-Test data)")
        gated_fade(X)
    print("\n=== 3. implied vs forward realised volatility (Deribit DVOL)")
    work = os.environ.get("TSFM_EVAL_WORK", os.path.join(HERE, "..", "tsfm_eval", "work"))
    vrp("repo", os.path.join(HERE, "..", "..", "binance_btcusdt_1min_ccxt.csv"), 1760140800000, 1762732800000)
    vrp("post", os.path.join(work, "data", "btcusdt_1m_2026-09-20_2026-10-30.csv"), 1789862400000, 1793318400000)
