"""Geometry features in the regression lab (lead, 2026-10-10, after the owner's "что по геометрии"): does the geometry of
H24/H43 (slope, distance, crossing, squeeze) add anything to the regression's 13 features when computed on textbook
indicators? Same 24 slices, same evaluation as lab.py. Sets:
  geom      the geometry of 5 textbook channels (EMA20, RSI14, MACD hist, BB mid(SMA20), Stoch %K), per channel:
            slope over 3 and 10 bars / own 1-bar change sd, price distance to the channel / close sd (price-type channels),
            smooth crossing in the last 5 bars, squeeze = sd(last 10) / sd(window)
  tb7geom   tb7 + geom
usage: python geom_lab.py"""
import os, sys
import numpy as np
os.chdir(r"D:\nt\nt_tactical"); sys.path.insert(0, "runs/tactical/lab")
import lab  # noqa: E402
from lab import ema, parts  # noqa: E402


def channels(W):
    o, h, l, c, v, d, sd = parts(W)
    e20 = ema(c, 20)
    up, dn = np.clip(d, 0, None), np.clip(-d, 0, None)
    rsi = 100 - 100 / (1 + ema(up, 14) / (ema(dn, 14) + 1e-12)); rsi = np.concatenate([rsi[:, :1], rsi], 1)
    macd = ema(c, 12) - ema(c, 26); hist = macd - ema(macd, 9)
    sma20 = np.stack([np.convolve(row, np.ones(20) / 20, mode="full")[:c.shape[1]] for row in c])
    sma20[:, :19] = c[:, :19]
    lo = np.minimum.accumulate(l, axis=1); hi = np.maximum.accumulate(h, axis=1)
    lo14 = np.stack([np.min(l[:, max(0, t - 13):t + 1], 1) for t in range(c.shape[1])], 1)
    hi14 = np.stack([np.max(h[:, max(0, t - 13):t + 1], 1) for t in range(c.shape[1])], 1)
    stoch = (c - lo14) / (hi14 - lo14 + 1e-12)
    return c, sd, {"ema20": (e20, True), "rsi": (rsi, False), "hist": (hist, False), "sma20": (sma20, True), "stoch": (stoch, False)}


def fs_geom(W):
    c, sd, ch = channels(W.astype(np.float64)); f = []
    for name, (x, price_like) in ch.items():
        sx = np.diff(x, axis=1).std(1) + 1e-12
        for k in (3, 10):
            f.append((x[:, -1] - x[:, -1 - k]) / (k * sx))
        if price_like:
            dist = (c - x) / sd[:, None]
            f.append(np.arcsinh(dist[:, -1]))
            f.append((np.tanh(dist[:, -1]) - np.tanh(dist[:, -6])) / 2)
        f.append(x[:, -10:].std(1) / (x.std(1) + 1e-12))
    return np.stack(f, 1)


lab.SETS["geom"] = fs_geom
lab.SETS["tb7geom"] = lambda W: np.column_stack([lab.fs_tb7(W), fs_geom(W)])
for s in ("tb7", "geom", "tb7geom"):
    lab.run(s, "logreg", 0.1, tag=f"{s}_geomlab")
lab.run("tb7geom", "logreg", 0.001, tag="tb7geom_C0.001_geomlab")
