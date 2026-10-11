"""Hourly long-context features (exploratory, dev only). All causal: the value for anchor i (the model sees bars <= i-1, predicts
close[i]/close[i-1]-1) reads only bars <= i-1. usage: python hfeats.py build -> D:/nt/nt_hlong_cache/h.npz"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); TAC = os.path.dirname(HERE)
CSV = "D:/nt/neural_trade/Bitcoin_BTCUSDT.csv"; CACHE = "D:/nt/nt_hlong_cache"
KS = (1, 4, 12, 24, 72, 168, 336); FIRST = 800          # first anchor index: every trailing statistic (720 bars) is defined
N_SEQ = 7; N_VEC = 20; CLIP = 8.0


def load_hourly():
    s = pd.read_csv(CSV, usecols=["timestamp", "open", "high", "low", "close", "volume"])
    s.index = pd.to_datetime(s["timestamp"]); s = s.drop(columns="timestamp").sort_index()
    return s.resample("60min").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()


def _roll_rms(x, k):
    c = np.concatenate([[0.0], np.cumsum(x)]); n = len(x); out = np.full(n, np.nan)
    out[k - 1:] = np.sqrt(np.maximum((c[k:] - c[:-k]) / k, 0.0)); return out


def per_bar(A, hour, dow):
    """A [n,5] (o,h,l,c,v) -> vec [n,20], seq [n,7]; row j uses bars <= j only (NaN where the history is short)."""
    n = len(A); o, h, l, c, v = (A[:, i].astype(np.float64) for i in range(5))
    lc = np.log(c); lr = np.concatenate([[0.0], np.diff(lc)]); lr2 = lr * lr
    s = {k: _roll_rms(lr2, k) for k in (6, 24, 168, 720)}; sg = np.maximum(s[168], 1e-5)
    cols = []
    for k in KS:
        r = np.full(n, np.nan); r[k:] = lc[k:] - lc[:-k]; cols.append(r / (sg * np.sqrt(k)))
    cols += [np.log(s[24] / sg + 1e-9), np.log(s[168] / np.maximum(s[720], 1e-5) + 1e-9), np.log(s[6] / np.maximum(s[24], 1e-5) + 1e-9)]
    for k in (24, 168, 336):
        hi = pd.Series(h).rolling(k).max().to_numpy(); lo = pd.Series(l).rolling(k).min().to_numpy()
        cols.append((c - lo) / np.maximum(hi - lo, 1e-9))
    vm = {k: pd.Series(v).rolling(k).mean().to_numpy() for k in (24, 168)}
    cols += [np.log((vm[24] + 1e-9) / (vm[168] + 1e-9)), np.log1p(v / (vm[168] + 1e-9))]
    cols.append((c - o) / np.maximum(h - l, 1e-9 * c))
    ph = 2 * np.pi * hour / 24.0; pw = 2 * np.pi * dow / 7.0
    cols += [np.sin(ph), np.cos(ph), np.sin(pw), np.cos(pw)]
    vec = np.stack(cols, 1)
    seq = np.stack([lr / sg, np.log(h / c) / sg, np.log(l / c) / sg, np.log(c / o) / sg, np.log1p(v / (vm[168] + 1e-9)),
                    np.sin(ph), np.cos(ph)], 1)
    seq[:, :4] = np.clip(seq[:, :4], -CLIP, CLIP)
    return vec, seq


def anchors(n):
    """anchor i in [FIRST, n-1]: window ends at bar i-1, label bar i."""
    return np.arange(FIRST, n)


def build_all():
    s = load_hourly(); A = s.values.astype(np.float64); n = len(A)
    hour = s.index.hour.to_numpy(np.float64); dow = s.index.dayofweek.to_numpy(np.float64)
    vec, seq = per_bar(A, hour, dow); anch = anchors(n)
    t = s.index[anch - 1].values.astype("datetime64[ns]").astype(np.int64)
    r = A[anch, 3] / A[anch - 1, 3] - 1
    for d in ("lab",):
        sys.path.insert(0, os.path.join(TAC, d))
    import lab
    W = A[anch[:, None] + np.arange(-60, 0)].astype(np.float32)
    tb7 = np.nan_to_num(np.asarray(lab.fs_tb7(W), np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return dict(t=t, r=r, anch=anch, vec=np.nan_to_num(vec[anch - 1]).astype(np.float32), seq=np.nan_to_num(seq).astype(np.float32),
                tb7=tb7)


if __name__ == "__main__":
    if sys.argv[1] == "build":
        os.makedirs(CACHE, exist_ok=True); d = build_all()
        np.savez(f"{CACHE}/h.npz", **d)
        print({k: v.shape for k, v in d.items()}, pd.Timestamp(d["t"][0]), pd.Timestamp(d["t"][-1]), flush=True)
