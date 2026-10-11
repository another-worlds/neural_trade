"""Feature blocks for the 'recent' study (SPEC.md): tb7, rich, ctx, gR, novel, har for every anchor of a bar cell. All causal.
Window blocks (tb7, rich, gR, novel) read only the 60 bars A[i-60:i]; ctx and har are per-bar trailing statistics read at the
anchor's last bar i-1. usage: python feats.py build 1m|1h  -> D:/nt/nt_recent_cache/<bar>_<block>.npy, <bar>_t.npy, <bar>_r.npy"""
import sys as _sys; _sys.path.insert(0, "D:/nt/nt_recut/runs/tactical/common")  # noqa: E402
import cutoff as C  # noqa: E402  (H66: no bar after C.CUTOFF is read)
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); TAC = os.path.dirname(HERE)
for d in ("lab", "geom2", "novel", "newnet"):
    sys.path.insert(0, os.path.join(TAC, d))
import lab                                    # noqa: E402  (chdirs to the checkout root)
import geomlib as G                           # noqa: E402
import features as NF                         # noqa: E402
import data as ND                             # noqa: E402

CSV = C.CSV
CACHE = "D:/nt/nt_recut_cache/recent"
L = 60
BAR_MIN = {"1m": 1, "1h": 60}
STRIDE = {"1m": 10, "1h": 1}
HOR = {"1m": 15, "1h": 1}                     # label horizon in bars
START = pd.Timestamp("2019-12-01")            # first anchor kept (the 365-day window of 2021-01 starts in 2020-01)
BLOCKS = ("tb7", "rich", "ctx", "gR", "novel", "har")
HAR_SCALES = (15, 60, 240, 1440)


def deadband(bar):
    return 5e-4 * np.sqrt(HOR[bar] * BAR_MIN[bar] / 15.0)


def load_bars(bar):
    s = pd.read_csv(CSV, usecols=["timestamp", "open", "high", "low", "close", "volume"])
    s.index = pd.to_datetime(s["timestamp"]); s = s.drop(columns="timestamp").sort_index()
    if bar != "1m":
        s = s.resample(f"{BAR_MIN[bar]}min").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    return s


def har_features(A, tod, dow):
    """[T, 8] float64, causal: log realised vol over the last 15/60/240/1440 bars + sin/cos of hour and of day of week."""
    lc = np.log(np.maximum(A[:, 3], 1e-9)); lr = np.concatenate([[0.0], np.diff(lc)]); out = []
    for k in HAR_SCALES:
        s2, n = ND._trailing(lr * lr, k)
        out.append(np.log(np.sqrt(np.maximum(s2 / n, 0.0)) + 1e-7))
    ph = 2 * np.pi * tod / 1440.0; pw = 2 * np.pi * dow / 7.0
    return np.stack(out + [np.sin(ph), np.cos(ph), np.sin(pw), np.cos(pw)], 1)


def per_bar(A, tod, dow):
    """the per-bar blocks over the whole series (float64)."""
    return {"ctx": ND.context_features(A, tod, dow), "har": har_features(A, tod, dow)}


def window_blocks(W):
    """W [N, 60, 5] float32 -> the four window blocks (NaN-free float32)."""
    nz = lambda x: np.nan_to_num(np.asarray(x, np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    nov = NF.compute(W); novel = np.concatenate([nov[f] for f in NF.FAMILIES], 1)
    return {"tb7": nz(lab.fs_tb7(W)), "rich": nz(lab.fs_rich(W)), "gR": nz(G.geom_rich(W)), "novel": nz(novel)}


def blocks_at(A, tod, dow, anch, pb=None):
    """all six blocks for the anchors `anch` (anchor i = window A[i-60:i]) of the series A; pb = per_bar(...) if already known."""
    pb = pb or per_bar(A, tod, dow)
    W = A[anch[:, None] + np.arange(-L, 0)].astype(np.float32)
    out = window_blocks(W)
    for k in ("ctx", "har"):
        out[k] = np.nan_to_num(pb[k][anch - 1]).astype(np.float32)
    return out


def _chunk(args):
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass
    anch, = args
    return blocks_at(_G["A"], _G["tod"], _G["dow"], anch, _G["pb"])


_G = {}


def _init(A, tod, dow, pb):
    _G.update(A=A, tod=tod, dow=dow, pb=pb)


def build(bar, workers=6, chunk=4000):
    import multiprocessing as mp
    os.makedirs(CACHE, exist_ok=True)
    s = load_bars(bar); A = s.values.astype(np.float64); n = len(A); H = HOR[bar]
    tod = (s.index.hour * 60 + s.index.minute).to_numpy(np.float64); dow = s.index.dayofweek.to_numpy(np.float64)
    pb = per_bar(A, tod, dow)
    anch = np.arange(L, n - 1, STRIDE[bar]); anch = anch[anch + H - 1 < n]
    t = s.index[anch - 1].values.astype("datetime64[ns]").astype(np.int64)
    keep = t >= START.value; anch, t = anch[keep], t[keep]
    r = A[anch + H - 1, 3] / A[anch - 1, 3] - 1
    print(f"{bar}: {len(anch)} anchors from {pd.Timestamp(t[0])} to {pd.Timestamp(t[-1])}", flush=True)
    parts = [(anch[i:i + chunk],) for i in range(0, len(anch), chunk)]
    with mp.Pool(workers, initializer=_init, initargs=(A, tod, dow, pb)) as pool:
        res = []
        for j, o in enumerate(pool.imap(_chunk, parts)):
            res.append(o)
            if j % 10 == 0:
                print(f"  chunk {j}/{len(parts)}", flush=True)
    for b in BLOCKS:
        X = np.concatenate([o[b] for o in res]); np.save(f"{CACHE}/{bar}_{b}.npy", X); print(b, X.shape, flush=True)
    np.save(f"{CACHE}/{bar}_t.npy", t); np.save(f"{CACHE}/{bar}_r.npy", r)


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(sys.argv[2])
