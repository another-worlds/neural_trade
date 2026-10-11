"""Order-flow features from OHLCV bars (SPEC.md): bulk volume classification, VPIN, A/D, volume-weighted return gap, Kyle lambda.
Everything is causal: the row of bar t is a function of bars <= t only. Anchors and next-open labels at the bottom."""
import os, sys
import numpy as np
import pandas as pd
from scipy.special import ndtr

CSV = "D:/nt/neural_trade/Bitcoin_BTCUSDT.csv"
CACHE = "D:/nt/nt_flow_cache"
L = 60
CELLS = {"1m": dict(bar_min=1, H=15, stride=10), "5m": dict(bar_min=5, H=3, stride=2)}
NAMES = ["imb5", "imb15", "imb60", "vpin", "ad15", "ad60", "vwd15", "vwd60", "kyle60"]
DB = 5e-4
N_BUCKETS = 50


def load_bars(bar):
    s = pd.read_csv(CSV, usecols=["timestamp", "open", "high", "low", "close", "volume"])
    s.index = pd.to_datetime(s["timestamp"]); s = s.drop(columns="timestamp").sort_index()
    m = CELLS[bar]["bar_min"]
    if m != 1:
        s = s.resample(f"{m}min").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    return s


def _rs(x, k):
    return pd.Series(x).rolling(k, min_periods=k).sum().to_numpy()


def flow_features(o, h, l, c, v, bar_min):
    """[N, 9] float64 (NaN in the warm-up), row t uses bars <= t only. o is unused (kept for the signature)."""
    n = len(c); sig_n = 240 // bar_min; day = 1440 // bar_min
    dP = np.concatenate([[0.0], np.diff(c)])
    sigma = pd.Series(dP).rolling(sig_n, min_periods=sig_n).std().to_numpy()
    z = np.where(sigma > 0, dP / np.where(sigma > 0, sigma, 1.0), 0.0)
    z = np.where(np.isnan(sigma), np.nan, z)
    sv = v * (2 * ndtr(z) - 1)
    out = {}
    for k in (5, 15, 60):
        sV = _rs(v, k)
        out[f"imb{k}"] = np.where(sV > 0, _rs(sv, k) / np.where(sV > 0, sV, 1), 0.0)
    # VPIN on a volume clock: bucket size = trailing 24 h volume ending at t-1, over 50
    dv = np.concatenate([[np.nan], _rs(v, day)[:-1]])
    vb = dv / N_BUCKETS
    ok = np.isfinite(vb) & (vb > 0) & np.isfinite(sv)
    u = np.where(ok, v / np.where(ok, vb, 1.0), 0.0)
    cum = np.cumsum(u); bid = np.floor(cum).astype(np.int64)
    nb = int(bid[-1]) + 1
    S = np.bincount(bid[ok], weights=sv[ok], minlength=nb); V = np.bincount(bid[ok], weights=v[ok], minlength=nb)
    aS = np.abs(S); ca = np.concatenate([[0.0], np.cumsum(aS)]); cv = np.concatenate([[0.0], np.cumsum(V)])
    m = bid - 1                                                    # last completed bucket at bar t
    first_ok = np.argmax(ok) if ok.any() else n
    valid = (m >= N_BUCKETS - 1) & (np.arange(n) >= first_ok)
    mm = np.clip(m, 0, nb - 1); lo = np.clip(mm - N_BUCKETS + 1, 0, nb)
    num = ca[mm + 1] - ca[lo]; den = cv[mm + 1] - cv[lo]
    out["vpin"] = np.where(valid & (den > 0), num / np.where(den > 0, den, 1), np.nan)
    rng = h - l
    clv = np.where(rng > 0, ((c - l) - (h - c)) / np.where(rng > 0, rng, 1), 0.0)
    for k in (15, 60):
        sV = _rs(v, k)
        out[f"ad{k}"] = np.where(sV > 0, _rs(clv * v, k) / np.where(sV > 0, sV, 1), 0.0)
    lr = np.concatenate([[np.nan], np.diff(np.log(np.maximum(c, 1e-9)))])
    for k in (15, 60):
        sV = _rs(v, k)
        vw = np.where(sV > 0, _rs(v * lr, k) / np.where(sV > 0, sV, 1), 0.0)
        out[f"vwd{k}"] = k * vw - _rs(lr, k)
    mv = pd.Series(v).rolling(day, min_periods=day).mean().to_numpy()
    svn = sv / np.where(mv > 0, mv, np.nan)
    a = pd.Series(lr); b = pd.Series(svn)
    var = b.rolling(60, min_periods=60).var().to_numpy(); cov = a.rolling(60, min_periods=60).cov(b).to_numpy()
    out["kyle60"] = np.where(var > 0, cov / np.where(var > 0, var, 1), 0.0)
    out["kyle60"] = np.where(np.isnan(var) | np.isnan(cov), np.nan, out["kyle60"])
    return np.stack([out[k] for k in NAMES], 1)


def anchors(n, H, stride):
    """anchor a = window A[a-L:a]; entry open[a], exit close[a+H-1]."""
    a = np.arange(L, n, stride)
    return a[a + H - 1 < n]


def next_open_return(A, a, H):
    return A[a + H - 1, 3] / A[a, 0] - 1


def build_cell(bar):
    """writes CACHE/<bar>_{tb7,flow,t,r}.npy"""
    here = os.path.dirname(os.path.abspath(__file__)); root = os.path.dirname(os.path.dirname(os.path.dirname(here)))
    sys.path.insert(0, os.path.join(root, "runs", "tactical", "lab"))
    import lab
    cfg = CELLS[bar]; s = load_bars(bar); A = s.values.astype(np.float64); n = len(A)
    F = flow_features(A[:, 0], A[:, 1], A[:, 2], A[:, 3], A[:, 4], cfg["bar_min"])
    a = anchors(n, cfg["H"], cfg["stride"])
    Xf = F[a - 1]
    good = np.isfinite(Xf).all(1); a, Xf = a[good], Xf[good]
    t = s.index[a - 1].values.astype("datetime64[ns]").astype(np.int64)
    r = next_open_return(A, a, cfg["H"])
    tb = []
    for i in range(0, len(a), 40000):
        W = A[a[i:i + 40000, None] + np.arange(-L, 0)].astype(np.float32)
        tb.append(np.nan_to_num(np.asarray(lab.fs_tb7(W), np.float32), nan=0.0, posinf=0.0, neginf=0.0))
    os.makedirs(CACHE, exist_ok=True)
    np.save(f"{CACHE}/{bar}_tb7.npy", np.concatenate(tb)); np.save(f"{CACHE}/{bar}_flow.npy", Xf.astype(np.float32))
    np.save(f"{CACHE}/{bar}_t.npy", t); np.save(f"{CACHE}/{bar}_r.npy", r)
    print(bar, len(a), "anchors", pd.Timestamp(t[0]), pd.Timestamp(t[-1]), flush=True)


if __name__ == "__main__":
    try:
        import psutil; psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass
    if sys.argv[1] == "build":
        build_cell(sys.argv[2])
