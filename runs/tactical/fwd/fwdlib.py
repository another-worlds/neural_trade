"""Forward-candidate study: data truncation, feature cache and the monthly walk-forward with trade selection.
NO bar on or after CUTOFF is ever loaded (load_bars cuts while reading the file, chunk by chunk)."""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); TAC = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(TAC, "recent")); sys.path.insert(0, os.path.join(TAC, "equity"))
CSV = os.environ.get("NT_FWD_CSV", "D:/nt/neural_trade/Bitcoin_BTCUSDT.csv")
CUTOFF = pd.Timestamp("2024-07-01")
CACHE = "D:/nt/nt_fwd_cache"
L = 60; BAR_MIN = 60; HMAX = 4
FIRST_MONTH, LAST_MONTH = "2019-01", "2024-06"
COST = 2.0


def load_bars(csv=None, cutoff=CUTOFF):
    """1-hour bars from the 1-minute file, reading chunks and stopping at the first timestamp >= cutoff."""
    parts = []
    for ch in pd.read_csv(csv or CSV, usecols=["timestamp", "open", "high", "low", "close", "volume"], chunksize=200_000):
        ts = pd.to_datetime(ch["timestamp"])
        keep = ts < cutoff
        if keep.any():
            c = ch[keep].copy(); c.index = ts[keep]; parts.append(c.drop(columns="timestamp"))
        if not keep.all():
            break
    s = pd.concat(parts).sort_index()
    assert s.index.max() < cutoff
    return s.resample("60min").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()


def build_cache(csv=None, cache=CACHE):
    import feats as F0                # noqa  (sets the sys.path for lab)
    import lab
    s = load_bars(csv); A = s.values.astype(np.float64); n = len(A)
    anch = np.arange(L, n - (HMAX - 1)); t = s.index[anch - 1].values.astype("datetime64[ns]").astype(np.int64)
    W = A[anch[:, None] + np.arange(-L, 0)].astype(np.float32)
    nz = lambda x: np.nan_to_num(np.asarray(x, np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    os.makedirs(cache, exist_ok=True)
    np.save(f"{cache}/tb7.npy", nz(lab.fs_tb7(W))); np.save(f"{cache}/rich.npy", nz(lab.fs_rich(W)))
    np.save(f"{cache}/t.npy", t); np.save(f"{cache}/anch.npy", anch)
    np.save(f"{cache}/open.npy", A[:, 0]); np.save(f"{cache}/close.npy", A[:, 3])
    return len(anch)


def next_open_ret(op, cl, anch, H):
    """H-bar return: entry = open of bar anch (the bar after the signal bar), exit = close of bar anch+H-1."""
    return cl[anch + H - 1] / op[anch] - 1


def deadband(H):
    return 5e-4 * np.sqrt(H * BAR_MIN / 15.0)


def month_predictions(X, t, r, kind, T, H, first=FIRST_MONTH, last=LAST_MONTH):
    """Yield (month, test_index_array, p, r_test) for each feasible month (wf.split_masks / wf._fit / wf.CLIP)."""
    import wf
    from sklearn.preprocessing import StandardScaler
    ti = pd.DatetimeIndex(t.astype("datetime64[ns]")); mask = np.abs(r) > deadband(H); y = (r > 0).astype(int)
    for m in pd.period_range(first, last, freq="M"):
        m0, m1 = m.start_time, (m + 1).start_time
        tr, te = wf.split_masks(ti, m0, m1, H, BAR_MIN, T); trm = tr & mask; tem = te & mask
        if trm.sum() < wf.MIN_TRAIN or tem.sum() < wf.MIN_TEST or len(np.unique(y[trm])) < 2 or len(np.unique(y[tem])) < 2:
            continue
        Xtr = np.asarray(X[trm], np.float32); sc = StandardScaler().fit(Xtr)
        g = lambda Z: np.clip(sc.transform(np.asarray(Z, np.float32)), -wf.CLIP, wf.CLIP)
        mdl, _ = wf._fit(kind, g(Xtr), y[trm], gap_rows=H + 1)
        idx = np.flatnonzero(te); p = mdl.predict_proba(g(X[idx]))[:, 1]
        yield str(m), idx, p, r[idx]


def trades_from(preds, t, H, frac):
    """preds: list of (month, idx, p, r). Top `frac` most confident per month, then a non-overlapping sequence over all months.
    Returns arrays sorted by entry: entry_ns, exit_ns, side, ret(side-adjusted fraction), month_label."""
    import eqlib as E
    ent, ex, side, ret, mon = [], [], [], [], []
    bar_ns = BAR_MIN * 60 * 10 ** 9
    for m, idx, p, r in preds:
        conf = np.abs(p - 0.5); k = conf >= np.quantile(conf, 1 - frac)
        t0 = t[idx][k]
        ent.append(t0 + bar_ns); ex.append(t0 + (H + 1) * bar_ns)
        sd = np.where(p[k] > 0.5, 1, -1); side.append(sd); ret.append(sd * r[k]); mon += [m] * int(k.sum())
    ent, ex, side, ret = (np.concatenate(a) for a in (ent, ex, side, ret)); mon = np.array(mon)
    o = np.argsort(ent, kind="stable"); ent, ex, side, ret, mon = ent[o], ex[o], side[o], ret[o], mon[o]
    keep, _ = E.non_overlapping(ent, ex)
    return ent[keep], ex[keep], side[keep], ret[keep], mon[keep]


def month_table(ret, mon, cost=COST):
    """monthly number: sum of net bps of the month's trades (entry month); also trades and hit."""
    net = ret * 1e4 - cost
    df = pd.DataFrame({"m": mon, "net": net, "win": net > 0}).groupby("m").agg(sum_bps=("net", "sum"), mean_bps=("net", "mean"), hit=("win", "mean"), n=("net", "size"))
    return df


def score_candidate(ent, ex, ret, mon, start_ns):
    import eqlib as E
    mt = month_table(ret, mon)
    st = E.equity_stats(ret, ent, ex, start_ns, COST)
    return {"score_mean_monthly_net_bps": float(mt["sum_bps"].mean()), "n_months": int(len(mt)), "n_trades": int(len(ret)),
            "max_dd_pct": st["max_drawdown_pct"], "hit": st["hit"], "mean_trade_net_bps": st["mean_bps"]}
