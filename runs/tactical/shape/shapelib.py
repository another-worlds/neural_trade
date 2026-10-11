"""Shape study (SPEC.md): data loader with the hard cutoff, path-shape targets, causal features, the generic dataset builder
and the monthly walk-forward worker. Everything is a pure function of the OHLCV arrays, so the same code runs on the real file
and on the simulated paths of mc.py.

Anchor i = a decision at the close of bar i-1; the target window is the T bars i..i+T-1; per-bar statistics are read at bar i-1."""
import os, sys, math, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); TAC = os.path.dirname(HERE)
CSV = os.environ.get("SHAPE_CSV", "D:/nt/neural_trade/Bitcoin_BTCUSDT.csv")
CUTOFF = pd.Timestamp("2025-07-20 23:59")            # last bar read, inclusive (owner rule after H66)
CACHE = os.environ.get("SHAPE_CACHE", "D:/nt/nt_shape_cache")
STRIDE = 10; WARM = 1500; HDIR = 15; DB = 5e-4
TS = (30, 60, 120)
TARGETS = ("rho", "vr", "z2", "z3")


# ------------------------------------------------------------------ data with the hard cutoff
def load_bars(path=None, cutoff=None, chunk=500_000):
    """OHLCV frame indexed by time, sorted, with NO row after `cutoff`. The file is read in chunks and reading stops at the first
    chunk that crosses the cutoff (the rows past it are dropped at once and never reach any array)."""
    path = path or CSV; cutoff = CUTOFF if cutoff is None else pd.Timestamp(cutoff); parts = []
    for ch in pd.read_csv(path, usecols=["timestamp", "open", "high", "low", "close", "volume"], chunksize=chunk):
        ts = pd.to_datetime(ch["timestamp"]); keep = (ts <= cutoff).to_numpy()
        if keep.any():
            c = ch[keep].copy(); c.index = ts[keep]; parts.append(c.drop(columns="timestamp"))
        if not keep.all():
            break
    s = pd.concat(parts).sort_index()
    assert s.index.max() <= cutoff
    return s


# ------------------------------------------------------------------ small helpers
def cs0(x):
    """cumulative sum with a leading 0: cs0(x)[k] = sum x[:k]."""
    out = np.empty(len(x) + 1); out[0] = 0.0; np.cumsum(x, out=out[1:]); return out


def roll_trailing(x, k):
    """S[j] = sum x[j-k+1..j] (NaN while j < k-1)."""
    c = cs0(x); out = np.full(len(x), np.nan); out[k - 1:] = c[k:] - c[:len(x) - k + 1]; return out


def roll_ext(x, k, fn):
    """trailing rolling max/min over k bars ending at j (NaN for j < k-1)."""
    s = pd.Series(x).rolling(k)
    return (s.max() if fn == "max" else s.min()).to_numpy()


def logs(A):
    return np.log(np.maximum(A[:, :4], 1e-9))


# ------------------------------------------------------------------ targets (the future window i..i+T-1)
def targets_at(A, anch, T):
    """rho, vr, z2, z3 for the windows of T bars starting at each anchor i (bars i..i+T-1). NaN where the window does not fit or is flat.
    rho = |lnC_end - lnC_{i-1}| / (ln max(C_{i-1}, maxH) - ln min(C_{i-1}, minL)); logs make every target reflection invariant."""
    n = len(A); L = logs(A); lo, lh, ll, lc = L[:, 0], L[:, 1], L[:, 2], L[:, 3]
    r = np.concatenate([[0.0], np.diff(lc)])
    c1 = cs0(r * r); p1 = np.concatenate([[0.0], r[1:] * r[:-1]]); c_p1 = cs0(p1)
    p2 = np.zeros(n); p2[2:] = r[2:] * r[:-2]; c_p2 = cs0(p2)
    ok = anch + T - 1 < n
    a = np.where(ok, anch, 1); e = a + T                     # window [a, e)
    sr = lc[e - 1] - lc[a - 1]; q = c1[e] - c1[a]
    lag1 = c_p1[e] - c_p1[a + 1]; lag2 = c_p2[e] - c_p2[a + 2]
    allp = (sr * sr - q) / 2.0
    hmax = pd.Series(lh).rolling(T).max().to_numpy()[e - 1]; lmin = pd.Series(ll).rolling(T).min().to_numpy()[e - 1]
    top = np.maximum(hmax, lc[a - 1]); bot = np.minimum(lmin, lc[a - 1]); rng = top - bot
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = np.where(rng > 1e-12, np.abs(sr) / rng, np.nan)
        flat = q > 1e-18
        vr = np.where(flat, sr * sr / q, np.nan)
        z2 = np.where(flat, (allp - lag1) / q, np.nan)
        z3 = np.where(flat, (allp - lag1 - lag2) / q, np.nan)
    out = {"rho": rho, "vr": vr, "z2": z2, "z3": z3}
    for k in out:
        out[k] = np.where(ok, out[k], np.nan)
    return out


def direction_return(A, anch, H=HDIR):
    """next-open label return: close of bar i+H-1 over the OPEN of bar i."""
    n = len(A); ok = anch + H - 1 < n; a = np.where(ok, anch, 0)
    r = A[np.minimum(a + H - 1, n - 1), 3] / A[a, 0] - 1
    return np.where(ok, r, np.nan)


# ------------------------------------------------------------------ features (read at bar b = i-1)
def clock_features(tmin):
    """Fourier clock of the anchor's open time; tmin = minutes since the epoch (int)."""
    tod = (tmin % 1440).astype(float); mih = (tmin % 60).astype(float); dow = ((tmin // 1440 + 3) % 7).astype(float)
    cols = []
    for k in (1, 2, 3):
        cols += [np.sin(2 * np.pi * k * tod / 1440), np.cos(2 * np.pi * k * tod / 1440)]
    for per in (60, 30, 20, 15):
        cols += [np.sin(2 * np.pi * mih / per), np.cos(2 * np.pi * mih / per)]
    for k in (1, 2):
        cols += [np.sin(2 * np.pi * k * dow / 7), np.cos(2 * np.pi * k * dow / 7)]
    return np.stack(cols, 1)


def per_bar_features(A):
    """dict of per-bar arrays; element j uses only bars <= j."""
    n = len(A); L = logs(A); lo, lh, ll, lc = L[:, 0], L[:, 1], L[:, 2], L[:, 3]; V = A[:, 4]
    r = np.concatenate([[0.0], np.diff(lc)]); r2 = r * r; ar = np.abs(r)
    f = {}
    rv = {k: roll_trailing(r2, k) for k in (15, 60, 240, 1440)}
    for k, v in rv.items():
        f[f"lrv{k}"] = np.log(np.sqrt(np.maximum(v, 0) / k) + 1e-7)
    f["vsurp1"] = f["lrv15"] - f["lrv240"]; f["vsurp2"] = f["lrv60"] - f["lrv1440"]
    sig60 = np.sqrt(np.maximum(rv[60], 0) / 60)
    bp = np.zeros(n); bp[1:] = ar[1:] * ar[:-1]
    f["bpv"] = (math.pi / 2) * roll_trailing(bp, 60) / (rv[60] + 1e-12)
    sig_prev = np.full(n, np.nan); sig_prev[1:] = np.sqrt(np.maximum(rv[1440][:-1], 0) / 1440)
    ev = np.nan_to_num(ar >= 4 * sig_prev, nan=False) & (sig_prev > 0)
    last = np.maximum.accumulate(np.where(ev, np.arange(n), -1))
    tsl = np.where(last < 0, 1440, np.minimum(np.arange(n) - last, 1440)); f["tsl"] = np.log1p(tsl.astype(float))
    gap = np.zeros(n); gap[1:] = np.abs(lo[1:] - lc[:-1])
    f["bounce"] = roll_trailing(gap, 60) / 60 / (sig60 + 1e-7)
    v60 = roll_trailing(V, 60) / 60; v1440 = roll_trailing(V, 1440) / 1440
    f["volsurp"] = np.log((v60 + 1e-3) / (v1440 + 1e-3))
    ret60 = np.full(n, np.nan); ret60[60:] = lc[60:] - lc[:-60]
    f["cgw"] = f["volsurp"] * np.abs(ret60) / (sig60 * math.sqrt(60) + 1e-7)
    rsp = roll_trailing(r2 * (r > 0), 60); rsn = roll_trailing(r2 * (r < 0), 60)
    f["rsp"] = np.log((rsp + 1e-12) / (rv[60] + 1e-12)); f["rsn"] = np.log((rsn + 1e-12) / (rv[60] + 1e-12))
    # backward rho
    for k in (30, 60, 240):
        hm = roll_ext(lh, k, "max"); lm = roll_ext(ll, k, "min")
        c0 = np.full(n, np.nan); c0[k:] = lc[:-k]
        top = np.maximum(hm, c0); bot = np.minimum(lm, c0); num = np.abs(lc - c0)
        with np.errstate(invalid="ignore", divide="ignore"):
            f[f"prho{k}"] = np.where(top - bot > 1e-12, num / (top - bot), np.nan)
    # variance ratios over the previous 240 bars
    for q in (5, 15, 60):
        s = np.full(n, np.nan); s[q:] = lc[q:] - lc[:-q]
        f[f"pvr{q}"] = roll_trailing(np.nan_to_num(s) ** 2, 240) / (q * roll_trailing(r2, 240) + 1e-18)
    for k in (1, 2, 3, 4, 5):
        pk = np.zeros(n); pk[k:] = r[k:] * r[:-k]
        f[f"ac{k}"] = roll_trailing(pk, 240) / (roll_trailing(r2, 240) + 1e-18)
    return f


BASE_COLS = ["lrv15", "lrv60", "lrv240", "lrv1440", "vsurp1", "vsurp2", "bpv", "tsl", "bounce", "volsurp", "cgw"]
CHAL_COLS = ["prho30", "prho60", "prho240", "pvr5", "pvr15", "pvr60", "ac1", "ac2", "ac3", "ac4", "ac5", "rsp", "rsn"]


def tb7_block(A, anch, workers=1):
    """lab.fs_tb7 on the windows A[i-60:i] (13 features); lab chdirs on import, so restore the cwd."""
    cwd = os.getcwd(); sys.path.insert(0, os.path.join(TAC, "lab"))
    try:
        import lab
    finally:
        os.chdir(cwd)
    out = np.empty((len(anch), 13), np.float32)
    step = 5000
    for s in range(0, len(anch), step):
        a = anch[s:s + step]; W = A[a[:, None] + np.arange(-60, 0)].astype(np.float32)
        out[s:s + step] = np.nan_to_num(np.asarray(lab.fs_tb7(W), np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return out


def _tb7_chunk(args):
    A, anch = args
    try:
        import psutil; psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass
    return tb7_block(A, anch)


def build_dataset(df, workers=1, last_anchor_slack=0):
    """df: OHLCV frame (time index, 1-minute). Returns a dict of arrays on the anchors (stride STRIDE)."""
    A = df[["open", "high", "low", "close", "volume"]].to_numpy(np.float64); n = len(A)
    tmin = (df.index.values.astype("datetime64[m]").astype(np.int64))
    anch = np.arange(WARM, n - HDIR, STRIDE)
    t = df.index.values.astype("datetime64[ns]")[anch - 1].astype(np.int64)       # anchor time = open of the last known bar
    pb = per_bar_features(A); b = anch - 1
    Xb = np.concatenate([clock_features(tmin[anch]), np.stack([pb[c][b] for c in BASE_COLS], 1)], 1)
    Xc = np.stack([pb[c][b] for c in CHAL_COLS], 1)
    tg = {f"{k}_{T}": v for T in TS for k, v in targets_at(A, anch, T).items()}
    if workers > 1:
        import multiprocessing as mp
        parts = np.array_split(anch, workers * 4)
        with mp.Pool(workers) as pool:
            tb7 = np.concatenate(pool.map(_tb7_chunk, [(A, p) for p in parts]))
    else:
        tb7 = tb7_block(A, anch)
    return {"t": t, "anch": anch, "Xb": Xb.astype(np.float32), "Xc": Xc.astype(np.float32), "tb7": tb7,
            "rdir": direction_return(A, anch), **{k: v for k, v in tg.items()}}


def save_dataset(ds, tag):
    os.makedirs(CACHE, exist_ok=True); np.savez(f"{CACHE}/{tag}.npz", **ds)


def load_dataset(tag):
    z = np.load(f"{CACHE}/{tag}.npz"); return {k: z[k] for k in z.files}


# ------------------------------------------------------------------ the walk-forward worker
def split_masks(t, m0, m1, T, train_days=365):
    """train: the anchor's target window CLOSES by the month start (anchor time + (T+1) minutes <= m0), at most train_days old;
    test: anchors in [m0, m1). t = anchor times (ns ints)."""
    t = pd.DatetimeIndex(np.asarray(t, np.int64).astype("datetime64[ns]")); lab_end = t + pd.Timedelta(minutes=T + 1)
    tr = (t >= m0 - pd.Timedelta(days=train_days)) & (lab_end <= m0); te = (t >= m0) & (t < m1)
    return np.asarray(tr), np.asarray(te)


def _std(Xtr, X):
    Xtr = np.nan_to_num(Xtr, nan=0.0, posinf=0.0, neginf=0.0); mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-9
    return np.clip((np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0) - mu) / sd, -8, 8)


def rank_gauss(y):
    from scipy.stats import norm, rankdata
    return norm.ppf((rankdata(y) - 0.5) / len(y))


def abs_logit(ds, tr, te, gap=40):
    """|logit| of the tb7 direction model: cross-fitted (2 contiguous halves, gap) on the training rows, full fit on the test rows.
    Also returns the full model's AUC on the test rows (dead-band mask)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    X = ds["tb7"]; r = ds["rdir"]; y = (r > 0).astype(int); m = np.abs(r) > DB
    itr = np.flatnonzero(tr); ite = np.flatnonzero(te)
    def fit(idx):
        idx = idx[m[idx]]
        mu = X[idx].mean(0); sd = X[idx].std(0) + 1e-9
        mdl = LogisticRegression(C=0.1, max_iter=500).fit(np.clip((X[idx] - mu) / sd, -8, 8), y[idx])
        return lambda J: mdl.decision_function(np.clip((X[J] - mu) / sd, -8, 8))
    h = len(itr) // 2; A, B = itr[:h], itr[h:]
    out_tr = np.empty(len(itr)); out_tr[:h] = fit(B[gap:])(A); out_tr[h:] = fit(A[:-gap])(B)
    full = fit(itr); out_te = full(ite)
    mt = m[ite]
    auc = float(roc_auc_score(y[ite][mt], out_te[mt])) if mt.sum() > 100 and len(np.unique(y[ite][mt])) > 1 else float("nan")
    return np.abs(out_tr), np.abs(out_te), auc


def fit_predict(model, Xtr, ytr, Xte):
    if model == "ridge":
        from sklearn.linear_model import Ridge
        return Ridge(alpha=100.0).fit(Xtr, ytr).predict(Xte)
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_iter=100, learning_rate=0.08, max_depth=4, min_samples_leaf=200,
                                         l2_regularization=1.0, random_state=0).fit(Xtr, ytr).predict(Xte)


MODELS = ("ridge", "hgb")
MIN_TRAIN, MIN_TEST = 5000, 500


def score_month(ds, month, models=MODELS, train_days=365, only=None):
    """one scored month -> {'month', 'auc_tb7', 'cells': [{target, T, model, n, sp_b, sp_c, mse_b, mse_c}]}"""
    from scipy.stats import spearmanr
    m = pd.Period(month, "M"); m0, m1 = m.start_time, (m + 1).start_time; cells = []
    tr60, te = split_masks(ds["t"], m0, m1, 60, train_days)
    if tr60.sum() < MIN_TRAIN or te.sum() < MIN_TEST:
        return None
    ltr, lte, auc = abs_logit(ds, tr60, te)
    for T in TS:
        tr, te = split_masks(ds["t"], m0, m1, T, train_days)
        itr, ite = np.flatnonzero(tr), np.flatnonzero(te)
        # |logit| was cross-fitted on the T=60 training rows; map it to this T's rows (same anchors, a superset/subset by a few rows)
        pos = {int(j): k for k, j in enumerate(np.flatnonzero(tr60))}
        keep = np.array([int(j) in pos for j in itr]); itr = itr[keep]
        lg_tr = ltr[[pos[int(j)] for j in itr]]; lg_te = lte
        Xb_tr, Xb_te = ds["Xb"][itr], ds["Xb"][ite]
        Xc_tr = np.concatenate([Xb_tr, ds["Xc"][itr], lg_tr[:, None].astype(np.float32)], 1)
        Xc_te = np.concatenate([Xb_te, ds["Xc"][ite], lg_te[:, None].astype(np.float32)], 1)
        for tgt in TARGETS:
            if only is not None and not any((tgt, md) in only for md in models):
                continue
            y = ds[f"{tgt}_{T}"]; ytr, yte = y[itr], y[ite]
            vtr, vte = np.isfinite(ytr), np.isfinite(yte)
            if vtr.sum() < MIN_TRAIN or vte.sum() < MIN_TEST:
                continue
            yfit = ytr[vtr] if tgt == "rho" else rank_gauss(ytr[vtr])
            sb_tr, sb_te = _std(Xb_tr[vtr], Xb_tr[vtr]), _std(Xb_tr[vtr], Xb_te[vte])
            sc_tr, sc_te = _std(Xc_tr[vtr], Xc_tr[vtr]), _std(Xc_tr[vtr], Xc_te[vte])
            for model in models:
                if only is not None and (tgt, model) not in only:
                    continue
                pb = fit_predict(model, sb_tr, yfit, sb_te); pc = fit_predict(model, sc_tr, yfit, sc_te)
                rec = {"target": tgt, "T": T, "model": model, "n": int(vte.sum()),
                       "sp_b": float(spearmanr(pb, yte[vte])[0]), "sp_c": float(spearmanr(pc, yte[vte])[0])}
                if tgt == "rho":
                    rec["mse_b"] = float(np.mean((pb - yte[vte]) ** 2)); rec["mse_c"] = float(np.mean((pc - yte[vte]) ** 2))
                cells.append(rec)
    return {"month": str(m), "auc_tb7": auc, "cells": cells}
