"""Volatility against standard forecasts (tactical, owner 2026-10-09: "Волатильность против простого эталона").
Question: do the learned volatility predictors (the new network's tower, the ridge magnitude model, an HGB on the hourly
context features) beat STANDARD volatility forecasts (trailing realised vol, EWMA, GARCH(1,1), HAR-RV), or only the naive 60-bar
trailing realised vol the network used as its baseline?

Fixed before any run (see also the task statement in runs/tactical/LOG.md H34-H36):
  target    realised |r| over the horizon (minute: 10/15/20 bars on the 24 lab slices' val blocks; hourly: 12 h, 8 walk-forward folds).
  metrics   per horizon Spearman(prediction, |r|) and QLIKE on r^2: mean(x - log x - 1), x = r^2 / s^2, s^2 the predicted variance of
            the horizon-return (per-bar variance x h for per-bar forecasts). Mean over horizons per unit (slice / fold). Floors:
            r^2 >= 1e-10 (a flat price gives r = 0) and s^2 >= 2% of the forecast's own train-span median (floor_s2: a flat
            tape gives zero trailing variance and one return after it would make QLIKE 4e8 in a train year).
            qlike_raw  s^2 as the forecast gives it. qlike_cal  s^2 times one scale c per horizon fitted on the TRAIN span
            (c = mean(x) there, the QLIKE-optimal scale): removes a pure level bias (1-minute microstructure, jumps) that
            a ranking-only comparison would not see. Spearman needs no scale. Both are reported; ranking (best baseline)
            uses qlike_cal.
  units     the paired per-unit differences with a 95% t-interval over units (24 slices / 8 folds).
  training  every fitted quantity (lambda, GARCH parameters, HAR / ridge / HGB coefficients, the scales c) uses only bars whose
            labels end before the val block (train span: 1 year ending 81 windows before the val block, as newnet/data.py;
            fewer for early slices). A forecast at bar t reads only bars <= t (trailing windows and filters are causal).
  HAR       log of the mean squared 1-bar return over the next h bars on log mean squared return over the last 15, 60, 240, 1440
            bars (hourly: 12, 24, 168, 720); OLS per horizon. HAR+tod adds the sin/cos of the time of day (first two harmonics)
            at the middle of the forecast interval (known in advance).
  "best simple baseline" = the best of forecasts 1-6 (minute) / the standard forecasts (hourly) by the mean of the metric over
            units. It is chosen on the same units it is judged on, which can only favour the baseline.
usage: python volbench.py minute | hourly | report"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
from scipy import stats
from scipy.optimize import minimize
from scipy.signal import lfilter

HERE = os.path.dirname(os.path.abspath(__file__))
TAC = os.path.dirname(HERE)
OUT = os.path.join(HERE, "results.jsonl")
CSV = "D:/nt/neural_trade/Bitcoin_BTCUSDT.csv"
LAB_CACHE = os.environ.get("LAB_CACHE", "D:/nt/nt_tactical/runs/tactical/lab/cache")
SERIES_CACHE = os.environ.get("SERIES_CACHE", "D:/nt/nt_tactical_newnet/runs/tactical/newnet/cache")
HZ = (10, 15, 20)
WIN, GAPB, SPAN = 60, 81, 525600
R2_FLOOR, S2_FLOOR = 1e-10, 1e-14
TRAILING = (15, 60, 240, 1440)
HAR_LAGS = (15, 60, 240, 1440)
EWMA_GRID = (0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999, 0.9995)
T975 = lambda n: float(stats.t.ppf(0.975, n - 1))  # noqa: E731


# ---------------------------------------------------------------- generic pieces
def spearman(a, b):
    return float(stats.spearmanr(a, b)[0])


def qlike(r2, s2):
    x = np.maximum(r2, R2_FLOOR) / np.maximum(s2, S2_FLOOR)
    return float(np.mean(x - np.log(x) - 1.0))


FLOOR_REL = 0.02


def floor_s2(s2, ref):
    """Variance forecasts floored at 2% of the median of the same forecast on the train span (per horizon column): a flat tape gives
    a zero trailing variance, and one return after it would otherwise dominate QLIKE (4e8 in a train year) and the scale c."""
    ref = np.atleast_2d(np.asarray(ref).T).T
    return np.maximum(s2, FLOOR_REL * np.median(ref, axis=0)[None, :] if s2.ndim == 2 else FLOOR_REL * np.median(ref))


def scale_c(r2, s2):
    """QLIKE-optimal multiplicative scale of s2 on a sample (mean of x)."""
    return float(np.mean(np.maximum(r2, R2_FLOOR) / np.maximum(s2, S2_FLOOR)))


def cum_sq(lr):
    return np.concatenate([[0.0], np.cumsum(lr * lr)])


def trailing_var(cs, k):
    """Per-bar variance (mean squared 1-bar return) over the k bars ending at each bar t, inclusive (fewer at the start)."""
    t = np.arange(len(cs) - 1)
    lo = np.maximum(t + 1 - k, 0)
    return (cs[t + 1] - cs[lo]) / (t + 1 - lo)


def future_var(cs, t, h):
    """Mean squared 1-bar return over bars t+1 .. t+h (the realised variance the HAR regression targets)."""
    return (cs[t + h + 1] - cs[t + 1]) / h


def ewma_var(lr, lam, v0):
    """RiskMetrics: v_t = lam v_{t-1} + (1-lam) r_t^2, the variance estimate AFTER seeing bar t (a forecast for t+1)."""
    return lfilter([1.0 - lam], [1.0, -lam], lr * lr, zi=[lam * v0])[0]


def garch_filter(z, omega, alpha, beta, u0):
    """u[t] = omega + alpha z_t^2 + beta u[t-1]: the one-step-ahead conditional variance given bars <= t."""
    return lfilter([alpha], [1.0, -beta], z * z + omega / alpha, zi=[beta * u0])[0]


def garch_fit(r):
    """Gaussian MLE of GARCH(1,1) with zero mean on the series r. Returns (omega, alpha, beta) in r's units."""
    s2 = float(np.mean(r * r)); z = r / math.sqrt(s2)

    def nll(p):
        om, al, be = p
        if al + be >= 0.99995:
            return 1e12
        u = garch_filter(z, om, al, be, 1.0)
        h = np.maximum(u[:-1], 1e-12)                     # the variance of z[1:] given bars before them
        return 0.5 * float(np.sum(np.log(h) + z[1:] ** 2 / h)) / len(z)

    best = None
    for st in ((0.02, 0.08, 0.90), (0.05, 0.15, 0.80), (0.005, 0.04, 0.95)):
        res = minimize(nll, st, method="L-BFGS-B", bounds=[(1e-7, 2.0), (1e-4, 0.8), (0.05, 0.9999)], options=dict(maxiter=200))
        if best is None or res.fun < best.fun:
            best = res
    om, al, be = best.x
    return om * s2, al, be


def garch_horizon_var(u_next, omega, alpha, beta, h):
    """Variance of the h-bar return at bar t: sum_{k=1..h} E[var_{t+k}], u_next = E[var_{t+1}]."""
    phi = alpha + beta; V = omega / (1.0 - phi)
    return V * h + (u_next - V) * (1.0 - phi ** h) / (1.0 - phi)


def har_design(logv, tod_mid=None):
    """Columns: constant, log trailing variances (list of arrays), optional time-of-day harmonics."""
    cols = [np.ones(len(logv[0]))] + list(logv)
    if tod_mid is not None:
        for k in (1, 2):
            cols += [np.sin(2 * np.pi * k * tod_mid / 1440.0), np.cos(2 * np.pi * k * tod_mid / 1440.0)]
    return np.stack(cols, 1)


def ols(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]


def t_ci(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) < 2:
        return [float("nan")] * 3
    m = v.mean(); hw = T975(len(v)) * v.std(ddof=1) / math.sqrt(len(v))
    return [float(m), float(m - hw), float(m + hw)]


class Unit:
    """Scores of several forecasts on one unit (slice / fold); `add` takes the horizon predictions."""
    def __init__(self, name, r_by_h, r2cal_by_h=None):
        self.name = name; self.r = r_by_h; self.rows = {}

    def add(self, fname, score_by_h, s2_by_h=None, s2_cal_by_h=None, c_by_h=None):
        """score_by_h: [n, H] any monotone score of |r| (Spearman); s2_by_h: [n, H] variance forecast of the h-return (or None)."""
        H = len(self.r)
        sp = [spearman(score_by_h[:, i], np.abs(self.r[i])) for i in range(H)]
        row = dict(unit=self.name, forecast=fname, spearman_h=sp, spearman=float(np.mean(sp)))
        if s2_by_h is not None:
            r2 = [x * x for x in self.r]
            s2_by_h = np.asarray(s2_by_h).reshape(len(self.r[0]), -1)
            row["qlike_raw"] = float(np.mean([qlike(r2[i], s2_by_h[:, i]) for i in range(H)]))
            row["qlike_cal"] = float(np.mean([qlike(r2[i], s2_by_h[:, i] * c_by_h[i]) for i in range(H)]))
            row["scale_c"] = [float(c) for c in c_by_h]
        self.rows[fname] = row
        return row


# ---------------------------------------------------------------- minute
def load_minute():
    import pandas as pd
    d = pd.read_csv(CSV, usecols=["timestamp", "close"])
    t = pd.to_datetime(d["timestamp"])
    close = d["close"].to_numpy(np.float64)
    tod = (t.dt.hour * 60 + t.dt.minute).to_numpy(np.float64)
    return close, tod


def locate_val(close, Wva):
    sys.path.insert(0, os.path.join(TAC, "newnet"))
    import data as dm
    return dm.locate_val(close, Wva)


def minute_forecasts(close, tod, lr, cs, vs, nva, s0, n, rtr, rva, vec=None):
    """All forecasts for one slice. t_tr / t_va are the last bars of the train / val windows. Returns {name: (score [n,3], s2 [n,3] or None)}
    for the val points and the same for the train points (needed for the scales c and the ridge's own fit)."""
    t_tr = s0 + WIN - 1 + np.arange(n); t_va = vs + WIN - 1 + np.arange(nva)
    h_arr = np.array(HZ, float)
    out_va, out_tr = {}, {}

    def put(name, fn):
        out_tr[name] = fn(t_tr); out_va[name] = fn(t_va)

    # (1)-(2) trailing realised vol: per-bar variance x h
    for k in TRAILING:
        v = trailing_var(cs, k)
        put(f"rv{k}", lambda t, v=v: (lambda s2: (s2, s2))(v[t][:, None] * h_arr[None, :]))
    # (3) EWMA, lambda by calibrated QLIKE on the train span
    seg = slice(s0, t_va[-1] + 1)
    v0 = float(np.mean(lr[s0:s0 + 1440] ** 2))
    best = None
    for lam in EWMA_GRID:
        e = ewma_var(lr[seg], lam, v0)
        s2 = e[t_tr - s0][:, None] * h_arr[None, :]
        q = np.mean([qlike_cal_only(rtr[:, i] ** 2, s2[:, i]) for i in range(3)])
        if best is None or q < best[0]:
            best = (q, lam)
    lam = best[1]; e = ewma_var(lr[seg], lam, v0)
    put("ewma", lambda t: (lambda s2: (s2, s2))(e[t - s0][:, None] * h_arr[None, :]))
    # (4) GARCH(1,1) fitted on the train bars only (1-bar returns s0+1 .. t_tr[-1])
    om, al, be = garch_fit(lr[s0 + 1:t_tr[-1] + 1])
    u = garch_filter(lr[seg], om, al, be, v0)
    put("garch", lambda t: (lambda s2: (s2, s2))(np.stack([garch_horizon_var(u[t - s0], om, al, be, h) for h in HZ], 1)))
    # (5)-(6) HAR-RV (+ time of day)
    logv = [np.log(trailing_var(cs, k) + 1e-12) for k in HAR_LAGS]
    sub = t_tr[::5]
    for name, use_tod in (("har", False), ("har_tod", True)):
        coefs = []
        for h in HZ:
            Xh = har_design([lv[sub] for lv in logv], (tod[sub] + h / 2.0) if use_tod else None)
            coefs.append(ols(Xh, np.log(future_var(cs, sub, h) + 1e-12)))

        def f(t, coefs=coefs, use_tod=use_tod):
            cols = []
            for hi, h in enumerate(HZ):
                Xh = har_design([lv[t] for lv in logv], (tod[t] + h / 2.0) if use_tod else None)
                cols.append(np.exp(Xh @ coefs[hi]) * h)
            s2 = np.stack(cols, 1)
            return s2, s2
        put(name, f)
    # (7) ridge magnitude model on the network's 60 per-bar features (rich + volatility-level context), log(|r| + 1e-5)
    if vec is not None:
        out_va["ridge"], out_tr["ridge"] = ridge_forecast(vec, t_tr, t_va, rtr, n)
    return out_va, out_tr, dict(ewma_lambda=lam, garch=[om, al, be])


def qlike_cal_only(r2, s2):
    x = np.maximum(r2, R2_FLOOR) / np.maximum(s2, S2_FLOOR)
    return float(np.log(np.mean(x)) - np.mean(np.log(x)))


def ridge_forecast(vec, t_tr, t_va, rtr, n):
    """Ridge on standardised features (stride so that at most ~150k rows), alpha on the last 15% of train (80 bars apart), as
    longtrain.mag_model. Score = predicted log|r|; variance forecast = exp(2 pred), its scale fitted on the held-out last 15%."""
    from sklearn.linear_model import Ridge
    k = max(1, -(-n // 150000)); idx = np.arange(0, n, k)
    X = np.asarray(vec[t_tr[idx]], np.float32); Xv = np.asarray(vec[t_va], np.float32)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    X = np.clip((X - mu) / sd, -8, 8); Xv = np.clip((Xv - mu) / sd, -8, 8)
    Y = np.log(np.abs(rtr[idx]) + 1e-5); m = len(idx); cut = int(round(0.85 * m)); gap = max(1, 80 // k)
    P_va = np.zeros((len(t_va), 3)); c_ok = np.zeros((m - cut, 3))
    for h in range(3):
        errs = []
        for a in (1.0, 100.0, 1e4):
            mdl = Ridge(alpha=a).fit(X[:cut - gap], Y[:cut - gap, h]); errs.append((float(np.mean((mdl.predict(X[cut:]) - Y[cut:, h]) ** 2)), a))
        a = min(errs)[1]
        c_ok[:, h] = Ridge(alpha=a).fit(X[:cut - gap], Y[:cut - gap, h]).predict(X[cut:])
        full = Ridge(alpha=a).fit(X, Y[:, h]); P_va[:, h] = full.predict(Xv)
    # the scale c comes from the held-out part: stored as the "train" s2 over those points only (see score_slice)
    return (P_va, np.exp(2 * P_va)), (None, idx[cut:], np.exp(2 * c_ok))


def score_slice(name, close, tod, lr, cs, series_vec=None):
    D = np.load(f"{LAB_CACHE}/{name}.npz")
    rva = D["rva"].astype(np.float64); nva = len(rva)
    vs = locate_val(close, D["Wva"])
    avail = vs - GAPB - (1440 - WIN + 1) + 1
    n = int(min(SPAN, avail)); s0 = vs - GAPB - (n - 1)
    t_tr = s0 + WIN - 1 + np.arange(n); t_va = vs + WIN - 1 + np.arange(nva)
    rtr = np.stack([close[t_tr + h] / close[t_tr] - 1.0 for h in HZ], 1)
    rv = np.stack([close[t_va + h] / close[t_va] - 1.0 for h in HZ], 1)
    assert np.allclose(rv, rva, rtol=1e-4, atol=2e-6), "val returns differ from the lab cache"
    assert t_tr[-1] + max(HZ) < vs, "train labels reach the val block"
    out_va, out_tr, info = minute_forecasts(close, tod, lr, cs, vs, nva, s0, n, rtr, rva, series_vec)
    U = Unit(name, [rv[:, i] for i in range(3)])
    for f, v in out_va.items():
        if f == "ridge":
            (P_va, s2_va), (c_ok, hold_idx, s2_ho) = v, out_tr["ridge"]
            s2_ho = floor_s2(s2_ho, s2_ho)
            c = [scale_c(rtr[hold_idx, i] ** 2, s2_ho[:, i]) for i in range(3)]
            U.add(f, P_va, floor_s2(s2_va, s2_ho), None, c)
        else:
            sc, s2 = v
            s2_tr = floor_s2(out_tr[f][1], out_tr[f][1]); s2 = floor_s2(s2, out_tr[f][1])
            c = [scale_c(rtr[:, i] ** 2, s2_tr[:, i]) for i in range(3)]
            U.add(f, sc, s2, None, c)
    rows = list(U.rows.values())
    for r in rows:
        r.update(setup="minute", n_val=int(nva), n_train=int(n), full_span=bool(n == SPAN), **info)
    return rows


# ---------------------------------------------------------------- hourly
def hourly_data():
    import pandas as pd
    s = pd.read_csv(CSV, usecols=["timestamp", "open", "high", "low", "close", "volume"])
    s.index = pd.to_datetime(s["timestamp"]); s = s.drop(columns="timestamp").sort_index()
    b = s.resample("1h").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    b.columns = list("ohlcv")
    return b


def hourly_features(b, anch):
    """The `ctx` feature set of hourly.py (fs_rich on the 60-bar window + hourly context at the last bar), imported from it."""
    cwd = os.getcwd()
    sys.path.insert(0, os.path.join(TAC, "hourly")); sys.path.insert(0, os.path.join(TAC, "lab"))
    try:
        from lab import fs_rich              # this worktree's lab.py first, so hourly.py's own import finds it in sys.modules
        import hourly as hp
    finally:
        os.chdir(cwd)
    A = b.values.astype(np.float64)
    W = np.stack([A[i - hp.L:i] for i in anch]).astype(np.float32)
    F = np.column_stack([fs_rich(W), hp.ctx_features(b)[anch - 1]])
    return np.nan_to_num(F).astype(np.float64), hp


HOURLY_T = 12
H_TRAILING = (12, 24, 168, 720)
H_HAR_LAGS = (12, 24, 168, 720)
H_EWMA_GRID = (0.8, 0.9, 0.94, 0.97, 0.98, 0.99, 0.995)


def hourly_fold(f, b, A, lr, cs, anch, tf, rT, F, tr, te, hp):
    from sklearn.ensemble import HistGradientBoostingRegressor
    T = HOURLY_T; tod_h = b.index.hour.values.astype(float) * 60.0
    ttr, tte = tf[tr], tf[te]; rtr, rte = rT[tr], rT[te]
    last = anch[tr[-1]] + T - 1                              # the last bar any train label reads
    assert last < tf[te[0]], "train labels reach the test block"
    U = Unit(f"fold{f}", [rte])
    v0 = float(np.mean(lr[1:721] ** 2))
    cand = {}
    for k in H_TRAILING:
        v = trailing_var(cs, k); cand[f"rv{k}"] = (v[ttr] * T, v[tte] * T)
    best = None
    for lam in H_EWMA_GRID:
        e = ewma_var(lr[:last + 1], lam, v0); q = qlike_cal_only(rtr ** 2, e[ttr] * T)
        if best is None or q < best[0]:
            best = (q, lam)
    lam = best[1]; e = ewma_var(lr[:tf[te[-1]] + 1], lam, v0); cand["ewma"] = (e[ttr] * T, e[tte] * T)
    om, al, be = garch_fit(lr[1:last + 1]); u = garch_filter(lr[:tf[te[-1]] + 1], om, al, be, v0)
    cand["garch"] = (garch_horizon_var(u[ttr], om, al, be, T), garch_horizon_var(u[tte], om, al, be, T))
    logv = [np.log(trailing_var(cs, k) + 1e-12) for k in H_HAR_LAGS]
    for name, use_tod in (("har", False), ("har_tod", True)):
        sub = ttr[::2]
        Xs = har_design([lv[sub] for lv in logv], (tod_h[sub] + T * 30.0) if use_tod else None)
        co = ols(Xs, np.log(future_var(cs, sub, T) + 1e-12))
        Xf = lambda t: har_design([lv[t] for lv in logv], (tod_h[t] + T * 30.0) if use_tod else None)  # noqa: E731
        cand[name] = (np.exp(Xf(ttr) @ co) * T, np.exp(Xf(tte) @ co) * T)
    for name, (s2tr, s2te) in cand.items():
        s2te = floor_s2(s2te[:, None], s2tr[:, None])[:, 0]; s2tr = floor_s2(s2tr[:, None], s2tr[:, None])[:, 0]
        c = scale_c(rtr ** 2, s2tr)
        U.add(name, s2te[:, None], s2te[:, None], None, [c])
    # learned: HGB on ctx features, log(|r| + 1e-5); scale from the held-out last 20% of the train block
    Y = np.log(np.abs(rT) + 1e-5); cut = int(round(0.8 * len(tr))); gap = 2 * T
    mk = lambda: HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=300, random_state=0)  # noqa: E731
    ho = mk().fit(F[tr[:cut - gap]], Y[tr[:cut - gap]]); s2ho = np.exp(2 * ho.predict(F[tr[cut:]]))[:, None]
    c = scale_c(rtr[cut:] ** 2, floor_s2(s2ho, s2ho)[:, 0])
    full = mk().fit(F[tr], Y[tr]); p = full.predict(F[te])
    U.add("hgb_ctx", p[:, None], floor_s2(np.exp(2 * p)[:, None], s2ho)[:, 0:1], None, [c])
    rows = list(U.rows.values())
    for r in rows:
        r.update(setup="hourly", n_val=int(len(te)), n_train=int(len(tr)), ewma_lambda=lam, garch=[om, al, be])
    return rows


def run_hourly():
    b = hourly_data(); A = b.values.astype(np.float64); L = 60; T = HOURLY_T
    anch = np.arange(L + 168, len(A) - T)
    import pandas as pd
    HOLD = pd.Timestamp("2024-07-01")
    t_exit = b.index[anch + T - 1]
    F, hp = hourly_features(b, anch)
    lc = np.log(A[:, 3]); lr = np.concatenate([[0.0], np.diff(lc)]); cs = cum_sq(lr)
    tf = anch - 1; rT = A[anch + T - 1, 3] / A[anch - 1, 3] - 1
    dev = np.arange(len(anch))[t_exit < HOLD]
    edges = np.linspace(len(dev) * 0.3, len(dev), 9).astype(int); rows = []
    for f in range(8):
        tr = dev[:max(1, edges[f] - 2 * T)]; te = dev[edges[f]:edges[f + 1]]
        t0 = time.time(); r = hourly_fold(f, b, A, lr, cs, anch, tf, rT, F, tr, te, hp); rows += r
        print(f"hourly fold {f}: {len(te)} test, {len(tr)} train, {time.time() - t0:.0f}s; " +
              " ".join(f"{x['forecast']} {x['spearman']:.3f}" for x in r), flush=True)
    return rows


def run_minute(names=None):
    close, tod = load_minute()
    lr = np.concatenate([[0.0], np.diff(np.log(np.maximum(close, 1e-9)))]); cs = cum_sq(lr)
    sc = np.load(f"{SERIES_CACHE}/series_close.npy", mmap_mode="r")
    assert len(sc) == len(close) and np.allclose(sc[::997], close[::997]), "the series cache does not match the file"
    vec = np.load(f"{SERIES_CACHE}/series_vec.npy", mmap_mode="r")
    names = names or sorted(f[:-4] for f in os.listdir(LAB_CACHE) if f.endswith(".npz"))
    rows = []
    for nm in names:
        t0 = time.time(); r = score_slice(nm, close, tod, lr, cs, vec); rows += r
        print(f"minute {nm}: {time.time() - t0:.0f}s; " + " ".join(f"{x['forecast']} {x['spearman']:.3f}" for x in r), flush=True)
    return rows


def tower_rows(minute_rows):
    """The network's recorded per-slice volatility Spearman (1-year span, 24 slices): patch and linear arch towers. Spearman only
    (the tower predicts log|r| with a Laplace NLL; it is not retrained here). Its own recorded trailing-60 baseline is kept for a check."""
    out = []
    for line in open(os.path.join(TAC, "newnet", "results.jsonl")):
        r = json.loads(line)
        if r["span"] != "1y" or r["n_slices"] != 24 or r["seed"] != 0:
            continue
        ps = r["per_slice"]
        for i, nm in enumerate(r["slices"]):
            out.append(dict(setup="minute", unit=nm, forecast=f"tower_{r['arch']}", spearman=float(ps["vol_rho"][i]),
                            spearman_h=[float(ps[f"vol_rho_h{h}"][i]) for h in range(3)], recorded=True,
                            recorded_base_spearman=float(ps["vol_rho_base"][i])))
    return out


# ---------------------------------------------------------------- report
STANDARD_MIN = ["rv60", "rv15", "rv240", "rv1440", "ewma", "garch", "har", "har_tod"]
STANDARD_HR = ["rv12", "rv24", "rv168", "rv720", "ewma", "garch", "har", "har_tod"]


def table(rows, setup, standard, learned):
    R = [r for r in rows if r["setup"] == setup]
    units = sorted({r["unit"] for r in R}); by = {}
    for r in R:
        by.setdefault(r["forecast"], {})[r["unit"]] = r
    res = {}
    for metric, better in (("spearman", max), ("qlike_cal", min)):
        base = [f for f in standard if f in by and all(metric in by[f][u] for u in units)]
        score = {f: np.mean([by[f][u][metric] for u in units]) for f in base}
        pick = better(base, key=lambda f: score[f]); res[metric] = dict(best=pick, cis={}, paired={})
        for f in by:
            us = [u for u in units if u in by[f] and metric in by[f][u] and u in by[pick]]
            if len(us) < 2:
                continue
            v = np.array([by[f][u][metric] for u in us]); res[metric]["cis"][f] = t_ci(v) + [len(us)]
            d = np.array([by[f][u][metric] - by[pick][u][metric] for u in us]); res[metric]["paired"][f] = t_ci(d) + [len(us)]
    for f in by:
        if "qlike_raw" in next(iter(by[f].values())):
            v = np.array([by[f][u]["qlike_raw"] for u in units]); res.setdefault("qlike_raw", {})[f] = t_ci(v)
    return res


def report():
    rows = [json.loads(line) for line in open(OUT) if json.loads(line).get("kind") == "unit"]
    lines = []
    for setup, std, lrn in (("minute", STANDARD_MIN, ["ridge", "tower_patch", "tower_linear"]), ("hourly", STANDARD_HR, ["hgb_ctx"])):
        if not any(r["setup"] == setup for r in rows):
            continue
        res = table(rows, setup, std, lrn); nunits = len({r["unit"] for r in rows if r["setup"] == setup})
        lines.append(f"\n== {setup}: {nunits} units; best simple baseline: Spearman -> {res['spearman']['best']}, QLIKE(cal) -> {res['qlike_cal']['best']}")
        lines.append(f"{'forecast':14s} {'Spearman mean [95% CI]':28s} {'d vs best':28s} {'QLIKE_cal mean [95% CI]':28s} {'d vs best (neg=better)':28s} QLIKE_raw")
        names = [f for f in std + lrn if f in res["spearman"]["cis"]]
        for f in names:
            sp = res["spearman"]["cis"][f]; dsp = res["spearman"]["paired"][f]
            ql = res["qlike_cal"]["cis"].get(f); dql = res["qlike_cal"]["paired"].get(f); qr = res.get("qlike_raw", {}).get(f)
            fmt = lambda c: f"{c[0]:+.4f} [{c[1]:+.4f},{c[2]:+.4f}]"  # noqa: E731
            lines.append(f"{f:14s} {fmt(sp):28s} {fmt(dsp):28s} " + (f"{fmt(ql):28s} {fmt(dql):28s} {qr[0]:.4f}" if ql else "n/a"))
        res["learned"] = lrn
        with open(OUT, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(dict(kind="summary", setup=setup, n_units=nunits, **res)) + "\n")
        for f in lrn:
            if f in res["spearman"]["paired"]:
                lo = res["spearman"]["paired"][f][1]
                lines.append(f"  -> {f} vs {res['spearman']['best']} on Spearman: CI lower bound {lo:+.4f} ({'ABOVE 0' if lo > 0 else 'includes 0 or below'})")
            if f in res["qlike_cal"]["paired"]:
                hi = res["qlike_cal"]["paired"][f][2]
                lines.append(f"  -> {f} vs {res['qlike_cal']['best']} on QLIKE(cal): CI upper bound {hi:+.4f} ({'BELOW 0 (better)' if hi < 0 else 'includes 0 or worse'})")
    print("\n".join(lines))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["minute", "hourly", "report"]); ap.add_argument("--slices", default="")
    a = ap.parse_args()
    if a.cmd == "report":
        txt = report(); open(os.path.join(HERE, "report.txt"), "w", encoding="utf-8").write(txt + "\n"); return
    rows = run_minute(a.slices.split(",") if a.slices else None) if a.cmd == "minute" else run_hourly()
    if a.cmd == "minute":
        rows += tower_rows(rows)
    with open(OUT, "a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(dict(kind="unit", **r)) + "\n")


if __name__ == "__main__":
    main()
