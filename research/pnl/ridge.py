"""T1-LIN and T2-LIN (SPEC 'track1' 2, 'track2' 2): ridge forecaster of r_{t,H}/sigma_{t,H} plus cost-aware hysteresis.

Per fold (annual expanding window):
- rows: every minute (Track 1, 42 intraday columns) or 5-minute origins (Track 2, 55 columns);
- target y = clip(ln(c_{t+H}/c_t) / sigma_{t,H}, +-6); sigma_{t,H} = sigma60 sqrt(H) for H <= 60, sigma_t sqrt(H) beyond;
- grid: H x theta {0.5, 1, 2} x lambda {1, 10, 1e2, 1e3, 1e4} (45 configurations for Track 1, 90 for Track 2);
- purged, blocked 5-fold CV on the training window (contiguous blocks; rows whose label reaches into a CV test block
  are purged with the largest H of the grid; 1-day embargo after each block); each CV block is traded by the
  hysteresis state machine and scored on engine F (c_in 3 bps, c_out 5 bps) per minute; score = mean daily net;
- the best configuration is refit on the whole (purged) training window; if its CV score <= 0 the year is CASH.
Outputs: per-bar forecast mu (bps), decision mask, hold H, threshold and maker window for the dev-OOS engine-E run,
and the fold-S model (scaler + weights) frozen for the seal.

  python ridge.py 1|2
"""
import hashlib
import json
import os
import sys
import time

import numpy as np

import fast
import folds as FO
import prep
from policy import C_RT

LAMBDAS = (1.0, 10.0, 1e2, 1e3, 1e4)
THETAS = (0.5, 1.0, 2.0)
HS = {1: (1, 5, 15), 2: (1, 5, 15, 60, 240, 1440)}
C_IN, C_OUT = 3e-4, 5e-4
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def rows_for(st, track, a, b):
    r = np.arange(a, b)
    return r if track == 1 else r[prep.decision_mask(st.b.t[a:b], 5)]


def design(st, scaler, rows, chunk=250_000):
    """Standardised features with an intercept column, float32."""
    Z = np.empty((len(rows), len(scaler.cols) + 1), np.float32)
    for i in range(0, len(rows), chunk):
        Z[i:i + chunk, :-1] = scaler(st.X, rows[i:i + chunk])
    Z[:, -1] = 1.0
    return Z


def gram(Z, Y, i, j, chunk=250_000):
    p = Z.shape[1]
    G, b = np.zeros((p, p)), np.zeros((p, Y.shape[1]))
    for k in range(i, j, chunk):
        z = Z[k:min(k + chunk, j)].astype(np.float64)
        G += z.T @ z
        b += z.T @ Y[k:min(k + chunk, j)]
    return G, b


def solve(G, b, lam):
    P = np.eye(G.shape[0]) * lam
    P[-1, -1] = 0.0                                                    # intercept unpenalised
    return np.linalg.solve(G + P, b)


def targets(st, rows, Hs):
    Y = np.empty((len(rows), len(Hs)))
    for k, H in enumerate(Hs):
        Y[:, k] = np.clip((st.lc[rows + H] - st.lc[rows]) / st.sigma_H(rows, H), -6, 6)
    return Y


def hyst_daily(st, mu_rows, rows, m0, m1, H, thr):
    """Trade one contiguous minute range [m0, m1) from forecasts at `rows` (bps); engine F per minute; daily sums."""
    mu = np.full(m1 - m0, np.nan)
    mu[rows - m0] = mu_rows
    dec = ~np.isnan(mu)
    x = fast.hysteresis_positions(mu, dec, int(H), float(thr))
    net = fast.engine_f_minutes(x, st.R1[m0:m1], st.F1[m0:m1], C_IN, C_OUT, 0.0)
    d = st.day[m0:m1] - st.day[m0]
    return np.bincount(d, weights=net), np.abs(np.diff(np.r_[0.0, x])).sum()


def fit_fold(st, track, test_start, test_end, n_cv=5, log=print):
    Hs = HS[track]
    Hmax = max(Hs)
    cols = st.cols(track)
    a_ts, b_ts = FO.train_range(track, test_start, Hmax)
    a, b = st.idx(a_ts), st.idx(b_ts)
    rows = rows_for(st, track, a, b)
    t0 = time.time()
    scaler = prep.Scaler(st.X, rows, cols)
    Z = design(st, scaler, rows)
    Y = targets(st, rows, Hs)
    n = len(rows)
    edges = np.linspace(0, n, n_cv + 1).astype(int)
    G_all, b_all = gram(Z, Y, 0, n)
    blocks = [gram(Z, Y, edges[j], edges[j + 1]) for j in range(n_cv)]
    log(f"  rows {n:,}  design+gram {time.time() - t0:.1f}s")
    cfgs = [(H, th, lam) for H in Hs for th in THETAS for lam in LAMBDAS]
    daily = {c: [] for c in cfgs}
    turn = {c: 0.0 for c in cfgs}
    for j in range(n_cv):
        s_j, e_j = edges[j], edges[j + 1]
        bs, be = rows[s_j], rows[e_j - 1] + 1                          # bar range of the CV test block
        pre = (np.searchsorted(rows, bs - Hmax - 1), s_j)               # labels reaching into the block
        post = (e_j, np.searchsorted(rows, be + 1440))                  # 1-day embargo after the block
        G, bb = G_all - blocks[j][0], b_all - blocks[j][1]
        for (i0, i1) in (pre, post):
            if i1 > i0:
                g2, b2 = gram(Z, Y, i0, i1)
                G, bb = G - g2, bb - b2
        Zj = Z[s_j:e_j].astype(np.float64)
        sig = {H: st.sigma_H(rows[s_j:e_j], H) * 1e4 for H in Hs}
        for lam in LAMBDAS:
            W = solve(G, bb, lam)
            P = Zj @ W
            for k, H in enumerate(Hs):
                mu = P[:, k] * sig[H]
                for th in THETAS:
                    dly, tv = hyst_daily(st, mu, rows[s_j:e_j], bs, be, H, th * C_RT)
                    daily[(H, th, lam)].append(dly)
                    turn[(H, th, lam)] += tv
    score = {c: float(np.mean(np.concatenate(daily[c]))) for c in cfgs}
    best = max(cfgs, key=lambda c: score[c])
    cash = score[best] <= 0.0
    H, th, lam = best
    k = Hs.index(H)
    w = solve(G_all, b_all, lam)[:, k]
    log(f"  best H={H} theta={th} lambda={lam:g} CV mean daily net {score[best] * 1e4:.2f} bps"
        f"{'  -> CASH' if cash else ''}  ({time.time() - t0:.1f}s)")
    return {"best": best, "cash": cash, "score": score, "turnover": turn, "w": w, "scaler": scaler, "H": H,
            "theta": th, "lam": lam, "cols": cols, "train": (int(a), int(b)), "n_rows": int(n)}


def predict(st, fit, track, s, e, chunk=250_000):
    rows = rows_for(st, track, s, e)
    mu = np.empty(len(rows))
    for i in range(0, len(rows), chunk):
        r = rows[i:i + chunk]
        Z = np.c_[fit["scaler"](st.X, r).astype(np.float64), np.ones(len(r))]
        mu[i:i + chunk] = (Z @ fit["w"]) * st.sigma_H(r, fit["H"]) * 1e4
    return rows, mu


def run_track(track, st=None, log=print):
    st = st or prep.Store()
    n = st.n
    mu = np.full(n, np.nan, np.float32)
    H_arr = np.ones(n, np.int64)
    thr = np.full(n, np.inf)
    summary = {"track": track, "folds": {}}
    for name, ts0, ts1 in FO.FOLDS + [FO.FOLD_S]:
        log(f"T{track}-LIN fold {name}")
        fit = fit_fold(st, track, ts0, ts1, log=log)
        info = {"H": fit["H"], "theta": fit["theta"], "lambda": fit["lam"], "cash": fit["cash"],
                "cv_mean_daily_bps": fit["score"][fit["best"]] * 1e4, "n_rows": fit["n_rows"], "train": fit["train"],
                "cv_table_bps": {f"H{c[0]}_th{c[1]}_lam{c[2]:g}": v * 1e4 for c, v in fit["score"].items()},
                "cv_turnover": {f"H{c[0]}_th{c[1]}_lam{c[2]:g}": v for c, v in fit["turnover"].items()}}
        if name == "S":
            os.makedirs(prep.WORK, exist_ok=True)
            path = os.path.join(prep.WORK, f"ridge_T{track}_S.npz")
            np.savez(path, w=fit["w"], med=fit["scaler"].med, iqr=fit["scaler"].iqr, cols=fit["scaler"].cols,
                     H=fit["H"], theta=fit["theta"], lam=fit["lam"], cash=fit["cash"])
            info["bundle"] = path
            info["bundle_sha256"] = hashlib.sha256(open(path, "rb").read()).hexdigest()
        else:
            s, e = st.idx(ts0), st.idx(ts1)
            rows, m = predict(st, fit, track, s, e)
            if fit["cash"]:
                m[:] = 0.0                                             # cash year: exit and stay flat
            mu[rows] = m
            H_arr[s:e] = fit["H"]
            thr[s:e] = fit["theta"] * C_RT
            info["mu_abs_median_bps"] = float(np.median(np.abs(m)))
            info["share_above_thr"] = float(np.mean(np.abs(m) > fit["theta"] * C_RT))
        summary["folds"][name] = info
    dec = ~np.isnan(mu)
    np.savez(os.path.join(prep.WORK, f"ridge_T{track}_dev.npz"), mu=mu, H=H_arr, thr=thr, dec=dec)
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(summary, open(os.path.join(RESULTS, f"ridge_T{track}.json"), "w"), indent=1)
    return summary


if __name__ == "__main__":
    run_track(int(sys.argv[1]))
