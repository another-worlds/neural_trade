"""Harness checks (SPEC 'data_and_features' CHECKS 1-6, simulator TEST 12); any failure is kill K0.

1 causality: bars after t0 replaced by noise / a permutation leave feature rows <= t0 bit-identical (20 cuts on a
  150k-bar window)
2 funding alignment: a print at 08:00 is charged only to the position held over the 07:59 bar
3 fold manifest: training, inner-validation and test ranges are disjoint and purged (every track, fold and h)
4 deliberate leak: features shifted by -1 bar make T1-LIN (fold Y2023) strongly profitable on its test year
5 labels shuffled within each day: T1-LIN refit on them, traded as sign(mu) every minute at zero cost on its test
  year: gross PnL within 2 SE of 0
6 load() never returns a bar at or after SEAL_START
12 reconciliation (dev OOS): engine-F daily PnL of the realised C7-B exposure with engine-E fees (the p_sel input)
  versus engine-E daily log returns, for B&H, T2-P and the plain fade: |gap| < 0.5% of equity per year

  python checks.py   -> results/checks.json
"""
import json
import os

import numpy as np

import data as D
import fast
import features as FE
import folds as FO
import prep
import ridge
import sim
import stats as ST
from policy import C_RT

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def copy_slice(b, a, e):
    """Bars.slice returns views; the checks modify bars, so they work on copies."""
    out = b.slice(a, e)
    for k, v in list(out.__dict__.items()):
        if isinstance(v, np.ndarray):
            out.__dict__[k] = v.copy()
    return out


def check_causality(b, n_cuts=20, win=150_000, seed=0):
    rs = np.random.default_rng(seed)
    a = 1_000_000
    c_before = b.c[a:a + win].copy()
    bb = copy_slice(b, a, a + win)
    X0, _, _ = FE.build(bb)
    worst = 0.0
    for t0 in rs.integers(50_000, win - 1000, n_cuts):
        for mode in ("noise", "perm"):
            b2 = copy_slice(b, a, a + win)
            k = t0 + 1
            if mode == "noise":
                m = 1 + rs.normal(0, 1e-3, win - k)
                for arr in (b2.o, b2.h, b2.l, b2.c):
                    arr[k:] = arr[k:] * m
                b2.h[k:] = np.maximum.reduce([b2.h[k:], b2.o[k:], b2.c[k:]])
                b2.l[k:] = np.minimum.reduce([b2.l[k:], b2.o[k:], b2.c[k:]])
                b2.v[k:] = b2.v[k:] * rs.uniform(0.5, 2, win - k)
            else:
                p = rs.permutation(win - k) + k
                for name in ("o", "h", "l", "c", "v", "qv", "n", "tbb"):
                    arr = getattr(b2, name)
                    arr[k:] = arr[p]
            X1, _, _ = FE.build(b2)
            same = np.array_equal(X0[:k], X1[:k], equal_nan=True)
            if not same:
                worst = max(worst, float(np.nanmax(np.abs(X0[:k] - X1[:k]))))
            if not same:
                return {"pass": False, "t0": int(t0), "mode": mode, "max_diff": worst}
    assert np.array_equal(b.c[a:a + win], c_before)          # the store itself was not touched
    return {"pass": True, "cuts": n_cuts, "modes": ["noise", "perm"], "window": win}


def check_funding_alignment():
    n = 1440
    t = 1600041600 + 60 * np.arange(n)                       # 2020-09-14 00:00 UTC
    c = np.full(n, 30000.0)
    z = np.ones(n, np.float32)
    i8 = int(np.searchsorted(t, t[0] + 8 * 3600))
    ft, fr = np.array([t[i8]]), np.array([1e-3])
    b = D.Bars(t, c.copy(), c.copy(), c.copy(), c.copy(), z, z, z, z * 0.5, 1, ft, fr, np.zeros(1, bool))
    out = {}
    # realistic fills (C10-B: taker at the next open): a decision at the 07:58 close fills at the 07:59 open and is
    # held over the 07:59 bar -> pays; a decision at the 07:59 close fills at the 08:00 open, after the print -> free.
    for name, dec_bar in (("held_over_0759", i8 - 2), ("decided_at_0759_close", i8 - 1)):
        tg = np.full(n, np.nan)
        tg[dec_bar] = 1.0
        r = sim.simulate(b, sim.TargetPolicy(tg[1:], 1, K=1), 1, n, fee="C10", fill="B", close_at_end=False)
        out[name] = float(r["funding"])
    # nominal (N) fills happen AT the decision close, so the position exists at the end of the 07:59 bar and pays,
    # exactly as engine F charges x_k for funding in (t_k, t_k + h]; E-N and F therefore agree (reconciliation only).
    tg = np.full(n, np.nan)
    tg[i8 - 1] = 1.0
    out["nominal_decided_at_0759_close"] = float(sim.simulate(b, sim.TargetPolicy(tg[1:], 1, K=1), 1, n, fee="C10",
                                                              fill="N", close_at_end=False)["funding"])
    ok = (abs(out["held_over_0759"] - 1e-3) < 1e-12 and out["decided_at_0759_close"] == 0.0
          and abs(out["nominal_decided_at_0759_close"] - 1e-3) < 1e-12)
    return {"pass": bool(ok), **out}


def check_fold_manifest(st):
    bad = []
    for track, hs in ((1, (1, 5, 15)), (2, (1, 5, 15, 60, 240, 1440))):
        for name, s, e in FO.FOLDS + [FO.FOLD_S]:
            for h in hs:
                a, b = FO.train_range(track, s, h)
                (ta, tb), (va, vb) = FO.inner_split(track, s, h)
                # rows used: ridge/test labels from rows < b; nets sample rows r with r + h + 1 bar < tb (train.py);
                # validation rows < vb = b. Every label (t + h, plus the 1-bar fill lag) ends before the next block.
                last_label_end = (b - 60) + (h + 1) * 60
                if not (a == ta < tb and tb + FO.DAY <= va < vb == b and last_label_end < s < e):
                    bad.append((track, name, h))
                if s >= FO.SEAL_START and name != "S":
                    bad.append((track, name, h, "test in seal"))
    return {"pass": not bad, "bad": bad}


class Shifted:
    """Features at t+1 served for row t (a deliberate one-bar leak)."""

    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, key):
        rows = key[0] if isinstance(key, tuple) else key
        rows = np.minimum(np.asarray(rows) + 1, len(self.X) - 1)
        return self.X[rows] if not isinstance(key, tuple) else self.X[(rows,) + key[1:]]


def ridge_test_year(st, fold="Y2023", zero_cost=False, shuffle=False, always_in=False):
    name, ts0, ts1 = next(f for f in FO.FOLDS if f[0] == fold)
    orig = ridge.targets
    if shuffle:
        def shuffled(st_, rows, Hs):
            Y = orig(st_, rows, Hs)
            d = st_.day[rows]
            rs = np.random.default_rng(0)
            for dd in np.unique(d):
                m = np.flatnonzero(d == dd)
                Y[m] = Y[rs.permutation(m)]
            return Y
        ridge.targets = shuffled
    try:
        fit = ridge.fit_fold(st, 1, ts0, ts1, log=lambda *_: None)
    finally:
        ridge.targets = orig
    s, e = st.idx(ts0), st.idx(ts1)
    rows, mu = ridge.predict(st, fit, 1, s, e)
    m = np.full(e - s, np.nan)
    m[rows - s] = mu
    if always_in:                                   # sign(mu) every minute: a forecaster that always trades
        x = np.sign(np.nan_to_num(m))
    else:
        x = fast.hysteresis_positions(m, ~np.isnan(m), int(fit["H"]), float(fit["theta"] * C_RT))
    c_in, c_out = (0.0, 0.0) if zero_cost else (3e-4, 5e-4)
    net = fast.engine_f_minutes(x, st.R1[s:e], st.F1[s:e], c_in, c_out, 0.0)
    daily = np.bincount(st.day[s:e] - st.day[s], weights=net)
    return {"H": fit["H"], "theta": fit["theta"], "lambda": fit["lam"], "cv_bps": fit["score"][fit["best"]] * 1e4,
            "mean_daily_bps": float(daily.mean() * 1e4), "se_daily_bps": float(daily.std(ddof=1) / np.sqrt(len(daily)) * 1e4),
            "turnover": float(np.abs(np.diff(np.r_[0.0, x])).sum())}


def check_reconciliation(st):
    path = os.path.join(prep.WORK, "dev_daily.npz")
    if not os.path.exists(path):
        return {"pass": None, "note": "run run_dev.py first"}
    dd = dict(np.load(path))
    ex = dict(np.load(os.path.join(prep.WORK, "dev_expo.npz")))
    s, e = st.idx(FO.DEV_OOS[0]), st.n
    r1 = np.r_[0.0, np.diff(st.lc)][s:e]
    f1 = st.b.funding[s:e]
    import gate as G
    out, ok = {}, True
    for name in ("B&H", "T2-P", "plain_fade"):
        if name not in ex:
            out[name] = "missing"
            ok = False
            continue
        x = G.lead(G.held(ex[name], r1))
        fdaily = ST.engine_f_daily(x, r1, G.lead(f1, 0.0), st.day[s:e], c_in=0.0, c_out=0.0)[0] - dd[f"fees|{name}"]
        eday = np.expm1(dd[f"{name}|C7B"])
        gap = (fdaily - eday)
        years = len(gap) / 365.0
        per_year = float(np.abs(gap.sum()) / years)
        out[name] = {"gap_per_year": per_year, "max_abs_daily_gap_bps": float(np.abs(gap).max() * 1e4)}
        ok &= per_year < 0.005
    return {"pass": bool(ok), **out}


def main():
    out = {}
    st = prep.Store()
    out["6_seal_truncation"] = {"pass": bool(st.b.t.max() < D.SEAL_START), "last_t": str(D.utc(int(st.b.t.max())))}
    print(out["6_seal_truncation"], flush=True)
    out["2_funding_alignment"] = check_funding_alignment()
    print(out["2_funding_alignment"], flush=True)
    out["3_fold_manifest"] = check_fold_manifest(st)
    print(out["3_fold_manifest"], flush=True)
    out["1_causality"] = check_causality(st.b)
    print(out["1_causality"], flush=True)
    honest = ridge_test_year(st)
    leak = ridge_test_year(prep.Store(bars=st.b, X=Shifted(st.X), names=st.names, sigma=st.sigma))
    out["4_deliberate_leak"] = {"pass": bool(leak["mean_daily_bps"] > 10 * max(honest["mean_daily_bps"], 1.0)),
                                "honest": honest, "leaked": leak}
    print(out["4_deliberate_leak"], flush=True)
    sh = ridge_test_year(st, zero_cost=True, shuffle=True, always_in=True)
    out["5_shuffled_labels"] = {"pass": bool(abs(sh["mean_daily_bps"]) <= 2 * sh["se_daily_bps"] + 1e-12), **sh}
    print(out["5_shuffled_labels"], flush=True)
    out["12_reconciliation"] = check_reconciliation(st)
    print(out["12_reconciliation"], flush=True)
    out["all_pass"] = all(v.get("pass") is not False for v in out.values() if isinstance(v, dict))
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(out, open(os.path.join(RESULTS, "checks.json"), "w"), indent=1, default=float)
    print("ALL PASS" if out["all_pass"] else "SOME CHECK FAILED")
    return out


if __name__ == "__main__":
    main()
