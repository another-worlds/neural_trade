"""Night experiment 3: geometry as VOLATILITY features, in volbench's protocol (runs/tactical/vol/volbench.py: the same 24 minute
slices with their 1-year train spans and 8 hourly dev folds, target |r| over the horizon, Spearman and QLIKE with the train-held-out
scale, paired t-intervals over units, best standard baseline picked among rv/EWMA/GARCH/HAR on the same units).
Learned models here (all log(|r|+1e-5) targets, HistGradientBoosting unless named, train points strided by 10):
  hgb_har       HAR logs (4 lags) + time-of-day harmonics                      (the readout WITHOUT geometry; the control)
  hgb_har_vg    + geometry of the volatility-type channels (ATR, band width, Bollinger/Keltner/Donchian edges, ADX, OBV)
  hgb_har_gR    + geometry of all 22 channels (crossing windows 3/5/10/20)
  hgb_gR        geometry alone
  ridge_har / ridge_har_vg   the linear versions
Hourly adds hgb_ctx (volbench's learned model), hgb_ctx_gR.
usage: python volgeom.py minute --shard i --of k | hourly | report"""
import argparse, json, math, os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); TAC = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(TAC, "vol")); sys.path.insert(0, os.path.join(TAC, "lab"))
import volbench as V  # noqa: E402
import geomlib as G  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402

VOL_CH = ("atr14", "bb_width", "bb_up", "bb_lo", "kel_up", "kel_lo", "don_up", "don_lo", "adx14", "obv")
LEARNED = ["hgb_har", "hgb_har_vg", "hgb_har_gR", "hgb_gR", "ridge_har", "ridge_har_vg"]
LEARNED_H = ["hgb_har", "hgb_har_vg", "hgb_har_gR", "hgb_gR", "hgb_ctx_ref", "hgb_ctx_gR"]
STRIDE = 10


def vg(W): return G.geom(W, VOL_CH, (5,))
def gR(W): return G.geom_rich(W)


def mk_hgb(): return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=300, random_state=0)


def fit_predict(kind, Xtr, Ytr, Xte, gap):
    """One horizon: fit on the first 85% (alpha / scale), then on all. Returns (pred_te, s2_te, c-scale s2 hold-out pred, hold-out idx)."""
    m = len(Xtr); cut = int(round(0.85 * m))
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    f = (lambda X: np.clip((X - mu) / sd, -8, 8)) if kind == "ridge" else (lambda X: X)
    make = (lambda: Ridge(alpha=100.0)) if kind == "ridge" else mk_hgb
    ho = make().fit(f(Xtr[:cut - gap]), Ytr[:cut - gap]); p_ho = ho.predict(f(Xtr[cut:]))
    full = make().fit(f(Xtr), Ytr)
    return full.predict(f(Xte)), p_ho, cut


def add_forecast(U, name, Xtr, Ytr, rtr, Xte, gap, kind, nh):
    """Fit per horizon (columns of Ytr / rtr), add the unit row: Spearman of the predicted log|r|, QLIKE with the held-out scale."""
    P = np.zeros((len(Xte), nh)); S2 = np.zeros((len(Xte), nh)); C = []
    for i in range(nh):
        p, p_ho, cut = fit_predict(kind, Xtr, Ytr[:, i], Xte, gap)
        s2_ho = np.exp(2 * p_ho)[:, None]; s2_ho = V.floor_s2(s2_ho, s2_ho)[:, 0]
        C.append(V.scale_c(rtr[cut:, i] ** 2, s2_ho))
        P[:, i] = p; S2[:, i] = V.floor_s2(np.exp(2 * p)[:, None], np.exp(2 * p_ho)[:, None])[:, 0]
    return U.add(name, P, S2, None, C)


def ohlcv():
    d = pd.read_csv(V.CSV, usecols=["timestamp", "open", "high", "low", "close", "volume"])
    return d[["open", "high", "low", "close", "volume"]].to_numpy(np.float32)


def windows(A, t):
    return A[t[:, None] + np.arange(-V.WIN + 1, 1)[None, :]]


def run_minute(shard, of):
    close, tod = V.load_minute(); A = ohlcv(); assert len(A) == len(close)
    lr = np.concatenate([[0.0], np.diff(np.log(np.maximum(close, 1e-9)))]); cs = V.cum_sq(lr)
    logv = [np.log(V.trailing_var(cs, k) + 1e-12) for k in V.HAR_LAGS]
    names = sorted(f[:-4] for f in os.listdir(V.LAB_CACHE) if f.endswith(".npz"))[shard::of]
    out = os.path.join(HERE, f"volgeom_rows_{shard}.jsonl")
    for nm in names:
        t0 = time.time(); D = np.load(f"{V.LAB_CACHE}/{nm}.npz"); rva = D["rva"].astype(np.float64); nva = len(rva)
        vs = V.locate_val(close, D["Wva"]); avail = vs - V.GAPB - (1440 - V.WIN + 1) + 1
        n = int(min(V.SPAN, avail)); s0 = vs - V.GAPB - (n - 1)
        t_tr = s0 + V.WIN - 1 + np.arange(n); t_va = vs + V.WIN - 1 + np.arange(nva)
        sub = t_tr[::STRIDE]
        rtr = np.stack([close[sub + h] / close[sub] - 1.0 for h in V.HZ], 1)
        rv = np.stack([close[t_va + h] / close[t_va] - 1.0 for h in V.HZ], 1)
        assert np.allclose(rv, rva, rtol=1e-4, atol=2e-6) and sub[-1] + max(V.HZ) < vs
        Ytr = np.log(np.abs(rtr) + 1e-5)
        def har(t):
            ang = lambda k: 2 * np.pi * k * (tod[t] + 7.5) / 1440.0
            return np.column_stack([lv[t] for lv in logv] + [np.sin(ang(1)), np.cos(ang(1)), np.sin(ang(2)), np.cos(ang(2))])
        Htr, Hva = har(sub), har(t_va)
        Wtr, Wva = windows(A, sub), windows(A, t_va)
        Gtr, Gva = {"vg": vg(Wtr), "gR": gR(Wtr)}, {"vg": vg(Wva), "gR": gR(Wva)}
        U = V.Unit(nm, [rv[:, i] for i in range(3)]); gap = max(1, 80 // STRIDE)
        jobs = [("hgb_har", "hgb", Htr, Hva), ("hgb_har_vg", "hgb", np.hstack([Htr, Gtr["vg"]]), np.hstack([Hva, Gva["vg"]])),
                ("hgb_har_gR", "hgb", np.hstack([Htr, Gtr["gR"]]), np.hstack([Hva, Gva["gR"]])), ("hgb_gR", "hgb", Gtr["gR"], Gva["gR"]),
                ("ridge_har", "ridge", Htr, Hva), ("ridge_har_vg", "ridge", np.hstack([Htr, Gtr["vg"]]), np.hstack([Hva, Gva["vg"]]))]
        for fname, kind, Xtr, Xte in jobs:
            row = add_forecast(U, fname, Xtr, Ytr, rtr, Xte, gap, kind, 3)
            row.update(setup="minute", n_val=int(nva), n_train=int(len(sub)))
            open(out, "a", encoding="utf-8").write(json.dumps(dict(kind="unit", **row)) + "\n")
        print(f"minute {nm}: {time.time() - t0:.0f}s " + " ".join(f"{k} {v['spearman']:.3f}" for k, v in U.rows.items()), flush=True)


def run_hourly():
    b = V.hourly_data(); A = b.values.astype(np.float64); L = 60; T = V.HOURLY_T
    anch = np.arange(L + 168, len(A) - T); HOLD = pd.Timestamp("2024-07-01"); t_exit = b.index[anch + T - 1]
    F, hp = V.hourly_features(b, anch)
    lc = np.log(A[:, 3]); lr = np.concatenate([[0.0], np.diff(lc)]); cs = V.cum_sq(lr)
    tf = anch - 1; rT = A[anch + T - 1, 3] / A[anch - 1, 3] - 1
    W = np.stack([A[i - L:i] for i in anch]).astype(np.float32)
    Gv, Gr = vg(W), gR(W)
    logv = [np.log(V.trailing_var(cs, k) + 1e-12) for k in V.H_HAR_LAGS]
    hr = b.index.hour.values.astype(float) * 60.0
    ang = lambda k: 2 * np.pi * k * (hr[tf] + T * 30.0) / 1440.0
    H = np.column_stack([lv[tf] for lv in logv] + [np.sin(ang(1)), np.cos(ang(1)), np.sin(ang(2)), np.cos(ang(2))])
    dev = np.arange(len(anch))[t_exit < HOLD]; edges = np.linspace(len(dev) * 0.3, len(dev), 9).astype(int)
    out = os.path.join(HERE, "volgeom_rows_hourly.jsonl"); Y = np.log(np.abs(rT) + 1e-5)
    for f in range(8):
        t0 = time.time(); tr = dev[:max(1, edges[f] - 2 * T)]; te = dev[edges[f]:edges[f + 1]]
        assert anch[tr[-1]] + T - 1 < tf[te[0]]
        U = V.Unit(f"fold{f}", [rT[te]]); gap = 2 * T
        jobs = [("hgb_har", H), ("hgb_har_vg", np.hstack([H, Gv])), ("hgb_har_gR", np.hstack([H, Gr])), ("hgb_gR", Gr),
                ("hgb_ctx_ref", F), ("hgb_ctx_gR", np.hstack([F, Gr]))]
        for fname, X in jobs:
            # fit_predict holds out the last 15% for the scale (volbench's hourly used 20%; the control is run the same way here)
            row = add_forecast(U, fname, X[tr], Y[tr][:, None], rT[tr][:, None], X[te], gap, "hgb", 1)
            row.update(setup="hourly", n_val=int(len(te)), n_train=int(len(tr)))
            open(out, "a", encoding="utf-8").write(json.dumps(dict(kind="unit", **row)) + "\n")
        print(f"hourly fold {f}: {time.time() - t0:.0f}s " + " ".join(f"{k} {v['spearman']:.3f}" for k, v in U.rows.items()), flush=True)


def report():
    base = [json.loads(l) for l in open(os.path.join(TAC, "vol", "results.jsonl")) if json.loads(l).get("kind") == "unit"]
    mine = []
    for fn in sorted(os.listdir(HERE)):
        if fn.startswith("volgeom_rows_"):
            mine += [json.loads(l) for l in open(os.path.join(HERE, fn))]
    lines = []
    for setup, std, lrn in (("minute", V.STANDARD_MIN, LEARNED + ["ridge", "tower_patch", "tower_linear"]),
                            ("hourly", V.STANDARD_HR, LEARNED_H + ["hgb_ctx"])):
        rows = [r for r in base + mine if r["setup"] == setup]
        res = V.table(rows, setup, std, lrn); units = len({r["unit"] for r in mine if r["setup"] == setup})
        best_sp, best_ql = res["spearman"]["best"], res["qlike_cal"]["best"]
        lines.append(f"\n== {setup}: {units} units. best standard: Spearman -> {best_sp}, QLIKE(cal) -> {best_ql}")
        fmt = lambda c: f"{c[0]:+.4f} [{c[1]:+.4f},{c[2]:+.4f}]"
        lines.append(f"{'forecast':14s} {'Spearman':28s} {'d vs best std':28s} {'d vs hgb_har (geometry effect)':32s} {'QLIKE_cal':28s} {'d vs best std':28s} d vs hgb_har")
        by = {}
        for r in rows:
            by.setdefault(r["forecast"], {})[r["unit"]] = r
        def pair(f, g, m):
            us = [u for u in by[f] if u in by.get(g, {}) and m in by[f][u] and m in by[g][u]]
            return V.t_ci(np.array([by[f][u][m] - by[g][u][m] for u in us])) if len(us) > 1 else None
        for f in std + lrn:
            if f not in res["spearman"]["cis"]:
                continue
            ctrl = "hgb_har" if not f.startswith("ridge") else "ridge_har"
            if f == "hgb_ctx_gR": ctrl = "hgb_ctx_ref"
            ps = pair(f, ctrl, "spearman") if f != ctrl and ctrl in by else None
            pq = pair(f, ctrl, "qlike_cal") if f != ctrl and ctrl in by else None
            ql = res["qlike_cal"]["cis"].get(f); dql = res["qlike_cal"]["paired"].get(f)
            lines.append(f"{f:14s} {fmt(res['spearman']['cis'][f]):28s} {fmt(res['spearman']['paired'][f]):28s} {(fmt(ps) if ps else '-'):32s} "
                         f"{(fmt(ql) if ql else 'n/a'):28s} {(fmt(dql) if dql else 'n/a'):28s} {(fmt(pq) if pq else '-')}")
    txt = "\n".join(lines); print(txt); open(os.path.join(HERE, "volgeom_report.txt"), "w", encoding="utf-8").write(txt + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--shard", type=int, default=0); ap.add_argument("--of", type=int, default=1)
    a = ap.parse_args()
    {"minute": lambda: run_minute(a.shard, a.of), "hourly": run_hourly, "report": report}[a.cmd]()
