"""usage: python run_sizing.py sel | select | look      (SPEC.md)
sel    : signals of both side signals and the HAR state on the SELECTION months 2021-01..2024-06 (data truncated before 2024-07-01);
         evaluates the six rules, writes results_sel.json
select : picks at most one rule per side signal from results_sel.json (Calmar at 2 bps), writes selection.json
look   : refuses unless selection.json exists; the ONE look 2024-07..2025-07-20, writes results_look.json
"""
import json, os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); TAC = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(TAC, "recent")); sys.path.insert(0, os.path.join(TAC, "fwd"))
import sizelib as S  # noqa: E402
import wf  # noqa: E402
import fwdlib as F  # noqa: E402

PHASES = {"sel": ("2021-01", "2024-06", "2024-07-01", "2021-01-01", "2024-07-01"),
          "look": ("2024-07", "2025-07", "2025-07-21", "2024-07-01", "2025-07-21")}   # first, last month, data end (exclusive), period start/end
BAR_NS = 3600 * 10 ** 9
CACHE_LOOK = "D:/nt/nt_sizing_cache/fwd"


def a_preds(first, last, data_end):
    """Side signal (a): 1-hour rich, L1 logistic, 365 days, H = 1 bar (recent/wf.py), top 10% of |P-0.5| per month."""
    blocks, t, r = wf.load_cell("1h")
    X = np.asarray(blocks["rich"], np.float32); ti = pd.DatetimeIndex(t)
    sel = np.asarray(ti < pd.Timestamp(data_end))
    T = 365; X, ti, r = X[sel], ti[sel], r[sel]
    f, l = first, last
    keep = np.asarray((ti >= pd.Period(f, "M").start_time - pd.Timedelta(days=T + 1)) & (ti < (pd.Period(l, "M") + 1).start_time))
    X, ti, r = X[keep], ti[keep], r[keep]
    from sklearn.preprocessing import StandardScaler
    H, bm = wf.HOR["1h"], wf.BAR_MIN["1h"]; mask = np.abs(r) > wf.deadband("1h"); y = (r > 0).astype(int)
    for m in pd.period_range(f, l, freq="M"):
        m0, m1 = m.start_time, (m + 1).start_time
        tr, te = wf.split_masks(ti, m0, m1, H, bm, T); trm, tem = tr & mask, te & mask
        if trm.sum() < wf.MIN_TRAIN or tem.sum() < wf.MIN_TEST or len(np.unique(y[tem])) < 2 or len(np.unique(y[trm])) < 2:
            continue
        Xtr = np.asarray(X[trm], np.float32); sc = StandardScaler().fit(Xtr)
        g = lambda Z: np.clip(sc.transform(np.asarray(Z, np.float32)), -wf.CLIP, wf.CLIP)
        mdl, _ = wf._fit("l1", g(Xtr), y[trm], gap_rows=H + 1)
        idx = np.flatnonzero(te); p = mdl.predict_proba(g(X[idx]))[:, 1]
        yield str(m), ti[idx].values.astype("datetime64[ns]").astype(np.int64), p, r[idx]


def b_preds(first, last, cache):
    """Side signal (b): frozen H62 candidate: 1-hour rich, logistic C = 0.1, 730 days, H = 4 hours (fwd/fwdlib.py)."""
    X = np.load(f"{cache}/rich.npy", mmap_mode="r"); t = np.load(f"{cache}/t.npy"); anch = np.load(f"{cache}/anch.npy")
    op = np.load(f"{cache}/open.npy"); cl = np.load(f"{cache}/close.npy")
    r = F.next_open_ret(op, cl, anch, 4)
    for m, idx, p, rr in F.month_predictions(np.asarray(X), t, r, "logreg", 730, 4, first, last):
        yield m, t[idx], p, rr


def signals(preds, H, frac=0.10):
    """Top `frac` of |P-0.5| per month -> arrays (no non-overlap yet): t_open (signal bar open), entry, exit, side, r_long, month."""
    to, side, rl, mon = [], [], [], []
    for m, tt, p, r in preds:
        k = np.abs(p - 0.5) >= np.quantile(np.abs(p - 0.5), 1 - frac)
        to.append(tt[k]); side.append(np.where(p[k] > 0.5, 1, -1)); rl.append(r[k]); mon += [m] * int(k.sum())
    to, side, rl = (np.concatenate(a) for a in (to, side, rl))
    o = np.argsort(to, kind="stable")
    to, side, rl, mon = to[o], side[o], rl[o], np.array(mon)[o]
    return {"t_open": to, "entry": to + BAR_NS, "exit": to + (H + 1) * BAR_NS, "side": side, "r_long": rl, "month": mon}


def build_look_cache():
    """The 1-hour rich cache up to the cutoff (same code as fwdlib.build_cache, other bound and directory)."""
    if os.path.exists(f"{CACHE_LOOK}/rich.npy"):
        return
    import feats as F0  # noqa
    import lab
    s = F.load_bars(F.C.CSV, pd.Timestamp("2025-07-21")); S.C.assert_cut(s.index, "look bars")
    A = s.values.astype(np.float64); n = len(A); L = F.L
    anch = np.arange(L, n - (F.HMAX - 1)); t = s.index[anch - 1].values.astype("datetime64[ns]").astype(np.int64)
    W = A[anch[:, None] + np.arange(-L, 0)].astype(np.float32)
    nz = lambda x: np.nan_to_num(np.asarray(x, np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    os.makedirs(CACHE_LOOK, exist_ok=True)
    np.save(f"{CACHE_LOOK}/rich.npy", nz(lab.fs_rich(W)))
    np.save(f"{CACHE_LOOK}/t.npy", t); np.save(f"{CACHE_LOOK}/anch.npy", anch)
    np.save(f"{CACHE_LOOK}/open.npy", A[:, 0]); np.save(f"{CACHE_LOOK}/close.npy", A[:, 3])


def run_phase(phase):
    first, last, data_end, p0, p1 = PHASES[phase]; t0 = time.time(); wf.lowprio()
    rv = S.hourly_rv(end=data_end); times = rv.index.values.astype("datetime64[ns]")
    assert rv.index.max() < pd.Timestamp(data_end) and rv.index.max() <= pd.Timestamp(S.C.CUTOFF)
    gaps = int((np.diff(times.astype(np.int64)) != BAR_NS).sum())
    print(f"{phase}: {len(rv)} hourly bars to {rv.index.max()}, {gaps} gaps ({time.time() - t0:.0f}s)", flush=True)
    cache = F.CACHE if phase == "sel" else CACHE_LOOK
    if phase == "look":
        build_look_cache()
    tcache = np.load(f"{cache}/t.npy"); assert pd.Timestamp(tcache.max()) < pd.Timestamp(data_end)
    start_ns, end_ns = int(pd.Timestamp(p0).value), int(pd.Timestamp(p1).value)
    out = {"phase": phase, "months": [first, last], "period": [p0, p1], "hourly_gaps": gaps, "signals": {}}
    for name, H, gen in (("a_1h_rich_l1_365", 1, a_preds(first, last, data_end)), ("b_h62_4h_rich_logreg_730", 4, b_preds(first, last, cache))):
        sg = signals(list(gen), H)
        bar = pd.Index(times).get_indexer(sg["t_open"].astype("datetime64[ns]")); assert (bar >= 0).all(), "signal bar not in the hourly index"
        sg["bar"] = bar
        st = S.state(rv.values, times, H, "2018-01", last)
        for k in ("sigma", "target", "dlv", "thr"):
            assert np.isfinite(st[k][bar]).all(), f"{name}: {k} not available at some signal"
        # HAR sanity on the phase: rank correlation of the forecast with the realised log mean RV of the next H bars
        y = S.har_target(rv.values, H); ph = (pd.DatetimeIndex(times) >= pd.Timestamp(p0)) & np.isfinite(y) & np.isfinite(st["pred"])
        from scipy.stats import spearmanr
        har_sp = float(spearmanr(st["pred"][ph], y[ph])[0])
        res = S.evaluate(sg, st, start_ns, end_ns)
        out["signals"][name] = {"H": H, "n_signals": int(len(bar)), "har_spearman_pred_vs_realised": har_sp,
                                "signal_months": sorted(set(sg["month"])), "rules": res}
        print(name, "signals", len(bar), "HAR spearman %.3f" % har_sp, f"({time.time() - t0:.0f}s)", flush=True)
        for rule in S.RULES:
            s = res[rule]["levels"]["2"]["stats"]
            print(f"  {rule:22s} n {s['n_trades']:5d} hit {s['hit']:.3f} mean {s['mean_bps']:+7.2f} cum {s['cum_return_pct']:+8.1f}% "
                  f"dd {s['max_drawdown_pct']:5.1f}% calmar {s['calmar']:+6.2f} m>60 {s['months_hit_above_60']}/{s['n_months']} "
                  f"ddm<5 {s['share_months_dd_below_5']:.2f}", flush=True)
    out["seconds"] = round(time.time() - t0, 1)
    json.dump(out, open(f"{HERE}/results_{phase}.json", "w"), indent=1)
    print(f"done {out['seconds']}s")


def select():
    r = json.load(open(f"{HERE}/results_sel.json")); picks = {}; table = []
    for name, d in r["signals"].items():
        cal = {rule: d["rules"][rule]["levels"]["2"]["stats"]["calmar"] for rule in S.RULES}
        base = cal["R0_fixed"]
        best = max((x for x in S.RULES if x != "R0_fixed"), key=lambda x: cal[x])
        picks[name] = best if cal[best] > base else None
        table.append((name, cal, base, best))
    json.dump({"picks": picks, "calmar_2bps_sel": {n: c for n, c, _, _ in table}}, open(f"{HERE}/selection.json", "w"), indent=1)
    for n, c, b, best in table:
        print(n, {k: round(v, 3) for k, v in c.items()}, "-> pick:", picks[n])


if __name__ == "__main__":
    ph = sys.argv[1]
    if ph == "select":
        select()
    elif ph == "look":
        if not os.path.exists(f"{HERE}/selection.json"):
            sys.exit("look refused: selection.json missing")
        run_phase("look")
    else:
        run_phase("sel")
