"""Long-train ladder (tactical, owner 2026-10-09): does the rebuild ladder change with far more training data?
The 24 lab slices keep their val (scoring) block (asserted equal to the cache); the training block grows backwards from
81 bars before the val block's first window (the cache's own 80-bar purge gap) over 11.5 days (= the cache), 90 days, 365 days, 3 years
(or all the history before the slice: the actual window count is recorded). Features come from the continuous minute series
(lab functions on 60-bar windows, chunked); nothing windowed is stored. Steps are ladder.py's L0 / L5 / L6 / L7, unchanged logic
(L2 C per horizon on the last 15% of train, early stopping on it, GRU on ladder.seq_channels), plus a magnitude read.

Training rows per step are capped by a stride over the (recent-aligned) window starts: L0/L5/L6 at most CAP_LIN rows, L7 at most
CAP_SEQ rows (11.5 d and 90 d: L0/L5/L6 full; 11.5 d: all steps full = the ladder). L0full = L0 on every window (stride 1).
Magnitude: Ridge (alpha on the last 15% of train) on log(|r| + 1e-5) per horizon, from `rich` (magA) or `rich` + 3 volatility-level
features (magB: log sd/close, log mean range/close, log1p mean volume). The magnitude-filtered honest tail: thresholds from the first
half of val (median predicted |r| of h1, then the top-q P(up) confidence among those windows), applied to the second half after
the lab's gap.

usage: python longtrain.py --slice-index 0 [--spans 11.5d,90d,365d,3y] [--steps L0,L5,L6,L7]  (one slice -> parts/<slice>.json)
       python longtrain.py --merge | --report"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import argparse, glob, json, math, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ladder  # noqa: E402  (imports lab, chdirs to the repo root)
import lab  # noqa: E402
import tensorflow as tf  # noqa: E402
from numpy.lib.stride_tricks import sliding_window_view  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.linear_model import LogisticRegression, Ridge  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

CSV = "D:/nt/neural_trade/Bitcoin_BTCUSDT.csv"
PARTS, OUT = f"{HERE}/longtrain_parts", f"{HERE}/longtrain_results.jsonl"
SPANS = {"11.5d": 16560, "90d": 129600, "365d": 525600, "3y": 1576800}   # windows (1-minute stride)
HZ = (10, 15, 20); GAPB = 81; CAP_LIN, CAP_SEQ = 130_000, 45_000; CHUNK = 20_000
NT = int(os.environ.get("LT_THREADS", "4"))


def lowprio():
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass


def load_series():
    import pandas as pd
    return pd.read_csv(CSV, usecols=["open", "high", "low", "close", "volume"]).to_numpy(np.float64)


def windows(A, s0, n):
    """[n,60,5] windows starting at bars s0 .. s0+n-1 (a view -> copy per chunk by the caller)."""
    return sliding_window_view(A[s0:s0 + n + 59], 60, axis=0).transpose(0, 2, 1)


def locate(A, Wva):
    w = Wva[0].astype(np.float64)
    for i in np.where(np.abs(A[:, 3] - w[0, 3]) < 1e-2)[0]:
        if i + 60 <= len(A) and np.allclose(A[i:i + 60], w, rtol=1e-5, atol=1e-6):
            return int(i)
    raise RuntimeError("val block not found in the series")


def level_feats(Wf):
    o, h, l, c, v, d, sd = lab.parts(Wf)
    return np.stack([np.log(sd / c[:, -1] + 1e-12), np.log(np.mean(h - l, 1) / c[:, -1] + 1e-12), np.log1p(v.mean(1))], 1)


def feats(Wf):
    """float32 window chunk -> tb7 [n,13], rich [n,44], level [n,3] (float32, nan_to_num as the ladder)."""
    return tuple(np.nan_to_num(f).astype(np.float32) for f in (lab.fs_tb7(Wf), lab.fs_rich(Wf), level_feats(Wf)))


def build_long(A, s0, n):
    """Features of windows s0..s0+n-1 in chunks, and their labels close[last+h]/close[last]-1."""
    T, R, G = [], [], []
    for i in range(0, n, CHUNK):
        k = min(CHUNK, n - i); Wf = np.ascontiguousarray(windows(A, s0 + i, k)).astype(np.float32)
        a, b, g = feats(Wf); T.append(a); R.append(b); G.append(g)
    last = s0 + 59 + np.arange(n)
    ret = np.stack([A[last + h, 3] / A[last, 3] - 1 for h in HZ], 1).astype(np.float32)
    return np.concatenate(T), np.concatenate(R), np.concatenate(G), ret


# ------------------------------------------------------------------ the steps on one (slice, span)
def strided(n, cap):
    k = max(1, math.ceil(n / cap)); return np.arange(n)[::-1][::k][::-1], k       # keep the most recent windows


def make_S(F, ret, db, Fva, rows, scaler):
    Ftr = scaler.transform(F[rows]).astype(np.float32); r = ret[rows]
    n = len(rows); k = 1
    cut = int(round((1 - ladder.INNER) * n))
    return dict(Ftr=Ftr, Fva=scaler.transform(Fva).astype(np.float32), Y=(r > 0).astype(np.float32),
                M=(np.abs(r) > db).astype(np.float32), itr=np.arange(0, cut - ladder.GAP), iva=np.arange(cut, n), rtr=r)


def l7(S, Q, Qva, l5):
    """ladder.residual_step(seq=True) with the L5 stage handed in (identical logic)."""
    Cs, inner, full, l5model = l5; F, Y, M, itr, iva = S["Ftr"], S["Y"], S["M"], S["itr"], S["iva"]
    Xs = [F, Q]; tf.keras.utils.set_random_seed(0)
    model, lin = ladder.make_model(F.shape[1], seq=True); lin.set_weights(inner)
    be, _ = ladder.fit(model, lin, [x[itr] for x in Xs], Y[itr], M[itr], ladder.lam_vec(Cs, M[itr]), lr=ladder.NL_LR,
                       epochs=ladder.NL_EPOCHS, batch=ladder.NL_BATCH, val=([x[iva] for x in Xs], Y[iva], M[iva]),
                       patience=ladder.NL_PATIENCE, decay=False)
    if be == 0:
        return ladder.predict(l5model, [S["Fva"]]), be
    tf.keras.utils.set_random_seed(0)
    model, lin = ladder.make_model(F.shape[1], seq=True); lin.set_weights(full)
    ladder.fit(model, lin, Xs, Y, M, ladder.lam_vec(Cs, M), lr=ladder.NL_LR, epochs=be, batch=ladder.NL_BATCH, decay=False)
    return ladder.predict(model, [S["Fva"], Qva]), be


def l6(S, l5):
    Cs, inner, full, l5model = l5; F, Y, M, itr, iva = S["Ftr"], S["Y"], S["M"], S["itr"], S["iva"]
    tf.keras.utils.set_random_seed(0)
    model, lin = ladder.make_model(F.shape[1], mlp=True); lin.set_weights(inner)
    be, _ = ladder.fit(model, lin, [F[itr]], Y[itr], M[itr], ladder.lam_vec(Cs, M[itr]), lr=ladder.NL_LR, epochs=ladder.NL_EPOCHS,
                       batch=ladder.NL_BATCH, val=([F[iva]], Y[iva], M[iva]), patience=ladder.NL_PATIENCE, decay=False)
    if be == 0:
        return ladder.predict(l5model, [S["Fva"]]), be
    tf.keras.utils.set_random_seed(0)
    model, lin = ladder.make_model(F.shape[1], mlp=True); lin.set_weights(full)
    ladder.fit(model, lin, [F], Y, M, ladder.lam_vec(Cs, M), lr=ladder.NL_LR, epochs=be, batch=ladder.NL_BATCH, decay=False)
    return ladder.predict(model, [S["Fva"]]), be


def mag_model(Xtr, ret, Xva):
    """Ridge per horizon on log(|r| + 1e-5); alpha on the last 15% of train. Returns predicted log|r| on val [N,3]."""
    Y = np.log(np.abs(ret) + 1e-5); n = len(Xtr); cut = int(round(0.85 * n)); P = np.zeros((len(Xva), 3))
    for h in range(3):
        best = min(((float(np.mean((Ridge(alpha=a).fit(Xtr[:cut - ladder.GAP], Y[:cut - ladder.GAP, h]).predict(Xtr[cut:]) - Y[cut:, h]) ** 2)), a)
                    for a in (1.0, 100.0, 1e4)))
        P[:, h] = Ridge(alpha=best[1]).fit(Xtr, Y[:, h]).predict(Xva)
    return P


def mag_tail(rva, db, P, mag, rng):
    """Honest tail restricted to windows with predicted |r| (h1) above the median of the first half of val."""
    n = len(P); A = np.zeros(n, bool); A[:n // 2] = True; B = np.zeros(n, bool); B[n // 2 + lab.GAP:] = True
    s = P.mean(1); conf = np.abs(s - 0.5); ret = rva[:, 1]; m = np.abs(ret) > db; mg = mag[:, 1]
    hi = mg >= np.median(mg[A]); out = {}
    for cov in (10, 5):
        thr = np.quantile(conf[A & hi], 1 - cov / 100); k = B & hi & (conf >= thr) & m
        mv = np.sign(s[k] - 0.5) * ret[k] * 1e4
        out[f"mon{cov}_hit"] = float(np.mean(mv > 0)) if k.sum() else float("nan")
        out[f"mon{cov}_bps"] = float(np.mean(mv)) if k.sum() else float("nan")
        out[f"mon{cov}_n"] = int(k.sum())
        null = [np.mean(np.sign(s[k] - 0.5) * rng.choice([-1, 1], k.sum()) * ret[k] * 1e4) for _ in range(300)] if k.sum() else [float("nan")]
        out[f"mon{cov}_null95"] = float(np.quantile(null, 0.95))
    return out


def mag_skill(rva, mag):
    return {f"magrho_h{h}": float(stats.spearmanr(mag[:, h], np.log(np.abs(rva[:, h]) + 1e-5))[0]) for h in range(3)}


def score(D, rtr_used, P, rng):
    return {k: float(v) for k, v in lab.evaluate_slice(dict(deadband=D["deadband"], rtr=rtr_used, rva=D["rva"]), P, rng).items()}


# ------------------------------------------------------------------ one slice
def run_slice(path, spans, steps, A, log):
    t0 = time.time(); D = np.load(path); db = float(D["deadband"]); name = os.path.basename(path)[:-4]; rva = D["rva"]
    Wva = D["Wva"]; vs = locate(A, Wva); nva = len(Wva)
    assert np.allclose(windows(A, vs, nva).astype(np.float32), Wva, rtol=1e-5, atol=1e-6), "Wva differs from the series"
    lastv = vs + 59 + np.arange(nva)
    rv = np.stack([A[lastv + h, 3] / A[lastv, 3] - 1 for h in HZ], 1)
    assert np.allclose(rv, rva, rtol=1e-4, atol=2e-6), "rva differs from the series"
    avail = vs - 80                                    # window starts 0 .. vs-81
    nmax = min(max(SPANS[s] for s in spans), avail); s0 = vs - GAPB - (nmax - 1)
    Ttb, Rrich, Glev, RET = build_long(A, s0, nmax)
    log(f"{name}: val start {vs}, available {avail} windows, built {nmax} in {time.time() - t0:.0f}s")
    rng = np.random.default_rng(0); fva_tb = np.nan_to_num(lab.fs_tb7(Wva)); fva_rich = np.nan_to_num(lab.fs_rich(Wva))
    fva_lev = np.nan_to_num(level_feats(Wva.astype(np.float32)))
    recs = []
    for sp in spans:
        n = min(SPANS[sp], avail); lo = nmax - n
        if sp == "11.5d":                              # must reproduce the cache's train
            Wtr = D["Wtr"]; assert len(Wtr) == n or n < SPANS[sp], (len(Wtr), n)
            nn = len(Wtr); assert np.allclose(windows(A, vs - GAPB - (nn - 1), nn).astype(np.float32), Wtr, rtol=1e-5, atol=1e-6), "Wtr differs"
            assert np.allclose(RET[-nn:], D["rtr"], rtol=1e-4, atol=2e-6), "rtr differs"
            n = nn; lo = nmax - n
        base = dict(slice=name, span=sp, n_windows=int(n), days=round(n / 1440, 1), full=bool(n == SPANS[sp]))
        rows, k = strided(n, CAP_LIN); rows = rows + lo
        ret = RET[rows]
        sc_tb = StandardScaler().fit(Ttb[rows]); sc_r = StandardScaler().fit(Rrich[rows])
        S_tb = make_S(Ttb, RET, db, fva_tb, rows, sc_tb); S_r = make_S(Rrich, RET, db, fva_rich, rows, sc_r)
        common = dict(base, stride=int(k), n_rows=int(len(rows)))
        if "L0" in steps:
            t1 = time.time(); P = np.zeros((nva, 3))
            for i in range(3):
                m = S_tb["M"][:, i] > 0
                P[:, i] = LogisticRegression(C=ladder.C0, max_iter=1000).fit(S_tb["Ftr"][m], S_tb["Y"][m, i].astype(int)).predict_proba(S_tb["Fva"])[:, 1]
            recs.append(dict(common, step="L0", fit_s=round(time.time() - t1, 1), **score(D, ret, P, rng)))
            if k > 1:                                  # L0 on every window
                allr = np.arange(lo, nmax); sc2 = StandardScaler().fit(Ttb[allr]); S2 = make_S(Ttb, RET, db, fva_tb, allr, sc2)
                t1 = time.time(); P = np.zeros((nva, 3))
                for i in range(3):
                    m = S2["M"][:, i] > 0
                    P[:, i] = LogisticRegression(C=ladder.C0, max_iter=1000).fit(S2["Ftr"][m], S2["Y"][m, i].astype(int)).predict_proba(S2["Fva"])[:, 1]
                recs.append(dict(base, step="L0full", stride=1, n_rows=int(len(allr)), fit_s=round(time.time() - t1, 1), **score(D, RET[allr], P, rng)))
                del S2
        l5 = None
        if steps & {"L5", "L6", "L7"}:
            tf.keras.utils.set_random_seed(0); t1 = time.time(); l5 = ladder.l5_stage(S_r); l5_s = time.time() - t1
            P5 = ladder.predict(l5[3], [S_r["Fva"]]); C = [float(c) for c in l5[0]]
            recs.append(dict(common, step="L5", fit_s=round(l5_s, 1), C=C, **score(D, ret, P5, rng)))
            # magnitude read on the L5 predictions (the model the filter would sit on), plus on L0's P(up)
            for tag, X, Xv in (("magA", S_r["Ftr"], S_r["Fva"]),
                               ("magB", np.hstack([S_r["Ftr"], StandardScaler().fit(Glev[rows]).transform(Glev[rows])]),
                                np.hstack([S_r["Fva"], StandardScaler().fit(Glev[rows]).transform(fva_lev)]))):
                t1 = time.time(); mag = mag_model(X, ret, Xv); ms = round(time.time() - t1, 1)
                recs.append(dict(common, step="L5", mag=tag, fit_s=ms, **mag_skill(rva, mag), **mag_tail(rva, db, P5, mag, rng)))
                if tag == "magB":
                    magB = mag
        if "L6" in steps:
            t1 = time.time(); P, be = l6(S_r, l5)
            recs.append(dict(common, step="L6", fit_s=round(time.time() - t1 + l5_s, 1), best_epoch=int(be), **score(D, ret, P, rng)))
            recs.append(dict(common, step="L6", mag="magB", fit_s=0.0, **mag_tail(rva, db, P, magB, rng)))
        if "L7" in steps:
            sub, k7 = strided(len(rows), CAP_SEQ)        # a stride over the L5 rows
            S7 = dict(S_r); S7["Ftr"], S7["Y"], S7["M"] = S_r["Ftr"][sub], S_r["Y"][sub], S_r["M"][sub]
            n7 = len(sub); cut = int(round((1 - ladder.INNER) * n7)); S7["itr"], S7["iva"] = np.arange(0, cut - ladder.GAP), np.arange(cut, n7)
            st = s0 + rows[sub]                          # absolute window starts of the L7 rows
            Q = np.concatenate([ladder.seq_channels(A[st[i:i + 5000, None] + np.arange(60)].astype(np.float32)) for i in range(0, n7, 5000)])
            mu, sdq = Q.mean((0, 1)), Q.std((0, 1)) + 1e-6
            Qva = (ladder.seq_channels(Wva) - mu) / sdq; Q = (Q - mu) / sdq
            t1 = time.time(); P, be = l7(S7, Q, Qva, l5)
            recs.append(dict(common, step="L7", n_rows=int(n7), stride=int(k * k7), fit_s=round(time.time() - t1 + l5_s, 1), best_epoch=int(be),
                             **score(D, ret, P, rng)))
            recs.append(dict(common, step="L7", mag="magB", n_rows=int(n7), stride=int(k * k7), fit_s=0.0, **mag_tail(rva, db, P, magB, rng)))
            del Q, Qva, S7
        log(f"{name} {sp}: n {n} done at {time.time() - t0:.0f}s")
    os.makedirs(PARTS, exist_ok=True)
    json.dump(recs, open(f"{PARTS}/{name}.json", "w"))
    return recs


# ------------------------------------------------------------------ merge and report
def merge():
    rows = []
    for p in sorted(glob.glob(f"{PARTS}/*.json")):
        rows += json.load(open(p))
    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(len(rows), "records ->", OUT)


def tci(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]; m = v.mean()
    if len(v) < 2:
        return m, m, m
    h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / math.sqrt(len(v)); return m, m - h, m + h


def report(only_full=False):
    R = [json.loads(l) for l in open(OUT, encoding="utf-8")]
    sel = lambda step, span, mag=None: {r["slice"]: r for r in R if r["step"] == step and r["span"] == span and r.get("mag") == mag}
    spans = [s for s in SPANS if any(r["span"] == s for r in R)]
    fullset = {sp: {r["slice"] for r in R if r["span"] == sp and r["full"]} for sp in spans}
    for subset in ("all 24 slices", "slices with the full span only"):
        print(f"\n== {subset} ==")
        print("span | step | n_sl | mean n_win | mean3 AUC [95% CI] | d vs L0 same span [CI] | d vs same step 11.5d [CI] | "
              "top10 hit | bps (null95) | mag-filtered top10 hit | bps (null95) | n trades | fit s")
        for sp in spans:
            for st in ("L0", "L0full", "L5", "L6", "L7"):
                d = sel(st, sp)
                if not d:
                    continue
                keys = sorted(d) if subset.startswith("all") else sorted(k for k in d if k in fullset[sp])
                if len(keys) < 2:
                    continue
                a = np.array([d[k]["auc3"] for k in keys]); m = tci(a)
                l0 = sel("L0", sp); d0 = tci(a - np.array([l0[k]["auc3"] for k in keys])) if st != "L0" else None
                b = sel(st, "11.5d"); dd = tci(a - np.array([b[k]["auc3"] for k in keys])) if (st in b or b) and sp != "11.5d" and all(k in b for k in keys) else None
                md = sel(st, sp, "magB") if st in ("L5", "L6", "L7") else {}
                mk = [k for k in keys if k in md]
                fm = lambda x: "-" if x is None else "%+.4f [%+.4f,%+.4f]" % x
                print(f"{sp} | {st} | {len(keys)} | {np.mean([d[k]['n_windows'] for k in keys]):.0f} | {m[0]:.4f} [{m[1]:.4f},{m[2]:.4f}] | {fm(d0)} | {fm(dd)} | "
                      f"{np.mean([d[k]['hon10_hit'] for k in keys]):.4f} | {np.mean([d[k]['hon10_bps'] for k in keys]):+.2f} ({np.mean([d[k]['hon10_null95'] for k in keys]):+.2f}) | "
                      + (f"{np.nanmean([md[k]['mon10_hit'] for k in mk]):.4f} | {np.nanmean([md[k]['mon10_bps'] for k in mk]):+.2f} ({np.nanmean([md[k]['mon10_null95'] for k in mk]):+.2f}) | "
                         f"{np.mean([md[k]['mon10_n'] for k in mk]):.0f}" if mk else "- | - | -") + f" | {np.mean([d[k]['fit_s'] for k in keys]):.1f}")
    print("\nmagnitude skill (Spearman of predicted vs realised log|r|, val, mean over slices) and the mag read on L5, by variant:")
    for sp in spans:
        for tag in ("magA", "magB"):
            d = sel("L5", sp, tag)
            if d:
                print(f"{sp} {tag}: rho h0/h1/h2 " + "/".join(f"{np.mean([r[f'magrho_h{h}'] for r in d.values()]):.3f}" for h in range(3)) +
                      f" | top10 mag-filtered hit {np.nanmean([r['mon10_hit'] for r in d.values()]):.4f} bps {np.nanmean([r['mon10_bps'] for r in d.values()]):+.2f} "
                      f"(null95 {np.nanmean([r['mon10_null95'] for r in d.values()]):+.2f}) | top5 hit {np.nanmean([r['mon5_hit'] for r in d.values()]):.4f} bps {np.nanmean([r['mon5_bps'] for r in d.values()]):+.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--slice-index", type=int, nargs="*"); ap.add_argument("--spans", default="11.5d,90d,365d,3y")
    ap.add_argument("--steps", default="L0,L5,L6,L7"); ap.add_argument("--merge", action="store_true"); ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.merge:
        merge()
    elif a.report:
        report()
    else:
        lowprio(); tf.config.threading.set_intra_op_parallelism_threads(NT); tf.config.threading.set_inter_op_parallelism_threads(2)
        paths = sorted(glob.glob(lab.CACHE)); A = load_series()
        for i in a.slice_index:
            run_slice(paths[i], a.spans.split(","), set(a.steps.split(",")), A, lambda s: print(s, flush=True))
