"""O1 runner (SPEC.md). Stages: features (cache), walk-forward months + frozen forward (results.jsonl), tables (tables.txt), interp.
usage: run_o1.py [run|tables|interp] [--limit N months]"""
import os, sys, json, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import o1_lib as L  # noqa: E402
from scipy import stats  # noqa: E402
T = L.T
OUT = L.HERE; RES = os.path.join(OUT, "results.jsonl"); LOG = os.path.join(OUT, "run.log")


def log(*a):
    msg = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def nice():
    from threadpoolctl import threadpool_limits; threadpool_limits(limits=4)     # BLAS threads: many starve at low priority
    try:
        import psutil; psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass


def free_gb():
    import psutil; return psutil.virtual_memory().available / 2 ** 30


def load_cache(name, loader, first):
    os.makedirs(L.SCRATCH, exist_ok=True); p = os.path.join(L.SCRATCH, name + ".npz")
    if os.path.exists(p):
        z = np.load(p); return {k: z[k] for k in z.files}
    assert free_gb() >= 8, "less than 8 GB free RAM"
    t0 = time.time(); s = loader(); log(name, "bars", len(s), s.index.min(), s.index.max())
    F = L.build_features(s, first); np.savez(p, **F); log(name, "features", len(F["ta"]), "anchors", f"{time.time() - t0:.0f}s")
    return F


def load_train_bars():
    return T.load_bars(T.TRAIN_CSV, L.LOAD_FROM, T.CUTOFF)


def done_keys():
    if not os.path.exists(RES):
        return set()
    return {(r["span"], r["month"], r["H"]) for r in map(json.loads, open(RES, encoding="utf-8"))}


def write(rows):
    with open(RES, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def run(limit=None):
    nice(); Ftr = load_cache("train", load_train_bars, T.WARM)
    Ffw = load_cache("fwd", lambda: T.load_bars(T.FWD_CSV, T.FWD_START, T.FWD_LOAD_END), L.FWD_FIRST)
    assert pd.Timestamp(int(Ftr["ta"].max()), unit="ns") <= T.CUTOFF + pd.Timedelta(minutes=1)
    assert pd.Timestamp(int(Ffw["ta"].min()), unit="ns") >= T.FWD_START
    done = done_keys(); t_all = time.time()
    months = [("sel", m) for m in L.M_SEL] + [("look", m) for m in L.M_LOOK]
    if limit:
        months = months[:limit]
    for span, mo in months:
        p = pd.Period(mo, "M"); m0, m1 = p.start_time, (p + 1).start_time; t0 = time.time()
        for H in L.HORIZONS:
            if (span, mo, H) in done:
                continue
            st = L.fit_state(Ftr, H, m0, groups=(H == 60))
            te = L.test_rows(Ftr["ta"], Ftr[f"ok{H}"], m0, m1)
            assert (Ftr["ta"][st["tr"]].astype("datetime64[ns]") + np.timedelta64(H, "m") <= np.datetime64(m0, "ns")).all()
            write(L.score_state(st, Ftr, te, span, mo)); del st
        log(span, mo, f"{time.time() - t0:.0f}s", f"free {free_gb():.0f} GB")
    if limit:
        return
    # frozen fit on the 365 days ending at the cutoff, scored on the forward months
    for H in L.HORIZONS:
        if all(("fwd", m, H) in done for m in T.FWD_MONTHS):
            continue
        t0 = time.time(); st = L.fit_state(Ftr, H, L.TRAIN_END, groups=(H == 60))
        assert (Ftr["ta"][st["tr"]].astype("datetime64[ns]") + np.timedelta64(H, "m") <= np.datetime64(L.TRAIN_END, "ns")).all()
        if H == 60:
            np.savez(os.path.join(OUT, "frozen_coefs_h60.npz"),
                     **{f"{nm}_{k}": st["m"][("lnrv", "full", nm)][1][k] for nm in ("R1", "R2") for k in ("idx", "w")})
        for mo in T.FWD_MONTHS:
            p = pd.Period(mo, "M"); te = L.test_rows(Ffw["ta"], Ffw[f"ok{H}"], p.start_time, (p + 1).start_time)
            if len(te) == 0:
                log("fwd", mo, H, "no rows"); continue
            write(L.score_state(st, Ffw, te, "fwd", mo))
        log("frozen H", H, f"{time.time() - t0:.0f}s"); del st
    log("total", f"{time.time() - t_all:.0f}s")


# ------------------------------------------------------------------ tables
def load_rows():
    return pd.DataFrame([json.loads(x) for x in open(RES, encoding="utf-8")])


def series(df, span, H, model, col, target="lnrv"):
    d = df[(df.span == span) & (df.H == H) & (df.model == model) & (df.target == target)].set_index("month")[col]
    return d.astype(float)


def paired(df, span, H, a, b, col, target="lnrv"):
    """a - b per month: mean, CI, months positive, n, two-sided t p."""
    x = series(df, span, H, a, col, target); y = series(df, span, H, b, col, target); d = (x - y).dropna()
    mu, lo, hi, t = T.ci(d.values); p = 2 * stats.t.sf(abs(t), len(d) - 1)
    return {"mean": mu, "lo": lo, "hi": hi, "t": t, "n_pos": int((d > 0).sum()), "n_neg": int((d < 0).sum()), "n": int(len(d)), "p": float(p)}


def decide_gstar(df):
    g1 = paired(df, "sel", 60, "R1", "R0", "spearman")["mean"]; g2 = paired(df, "sel", 60, "R2", "R0", "spearman")["mean"]
    return ("R2" if g2 - g1 >= 0.002 else "R1"), g1, g2


def holm(ps):
    o = sorted(range(len(ps)), key=lambda i: ps[i]); adj = [0] * len(ps); run_ = 0
    for r, i in enumerate(o):
        run_ = max(run_, min(1, (len(ps) - r) * ps[i])); adj[i] = run_
    return adj


def criterion(df, span, gs):
    s = paired(df, span, 60, gs, "R0", "spearman"); q = paired(df, span, 60, gs, "R0", "qlike")
    ok = (s["mean"] >= 0.005) and (s["lo"] > 0) and (q["mean"] <= 0)
    return ok, s, q


def fmt(d, nd=4):
    return f"{d['mean']:+.{nd}f} [{d['lo']:+.{nd}f},{d['hi']:+.{nd}f}] +{d['n_pos']}/{d['n']}"


def top_coefs(gs, k=15):
    z = np.load(os.path.join(OUT, "frozen_coefs_h60.npz")); idx, w = z[f"{gs}_idx"], z[f"{gs}_w"]; names = L.feature_names(); n3 = len(L.S3_NAMES)
    rows = [(int(i), float(c)) for i, c in zip(idx, w) if i >= n3]
    rows.sort(key=lambda r: -abs(r[1]))
    return [{"col": i, "name": names[i], "coef": c, "channel": names[i].split(".")[1], "group": L.group_of(names[i].split(".")[1])}
            for i, c in rows[:k]]


def tables():
    df = load_rows(); out = []; P = out.append
    gs, g1, g2 = decide_gstar(df)
    P(f"O1 tables. Months: sel {df[df.span=='sel'].month.nunique()}, look {df[df.span=='look'].month.nunique()}, "
      f"fwd {df[df.span=='fwd'].month.nunique()}")
    P(f"\nSELECTION: mean Spearman gain over R0 at H=60: R1 {g1:+.4f}, R2 {g2:+.4f}; R2 - R1 = {g2-g1:+.4f} (R2 needs >= +0.002) -> G* = {gs}")
    sh = paired(df, "sel", 60, "Rsh", "R0", "spearman"); gate = abs(sh["mean"]) <= 0.003
    P(f"SANITY GATE: Rsh - R0 mean Spearman at H=60 on selection months = {sh['mean']:+.4f} (limit +-0.003): {'PASS' if gate else 'FAIL (broken)'}")
    for tgt, title in (("lnrv", "PRIMARY target ln RV(a,a+H)"), ("absr", "SECONDARY target ln(|r_H|+1e-5) (ridge only)")):
        P(f"\n===== {title} =====")
        for span in ("sel", "look", "fwd"):
            for H in L.HORIZONS:
                P(f"\n-- span {span}  H={H}  target {tgt} --")
                models = ["R0", "R1", "R2", "Rsh"] + (["H0", "H2"] if tgt == "lnrv" else [])
                P(f"{'model':6s} {'Spearman':>9s} {'QLIKE':>9s}")
                for m in models:
                    sp = series(df, span, H, m, "spearman", tgt); ql = series(df, span, H, m, "qlike", tgt)
                    P(f"{m:6s} {sp.mean():9.4f} {ql.mean():9.4f}   (n months {len(sp)})")
                P("paired vs R0 (mean [95% t-CI] +months positive/n):  Spearman gain | QLIKE diff (negative = better)")
                for m in [x for x in models if x != "R0"] + (["H2-H0"] if tgt == "lnrv" else []):
                    a, b = ("H2", "H0") if m == "H2-H0" else (m, "R0")
                    s = paired(df, span, H, a, b, "spearman", tgt); q = paired(df, span, H, a, b, "qlike", tgt)
                    P(f"  {m:6s}{'(vs H0)' if m=='H2-H0' else '       '} {fmt(s)}  |  {fmt(q, 4)} (better in {q['n_neg']}/{q['n']})")
    P("\n===== PRE-REGISTERED CRITERION (G* vs R0 at H=60, primary target) =====")
    res = {}
    for span in ("sel", "look", "fwd"):
        ok, s, q = criterion(df, span, gs); res[span] = ok
        P(f"{span:5s} G*={gs}: Spearman gain {fmt(s)}; QLIKE diff {fmt(q)}; gain>=+0.005 {s['mean']>=0.005}, lower>0 {s['lo']>0}, "
          f"QLIKE<=0 {q['mean']<=0} -> {'PASS' if ok else 'FAIL'}")
    if not gate:
        verdict = "broken"
    elif not res["look"]:
        verdict = "no"
    elif res["fwd"]:
        verdict = "holds"
    else:
        verdict = "look only"
    P(f"\nVERDICT: {verdict}")
    P("\n===== SECONDARY H=15 and 240 (G* vs R0 Spearman gain; Holm over the two within each span; not part of the verdict) =====")
    for span in ("sel", "look", "fwd"):
        ss = [paired(df, span, H, gs, "R0", "spearman") for H in (15, 240)]; adj = holm([x["p"] for x in ss])
        for H, x, ad in zip((15, 240), ss, adj):
            P(f"{span:5s} H={H:3d}: {fmt(x)}  p={x['p']:.4f} Holm p={ad:.4f}")
    P("\n===== INTERPRETATION =====")
    P("Incremental Spearman of each channel group (its G1 columns) added alone to S3, vs R0, H=60, selection months:")
    for g in L.GROUPS:
        P(f"  {g:12s} {fmt(paired(df, 'sel', 60, 'grp:' + g, 'R0', 'spearman'))}")
    P(f"  {'all (G*)':12s} {fmt(paired(df, 'sel', 60, gs, 'R0', 'spearman'))}")
    P("\nTop 15 standardised geometry coefficients of the frozen G* at H=60 (ridge on ln RV; + means higher forecast RV):")
    for r in top_coefs(gs):
        P(f"  {r['coef']:+.4f}  {r['name']:34s} channel {r['channel']:10s} group {r['group']}")
    open(os.path.join(OUT, "tables.txt"), "w", encoding="utf-8").write("\n".join(out) + "\n")
    print("\n".join(out))


def interp():
    gs, _, _ = decide_gstar(load_rows()); top = top_coefs(gs); F = load_cache("train", load_train_bars, T.WARM)
    s = load_train_bars(); A = s.values.astype(np.float64); idx = s.index
    ta = F["ta"].astype("datetime64[ns]"); sel = np.where((ta >= np.datetime64("2021-01-01")) & (ta < np.datetime64("2024-07-01")) & F["ok60"])[0]
    X = np.concatenate([F["S3"], F["G1"], F["G15"]], 1)
    res = {"gstar": gs, "top": top, "examples": []}
    for r in top[:3]:
        v = X[sel, r["col"]]; order = np.argsort(v); picks = [order[int(q * (len(order) - 1))] for q in (0.02, 0.10, 0.40, 0.60, 0.90, 0.98)]
        ex = []
        for pi in picks:
            a = int(T.WARM + 7 * sel[pi]); is15 = r["name"].startswith("g15.")
            W = L.agg15(A, [a])[0] if is15 else A[a - 60:a]
            ts = [str(idx[a - 900 + 15 * k]) for k in range(60)] if is15 else [str(t) for t in idx[a - 60:a]]
            assert pd.Timestamp(ts[-1]) < pd.Timestamp("2024-07-01")
            ex.append({"anchor_time": str(idx[a - 1] + pd.Timedelta(minutes=1)), "value": float(v[pi]), "scale": "15min" if is15 else "1min",
                       "bar_times": ts, "ohlcv": W.tolist()})
        res["examples"].append({"feature": r["name"], "coef": r["coef"], "windows": ex})
    json.dump(res, open(os.path.join(OUT, "interp.json"), "w"))
    log("interp.json written", [e["feature"] for e in res["examples"]])


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    lim = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    {"run": lambda: run(lim), "tables": tables, "interp": interp}[cmd]()
