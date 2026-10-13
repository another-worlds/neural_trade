"""O5 runner (SPEC.md). usage: run_o5.py [run|tables]. Order: frame (cache) -> selection (H=60: S3, L0, N0) -> base choice B* -> judgement span (all H,
5 models) -> tables.txt. Resumable through results.jsonl."""
import json, os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import o5_lib as P  # noqa: E402
O3 = P.O3; T = P.T
RES = os.path.join(HERE, "results.jsonl"); LOG = os.path.join(HERE, "run.log"); CHOICE = os.path.join(HERE, "choice.json")


def log(*a):
    msg = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def nice():
    from threadpoolctl import threadpool_limits; threadpool_limits(limits=4)
    try:
        import psutil; psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass


def free_gb():
    import psutil; return psutil.virtual_memory().available / 2 ** 30


def load_frame():
    os.makedirs(P.SCRATCH, exist_ok=True); p = os.path.join(P.SCRATCH, "frame.npz")
    if os.path.exists(p):
        z = np.load(p); return {k: z[k] for k in z.files}
    assert free_gb() >= 8, "less than 8 GB free RAM"
    t0 = time.time(); s = T.load_bars(T.TRAIN_CSV, P.LOAD_FROM, P.LOAD_HI)
    assert s.index.max() <= P.LOAD_HI < T.CUTOFF
    gaps = s.index.to_series().diff().dropna(); log("bars", len(s), s.index.min(), s.index.max(), "non-1-minute steps", int((gaps != pd.Timedelta(minutes=1)).sum()))
    F = P.build_frame(s, P.FIRST5); del s
    log("frame anchors", len(F["ta"]), pd.Timestamp(F["ta"][0]), pd.Timestamp(F["ta"][-1]), f"{time.time() - t0:.0f}s",
        "G finite", bool(np.isfinite(F["G"]).all()), "XB finite", bool(np.isfinite(F["XB"]).all()))
    np.savez(p, **F); return F


def done_keys():
    if not os.path.exists(RES):
        return set()
    return {(r["span"], r["month"], r["H"]) for r in map(json.loads, open(RES, encoding="utf-8"))}


def write(rows):
    with open(RES, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def rows_df():
    return pd.DataFrame([json.loads(x) for x in open(RES, encoding="utf-8")])


def run_span(F, span, months, Hs, models, done):
    for mo in months:
        m0, m1 = P.month_bounds(mo); t0 = time.time()
        for H in Hs:
            if (span, mo, H) in done:
                continue
            st = P.fit_h(F, H, m0, models=models); te = O3.test_rows(F["ta"], F[f"ok{H}"], m0, m1)
            assert len(te) > 0 and pd.Timestamp(st["last_label_end"]) <= m0
            rows = P.score(st, F, te, span, mo)
            for r in rows:
                r["train_days"] = round(float((m0 - pd.Timestamp(st["first_ta"])).total_seconds() / 86400), 1)
            write(rows)
        log(span, mo, f"{time.time() - t0:.0f}s", f"free {free_gb():.0f} GB")


def choose_base():
    df = rows_df(); d = df[(df.span == "sel") & (df.H == 60)]
    sp = {m: float(d[d.model == m].spearman.mean()) for m in ("S3", "L0", "N0")}
    ql = {m: float(d[d.model == m].qlike.mean()) for m in ("S3", "L0", "N0")}
    base = "N0" if sp["N0"] > sp["L0"] else "L0"
    ch = {"base": base, "candidate": {"L0": "L1", "N0": "N1"}[base], "sel_spearman_h60": sp, "sel_qlike_h60": ql,
          "n_months": int(d[d.model == "L0"].month.nunique())}
    json.dump(ch, open(CHOICE, "w"), indent=1); return ch


def run():
    nice(); t_all = time.time(); F = load_frame(); done = done_keys()
    assert pd.Timestamp(int(F["ta"].max()) if F["ta"].dtype.kind == "i" else F["ta"].max()) <= P.LOAD_HI
    run_span(F, "sel", P.S_MONTHS, (60,), ("S3", "L0", "N0"), done)
    ch = choose_base(); log("BASE CHOICE", ch)
    run_span(F, "judge", P.J_MONTHS, P.HORIZONS, tuple(P.SPECS), done)
    log("total", f"{time.time() - t_all:.0f}s")
    json.dump({"runtime_s": time.time() - t_all}, open(os.path.join(HERE, "runtime.json"), "w"))


def fmt(d, nd=4):
    return "n/a" if d is None else f"{d['mean']:+.{nd}f} [{d['lo']:+.{nd}f},{d['hi']:+.{nd}f}] +{d['pos']}/{d['n']} (min {d['min']:+.{nd}f}, max {d['max']:+.{nd}f})"


def tables():
    df = rows_df(); ch = json.load(open(CHOICE)); base, cand = ch["base"], ch["candidate"]; out = []; Q = out.append
    sel = df[df.span == "sel"]; jd = df[df.span == "judge"]
    Q(f"O5 tables. selection months {sel.month.nunique()}, judgement months {jd.month.nunique()} ({jd.month.min()}..{jd.month.max()})")
    Q("\n===== BASE CHOICE on the selection months 2021-01..2024-06, H=60 (mean Spearman; the better of L0 and N0) =====")
    for m in ("S3", "L0", "N0"):
        Q(f"  {m:3s} Spearman {ch['sel_spearman_h60'][m]:.4f}  QLIKE {ch['sel_qlike_h60'][m]:.4f}")
    Q(f"  B* = {base} ({'BANK ridge' if base == 'L0' else 'HGB on S3+bank'}); candidate = {cand}; N0 - L0 = {ch['sel_spearman_h60']['N0'] - ch['sel_spearman_h60']['L0']:+.4f}")
    mo = P.J_MONTHS
    for H in P.HORIZONS:
        Q(f"\n===== JUDGEMENT 2018-07..2020-12  H={H}: mean Spearman | mean QLIKE =====")
        d = jd[jd.H == H]
        for m in ("S3", "L0", "L1", "N0", "N1"):
            x = d[d.model == m]; Q(f"  {m:3s} {x.spearman.mean():.4f} | {x.qlike.mean():.4f}   (n months {x.month.nunique()})")
        Q("paired (a - b), per month: mean [95% t-CI] +positive/n (min, max)   |  QLIKE diff (negative = better)")
        for a, b in dict.fromkeys((("L1", "L0"), ("N1", "N0"), (base, "S3"))):
            s = P.paired(jd, a, b, "spearman", mo, H); q = P.paired(jd, a, b, "qlike", mo, H)
            Q(f"  {a}-{b}  Spearman {fmt(s)}  |  QLIKE {fmt(q)}")
    Q("\n===== PRE-REGISTERED CRITERION (H=60, candidate - B*, 30 months) =====")
    s = P.paired(jd, cand, base, "spearman", mo, 60); q = P.paired(jd, cand, base, "qlike", mo, 60); c = P.criterion(s, q, 20)
    Q(f"B*={base} candidate={cand}: Spearman gain {fmt(s)}; QLIKE diff {fmt(q)}")
    Q(f"  gain>=+0.003 {c['c1_mean_ge_0.003']}, lower>0 {c['c2_lo_gt_0']}, positive months>=20 {c['c3_months']} ({s['pos']}/{s['n']}), QLIKE<=0 {c['c4_qlike_le_0']} -> {'PASS' if c['pass'] else 'FAIL'}")
    Q("VERDICT: " + ("geometry adds over the best base" if c["pass"] else "geometry does not add over the best base"))
    Q("\n===== SECONDARY H=15 and 240 (no verdict; Holm over the two horizons within each comparison) =====")
    for a, b in dict.fromkeys((("L1", "L0"), ("N1", "N0"), (base, "S3"))):
        ss = [P.paired(jd, a, b, "spearman", mo, H) for H in (15, 240)]; adj = P.holm([x["p"] for x in ss])
        for H, x, ad in zip((15, 240), ss, adj):
            Q(f"  {a}-{b} H={H:3d}: Spearman {fmt(x)}  p={x['p']:.4f} Holm p={ad:.4f}")
    Q("\n===== per-month candidate - B* at H=60 (Spearman / QLIKE), train days =====")
    d = jd[jd.H == 60]
    for m in mo:
        a = d[(d.month == m) & (d.model == cand)].iloc[0]; b = d[(d.month == m) & (d.model == base)].iloc[0]
        Q(f"  {m} {a.spearman - b.spearman:+.4f} / {a.qlike - b.qlike:+.4f}  days {a.train_days}  n_te {a.n_te}")
    Q("\n===== by year, candidate - B* Spearman at H=60 =====")
    for y in ("2018", "2019", "2020"):
        ms = [m for m in mo if m.startswith(y)]; x = P.paired(jd, cand, base, "spearman", ms, 60); Q(f"  {y}: {fmt(x)}")
    open(os.path.join(HERE, "tables.txt"), "w", encoding="utf-8").write("\n".join(out) + "\n")
    json.dump({"choice": ch, "criterion": c, "spearman": s, "qlike": q}, open(os.path.join(HERE, "summary.json"), "w"), indent=1, default=float)
    print("\n".join(out))


if __name__ == "__main__":
    {"run": run, "tables": tables}[sys.argv[1] if len(sys.argv) > 1 else "run"]()
