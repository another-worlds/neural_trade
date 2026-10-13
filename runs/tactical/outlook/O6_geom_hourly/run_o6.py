"""O6 runner (SPEC.md). usage: run_o6.py [run|tables]. Frame (cache) -> judgement span 2019-01..2020-12 (L1, L2, L2sh at 15/60/240) -> spent spans
(descriptive) -> tables. Resumable through results.jsonl."""
import json, os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import o6_lib as P  # noqa: E402
O3 = P.O3; T = P.T
RES = os.path.join(HERE, "results.jsonl"); LOG = os.path.join(HERE, "run.log")
SANITY = 0.003


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
    assert s.index.max() <= P.LOAD_HI <= T.CUTOFF
    gaps = s.index.to_series().diff().dropna(); log("bars", len(s), s.index.min(), s.index.max(), "non-1-minute steps", int((gaps != pd.Timedelta(minutes=1)).sum()))
    F = P.build_frame6(s, P.FIRST6); del s
    log("frame anchors", len(F["ta"]), pd.Timestamp(F["ta"][0]), pd.Timestamp(F["ta"][-1]), f"{time.time() - t0:.0f}s",
        "G finite", bool(np.isfinite(F["G"]).all()), "XB finite", bool(np.isfinite(F["XB"]).all()), "G shape", F["G"].shape)
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


def run_span(F, span, months, done):
    for mo in months:
        m0, m1 = P.month_bounds(mo); t0 = time.time()
        for H in P.HORIZONS:
            if (span, mo, H) in done:
                continue
            key = int(mo.replace("-", ""))
            st = P.fit_h(F, H, m0, key=key); te = O3.test_rows(F["ta"], F[f"ok{H}"], m0, m1)
            assert len(te) > 0 and pd.Timestamp(st["last_label_end"]) <= m0
            rows = P.score(st, F, te, span, mo, key=key)
            for r in rows:
                r["train_days"] = round(float((m0 - pd.Timestamp(st["first_ta"])).total_seconds() / 86400), 1)
            write(rows)
        log(span, mo, f"{time.time() - t0:.0f}s", f"free {free_gb():.0f} GB")


def run():
    nice(); t_all = time.time(); F = load_frame(); done = done_keys()
    assert pd.Timestamp(F["ta"].max()) <= T.CUTOFF
    run_span(F, "judge", P.J_MONTHS, done)
    run_span(F, "spent1", P.S_SPENT1, done)
    run_span(F, "spent2", P.S_SPENT2, done)
    log("total", f"{time.time() - t_all:.0f}s")
    json.dump({"runtime_s": time.time() - t_all}, open(os.path.join(HERE, "runtime.json"), "w"))


def fmt(d, nd=4):
    return "n/a" if d is None else f"{d['mean']:+.{nd}f} [{d['lo']:+.{nd}f},{d['hi']:+.{nd}f}] +{d['pos']}/{d['n']} (min {d['min']:+.{nd}f}, max {d['max']:+.{nd}f})"


def tables():
    df = rows_df(); out = []; Q = out.append; jd = df[df.span == "judge"]; mo = P.J_MONTHS
    Q(f"O6 tables. judgement months {jd.month.nunique()} ({jd.month.min()}..{jd.month.max()}); rows {len(df)}")
    for H in P.HORIZONS:
        Q(f"\n===== JUDGEMENT 2019-01..2020-12  H={H}: mean Spearman | mean QLIKE =====")
        d = jd[jd.H == H]
        for m in P.MODELS:
            x = d[d.model == m]; Q(f"  {m:5s} {x.spearman.mean():.4f} | {x.qlike.mean():.4f}   (n months {x.month.nunique()})")
        Q("paired (a - b), per month: mean [95% t-CI] +positive/n (min, max)   |  QLIKE diff (negative = better)")
        for a, b in (("L2", "L1"), ("L2sh", "L1")):
            s = P.paired(jd, a, b, "spearman", mo, H); q = P.paired(jd, a, b, "qlike", mo, H)
            Q(f"  {a}-{b}  Spearman {fmt(s)}  |  QLIKE {fmt(q)}")
    Q("\n===== SECONDARY H=15 and 60, L2 - L1 Spearman (no verdict; Holm over the two horizons) =====")
    ss = [P.paired(jd, "L2", "L1", "spearman", mo, H) for H in (15, 60)]; adj = P.holm([x["p"] for x in ss])
    for H, x, ad in zip((15, 60), ss, adj):
        Q(f"  L2-L1 H={H:3d}: Spearman {fmt(x)}  p={x['p']:.4f} Holm p={ad:.4f}")
    Q("\n===== DESCRIPTIVE spent spans (no verdict; second/third look at these months): L2 - L1 and L2sh - L1 =====")
    for span, label in (("spent1", "2021-01..2024-06"), ("spent2", "2024-07..2025-07 (2025-07 ends 07-20)")):
        d = df[df.span == span]; ms = sorted(d.month.unique())
        for H in P.HORIZONS:
            for a in ("L2", "L2sh"):
                s = P.paired(d, a, "L1", "spearman", ms, H); q = P.paired(d, a, "L1", "qlike", ms, H)
                Q(f"  {label} H={H:3d} {a}-L1  Spearman {fmt(s)}  |  QLIKE {fmt(q)}")
    Q("\n===== PRE-REGISTERED CRITERION (H=240, L2 - L1, 24 judgement months) =====")
    s = P.paired(jd, "L2", "L1", "spearman", mo, 240); q = P.paired(jd, "L2", "L1", "qlike", mo, 240); c = P.criterion(s, q, 16)
    Q(f"Spearman gain {fmt(s)}; QLIKE diff {fmt(q)}")
    Q(f"  gain>=+0.003 {c['c1_mean_ge_0.003']}, lower>0 {c['c2_lo_gt_0']}, positive months>=16 {c['c3_months']} ({s['pos']}/{s['n']}), QLIKE<=0 {c['c4_qlike_le_0']} -> {'PASS' if c['pass'] else 'FAIL'}")
    sh = P.paired(jd, "L2sh", "L1", "spearman", mo, 240); gate = abs(sh["mean"]) <= SANITY
    Q(f"\n===== SANITY GATE (H=240, L2sh - L1 within +-{SANITY}) =====\n  L2sh-L1 mean {sh['mean']:+.4f} [{sh['lo']:+.4f},{sh['hi']:+.4f}] -> {'ok' if gate else 'BROKEN'}")
    verdict = "broken" if not gate else ("pass" if c["pass"] else "fail")
    Q("VERDICT: " + verdict)
    Q("\n===== per-month L2 - L1 at H=240 (Spearman / QLIKE), train days =====")
    d = jd[jd.H == 240]
    for m in mo:
        a = d[(d.month == m) & (d.model == "L2")].iloc[0]; b = d[(d.month == m) & (d.model == "L1")].iloc[0]
        Q(f"  {m} {a.spearman - b.spearman:+.4f} / {a.qlike - b.qlike:+.4f}  days {a.train_days}  n_te {a.n_te}")
    open(os.path.join(HERE, "tables.txt"), "w", encoding="utf-8").write("\n".join(out) + "\n")
    json.dump({"verdict": verdict, "criterion": c, "spearman": s, "qlike": q, "sanity": sh, "sanity_ok": bool(gate)},
              open(os.path.join(HERE, "summary.json"), "w"), indent=1, default=float)
    print("\n".join(out))


if __name__ == "__main__":
    {"run": run, "tables": tables}[sys.argv[1] if len(sys.argv) > 1 else "run"]()
