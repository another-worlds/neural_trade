"""O4 Part A (SPEC.md): retrospective 2017-09..2018-12. usage: python run_o4.py -> resultsA.jsonl, tablesA.txt, summaryA.json."""
import json, os, sys, time
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import o4_lib as O  # noqa: E402
L = O.L; T = O.T
try:
    import psutil; psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
except Exception:
    pass


def main():
    t0 = time.time(); el = lambda: f"{time.time() - t0:.0f}s"
    s = T.load_bars(T.TRAIN_CSV, O.A_LO, O.A_LOAD_HI); print("bars", len(s), s.index.min(), s.index.max(), el(), flush=True)
    gaps = s.index.to_series().diff().dropna(); nm = gaps[gaps != pd.Timedelta(minutes=1)]; print("non-1-minute steps", len(nm), flush=True)
    F = L.build_features(s); print("anchors", len(F["anch"]), "first", pd.Timestamp(F["ta"][0]), "last", pd.Timestamp(F["ta"][-1]), el(), flush=True)
    rows = []; days = {}
    for m in O.A_MONTHS:
        m0, m1 = O.month_bounds(m)
        for H in L.HORIZONS:
            st = O.fit_rb(F, H, m0); te = L.test_rows(F["ta"], F[f"ok{H}"], m0, m1)
            assert len(te) > 0 and pd.Timestamp(st["last_label_end"]) <= m0
            rows.extend(O.score_rb(st, F, te, "A", m))
            days[(m, H)] = round(float((m0 - pd.Timestamp(st["first_ta"])).total_seconds() / 86400), 1)
        d = {(x["H"], x["model"]): round(x["spearman"], 4) for x in rows[-6:]}
        print(m, "train days H60", days[(m, 60)], "n_tr", rows[-3]["n_tr"], d, el(), flush=True)
    with open(f"{HERE}/resultsA.jsonl", "w") as f:
        for r in rows:
            r["train_days"] = days[(r["month"], r["H"])]; f.write(json.dumps(r) + "\n")
    df = pd.DataFrame([json.loads(x) for x in open(f"{HERE}/resultsA.jsonl")]); lines = []
    fm = lambda d: "n/a" if d is None else f"{d['mean']:+.4f} [{d['lo']:+.4f},{d['hi']:+.4f}] +{d['pos']}/{d['n']}"
    res = {}
    lines.append(f"== O4 Part A: {len(O.A_MONTHS)} months 2017-09..2018-12; bars {len(s)} ({s.index.min()} .. {s.index.max()}); non-1-minute steps {len(nm)} ==")
    for H in L.HORIZONS:
        d = df[df.H == H]
        lines.append(f"== H={H}: mean Spearman / mean QLIKE ==")
        lines.append("   " + "  ".join(f"{m} {d[d.model == m].spearman.mean():+.4f}/{d[d.model == m].qlike.mean():.4f}" for m in O.MODELS))
        sp = O.paired(d, "BANK", "R0", "spearman", O.A_MONTHS); ql = O.paired(d, "BANK", "R0", "qlike", O.A_MONTHS); res[H] = {"sp": sp, "ql": ql}
        lines.append(f"   BANK-R0  dSpearman {fm(sp)}   dQLIKE {fm(ql)}")
    ps = [res[H]["sp"]["p"] for H in (15, 240)]; hp = O.holm2(ps)
    lines.append(f"== Holm (BANK-R0 Spearman, H 15/240; 2 tests): p raw {ps[0]:.4f}/{ps[1]:.4f} adjusted {hp[0]:.4f}/{hp[1]:.4f}")
    crit = O.criterion(res[60]["sp"], res[60]["ql"], 11)
    verdict = "confirmed" if crit["pass"] else "not confirmed"
    lines.append(f"== CRITERION A (H=60, BANK - R0): {crit}")
    lines.append(f"== VERDICT A: {verdict}")
    lines.append("-- training days per month (H=60) and n_tr (H15/60/240):")
    for m in O.A_MONTHS:
        lines.append(f"   {m} days {days[(m, 60)]:6.1f}  n_tr " + "/".join(str(int(df[(df.month == m) & (df.H == H) & (df.model == 'R0')].n_tr.iloc[0])) for H in L.HORIZONS))
    lines.append("-- per month BANK-R0 (H: dSpearman/dQLIKE):")

    def g(m, H, mod, col):
        return df[(df.month == m) & (df.H == H) & (df.model == mod)][col].iloc[0]
    for m in O.A_MONTHS:
        lines.append(f"   {m} " + "  ".join(f"H{H} {g(m, H, 'BANK', 'spearman') - g(m, H, 'R0', 'spearman'):+.4f}/{g(m, H, 'BANK', 'qlike') - g(m, H, 'R0', 'qlike'):+.4f}" for H in L.HORIZONS))
    lines.append("== per-month rows: month H model n_te sp ql")
    for _, r in df.iterrows():
        lines.append(f"{r.month} H{r.H} {r.model:4s} n={r.n_te} sp={r.spearman:+.4f} ql={r.qlike:.4f}")
    open(f"{HERE}/tablesA.txt", "w").write("\n".join(lines))
    json.dump({"verdict": verdict, "criterion": crit, "res": {str(k): v for k, v in res.items()}, "holm_15_240": hp, "runtime_s": time.time() - t0},
              open(f"{HERE}/summaryA.json", "w"), indent=1, default=float)
    print("\n".join(lines[:12])); print("runtime", el())


if __name__ == "__main__":
    main()
