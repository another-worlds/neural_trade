"""O3 run (SPEC.md). usage: python run_o3.py [--analyze]  -> results.jsonl, kernels.json, windows.json, tables.txt (log to stdout)."""
import json, os, sys, time
import numpy as np
import pandas as pd
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import o3_lib as L  # noqa: E402
T = L.T
try:
    import psutil; psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
except Exception:
    pass
MODELS = ["R0", "R0f", "BANK", "BANKf", "LK", "LKsh"]


def month_bounds(m):
    p = pd.Period(m, freq="M"); return p.start_time, (p + 1).start_time


def pick_windows(s):
    """quiet / shock 2-day windows: among days 2024-07-01..2025-07-18 (the window is the day and the next one, all before 2025-07-20),
    the day with the smallest / largest (max high - min low) / mean close."""
    d = s.loc["2024-07-01":"2025-07-18"]; g = d.groupby(d.index.normalize())
    rng = (g["high"].max() - g["low"].min()) / g["close"].mean()
    rng = rng[g.size() == 1440]
    return {"quiet": rng.idxmin(), "shock": rng.idxmax()}, rng


def main():
    t0 = time.time(); el = lambda: f"{time.time() - t0:.0f}s"
    tr_bars = T.load_bars(T.TRAIN_CSV, L.LOAD_FROM, T.CUTOFF); fw_bars = T.load_fwd()
    print("bars train", len(tr_bars), tr_bars.index.min(), tr_bars.index.max(), "fwd", len(fw_bars), fw_bars.index.min(), fw_bars.index.max(), el(), flush=True)
    FT = L.build_features(tr_bars); FF = L.build_features(fw_bars)
    print("anchors train", len(FT["anch"]), "fwd", len(FF["anch"]), "first fwd anchor", pd.Timestamp(FF["ta"][0]), el(), flush=True)
    rows = []; kern = {"taus": list(L.TAUS), "channels": list(L.CHANNELS), "channel_of_indicator": L.CHMAP.tolist(), "frozen": {}, "monthly_H60": {}}

    def log_rows(r):
        rows.extend(r)

    for span, months in (("sel", L.M_SEL), ("look", L.M_LOOK)):
        for m in months:
            m0, m1 = month_bounds(m)
            for H in L.HORIZONS:
                st = L.fit_month(FT, H, m0); te = L.test_rows(FT["ta"], FT[f"ok{H}"], m0, m1)
                r = L.score_state(st, FT, te, span, m); log_rows(r)
                if H == 60:
                    kern["monthly_H60"][m] = L.kernel_record(st)
            d = {x["model"]: round(x["spearman"], 4) for x in rows[-6:]}
            print(span, m, "H240", d, "n", rows[-1]["n_te"], el(), flush=True)
    # frozen fit at the cutoff, forward months
    frozen = {}
    for H in L.HORIZONS:
        frozen[H] = L.fit_month(FT, H, L.TRAIN_END); kern["frozen"][str(H)] = L.kernel_record(frozen[H])
        print("frozen H", H, "n_tr", frozen[H]["n_tr"], "LK epochs", frozen[H]["m"]["LK"]["epochs"], el(), flush=True)
    for m in L.M_FWD:
        m0, m1 = month_bounds(m)
        for H in L.HORIZONS:
            te = L.test_rows(FF["ta"], FF[f"ok{H}"], m0, m1); log_rows(L.score_state(frozen[H], FF, te, "fwd", m))
        print("fwd", m, {x["model"]: round(x["spearman"], 4) for x in rows[-6:]}, "n", rows[-1]["n_te"], el(), flush=True)
    with open(f"{HERE}/results.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    json.dump(kern, open(f"{HERE}/kernels.json", "w"))
    write_windows(tr_bars, FT, frozen[60])
    n_drop = {f"{nm}_H{H}": {"anchors": int(len(F["ta"])), "dropped": int((~F[f"ok{H}"]).sum())}
              for H in L.HORIZONS for nm, F in (("train", FT), ("fwd", FF))}
    json.dump(n_drop, open(f"{HERE}/dropped_rows.json", "w"))
    print("runtime", el(), flush=True); json.dump({"runtime_s": time.time() - t0}, open(f"{HERE}/runtime.json", "w"))
    analyze()


def write_windows(s, FT, st60):
    days, rng = pick_windows(s); A = FT["A"]; out = {"rule": "among full 1,440-bar days 2024-07-01..2025-07-18, quiet = smallest (max high - min low)/mean close, shock = largest; "
                                                          "the window is that day and the next (2,880 bars)", "taus": list(L.TAUS), "kernel": L.kernel_record(st60), "windows": {}}
    m = st60["m"]["LK"]; P = L.softmax(m["p"]["a"]); cs2 = np.concatenate([[0.0], np.cumsum(np.concatenate([[0.0], np.diff(np.log(A[:, 3]))]) ** 2)])
    for name, day in days.items():
        i0 = int(s.index.get_loc(day)); idx = np.arange(i0, i0 + 2880); bk = L.bank_at(A, idx)[:, L.CHMAP, :]
        I = np.log(np.einsum("nkj,kj->nk", bk, P) + L.EPS_CH[L.CHMAP]); Z = (I - np.array(m["zm"])) / np.array(m["zs"])
        lrv = lambda w: np.log(np.maximum(cs2[idx + 1] - cs2[idx + 1 - w], 1e-30) / w)
        out["windows"][name] = {"day": str(day.date()), "daily_range": float(rng[day]), "t": [str(x) for x in s.index[idx]], "close": A[idx, 3].tolist(),
                                "I": np.round(I, 5).T.tolist(), "Z": np.round(Z, 5).T.tolist(), "ln_rv60_per_bar": np.round(lrv(60), 5).tolist(),
                                "ln_rv1440_per_bar": np.round(lrv(1440), 5).tolist()}
    json.dump(out, open(f"{HERE}/windows.json", "w"))


# ------------------------------------------------------------------ analysis
def paired(df, a, b, col, months):
    pa = df[df.model == a].set_index("month")[col]; pb = df[df.model == b].set_index("month")[col]
    d = (pa - pb).reindex(months).dropna()
    if len(d) < 3:
        return None
    mu, lo, hi, t = T.ci(d.values); p = float(stats.ttest_1samp(d.values, 0).pvalue)
    return {"mean": mu, "lo": lo, "hi": hi, "p": p, "pos": int((d > 0).sum()), "n": int(len(d))}


def passes(sp, ql):
    return bool(sp and ql and sp["mean"] >= 0.005 and sp["lo"] > 0 and ql["mean"] <= 0)


def analyze():
    df = pd.DataFrame([json.loads(x) for x in open(f"{HERE}/results.jsonl")]); lines = []; res = {}
    spans = {"sel": L.M_SEL, "look": L.M_LOOK, "fwd": L.M_FWD}
    f = lambda d: "n/a" if d is None else f"{d['mean']:+.4f} [{d['lo']:+.4f},{d['hi']:+.4f}] +{d['pos']}/{d['n']}"
    for span, months in spans.items():
        for H in L.HORIZONS:
            d = df[(df.span == span) & (df.H == H)]
            lines.append(f"== span {span} ({len(months)} months) H={H}: mean Spearman / mean QLIKE ==")
            lines.append("   " + "  ".join(f"{m} {d[d.model == m].spearman.mean():+.4f}/{d[d.model == m].qlike.mean():.4f}" for m in MODELS))
            for a, b in (("LK", "R0"), ("BANK", "R0"), ("LK", "BANK"), ("LKsh", "R0"), ("LK", "R0f"), ("LK", "BANKf")):
                sp = paired(d, a, b, "spearman", months); ql = paired(d, a, b, "qlike", months)
                res[f"{span}|{H}|{a}-{b}"] = {"sp": sp, "ql": ql}
                lines.append(f"   {a:5s}-{b:5s}  dSpearman {f(sp)}   dQLIKE {f(ql)}")
    # sanity gate, criteria
    g = res["sel|60|LKsh-R0"]["sp"]["mean"]; broken = abs(g) > 0.003
    lines.append(f"== SANITY GATE (sel, H=60): LKsh - R0 mean Spearman {g:+.4f} (limit +-0.003) -> {'BROKEN' if broken else 'ok'}")
    pas = {s: passes(res[f"{s}|60|LK-R0"]["sp"], res[f"{s}|60|LK-R0"]["ql"]) for s in spans}
    syn = {s: bool(pas[s] and res[f"{s}|60|LK-BANK"]["sp"]["mean"] >= 0 and res[f"{s}|60|LK-BANK"]["ql"]["mean"] <= 0) for s in spans}
    verdict = "broken" if broken else ("holds" if pas["look"] and pas["fwd"] else "look only" if pas["look"] else "no")
    lines.append(f"== PASS (LK vs R0, H=60): sel {pas['sel']}, look {pas['look']}, fwd {pas['fwd']}")
    lines.append(f"== VERDICT: {verdict}   |   synthetic adds (look): {'yes' if syn['look'] else 'no'}   (fwd, descriptive: {'yes' if syn['fwd'] else 'no'}; sel: {'yes' if syn['sel'] else 'no'})")
    for s in spans:
        ps = [res[f"{s}|{H}|LK-R0"]["sp"]["p"] for H in (15, 240)]; o = np.argsort(ps); holm = [None, None]
        holm[o[0]] = min(1, ps[o[0]] * 2); holm[o[1]] = min(1, max(ps[o[1]], holm[o[0]]))
        lines.append(f"== Holm (LK-R0 Spearman, H 15/240, {s}): p raw {ps[0]:.4f}/{ps[1]:.4f} adjusted {holm[0]:.4f}/{holm[1]:.4f}")
    lines.append("== per-month rows: span month H model n spearman qlike")
    for _, r in df.iterrows():
        lines.append(f"{r.span} {r.month} H{r.H} {r.model:5s} n={r.n_te} sp={r.spearman:+.4f} ql={r.qlike:.4f}")
    open(f"{HERE}/tables.txt", "w").write("\n".join(lines))
    json.dump({"verdict": verdict, "pass": pas, "synthetic_adds": syn, "res": res}, open(f"{HERE}/summary.json", "w"), indent=1, default=float)
    print("\n".join(lines[:60]))


if __name__ == "__main__":
    analyze() if "--analyze" in sys.argv else main()
