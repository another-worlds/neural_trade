"""T05 Look A (SPEC.md): 2017-09..2018-12 on the cut file, the chosen model against M0 and M1. Refuses to run unless selection.json is tracked,
unmodified and committed. Usage: python run_lookA.py -> resultsA.json, tables.txt"""
import json, os, subprocess, sys, time
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import t05_lib as T  # noqa: E402


def assert_selection_committed(here=HERE):
    """selection.json is tracked and clean (no uncommitted change, not staged); returns the commit that last touched it."""
    p = os.path.join(here, "selection.json")
    assert os.path.exists(p), "selection.json missing: run run_select.py first"
    g = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, cwd=here)
    assert g("ls-files", "--error-unmatch", "selection.json").returncode == 0, "selection.json is not tracked: commit it before Look A"
    assert g("status", "--porcelain", "--", "selection.json").stdout.strip() == "", "selection.json has uncommitted changes"
    return g("log", "-1", "--format=%H", "--", "selection.json").stdout.strip()


def verdict(df, ch, need_frac=11 / 16):
    """The four criteria for `ch` against each of M0 and M1. Criterion 2: better in at least need_frac of the months (11 of 16 = 0.6875 on Look A, 0.8 on Look B)."""
    out = {}
    piv = df.pivot(index="month", columns="model", values="iscore_bps"); pooled = {m: float(d.n_in.sum() / d.n.sum()) for m, d in df.groupby("model")}
    sd = {m: float(d.cover.std(ddof=1)) for m, d in df.groupby("model")}
    for ref in ("M0", "M1"):
        d = (piv[ch] - piv[ref]).to_numpy(); mu, lo, hi, p = T.L3.ci(d); nb = int((d < 0).sum())
        c = {"mean_diff_bps": mu, "ci": [lo, hi], "p_lt0": p, "months_better": nb, "n_months": int(len(d)),
             "c1_ci_below_0": bool(hi < 0), "c2_better_in_enough_months": bool(nb >= need_frac * len(d) - 1e-9), "c3_pooled_cover_in_band": bool(0.88 <= pooled[ch] <= 0.92),
             "c4_cover_sd_le_ref": bool(sd[ch] <= sd[ref]), "pooled_cover": pooled[ch], "cover_sd": sd[ch], "ref_cover_sd": sd[ref]}
        c["success"] = bool(c["c1_ci_below_0"] and c["c2_better_in_enough_months"] and c["c3_pooled_cover_in_band"] and c["c4_cover_sd_le_ref"]); out[f"{ch}_vs_{ref}"] = c
    out["success_all"] = bool(all(out[f"{ch}_vs_{r}"]["success"] for r in ("M0", "M1")))
    return out


def main():
    t0 = time.time(); log = lambda m: print(f"[{time.time() - t0:5.0f}s] {m}", flush=True)
    sha = assert_selection_committed(); sel = json.load(open(f"{HERE}/selection.json")); ch = sel["chosen"]
    log(f"selection.json committed at {sha}; chosen {ch}")
    if ch is None:
        raise SystemExit("no model qualified in selection: nothing to test")
    s = T.load(T.CUT_CSV, T.LOOKA_LOAD); B = T.make_data(s); log(f"bars {len(s)} {s.index.min()} .. {s.index.max()}, anchors {len(B['D'].anch)}")
    names = ["M0", "M1", ch]
    df = T.walk_forward(B, T.LOOKA_MONTHS, names, log=log); summ = T.summarise(df)
    v = verdict(df, ch)
    # M0 vs M1 on this block (the T03 question on untouched history): negative = M0 better
    piv = df.pivot(index="month", columns="model", values="iscore_bps"); d = (piv["M0"] - piv["M1"]).to_numpy(); mu, lo, hi, p = T.L3.ci(d)
    m01 = {"mean_diff_bps": mu, "ci": [lo, hi], "p_lt0": p, "months_M0_better": int((d < 0).sum()), "n_months": int(len(d))}
    first_days = df[df.model == "M0"].sort_values("month")[["month", "n_train", "n_cal", "train_days"]].to_dict("records")
    res = {"selection_commit": sha, "chosen": ch, "summary": summ, "verdict": v, "M0_vs_M1": m01, "train_days_by_month": first_days, "runtime_s": time.time() - t0,
           "rows": json.loads(df.to_json(orient="records"))}
    json.dump(res, open(f"{HERE}/resultsA.json", "w"), indent=1)
    lines = [open(f"{HERE}/tables_selection.txt").read(), "", f"== T05 Look A: 2017-09..2018-12 (16 months), chosen {ch} (selection.json at {sha}) ==",
             "model  mean_iscore_bps  pooled_cover  monthly_cover_sd  mean_width_bps"]
    for m in names:
        x = summ[m]; lines.append(f"{m:5s} {x['iscore_bps']:14.3f} {x['pooled_cover']:12.4f} {x['cover_sd']:16.4f} {x['width_bps']:14.2f}")
    for k, c in v.items():
        if k != "success_all":
            lines.append(f"{k}: diff {c['mean_diff_bps']:+.3f} bps [{c['ci'][0]:+.3f},{c['ci'][1]:+.3f}], better {c['months_better']}/{c['n_months']}, "
                         f"pooled cover {c['pooled_cover']:.4f}, cover sd {c['cover_sd']:.4f} (ref {c['ref_cover_sd']:.4f}) -> c1 {c['c1_ci_below_0']} c2 {c['c2_better_in_enough_months']} "
                         f"c3 {c['c3_pooled_cover_in_band']} c4 {c['c4_cover_sd_le_ref']}  SUCCESS {c['success']}")
    lines.append(f"VERDICT (success against BOTH M0 and M1): {v['success_all']}")
    lines.append(f"M0 - M1 on this block: {mu:+.3f} bps [{lo:+.3f},{hi:+.3f}], M0 better in {m01['months_M0_better']}/16")
    lines.append("train days used per month (first 3): " + str([(r['month'], round(r['train_days'], 1), r['n_train']) for r in first_days[:3]]))
    lines.append("== per month (model: n, cover, width, iscore) ==")
    for _, r in df.sort_values(["month", "model"]).iterrows():
        lines.append(f"{r.month} {r.model} n={r.n} cov={r.cover:.4f} w={r.width_bps:.2f} is={r.iscore_bps:.2f}")
    open(f"{HERE}/tables.txt", "w").write("\n".join(lines)); print("\n".join(lines[:40])); log("done")


if __name__ == "__main__":
    main()
