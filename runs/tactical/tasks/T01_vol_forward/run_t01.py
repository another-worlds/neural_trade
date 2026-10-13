"""T01 run (SPEC.md). usage: python run_t01.py  -> results.json, tables.txt (log to stdout)."""
import json, os, sys, time
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import t01_lib as T  # noqa: E402

try:
    import psutil; psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
except Exception:
    pass

H72 = {"skill_ar": (0.1633, 0.1466, 0.1801), "r2_rw": 0.2047}      # gaps/look3.json (target A, S3, 2024-07..2025-07)


def main():
    t0 = time.time(); el = lambda: f"{time.time() - t0:.0f}s"
    tr_bars = T.load_train(); print("train bars", len(tr_bars), tr_bars.index.min(), tr_bars.index.max(), el(), flush=True)
    fw_bars = T.load_fwd(); print("fwd bars", len(fw_bars), fw_bars.index.min(), fw_bars.index.max(), el(), flush=True)
    gaps = int((fw_bars.index.to_series().diff().dropna() != pd.Timedelta(minutes=1)).sum()); print("fwd non-1-minute steps", gaps, flush=True)
    BT = T.build(tr_bars, 1441); BF = T.build(fw_bars, T.WARM); print("anchors train", len(BT["D"].anch), "fwd", len(BF["D"].anch), el(), flush=True)

    # arm F: fit ONCE on the 365 days ending at the cutoff (label end <= cutoff), freeze
    m0 = T.CUTOFF + pd.Timedelta(minutes=1)
    tr = {tg: T.train_idx(BT, tg, m0) for tg in T.TARGETS}
    for tg in T.TARGETS:
        ta = BT["D"].ta[tr[tg]]; print("train", tg, "rows", len(tr[tg]), "first", ta.min(), "last label end", ta.max() + pd.Timedelta(minutes=T.TARGETS[tg]), flush=True)
    frozen = T.fit(BT, tr); rows = []
    for m in pd.PeriodIndex(T.FWD_MONTHS, freq="M"):
        r = T.score_month(frozen, BF, m, "F"); rows += r
        print("F", m, {(x["target"], x["model"]): round(x["skill_ar"], 4) for x in r if x["model"] == "S3"}, "n", r[0]["n_te"], el(), flush=True)
    # arm R: refit each month M >= 2025-11 on the forward file's bars before M (expanding window from 2025-10-01)
    for m in pd.period_range("2025-11", "2026-09", freq="M"):
        m0 = m.start_time; trr = {tg: T.train_idx(BF, tg, m0, start=T.FWD_START) for tg in T.TARGETS}
        mod = T.fit(BF, trr); r = T.score_month(mod, BF, m, "R"); rows += r
        for x in r:
            x["n_tr"] = int(len(trr[x["target"]]))
        print("R", m, {(x["target"], x["model"]): round(x["skill_ar"], 4) for x in r if x["model"] == "S3"}, "ntr", len(trr["A"]), el(), flush=True)
    df = pd.DataFrame(rows)

    lines = ["== T01: forward test of H72 (SPEC.md). Scored months arm F 2025-10..2026-09 (12), arm R 2025-11..2026-09 (11) =="]
    lines.append(f"forward file non-1-minute steps: {gaps}; train rows A {len(tr['A'])}, B {len(tr['B'])}")
    res = {"h72_look": H72, "train_rows": {k: int(len(v)) for k, v in tr.items()}, "fwd_nonminute_steps": gaps}
    for arm in ("F", "R"):
        for tg in T.TARGETS:
            d = df[(df.arm == arm) & (df.target == tg)]; mods = T.MODELS_A if tg == "A" else T.MODELS_B
            lines.append(f"-- arm {arm} target {tg}: mean over months (n={d.month.nunique()}) MSE / Spearman / R2 vs rw / skill vs ar [95% t-CI]")
            for k in mods:
                x = d[d.model == k]
                if len(x) < 2:
                    continue
                s = T.ci(x.skill_ar); r2 = T.ci(x.r2_rw); sp = T.ci(x.spearman.fillna(0))
                lines.append(f"   {k:5s} mse {x.mse.mean():.4f} spearman {x.spearman.mean():+.4f} [{sp[1]:+.4f},{sp[2]:+.4f}]  R2_rw {r2[0]:+.4f} [{r2[1]:+.4f},{r2[2]:+.4f}]  "
                             f"skill_ar {s[0]:+.4f} [{s[1]:+.4f},{s[2]:+.4f}]")
                res.setdefault(arm, {}).setdefault(tg, {})[k] = {"mse": float(x.mse.mean()), "spearman": sp[0], "spearman_ci": sp[1:3], "r2_rw": r2[0],
                                                                  "r2_rw_ci": r2[1:3], "skill_ar": s[0], "skill_ar_ci": s[1:3]}
    # verdict (SPEC): arm F, S3, target A
    verdict = {}
    for nm in ("S3", "S1"):
        x = df[(df.arm == "F") & (df.target == "A") & (df.model == nm)]; mu, lo, hi, t = T.ci(x.skill_ar); r2 = float(x.r2_rw.mean())
        verdict[nm] = {"skill_ar": [mu, lo, hi], "r2_rw": r2, "c1_mean_ge_0.02": bool(mu >= 0.02), "c2_lo_gt_0": bool(lo > 0), "c3_r2_gt_0": bool(r2 > 0),
                       "pass": bool(mu >= 0.02 and lo > 0 and r2 > 0), "months_positive": int((x.skill_ar > 0).sum()), "n_months": int(len(x))}
    res["verdict"] = verdict
    lines.append(f"== VERDICT arm F, target A: S3 (primary) {verdict['S3']}\n   S1 (secondary) {verdict['S1']}\n   H72 look: {H72}")
    lines.append("== per-month tables (arm, month, target, model: n_te, dropped, mse, R2_rw, spearman, skill_ar) ==")
    for _, r in df.iterrows():
        lines.append(f"{r.arm} {r.month} {r.target} {r.model:5s} n={r.n_te} drop={r.n_dropped} mse={r.mse:.4f} R2={r.r2_rw:+.4f} sp={r.spearman:+.4f} skill={r.skill_ar:+.4f}")
    res["runtime_s"] = time.time() - t0
    json.dump({"summary": res, "rows": json.loads(df.to_json(orient="records"))}, open(f"{HERE}/results.json", "w"), indent=1)
    open(f"{HERE}/tables.txt", "w").write("\n".join(lines)); print("\n".join(lines[:60])); print("done", el())


if __name__ == "__main__":
    main()
