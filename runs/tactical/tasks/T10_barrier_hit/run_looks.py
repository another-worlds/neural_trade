"""T10 looks (SPEC.md): Look 1 = 2024-07..2025-07-20 (cut file, monthly walk-forward), Look 2 = 2025-11..2026-09 (forward file, models frozen as
fitted on the 365 days to 2025-07-20; 2025-10 only calibrates H2 and B0). Runs ONLY after selection.json is committed and clean.
usage: python run_looks.py -> results.json, tables.txt, look_rows.npz"""
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import pandas as pd

import t10_lib as L

SEL = os.path.join(HERE, "selection.json")
HEAD = L.assert_selection_committed(SEL)                      # first: selection tracked, clean, in HEAD (no data is read before this)
CHOSEN = json.load(open(SEL))["chosen"]
SEL_SHA = hashlib.sha256(open(SEL, "rb").read()).hexdigest()

try:
    import psutil; psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
except Exception:
    pass


def verdict(look):
    c = look[f"{L.PRIMARY[0]}|{L.PRIMARY[1]}"][CHOSEN]
    ok1 = c["skill_vs_B1"] >= 0.02 and c["skill_ci"][0] > 0; ok2 = c["ece"] < 0.01
    return {"skill": c["skill_vs_B1"], "ci": c["skill_ci"], "ece": c["ece"], "skill_ok": bool(ok1), "ece_ok": bool(ok2), "pass": bool(ok1 and ok2)}


def holm_cells(look):
    keys = [f"{X}|{T}" for T in L.TS for X in L.XS]; ps = [look[k][CHOSEN]["skill_p"] for k in keys]; adj = L.T3.holm(ps)
    return {k: {"p": float(p), "p_holm": float(a)} for k, p, a in zip(keys, ps, adj)}


def main():
    t0 = time.time(); el = lambda: f"{time.time() - t0:.0f}s"
    print("selection ok: HEAD", HEAD, "chosen", CHOSEN, "sha", SEL_SHA, flush=True)
    cut = L.guard.load_ohlcv("cut"); fwd = L.guard.load_ohlcv("fwd")
    print("cut", len(cut), cut.index.max(), "| fwd", len(fwd), fwd.index.min(), fwd.index.max(), el(), flush=True)
    Pc = L.prep(cut, "cut"); Pf = L.prep(fwd, "fwd")
    m1 = list(pd.period_range(L.LOOK1_FIRST, L.LOOK1_LAST, freq="M")); m2 = list(pd.period_range(L.LOOK2_FIRST, L.LOOK2_LAST, freq="M"))
    out = {"head": HEAD, "selection_sha256": SEL_SHA, "chosen": CHOSEN, "look1": {}, "look2": {}, "info": {}, "frozen": {}}; rows = {}
    for T in L.TS:
        Bc = L.build(Pc, T); Bf = L.build(Pf, T)
        print(f"T={T}: cut anchors {len(Bc['a'])} (valid {int(Bc['ok'].sum())}), fwd anchors {len(Bf['a'])} (valid {int(Bf['ok'].sum())})", el(), flush=True)
        R1 = L.walk(Bc, T, m1)
        fr = L.fit_models(Bc, T, L.CUTOFF + pd.Timedelta(minutes=1))
        R2 = L.walk(Bf, T, m2, frozen=fr)
        out["frozen"][str(T)] = {"n_train": fr["n_train"], "last_label_end": fr["last_label_end"], "har_coef": fr["har"]["coef"].tolist(), "har_c": fr["har"]["c"]}
        out["info"][f"look1|{T}"] = R1["info"]; out["info"][f"look2|{T}"] = R2["info"]
        for nm, R in (("look1", R1), ("look2", R2)):
            S = L.summarize(R, L.boot_index(R["day"], L.BOOT, seed=0))
            for X in L.XS:
                out[nm][f"{X}|{T}"] = S[X]
            rows[f"{nm}|{T}|month"] = R["month"]; rows[f"{nm}|{T}|ta"] = R["ta"]; rows[f"{nm}|{T}|day"] = R["day"]
            for X in L.XS:
                rows[f"{nm}|{T}|y|{X}"] = R["y"][X]
                for m in L.MODELS:
                    rows[f"{nm}|{T}|p|{m}|{X}"] = R["p"][m][X].astype(np.float32)
        print(f"T={T} done", el(), flush=True)
    for nm in ("look1", "look2"):
        out[nm]["holm"] = holm_cells(out[nm]); out[nm]["verdict"] = verdict(out[nm])
    v1, v2 = out["look1"]["verdict"], out["look2"]["verdict"]
    out["success"] = bool(v1["pass"] and v2["pass"])
    out["runtime_s"] = time.time() - t0
    json.dump(out, open(os.path.join(HERE, "results.json"), "w"), indent=1)
    np.savez_compressed(os.path.join(HERE, "look_rows.npz"), **rows)
    lines = [f"T10 looks. chosen={CHOSEN} (selection.json sha256 {SEL_SHA}), run at HEAD {HEAD}", ""]
    for nm, title in (("look1", "Look 1 2024-07..2025-07-20 (cut file, monthly walk-forward)"), ("look2", "Look 2 2025-11..2026-09 (forward file, frozen fit to 2025-07-20)")):
        lines.append(title)
        for T in L.TS:
            for X in L.XS:
                r = out[nm][f"{X}|{T}"]; h = out[nm]["holm"][f"{X}|{T}"]
                lines.append(f" X={X} bps T={T} min n={r['B1']['n']} base_rate={r['B1']['base_rate']:.4f}" + ("   <-- PRIMARY" if (X, T) == L.PRIMARY else ""))
                for m in L.MODELS:
                    d = r[m]; ci = f" CI[{100 * d['skill_ci'][0]:+.2f},{100 * d['skill_ci'][1]:+.2f}] p={d['skill_p']:.4f}" if "skill_ci" in d else ""
                    ex = f" holm_p={h['p_holm']:.4f}" if m == CHOSEN else ""
                    lines.append(f"    {m}{'*' if m == CHOSEN else ' '} brier={d['brier']:.5f} skill_vs_B1={100 * d['skill_vs_B1']:+.2f}%{ci}{ex} ece={100 * d['ece']:.2f}pp auc={d['auc']:.4f} logloss={d['logloss']:.4f}")
        v = out[nm]["verdict"]
        lines.append(f" primary verdict: skill {100 * v['skill']:+.2f}% CI [{100 * v['ci'][0]:+.2f},{100 * v['ci'][1]:+.2f}] (>=2% and CI>0: {v['skill_ok']}); ECE {100 * v['ece']:.2f}pp (<1pp: {v['ece_ok']}); pass={v['pass']}")
        lines.append("")
    lines.append(f"SUCCESS (both looks pass both criteria): {out['success']}")
    open(os.path.join(HERE, "tables.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines)); print("total", el(), flush=True)


if __name__ == "__main__":
    main()
