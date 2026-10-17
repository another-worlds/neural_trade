"""T15 Day 3: summary table and recommendation (NOTE_day3.md). Reads Day 1 / Day 2 results and the stored T14 results; no harness run.
Run: D:/nt/nt_mm_env/python.exe day3.py"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import t15_lib as T  # noqa: E402
import m9_lib as M9  # noqa: E402

DAY = "day3"; OUT = os.path.join(HERE, DAY)
D1 = T.load_json(os.path.join(HERE, "day1", "results_day1.json"))
D2 = T.load_json(os.path.join(HERE, "day2", "results_day2.json"))
# (label, day1 key, day2 key or None, frozen artefact?, harness results path)
VARS = [("M7_gbm", "M7_gbm", "M7_gbm", False), ("M7c", "M7c_gbm_resid", "M7c", True), ("M9h6", "T15_h6", "M9h6", True), ("T15_h4", "T15_h4", None, False),
        ("T15_h8", "T15_h8", None, False), ("T15_h12", "T15_h12", None, False), ("M9_prof_str (24 slots)", "M9_prof_str", None, False)]


def res_of(key):
    p = os.path.join(HERE, "day1", key, "results.json")
    return T.load_json(p) if os.path.exists(p) else T.stored(key)


def look_means(res, tg):
    s = T.series(res, tg); return {lk: float(v[1].mean()) for lk, v in s.items()}


def main():
    os.makedirs(OUT, exist_ok=True)
    ref = res_of("M7_gbm"); gref = {tg: float(T.pooled(ref, tg)[1].mean()) for tg in "AB"}; lref = {tg: look_means(ref, tg) for tg in "AB"}
    f = {"A": lambda x: "n/a" if x is None else f"{x:+.4f}", "B": lambda x: "n/a" if x is None else f"{100 * x:+.2f}%"}
    ci = lambda tg, d: f"{f[tg](d['mean'])} [{f[tg](d['lo'])}, {f[tg](d['hi'])}]"
    rows = {}; L = []
    for lab, k1, k2, frozen in VARS:
        o1 = D1["views"][k1]; r = res_of(k1); row = {"variant": lab, "n_cols": o1["n_cols"], "frozen_artefact": frozen}
        for tg in "AB":
            lm = look_means(r, tg)
            row[tg] = {"pooled": o1[tg]["pooled"], "R": o1[tg]["R"], "R_look1": lm["look1"] / lref[tg]["look1"], "R_look2": lm["look2"] / lref[tg]["look2"],
                       "A_stable_windows": o1[tg]["n_windows_lo_gt0"], "stable": o1[tg]["stable"]}
        row["min_R"] = min(row["A"]["R"], row["B"]["R"])
        row["min_R_target_x_look"] = min(row["A"]["R_look1"], row["A"]["R_look2"], row["B"]["R_look1"], row["B"]["R_look2"])
        if k2:
            o2 = D2[k2]; row["day2"] = {"passes_module": o2["passes"]["module"], "target": {}}
            for tg in "AB":
                chk = o2["passes"][tg]["checks"]
                defined = ["C-H", "C-W"] + [f"C-P {p}" for p in ("cut", "fwd", "both")] + [f"C-S +{s}" for s in (6, 12, 18)]
                untest = [c for c in defined if c not in chk]
                row["day2"]["target"][tg] = {"pass": o2["passes"][tg]["pass"], "checks": chk, "n_passed": int(sum(chk.values())), "n_testable": len(chk), "untestable": untest,
                                              "failed": [c for c, v in chk.items() if not v]}
        rows[lab] = row
    elig = {lab: {"pass": r["day2"]["passes_module"], "R_A": r["A"]["R"], "R_B": r["B"]["R"], "n_cols": r["n_cols"]} for lab, r in rows.items() if r["frozen_artefact"] and "day2" in r}
    choice = M9.choose_freeze(elig)
    alt_elig = {lab: e for lab, e in elig.items() if e["pass"]}
    alt = max(alt_elig, key=lambda lab: rows[lab]["min_R_target_x_look"]) if alt_elig else None
    out = {"rows": rows, "eligible": elig, "recommended": choice, "alternative_reading_choice": alt, "M7_gbm_pooled": gref}
    T.dump(out, os.path.join(OUT, "summary_table.json"))

    L += ["T15 SUMMARY TABLE (pooled 24 months; Look 1 = Bitstamp 13 months, Look 2 = Binance 11 months). R = pooled gain / M7_gbm pooled gain "
          f"(M7_gbm: A {f['A'](gref['A'])}, B {f['B'](gref['B'])}).", ""]
    hdr = f"{'variant':22s} {'cols':>4s} | {'A pooled [t-CI]':>30s} {'R_A':>5s} | {'B pooled [t-CI]':>30s} {'R_B':>5s} | {'minR':>5s} {'minR t x l':>10s} | A-stable (windows) | Day 2 module pass | eligible"
    L += [hdr, "-" * len(hdr)]
    for lab, r in rows.items():
        d2 = r.get("day2"); p = "controls not run" if not d2 else ("PASS" if d2["passes_module"] else "FAIL")
        el = "frozen artefact" if r["frozen_artefact"] else "no (not a frozen artefact)"
        L.append(f"{lab:22s} {r['n_cols']:4d} | {ci('A', r['A']['pooled']):>30s} {r['A']['R']:5.2f} | {ci('B', r['B']['pooled']):>30s} {r['B']['R']:5.2f} | {r['min_R']:5.2f} {r['min_R_target_x_look']:10.2f} | "
                 f"{'STABLE' if r['A']['stable'] else 'no':6s} ({r['A']['A_stable_windows']}/4)    | {p:17s} | {el}")
    L += ["", "Per-look ratios R (look1 / look2): " + "; ".join(f"{lab}: A {r['A']['R_look1']:.2f}/{r['A']['R_look2']:.2f}, B {r['B']['R_look1']:.2f}/{r['B']['R_look2']:.2f}" for lab, r in rows.items()), ""]
    L.append("DAY 2 CONTROLS (a target survives a control iff the CI lower bound is above 0; 'untestable' = fewer than 5 scored months):")
    for lab, r in rows.items():
        if "day2" not in r:
            continue
        for tg in "AB":
            t = r["day2"]["target"][tg]
            L.append(f"  {lab:8s} target {tg}: pass {t['pass']}; testable controls passed {t['n_passed']}/{t['n_testable']}; failed {t['failed'] or 'none'}; untestable {t['untestable'] or 'none'}")
    L += ["", f"RECOMMENDATION (M9 freeze rule over eligible frozen artefacts that pass Day 2): {choice or 'NONE'}",
          f"alternative reading (min over target x look): {alt or 'NONE'}", f"eligible rows: {json.dumps(elig)}"]
    open(os.path.join(OUT, "summary_table.txt"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))

    # markdown table + recommendation text + report snippet
    md = ["| variant | cols | A pooled | R_A | B pooled | R_B | min R | Day 1 A-stable | Day 2 | verdict |", "|---|---|---|---|---|---|---|---|---|---|"]
    for lab, r in rows.items():
        d2 = r.get("day2"); verdict = ("RECOMMENDED by rule" if lab == choice else ("passes Day 2, not a frozen artefact (reference)" if d2 and d2["passes_module"] else
                                                                                       ("fails Day 2" if d2 else "controls not run")))
        md.append(f"| {lab} | {r['n_cols']} | {ci('A', r['A']['pooled'])} | {r['A']['R']:.2f} | {ci('B', r['B']['pooled'])} | {r['B']['R']:.2f} | {r['min_R']:.2f} | "
                  f"{'yes' if r['A']['stable'] else 'no'} ({r['A']['A_stable_windows']}/4) | {'pass' if d2 and d2['passes_module'] else ('fail' if d2 else '-')} | {verdict} |")
    open(os.path.join(OUT, "summary_table.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    rec = [f"# T15 recommendation (rule: NOTE_day2 / NOTE_day3)", "", f"Recommended for the prospective freeze: **{choice or 'none'}**.", "",
           "Eligible = frozen artefacts that pass Day 2 (survive hour and weekend shuffles, and every testable pool and start-date variant, on at least one target).", "", *md, ""]
    open(os.path.join(OUT, "recommendation.md"), "w", encoding="utf-8").write("\n".join(rec) + "\n")
    snippet = ["# Proposed replacement rows for the morning report's prospective-bets table (NOT applied; the report is the lead's to edit)", "",
               "| module | frozen file | T15 verdict | key numbers |", "|---|---|---|---|"]
    for lab in ("M7c", "M9h6"):
        r = rows[lab]; d2 = r["day2"]; status = "RECOMMENDED by the pre-registered rule" if lab == choice else ("passes Day 2" if d2["passes_module"] else "fails Day 2")
        fails = sorted({c for tg in "AB" for c in d2["target"][tg]["failed"]})
        snippet.append(f"| {lab} | {'M7c frozen files' if lab == 'M7c' else 'frozen_P_M9h6_prof_str.json'} | {status} | pooled A {ci('A', r['A']['pooled'])}, B {ci('B', r['B']['pooled'])}; min R {r['min_R']:.2f}; "
                       f"failed controls: {', '.join(fails) if fails else 'none'}; untestable: {', '.join(sorted({c for tg in 'AB' for c in d2['target'][tg]['untestable']})) or 'none'} |")
    open(os.path.join(OUT, "prospective_bets_update.md"), "w", encoding="utf-8").write("\n".join(snippet) + "\n")


if __name__ == "__main__":
    T.below_normal(); T.run_job(DAY, "summary table and recommendation", main, "day3/summary_table.txt, day3/recommendation.md, day3/prospective_bets_update.md")
