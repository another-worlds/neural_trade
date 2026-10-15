"""T11 stage 2 (SPEC.md): the report-only looks, AFTER selection.json is committed. (a) the cut file 2017-09..2018-12 (partly seen, T04-A); (b) the Binance
file 2025-11..2026-09, monthly refit on the forward file's own bars from 2025-10-01 (arm R as T01 / T03), read only to 2026-09-30 23:59. Nothing here
chooses anything. usage: python run_report.py -> results.json, tables.txt (log to stdout). Refuses unless selection.json is committed unchanged."""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import t11_lib as L  # noqa: E402
T = L.T

try:
    import psutil; psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
except Exception:
    pass


def table(rows, label, chosen):
    lines = [f"== {label}: {len(rows)} months {rows[0]['month']}..{rows[-1]['month']}, target A: mean over months of skill vs ar [95% t-CI] / months positive =="]
    for k in ["rw", "ar"] + L.VARIANTS:
        sk = [r["skill_ar"][k] for r in rows]
        if k in ("rw", "ar"):
            lines.append(f"   {k:3s} mse {np.mean([r['mse'][k] for r in rows]):.5f}  skill_ar {np.mean(sk):+.4f}"); continue
        mu, lo, hi, _ = T.ci(sk)
        lines.append(f"   {k:3s}{'*' if k == chosen else ' '} mse {np.mean([r['mse'][k] for r in rows]):.5f}  skill_ar {mu:+.4f} [{lo:+.4f},{hi:+.4f}]  positive {int((np.array(sk) > 0).sum())}/{len(sk)}")
    lines.append("-- paired differences, variant minus V0 [95% t-CI] / months positive (target A); secondary (|r_15|): Spearman diff, QLIKE diff")
    comps = {}
    for v in L.VARIANTS[1:]:
        c = L.paired(rows, v); comps[v] = c; d, q, sp, ql = c["skill_ar_diff"], c["mse_ratio_skill"], c["spearman_diff"], c["qlike_diff"]
        lines.append(f"   {v}{'*' if v == chosen else ' '}-V0 skill_ar diff {d['mean']:+.4f} [{d['ci'][0]:+.4f},{d['ci'][1]:+.4f}] pos {d['positive']}/{len(rows)}   1-mse/mse_V0 {q['mean']:+.4f} "
                     f"| R: spearman {sp['mean']:+.4f} [{sp['ci'][0]:+.4f},{sp['ci'][1]:+.4f}] qlike {ql['mean']:+.5f} (negative = better)")
    lines.append("-- per month: month n_tr n_te | skill_ar " + "/".join(L.VARIANTS))
    for r in rows:
        lines.append(f"{r['month']} {r['n_tr']:6d} {r['n_te']:5d} | " + " ".join(f"{r['skill_ar'][k]:+.4f}" for k in L.VARIANTS))
    return lines, comps


def main():
    t0 = time.time(); el = lambda: f"{time.time() - t0:.0f}s"
    sel_path = os.path.join(HERE, "selection.json"); commit = L.assert_committed(sel_path); sel = json.load(open(sel_path)); chosen = sel["chosen"]
    print("selection.json committed in", commit, "chosen", chosen, flush=True)
    log = lambda m, r: print(m, "ntr", r["n_tr"], "n_te", r["n_te"], {k: round(v, 4) for k, v in r["skill_ar"].items() if k.startswith("V")}, el(), flush=True)
    # (a) 2017-09..2018-12, the cut file (file time -> UTC by timefix)
    s = T.load_bars(T.TRAIN_CSV, L.A_LO, L.A_HI); print("A bars", len(s), s.index.min(), s.index.max(), el(), flush=True)
    BA = L.build(s, 1441, "bitstamp"); print("A drops", BA["drops"], flush=True)
    rowsA = L.walk_forward(BA, L.A_MONTHS, "report_2017_18", log=log)
    # (b) Binance 2025-11..2026-09, refit on the forward bars from 2025-10-01 (an expanding window), reads stop at 2026-09-30 23:59
    fw = L.load_freeze_bars(T.FWD_CSV); print("B bars", len(fw), fw.index.min(), fw.index.max(), el(), flush=True)
    assert fw.index.max() <= L.FW_HI
    BB = L.build(fw, T.WARM, "binance"); print("B drops", BB["drops"], flush=True)
    rowsB = L.walk_forward(BB, L.FW_MONTHS, "report_binance_2025_26", fixed_start=L.FW_LO, log=log)
    la, ca = table(rowsA, "REPORT-ONLY cut file 2017-09..2018-12 (partly seen: T04-A)", chosen)
    lb, cb = table(rowsB, "REPORT-ONLY Binance 2025-11..2026-09 (monthly refit on the forward file; seen by H83)", chosen)
    lines = [f"selection.json committed in {commit}; chosen {chosen}; selection mean skill diffs {sel['mean_skill_ar_diff_vs_V0']}"] + la + [""] + lb
    json.dump({"selection_commit": commit, "chosen": chosen, "A_2017_18": {"rows": rowsA, "comparisons": ca, "drops": BA["drops"]},
               "binance_2025_26": {"rows": rowsB, "comparisons": cb, "drops": BB["drops"], "n_bars": int(len(fw)), "last_bar": str(fw.index.max())},
               "runtime_s": time.time() - t0}, open(f"{HERE}/results.json", "w"), indent=1)
    open(f"{HERE}/tables.txt", "w").write("\n".join(lines)); print("\n".join(l for l in lines if not l[:2].isdigit())); print("done", el())


if __name__ == "__main__":
    main()
