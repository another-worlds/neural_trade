"""T07 selection on 2019-01 .. 2024-06 ONLY. Writes selection.json (+ selection_tables.txt); reads no bar on or after 2024-07-01.
The next step (after `git commit` of selection.json) is run_looks.py. usage: python run_sel.py [--workers 6]"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import t07_lib as T
import analysis as A
import pipeline as P

OUT = HERE


def main():
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 6
    t0 = time.time()
    b = P.build_main("sel", T.SEL_END, T.SEL_LAST, workers)
    grid = b["grid"]; h0 = grid.hour_of("2019-01-01"); h1 = grid.nh
    assert grid.index[-1] < T.SEL_END
    view = grid.view(h0, h1); pos = b["pos"][:, h0 * 60:h1 * 60]; har = T.har_scale(b["sig"][h0:h1], b["med"][h0:h1])
    combos, (g, u, p) = A.combos_for(view, pos, har)
    sl = {"sel": slice(0, view.nh)}
    mem, r2m = A.table_for({m: pos[i] for i, m in enumerate(T.MEMBERS)}, view, sl)
    com, r2c = A.table_for(combos, view, sl)
    bh = A.buy_hold(view, sl)
    res = {"period": "2019-01-01 .. 2024-06-30 23:59", "hours": int(view.nh), "members": mem, "combos": com, "buy_hold": bh, "null": {}, "eligible": {}}
    lines = []
    for c in T.COMBOS:
        st = com[c]["sel"]["2"]
        nul = A.null_summary(combos[c], view, sl["sel"], T.VERDICT_COST)
        res["null"][c] = nul
        res["eligible"][c] = bool(st["trades_per_year"] >= 100 and st["total_return"] > nul["p95"])
    elig = [c for c in T.COMBOS if res["eligible"][c]]
    pool = elig if elig else list(T.COMBOS)
    cal = {c: com[c]["sel"]["2"]["calmar"] for c in pool}
    chosen = max(cal, key=lambda c: (cal[c] if np.isfinite(cal[c]) else -1e9))
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True).stdout.strip()
    sel = {"rule": "highest Calmar at 2 bps on 2019-01..2024-06 among combinations with >= 100 trades/year and cumulative return above the random-side null p95",
           "any_eligible": bool(elig), "chosen": chosen, "chosen_is_eligible": bool(res["eligible"][chosen]),
           "calmar_2bps": {c: com[c]["sel"]["2"]["calmar"] for c in T.COMBOS}, "total_return_2bps": {c: com[c]["sel"]["2"]["total_return"] for c in T.COMBOS},
           "max_dd_2bps": {c: com[c]["sel"]["2"]["max_dd"] for c in T.COMBOS}, "trades_per_year_2bps": {c: com[c]["sel"]["2"]["trades_per_year"] for c in T.COMBOS},
           "null_p95_2bps": {c: res["null"][c]["p95"] for c in T.COMBOS}, "eligible": res["eligible"],
           "sigma_ref": "median of the members' trailing 30-day sigma (scale cap 5); HAR sigma* = in-sample median of the month's training span",
           "data_last_bar_read": str(grid.index[-1]), "code_head_when_selected": sha, "seconds": round(time.time() - t0)}
    json.dump(sel, open(f"{OUT}/selection.json", "w"), indent=1)
    json.dump(res, open(f"{OUT}/selection_full.json", "w"), indent=1)
    np.savez(f"{T.CACHE}/sel_hourly.npz", g=g, u=u, p=p, **{f"c_{c}": combos[c].astype(np.float32) for c in T.COMBOS})
    # tables
    out = ["Selection period 2019-01-01 .. 2024-06-30 (in-sample of selection, not evidence of skill)", f"hours {view.nh}", ""]
    out.append("Buy-and-hold (no cost): " + ", ".join(f"{k} {v:+.3f}" if k != "calmar" else f"{k} {v:.2f}" for k, v in bh["sel"].items()))
    for title, tab in (("MEMBERS", mem), ("COMBINATIONS", com)):
        out.append(""); out.append(title)
        out.append(f"{'name':5s} {'cost':>4s} {'total':>9s} {'ann':>8s} {'maxDD':>7s} {'Calmar':>7s} {'hit':>6s} {'trades/y':>9s}")
        for nm, d in tab.items():
            for c in ("0", "1", "2", "5"):
                s = d["sel"][c]
                out.append(f"{nm:5s} {c:>4s} {s['total_return']:+9.3f} {s['ann_return']:+8.3f} {s['max_dd']:7.3f} {s['calmar']:7.2f} {s['hit']:6.3f} {s['trades_per_year']:9.0f}")
    out.append(""); out.append("SELECTION TABLE (2 bps): Calmar, total return, trades/year, null p95, eligible")
    for c in T.COMBOS:
        out.append(f"{c}  Calmar {sel['calmar_2bps'][c]:+.2f}  total {sel['total_return_2bps'][c]:+.3f}  maxDD {sel['max_dd_2bps'][c]:.3f}  trades/y {sel['trades_per_year_2bps'][c]:.0f}  "
                   f"null p95 {sel['null_p95_2bps'][c]:+.3f}  eligible {sel['eligible'][c]}")
    out.append(f"CHOSEN: {chosen} (eligible: {sel['chosen_is_eligible']}; any eligible: {sel['any_eligible']})")
    open(f"{OUT}/selection_tables.txt", "w").write("\n".join(out)); print("\n".join(out))


if __name__ == "__main__":
    main()
