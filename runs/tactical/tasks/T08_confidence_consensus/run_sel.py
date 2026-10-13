"""T08 selection on 2019-01 .. 2024-06 ONLY. Writes selection.json (both finalists) + selection_tables.txt; reads no bar on or after 2024-07-01.
The next step (after `git commit` of selection.json) is run_looks.py."""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import t08_lib as L
import scores as SC
import t08_analysis as A
import t07_lib as T


def fmt(res, name):
    s = res[name]["sel"]["2"]
    return (f"{name:16s} Calmar {s['calmar']:+7.2f}  return {s['total_return']:+8.3f}  maxDD {s['max_dd']:6.3f}  trades/y {s['trades_per_year']:7.1f}  "
            f"hit@2bps {s['hit']:.3f}  mean bps {s['mean_bps']:+7.2f}  n {s['n_trades']}")


def main():
    t0 = time.time(); os.makedirs(L.CACHE, exist_ok=True)
    b = SC.build_main(T.SEL_END, T.SEL_LAST, "sel")
    grid = b["grid"]; assert grid.index[-1] < T.SEL_END
    SC.save(f"{L.CACHE}/sel_scores.npz", b)
    S = b["S"]; months = L.month_labels(grid.hour_t); C = L.percentiles(S, months)
    Hr = T.hsum(grid.r); h0 = grid.hour_of("2019-01-01"); h1 = grid.nh
    res, trades = A.eval_configs(S, C, Hr, {"sel": (h0, h1)})
    a, bn = L.config_names()
    fa, fb = A.finalist(res, a), A.finalist(res, bn)
    dec = A.member_deciles(S, C, Hr, h0, h1); cdec = A.consensus_deciles(S, C, Hr, h0, h1)
    cover = {m: float(np.isfinite(S[i, h0:h1]).mean()) for i, m in enumerate(L.MEMBERS)}
    ccov = {m: float(np.isfinite(C[i, h0:h1]).mean()) for i, m in enumerate(L.MEMBERS)}
    bh = L.buy_hold(Hr, h0, h1)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True).stdout.strip()
    sel = {"rule": "per family: highest Calmar at 2 bps on 2019-01..2024-06 among configurations with >= 30 trades per year",
           "finalist_A": fa, "finalist_B": fb,
           "calmar_2bps": {n: res[n]["sel"]["2"]["calmar"] for n in a + bn},
           "total_return_2bps": {n: res[n]["sel"]["2"]["total_return"] for n in a + bn},
           "trades_per_year_2bps": {n: res[n]["sel"]["2"]["trades_per_year"] for n in a + bn},
           "hit_2bps": {n: res[n]["sel"]["2"]["hit"] for n in a + bn},
           "score_coverage_by_member": cover, "data_last_bar_read": str(grid.index[-1]), "code_head_when_selected": sha, "seconds": round(time.time() - t0)}
    json.dump(sel, open(f"{HERE}/selection.json", "w"), indent=1)
    json.dump({"configs": res, "deciles_member": dec, "deciles_consensus": cdec, "buy_hold": bh, "conf_coverage": ccov}, open(f"{HERE}/selection_full.json", "w"), indent=1)
    out = ["Selection period 2019-01-01 .. 2024-06-30 (in-sample of selection, not evidence of skill)", f"hours {h1 - h0}, years {(h1 - h0) / T.HOURS_PER_YEAR:.2f}",
           f"Buy-and-hold (no cost): return {bh['total_return']:+.3f} maxDD {bh['max_dd']:.3f} Calmar {bh['calmar']:.2f}", "",
           "Hourly score coverage by member (share of hours with a score): " + ", ".join(f"{m} {v:.3f}" for m, v in cover.items()),
           "Percentile-confidence coverage: " + ", ".join(f"{m} {v:.3f}" for m, v in ccov.items()), "",
           "FAMILY A (one member, c >= q, 4-hour holds), 2 bps"]
    out += [fmt(res, n) for n in a]
    out += ["", "FAMILY B (consensus), 2 bps"] + [fmt(res, n) for n in bn]
    out += ["", "All configurations at 0 / 1 / 2 / 5 bps: Calmar / return"]
    for n in a + bn:
        out.append(f"{n:16s} " + " | ".join(f"{c}bps {res[n]['sel'][c]['calmar']:+6.2f} {res[n]['sel'][c]['total_return']:+8.3f}" for c in ("0", "1", "2", "5")))
    out += ["", "HIT RATE BY CONFIDENCE DECILE per member, selection (all hourly decisions; gross hit of the member's sign against the 4-hour",
            "open-to-open return; net = after 2 bps; n = decisions; decile 10 is the most confident)"]
    for m, rows in dec.items():
        out.append(f"{m}  " + " ".join(f"{r['hit']:.3f}" for r in rows) + f"   net@2bps: " + " ".join(f"{r['hit_net']:.3f}" for r in rows) + f"   n: " + " ".join(str(r['n']) for r in rows))
    out += ["", "HIT RATE BY CONFIDENCE DECILE of the consensus rules (q-free consensus confidence; K1 min c of M1-M3, K3 min c of all six, K2 third-highest c of",
            "the leading direction without a veto), selection"]
    for k, rows in cdec.items():
        out.append(f"{k}  " + " ".join(f"{r['hit']:.3f}" for r in rows) + f"   net@2bps: " + " ".join(f"{r['hit_net']:.3f}" for r in rows) + f"   n: " + " ".join(str(r['n']) for r in rows))
    out += ["", f"FINALIST A: {fa}", f"FINALIST B: {fb}"]
    open(f"{HERE}/selection_tables.txt", "w").write("\n".join(out)); print("\n".join(out))


if __name__ == "__main__":
    main()
