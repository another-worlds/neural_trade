"""Tables of REPORT.md from results_sel.json / results_look.json / selection.json (no new computation)."""
import json, os, math
H = os.path.dirname(os.path.abspath(__file__))
sel, look, pick = (json.load(open(f"{H}/{f}")) for f in ("results_sel.json", "results_look.json", "selection.json"))
RULES = ["R0_fixed", "R1_voltarget", "R2_volchange_filter", "R3_vt_plus_filter", "R4_brake", "R5_all"]
out = []
def table(res, name, cost):
    rows = ["| rule | trades | hit | mean bps | cum % | max DD % | Calmar | months hit>60% | months DD<5% | avg w | null cum p95 | null DD p5 / med | P(null cum >= actual) |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in RULES:
        L = res["signals"][name]["rules"][r]["levels"][str(cost)]; s, n = L["stats"], L["null"]
        rows.append(f"| {r} | {s['n_trades']} | {s['hit']:.3f} | {s['mean_bps']:+.2f} | {s['cum_return_pct']:+.1f} | {s['max_drawdown_pct']:.1f} | {s['calmar']:+.2f} | {s['months_hit_above_60']}/{s['n_months']} | {s['share_months_dd_below_5']:.2f} | {s['avg_weight']:.2f} | {n['cum_p95']:+.1f} | {n['dd_p05']:.1f} / {n['dd_median']:.1f} | {n['p_cum_ge_actual']:.3f} |")
    return "\n".join(rows)
for ph, res in (("SELECTION 2021-01..2024-06", sel), ("LOOK 2024-07-01..2025-07-20", look)):
    out.append(f"## {ph}\n")
    for name in res["signals"]:
        d = res["signals"][name]
        out.append(f"### {name} ({d['n_signals']} signals; HAR Spearman of the forecast with the realised log-vol of the trade window {d['har_spearman_pred_vs_realised']:.3f}; months {len(d['signal_months'])})\n")
        for c in (2, 0, 1, 5):
            out.append(f"**{c} bps round trip**\n\n" + table(res, name, c) + "\n")
out.append("## Verdicts (look, 2 bps; rules of SPEC.md)\n")
for name, p in pick["picks"].items():
    rule = p or "R0_fixed"
    L = look["signals"][name]["rules"][rule]["levels"]["2"]; s = L["stats"]; n = L["null"]
    b = look["signals"][name]["rules"]["R0_fixed"]["levels"]["2"]["stats"]
    se = math.sqrt(0.25 / s["n_trades"]); lo = s["hit"] - 1.96 * se
    dd_ok = s["max_drawdown_pct"] < 5 and s["cum_return_pct"] > 0 and s["share_months_dd_below_5"] >= 0.9
    hit_ok = s["hit"] > 0.6 and lo > 0.5
    beats = (p is not None) and s["calmar"] > b["calmar"] and s["cum_return_pct"] > n["cum_p95"]
    out.append(f"- **{name}**: chosen rule in selection = {p}. Look at 2 bps ({rule}): {s['n_trades']} trades, hit {s['hit']:.3f} (95% binomial lower bound {lo:.3f}), "
               f"max DD {s['max_drawdown_pct']:.1f}%, cum {s['cum_return_pct']:+.1f}%, months DD<5% {s['share_months_dd_below_5']:.2f}. "
               f"Drawdown goal: {'MET' if dd_ok else 'NOT met'}. Hit goal: {'MET' if hit_ok else 'NOT met'}. Goal met: {'YES' if dd_ok and hit_ok else 'NO'}. "
               f"Sizing beats R0 (Calmar {s['calmar']:+.2f} vs {b['calmar']:+.2f}, and cum above null p95 {n['cum_p95']:+.1f}): {'yes' if beats else 'no' if p else 'n/a (no rule chosen)'}.")
out.append(f"\nRun times: selection {sel['seconds']} s, look {look['seconds']} s (CPU, below-normal priority).")
open(f"{H}/tables.md", "w", encoding="utf-8").write("\n".join(out))
print("\n".join(out[-4:]))
