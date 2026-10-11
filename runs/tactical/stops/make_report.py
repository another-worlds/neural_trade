"""Builds tables.md (all generated numbers) from results_sel.json, selection.json, results_look.json. usage: python make_report.py"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RULES = ("X0_time", "X1_sym1", "X2_sym15", "X3_sym2", "X4_asym", "X5_trail1", "X6_best_plus_filter")
sel = json.load(open(f"{HERE}/selection.json"))["combos"]


def row(rule, d):
    lv = d["rules"][rule]["levels"]["2"]; s = lv["stats"]; nl = lv["null"]
    if s["n_trades"] == 0:
        return f"| {rule} | 0 |" + " |" * 11
    und = d["rules"][rule]["underlying"]
    name = rule if rule != "X6_best_plus_filter" else f"X6 ({und}+filter)"
    return (f"| {name} | {s['n_trades']} | {s['hit']:.3f} | {s['mean_bps']:+.2f} | {s['cum_return_pct']:+.1f} | {s['max_drawdown_pct']:.1f} | "
            f"{s['calmar']:+.2f} | {s['months_hit_above_60']}/{s['n_months']} | {s['share_months_dd_below_5']:.2f} | "
            f"{s['share_stop']:.2f}/{s['share_tp']:.2f} | {nl['cum_p95']:+.1f} | {nl['p_cum_ge_actual']:.3f} |")


HEAD = ("| rule | n | hit | mean bps | cum % | max DD % | Calmar | months hit>60 | months DD<5 | stop/tp share | null cum p95 | P(null>=) |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|")


def phase_tables(res, title, chosen_of):
    out = [f"## {title}\n"]
    for combo, d in res["combos"].items():
        chk = d["check_x0_vs_r_long"]
        out.append(f"### {combo} (signals {d['n_signals']}, filtered by the vol-change rule {d['n_filtered_out']}, best5 {d['best5']}, "
                   f"chosen in selection: {chosen_of(combo)}; X0 vs hourly r_long: max abs diff {chk['max_abs_diff_vs_r_long']:.1e}, "
                   f"{chk['n_diff_gt_1e-9']} differ, {chk['n_invalid']} without minute path)\n")
        out.append("At 2 bps:\n\n" + HEAD)
        out += [row(r, d) for r in RULES]
        out.append("")
        out.append("Costs 0/1/2/5 bps (cum % / max DD % / hit):\n\n| rule | 0 | 1 | 2 | 5 |\n|---|---|---|---|---|")
        for r in RULES:
            cells = []
            for c in ("0", "1", "2", "5"):
                s = d["rules"][r]["levels"][c]["stats"]
                cells.append("n/a" if s["n_trades"] == 0 else f"{s['cum_return_pct']:+.1f} / {s['max_drawdown_pct']:.1f} / {s['hit']:.3f}")
            out.append(f"| {r} | " + " | ".join(cells) + " |")
        out.append("")
    return out


def frontier_tables(res, title, chosen_of):
    out = [f"## {title}\n", "Fixed notional, cumulative return % / max DD % at weight w (pnl and cost x w), 2 bps; `w(DD<5)` the largest weight on a 0.01 grid "
           "with max DD < 5% and the return there.\n"]
    for combo, d in res["combos"].items():
        rules = ["X0_time"]; ch = chosen_of(combo)
        if ch:
            rules.append(ch)
        out.append(f"### {combo}\n")
        out.append("| rule | " + " | ".join(f"{w:.1f}" for w in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)) + " | w(DD<5) | cum % there |\n|---|" + "---|" * 12)
        for r in rules:
            f = d["rules"][r]["levels"]["2"]["frontier"]
            if f is None:
                continue
            out.append(f"| {r} | " + " | ".join(f"{x['cum_pct']:+.1f}/{x['dd_pct']:.1f}" for x in f["rows"]) + f" | {f['w_dd5']:.2f} | {f['cum_at_w_dd5']:+.1f} |")
        out.append("")
    return out


def verdicts(res):
    out = ["## Verdicts on the look (pre-registered rules, 2 bps)\n", "| combination | chosen | n | hit | cum % | max DD % | months DD<5 | DD goal | hit goal | beats X0 (Calmar and null p95) |\n|---|---|---|---|---|---|---|---|---|---|"]
    for combo, d in res["combos"].items():
        ch = sel[combo]["chosen"]
        if ch is None:
            out.append(f"| {combo} | none | | | | | | no | no | no rule chosen: \"no\" |"); continue
        lv = d["rules"][ch]["levels"]["2"]; s = lv["stats"]; b = d["rules"]["X0_time"]["levels"]["2"]["stats"]
        if s["n_trades"] == 0:
            out.append(f"| {combo} | {ch} | 0 | | | | | no | no | no |"); continue
        n = s["n_trades"]; hit = s["hit"]; se = (0.25 / n) ** 0.5
        dd_goal = s["max_drawdown_pct"] < 5 and s["cum_return_pct"] > 0 and s["share_months_dd_below_5"] >= 0.9 and n >= 100
        hit_goal = hit > 0.60 and hit - 1.96 * ((hit * (1 - hit)) / n) ** 0.5 > 0.5
        beats = s["calmar"] > b["calmar"] and s["cum_return_pct"] > lv["null"]["cum_p95"]
        out.append(f"| {combo} | {ch} | {n} | {hit:.3f} | {s['cum_return_pct']:+.1f} | {s['max_drawdown_pct']:.1f} | {s['share_months_dd_below_5']:.2f} | "
                   f"{'YES' if dd_goal else 'no'} | {'YES' if hit_goal else 'no'} | {'YES' if beats else 'no'} (Calmar {s['calmar']:+.2f} vs X0 {b['calmar']:+.2f}; null p95 {lv['null']['cum_p95']:+.1f}) |")
    return out


if __name__ == "__main__":
    rs = json.load(open(f"{HERE}/results_sel.json")); rl = json.load(open(f"{HERE}/results_look.json"))
    ch = lambda c: sel[c]["chosen"]
    lines = ["# Stops study: generated tables (make_report.py; every number comes from results_sel.json / selection.json / results_look.json)\n"]
    lines += phase_tables(rs, "Selection 2021-01..2024-06 (all rules; the choice used the Calmar at 2 bps)", ch)
    lines += phase_tables(rl, "The ONE look 2024-07-01..2025-07-20", ch)
    lines += verdicts(rl)
    lines += [""] + frontier_tables(rs, "Leverage frontier, selection", ch) + frontier_tables(rl, "Leverage frontier, look", ch)
    open(f"{HERE}/tables.md", "w").write("\n".join(lines) + "\n")
    print("wrote tables.md")
