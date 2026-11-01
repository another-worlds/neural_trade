"""Tables for the write-up from results/*.json (dev) and results/seal.json (after the seal run).

  python report.py   -> results/tables.md
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
ORDER = {1: ["T1-FADEV", "T1-LIN", "T1-NET", "T1-NT", "T1-NET-W", "plain_fade", "intraday_trend", "B&H", "cash"],
         2: ["T2-P", "T2-LIN", "T2-NET", "T2-NET-W", "tsmom_unscaled", "voltarget_long", "fade60", "B&H", "cash"]}


def f(v, nd=0):
    return "-" if v is None else f"{v:,.{nd}f}"


def dev_tables():
    runs = json.load(open(os.path.join(RES, "dev_runs.json")))
    gate = json.load(open(os.path.join(RES, "gate.json")))
    md = []
    for t in (1, 2):
        g = gate["tracks"][str(t)]
        md += [f"### Track {t}: dev OOS 2021-01-01..2025-10-31 (1,765 days), net of fees and funding", "",
               "| strategy | C7-B %/yr | Sharpe C7-B | C7-S %/yr | C4-B %/yr | C10-B %/yr | trades/yr | mean x | "
               "G1 p | gates passed |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
        for name in ORDER[t]:
            if name not in runs:
                continue
            sc = runs[name]["scenarios"]
            gi = g["candidates"].get(name)
            tr = sc["C7B"].get("trades") or {}
            gates = "-"
            if gi:
                gates = ", ".join(k for k, ok in gi["gates"].items() if ok) or "none"
            md.append(f"| {name} | {f(sc['C7B']['mean_daily_bps'] * 3.65, 1)} | {f(sc['C7B']['sharpe'], 2)} | "
                      f"{f(sc['C7S']['mean_daily_bps'] * 3.65, 1)} | {f(sc['C4B']['mean_daily_bps'] * 3.65, 1)} | "
                      f"{f(sc['C10B']['mean_daily_bps'] * 3.65, 1)} | {f(tr.get('trades_per_year'))} | "
                      f"{f(sc['C7B'].get('mean_x'), 2)} | {f(gi.get('p_maxT') if gi else None, 3)} | {gates} |")
        md += ["", f"p_sel (track) = {g['p_sel']:.3f}; deployed: **{g['deployed']}**; information only: "
               f"{g['information_only']}", ""]
    return md


def seal_tables():
    p = os.path.join(RES, "seal.json")
    if not os.path.exists(p):
        return ["(seal not run)"]
    s = json.load(open(p))
    md = ["### Sealed test 2025-11-01..2026-10-30 (frozen fold-S models)", "",
          "| strategy | C7-B full bps/day | C7-B clean bps/day | Sharpe full | C7-S full | C10-S full | trades |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for name, v in s["strategies"].items():
        md.append(f"| {name} | {f(v['C7B']['full']['mean_daily_bps'], 2)} | {f(v['C7B']['clean']['mean_daily_bps'], 2)} | "
                  f"{f(v['C7B']['full']['sharpe'], 2)} | {f(v['C7S']['full']['mean_daily_bps'], 2)} | "
                  f"{f(v['C10S']['full']['mean_daily_bps'], 2)} | {v['C7B']['trades']} |")
    md += ["", "Verdicts:", ""] + [f"- {k}: **{v['verdict']}** ({v['role']})" for k, v in s["verdicts"].items()]
    return md


def main():
    md = ["# PnL study tables", ""] + dev_tables() + [""] + seal_tables()
    open(os.path.join(RES, "tables.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
