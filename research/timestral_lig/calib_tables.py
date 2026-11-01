"""Markdown tables from calibration.json (v2: selection-adjusted gate, Holm, permutation p for test-picked rules)."""
import json
import os
import sys

import train

R = json.load(open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(train.WORK, "calibration.json")))


def rule_txt(p):
    if not p:
        return "–"
    filt = "no conf filter" if p["keep"] == 1 else f"{p['conf']} top {p['keep']:.0%}"
    return f"H={p['H']}, {p['align']}, {filt}, {p['side']}"


def acc(t):
    if not t or "accuracy" not in t:
        return "no trades"
    s = f"{t['accuracy']:.1%} [{t['acc_ci95'][0]:.1%}, {t['acc_ci95'][1]:.1%}]"
    return s + (" (too few trades)" if t.get("insufficient") else "")


def money(t):
    if not t or "pnl_bps" not in t:
        return "– | – | –"
    return f"{t['pnl_bps']:+.0f} (t {t['t']}) | {t['net_bps_cost4']:+.0f} | {t['net_bps_cost10']:+.0f}"


for ds, d in R.items():
    for proto, label in (("P-A", "calibrated on pre-Test forecasts, evaluated on the full Test"),
                         ("P-B", "calibrated on the first half of Test, evaluated on the second half")):
        print(f"\n#### {ds}, {proto} ({label})\n")
        print("| Model | Rule | Chosen on calibration | Calib acc (p_sel) | Decision | Test accuracy [95% CI] | "
              "p one-sided (Holm) | Trades / decided (per day) | Gross bps (t) | Net 4 bps | Net 10 bps |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for name, r in d[proto].items():
            for rule in ("W", "R"):
                x = r[rule]
                p = x.get("params")
                cal = f"{p['calib_accuracy']:.1%} ({p['p_sel']:.3f})" if p else "–"
                if x.get("decision") == "deploy":
                    t = x["test"]
                    pv = f"{t['p_one_sided']:.3f} ({t.get('p_holm', float('nan')):.2f})" if "p_one_sided" in t else "–"
                    print(f"| {name} | {rule} | {rule_txt(p)} | {cal} | deploy | {acc(t)} | {pv} | "
                          f"{t['trades']} / {t['decided']} ({t['per_day']}) | {money(t)} |")
                else:
                    t = x.get("test_if_deployed")
                    print(f"| {name} | {rule} | {rule_txt(p)} | {cal} | **abstain** | (if deployed: {acc(t)}) | – | "
                          f"{(str(t['trades']) + ' / ' + str(t['decided'])) if t else '–'} | {money(t) if t else '– | – | –'} |")
        print(f"\nDescriptive only, rule picked on the evaluation data itself (p_sel = permutation p of the best of the grid):\n")
        print("| Model | Picked rule | Accuracy | Decided trades | p_sel | Gross bps |")
        print("|---|---|---|---|---|---|")
        for name, r in d[proto].items():
            x = r.get("test_picked")
            if x:
                t, p = x["test"], x["params"]
                print(f"| {name} | {rule_txt(p)} | {t['accuracy']:.1%} | {t['decided']} | {p['p_sel']:.3f} | {t['pnl_bps']:+.0f} |")
