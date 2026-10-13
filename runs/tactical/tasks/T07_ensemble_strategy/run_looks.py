"""T07 Look 1 (2024-07 .. 2025-07-20) and the forward verdict (2025-11 .. 2026-09 on the Binance file), for the combination in selection.json.
Refuses to run unless selection.json is tracked and clean in git (the order in the SPEC is binding). usage: python run_looks.py [--workers 6]"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import pandas as pd
import t07_lib as T
import analysis as A
import pipeline as P

OUT = HERE
SEL = f"{OUT}/selection.json"


def fmt_row(nm, d, per):
    cells = []
    for c in ("0", "1", "2", "5"):
        s = d[per][c]
        cells.append(f"{s['total_return']:+7.3f} {s['calmar']:6.2f} {s['max_dd']:6.3f} {s['hit']:5.3f} {s['trades_per_year']:6.0f}")
    return f"{nm:5s} | " + " | ".join(cells)


def table_text(title, tab, per, bh):
    out = ["", title, "(per cost 0 | 1 | 2 | 5 bps: return, Calmar, maxDD, trade hit, trades/yr)"]
    for nm, d in tab.items():
        out.append(fmt_row(nm, d, per))
    s = bh[per]
    out.append(f"B&H   | return {s['total_return']:+.3f}  ann {s['ann_return']:+.3f}  maxDD {s['max_dd']:.3f}  Calmar {s['calmar']:.2f}  (no cost)")
    return out


def main():
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 6
    sha256 = T.assert_selection_committed(SEL)
    sel = json.load(open(SEL)); chosen = sel["chosen"]
    t0 = time.time(); res = {"selection_json_sha256": sha256, "chosen": chosen, "chosen_eligible_in_selection": sel["chosen_is_eligible"]}
    # ------------------------------------------------------------ Look 1 (main file, to the cutoff)
    b = P.build_main("look", None, "2025-07", workers)
    grid = b["grid"]; h0 = grid.hour_of("2019-01-01"); hs = grid.hour_of("2024-07-01") - h0; h1 = grid.nh
    view = grid.view(h0, h1); pos = b["pos"][:, h0 * 60:h1 * 60]; har = T.har_scale(b["sig"][h0:h1], b["med"][h0:h1])
    combos, (g, u, p) = A.combos_for(view, pos, har)
    sl = {"sel": slice(0, hs), "look1": slice(hs, view.nh)}
    mem, r2m = A.table_for({m: pos[i] for i, m in enumerate(T.MEMBERS)}, view, sl)
    com, r2c = A.table_for(combos, view, sl)
    dg = {"D3": T.vote_minute(pos)}; dg["D6"] = dg["D3"] * np.repeat(har, 60)
    dgt, _ = A.table_for(dg, view, sl)
    bh = A.buy_hold(view, sl)
    res["look1"] = {"diagnostic_minute_vote": dgt, "members": mem, "combos": com, "buy_hold": bh, "null": {}, "period": "2024-07-01 .. 2025-07-20 23:59",
                    "sel_recheck_calmar_2bps": {c: com[c]["sel"]["2"]["calmar"] for c in T.COMBOS}}
    for c in T.COMBOS:
        res["look1"]["null"][c] = A.null_summary(combos[c], view, sl["look1"], T.VERDICT_COST)
    res["look1"]["monthly_chosen_2bps"] = A.monthly_summary(r2c[(chosen, "look1")], view.hour_t[sl["look1"]])
    res["look1"]["corr_2bps"] = A.correlation(T.net(g, u, 2)[:, sl["look1"]]).round(3).tolist()
    res["meta_look"] = {k: {kk: vv for kk, vv in v.items() if kk != "p"} for k, v in b["meta"].items()}
    np.savez(f"{T.CACHE}/look_hourly.npz", g=g, u=u, p=p, hour_t=view.hour_t.values.astype("datetime64[m]"),
             **{f"c_{c}": combos[c].astype(np.float32) for c in T.COMBOS}, r=view.r)
    # ------------------------------------------------------------ forward (Binance file)
    f = P.build_fwd(b, workers=workers)
    fg = f["grid"]; fh0 = fg.hour_of("2025-11-01"); fh1 = fg.nh
    assert fg.index[fh0 * 60] == pd.Timestamp("2025-11-01") and fg.index[-1] == T.FWD_LAST
    fview = fg.view(0, fg.nh); fhar = T.har_scale(f["sig"], f["sig_star"])
    fcombos, (fg_, fu_, fp_) = A.combos_for(fview, f["pos"], fhar)
    fsl = {"fwd": slice(fh0, fg.nh)}
    fmem, fr2m = A.table_for({m: f["pos"][i] for i, m in enumerate(T.MEMBERS)}, fview, fsl)
    fcom, fr2c = A.table_for(fcombos, fview, fsl)
    fdg = {"D3": T.vote_minute(f["pos"])}; fdg["D6"] = fdg["D3"] * np.repeat(fhar, 60)
    fdgt, _ = A.table_for(fdg, fview, fsl)
    fbh = A.buy_hold(fview, fsl)
    fnull = {c: A.null_summary(fcombos[c], fview, fsl["fwd"], T.VERDICT_COST) for c in T.COMBOS}
    ht = fview.hour_t[fsl["fwd"]]
    fres = {"diagnostic_minute_vote": fdgt, "members": fmem, "combos": fcom, "buy_hold": fbh, "null": fnull, "period": "2025-11-01 .. 2026-09-30 23:59 (calibration month 2025-10)",
            "sigma_star": f["sig_star"], "meta": {k: {kk: vv for kk, vv in v.items() if kk != "p"} for k, v in f["meta"].items()},
            "corr_2bps": A.correlation(T.net(fg_, fu_, 2)[:, fsl["fwd"]]).round(3).tolist(),
            "monthly": {c: A.monthly_summary(fr2c[(c, "fwd")], ht) for c in T.COMBOS}}
    # verdict on the chosen combination
    rh = fr2c[(chosen, "fwd")]; st = fcom[chosen]["fwd"]["2"]; mo = fres["monthly"][chosen]
    crit = {"1 mean monthly net return > 0 and 95% t lower bound > 0": bool(mo["mean"] > 0 and mo["ci95"][0] > 0),
            "2 cumulative return above the null p95": bool(st["total_return"] > fnull[chosen]["p95"]),
            "3 Calmar above buy-and-hold's": bool(np.isfinite(st["calmar"]) and st["calmar"] > fbh["fwd"]["calmar"]),
            "4 max drawdown < 15%": bool(st["max_dd"] < 0.15)}
    lev, levret = T.leverage_for_dd(rh, 0.05)
    fres["verdict"] = {"chosen": chosen, "criteria": crit, "success": bool(all(crit.values())), "stats_2bps": st, "null_2bps": fnull[chosen],
                       "monthly": mo, "buy_hold": fbh["fwd"],
                       "owner_goal": {"hit_gt_60": bool(st["hit"] > 0.6), "trade_hit": st["hit"], "max_dd_lt_5pct": bool(st["max_dd"] < 0.05),
                                      "leverage_for_dd_lt_5pct": lev, "return_at_that_leverage": levret,
                                      "months_positive_share": mo["share_positive"]}}
    res["forward"] = fres
    np.savez(f"{T.CACHE}/fwd_hourly.npz", g=fg_, u=fu_, p=fp_, hour_t=fview.hour_t.values.astype("datetime64[m]"),
             **{f"c_{c}": fcombos[c].astype(np.float32) for c in T.COMBOS}, r=fview.r, fh0=fh0)
    res["seconds"] = round(time.time() - t0)
    json.dump(res, open(f"{OUT}/results.json", "w"), indent=1, default=float)
    # ------------------------------------------------------------ tables.txt
    out = [f"T07 results. selection.json sha256 {sha256}; chosen {chosen} (eligible in selection: {sel['chosen_is_eligible']})"]
    out += table_text("MEMBERS, selection 2019-01..2024-06 (in-sample of selection)", mem, "sel", bh)
    out += table_text("COMBINATIONS, selection (recomputed on the longer history; see recheck in results.json)", com, "sel", bh)
    out += table_text("MEMBERS, Look 1 2024-07..2025-07-20", mem, "look1", bh)
    out += table_text("COMBINATIONS, Look 1", com, "look1", bh)
    out += table_text("MEMBERS, forward 2025-11..2026-09 (frozen models)", fmem, "fwd", fbh)
    out += table_text("COMBINATIONS, forward", fcom, "fwd", fbh)
    out += table_text("DIAGNOSTIC (not one of the six, not selectable): the vote per minute, Look 1", dgt, "look1", bh)
    out += table_text("DIAGNOSTIC: the vote per minute, forward", fdgt, "fwd", fbh)
    for per, key in (("Look 1", res["look1"]), ("forward", fres)):
        out += ["", f"Random-side null at 2 bps, cumulative return, {per}: " + ", ".join(f"{c} p50 {v['p50']:+.3f} p95 {v['p95']:+.3f}" for c, v in key["null"].items())]
        cm = key["corr_2bps"]
        out += [f"Member correlation of hourly own-book net returns, {per}:", "      " + " ".join(f"{m:>6s}" for m in T.MEMBERS)]
        out += [f"{T.MEMBERS[i]:5s} " + " ".join(f"{x:+6.2f}" for x in row) for i, row in enumerate(cm)]
    v = fres["verdict"]
    out += ["", f"FORWARD VERDICT for {chosen} at 2 bps: success = {v['success']}"] + [f"  {k}: {'PASS' if x else 'FAIL'}" for k, x in v["criteria"].items()]
    out += [f"  stats: {json.dumps({k: round(x, 4) if isinstance(x, float) else x for k, x in v['stats_2bps'].items()})}",
            f"  null p50 {v['null_2bps']['p50']:+.4f} p95 {v['null_2bps']['p95']:+.4f}; B&H {json.dumps({k: round(x, 4) for k, x in v['buy_hold'].items()})}",
            f"  monthly mean {v['monthly']['mean']:+.5f} CI95 [{v['monthly']['ci95'][0]:+.5f}, {v['monthly']['ci95'][1]:+.5f}], positive months {v['monthly']['share_positive']:.2f} of {v['monthly']['n_months']}",
            f"  owner goal: {json.dumps(v['owner_goal'])}"]
    open(f"{OUT}/tables.txt", "w").write("\n".join(out)); print("\n".join(out))


if __name__ == "__main__":
    main()
