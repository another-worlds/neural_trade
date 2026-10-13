"""T08 Look 1 (2024-07 .. 2025-07-20) and the forward verdict (2025-11 .. 2026-09 on the Binance file) for the two finalists in selection.json.
Refuses to run unless selection.json is tracked and clean in git (the order in the SPEC is binding)."""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import pandas as pd
import t08_lib as L
import t08_scores as SC
import t08_analysis as A
import t07_lib as T

SEL = f"{HERE}/selection.json"


def fmt_row(nm, d, per):
    cells = []
    for c in ("0", "1", "2", "5"):
        s = d[per][c]
        cells.append(f"{s['total_return']:+7.3f} {s['calmar']:6.2f} {s['max_dd']:6.3f} {s['hit']:5.3f} {s['mean_bps']:+7.1f} {s['trades_per_year']:6.0f}")
    return f"{nm:16s} | " + " | ".join(cells)


def table_text(title, tab, per, bh, names):
    out = ["", title, "(per cost 0 | 1 | 2 | 5 bps: return, Calmar, maxDD, trade hit, mean net bps/trade, trades/yr)"]
    out += [fmt_row(n, tab[n], per) for n in names]
    out.append(f"{'B&H':16s} | return {bh['total_return']:+.3f}  ann {bh['ann_return']:+.3f}  maxDD {bh['max_dd']:.3f}  Calmar {bh['calmar']:.2f}  (no cost)")
    return out


def dec_text(title, dec):
    out = ["", title, "(decile 1..10 by percentile confidence; gross hit | after 2 bps | n)"]
    for k, rows in dec.items():
        out.append(f"{k:3s} hit  " + " ".join(f"{r['hit']:.3f}" for r in rows))
        out.append(f"    net  " + " ".join(f"{r['hit_net']:.3f}" for r in rows))
        out.append(f"    n    " + " ".join(f"{r['n']:5d}" for r in rows))
    return out


def stage_eval(S, hour_t, Hr, slices, names):
    months = L.month_labels(hour_t); C = L.percentiles(S, months)
    res, trades = A.eval_configs(S, C, Hr, slices)
    return C, res, trades


def finalist_verdict(name, ent, side, Hr, hour_t, h0, h1, other):
    k = (ent >= h0) & (ent < h1); e, s = ent[k], side[k]
    st, rh = L.stats_for(ent, side, Hr, L.VERDICT_COST, h0, h1)
    nb = L.trade_net_bps(e, s, Hr, L.VERDICT_COST)
    boot = L.week_bootstrap(nb, pd.DatetimeIndex(hour_t)[e])
    nul = L.null_cum(ent, side, Hr, L.VERDICT_COST, h0, h1, draws=2000)
    return dict(name=name, stats_2bps=st, n_trades=int(k.sum()), bootstrap=boot, null_p50=float(np.quantile(nul, .5)), null_p95=float(np.quantile(nul, .95)),
                null_p05=float(np.quantile(nul, .05)), net_bps_mean=float(nb.mean()) if len(nb) else float("nan"), rh=rh)


def main():
    sha256 = L.assert_selection_committed(SEL)
    sel = json.load(open(SEL)); names = [n for n in (sel["finalist_A"], sel["finalist_B"]) if n]
    a, b = L.config_names(); allnames = a + b
    t0 = time.time(); res = {"selection_json_sha256": sha256, "finalists": names}
    # ------------------------------------------------------------ Look 1 (main file, to the cutoff)
    bm = SC.build_main(None, "2025-07", "look"); grid = bm["grid"]; S = bm["S"]
    SC.save(f"{L.CACHE}/look_scores.npz", bm)
    Hr = T.hsum(grid.r); hs0 = grid.hour_of("2019-01-01"); hl0 = grid.hour_of("2024-07-01"); hl1 = grid.nh
    sel_sc = SC.load(f"{L.CACHE}/sel_scores.npz")
    ns = sel_sc["S"].shape[1]
    a_, b_ = np.nan_to_num(S[:, :ns - 8], nan=9.0), np.nan_to_num(sel_sc["S"][:, :ns - 8], nan=9.0)
    same = bool(S.shape[1] >= ns and np.allclose(a_, b_, atol=1e-6))
    res["sel_scores_identical_in_look_build_except_last_8_hours"] = same
    C, tab, trades = stage_eval(S, grid.hour_t, Hr, {"sel": (hs0, hl0), "look1": (hl0, hl1)}, allnames)
    bh = {"sel": L.buy_hold(Hr, hs0, hl0), "look1": L.buy_hold(Hr, hl0, hl1)}
    ldec_m = A.member_deciles(S, C, Hr, hl0, hl1); ldec_c = A.consensus_deciles(S, C, Hr, hl0, hl1)
    lv = {n: finalist_verdict(n, *trades[n], Hr, grid.hour_t, hl0, hl1, None) for n in names}
    res["look1"] = {"period": "2024-07-01 .. 2025-07-20 23:59", "configs": tab, "buy_hold": bh, "deciles_member": ldec_m, "deciles_consensus": ldec_c,
                    "finalists": {n: {k: v for k, v in d.items() if k != "rh"} for n, d in lv.items()},
                    "sel_recheck_calmar_2bps": {n: tab[n]["sel"]["2"]["calmar"] for n in names}}
    # ------------------------------------------------------------ forward (Binance file)
    fb = SC.build_fwd(); fg = fb["grid"]; fS = fb["S"]
    SC.save(f"{L.CACHE}/fwd_scores.npz", fb)
    assert fg.index[0] == pd.Timestamp("2025-10-01") and fg.index[-1] == T.FWD_LAST
    fHr = T.hsum(fg.r); fh0 = fg.hour_of("2025-11-01"); fh1 = fg.nh
    assert fg.index[fh0 * 60] == pd.Timestamp("2025-11-01")
    fC, ftab, ftrades = stage_eval(fS, fg.hour_t, fHr, {"fwd": (fh0, fh1)}, allnames)
    fbh = L.buy_hold(fHr, fh0, fh1)
    fdec_m = A.member_deciles(fS, fC, fHr, fh0, fh1); fdec_c = A.consensus_deciles(fS, fC, fHr, fh0, fh1)
    fv = {n: finalist_verdict(n, *ftrades[n], fHr, fg.hour_t, fh0, fh1, None) for n in names}
    holm = L.holm({n: fv[n]["bootstrap"]["p"] for n in names}) if names else {}
    verdict = {}
    for n in names:
        d = fv[n]; st = d["stats_2bps"]; boot = d["bootstrap"]
        c1 = d["n_trades"] >= 30
        crit = {"1 at least 30 forward trades": bool(c1),
                "2 mean net bps/trade > 0 with the 95% block-bootstrap lower bound > 0": bool(c1 and boot["mean"] > 0 and boot["lo"] > 0),
                "2h Holm over the two finalists": bool(holm.get(n, False)),
                "3 cumulative return above the random-side null p95": bool(c1 and st["total_return"] > d["null_p95"]),
                "4 max drawdown < 15%": bool(c1 and st["max_dd"] < 0.15)}
        lev, levret = T.leverage_for_dd(d["rh"], 0.05) if len(d["rh"]) else (float("nan"), float("nan"))
        hit0 = ftab[n]["fwd"]["0"]["hit"]
        verdict[n] = dict(criteria=crit, success=bool(c1 and crit["2 mean net bps/trade > 0 with the 95% block-bootstrap lower bound > 0"] and crit["2h Holm over the two finalists"]
                                                      and crit["3 cumulative return above the random-side null p95"] and crit["4 max drawdown < 15%"]),
                          verdict_text=("inconclusive: too few trades" if not c1 else None),
                          stats_2bps=st, bootstrap=boot, null={"p05": d["null_p05"], "p50": d["null_p50"], "p95": d["null_p95"]}, buy_hold=fbh,
                          owner_goal={"trade_hit_2bps": st["hit"], "trade_hit_0bps": hit0, "hit_gt_60_at_2bps": bool(st["hit"] > 0.6), "max_dd": st["max_dd"],
                                      "max_dd_lt_5pct": bool(st["max_dd"] < 0.05), "leverage_for_dd_lt_5pct": lev, "return_at_that_leverage": levret,
                                      "trades_per_year": st["trades_per_year"]})
    res["forward"] = {"period": "2025-11-01 .. 2026-09-30 23:59 (calibration month 2025-10)", "configs": ftab, "buy_hold": fbh, "deciles_member": fdec_m,
                      "deciles_consensus": fdec_c, "holm_p": {n: fv[n]["bootstrap"]["p"] for n in names}, "holm_rejected": holm, "verdict": verdict}
    res["seconds"] = round(time.time() - t0)
    json.dump(res, open(f"{HERE}/results.json", "w"), indent=1, default=float)
    # ------------------------------------------------------------ tables.txt
    out = [f"T08 results. selection.json sha256 {sha256}; finalists {names}"]
    out += table_text("FINALISTS, selection 2019-01..2024-06 (in-sample of selection)", tab, "sel", bh["sel"], names)
    out += table_text("FINALISTS, Look 1 2024-07..2025-07-20", tab, "look1", bh["look1"], names)
    out += table_text("FINALISTS, forward 2025-11..2026-09 (frozen models)", ftab, "fwd", fbh, names)
    out += table_text("ALL 27 CONFIGURATIONS, Look 1 (transparency)", tab, "look1", bh["look1"], allnames)
    out += table_text("ALL 27 CONFIGURATIONS, forward (transparency; no choice uses it)", ftab, "fwd", fbh, allnames)
    out += dec_text("HIT BY CONFIDENCE DECILE per member, Look 1", ldec_m) + dec_text("HIT BY CONFIDENCE DECILE of the consensus rules, Look 1", ldec_c)
    out += dec_text("HIT BY CONFIDENCE DECILE per member, FORWARD", fdec_m) + dec_text("HIT BY CONFIDENCE DECILE of the consensus rules, FORWARD", fdec_c)
    for n in names:
        d = lv[n]; out.append(f"Look 1 {n} at 2 bps: trades {d['n_trades']}, mean {d['net_bps_mean']:+.2f} bps, week-bootstrap 95% [{d['bootstrap']['lo']:+.2f}, {d['bootstrap']['hi']:+.2f}], "
                              f"null p95 {d['null_p95']:+.3f}, return {d['stats_2bps']['total_return']:+.3f}")
    for n in names:
        v = verdict[n]; d = fv[n]
        out += ["", f"FORWARD VERDICT {n} at 2 bps: success = {v['success']}" + (f"  ({v['verdict_text']})" if v["verdict_text"] else "")]
        out += [f"  {k}: {'PASS' if x else 'FAIL'}" for k, x in v["criteria"].items()]
        out += [f"  trades {d['n_trades']}, mean net {d['net_bps_mean']:+.2f} bps/trade, week-bootstrap 95% [{d['bootstrap']['lo']:+.2f}, {d['bootstrap']['hi']:+.2f}], p {d['bootstrap']['p']:.4f}, weeks {d['bootstrap']['n_weeks']}",
                f"  return {d['stats_2bps']['total_return']:+.4f}, null p05 {d['null_p05']:+.4f} p50 {d['null_p50']:+.4f} p95 {d['null_p95']:+.4f}, maxDD {d['stats_2bps']['max_dd']:.4f}, B&H {fbh['total_return']:+.3f} (DD {fbh['max_dd']:.3f})",
                f"  owner goal: {json.dumps(v['owner_goal'], default=float)}"]
    open(f"{HERE}/tables.txt", "w").write("\n".join(out)); print("\n".join(out))


if __name__ == "__main__":
    main()
