"""usage: python run_stops.py sel | select | look      (SPEC.md)
sel    : signals (both side signals, both threshold versions) and HAR state on the SELECTION months 2021-01..2024-06 (data truncated before
         2024-07-01); the six exit rules plus the derived X6; writes results_sel.json
select : picks best5 and the chosen rule per combination (Calmar at 2 bps), writes selection.json
look   : refuses unless selection.json exists; the ONE look 2024-07-01..2025-07-20, writes results_look.json
"""
import json, os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); TAC = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(TAC, "sizing"))
import stoplib as L  # noqa: E402
import run_sizing as R  # noqa: E402  (a_preds, b_preds, build_look_cache of the sizing study; imports wf and fwdlib)

S = L.S
COSTS = L.E.COSTS
# phase: (first signal month, last month, data end (exclusive), period start, period end)
PHASES = {"sel": ("2021-01", "2024-06", "2024-07-01", "2021-01-01", "2024-07-01"),
          "look": ("2024-07", "2025-07", "2025-07-21", "2024-07-01", "2025-07-21")}
NAMES = (("a_1h_rich_l1_365", 1), ("b_h62_4h_rich_logreg_730", 4))
VERSIONS = ("prev", "same")


def evaluate(pre, sg, keep_f, start_ns, end_ns, best5, draws=500):
    """All rules of one combination. pre: precompute() output (X0..X5); X6 = pre[best5] with the filter."""
    entry, side = sg["entry"], sg["side"]; ones = np.ones(len(entry), bool)
    res = {}
    for rule in L.ALL_RULES:
        key, keep = (best5, keep_f) if rule == "X6_best_plus_filter" else (rule, ones)
        acts = {c: L.actual_stats(pre, key, entry, side, keep, c, start_ns, end_ns) for c in COSTS}
        null = L.null_stats(pre, key, entry, keep, COSTS, {c: (acts[c][0].get("cum_return_pct", 0.0) if acts[c][0]["n_trades"] else 0.0) for c in COSTS}, draws)
        levels = {}
        for c in COSTS:
            s, det = acts[c]
            levels[str(c)] = {"stats": s, "null": null[str(c)], "frontier": L.frontier(det["pnl"]) if s["n_trades"] else None}
        res[rule] = {"underlying": key, "levels": levels}
    return res


def run_phase(phase):
    first_sig, last, data_end, p0, p1 = PHASES[phase]; t0 = time.time(); R.wf.lowprio()
    first_pred = str(pd.Period(first_sig, "M") - 1)
    sel = json.load(open(f"{HERE}/selection.json")) if phase == "look" else None
    rv = S.hourly_rv(end=data_end); times = rv.index.values.astype("datetime64[ns]")
    assert rv.index.max() < pd.Timestamp(data_end) and rv.index.max() <= pd.Timestamp(S.C.CUTOFF)
    mins = L.load_minutes(pd.Period(first_pred, "M").start_time - pd.Timedelta(days=1), data_end)
    assert mins["t"].max() < pd.Timestamp(data_end).value and mins["t"].max() <= pd.Timestamp(S.C.CUTOFF).value
    print(f"{phase}: {len(rv)} hourly bars, {len(mins['t'])} minute bars to {pd.Timestamp(mins['t'].max())} ({time.time() - t0:.0f}s)", flush=True)
    if phase == "look":
        R.build_look_cache()
    cache = R.F.CACHE if phase == "sel" else R.CACHE_LOOK
    tcache = np.load(f"{cache}/t.npy"); assert pd.Timestamp(tcache.max()) < pd.Timestamp(data_end)
    start_ns, end_ns = int(pd.Timestamp(p0).value), int(pd.Timestamp(p1).value)
    out = {"phase": phase, "first_signal_month": first_sig, "last_month": last, "period": [p0, p1], "combos": {}}
    gens = {"a_1h_rich_l1_365": lambda: R.a_preds(first_pred, last, data_end), "b_h62_4h_rich_logreg_730": lambda: R.b_preds(first_pred, last, cache)}
    for name, H in NAMES:
        preds = list(gens[name]())
        st = S.state(rv.values, times, H, "2018-01", last)
        for version in VERSIONS:
            sg = L.signals_v(preds, H, version, first_sig)
            combo = f"{name}|{version}"
            bar = pd.Index(times).get_indexer(sg["t_open"].astype("datetime64[ns]")); assert (bar >= 0).all()
            for k in ("sigma", "target", "dlv", "thr"):
                assert np.isfinite(st[k][bar]).all(), f"{combo}: {k} not available at some signal"
            sig_trade = st["sigma"][bar] * np.sqrt(H)
            pre = L.precompute(mins, sg["entry"], sg["exit"], sig_trade)
            valid = np.isfinite(pre["X0_time"]["gross"]).all(1)
            # check: the X0 gross equals the frozen hourly r_long (entry minute / last horizon minute equal the hourly open / close)
            diff = np.abs(pre["X0_time"]["gross"][valid, 0] - sg["r_long"][valid])
            chk = {"n_signals": int(len(bar)), "n_invalid": int((~valid).sum()), "max_abs_diff_vs_r_long": float(diff.max()), "n_diff_gt_1e-9": int((diff > 1e-9).sum())}
            if not valid.all():
                sg = {k: v[valid] for k, v in sg.items()}; bar = bar[valid]
                pre = {r: {k: v[valid] for k, v in d.items()} for r, d in pre.items()}
            keep_f = ~(st["dlv"][bar] > st["thr"][bar])
            if phase == "sel":
                cal = {r: L.actual_stats(pre, r, sg["entry"], sg["side"], np.ones(len(bar), bool), 2, start_ns, end_ns)[0]["calmar"] for r in L.STOP_RULES}
                best5 = max(cal, key=cal.get)
            else:
                best5 = sel["combos"][combo]["best5"]
            res = evaluate(pre, sg, keep_f, start_ns, end_ns, best5)
            out["combos"][combo] = {"H": H, "version": version, "check_x0_vs_r_long": chk, "best5": best5, "n_signals": int(len(bar)),
                                    "signal_months": sorted(set(sg["month"])), "n_filtered_out": int((~keep_f).sum()), "rules": res}
            print(combo, chk, "best5", best5, f"({time.time() - t0:.0f}s)", flush=True)
            for rule in L.ALL_RULES:
                s = res[rule]["levels"]["2"]["stats"]
                if s["n_trades"] == 0:
                    print(f"  {rule:20s} no trades"); continue
                print(f"  {rule:20s} n {s['n_trades']:5d} hit {s['hit']:.3f} mean {s['mean_bps']:+7.2f} cum {s['cum_return_pct']:+8.1f}% "
                      f"dd {s['max_drawdown_pct']:5.1f}% calmar {s['calmar']:+6.2f} m>60 {s['months_hit_above_60']}/{s['n_months']} "
                      f"ddm<5 {s['share_months_dd_below_5']:.2f} stop {s['share_stop']:.2f} tp {s['share_tp']:.2f}", flush=True)
    out["seconds"] = round(time.time() - t0, 1)
    json.dump(out, open(f"{HERE}/results_{phase}.json", "w"), indent=1)
    print(f"done {out['seconds']}s")


def select():
    r = json.load(open(f"{HERE}/results_sel.json")); picks = {}
    for combo, d in r["combos"].items():
        cal = {rule: d["rules"][rule]["levels"]["2"]["stats"].get("calmar", float("-inf")) for rule in L.ALL_RULES}
        base = cal["X0_time"]
        best = max((x for x in L.ALL_RULES if x != "X0_time"), key=lambda x: cal[x])
        picks[combo] = {"best5": d["best5"], "chosen": best if cal[best] > base else None, "calmar_2bps_sel": cal}
        print(combo, {k: round(v, 3) for k, v in cal.items()}, "-> best5", d["best5"], "chosen", picks[combo]["chosen"])
    json.dump({"combos": picks}, open(f"{HERE}/selection.json", "w"), indent=1)


if __name__ == "__main__":
    ph = sys.argv[1]
    if ph == "select":
        select()
    elif ph == "look":
        if not os.path.exists(f"{HERE}/selection.json"):
            sys.exit("look refused: selection.json missing")
        run_phase("look")
    else:
        run_phase("sel")
