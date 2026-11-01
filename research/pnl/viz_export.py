"""Data export for the PnL viewer (descriptive; after the seal was scored once).

Re-simulates each shown strategy at C7-B with engine E over dev OOS (2021-01-01..2025-10-31) and over the sealed year
with the frozen fold-S models, and asserts that the daily net log returns are bit-identical to the stored results
(results/dev_daily via $PNL_WORK, results/seal_daily.npz). Nothing is re-scored; the seal access is logged by data.py.

Writes to OUT (default $PNL_WORK/viewer):
  meta.json                     strategies, stats (dev and seal), time origin, seal boundary
  price_hour.bin                float32 close at the end of every UTC hour from 2021-01-01 (NaN where missing)
  m/YYYY-MM.txt                 base64 of float32 1-min closes for each month (NaN for missing minutes)
  s/<id>.json                   per strategy: trades, exposure change points, PnL points (base64 typed arrays)

  PNL_UNSEAL=<sha256 of prereg.json> python viz_export.py [OUT]
"""
import base64
import json
import os
import sys

import numpy as np

import data as D
import features as FE
import folds as FO
import prep
import rules
import run_dev
import run_seal
import sim

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
SHOW = [("T1-LIN", 1), ("T1-NET", 1), ("T1-FADEV", 1), ("T1-NT", 1), ("T2-P", 2), ("T2-NET", 2), ("T2-LIN", 2),
        ("B&H", 0), ("tsmom_unscaled", 0)]
T0 = FO.DEV_OOS[0]                                   # 2021-01-01 00:00 UTC


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def run(b, fac, s, e):
    return sim.simulate(b, fac(), s, e, fee="C7", fill="B", record_exposure=True)


def pieces(b, r, s, e, log_offset):
    """Per-bar close times (minutes), exposure and chained log equity for one run. The run closes any open position
    as taker after the last recorded close, so the last point takes the scored total and a flat position."""
    tmin = (b.t[s:e] // 60 + 1).astype(np.int64)                   # close time of each bar, in minutes
    L = r["equity_log"].copy()
    L[-1] = float(np.sum(r["daily"]))
    x = r["exposure"].astype(np.float64).copy()
    x[-1] = 0.0
    return tmin, x, L + log_offset


def exposure_points(tmin, x, fill_bars, start_bar, tol=0.01):
    """Kept points: every fill bar and the bar before it (exact), plus drift changes of tol or more."""
    keep = np.zeros(len(x), bool)
    keep[0] = keep[-1] = True
    fb = np.asarray(fill_bars, np.int64) - start_bar
    fb = fb[(fb >= 0) & (fb < len(x))]
    keep[fb] = True
    keep[np.maximum(fb - 1, 0)] = True
    last = x[0]
    d = np.abs(np.diff(x))
    cand = np.flatnonzero(d > 1e-9) + 1
    for i in cand:
        if abs(x[i] - last) >= tol or (x[i] == 0.0) != (last == 0.0):
            keep[i] = True
            keep[i - 1] = True                                     # the value just before the change (step look)
            last = x[i]
    idx = np.flatnonzero(keep)
    return tmin[idx], x[idx]


def equity_points(tmin, L, fill_bars, start_bar):
    hour_end = (tmin % 60) == 0                                     # closes at hh:59 -> close time hh+1:00
    keep = hour_end.copy()
    keep[0] = keep[-1] = True
    fb = np.asarray(fill_bars, np.int64) - start_bar
    fb = fb[(fb >= 0) & (fb < len(L))]
    keep[fb] = True
    keep[np.maximum(fb - 1, 0)] = True
    idx = np.flatnonzero(keep)
    return tmin[idx], L[idx]


def trade_cols(b, trades, period):
    if not trades:
        return np.zeros((0, 9))
    rows = [(b.t[tr["entry_bar"]] // 60, b.t[tr["exit_bar"]] // 60, tr["side"], tr["entry_px"], tr["exit_px"],
             tr["net_bps"], tr["gross_bps"], (b.t[tr["exit_bar"]] - b.t[tr["entry_bar"]]) // 60, period)
            for tr in trades]                                      # hold in wall-clock minutes (the seal has a gap day)
    return np.array(rows, np.float64)


def export_price(b, out):
    s = b.index_of(T0)
    t = b.t[s:]
    c = b.c[s:].astype(np.float32)
    m = (t - T0) // 60
    grid = np.full(int(m[-1]) + 1, np.nan, np.float32)
    grid[m] = c
    nh = len(grid) // 60 + (1 if len(grid) % 60 else 0)
    pad = np.full(nh * 60, np.nan, np.float32)
    pad[:len(grid)] = grid
    hours = pad.reshape(nh, 60)
    last = np.where(np.isnan(hours[:, -1]), np.nan, hours[:, -1])
    # hour close: last available minute of the hour
    for j in range(58, -1, -1):
        miss = np.isnan(last)
        if not miss.any():
            break
        last[miss] = hours[miss, j]
    last.astype("<f4").tofile(os.path.join(out, "price_hour.bin"))
    os.makedirs(os.path.join(out, "m"), exist_ok=True)
    months = []
    import datetime as dt
    d0 = dt.datetime(2021, 1, 1, tzinfo=dt.timezone.utc)
    y, mo = 2021, 1
    while True:
        a = dt.datetime(y, mo, 1, tzinfo=dt.timezone.utc)
        bnext = dt.datetime(y + (mo == 12), mo % 12 + 1, 1, tzinfo=dt.timezone.utc)
        i0 = int((a - d0).total_seconds() // 60)
        i1 = int((bnext - d0).total_seconds() // 60)
        if i0 >= len(grid):
            break
        seg = grid[i0:min(i1, len(grid))]
        name = f"{y:04d}-{mo:02d}"
        with open(os.path.join(out, "m", name + ".txt"), "w") as fh:     # base64 float32 (artifacts serve text)
            fh.write(base64.b64encode(seg.astype("<f4").tobytes()).decode())
        months.append({"name": name, "start_min": i0, "n": int(len(seg))})
        y, mo = (y + 1, 1) if mo == 12 else (y, mo + 1)
    return {"t0_min": int(T0 // 60), "n_hours": int(nh), "months": months,
            "price_min": float(np.nanmin(c)), "price_max": float(np.nanmax(c))}


def stats_block():
    gate = json.load(open(os.path.join(RES, "gate.json")))
    runs = json.load(open(os.path.join(RES, "dev_runs.json")))
    seal = json.load(open(os.path.join(RES, "seal.json")))
    out = {}
    for name, track in SHOW:
        dv = runs[name]["scenarios"]["C7B"]
        sv = seal["strategies"][name]["C7B"]
        g = gate["tracks"].get(str(track), {}).get("candidates", {}).get(name) if track else None
        verdict = None
        for k, v in seal.get("verdicts", {}).items():
            if k.endswith(":" + name):
                verdict = v["verdict"]
        out[name] = {"dev": {"ann_pct": dv["mean_daily_bps"] * 3.65, "sharpe": dv["sharpe"], "max_dd": dv["max_dd"],
                             "trades": (dv.get("trades") or {}).get("count", 0)},
                     "seal": {"ann_pct": sv["full"]["mean_daily_bps"] * 3.65, "sharpe": sv["full"]["sharpe"],
                              "total_log_pct": sv["full"]["total_log_bps"] / 100, "trades": sv["trades"]},
                     "gates": g["gates"] if g else None, "p_timing": g.get("p_maxT") if g else None,
                     "verdict": verdict}
    return out


def main(out):
    os.makedirs(os.path.join(out, "s"), exist_ok=True)
    # ---------------- dev
    st = prep.Store()
    S = rules.build_signals(st.b, st.sigma)
    names = [n for n, _ in SHOW]
    facs = run_dev.factories(st, S, names)
    s, e = st.idx(FO.DEV_OOS[0]), st.n
    dev_daily = dict(np.load(os.path.join(prep.WORK, "dev_daily.npz")))
    dev = {}
    for name in names:
        r = run(st.b, facs[name][0], s, e)
        assert np.array_equal(r["daily"], dev_daily[f"{name}|C7B"]), f"dev mismatch {name}"
        dev[name] = r
        print("dev ok", name, len(r["trades"]), flush=True)
    # ---------------- seal (frozen fold-S models, as run_seal.py built them)
    prereg = json.load(open(os.path.join(HERE, "prereg.json")))
    b = D.load(1, seal=True)
    X, cols, sig = FE.build(b)
    st2 = prep.Store(bars=b, X=X, names=cols, sigma=sig.astype(np.float32).astype(np.float64))
    s2, e2 = st2.idx(FO.SEAL_START), st2.idx(int(b.t[-1]) + 60)
    S2 = rules.build_signals(b, st2.sigma)
    f2 = {}
    for track in (1, 2):
        for k, f in rules.policies(S2, track).items():
            f2.setdefault(k, f)
        f2[f"T{track}-LIN"] = run_seal.ridge_policy(st2, track, s2, e2)
        f2[f"T{track}-NET"] = run_seal.net_policy(st2, track, prereg["net_selection_S"][str(track)], s2, e2)
    f2["T1-NT"] = run_seal.nt_policy(st2, prereg, s2, e2, False)
    seal_daily = dict(np.load(os.path.join(RES, "seal_daily.npz")))
    seal = {}
    for name in names:
        r = run(b, f2[name], s2, e2)
        assert np.array_equal(r["daily"], seal_daily[f"{name}|C7B"]), f"seal mismatch {name}"
        seal[name] = r
        print("seal ok", name, len(r["trades"]), flush=True)
    del X
    meta = {"price": export_price(b, out), "seal_start_min": int(FO.SEAL_START // 60),
            "dev_start_min": int(FO.DEV_OOS[0] // 60), "end_min": int(b.t[-1] // 60 + 1),
            "refits_min": [int(f[1] // 60) for f in FO.FOLDS[1:]], "stats": stats_block(),
            "strategies": [{"id": n, "track": tr} for n, tr in SHOW], "scenario": "C7-B"}
    for name in names:
        rd, rs = dev[name], seal[name]
        t1, x1, L1 = pieces(st.b, rd, s, e, 0.0)
        t2, x2, L2 = pieces(b, rs, s2, e2, float(np.sum(rd["daily"])))
        fb1 = [f[0] for f in rd["fills"]]
        fb2 = [f[0] for f in rs["fills"]]
        et1, ex1 = exposure_points(t1, x1, fb1, s)
        et2, ex2 = exposure_points(t2, x2, fb2, s2)
        qt1, qL1 = equity_points(t1, L1, fb1, s)
        qt2, qL2 = equity_points(t2, L2, fb2, s2)
        tr = np.vstack([trade_cols(st.b, rd["trades"], 0), trade_cols(b, rs["trades"], 1)])
        doc = {"id": name,
               "expo_t": b64(np.r_[et1, et2].astype("<i4")), "expo_x": b64(np.r_[ex1, ex2].astype("<f4")),
               "pnl_t": b64(np.r_[qt1, qt2].astype("<i4")), "pnl_L": b64(np.r_[qL1, qL2].astype("<f4")),
               "seal_break_expo": int(len(et1)), "seal_break_pnl": int(len(qt1)),
               "tr_tin": b64(tr[:, 0].astype("<i4")), "tr_tout": b64(tr[:, 1].astype("<i4")),
               "tr_side": b64(tr[:, 2].astype("<i1")), "tr_pin": b64(tr[:, 3].astype("<f4")),
               "tr_pout": b64(tr[:, 4].astype("<f4")), "tr_net": b64(tr[:, 5].astype("<f4")),
               "tr_gross": b64(tr[:, 6].astype("<f4")), "tr_hold": b64(tr[:, 7].astype("<i4")),
               "tr_period": b64(tr[:, 8].astype("<u1")), "n_trades": int(len(tr)),
               "n_trades_dev": int(len(rd["trades"]))}
        fn = name.replace("&", "and")
        json.dump(doc, open(os.path.join(out, "s", fn + ".json"), "w"))
        print("wrote", fn, len(et1) + len(et2), "expo pts", len(qt1) + len(qt2), "pnl pts", len(tr), "trades",
              flush=True)
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(prep.WORK, "viewer"))
