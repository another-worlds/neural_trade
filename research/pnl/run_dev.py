"""Dev-OOS scoring (SPEC 'walk_forward' DEV OOS, 'baselines'): one contiguous engine-E run per strategy over
2021-01-01..2025-10-31 in all six scenarios (C7/C4/C10 x B/S), plus nominal C10-N for reconciliation.

Candidates: T1-FADEV, T1-LIN, T1-NET, T1-NT (Track 1); T2-P, T2-LIN, T2-NET (Track 2).
Baselines: cash, B&H perp, plain fade, intraday trend (T1); TSMOM unscaled, vol-target long, 60-min fade (T2);
static-exposure-matched per candidate; carry hurdle (reference). The random-timing null and p_sel are in gate.py.

Writes results/dev_runs.json (metrics per strategy and scenario), results/net_selection.json, $PNL_WORK/dev_daily.npz
(daily net log returns per strategy|scenario) and $PNL_WORK/dev_expo.npz (C7-B per-bar exposure), and appends one
line per run to results/runs.jsonl.

  python run_dev.py [names...]     (default: everything)
"""
import hashlib
import json
import os
import sys
import time

import numpy as np

import folds as FO
import metrics as M
import prep
import rules
import sim
from policy import BandPolicy, HysteresisPolicy

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
SCEN = [("C7", "B"), ("C7", "S"), ("C4", "B"), ("C4", "S"), ("C10", "B"), ("C10", "S")]
TRACK = {"T1-FADEV": 1, "T1-LIN": 1, "T1-NET": 1, "T1-NT": 1, "plain_fade": 1, "intraday_trend": 1,
         "T2-P": 2, "T2-LIN": 2, "T2-NET": 2, "tsmom_unscaled": 2, "voltarget_long": 2, "fade60": 2,
         "T1-NET-W": 1, "T2-NET-W": 2,
         "B&H": 0, "cash": 0}
CANDIDATES = {1: ("T1-FADEV", "T1-LIN", "T1-NET", "T1-NT"), 2: ("T2-P", "T2-LIN", "T2-NET")}
NET_HS = {1: (1, 5, 15), 2: (60, 240, 1440, 1, 5, 15)}
DEV_FOLDS = [f[0] for f in FO.FOLDS]


def k_track1(h):
    return 1 if h <= 5 else 3


# ------------------------------------------------------------------------------------------- net selection

def select_net(st, track, variant=None):
    """In-fold cell choice per fold from inner-validation ensemble U (SPEC 'IN-FOLD CELL CHOICE'). variant='mlpw':
    MLP cells come from the extended-warm-up retraining (descriptive T*-NET-W, CHANGES.log)."""
    nets = os.path.join(prep.WORK, "nets")
    nets_v = os.path.join(prep.WORK, "nets_" + variant) if variant else None

    def src(cell):
        return nets_v if (variant and cell.startswith("MLP")) else nets
    n = st.n
    tau = np.zeros(n, np.float32)
    band = np.zeros(n, np.float32)
    dec = np.zeros(n, bool)
    K = np.full(n, 15 if track == 2 else 1, np.int64)
    table = {}
    for fold, ts0, ts1 in FO.FOLDS + [FO.FOLD_S]:
        cells = {}
        for h in NET_HS[track]:
            tr = 1 if h <= 15 else 2
            path = os.path.join(nets, f"T{tr}_{fold}_h{h}.json")
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            js = json.load(open(path))
            for cell, v in js["cells"].items():
                cells[(h, cell)] = v["U_ens"]
            if variant:
                jv = json.load(open(os.path.join(nets_v, f"T{tr}_{fold}_h{h}.json")))
                for cell, v in jv["cells"].items():
                    cells[(h, cell)] = v["U_ens"]
        best = max(cells, key=lambda c: cells[c])
        h, cell = best
        if cell.startswith("MLP"):
            lin = (h, "LIN" + cell[3:])
            if cells[lin] >= 0.9 * cells[best]:
                best = lin
        cash = cells[best] <= 0.0
        table[fold] = {"h": best[0], "cell": best[1], "U_best_bps": cells[best] * 1e4, "cash": cash,
                       "cells_bps": {f"h{k[0]}_{k[1]}": u * 1e4 for k, u in cells.items()}}
        if fold == "S":
            continue
        s, e = st.idx(ts0), st.idx(ts1)
        if cash:
            m = np.arange(s, e)[prep.decision_mask(st.b.t[s:e], 60)]
            dec[m] = True                                 # tau = b = 0: go flat at the first decision, stay flat
            continue
        h, cell = best
        tr = 1 if h <= 15 else 2
        z = np.load(os.path.join(src(cell), f"T{tr}_{fold}_h{h}_test.npz"))
        rows = z["test_rows"]
        tau[rows] = z[f"{cell}_tau"]
        band[rows] = z[f"{cell}_b"]
        dec[rows] = True
        if track == 1:
            K[s:e] = k_track1(h)
    return tau, band, dec, K, table


# ------------------------------------------------------------------------------------------- strategies

def factories(st, S, which):
    """name -> (zero-arg policy factory, config dict)."""
    out = {}
    r1, r2 = rules.policies(S, 1), rules.policies(S, 2)
    for k in ("T1-FADEV", "plain_fade", "intraday_trend", "B&H", "cash"):
        out[k] = (r1[k], {"rule": k})
    for k in ("T2-P", "tsmom_unscaled", "voltarget_long", "fade60"):
        out[k] = (r2[k], {"rule": k})
    for track, name in ((1, "T1-LIN"), (2, "T2-LIN")):
        if name in which:
            z = np.load(os.path.join(prep.WORK, f"ridge_T{track}_dev.npz"))
            mu, H, thr, dec = z["mu"], z["H"], z["thr"], z["dec"]
            Kv, ch = (1, False) if track == 1 else (15, True)
            out[name] = ((lambda mu=mu, dec=dec, H=H, thr=thr, Kv=Kv, ch=ch:
                          HysteresisPolicy(mu, dec, H, thr, K=Kv, chase=ch)),
                         {"ridge": f"ridge_T{track}_dev.npz", "sha": hashlib.sha256(mu.tobytes()).hexdigest()})
    for track, name, var in ((1, "T1-NET", None), (2, "T2-NET", None), (1, "T1-NET-W", "mlpw"),
                             (2, "T2-NET-W", "mlpw")):
        if name in which:
            tau, band, dec, K, table = select_net(st, track, var)
            suffix = f"_{var}" if var else ""
            json.dump(table, open(os.path.join(RESULTS, f"net_selection_T{track}{suffix}.json"), "w"), indent=1)
            ch = track == 2
            out[name] = ((lambda tau=tau, band=band, dec=dec, K=K, ch=ch: BandPolicy(tau, band, dec, K=K, chase=ch)),
                         {"net_selection": table, "sha": hashlib.sha256(tau.tobytes() + band.tobytes()).hexdigest()})
    if "T1-NT" in which:
        path = os.path.join(prep.WORK, "nt", "T1-NT_dev.npz")
        if os.path.exists(path):
            z = np.load(path)
            out["T1-NT"] = ((lambda z=z: HysteresisPolicy(z["mu"], z["dec"], z["H"], z["thr"], K=z["K"],
                                                          chase=False)),
                            {"nt": path, "sha": hashlib.sha256(z["mu"].tobytes()).hexdigest()})
    return {k: v for k, v in out.items() if k in which}


def run_one(st, name, fac, fee, fill, s, e, record):
    t0 = time.time()
    r = sim.simulate(st.b, fac(), s, e, fee=fee, fill=fill, record_exposure=record)
    years = len(r["daily"]) / 365.0
    tr = r["trades"]
    summ = M.summary(r["daily"], r["days"], trades=tr if tr else None)
    attempts = r["n_maker"] + r["n_cancel"] + r["n_chase"]
    summ.update({"fees_frac": r["fees"], "funding_frac": r["funding"], "turnover_per_year": r["turnover"] / years,
                 "n_taker": r["n_taker"], "n_maker": r["n_maker"], "n_cancel": r["n_cancel"], "n_chase": r["n_chase"],
                 "maker_fill_rate": r["n_maker"] / attempts if attempts else None, "trims": r["trims"],
                 "mean_daily_bps": float(r["daily"].mean() * 1e4), "seconds": time.time() - t0})
    if record:
        x = r["exposure"]
        summ.update(M.exposure_stats(x))
    return r, summ


def main(which=None):
    st = prep.Store()
    S = rules.build_signals(st.b, st.sigma)
    s, e = st.idx(FO.DEV_OOS[0]), st.n
    which = which or list(TRACK)
    facs = factories(st, S, which)
    os.makedirs(RESULTS, exist_ok=True)
    out_path = os.path.join(RESULTS, "dev_runs.json")
    res = json.load(open(out_path)) if os.path.exists(out_path) else {}
    daily_p = os.path.join(prep.WORK, "dev_daily.npz")
    expo_p = os.path.join(prep.WORK, "dev_expo.npz")
    daily = dict(np.load(daily_p)) if os.path.exists(daily_p) else {}
    expo = dict(np.load(expo_p)) if os.path.exists(expo_p) else {}
    runs = open(os.path.join(RESULTS, "runs.jsonl"), "a")
    for name, (fac, cfg) in facs.items():
        res[name] = {"track": TRACK.get(name, 0), "config": {k: v for k, v in cfg.items() if k != "net_selection"},
                     "scenarios": {}}
        for fee, fill in SCEN + [("C10", "N")]:
            rec = (fee, fill) == ("C7", "B")
            r, summ = run_one(st, name, fac, fee, fill, s, e, rec)
            res[name]["scenarios"][fee + fill] = summ
            daily[f"{name}|{fee}{fill}"] = r["daily"]
            if rec:
                expo[name] = r["exposure"].astype(np.float32)
                daily["days"] = r["days"]
                daily[f"fees|{name}"] = r["daily_fees"]
                daily[f"trades|{name}"] = np.array([t["net_bps"] for t in r["trades"]], np.float64)
                daily[f"trades_gross|{name}"] = np.array([t["gross_bps"] for t in r["trades"]], np.float64)
            runs.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "name": name,
                                   "scenario": fee + fill, "config_sha": cfg.get("sha", cfg.get("rule")),
                                   "n_days": len(r["daily"]), "mean_daily_bps": summ["mean_daily_bps"],
                                   "sharpe": summ["sharpe"]}) + "\n")
            runs.flush()
            print(f"{name:16s} {fee}{fill}: {summ['mean_daily_bps']:8.2f} bps/d  Sharpe {summ['sharpe']:6.2f}  "
                  f"({summ['seconds']:.1f}s)", flush=True)
        # static-exposure-matched baseline for candidates (B5): the candidate's mean C7-B exposure, held constant
        if name in CANDIDATES[1] + CANDIDATES[2] + ("T1-NET-W", "T2-NET-W"):
            mx = float(np.mean(expo[name]))
            sname = f"static[{name}]"
            fac_s = (lambda mx=mx: sim.ConstPolicy(mx, K=15, chase=True))
            res[sname] = {"track": TRACK[name], "config": {"mean_x": mx}, "scenarios": {}}
            for fee, fill in SCEN:
                r, summ = run_one(st, sname, fac_s, fee, fill, s, e, False)
                res[sname]["scenarios"][fee + fill] = summ
                daily[f"{sname}|{fee}{fill}"] = r["daily"]
    res["carry_hurdle"] = {"track": 2, "log_total": rules.carry_hurdle(st.b, s, e),
                           "by_year": {y: rules.carry_hurdle(st.b, st.idx(a), st.idx(b))
                                       for y, a, b in FO.FOLDS}}
    json.dump(res, open(out_path, "w"), indent=1, default=float)
    np.savez(daily_p, **daily)
    np.savez(expo_p, **expo)
    return res


if __name__ == "__main__":
    main(sys.argv[1:] or None)
