"""T1-NT (SPEC 'track1' 4): fixed neural_trade forecasts (nt_adapter.py outputs in $PNL_WORK/nt) traded through the
common hysteresis layer with theta = 1 (|mu| > 8 bps), minimum hold h, Track 1 execution (maker K = 1 or 3, cancel).

Per fold, h in {1, 5, 15} is chosen on neural_trade's own validation block under engine F (c_in 3, c_out 5 bps,
per minute); CASH for the year if every h has mean daily net <= 0. Writes $PNL_WORK/nt/T1-NT_dev.npz (per-bar mu,
dec, H, thr, K for run_dev.py), $PNL_WORK/nt/T1-NT_S.json (frozen fold-S bundle and h for the seal) and
results/nt_selection.json.

  python nt_select.py
"""
import json
import os

import numpy as np

import fast
import folds as FO
import prep
from policy import C_RT

HS = (1, 5, 15)
NT = os.path.join(prep.WORK, "nt")
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def engine_f_score(st, t, mu_h, h):
    idx = np.searchsorted(st.b.t, t)
    ok = (idx < st.n) & (st.b.t[np.minimum(idx, st.n - 1)] == t)
    idx, m = idx[ok], mu_h[ok]
    s, e = int(idx.min()), int(idx.max()) + 1
    mu = np.full(e - s, np.nan)
    mu[idx - s] = m
    x = fast.hysteresis_positions(mu, ~np.isnan(mu), int(h), float(C_RT))
    net = fast.engine_f_minutes(x, st.R1[s:e], st.F1[s:e], 3e-4, 5e-4, 0.0)
    daily = np.bincount(st.day[s:e] - st.day[s], weights=net)
    return float(daily.mean()), float(np.abs(np.diff(np.r_[0.0, x])).sum())


def main():
    st = prep.Store()
    n = st.n
    mu = np.full(n, np.nan, np.float32)
    H = np.ones(n, np.int64)
    thr = np.full(n, C_RT)
    K = np.ones(n, np.int64)
    table = {}
    for name, ts0, ts1 in FO.FOLDS + [FO.FOLD_S]:
        vpath = os.path.join(NT, f"{name}_val.npz")
        if not os.path.exists(vpath):
            raise FileNotFoundError(vpath)
        v = np.load(vpath)
        scores = {h: engine_f_score(st, v["t"], v["mu"][:, j], h) for j, h in enumerate(HS)}
        best = max(HS, key=lambda h: scores[h][0])
        cash = scores[best][0] <= 0.0
        meta = json.load(open(os.path.join(NT, f"{name}.json")))
        table[name] = {"h": best, "cash": cash, "val_mean_daily_bps": {h: s[0] * 1e4 for h, s in scores.items()},
                       "val_turnover": {h: s[1] for h, s in scores.items()}, "bundle": meta.get("bundle")}
        if name == "S":
            json.dump({"bundle": meta.get("bundle"), "h": best, "cash": cash}, open(os.path.join(NT, "T1-NT_S.json"), "w"))
            continue
        z = np.load(os.path.join(NT, f"{name}.npz"))
        idx = np.searchsorted(st.b.t, z["t"])
        ok = (idx < n) & (st.b.t[np.minimum(idx, n - 1)] == z["t"])
        j = HS.index(best)
        mu[idx[ok]] = 0.0 if cash else z["mu"][ok, j]
        s, e = st.idx(ts0), st.idx(ts1)
        H[s:e], K[s:e] = best, (1 if best <= 5 else 3)
        table[name]["test_share_above_thr"] = float(np.mean(np.abs(z["mu"][ok, j]) > C_RT))
    np.savez(os.path.join(NT, "T1-NT_dev.npz"), mu=mu, dec=~np.isnan(mu), H=H, thr=thr, K=K)
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(table, open(os.path.join(RESULTS, "nt_selection.json"), "w"), indent=1, default=float)
    print(json.dumps(table, indent=1, default=float))


if __name__ == "__main__":
    main()
