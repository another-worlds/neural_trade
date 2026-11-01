"""Descriptive diagnostics of T1-LIN on dev OOS (not a gate, not a selection): what it trades, how concentrated its
PnL is, and which part of the stress scenario (latency, maker trade-through, taker slippage) removes its edge.

  python diag_t1lin.py  -> results/diag_t1lin.json
"""
import json
import os

import numpy as np

import folds as FO
import prep
import sim
from policy import HysteresisPolicy

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def main():
    st = prep.Store()
    z = np.load(os.path.join(prep.WORK, "ridge_T1_dev.npz"))
    s, e = st.idx(FO.DEV_OOS[0]), st.n

    def pol():
        return HysteresisPolicy(z["mu"], z["dec"], z["H"], z["thr"], K=1, chase=False)

    out = {"stress_decomposition_mean_daily_bps": {}}
    variants = {"C7-B (base)": ("C7", "B", None), "C7-S (stress)": ("C7", "S", None),
                "C7-B + 1 bar latency": ("C7", "B", {"lat": 1}),
                "C7-B + 1 bps maker trade-through": ("C7", "B", {"tt": 1e-4}),
                "C7-B + 1 bps taker slippage": ("C7", "B", {"slip": 1e-4}),
                "C10-B (all taker)": ("C10", "B", None), "C10-B + 1 bar latency": ("C10", "B", {"lat": 1}),
                "C10-B + 2 bar latency": ("C10", "B", {"lat": 2})}
    base = None
    for name, (fee, fill, dg) in variants.items():
        r = sim.simulate(st.b, pol(), s, e, fee=fee, fill=fill, diag=dg)
        g = np.array([t["gross_bps"] for t in r["trades"]])
        out["stress_decomposition_mean_daily_bps"][name] = {
            "mean_daily_bps": float(r["daily"].mean() * 1e4), "trades": len(r["trades"]),
            "mean_gross_bps_per_trade": float(g.mean()) if len(g) else None, "n_cancel": r["n_cancel"]}
        if name == "C7-B (base)":
            base = r
    tr = base["trades"]
    net = np.array([t["net_bps"] for t in tr])
    side = np.array([t["side"] for t in tr])
    eb = np.array([t["entry_bar"] for t in tr])
    hold = np.array([t["hold_minutes"] for t in tr])
    # direction relative to the last 1, 5 and 15 minutes before the decision (decision = entry bar - 1 for maker)
    dec = eb - 1
    rel = {}
    for L in (1, 5, 15):
        past = np.sign(st.lc[dec] - st.lc[dec - L])
        rel[f"follows_last_{L}m_share"] = float(np.mean(side == past))
    order = np.argsort(-np.abs(net))
    tot = net.sum()
    years = (st.b.t[eb] // 86400).astype("datetime64[D]").astype("datetime64[Y]").astype(int) + 1970
    sig15 = st.sigma60[dec] * 1e4
    out.update({
        "trades": len(tr), "mean_net_bps": float(net.mean()), "median_net_bps": float(np.median(net)),
        "hit_rate": float(np.mean(net > 0)), "median_hold_min": float(np.median(hold)),
        "long_share": float(np.mean(side > 0)), **rel,
        "top10_share_of_total_net": float(net[order[:10]].sum() / tot) if tot else None,
        "net_without_top10_bps": float(net[order[10:]].sum()),
        "trades_per_year": {int(y): int(np.sum(years == y)) for y in np.unique(years)},
        "net_per_year_bps": {int(y): float(net[years == y].sum()) for y in np.unique(years)},
        "sigma60_at_entry_bps_per_min_median": float(np.median(sig15)),
        "sigma60_dev_median_bps_per_min": float(np.median(st.sigma60[s:e]) * 1e4),
        "largest_trades": [{"time": str(np.datetime64(int(st.b.t[eb[k]]), "s")), "side": int(side[k]),
                            "net_bps": float(net[k]), "hold_min": int(hold[k])} for k in order[:10]],
    })
    json.dump(out, open(os.path.join(RESULTS, "diag_t1lin.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
