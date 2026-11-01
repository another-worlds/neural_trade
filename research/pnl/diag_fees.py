"""Descriptive fee-tier sensitivity on dev OOS (never a gate or a selection): Track 1 strategies at maker fees of
2 (standard), 1 and 0 bps, with taker 5 bps, under C4-B (maker both ways) and C7-B (maker in, taker out).
Maker fills at bar resolution assume a 1-tick trade-through and no queue position; a pass here would need tick or
L2 validation before it means anything (SPEC 'track1').

  python diag_fees.py  -> results/diag_fees.json
"""
import json
import os

import numpy as np

import folds as FO
import prep
import rules
import sim
from policy import BandPolicy, HysteresisPolicy

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def main():
    st = prep.Store()
    S = rules.build_signals(st.b, st.sigma)
    s, e = st.idx(FO.DEV_OOS[0]), st.n
    import run_dev
    tau, band, dec, K, _ = run_dev.select_net(st, 1)
    z = np.load(os.path.join(prep.WORK, "ridge_T1_dev.npz"))
    facs = {"T1-FADEV": rules.policies(S, 1)["T1-FADEV"], "plain_fade": rules.policies(S, 1)["plain_fade"],
            "T1-NET": lambda: BandPolicy(tau, band, dec, K=K, chase=False),
            "T1-LIN": lambda: HysteresisPolicy(z["mu"], z["dec"], z["H"], z["thr"], K=1, chase=False)}
    out = {}
    m0 = sim.MAKER
    try:
        for maker in (2e-4, 1e-4, 0.0):
            sim.MAKER = maker
            for fee in ("C4", "C7"):
                for name, fac in facs.items():
                    r = sim.simulate(st.b, fac(), s, e, fee=fee, fill="B")
                    out[f"{name}|{fee}|maker{maker * 1e4:g}"] = {"ann_pct": float(r["daily"].mean() * 365 * 100),
                                                                "sharpe": float(r["daily"].mean() / r["daily"].std()
                                                                                * np.sqrt(365)),
                                                                "trades": len(r["trades"])}
                    print(name, fee, maker * 1e4, out[f"{name}|{fee}|maker{maker * 1e4:g}"], flush=True)
    finally:
        sim.MAKER = m0
    json.dump(out, open(os.path.join(RESULTS, "diag_fees.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
