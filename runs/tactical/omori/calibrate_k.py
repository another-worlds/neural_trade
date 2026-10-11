"""Fix k per horizon: the multiple of sigma_hat_HAR * sqrt(T) at which the target event (high/low reach from C_{i-1} within T bars)
has a 2.0% frequency on the training year (2018-01..2018-12, windows closed by 2019-01-01) of the FIRST scored month. This reads no
test, selection or look month and no skill number. Output: results/k.json."""
import os, sys, json, math
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import omorilib as O

if __name__ == "__main__":
    B = O.Bars(); out = {}
    m0 = pd.Timestamp("2019-01-01")
    for T in O.TS:
        i_m0 = int(np.searchsorted(B.t_ns, np.datetime64(m0, "ns").astype(np.int64)))
        i_lo = int(np.searchsorted(B.t_ns, np.datetime64(m0 - pd.Timedelta(days=365), "ns").astype(np.int64)))
        anch = np.arange(max(i_lo, O.WARM + T + 5), i_m0 - T - 2, O.STRIDE)
        coef = O.fit_har(B, anch, T); sig = O.sigma_hat(B, coef, anch, T)
        lo, hi = 0.5, 12.0
        for _ in range(40):
            k = 0.5 * (lo + hi); r = float(np.nanmean(O.target_event(B, anch, T, k, sig)))
            lo, hi = (k, hi) if r > 0.02 else (lo, k)
        k = round(0.5 * (lo + hi), 3); r = float(np.nanmean(O.target_event(B, anch, T, k, sig)))
        out[str(T)] = {"k": k, "rate": r, "n_anchors": int(len(anch))}; print(T, out[str(T)], flush=True)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "results", "k.json"), "w"), indent=1)
