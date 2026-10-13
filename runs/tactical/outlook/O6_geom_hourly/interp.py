"""O6 interpretation: the top G60 coefficients at H = 240 of the frozen L2 fit (standardised ridge weights) and, for the top 3 features, 6 real
hourly windows (3 highest and 3 lowest feature value, anchors at least 3 days apart) from the cut file (all before 2025-07-20 23:59)."""
import json, os, sys
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import o6_lib as P  # noqa: E402

art = json.load(open(f"{HERE}/frozen_B.json")); m = P.model_from_json(art["states"]["240"]["m"]["L2"])
off = P.N_XB + P.N_G1G15; w60 = m["w"][off:]; names = P.G60_NAMES; assert len(w60) == len(names) == 164
order = np.argsort(-np.abs(w60)); top = [{"feature": names[i], "col": int(i), "coef_standardised": float(w60[i])} for i in order[:15]]
z = np.load(os.path.join(P.SCRATCH, "frame.npz")); ta = z["ta"]; anch = z["anch"]; Gm = z["G"]
s = P.T.load_bars(P.T.TRAIN_CSV, P.LOAD_FROM, P.T.CUTOFF); A = s.values.astype(np.float64); idx = s.index
out = {"note": "G60 coefficients of the frozen L2 ridge at H=240 (standardised features, fit on Binance 2025-10-05..2026-09-30); windows from the cut file",
       "frozen_sha256": art["sha256"], "top_g60_coefficients": top, "windows": {}}
for t in top[:3]:
    col = P.N_G1G15 + t["col"]; v = Gm[:, col].astype(np.float64); ws = []
    for label, o in (("highest", np.argsort(-v)), ("lowest", np.argsort(v))):
        chosen = []
        for i in o:
            if all(abs(int(anch[i]) - int(anch[j])) >= 3 * 1440 for j in chosen):
                chosen.append(int(i))
            if len(chosen) == 3:
                break
        for i in chosen:
            a = int(anch[i]); bars = P.aggm(A, [a], P.HOUR)[0]
            ws.append({"which": label, "feature_value": float(v[i]), "anchor_time": str(idx[a]), "last_bar_end": str(idx[a - 1]),
                       "hourly_ohlcv": np.round(bars, 4).tolist()})
    out["windows"][t["feature"]] = ws
assert all(pd.Timestamp(w["last_bar_end"]) <= P.T.CUTOFF for ws in out["windows"].values() for w in ws)
json.dump(out, open(f"{HERE}/interp.json", "w"), indent=1)
print(json.dumps(top, indent=1))
