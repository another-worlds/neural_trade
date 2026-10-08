"""Accuracy vs the confidence threshold (owner, 2026-10-08) and head combinations as a simple strategy, on saved
validation predictions (no retraining). Signals (all use only the model's own outputs):
  single_h      P(up) of horizon h
  mean3         mean P(up) of the 3 horizons
  agree3        mean3, but only bars where all 3 horizons point the same way
  agree3_hivar  agree3 and the predicted variance of horizon h in the run's top third (a big move expected)
For each signal and coverage (the share of bars kept, by |score - 0.5|): hit rate, mean gross move per trade in bps
(sign x realised return at horizon h, costs 0 per D-044), and a random null: the same number of bars and the same
long/short mix with random signs (500 draws) -> its 95th percentile of the mean move. CIs: per-run values, t over runs.
Bars overlap (the target spans h bars), so the effective sample is about n / horizon bars (D-012)."""
import glob, json, math, sys
import numpy as np

spec = sys.argv[1] if len(sys.argv) > 1 else "cand_c2_base"
H = int(sys.argv[2]) if len(sys.argv) > 2 else 1
files = sorted(glob.glob(f"runs/tactical/screens/{spec}/preds/*.npz"))
COV = [1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02]
rng = np.random.default_rng(0)
rows = {}
for f in files:
    d = np.load(f); y, lc = d["y"], d["last_close"].reshape(-1); db = float(d["deadband_bps"]) / 1e4
    P = np.stack([d[f"p_up_h{h}"] for h in range(3)], 1); V = d[f"var_h{H}"]
    ret = y[:, H] / lc
    m3 = P.mean(1); agree = (np.sign(P - 0.5) == np.sign(P[:, [0]] - 0.5)).all(1); hiv = V > np.quantile(V, 2 / 3)
    sigs = {"single_h": (P[:, H], np.ones(len(ret), bool)), "mean3": (m3, np.ones(len(ret), bool)),
            "agree3": (m3, agree), "agree3_hivar": (m3, agree & hiv)}
    for name, (s, ok) in sigs.items():
        conf = np.abs(s - 0.5)
        for cov in COV:
            thr = np.quantile(conf, 1 - cov)  # coverage is of ALL bars, the filter then applies on top
            k = ok & (conf >= thr) & (np.abs(ret) > db)
            if k.sum() < 30:
                continue
            side = np.sign(s[k] - 0.5); mv = side * ret[k] * 1e4
            hit = float(np.mean(mv > 0)); mean_bps = float(np.mean(mv))
            null = [np.mean(rng.permutation(side) * rng.choice([-1, 1], len(side)) * ret[k] * 1e4) for _ in range(200)]
            r = rows.setdefault((name, cov), {"hit": [], "bps": [], "null95": [], "n": []})
            r["hit"].append(hit); r["bps"].append(mean_bps); r["null95"].append(float(np.quantile(null, 0.95))); r["n"].append(int(k.sum()))
def ci(v):
    if len(v) < 2: return (float("nan"),) * 2
    m, se = float(np.mean(v)), float(np.std(v, ddof=1) / math.sqrt(len(v))); t = 2.23 if len(v) == 11 else 2.0
    return (m - t * se, m + t * se)
out = []
for (name, cov), r in sorted(rows.items(), key=lambda kv: (kv[0][0], -kv[0][1])):
    lo, hi = ci(r["hit"]); blo, bhi = ci(r["bps"])
    out.append({"signal": name, "coverage": cov, "runs": len(r["hit"]), "bars_per_run": int(np.mean(r["n"])),
                "n_eff_per_run": int(np.mean(r["n"]) / [10, 15, 20][H]), "hit": round(float(np.mean(r["hit"])), 4),
                "hit_ci": [round(lo, 4), round(hi, 4)], "gross_bps": round(float(np.mean(r["bps"])), 2),
                "gross_bps_ci": [round(blo, 2), round(bhi, 2)], "null95_bps": round(float(np.mean(r["null95"])), 2)})
for o in out:
    print(f"{o['signal']:13s} cov {o['coverage']:>4} | hit {o['hit']:.3f} [{o['hit_ci'][0]:.3f},{o['hit_ci'][1]:.3f}] | "
          f"{o['gross_bps']:+.2f} bps [{o['gross_bps_ci'][0]:+.2f},{o['gross_bps_ci'][1]:+.2f}] null95 {o['null95_bps']:+.2f} | bars {o['bars_per_run']} n_eff {o['n_eff_per_run']}")
json.dump(out, open(f"runs/tactical/probe/threshold_curves_{spec}_h{H}.json", "w"), indent=1)
