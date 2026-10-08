"""Hypotheses 9 (direction over-confidence) and 2-3 (ensemble, thresholds) checked honestly on saved predictions:
each run's validation block is split in time - the first half FITS (calibration / thresholds), a 20-bar gap (no shared
targets), the second half TESTS. Nothing is chosen on the test half.
  H9: temperature / Platt scaling of P(up) (logit a*z + b, fitted by log loss on the fit half) -> log loss, Brier, AUC on
      the test half vs the raw head and vs a constant (the fit half's up-rate).
  H2-3: confidence thresholds (|score - 0.5| quantiles for coverage 20/10/5%) and the high-variance cut taken on the fit half,
      applied to the test half; hit rate and gross bps per trade vs a random null of the same size and long/short mix.
usage: python honest_split.py <spec> [h=1]"""
import glob, json, math, sys
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

spec = sys.argv[1]; H = int(sys.argv[2]) if len(sys.argv) > 2 else 1; GAP = 20
rng = np.random.default_rng(0)
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
sig = lambda z: 1 / (1 + np.exp(-z))
ll = lambda y, p: float(-np.mean(y * np.log(np.clip(p, 1e-7, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-7, 1))))
acc = {}
SL = {}
for rf in glob.glob(f"runs/tactical/screens/{spec}/results*.jsonl"):
    for l in open(rf, encoding="utf-8"):
        r = json.loads(l); SL[r["trial_key"]] = r["data_end"][:10]
cur = [None]
def put(k, v): acc.setdefault(k, {}).setdefault(cur[0], []).append(float(v))
for f in sorted(glob.glob(f"runs/tactical/screens/{spec}/preds/*.npz")):
    cur[0] = SL.get(f.replace("\\", "/").split("/")[-1][:-4], f)
    d = np.load(f); y, lc = d["y"], d["last_close"].reshape(-1); db = float(d["deadband_bps"]) / 1e4
    n = len(lc); a_end = n // 2; b0 = a_end + GAP
    P = np.stack([d[f"p_up_h{h}"] for h in range(3)], 1); V = d[f"var_h{H}"]
    ret = y[:, H] / lc; m = np.abs(ret) > db; lab = (ret > 0).astype(float)
    A = np.zeros(n, bool); A[:a_end] = True; B = np.zeros(n, bool); B[b0:] = True
    # ---- H9 calibration of the single head
    p = P[:, H]; z = logit(p)
    fa = A & m; fb = B & m
    res = minimize(lambda w: ll(lab[fa], sig(w[0] * z[fa] + w[1])), [1.0, 0.0], method="Nelder-Mead")
    a, b = res.x; pc = sig(a * z + b); base = lab[fa].mean()
    put("platt_a", a)
    for name, q in (("raw", p), ("calibrated", pc), ("constant", np.full(n, base))):
        put(f"ll_{name}", ll(lab[fb], q[fb])); put(f"brier_{name}", np.mean((q[fb] - lab[fb]) ** 2))
    put("auc_raw", roc_auc_score(lab[fb], p[fb])); put("auc_calibrated", roc_auc_score(lab[fb], pc[fb]))
    put("mean_abs_p_dev_raw", np.mean(np.abs(p[fb] - 0.5))); put("mean_abs_p_dev_cal", np.mean(np.abs(pc[fb] - 0.5)))
    # ---- H2-3 thresholds from the fit half
    m3 = P.mean(1); agree = (np.sign(P - 0.5) == np.sign(P[:, [0]] - 0.5)).all(1); vcut = np.quantile(V[A], 2 / 3)
    for sname, s, ok in (("single", P[:, H], np.ones(n, bool)), ("mean3", m3, np.ones(n, bool)), ("agree3", m3, agree),
                         ("agree3_hivar", m3, agree & (V > vcut))):
        conf = np.abs(s - 0.5)
        for cov in (0.2, 0.1, 0.05):
            thr = np.quantile(conf[A], 1 - cov)
            k = B & ok & (conf >= thr) & m
            if k.sum() < 15:
                continue
            side = np.sign(s[k] - 0.5); mv = side * ret[k] * 1e4
            null = [np.mean(rng.permutation(side) * ret[k] * 1e4 * rng.choice([-1, 1], len(side))) for _ in range(300)]
            put(f"{sname}_{int(cov*100)}_hit", np.mean(mv > 0)); put(f"{sname}_{int(cov*100)}_bps", np.mean(mv))
            put(f"{sname}_{int(cov*100)}_null95", np.quantile(null, 0.95)); put(f"{sname}_{int(cov*100)}_n", k.sum())
TQ = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45}
def summ(byslice):
    v = np.array([np.mean(x) for x in byslice.values()]); m = v.mean()  # unit of inference = slice (seeds averaged)
    se = v.std(ddof=1) / math.sqrt(len(v)) if len(v) > 1 else float("nan"); t = TQ.get(len(v), 2.0)
    return round(float(m), 4), [round(float(m - t * se), 4), round(float(m + t * se), 4)], len(v)
out = {k: summ(v) for k, v in acc.items()}
json.dump(out, open(f"runs/tactical/probe/honest_split_{spec}_h{H}.json", "w"), indent=1)
for k in sorted(out):
    print(f"{k:28s} {out[k][0]:+.4f}  CI {out[k][1]}  slices {out[k][2]}")
