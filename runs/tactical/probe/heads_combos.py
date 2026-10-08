"""Head combinations on saved validation predictions (owner, 2026-10-08: the model is an implicit ensemble; we never
tested combining the heads). CPU only, no retraining. For each run and horizon:
  - direction AUC overall, and inside terciles of the predicted variance (does a "big move expected" make the call better?)
  - AUC of combined scores: mean P(up) of the 3 horizons; P(up) where the 3 horizons agree; P(up) x price sign agreement
  - hit rate on the most confident 10% and 20% of bars, overall and in the high-variance tercile
Averages over runs; per-run values in the json."""
import glob, json, sys
import numpy as np
from sklearn.metrics import roc_auc_score

spec = sys.argv[1] if len(sys.argv) > 1 else "cand_c2_base"
files = sorted(glob.glob(f"runs/tactical/screens/{spec}/preds/*.npz"))
res = {}
def put(k, v):
    if v is not None and np.isfinite(v):
        res.setdefault(k, []).append(float(v))
def auc(lab, s):
    return roc_auc_score(lab, s) if len(np.unique(lab)) == 2 and len(lab) > 20 else None
for f in files:
    d = np.load(f); y, lc = d["y"], d["last_close"].reshape(-1); db = float(d["deadband_bps"]) / 1e4
    P = np.stack([d[f"p_up_h{h}"] for h in range(3)], 1); V = np.stack([d[f"var_h{h}"] for h in range(3)], 1)
    D = np.stack([d[f"delta_h{h}"] for h in range(3)], 1)
    for h in range(3):
        ret = y[:, h] / lc; m = np.abs(ret) > db; lab = (ret > 0).astype(int)
        p, v = P[:, h], V[:, h]
        put(f"h{h}_auc_dir", auc(lab[m], p[m]))
        q = np.quantile(v, [1 / 3, 2 / 3])
        for name, sel in (("lowvar", v <= q[0]), ("midvar", (v > q[0]) & (v <= q[1])), ("highvar", v > q[1])):
            s = m & sel; put(f"h{h}_auc_dir_{name}", auc(lab[s], p[s]))
            put(f"h{h}_absmove_bps_{name}", 1e4 * np.mean(np.abs(ret[sel])))
        put(f"h{h}_auc_mean3", auc(lab[m], P[m].mean(1)))
        agree = (np.sign(P - 0.5) == np.sign(P[:, [h]] - 0.5)).all(1)
        put(f"h{h}_share_agree3", agree.mean())
        put(f"h{h}_auc_dir_when_agree3", auc(lab[m & agree], p[m & agree]))
        same = np.sign(p - 0.5) == np.sign(D[:, h])
        put(f"h{h}_auc_dir_when_price_agrees", auc(lab[m & same], p[m & same]))
        conf = np.abs(p - 0.5)
        for top in (0.1, 0.2):
            t = conf >= np.quantile(conf, 1 - top); s = m & t
            put(f"h{h}_hit_top{int(top*100)}", np.mean((p[s] > 0.5) == (lab[s] == 1)))
            s2 = s & (v > q[1]); put(f"h{h}_hit_top{int(top*100)}_highvar", np.mean((p[s2] > 0.5) == (lab[s2] == 1)) if s2.sum() > 20 else None)
out = {k: round(float(np.mean(v)), 4) for k, v in sorted(res.items())}
out["runs"] = len(files)
print(json.dumps(out, indent=1))
json.dump({"mean": out, "per_run": res}, open(f"runs/tactical/probe/heads_combos_{spec}.json", "w"), indent=1)
