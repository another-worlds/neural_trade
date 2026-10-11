"""Step 6 report: paired per-slice differences (geometry arm minus plain arm, same seed, 24 slices x 1y) from newnet_results.jsonl.
usage: python newnet_compare.py"""
import json, os, sys
import numpy as np
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__))
R = {}
for l in open(f"{HERE}/newnet_results.jsonl", encoding="utf-8"):
    r = json.loads(l); R[(r["tag"], r["seed"])] = r
def ci(d):
    d = np.asarray(d, float); h = stats.t.ppf(.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)); return f"{d.mean():+.4f} [{d.mean() - h:+.4f},{d.mean() + h:+.4f}]"
out = []
KEYS = [("auc3", "mean3 AUC"), ("d_auc3", "d vs own regression"), ("ll_h1", "h1 log loss (lower better)"), ("vol_rho", "vol Spearman"), ("hon10_bps", "top-10% bps"), ("mon10_bps", "mag-filtered top-10% bps")]
for arch in ("linear", "patch", "linear3y", "patch3y"):
    seeds = sorted({s for (t, s) in R if t.startswith(f"geom2_{arch}_base") and isinstance(s, int)})
    pairs = []
    for s in seeds:
        b = next((R[(t, s)] for (t, ss) in R if ss == s and t.startswith(f"geom2_{arch}_base")), None)
        g = next((R[(t, s)] for (t, ss) in R if ss == s and t.startswith(f"geom2_{arch}_geom")), None)
        if b and g:
            pairs.append((s, b, g))
    for s, b, g in pairs:
        out.append(f"{arch} seed {s}: plain mean3 {b['summary']['auc3'][0]:.4f}, geometry {g['summary']['auc3'][0]:.4f}; params {b['params']['total']} vs {g['params']['total']}; sec {b['seconds']:.0f} vs {g['seconds']:.0f}")
        for k, nm in KEYS:
            d = np.array(g["per_slice"][k]) - np.array(b["per_slice"][k]); out.append(f"    {nm:28s} geometry - plain {ci(d)}")
    if len(pairs) > 1:
        out.append(f"{arch}: mean over {len(pairs)} seeds (per-slice differences averaged over seeds, then t over slices)")
        for k, nm in KEYS:
            d = np.mean([np.array(g["per_slice"][k]) - np.array(b["per_slice"][k]) for s, b, g in pairs], 0); out.append(f"    {nm:28s} geometry - plain {ci(d)}")
txt = "\n".join(out); print(txt); open(f"{HERE}/newnet_compare.txt", "w", encoding="utf-8").write(txt + "\n")
