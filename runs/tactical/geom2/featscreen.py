"""Experiment 2b: single-feature screen. For every feature of tb7, rich and the 22-channel geometry gR, on every lab slice's val block:
signed AUC - 0.5 against the sign of the return at each horizon (deadband-masked), averaged over the 3 horizons. Across the 24 slices:
mean, t-statistic (mean / se), and the number of slices with the same sign. Bonferroni threshold: |t| above the two-sided t(23)
quantile at 0.05 / n_features. Also the same on the TRAIN block (it should agree if a feature carries a stable signal).
usage: python featscreen.py   (writes featscreen_report.txt)"""
import glob, os, sys
import numpy as np
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "lab"))
import lab, geomlib as G  # noqa: E402

names = {"tb7": [f"tb7_{i}" for i in range(13)], "rich": [f"rich_{i}" for i in range(43)], "gR": G.geom_names(None, (3, 5, 10, 20))}
fn = {"tb7": lab.fs_tb7, "rich": lab.fs_rich, "gR": G.geom_rich}
def auc_signed(F, r, db):
    out = np.zeros((F.shape[1], 3))
    for h in range(3):
        m = np.abs(r[:, h]) > db; y = r[m, h] > 0; n1 = y.sum(); n0 = len(y) - n1
        rk = stats.rankdata(F[m], axis=0); out[:, h] = (rk[y].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0) - 0.5
    return out.mean(1)
res = {k: {"va": [], "tr": []} for k in fn}
for p in sorted(glob.glob(lab.CACHE)):
    D = np.load(p); db = float(D["deadband"])
    for k, f in fn.items():
        res[k]["va"].append(auc_signed(np.nan_to_num(f(D["Wva"])), D["rva"], db)); res[k]["tr"].append(auc_signed(np.nan_to_num(f(D["Wtr"])), D["rtr"], db))
lines = []; tcrit_b = lambda m: stats.t.ppf(1 - 0.05 / (2 * m), 23)
allrows = []
for k in fn:
    va, tr = np.array(res[k]["va"]), np.array(res[k]["tr"])
    for j, nm in enumerate(names[k]):
        mv = va[:, j].mean(); t = mv / (va[:, j].std(ddof=1) / np.sqrt(len(va))); same = int((np.sign(va[:, j]) == np.sign(mv)).sum())
        allrows.append((k, nm, mv, t, same, tr[:, j].mean()))
nf = len(allrows); tc = tcrit_b(nf)
lines.append(f"{nf} features, {len(res['tb7']['va'])} slices; Bonferroni |t| threshold {tc:.2f}; AUC-0.5 on the val blocks (mean of 3 horizons)")
for grp in ("tb7", "rich", "gR"):
    g = [r for r in allrows if r[0] == grp]; big = [r for r in g if abs(r[3]) > tc]
    lines.append(f"{grp}: {len(g)} features, {len(big)} beyond the Bonferroni threshold, max |AUC-0.5| {max(abs(r[2]) for r in g):.4f}, max |t| {max(abs(r[3]) for r in g):.2f}")
lines.append("\nTop 20 by |t| overall:  group name  mean(AUC-0.5)  t  slices with the same sign  train-block mean")
for r in sorted(allrows, key=lambda r: -abs(r[3]))[:20]:
    lines.append(f"{r[0]:5s} {r[1]:28s} {r[2]:+.4f} t {r[3]:+.2f} same {r[4]:2d}/24 train {r[5]:+.4f}")
lines.append("\nTop 12 geometry features by |t|:")
for r in sorted([r for r in allrows if r[0] == "gR"], key=lambda r: -abs(r[3]))[:12]:
    lines.append(f"{r[0]:5s} {r[1]:28s} {r[2]:+.4f} t {r[3]:+.2f} same {r[4]:2d}/24 train {r[5]:+.4f}")
txt = "\n".join(lines); print(txt); open(os.path.join(HERE, "featscreen_report.txt"), "w", encoding="utf-8").write(txt + "\n")
