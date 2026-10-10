"""Report for readout.py rows (experiments 1 and 2): per (set, model) over the 24 slices.
mean3 AUC [95% t-CI over slices], paired diff vs tb7 logistic, paired diff vs the same model without geometry, h1 log loss minus the
constant's, honest top-10% hit and excess = hon10_bps - hon10_null95 [CI]. usage: python readout_report.py"""
import glob, json, os
import numpy as np
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__))
rows = [json.loads(l) for f in sorted(glob.glob(f"{HERE}/readout_rows_*.jsonl")) for l in open(f)]
by = {}
for r in rows:
    by.setdefault((r["set"], r["model"]), {})[r["slice"]] = r
def ci(v):
    v = np.asarray(v, float); h = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)); return f"{v.mean():+.4f} [{v.mean() - h:+.4f},{v.mean() + h:+.4f}]"
def paired(a, b, key):
    s = [x for x in a if x in b]; return np.array([a[x][key] - b[x][key] for x in s]) if len(s) > 1 else None
ref = by.get(("tb7", "logreg"), {})
out = []
sets = ["tb7", "rich", "g5", "gR", "tb7+g5", "rich+g5", "tb7+gR", "rich+gR"]
for m in ("logreg", "hgb", "mlp", "logreg_c01", "logreg_c001", "hgb_reg", "mlp_reg"):
    out.append(f"\n== model {m}: 24 slices; mean3 AUC, d vs tb7 logreg, d vs same model on the base set (tb7/rich), h1 logloss - constant, top-10% hit, excess over null95")
    for s in sets:
        d = by.get((s, m))
        if not d:
            continue
        n = len(d); a = np.array([x["auc3"] for x in d.values()]); base_set = s.split("+")[0] if "+" in s else None
        p1 = paired(d, ref, "auc3"); p2 = paired(d, by.get((base_set, m), {}), "auc3") if base_set else None
        ex = np.array([x["hon10_bps"] - x["hon10_null95"] for x in d.values()]); ll = np.array([x["ll_h1"] - x["ll_const_h1"] for x in d.values()])
        out.append(f"{s:9s} n={n:2d} feats {next(iter(d.values()))['n_feat']:3d} | AUC {ci(a)} | d vs tb7-logreg {ci(p1) if p1 is not None and (s, m) != ('tb7', 'logreg') else '-':27s} | "
                   f"d vs {base_set or 'base'}-{m} {ci(p2) if p2 is not None else '-':27s} | ll-const {ll.mean():+.4f} | hit {np.mean([x['hon10_hit'] for x in d.values()]):.3f} excess {ci(ex)}")
txt = "\n".join(out); print(txt); open(f"{HERE}/readout_report.txt", "w", encoding="utf-8").write(txt + "\n")
