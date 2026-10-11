"""Experiment 3b: single-feature volatility screen on the minute lab slices' val blocks. For every geometry feature (22 channels) and for the
window's own volatility (log sd of 1-bar changes), Spearman with log|r| at the 3 horizons (mean), raw and after removing the window volatility
(residual of log|r| and of the feature on log window sd, linear). Across the 24 slices: mean, t, same-sign count. Does the squeeze (short-term
sd over window sd) predict a larger coming move beyond the volatility level? usage: python volscreen.py"""
import glob, os, sys
import numpy as np
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "..", "lab"))
import lab, geomlib as G  # noqa: E402

names = G.geom_names(None, (3, 5, 10, 20)); rows = []
def resid(y, x):
    X = np.column_stack([np.ones(len(x)), x]); return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
for p in sorted(glob.glob(lab.CACHE)):
    D = np.load(p); W = D["Wva"]; r = np.abs(D["rva"]).astype(np.float64)
    F = G.geom_rich(W).astype(np.float64); lv = np.log(np.diff(W[..., 3].astype(np.float64), axis=1).std(1) + 1e-9)
    Y = np.log(r + 1e-5); raw = np.zeros((F.shape[1], 3)); res = np.zeros((F.shape[1], 3))
    from scipy.stats import rankdata
    rF = np.apply_along_axis(rankdata, 0, F); rlv = rankdata(lv)
    for h in range(3):
        ry = rankdata(Y[:, h]); ryr = resid(ry, rlv)
        for j in range(F.shape[1]):
            raw[j, h] = np.corrcoef(rF[:, j], ry)[0, 1]; res[j, h] = np.corrcoef(resid(rF[:, j], rlv), ryr)[0, 1]
    rows.append((raw.mean(1), res.mean(1), np.mean([np.corrcoef(rlv, rankdata(Y[:, h]))[0, 1] for h in range(3)])))
raw = np.array([x[0] for x in rows]); res = np.array([x[1] for x in rows]); base = np.array([x[2] for x in rows]); n = len(rows)
t = lambda a: a.mean(0) / (a.std(0, ddof=1) / np.sqrt(n))
lines = [f"{n} slices; window volatility alone: Spearman with log|r| {base.mean():+.3f} [t {base.mean() / (base.std(ddof=1) / np.sqrt(n)):.1f}]"]
tc = stats.t.ppf(1 - 0.05 / (2 * len(names)), n - 1)
by = {}
for j, nm in enumerate(names):
    by.setdefault(nm.split(".")[1].rstrip("0123456789"), []).append(j)
lines.append(f"{len(names)} geometry features; Bonferroni |t| > {tc:.2f}")
for typ, js in by.items():
    lines.append(f"  {typ:8s}: {len(js)} features; raw: max|rho| {np.abs(raw.mean(0)[js]).max():.3f}, beyond Bonferroni {int((np.abs(t(raw))[js] > tc).sum())}; "
                 f"after window-vol control: max|rho| {np.abs(res.mean(0)[js]).max():.3f}, beyond Bonferroni {int((np.abs(t(res))[js] > tc).sum())}")
lines.append("Top 12 after the window-volatility control:  name  rho  t  same sign")
for j in np.argsort(-np.abs(t(res)))[:12]:
    lines.append(f"  {names[j]:24s} {res.mean(0)[j]:+.3f} t {t(res)[j]:+.1f} same {int((np.sign(res[:, j]) == np.sign(res.mean(0)[j])).sum())}/{n}  (raw {raw.mean(0)[j]:+.3f})")
sq = [j for j, nm in enumerate(names) if nm.endswith("squeeze")]
lines.append("Squeeze features (after control): " + ", ".join(f"{names[j].split('.')[0]} {res.mean(0)[j]:+.3f}" for j in sq))
txt = "\n".join(lines); print(txt); open(os.path.join(HERE, "volscreen_report.txt"), "w", encoding="utf-8").write(txt + "\n")
