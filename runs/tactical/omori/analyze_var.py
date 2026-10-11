"""usage: python analyze_var.py sel|look -> results/var_<phase>_table.txt, var_<phase>_result.json (SPEC_var.md section 5).
Primary cells: alpha in {2.5%, 5%} x T in {15, 60} x tail in {left, right} (8). Gain = 1 - L(challenger)/L(FHS) in standardised units;
DM-type test: one-sided t test of the monthly mean loss differences over months (df = months-1); Bonferroni over the 8 cells."""
import os, sys, json, math
import numpy as np
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results")
PRIMARY_ALPHAS = (0.025, 0.05)


def lr_tests(counts, al):
    n00, n01, n10, n11 = counts; N = sum(counts); hits = n01 + n11; pi = hits / N

    def ll(p, k, m):
        return (k * math.log(p) if k else 0.0) + ((m - k) * math.log(1 - p) if m - k else 0.0)
    lr_uc = -2 * (ll(al, hits, N) - ll(pi, hits, N)) if 0 < pi < 1 else float("nan")
    a = n01 / max(n00 + n01, 1); b = n11 / max(n10 + n11, 1)
    l1 = ll(a, n01, n00 + n01) + ll(b, n11, n10 + n11) if 0 < a < 1 and 0 < b < 1 else float("nan")
    l0 = ll(pi, hits, N) if 0 < pi < 1 else float("nan")
    lr_ind = -2 * (l0 - l1) if np.isfinite(l1) else float("nan")
    return dict(hit=pi, p_uc=float(1 - stats.chi2.cdf(lr_uc, 1)), p_ind=float(1 - stats.chi2.cdf(lr_ind, 1)) if np.isfinite(lr_ind) else float("nan"),
                p_cc=float(1 - stats.chi2.cdf(lr_uc + lr_ind, 2)) if np.isfinite(lr_ind) else float("nan"), n=N)


def main(phase):
    rows = [json.loads(l) for l in open(f"{RES}/var_{phase}.jsonl", encoding="utf-8")]
    cells = {}
    for r in rows:
        for c in r["cells"]:
            cells.setdefault((r["T"], c["tail"], c["alpha"]), []).append((r["month"], c))
    out = []; L = [f"phase var_{phase}: {len(rows)} (month,T) rows"]
    for key in sorted(cells):
        T, tail, al = key; cs = [c for _, c in sorted(cells[key], key=lambda x: x[0])]
        n = len(cs); Lf = np.array([c["L_fhs"] for c in cs]); Lc = np.array([c["L_c"] for c in cs]); Lk = np.array([c["L_k"] for c in cs])

        def dm(a, b):
            d = a - b; se = d.std(ddof=1) / math.sqrt(n); t = d.mean() / se
            return dict(gain=float(d.mean() / a.mean()), t=float(t), p1=float(1 - stats.t.cdf(t, n - 1)), lo=float((d.mean() - stats.t.ppf(.975, n - 1) * se) / a.mean()))
        rec = dict(T=T, tail=tail, alpha=al, months=n, primary=al in PRIMARY_ALPHAS, c_vs_fhs=dm(Lf, Lc), k_vs_fhs=dm(Lf, Lk), c_vs_k=dm(Lk, Lc), L_fhs=float(Lf.mean()))
        rec["p_bonf8"] = min(1.0, 8 * rec["c_vs_fhs"]["p1"]) if rec["primary"] else None
        for arm in ("fhs", "k", "c"):
            cnt = np.sum([c[f"tr_{arm}"] for c in cs], axis=0).tolist(); rec[f"cov_{arm}"] = lr_tests(cnt, al)
            z = np.array([c[f"z2_{arm}"] for c in cs]); rec[f"z2_{arm}"] = dict(mean=float(z.mean()), p=float(2 * (1 - stats.t.cdf(abs(z.mean() / (z.std(ddof=1) / math.sqrt(n))), n - 1))))
            rec[f"hit_{arm}"] = float(np.mean([c[f"hit_{arm}"] for c in cs]))
        out.append(rec)
        L.append(f"T={T:2d} {tail:5s} a={al:.3f} n={n}  FZ0 gain C vs FHS {100*rec['c_vs_fhs']['gain']:+6.2f}% [lo {100*rec['c_vs_fhs']['lo']:+6.2f}] p1 {rec['c_vs_fhs']['p1']:.4f}"
                 + (f" Bonf8 {rec['p_bonf8']:.4f}" if rec["primary"] else " (secondary)")
                 + f" | FZ0-const vs FHS {100*rec['k_vs_fhs']['gain']:+6.2f}% | C vs const {100*rec['c_vs_k']['gain']:+6.2f}% p1 {rec['c_vs_k']['p1']:.4f}"
                 + f" | hit FHS/const/C {rec['hit_fhs']:.4f}/{rec['hit_k']:.4f}/{rec['hit_c']:.4f} | Kupiec/ind/cc p (C) {rec['cov_c']['p_uc']:.3f}/{rec['cov_c']['p_ind']:.3f}/{rec['cov_c']['p_cc']:.3f}"
                 + f" | Z2 FHS {rec['z2_fhs']['mean']:+.3f} (p {rec['z2_fhs']['p']:.3f}) C {rec['z2_c']['mean']:+.3f} (p {rec['z2_c']['p']:.3f})")
    res = dict(phase=phase, cells=out)
    if phase == "sel":
        sel = [dict(T=r["T"], tail=r["tail"], alpha=r["alpha"]) for r in out if r["primary"] and r["c_vs_fhs"]["gain"] >= 0.01 and r["p_bonf8"] < 0.05
               and r["cov_c"]["p_cc"] > 0.05 and r["z2_c"]["p"] > 0.05]
        res["cells_selected"] = sel; json.dump(dict(cells=sel), open(f"{RES}/var_selected.json", "w"), indent=1)
        L.append(f"selected primary cells (gain >= 1%, Bonferroni-8 p < 0.05, coverage cc p > 0.05, Z2 p > 0.05 for C): {sel}")
    else:
        sel = json.load(open(f"{RES}/var_selected.json"))["cells"]; k = len(sel); conf = []
        for s in sel:
            r = next(x for x in out if (x["T"], x["tail"], x["alpha"]) == (s["T"], s["tail"], s["alpha"]))
            ok = r["c_vs_fhs"]["gain"] >= 0.005 and min(1.0, k * r["c_vs_fhs"]["p1"]) < 0.05
            conf.append(dict(**s, gain=r["c_vs_fhs"]["gain"], p_bonf=min(1.0, k * r["c_vs_fhs"]["p1"]), confirmed=bool(ok)))
        res["confirmation"] = conf; L.append(f"confirmation: {conf}")
    json.dump(res, open(f"{RES}/var_{phase}_result.json", "w"), indent=1)
    open(f"{RES}/var_{phase}_table.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1])
