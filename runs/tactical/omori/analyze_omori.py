"""usage: python analyze_omori.py simA|simB|sel|look  -> results/<phase>_table.txt, <phase>_result.json (verdict against SPEC.md sections 4-5, 7)."""
import os, sys, json, math
import numpy as np
import pandas as pd
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results")


def tci(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]; n = len(v); m = v.mean(); se = v.std(ddof=1) / math.sqrt(n)
    h = stats.t.ppf(0.975, n - 1) * se; t = m / se
    return dict(mean=float(m), lo=float(m - h), hi=float(m + h), t=float(t), p1=float(1 - stats.t.cdf(t, n - 1)), n=n)


def holm(ps):
    o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * ps[i])); adj[i] = run
    return adj


def load(phase):
    rows = [json.loads(l) for l in open(f"{RES}/{phase}.jsonl", encoding="utf-8")]
    return pd.DataFrame(rows)


def main(phase):
    df = load(phase); out = {}; lines = []
    for T in sorted(df["T"].unique()):
        d = df[df["T"] == T].sort_values("month")
        sk = lambda a, b, kind="ll": 1 - d[f"{kind}_{a}"] / d[f"{kind}_{b}"]
        cells = {"ll_pow_vs_B": sk("pow", "B"), "bs_pow_vs_B": sk("pow", "B", "bs"), "ll_exp_vs_B": sk("exp", "B"), "ll_pow_vs_exp": sk("pow", "exp"),
                 "ll_pow_nofs_vs_B": sk("pow_nofs", "B"), "ll_fs_vs_B": sk("fs", "B"), "ll_B_vs_const": sk("B", "const"), "ll_pow_vs_const": sk("pow", "const")}
        r = {k: tci(v) for k, v in cells.items()}
        d = d.assign(year=d["month"].str[:4]); half = d["month"].map(lambda m: m[:4] + ("H1" if int(m[5:]) <= 6 else "H2"))
        py = d.groupby("year")["p"].mean(); out[str(T)] = dict(cells=r, p_mean=float(d["p"].mean()), p_by_year={k: float(v) for k, v in py.items()},
            p_year_range=float(py.max() - py.min()), c_share10=float((d["c"] == 10).mean()), coef_kernel=float(d["coef_kernel"].mean()),
            months=int(len(d)), events=int(d["events"].sum()), n_eff_month=float(d["n_eff"].mean()), rate=float(d["rate_tr"].mean()),
            pos_coef_share=float((d["coef_kernel"] > 0).mean()))
    if phase in ("sel", "look") or phase.startswith("sim"):
        ps = [out[str(T)]["cells"]["ll_pow_vs_B"]["p1"] for T in sorted(df["T"].unique())]; adj = holm(np.array(ps))
        for T, a in zip(sorted(df["T"].unique()), adj):
            out[str(T)]["holm_p"] = float(a)
    res = {"phase": phase, "by_T": out}
    if phase == "sel":
        o = out["30"]; c = o["cells"]["ll_pow_vs_B"]
        crit = dict(skill_ge_1pct=c["mean"] >= 0.01, lo_gt_0=c["lo"] > 0, holm_lt_05=o["holm_p"] < 0.05, p_in_06_13=0.6 <= o["p_mean"] <= 1.3,
                    p_years_within_02=o["p_year_range"] < 0.2)
        res["criteria"] = crit; res["selected"] = bool(all(crit.values()))
        json.dump(res, open(f"{RES}/selected.json", "w"), indent=1)
        dd = df[df["T"] == 30]      # the shock state #6 reuses: the selection's mean exponent (to the grid) and the modal c
        json.dump(dict(p_star=round(float(round(o["p_mean"] * 10) / 10), 1), c_star=float(dd["c"].mode().iloc[0])), open(f"{RES}/state.json", "w"))
    if phase == "look":
        c = out["30"]["cells"]["ll_pow_vs_B"]; res["confirmed"] = bool(c["mean"] >= 0.005 and c["lo"] > 0 and c["p1"] < 0.05)
    if phase.startswith("sim"):
        c = out["30"]["cells"]["ll_pow_vs_B"]; res["mean_skill"] = c["mean"]; res["p_mean"] = out["30"]["p_mean"]
        res["pass"] = (c["mean"] < 0.003) if phase == "simA" else bool(c["mean"] >= 0.01 and abs(out["30"]["p_mean"] - 0.9) <= 0.2)
    json.dump(res, open(f"{RES}/{phase}_result.json", "w"), indent=1)
    L = [f"phase {phase}"]
    for T, o in out.items():
        L.append(f"T={T}: months {o['months']}  events {o['events']}  mean n_eff/month {o['n_eff_month']:.0f}  train rate {o['rate']:.4f}")
        for k, v in o["cells"].items():
            L.append(f"   {k:20s} mean {100*v['mean']:+7.3f}%  CI [{100*v['lo']:+7.3f}, {100*v['hi']:+7.3f}]  t {v['t']:+6.2f}  p1 {v['p1']:.4f}")
        L.append(f"   Holm p (pow vs B) {o.get('holm_p', float('nan')):.4f}   p mean {o['p_mean']:.3f}  by year {json.dumps({k: round(v,2) for k,v in o['p_by_year'].items()})}  range {o['p_year_range']:.2f}  c=10 share {o['c_share10']:.2f}  kernel coef>0 share {o['pos_coef_share']:.2f}")
    for k in ("criteria", "selected", "confirmed", "pass"):
        if k in res:
            L.append(f"{k}: {res[k]}")
    open(f"{RES}/{phase}_table.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1])
