"""T06 analysis (CPU, seconds): reads the members' saved outputs, checks that each reproduces its logged per-slice AUC, builds E1 / E2 / E3
from the members that do, scores them against L0 on the 24 lab slices and the 6 FINAL slices, adds the walk-forward table, applies the
pre-registered criterion (SPEC.md) and writes results.json and tables.txt. Nothing is chosen from these numbers: the members and the
ensembles are fixed by SPEC.md; this script only reports. usage: python analyse.py"""
import glob, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); TAC = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import numpy as np
import ens_lib as E

PRED = os.environ.get("T06_PREDS", "D:/nt/nt_t06_preds")
LAB_CACHE = os.environ.get("LAB_CACHE", "D:/nt/nt_tactical/runs/tactical/lab/cache")
MEMBERS = ["L0", "L5", "L6", "L7", "R1y", "P1y"]
NAMES = {"L0": "sklearn logistic tb7 (L0)", "L5": "rich-43 linear (L5)", "L6": "+ residual MLP (L6)", "L7": "+ GRU on the window (L7)",
         "R1y": "regression on 1 y (newnet linear)", "P1y": "PatchTST-lite 1 y"}
# reproduction rule, fixed before any result: |mean over slices of (re-run - logged) auc3| <= 0.002 and no slice differs by more than 0.01
REPRO_MEAN, REPRO_MAX = 0.002, 0.01


def jl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def logged():
    """{member: {slice: auc3}} from the logged results (rebuild ladder; newnet verdict_try 1y and final_linear)."""
    out = {}
    names24 = sorted(os.path.basename(p)[:-4] for p in glob.glob(LAB_CACHE + "/*.npz"))
    latest = {}
    for r in jl(f"{TAC}/rebuild/results.jsonl"):
        if r["n_slices"] == 24:
            latest[r["step"]] = r
    for k in ("L0", "L5", "L6", "L7"):
        out[k] = dict(zip(names24, latest[k]["per_slice"]["auc3"]))
    nn = jl(f"{TAC}/newnet/results.jsonl")
    pick = lambda arch, tag: [r for r in nn if r["arch"] == arch and r.get("tag") == tag and r["seed"] == 0][0]
    r = pick("linear", "verdict_try"); out["R1y"] = dict(zip(r["slices"], r["per_slice"]["auc3"]))
    r = pick("linear", "final_linear"); out["R1y"].update(dict(zip(r["slices"], r["per_slice"]["auc3"])))
    r = pick("patch", "verdict_try"); out["P1y"] = dict(zip(r["slices"], r["per_slice"]["auc3"]))
    return out


def load_slice(name):
    a = np.load(f"{PRED}/{name}__lab.npz"); b = np.load(f"{PRED}/{name}__net.npz")
    assert np.allclose(a["rva"], b["rva"], rtol=1e-4, atol=2e-6), f"{name}: the lab and the network score different val blocks"
    d = dict(rva=a["rva"], rtr=a["rtr"], db=float(a["deadband"]), tail=a["tail"], val={}, tail_p={})
    for m in ("L0", "L5", "L6", "L7"):
        d["val"][m], d["tail_p"][m] = a[f"{m}_val"].astype(np.float64), a[f"{m}_tail"].astype(np.float64)
    for m in ("R1y", "P1y"):
        d["val"][m], d["tail_p"][m] = b[f"{m}_val"].astype(np.float64), b[f"{m}_tail"].astype(np.float64)
    return d


def fm(x, nd=4):
    return "%.*f [%.*f, %.*f]" % (nd, x[0], nd, x[1], nd, x[2])


def main():
    names24 = sorted(os.path.basename(p)[:-4] for p in glob.glob(LAB_CACHE + "/*.npz"))
    names6 = sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{TAC}/lab/cache_final/*.npz"))
    D = {n: load_slice(n) for n in names24 + names6}
    lg = logged(); res = dict(); lines = []
    # ---------------- reproduction
    repro = {}
    for m in MEMBERS:
        diffs = []
        for n in names24 + names6:
            if n in lg[m]:
                mine = E.score(D[n]["val"][m], D[n]["rva"], D[n]["rtr"], D[n]["db"])["auc3"]; diffs.append((n, mine, lg[m][n], mine - lg[m][n]))
        dd = np.array([x[3] for x in diffs])
        repro[m] = dict(n=len(dd), mean_diff=float(dd.mean()), mean_abs=float(np.abs(dd).mean()), max_abs=float(np.abs(dd).max()),
                        mean_mine=float(np.mean([x[1] for x in diffs])), mean_logged=float(np.mean([x[2] for x in diffs])),
                        ok=bool(abs(dd.mean()) <= REPRO_MEAN and np.abs(dd).max() <= REPRO_MAX),
                        per_slice=[dict(slice=x[0], rerun=round(x[1], 5), logged=round(x[2], 5)) for x in diffs])
    used = [m for m in MEMBERS if repro[m]["ok"]]; dropped = [m for m in MEMBERS if m not in used]
    res["reproduction"] = repro; res["used"] = used; res["dropped"] = dropped
    lines.append("== Reproduction of the members' logged per-slice mean-of-3 AUC (rule: |mean diff| <= %.3f and max |diff| <= %.2f) ==" % (REPRO_MEAN, REPRO_MAX))
    lines.append("member | slices checked | re-run mean | logged mean | mean diff | mean |diff| | max |diff| | reproduces")
    for m in MEMBERS:
        r = repro[m]
        lines.append(f"{m} {NAMES[m]} | {r['n']} | {r['mean_mine']:.4f} | {r['mean_logged']:.4f} | {r['mean_diff']:+.4f} | {r['mean_abs']:.4f} | {r['max_abs']:.4f} | {r['ok']}")
    lines.append("")
    # ---------------- per-slice scores
    cands = used + ["E1", "E2", "E3"]
    S = {c: {} for c in cands}
    for n, d in D.items():
        Pv = np.stack([d["val"][m] for m in used]); Pt = np.stack([d["tail_p"][m] for m in used])
        rtail = d["rtr"][d["tail"]]
        stk = E.fit_stacker(Pt, rtail, d["db"])
        P = dict({m: (d["val"][m], d["tail_p"][m]) for m in used})
        P["E1"] = (E.e1(Pv), E.e1(Pt)); P["E2"] = (E.e2(Pv), E.e2(Pt)); P["E3"] = (E.apply_stacker(stk, Pv), E.apply_stacker(stk, Pt))
        for c in cands:
            v, t = P[c]; sc = E.score(v, d["rva"], d["rtr"], d["db"]); thr = E.conf_threshold(t); tp = E.top_read(v, d["rva"], d["db"], thr)
            S[c][n] = dict(sc, thr=thr, top_hit=tp["hit"], top_bps=tp["bps"], top_n=tp["n"], ll_gap=sc["ll_h1"] - sc["ll_const_h1"])
    res["slices"] = {c: S[c] for c in cands}
    # ---------------- logit correlations (per slice, mean over the 24 lab slices)
    cm = np.zeros((len(used), len(used)))
    for n in names24:
        Z = np.stack([E.logit(D[n]["val"][m]).ravel() for m in used]); cm += E.mean_pair_corr(Z) / len(names24)
    res["logit_corr"] = dict(members=used, matrix=cm.round(4).tolist())
    lines.append("== Pairwise correlation of the members' logits on the validation rows (all three horizons stacked; mean over the 24 slices) ==")
    lines.append("      " + " ".join(f"{m:>6}" for m in used))
    for i, m in enumerate(used):
        lines.append(f"{m:>5} " + " ".join(f"{cm[i, j]:6.3f}" for j in range(len(used))))
    lines.append("")
    # ---------------- main table
    def arr(c, key, names):
        return np.array([S[c][n][key] for n in names])
    for title, names in (("24 lab slices (main)", names24), ("6 FINAL slices (confirmation)", names6)):
        lines.append(f"== {title}: mean-of-3 AUC; paired difference vs L0 (t-interval over slices); log loss h1 minus constant; top-10% read ==")
        lines.append("candidate | mean3 AUC [95% CI] | d vs L0 [95% CI] | slices > L0 | ll_h1 - const | top10 hit | top10 bps | trades (sum)")
        base = arr("L0", "auc3", names); tab = {}
        for c in cands:
            a = arr(c, "auc3", names); d = a - base
            row = dict(auc3=E.tci(a), d_vs_L0=E.tci(d), pos=int((d > 0).sum()), n=len(names), ll_gap=float(arr(c, "ll_gap", names).mean()),
                       hit=float(np.nanmean(arr(c, "top_hit", names))), bps=float(np.nanmean(arr(c, "top_bps", names))), trades=int(arr(c, "top_n", names).sum()))
            tab[c] = row
            lines.append(f"{c} | {fm(row['auc3'])} | {'-' if c == 'L0' else fm(row['d_vs_L0'])} | {'-' if c == 'L0' else str(row['pos']) + '/' + str(row['n'])} | "
                         f"{row['ll_gap']:+.4f} | {row['hit']:.4f} | {row['bps']:+.2f} | {row['trades']}")
        res["table_24" if names is names24 else "table_final"] = tab
        lines.append("")
    # ---------------- per-slice E1 - L0 (both sets)
    lines.append("== per-slice mean-of-3 AUC: L0, E1, E1 - L0 ==")
    for n in names24 + names6:
        lines.append(f"{n}{' (FINAL)' if n in names6 else ''} | {S['L0'][n]['auc3']:.4f} | {S['E1'][n]['auc3']:.4f} | {S['E1'][n]['auc3'] - S['L0'][n]['auc3']:+.4f}")
    lines.append("")
    # ---------------- walk-forward
    wf_path = os.path.join(HERE, "walkforward.json")
    if os.path.exists(wf_path):
        W = json.load(open(wf_path))["rows"]; months = [r["month"] for r in W]
        lines.append("== Walk-forward 2024-07 .. 2025-07-20, next-open labels (mean-of-3 AUC by month; 60-day train; stride 10) ==")
        lines.append("month | L0 | L5 | L6 | E1(L0,L5,L6) | E1 - L0 | n_test")
        for r in W:
            lines.append(f"{r['month']} | {r['L0']['auc3']:.4f} | {r['L5']['auc3']:.4f} | {r['L6']['auc3']:.4f} | {r['E1']['auc3']:.4f} | {r['E1']['auc3'] - r['L0']['auc3']:+.4f} | {r['n_test']}")
        wf = {}
        base = np.array([r["L0"]["auc3"] for r in W])
        lines.append("")
        lines.append("candidate | mean3 AUC over months [95% CI] | d vs L0 [95% CI] | months > L0 | h1 AUC | ll_h1 - const | top10 hit | top10 bps | trades")
        for c in ("L0", "L5", "L6", "E1"):
            a = np.array([r[c]["auc3"] for r in W]); d = a - base
            wf[c] = dict(auc3=E.tci(a), d_vs_L0=E.tci(d), pos=int((d > 0).sum()), auc_h1=float(np.mean([r[c]["auc_h1"] for r in W])),
                         ll_gap=float(np.mean([r[c]["ll_h1"] - r[c]["ll_const_h1"] for r in W])),
                         hit=float(np.nanmean([r[c]["top10_hit"] for r in W])), bps=float(np.nanmean([r[c]["top10_bps"] for r in W])),
                         trades=int(sum(r[c]["top10_n"] for r in W)))
            x = wf[c]
            lines.append(f"{c} | {fm(x['auc3'])} | {'-' if c == 'L0' else fm(x['d_vs_L0'])} | {'-' if c == 'L0' else str(x['pos']) + '/' + str(len(W))} | "
                         f"{x['auc_h1']:.4f} | {x['ll_gap']:+.4f} | {x['hit']:.4f} | {x['bps']:+.2f} | {x['trades']}")
        # reproduction of L0 h1 against the logged H59 (recut) months
        h59 = {r["month"]: r["auc"] for r in json.load(open(f"{TAC}/recut/lab/decay_nextopen.json"))}
        dd = [r["L0"]["auc_h1"] - h59[r["month"]] for r in W if r["month"] in h59]
        wf["L0_h1_vs_H59"] = dict(n=len(dd), mean_diff=float(np.mean(dd)), max_abs=float(np.max(np.abs(dd))))
        lines.append(f"\nL0 h1 monthly AUC vs the logged H59 (recut) months: {len(dd)} months, mean diff {np.mean(dd):+.4f}, max |diff| {np.max(np.abs(dd)):.4f}")
        res["walkforward"] = wf; res["walkforward_rows"] = W
        lines.append("")
    # ---------------- verdict
    t24, tf_ = res["table_24"], res["table_final"]
    d24 = t24["E1"]["d_vs_L0"]; dfin = tf_["E1"]["d_vs_L0"]
    crit_main = bool(d24[0] >= 0.005 and d24[1] > 0)
    crit_fin = bool(dfin[0] > 0)
    res["verdict"] = dict(E1_minus_L0_24=d24, E1_minus_L0_final=dfin, main_met=crit_main, final_same_sign=crit_fin, success=bool(crit_main and crit_fin),
                          rule="E1 - L0 >= +0.005 mean-of-3 AUC with the 95% CI over the 24 slices above 0, and the 6 FINAL slices have the same sign")
    lines.append("== Verdict against SPEC.md ==")
    lines.append(f"E1 - L0 on 24 slices: {d24[0]:+.4f} [{d24[1]:+.4f}, {d24[2]:+.4f}] (needs >= +0.005 and lower bound > 0): {'MET' if crit_main else 'NOT MET'}")
    lines.append(f"E1 - L0 on 6 FINAL: {dfin[0]:+.4f} (sign positive: {crit_fin})")
    lines.append(f"success = {crit_main and crit_fin}")
    json.dump(res, open(os.path.join(HERE, "results.json"), "w"), indent=1)
    open(os.path.join(HERE, "tables.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
