"""Tables (per experiment) from results.jsonl and results_ft.jsonl -> table4.txt. usage: python summary4.py"""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
f = lambda x: f"{x[0]:+.4f} [{x[1]:+.4f},{x[2]:+.4f}]"
rows = [json.loads(l) for l in open(HERE + "/results.jsonl", encoding="utf-8")]
out = []
exps = []
for r in rows:
    if r["experiment"] not in exps: exps.append(r["experiment"])
for e in exps:
    R = [r for r in rows if r["experiment"] == e]
    r0 = R[0]; t = r0["probes"]["a_tb7"]
    out.append(f"\n== {e}: {r0['n_slices']} slices with train start >= {r0['cutoff']} (first {r0['slices'][0]['train_start']}, last {r0['slices'][-1]['train_start']})")
    out.append(f"tb7: mean3 AUC {f(t['auc3'])} | vol ridge {f(t['vol_ridge'])} | vol boosting {f(t['vol_hgb'])}")
    out.append("variant (steps, params) | probe | mean3 AUC | d vs tb7 | vol ridge | d vs tb7 | vol boosting | d vs tb7 || vs v2c: dAUC | d ridge | d boosting | eff_rank")
    for r in R:
        m = r["pretrain"]; tag = f"{r['variant']} ({m.get('steps', 30000)} steps, {m.get('params', 26720)} par)"
        for p in ("b_emb", "c_both"):
            q, d = r["probes"][p], r["paired_vs_tb7"][p]; v = r["paired_vs_ref"].get(p)
            vs = f"{f(v['auc3'])} | {f(v['vol_ridge'])} | {f(v['vol_hgb'])}" if v else "(reference)"
            rk = f"{r['collapse']['eff_rank'][0]:.1f}" if r["collapse"] and p == "b_emb" else ""
            out.append(f"{tag} | {p} | {f(q['auc3'])} | {f(d['auc3'])} | {f(q['vol_ridge'])} | {f(d['vol_ridge'])} | {f(q['vol_hgb'])} | {f(d['vol_hgb'])} || {vs} | {rk}")
ft = os.path.join(HERE, "results_ft.jsonl")
if os.path.exists(ft):
    out.append("\n== exp5 fine-tuning (per-slice end-to-end; direction vs the tb7 logreg, volatility vs HAR-RV fitted on the same train block)")
    for l in open(ft, encoding="utf-8"):
        r = json.loads(l)
        if r["tag"] == "smoke": continue
        v = r["vol"]
        out.append(f"{r['tag']} ({r['n_slices']} slices, start >= {r['cutoff']})")
        out.append(f"  direction mean3 AUC: tb7 {f(r['dir']['base'])} | fine-tuned {f(r['dir']['ft'])} | d {f(r['dir']['d_ft_vs_base'])}")
        out.append(f"  vol Spearman: fine-tuned {f(v['ft'])} | tb7 ridge {f(v['ridge_tb7'])} | HAR-RV {f(v['har'])} | HAR+tod {f(v['har_tod'])}; d ft-HARtod {f(v['d_ft_vs_har_tod'])}")
        out.append(f"  vol B (linear part tb7+HAR): fine-tuned {f(v['B_ft'])} | ridge B {f(v['B_ridge'])}; d ft-HARtod {f(v['d_Bft_vs_har_tod'])}; d ft-ridgeB {f(v['d_Bft_vs_Bridge'])}")
        out.append(f"  best inner epoch (dir) per slice: {[x['dir_best_epoch'] for x in r['rows']]}; (vol) {[x['vol_best_epoch'] for x in r['rows']]}; (volB) {[x['volB_best_epoch'] for x in r['rows']]}")
txt = "\n".join(out); print(txt); open(HERE + "/table4.txt", "w", encoding="utf-8").write(txt + "\n")
