"""Table (variant x probe) from results.jsonl + loss curves and collapse diagnostics from the pretraining logs.
usage: python summary.py  -> prints the table, writes table.txt and loss_curves.png"""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
VARS = ["v1", "ctl", "a", "b", "c"]
NAME = {"v1": "v1 (CPU, 15 bars)", "ctl": "ctl (v1 task, v2 code)", "a": "a (60 bars, 3 seg)", "b": "b (masked ctx)", "c": "c (a+b)"}
f = lambda x: f"{x[0]:+.4f} [{x[1]:+.4f},{x[2]:+.4f}]"
rows = {}
for l in open(HERE + "/results.jsonl", encoding="utf-8"):
    r = json.loads(l); rows[r["variant"]] = r            # the last row per variant
out = []
t = rows[next(iter(rows))]["probes"]["a_tb7"]
out.append(f"tb7 alone ({rows[next(iter(rows))]['n_slices']} slices): mean3 AUC {f(t['auc3'])}; vol Spearman ridge {f(t['vol_ridge'])}, boosting {f(t['vol_hgb'])}\n")
out.append("variant | pretrain | probe | mean3 AUC | paired vs tb7 | vol ridge | paired vs tb7 | vol boosting | paired vs tb7")
for v in VARS:
    if v not in rows: continue
    r = rows[v]; m = r["pretrain"]
    for p in ("b_emb", "c_both"):
        q, d = r["probes"][p], r["paired_vs_tb7"][p]
        out.append(f"{NAME[v]} | {m.get('sec', 1077)}s {m['steps']} steps {m.get('device', 'cpu')} | {p} | {f(q['auc3'])} | {f(d['auc3'])} | "
                   f"{f(q['vol_ridge'])} | {f(d['vol_ridge'])} | {f(q['vol_hgb'])} | {f(d['vol_hgb'])}")
out.append("\npaired vs v1 (same slices): variant probe | d AUC | d vol ridge | d vol boosting")
for v in VARS:
    if v in rows and rows[v]["paired_vs_v1"]:
        for p, d in rows[v]["paired_vs_v1"].items(): out.append(f"{v} {p} | {f(d['auc3'])} | {f(d['vol_ridge'])} | {f(d['vol_hgb'])}")
out.append("\ncollapse (val embeddings over slices, mean [CI]): variant | std mean | std min | effective rank (of 32)")
for v in VARS:
    if v in rows: c = rows[v]["collapse"]; out.append(f"{v} | {c['std_mean'][0]:.3f} | {c['std_min'][0]:.3f} | {c['eff_rank'][0]:.1f}")
txt = "\n".join(out); print(txt); open(HERE + "/table.txt", "w", encoding="utf-8").write(txt + "\n")

fig, ax = plt.subplots(2, 3, figsize=(16, 8)); cols = {"v1": "#888", "ctl": "#3987e5", "a": "#d95926", "b": "#199e70", "c": "#7a3fb0"}
for v in VARS:
    p = f"{HERE}/ckpt/{v}/pretrain_log.jsonl"
    if not os.path.exists(p): continue
    L = [json.loads(l) for l in open(p)]; s = [x["step"] for x in L]
    ax[0, 0].plot(s, [x.get("loss", x.get("loss")) for x in L], c=cols[v], label=v)
    ax[0, 1].plot(s, [x.get("loss_fut", x.get("smooth_l1")) for x in L], c=cols[v], label=v)
    if v in ("b", "c"): ax[0, 2].plot(s, [x["loss_mask"] for x in L], c=cols[v], label=v)
    ax[1, 0].plot(s, [x["eff_rank"] for x in L], c=cols[v], label=v)
    ax[1, 1].plot(s, [x["std_mean"] for x in L], c=cols[v], label=v)
    ax[1, 2].plot(s, [x["cov"] for x in L], c=cols[v], label=v)
for a, ti in zip(ax.flat, ["total loss", "future-segment loss (smooth L1)", "masked-patch loss (smooth L1)", "effective rank (of 32)", "embedding std (mean over dims)", "VICReg covariance term"]):
    a.set_title(ti); a.set_xlabel("step"); a.legend(); a.grid(alpha=.3)
plt.tight_layout(); plt.savefig(HERE + "/loss_curves.png", dpi=110)
