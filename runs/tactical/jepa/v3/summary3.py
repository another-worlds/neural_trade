"""Table of the v3 probes (results_probe.jsonl) and pretraining loss at 30k/60k/120k steps. usage: python summary3.py > table.txt"""
import json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
f = lambda x: f"{x[0]:+.4f} [{x[1]:+.4f},{x[2]:+.4f}]"
rows = {}
for l in open(HERE + "/results_probe.jsonl"):
    r = json.loads(l); rows[r["variant"]] = r
print("variant | emb vol ridge | emb vol hgb | tb7+emb vol ridge | tb7+emb vol hgb | tb7+emb AUC | same, first 11 slices (data_end < 2024-07-01): tb7+emb ridge / hgb")
for v, r in rows.items():
    b, c = r["probes"]["b_emb"], r["probes"]["c_both"]
    print(f"{v} | {f(b['vol_ridge'])} | {f(b['vol_hgb'])} | {f(c['vol_ridge'])} | {f(c['vol_hgb'])} | {f(c['auc3'])} | "
          f"{np.mean(c['per_slice_vol_ridge'][:11]):.4f} / {np.mean(c['per_slice_vol_hgb'][:11]):.4f}")
print("\npretraining logs (mean over the 500-step window ending at the step): variant step total_loss fut mask cov eff_rank")
for s in (0, 1):
    L = {x["step"]: x for x in map(json.loads, open(f"{HERE}/ckpt/c_s{s}/pretrain_log.jsonl"))}
    for st in (30000, 60000, 120000):
        x = L[st]; print(f"s{s} {st} {x['loss']:.4f} {x['loss_fut']:.4f} {x['loss_mask']:.4f} {x['cov']:.4f} {x['eff_rank']:.1f}")
print("\nfinal 5000 steps mean loss: " + ", ".join(f"s{s} " + f"{np.mean([json.loads(l)['loss'] for l in open(f'{HERE}/ckpt/c_s{s}/pretrain_log.jsonl')][-10:]):.4f}" for s in (0, 1)))
