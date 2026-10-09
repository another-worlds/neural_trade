"""newnet2: share of fits whose best direction epoch is 0 (= the regression untouched), per record; python stop_stats.py"""
import json, numpy as np
for l in open("results.jsonl", encoding="utf-8"):
    r = json.loads(l)
    if not r["slice_info"] or r["arch"] == "linear": continue
    b = np.array([e["best_epoch_dir"] for e in r["slice_info"]]); ep = np.array([e["epochs_run"] for e in r["slice_info"]])
    print(f"{r.get('tag') or 'orig':16s} {r['arch']:6s} {r['span']:3s} seed {r['seed']} n={len(b):2d} best_epoch==0: {np.mean(b == 0):.2f}  best<=1: {np.mean(b <= 1):.2f}  median best {np.median(b):.0f}  median epochs run {np.median(ep):.0f}")
# regression's own numbers (from the _lin columns) on the 24 / 6 / FINAL slices
for l in open("results.jsonl", encoding="utf-8"):
    r = json.loads(l); ps = r["per_slice"]
    if r["arch"] == "patch" and r["seed"] == 0 and r.get("tag") in ("orig_3y", "final_trainlin", "verdict_try", "screen_trainlin") and "ll_h1_lin" in ps:
        print(f"regression on {r['tag']} slices ({r['span']}, n={r['n_slices']}): mean3 {np.mean(ps['auc3_lin']):.4f}, ll-const {np.mean(np.array(ps['ll_h1_lin']) - np.array(ps['ll_const_h1'])):+.4f}")
