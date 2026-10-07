"""Scale-up of the 10 best-AUC configurations of the old screen campaign (l1_*) to 7-day training blocks.
Owner (2026-10-07): 'take the top-10 high-AUC candidates to longer runs (7d), on the off chance'. This is an
explicit exception to the 2-minute-per-run rule for these runs only (D-063 stays for everything else).
Selection (fixed before any run): configs of the old campaign with >= 6 trials, ranked by mean direction AUC
(mean of h0-h2) over all their trials. NOTE: that ranking is partly luck (the top 10 lose ~0.05 AUC on a fresh seed,
see the 2026-10-07 assessment), which is what this experiment tests.
Layout: reference training block of 7 days = 10,080 train windows (MAX_SEQUENCE_COUNT 126,000 with the screen's
N_FOLDS 2 / 10% / 10% split), batch 256, 8 epochs, the same 4 slices the old campaign used, seed 0.
usage: python make_hc3.py   -> configs/tactical/hc3_cand01..10.yaml, hc3_default.yaml, runs/tactical/hc3_candidates.json"""
import glob, json, os, statistics as st

import yaml

os.chdir(r"D:\nt\nt_tactical")
cfg = {}
for f in glob.glob(r"D:/nt/neural_trade/runs/screens/l1_*/results*.jsonl"):
    for l in open(f, encoding="utf-8"):
        r = json.loads(l); a = r.get("direction_auc") or {}
        if all(a.get(h) and a[h].get("auc") is not None for h in ("h0", "h1", "h2")):
            key = (r["screen"], json.dumps(r["config_diff"], sort_keys=True))
            cfg.setdefault(key, []).append(st.mean(a[h]["auc"] for h in ("h0", "h1", "h2")))
ranked = sorted(((st.mean(v), k, len(v)) for k, v in cfg.items() if len(v) >= 6), reverse=True)[:10]
SLICES = ["2020-03-12T23:00:00", "2021-05-19T23:00:00", "2023-07-15T00:00:00", "2024-12-05T00:00:00"]
over = {"CSV_PATH": "D:/nt/neural_trade/Bitcoin_BTCUSDT.csv", "N_FOLDS": 2, "VAL_FRACTION": 0.1, "CAL_FRACTION": 0.1,
        "MAX_SEQUENCE_COUNT": 126000, "FOLD_INDEX": -2, "BATCH_SIZE": 256, "TRAIN_METRICS_EVERY": 10}


def spec(name, extra):
    o = dict(over); o.update(extra)
    return {"schema_version": 1, "name": name, "base_config": "../default.yaml", "overrides": o, "slices": SLICES,
            "seeds": [0], "run": {"calibrate": False, "epochs": 8},
            "rules": {"finite": True, "max_nonfinite_grad_steps": 0, "max_clipped_share": 1.0, "clip_skip_epochs": 1,
                      "min_train_loss_drop": -10.0, "max_term_share": 1.0}}


meta = []
for i, (m, k, n) in enumerate(ranked, 1):
    extra = json.loads(k[1])
    name = f"hc3_cand{i:02d}"
    yaml.safe_dump(spec(name, extra), open(f"configs/tactical/{name}.yaml", "w"), sort_keys=False)
    meta.append({"name": name, "old_screen": k[0], "old_mean_auc": round(m, 4), "old_trials": n, "config_diff": extra})
yaml.safe_dump(spec("hc3_default", {}), open("configs/tactical/hc3_default.yaml", "w"), sort_keys=False)
json.dump(meta, open("runs/tactical/hc3_candidates.json", "w"), indent=1)
for x in meta:
    print(x["name"], x["old_mean_auc"], x["old_screen"], str(x["config_diff"])[:90])
