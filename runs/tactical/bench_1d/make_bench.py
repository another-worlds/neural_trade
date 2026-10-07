"""Throughput benchmark on a 1-day training block (owner, 2026-10-07: "yes, but take a 1-day block, not 7 days").
Fixed before any run. Training block 1,440 windows (MAX_SEQUENCE_COUNT 18,000 with the screen's N_FOLDS 2 / 10% / 10%
split), 8 epochs, TRAIN_METRICS_EVERY 10, the 6 slices of the candidate check's C2 x seeds 0-1 (the owner's 6x2 design).
Arms (run one after another, never overlapping, with nothing else of ours on the GPU):
  A  bs256_x1   batch 256, one process
  B  bs256_x3   batch 256, three processes at once (shards 0-2/3 of the same 12 trials)
  C  bs1024_x1  batch 1024, one process
Measured: trials per hour (wall clock of the whole arm), seconds per training step (median of epochs 2-8 / steps),
and for C vs A all 9 outputs paired by (slice, seed). LR is NOT rescaled for C (a separate question if C is faster)."""
import os, sys
import yaml
os.chdir(r"D:\nt\nt_tactical"); sys.path.insert(0, "runs/tactical")
import make_hc2 as m

allv = m.slices(); climb = [s for i, s in enumerate(allv) if i % 5 != 4]
SLICES = climb[1::7][:6]
BASE = {"CSV_PATH": "D:/nt/neural_trade/Bitcoin_BTCUSDT.csv", "N_FOLDS": 2, "VAL_FRACTION": 0.1, "CAL_FRACTION": 0.1,
        "MAX_SEQUENCE_COUNT": 18000, "FOLD_INDEX": -2, "TRAIN_METRICS_EVERY": 10}
RULES = {"finite": True, "max_nonfinite_grad_steps": 0, "max_clipped_share": 1.0, "clip_skip_epochs": 1,
         "min_train_loss_drop": -10.0, "max_term_share": 1.0}
for name, bs in (("bench_bs256_x1", 256), ("bench_bs256_x3", 256), ("bench_bs1024_x1", 1024)):
    spec = {"schema_version": 1, "name": name, "base_config": "../default.yaml", "overrides": dict(BASE, BATCH_SIZE=bs),
            "slices": SLICES, "seeds": [0, 1], "run": {"calibrate": False, "epochs": 8, "save_predictions": True}, "rules": RULES}
    yaml.safe_dump(spec, open(f"configs/tactical/{name}.yaml", "w"), sort_keys=False)
print(SLICES)
