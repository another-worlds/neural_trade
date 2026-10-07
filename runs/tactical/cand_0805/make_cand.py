"""Specs for the candidate check (SPEC.md in this folder). Writes configs/tactical/cand_{c1,c2}_{skip,base}.yaml."""
import os, sys
import yaml
os.chdir(r"D:\nt\nt_tactical"); sys.path.insert(0, "runs/tactical")
import make_hc2 as m

allv = m.slices(); climb = [s for i, s in enumerate(allv) if i % 5 != 4]
C2_SLICES = climb[1::7][:6]
SCREEN = {"CSV_PATH": "D:/nt/neural_trade/Bitcoin_BTCUSDT.csv", "N_FOLDS": 2, "VAL_FRACTION": 0.1, "CAL_FRACTION": 0.1,
          "MAX_SEQUENCE_COUNT": 4500, "FOLD_INDEX": -2, "BATCH_SIZE": 64, "TRAIN_METRICS_EVERY": 1}
WEEK = dict(SCREEN, MAX_SEQUENCE_COUNT=126000, BATCH_SIZE=256, TRAIN_METRICS_EVERY=10)
RULES = {"finite": True, "max_nonfinite_grad_steps": 0, "max_clipped_share": 1.0, "clip_skip_epochs": 1,
         "min_train_loss_drop": -10.0, "max_term_share": 1.0}


def write(name, over, slices, seeds, extra):
    o = dict(over); o.update(extra)
    spec = {"schema_version": 1, "name": name, "base_config": "../default.yaml", "overrides": o, "slices": slices,
            "seeds": seeds, "run": {"calibrate": False, "epochs": 8, "save_predictions": True}, "rules": RULES}
    yaml.safe_dump(spec, open(f"configs/tactical/{name}.yaml", "w"), sort_keys=False)


SKIP = {"DIRECTION_HEAD_MODE": "skip_only"}
write("cand_c1_skip", SCREEN, ["2022-04-19T07:00:00"], list(range(3, 13)), SKIP)
write("cand_c1_base", SCREEN, ["2022-04-19T07:00:00"], list(range(3, 13)), {})
write("cand_c2_skip", WEEK, C2_SLICES, [0, 1], SKIP)
write("cand_c2_base", WEEK, C2_SLICES, [0, 1], {})
print("C2 slices:", C2_SLICES)
