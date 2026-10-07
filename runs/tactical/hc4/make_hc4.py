"""Hill-climb on the 7-day block (owner, 2026-10-07: "now let's form the hill-climb"). SPEC.md in this folder.
usage: python make_hc4.py <variant> [KEY=VALUE ...]      -> configs/tactical/hc4_<variant>.yaml (climb slices)
       python make_hc4.py <variant> --final [KEY=VALUE]  -> configs/tactical/hc4_<variant>_final.yaml (held-out slices)
Special keys: CALIBRATE=true|false (run.calibrate), EPOCHS=n (run.epochs). Every other key is a Config override."""
import os, sys
import yaml
os.chdir(r"D:\nt\nt_tactical"); sys.path.insert(0, "runs/tactical")
import make_hc2 as m

allv = m.slices(); climb = [s for i, s in enumerate(allv) if i % 5 != 4]
CLIMB = climb[1::7][:6]      # the same 6 slices as the candidate check's C2 (its default runs are the round-0 base)
FINAL = climb[4::7][:6]      # held out: used once per winner, never for a choice inside the climb
WEEK = {"CSV_PATH": "D:/nt/neural_trade/Bitcoin_BTCUSDT.csv", "N_FOLDS": 2, "VAL_FRACTION": 0.1, "CAL_FRACTION": 0.1,
        "MAX_SEQUENCE_COUNT": 126000, "FOLD_INDEX": -2, "BATCH_SIZE": 256, "TRAIN_METRICS_EVERY": 10}
RULES = {"finite": True, "max_nonfinite_grad_steps": 0, "max_clipped_share": 1.0, "clip_skip_epochs": 1,
         "min_train_loss_drop": -10.0, "max_term_share": 1.0}


def make(variant, kv, final=False):
    over = dict(WEEK); calibrate, epochs = False, 8
    for k, v in kv.items():
        if k == "CALIBRATE":
            calibrate = yaml.safe_load(v)
        elif k == "EPOCHS":
            epochs = int(v)
        else:
            over[k] = yaml.safe_load(v)
    name = f"hc4_{variant}" + ("_final" if final else "")
    spec = {"schema_version": 1, "name": name, "base_config": "../default.yaml", "overrides": over,
            "slices": FINAL if final else CLIMB, "seeds": [0, 1],
            "run": {"calibrate": calibrate, "epochs": epochs, "save_predictions": True}, "rules": RULES}
    yaml.safe_dump(spec, open(f"configs/tactical/{name}.yaml", "w"), sort_keys=False)
    return name


if __name__ == "__main__":
    a = sys.argv[1:]
    print(make(a[0], dict(x.split("=", 1) for x in a[1:] if "=" in x), "--final" in a))
