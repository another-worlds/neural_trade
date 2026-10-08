"""Indicator experiments on tiny runs (owner, 2026-10-08: "experiment with different indicators on tiny slices and runs").
Fixed before any run. Layout: a 1-day training block (MAX_SEQUENCE_COUNT 18,000 -> 1,440 train windows), batch 256,
8 epochs, the static excerpt, the 6 tactical climb slices x seeds 0-1 (the owner's 6x2 screening design), all 9 heads
measured, predictions saved. One variant = one change against `all` (today's 14 families x 3 instances):
  only_<family>   the network sees one indicator family (3 instances) - which family carries signal on its own
  none            no learned indicator family at all (raw OHLCV only), if the build accepts it
  frozen          all 14 families, periods frozen at their textbook starting values: indicator LR and gradient
                  multipliers 1e-9 and ADAPTIVE_INDICATORS false (no per-window shift) - the untested core claim
                  "learned beats textbook" (NT-033), as far as a tiny run can tell
Caveat recorded with the result: on a 1-day block the confidence head does not learn (journal H11) and direction AUC
has a noise of about +-0.03 per variant; tiny runs screen for large effects only.
usage: python make_ind.py -> configs/tactical/ind1_<variant>.yaml and runs/tactical/ind/tasks_ind1.txt"""
import os, sys
import yaml
os.chdir(r"D:\nt\nt_tactical"); sys.path.insert(0, "runs/tactical/hc4"); sys.path.insert(0, "runs/tactical")
import make_hc4 as H

FAMS_CLOSE = {"ma": [5, 10, 30], "macd": [{"fast": 12, "slow": 26, "signal": 9}, {"fast": 5, "slow": 35, "signal": 5},
                                           {"fast": 8, "slow": 17, "signal": 9}],
              "rsi": [9, 14, 21], "bb": [10, 20, 25]}
DEFAULT_OHLCV = {"atr": [7, 14, 28], "stoch": [{"k_period": 14, "d_period": 3}, {"k_period": 9, "d_period": 3},
                                                {"k_period": 21, "d_period": 5}],
                 "willr": [7, 14, 28], "keltner": [{"period": 20, "atr_period": 10}, {"period": 10, "atr_period": 10},
                                                    {"period": 40, "atr_period": 20}],
                 "obv": [10, 20, 40], "vwap": [10, 20, 40], "mfi": [7, 14, 28], "adx": [7, 14, 28], "cci": [10, 20, 40],
                 "donchian": [10, 20, 55]}
BASE = {"CSV_PATH": "D:/nt/neural_trade/Bitcoin_BTCUSDT_tactical.csv", "N_FOLDS": 2, "VAL_FRACTION": 0.1, "CAL_FRACTION": 0.1,
        "MAX_SEQUENCE_COUNT": 18000, "FOLD_INDEX": -2, "BATCH_SIZE": 256, "TRAIN_METRICS_EVERY": 10}
RULES = {"finite": True, "max_nonfinite_grad_steps": 0, "max_clipped_share": 1.0, "clip_skip_epochs": 1,
         "min_train_loss_drop": -10.0, "max_term_share": 1.0}


def write(name, extra):
    o = dict(BASE); o.update(extra)
    spec = {"schema_version": 1, "name": f"ind1_{name}", "base_config": "../default.yaml", "overrides": o,
            "slices": H.CLIMB, "seeds": [0, 1], "run": {"calibrate": False, "epochs": 8, "save_predictions": True}, "rules": RULES}
    yaml.safe_dump(spec, open(f"configs/tactical/ind1_{name}.yaml", "w"), sort_keys=False)
    return f"ind1_{name}"


names = [write("all", {})]
empty_close = {k: [] for k in FAMS_CLOSE}
for fam in list(FAMS_CLOSE) + list(DEFAULT_OHLCV):
    fams = dict(empty_close)
    fams[fam] = FAMS_CLOSE.get(fam) or DEFAULT_OHLCV[fam]
    names.append(write(f"only_{fam}", {"INDICATOR_FAMILIES": fams}))
names.append(write("none", {"INDICATOR_FAMILIES": dict(empty_close)}))
names.append(write("frozen", {"INDICATOR_LR_MULT": 1e-9, "INDICATOR_GRAD_MULT": 1e-9, "ADAPTIVE_INDICATORS": False}))
with open("runs/tactical/ind/tasks_ind1.txt", "w") as f:
    for n in names:
        for i in range(2):
            f.write(f"{n} {i}/2\n")
print(len(names), "variants:", " ".join(names))
