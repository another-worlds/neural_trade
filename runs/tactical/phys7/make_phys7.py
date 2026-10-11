"""Writes configs/tactical/phys7_<head>_<phys>.yaml (SPEC.md). Run from the repo root: python runs/tactical/phys7/make_phys7.py"""
import yaml

CLIMB = ['2017-05-02T12:00:00', '2018-11-16T02:00:00', '2020-03-30T04:00:00', '2021-10-13T18:00:00',
         '2023-04-29T08:00:00', '2024-11-11T23:00:00']   # = hc4 CLIMB (make_hc4.py), all before the 2025-07-20 cutoff
WEEK = {"CSV_PATH": "D:/nt/neural_trade/Bitcoin_BTCUSDT_tactical.csv", "N_FOLDS": 2, "VAL_FRACTION": 0.1, "CAL_FRACTION": 0.1,
        "MAX_SEQUENCE_COUNT": 126000, "FOLD_INDEX": -2, "BATCH_SIZE": 256, "TRAIN_METRICS_EVERY": 10, "DATA_TAIL_TRIM": True}
RULES = {"finite": True, "max_nonfinite_grad_steps": 0, "max_clipped_share": 1.0, "clip_skip_epochs": 1,
         "min_train_loss_drop": -10.0, "max_term_share": 1.0}
PHYS = ["LAMBDA_T_PERP", "LAMBDA_CASIMIR", "LAMBDA_HD", "LAMBDA_IFE", "LAMBDA_VAC", "LAMBDA_VAC_OVERFLOW"]

for head in ("none", "price"):
    for phys in ("on", "off"):
        over = dict(WEEK)
        if head == "none":
            over["PRICE_HEAD"] = "none"
        if phys == "off":
            over.update({k: 0 for k in PHYS})
        name = f"phys7_{head}_{phys}"
        spec = {"schema_version": 1, "name": name, "base_config": "../default.yaml", "overrides": over, "slices": CLIMB,
                "seeds": [0], "run": {"calibrate": False, "epochs": 4, "save_predictions": True}, "rules": RULES}
        yaml.safe_dump(spec, open(f"configs/tactical/{name}.yaml", "w"), sort_keys=False)
        print(name)
