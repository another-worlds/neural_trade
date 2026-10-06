"""Baseline v2 design (owner: 'formulate the baseline, start active experiments'), fixed BEFORE any v2 run.

Metric: mean over h0-h2 of validation direction AUC (screen layout: 4500 windows, 360 train / 450 val).
Slices: 50 DATA_END dates evenly spaced over 2017-03-01 .. 2025-07-20 (inside the protected span rule: the
newest 64 days of the file are never used). Every 5th (index % 5 == 4) is a FINAL slice (10), kept out of
every climb round and used once for the winner; the other 40 are CLIMB slices, split into 5 chunk specs
(climb[c::5], 8 slices each, so a partial run still spans all regimes and each process caches <= 8 slices).
Seeds: 3 per slice. A round = a variant over the 40 climb slices = 120 trials. Unit of inference = slice.
Candidate rule (fixed now): ADOPT-for-confirmation only if the 95% t-interval over the 40 per-slice mean
differences is entirely above 0 AND the mean diff >= +0.01; it is then checked once on the 10 FINAL slices.

usage: python make_hc2.py <variant> [KEY=VALUE ...]   -> configs/tactical/hc2_<variant>_c{0..4}.yaml
       python make_hc2.py <variant> --final [KEY=VALUE ...] -> hc2_<variant>_final.yaml (the 10 final slices)
"""
import datetime as dt
import sys

import yaml

START, END, N = dt.datetime(2017, 3, 1), dt.datetime(2025, 7, 20), 50


def slices():
    span = (END - START).total_seconds()
    return [(START + dt.timedelta(seconds=span * i / (N - 1))).replace(minute=0, second=0, microsecond=0)
            .strftime("%Y-%m-%dT%H:%M:%S") for i in range(N)]


def main():
    args = sys.argv[1:]
    variant, final = args[0], "--final" in args
    kv = dict(a.split("=", 1) for a in args[1:] if "=" in a)
    allv = slices()
    fin = [s for i, s in enumerate(allv) if i % 5 == 4]
    climb = [s for i, s in enumerate(allv) if i % 5 != 4]
    over = {"CSV_PATH": "D:/nt/neural_trade/Bitcoin_BTCUSDT.csv", "N_FOLDS": 2, "VAL_FRACTION": 0.1, "CAL_FRACTION": 0.1,
            "MAX_SEQUENCE_COUNT": 4500, "FOLD_INDEX": -2, "BATCH_SIZE": 64, "TRAIN_METRICS_EVERY": 1}
    for k, v in kv.items():
        over[k] = yaml.safe_load(v)
    groups = {"final": fin} if final else {f"c{c}": climb[c::5] for c in range(5)}
    for g, sl in groups.items():
        spec = {"schema_version": 1, "name": f"hc2_{variant}_{g}", "base_config": "../default.yaml", "overrides": over,
                "slices": sl, "seeds": [0, 1, 2], "run": {"calibrate": False, "epochs": 8},
                "rules": {"finite": True, "max_nonfinite_grad_steps": 0, "max_clipped_share": 1.0, "clip_skip_epochs": 1,
                          "min_train_loss_drop": -10.0, "max_term_share": 1.0}}
        path = f"configs/tactical/hc2_{variant}_{g}.yaml"
        yaml.safe_dump(spec, open(path, "w"), sort_keys=False)
        print(path, len(sl), "slices")


if __name__ == "__main__":
    main()
