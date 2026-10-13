"""T05 Look B freeze (SPEC.md): fit the chosen model, M0 and M1 on the Binance file 2025-10-02..2026-09-30 (every anchor whose label ended by
2026-10-01 00:00), store the fits and the 2026-09 calibration in frozen_B.json with its sha256 (frozen_B.sha256). Reads no bar of October 2026.
Usage: python freeze_B.py [--csv PATH --out DIR]"""
import argparse, hashlib, json, os, sys
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import t05_lib as T  # noqa: E402


def build_frozen(bars, names, cal_month="2026-09", meta=None):
    B = T.make_data(bars); fr = T.freeze(B, names, cal_month)
    fr["bars"] = {"first": str(bars.index.min()), "last": str(bars.index.max()), "n": int(len(bars)),
                  "close_sha256": hashlib.sha256(np.ascontiguousarray(bars["close"].to_numpy(np.float64)).tobytes()).hexdigest()}
    fr["spec"] = "SPEC.md @7a02d55"; fr["names"] = list(names); fr.update(meta or {})
    return fr


def write_frozen(fr, outdir):
    p = os.path.join(outdir, "frozen_B.json"); json.dump(fr, open(p, "w"), indent=1, sort_keys=True)
    h = T.sha256_file(p); open(os.path.join(outdir, "frozen_B.sha256"), "w").write(h + "  frozen_B.json\n"); return h


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--csv", default=T.FWD_CSV); ap.add_argument("--out", default=HERE); a = ap.parse_args()
    sel = json.load(open(os.path.join(HERE, "selection.json"))); ch = sel["chosen"]; names = ["M0", "M1"] + ([ch] if ch else [])
    bars = T.load(a.csv, T.FREEZE_LOAD)
    assert bars.index.max() <= T.FREEZE_LOAD[1]
    fr = build_frozen(bars, names, "2026-09", {"chosen": ch})
    h = write_frozen(fr, a.out)
    print("frozen", names, "n_train", fr["n_train"], "n_cal", fr["n_cal"], "train", fr["train_first"], "->", fr["train_last_label_end"], "bars", fr["bars"])
    print("sha256", h)


if __name__ == "__main__":
    main()
