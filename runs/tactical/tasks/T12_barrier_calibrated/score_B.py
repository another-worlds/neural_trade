"""T12 Look B scoring (SPEC.md), for a LATER session. usage: python score_B.py [--csv path]

Loads frozen_B.json (refuses on a sha256 mismatch), reads the Binance file from 2026-09-28 (feature history), finds the complete calendar months from
2026-10 (complete = the file holds the last 240-minute label of the month), scores H3 / H3c / B1 on them with NO refit (October's isotonic map is the stored
2026-09 map, later months fit on the previous month's frozen-H3 predictions; October is scored from 2026-10-02) and applies the Look A criteria.
The verdict is the first scoring with >= 3 complete months; with fewer, it says so and writes nothing. A later run writes resultsB_followup.json
instead of overwriting the verdict. Nothing here is run in the T12 task."""
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import t12_lib as T12  # noqa: E402

SCORE_FROM = pd.Timestamp("2026-09-28 00:00")


def load_frozen(path=os.path.join(HERE, "frozen_B.json")):
    art = json.load(open(path))
    if T12.sha_of(art) != art["sha256"]:
        raise ValueError("frozen_B.json does not match its sha256")
    return art


def load_bars(csv, first=SCORE_FROM):
    df = pd.read_csv(csv, usecols=T12.COLS); df.index = pd.to_datetime(df["timestamp"]); df = df.drop(columns="timestamp").sort_index()
    df = df[df.index >= first]; T12.guard.check_index(df.index, "forward file"); return df


def main():
    csv = sys.argv[sys.argv.index("--csv") + 1] if "--csv" in sys.argv else T12.guard.FWD_CSV
    art = load_frozen(); print("frozen sha256", art["sha256"]); bars = load_bars(csv); print("bars", len(bars), bars.index.min(), bars.index.max())
    res = T12.score(art, bars); print(res["status"])
    if res["verdict"] is None:
        return
    dst = os.path.join(HERE, "resultsB.json")
    if os.path.exists(dst):
        dst = os.path.join(HERE, "resultsB_followup.json")
    json.dump(res, open(dst, "w"), indent=1); v = res["verdict"]
    print("Look B", "PASS" if v["pass"] else "FAIL", {k: v[k] for k in ("skill", "skill_ci", "ece", "rel_brier_vs_H3", "c1_skill", "c2_ece", "c3_not_worse")}, "->", dst)


if __name__ == "__main__":
    main()
