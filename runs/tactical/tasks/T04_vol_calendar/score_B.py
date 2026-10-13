"""T04 Part B scoring (SPEC.md), for a LATER session. usage: python score_B.py [--csv path]

Loads frozen_B.json (refuses on a sha256 mismatch), reads the Binance file from 2026-09-28 (feature history), finds the complete calendar
months from 2026-10 (complete = the file reaches the first hour of the next month, for the 60-minute label), scores every frozen model on
them with NO refit and applies criterion B. The verdict is the first scoring with >= 3 complete months; with fewer, it says so and writes
nothing. A later run writes resultsB_followup.json instead of overwriting the verdict. Nothing here is run in the freeze task."""
import json, os, sys
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import t04_lib as L  # noqa: E402
T = L.T

SCORE_FROM = pd.Timestamp("2026-09-28 00:00"); FIRST_MONTH = pd.Period("2026-10", freq="M"); MIN_MONTHS = 3


def load_frozen(path=os.path.join(HERE, "frozen_B.json")):
    art = json.load(open(path))
    if L.sha_of(art) != art["sha256"]:
        raise ValueError("frozen_B.json does not match its sha256")
    return art


def complete_months(last_bar, first=FIRST_MONTH):
    out = []; m = first
    while (m + 1).start_time + pd.Timedelta(minutes=59) <= last_bar:
        out.append(m); m = m + 1
    return out


def score(art, bars, months=None):
    """Score the frozen models (no refit) on `months` (default: all complete months); returns rows, the verdict (None if < 3 months), the rest."""
    months = complete_months(bars.index.max()) if months is None else months
    out = {"n_months": len(months), "months": [str(m) for m in months]}
    if not months:
        return {**out, "rows": [], "verdict": None, "status": "no complete month"}
    B = L.build(bars, L.BASE); rows = [L.score_month(art["models"], B, m) for m in months]
    out["rows"] = rows; out["comparisons"] = {"S3C-S3": L.stats_for(rows, "S3C", "S3", "S3C-S3"), "S3-ar": L.stats_for(rows, "S3", "ar", "S3-ar"),
                                                "S3-harlvl": L.stats_for(rows, "S3", "harlvl", "S3-harlvl"), "C-ar": L.stats_for(rows, "C", "ar", "C-ar")} if len(rows) > 1 else {}
    out["verdict"] = L.verdict(rows) if len(rows) >= MIN_MONTHS else None
    out["status"] = "verdict" if len(rows) >= MIN_MONTHS else f"withheld: {len(rows)} complete months, need {MIN_MONTHS}"
    return out


def main():
    csv = sys.argv[sys.argv.index("--csv") + 1] if "--csv" in sys.argv else T.FWD_CSV
    art = load_frozen(); print("frozen sha256", art["sha256"])
    bars = T.load_bars(csv, SCORE_FROM, pd.Timestamp("2100-01-01")); print("bars", len(bars), bars.index.min(), bars.index.max())
    res = score(art, bars); res["frozen_sha256"] = art["sha256"]; print(res["status"])
    if res["verdict"] is None:
        return
    dst = os.path.join(HERE, "resultsB.json")
    if os.path.exists(dst):
        dst = os.path.join(HERE, "resultsB_followup.json")
    json.dump(res, open(dst, "w"), indent=1); print("criterion B:", "PASS" if res["verdict"]["pass"] else "FAIL", res["verdict"], "->", dst)
    for r in res["rows"]:
        print(r["month"], r["n_te"], " ".join(f"{k} {v:+.4f}" for k, v in r["skill_ar"].items()))


if __name__ == "__main__":
    main()
