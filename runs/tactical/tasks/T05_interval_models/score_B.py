"""T05 Look B scorer (SPEC.md): scores the chosen model, M0 and M1 from frozen_B.json on the months from 2026-10, once at least 3 COMPLETE
months exist, with the Look A criteria and 'better in at least 80% of months'. The fits are frozen; quantiles/corrections are recalibrated on
the previous month (the cal rule), so the first month uses exactly the stored 2026-09 calibration.
Usage: python score_B.py [--csv PATH] [--frozen frozen_B.json] [--out resultsB.json]"""
import argparse, json, os, sys
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import t05_lib as T  # noqa: E402
from run_lookA import verdict  # noqa: E402

FIRST_MONTH = pd.Period("2026-10", "M"); MIN_MONTHS = 3; NEED_FRAC = 0.8


def last_bar_time(path):
    with open(path, "rb") as f:
        f.seek(0, 2); n = f.tell(); f.seek(max(0, n - 400)); tail = f.read().decode().strip().splitlines()[-1]
    return pd.Timestamp(tail.split(",")[0])


def complete_months(last_bar, first=FIRST_MONTH):
    """months from `first` whose final minute (23:59 of the last day) is already in the file."""
    out = []; m = first
    while (m + 1).start_time - pd.Timedelta(minutes=1) <= last_bar:
        out.append(m); m += 1
    return out


def score(csv, frozen_path, sha_path=None, first=FIRST_MONTH, min_months=MIN_MONTHS):
    sha_path = sha_path or os.path.join(os.path.dirname(frozen_path), "frozen_B.sha256")
    want = open(sha_path).read().split()[0]; got = T.sha256_file(frozen_path)
    assert want == got, f"frozen file changed: {got} != {want}"
    fm = json.load(open(frozen_path)); names = fm["names"]; ch = fm.get("chosen")
    months = complete_months(last_bar_time(csv), first)
    if len(months) < min_months:
        return {"status": "waiting", "complete_months": [str(m) for m in months], "need": min_months}
    lo = (first - 1).start_time - pd.Timedelta(days=3); hi = (months[-1] + 1).start_time + pd.Timedelta(hours=2)
    bars = T.load(csv, (lo, hi), allow_late=True); B = T.make_data(bars)
    rows = []
    for mo in months:
        idx, n_all = T.L3.month_rows(B, T.H, mo); cidx = T.L3.cal_rows(B, T.H, mo)
        for nm in names:
            lo_, hi_ = T.frozen_bounds(nm, fm, B, idx, None if mo == first else cidx)    # month 1: the stored 2026-09 calibration
            d = T.score_bounds(B["r"][T.H][idx], lo_, hi_); d.update({"month": str(mo), "model": nm, "n_dropped": n_all - len(idx)}); rows.append(d)
    df = pd.DataFrame(rows); v = verdict(df, ch, NEED_FRAC) if ch else None
    return {"status": "scored", "months": [str(m) for m in months], "summary": T.summarise(df), "verdict": v, "chosen": ch, "rows": json.loads(df.to_json(orient="records"))}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--csv", default=T.FWD_CSV); ap.add_argument("--frozen", default=os.path.join(HERE, "frozen_B.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "resultsB.json")); a = ap.parse_args()
    res = score(a.csv, a.frozen)
    if res["status"] == "waiting":
        print("not enough complete months yet:", res); return
    json.dump(res, open(a.out, "w"), indent=1); print(json.dumps({k: res[k] for k in ("months", "summary", "verdict")}, indent=1))


if __name__ == "__main__":
    main()
