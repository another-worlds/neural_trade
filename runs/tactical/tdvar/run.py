"""tdvar runner (SPEC.md). usage: python run.py --stage sel|look [--part 0 --nparts 3] [--nn]
Resumable: one jsonl row per (month, W, H, model); a month's cells are skipped when the month is already in the done file."""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tdvar_lib as L  # noqa: E402

STAGES = {"sel": ("2019-01", "2024-06", "2024-07-01"), "look": ("2024-07", "2025-07", None)}


def lower_priority():
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="sel")
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--nparts", type=int, default=1)
    ap.add_argument("--nn", action="store_true")
    ap.add_argument("--months", default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    lower_priority()
    lo, hi, end = STAGES[a.stage]
    if a.months:
        lo, hi = a.months.split(":")
    months = list(pd.period_range(lo, hi, freq="M"))
    if a.nn and a.stage == "sel":
        months = [m for m in months if m >= pd.Period("2020-02") and (m.year * 12 + m.month) % 2 == 0]
    months = months[a.part::a.nparts]
    close, idx = L.load_bars()
    assert idx.max() < L.CUTOFF
    D = L.Data(close, idx)
    end_pos = int(idx.searchsorted(pd.Timestamp(end))) if end else len(idx)
    kind = "nn" if a.nn else "lin"
    rp = f"{HERE}/results_{kind}_{a.stage}{a.tag}_{a.part}.jsonl"
    mp = f"{HERE}/meta_{kind}_{a.stage}{a.tag}_{a.part}.jsonl"
    done = set(json.loads(l)["month"] for l in open(mp)) if os.path.exists(mp) else set()
    if a.nn:
        import tdvar_nn as N
    print(f"{kind} {a.stage} part {a.part}/{a.nparts}: {len(months)} months, anchors {len(D.anch)}, end_pos {end_pos}", flush=True)
    for m in months:
        if str(m) in done:
            continue
        t0w = time.time(); t0, t1 = L.month_bounds(m); rows = []; cells = {}
        for W in L.WINDOWS:
            for H in L.HORIZONS:
                t = time.time()
                out = (N.cell_nn if a.nn else L.cell)(D, t0, t1, W, H, end_pos)
                if out is None:
                    continue
                r, info = out
                for row in r:
                    rows.append(dict(month=str(m), W=W, H=H, trunc=bool(info["win_days_avail"] < 0.99 * W), **row))
                cells[f"{W}_{H}"] = dict(info, sec=round(time.time() - t, 2))
        with open(rp, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        with open(mp, "a") as f:
            f.write(json.dumps(dict(month=str(m), sec=round(time.time() - t0w, 1), cells=cells)) + "\n")
        print(m, f"{time.time() - t0w:.0f}s", f"{len(rows)} rows", flush=True)


if __name__ == "__main__":
    main()
