"""tdconfirm runner (SPEC.md). usage: python run_confirm.py --period conf|repl [--part i --nparts n]
Resumable: one jsonl row per (month, W, H, model); a month already in the meta file is skipped."""
import argparse
import json
import os
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import confirm_lib as C  # noqa: E402
import tdvar_lib as L  # noqa: E402

PERIODS = {"conf": ("2017-07", "2018-12", "2019-01-01"), "repl": ("2020-02", "2024-06", "2024-07-01")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", default="conf")
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--nparts", type=int, default=1)
    a = ap.parse_args()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass
    lo, hi, end = PERIODS[a.period]
    months = list(pd.period_range(lo, hi, freq="M"))[a.part::a.nparts]
    reg = C.load_reg()
    close, idx = L.load_bars()
    assert idx.max() < L.CUTOFF
    D = L.Data(close, idx)
    end_pos = int(idx.searchsorted(pd.Timestamp(end)))
    rp, mp = f"{HERE}/results_{a.period}_{a.part}.jsonl", f"{HERE}/meta_{a.period}_{a.part}.jsonl"
    done = set(json.loads(line)["month"] for line in open(mp)) if os.path.exists(mp) else set()
    print(f"{a.period} part {a.part}/{a.nparts}: {len(months)} months, reg {reg}, end_pos {end_pos}", flush=True)
    for m in months:
        if str(m) in done:
            continue
        t_m = time.time()
        t0, t1 = L.month_bounds(m)
        rows, cells, skipped = [], {}, []
        for W in C.WINDOWS:
            for H in C.HORIZONS:
                t = time.time()
                out = C.cell_confirm(D, t0, t1, W, H, end_pos, reg)
                if out is None:
                    skipped.append(f"{W}_{H}")
                    continue
                r, info = out
                for row in r:
                    rows.append(dict(month=str(m), W=W, H=H, trunc=bool(info["win_days_avail"] < 0.99 * W), **row))
                cells[f"{W}_{H}"] = dict(info, sec=round(time.time() - t, 2))
        with open(rp, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        with open(mp, "a") as f:
            f.write(json.dumps(dict(month=str(m), sec=round(time.time() - t_m, 1), skipped=skipped, cells=cells)) + "\n")
        print(m, f"{time.time() - t_m:.0f}s", f"{len(rows)} rows", "skipped", skipped, flush=True)


if __name__ == "__main__":
    main()
