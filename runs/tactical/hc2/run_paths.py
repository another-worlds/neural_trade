"""usage: python run_paths.py dev|look      minute-path tables (time exit, SL=TP 1.5 and 2 sigma) per horizon for the phase's months (SPEC.md).
look refuses without winner.json. Lockbox anchors stay NaN in the dev tables."""
import sys, os, json, time
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import hc2lib as H  # noqa: E402

_G = {}


def task(a):
    phase, h = a; t0 = time.time()
    try:
        import psutil; psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass
    inp = H.Inputs()
    rv = H.S.hourly_rv(end=H.DATA_END); times = rv.index.values.astype("datetime64[ns]")
    assert len(rv) == len(inp.op) and rv.index.max() < pd.Timestamp(H.DATA_END)
    st = H.S.state(rv.values, times, h, "2018-01", "2024-06")
    mins = H.L.load_minutes("2018-11-30", H.DATA_END)
    assert mins["t"].max() < pd.Timestamp(H.DATA_END).value
    months = H.MO.LOCKBOX if phase == "look" else H.MO.DEV
    p = H.build_paths(inp, mins, st["sigma"], h, months)
    H.save_paths(phase, h, p)
    n = int(np.isfinite(p["X0_time"]["gross"][:, 0]).sum())
    return h, n, time.time() - t0


if __name__ == "__main__":
    import multiprocessing as mp
    phase = sys.argv[1]
    if phase == "look":
        json.load(open(f"{HERE}/winner.json"))            # refuses without the committed winner
    os.makedirs(H.CACHE, exist_ok=True)
    with mp.Pool(3) as pool:
        for h, n, dt in pool.imap_unordered(task, [(phase, h) for h in H.HS]):
            print(f"{phase} H={h}: {n} anchors with paths, {dt:.0f}s", flush=True)
