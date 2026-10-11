"""Decay study driver (nt-tactical-decay). usage: python decay_run.py build|q1|q2|q3
Writes runs/tactical/decay/{q1,q2,q3}.json (monthly rows per cell). 6 worker processes, below-normal priority, 4 threads each."""
import os, sys, json, time
os.environ.setdefault("OMP_NUM_THREADS", "4")
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import decay_lib as D

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "decay")
HS = (1, 3, 10, 15, 30)


def _low():
    try:
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    except Exception:
        pass


def job(a):
    t0 = time.time(); kind, args = a
    if kind == "build":
        return a, D.build_cache(args), time.time() - t0
    bar, H, train_days, model, k = args
    rows = D.walk_forward(bar, H, train_days=train_days, kind=model, k=k)
    return a, rows, time.time() - t0


def cells(q):
    if q == "build":
        return [("build", b) for b in ("1m", "5m", "15m", "1h")]
    if q == "q1":
        return [("w", (b, H, 60, "logreg", 0.0)) for b in D.BAR_MIN for H in HS if H * D.BAR_MIN[b] <= 1440]
    if q == "q2":
        return [("w", ("1m", 15, d, m, 0.0)) for m in ("logreg", "hgb") for d in (60, 365, 1095)]
    if q == "q3":
        return [("w", ("1m", 15, 60, "logreg", k)) for k in (0.5, 1.0)] + [("w", ("5m", 3, 60, "logreg", k)) for k in (0.5, 1.0)]


if __name__ == "__main__":
    q = sys.argv[1]; os.makedirs(OUT, exist_ok=True); t0 = time.time(); res = {}
    with Pool(6, initializer=_low) as pool:
        for a, rows, dt in pool.imap_unordered(job, cells(q)):
            key = "|".join(map(str, a[1])) if a[0] == "w" else a[1]
            print(f"{q} {key}: {dt:.0f}s", flush=True)
            res[key] = {"time_s": round(dt, 1), "rows": rows} if a[0] == "w" else {"anchors": rows, "time_s": round(dt, 1)}
    if q != "build":
        json.dump(res, open(os.path.join(OUT, f"{q}.json"), "w"), indent=1)
    print(f"{q} total {time.time() - t0:.0f}s")
