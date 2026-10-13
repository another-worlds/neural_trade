"""usage: python run_preds.py dev|look [--workers 6] [--only feat:model:T:H,...]   (SPEC.md)
dev : predictions of WARMUP + the 54 DEV months for the 54 prediction sets (the lockbox months stay NaN).
look: predictions of the 12 LOCKBOX months for the sets in winner.json (+ the baseline's), refuses without winner.json."""
import sys, os, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import hc2lib as H  # noqa: E402

_inp = None


def lowprio():
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass


def task(a):
    global _inp
    phase, key = a; lowprio(); t0 = time.time()
    if _inp is None:
        _inp = H.Inputs()
    months = (H.MO.WARMUP,) + H.MO.DEV if phase == "dev" else H.MO.LOCKBOX
    p = H.predict_set(_inp, key[0], key[1], key[2], key[3], months, look=(phase == "look"))
    np.save(H.pred_path(phase, key), p)
    return key, time.time() - t0, int(np.isfinite(p).sum())


def keys_all():
    return [(f, m, T, h) for f in dict(H.LEVERS)["feat"] for m in dict(H.LEVERS)["model"] for T in dict(H.LEVERS)["T"] for h in H.HS]


if __name__ == "__main__":
    import multiprocessing as mp
    phase = sys.argv[1]; w = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 6
    os.makedirs(H.CACHE, exist_ok=True)
    if phase == "look":
        win = json.load(open(f"{HERE}/winner.json"))                 # refuses (FileNotFoundError) without the committed winner
        keys = set()
        for cfg in (win["winner"], H.BASELINE):
            for h in H.HS:
                keys.add(H.pkey(cfg, h))
        keys = sorted(keys)
    else:
        keys = keys_all()
    if "--only" in sys.argv:
        only = set(sys.argv[sys.argv.index("--only") + 1].split(","))
        keys = [k for k in keys if ":".join(map(str, k)) in only]
    keys = [k for k in keys if not os.path.exists(H.pred_path(phase, k))]
    print(f"{phase}: {len(keys)} prediction sets to build", flush=True)
    t0 = time.time()
    with mp.Pool(w) as pool:
        for key, dt, n in pool.imap_unordered(task, [(phase, k) for k in keys]):
            print(f"  {key} {dt:.0f}s n={n} (total {time.time() - t0:.0f}s)", flush=True)
