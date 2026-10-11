"""usage: python run_grid.py cache | run [--workers 4] | table"""
import json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import fwdlib as F  # noqa

SETS = ("tb7", "rich"); MODELS = ("logreg", "l1"); TS = (365, 730); HS = (1, 2, 4); FRACS = (0.10, 0.05)
OUT = f"{HERE}/grid.jsonl"


def task(a):
    name, kind, T, H = a; t0 = time.time()
    try:
        import psutil; psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass
    X = np.load(f"{F.CACHE}/{name}.npy", mmap_mode="r"); t = np.load(f"{F.CACHE}/t.npy"); anch = np.load(f"{F.CACHE}/anch.npy")
    op = np.load(f"{F.CACHE}/open.npy"); cl = np.load(f"{F.CACHE}/close.npy")
    r = F.next_open_ret(op, cl, anch, H)
    preds = list(F.month_predictions(np.asarray(X), t, r, kind, T, H))
    res = []
    for frac in F.__dict__.get("FRACS_", FRACS):
        ent, ex, side, ret, mon = F.trades_from(preds, t, H, frac)
        sc = F.score_candidate(ent, ex, ret, mon, int(F.pd.Timestamp("2019-01-01").value))
        res.append({"set": name, "model": kind, "T": T, "H": H, "top": frac, **sc})
    return {"cfg": [name, kind, T, H], "seconds": round(time.time() - t0, 1), "n_months_fit": len(preds), "rows": res}


if __name__ == "__main__":
    if sys.argv[1] == "cache":
        print("anchors", F.build_cache())
    elif sys.argv[1] == "run":
        import multiprocessing as mp
        w = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 4
        done = set()
        if os.path.exists(OUT):
            done = {tuple(json.loads(l)["cfg"]) for l in open(OUT, encoding="utf-8")}
        todo = [(s, m, T, H) for s in SETS for m in MODELS for T in TS for H in HS if (s, m, T, H) not in done]
        todo.sort(key=lambda x: x[1] != "l1")
        with mp.Pool(w) as pool, open(OUT, "a", encoding="utf-8") as fh:
            for d in pool.imap_unordered(task, todo):
                fh.write(json.dumps(d) + "\n"); fh.flush(); print(d["cfg"], d["n_months_fit"], "months", d["seconds"], "s", flush=True)
