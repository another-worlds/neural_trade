"""Resumable monthly runner. usage: python run_shape.py mcA|mcB|sel|look [--workers 6]
sel: real 2019-01..2024-06; look: 2024-07..2025-07 (ONLY the cells in results/selected.json); mcA/mcB: 2021-01..2023-06 of the simulated paths."""
import os, sys, json, time
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
os.environ.setdefault("OMP_NUM_THREADS", "1")
import shapelib as S

RES = os.path.join(HERE, "results"); os.makedirs(RES, exist_ok=True)
PHASES = {"mcA": ("mcA", "2021-01", "2023-06"), "mcB": ("mcB", "2021-01", "2023-06"),
          "sel": ("real", "2019-01", "2024-06"), "look": ("real", "2024-07", "2025-07")}
_G = {}


def _init(tag, only):
    try:
        import psutil; psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass
    _G["ds"] = S.load_dataset(tag); _G["only"] = only


def _job(m):
    t0 = time.time(); r = S.score_month(_G["ds"], m, only=_G["only"])
    return m, r, time.time() - t0


if __name__ == "__main__":
    phase = sys.argv[1]; w = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 6
    tag, first, last = PHASES[phase]; only = None
    if phase == "look":
        sel = json.load(open(f"{RES}/selected.json"))["cells"]
        if not sel:
            sys.exit("look refused: nothing selected")
        only = {(c["target"], c["model"]) for c in sel}
        if os.path.exists(f"{RES}/look.jsonl"):
            sys.exit("look refused: the one look already ran")
    path = f"{RES}/{phase}.jsonl"; done = set()
    if os.path.exists(path):
        done = {json.loads(l)["month"] for l in open(path, encoding="utf-8")}
    months = [str(m) for m in pd.period_range(first, last, freq="M") if str(m) not in done]
    print(f"{phase}: {len(months)} months to run, {len(done)} done", flush=True); t0 = time.time()
    import multiprocessing as mp
    with mp.Pool(w, initializer=_init, initargs=(tag, only)) as pool, open(path, "a", encoding="utf-8") as fh:
        for m, r, sec in pool.imap_unordered(_job, months):
            if r is None:
                print(f"  {m}: skipped (too few rows)", flush=True); continue
            fh.write(json.dumps(r) + "\n"); fh.flush(); print(f"  {m} {sec:.0f}s", flush=True)
    print(f"{phase} done in {time.time() - t0:.0f}s", flush=True)
