"""T15 shared helpers: hard-stop guard, progress file, generalised harmonic views, monthly-series statistics. CPU only, below-normal priority.
Run with D:/nt/nt_mm_env/python.exe. Data only through the T14 harness (core/, anchors, view cache); this file never reads a CSV.
The T14 cache is a COPY (D:/nt/nt_t15_cache: anchors, base, bars, desc) so that no T14 file is touched; the T15 views are written there.
"""
import os

os.environ.setdefault("T14_CACHE", "D:/nt/nt_t15_cache")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import ctypes
import json
import shutil
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
TASKS = os.path.dirname(HERE)
T14 = os.path.join(TASKS, "T14_multimodal")
for p in (os.path.join(T14, "M9_module"), os.path.join(T14, "M8_module"), os.path.join(T14, "M7_gbm"), os.path.join(T14, "M6_faiss"), os.path.join(T14, "core")):
    if p not in sys.path:
        sys.path.insert(0, p)
import m9_lib as M9  # noqa: E402
import m7_lib as L7  # noqa: E402
from m6_lib import KEYS  # noqa: E402

CACHE = os.environ["T14_CACHE"]
START_FILE = os.path.join(HERE, "t15_start.txt")
MIN_RAM_GB, MIN_DISK_GB, CAP_HOURS = 8.0, 20.0, 72.0
STR_COLS = M9.STR_COLS


class StopRun(RuntimeError):
    pass


def below_normal():
    L7.below_normal()


def free_ram_gb():
    import psutil
    return psutil.virtual_memory().available / 2 ** 30


def free_disk_gb(path="D:/"):
    return shutil.disk_usage(path).free / 2 ** 30


def wall_hours():
    if not os.path.exists(START_FILE):
        open(START_FILE, "w").write(str(time.time()))
    return (time.time() - float(open(START_FILE).read())) / 3600.0


def checks():
    return {"free_ram_gb": round(free_ram_gb(), 1), "free_disk_D_gb": round(free_disk_gb(), 1), "wall_hours": round(wall_hours(), 3)}


def guard(job):
    """Hard stops (T15 SPEC): free RAM < 8 GB, free D: < 20 GB, wall clock > 72 h. Raises StopRun (the driver writes progress and ends)."""
    c = checks()
    if c["free_ram_gb"] < MIN_RAM_GB:
        raise StopRun(f"before {job}: free RAM {c['free_ram_gb']} GB < {MIN_RAM_GB}")
    if c["free_disk_D_gb"] < MIN_DISK_GB:
        raise StopRun(f"before {job}: free disk D: {c['free_disk_D_gb']} GB < {MIN_DISK_GB}")
    if c["wall_hours"] > CAP_HOURS:
        raise StopRun(f"before {job}: wall clock {c['wall_hours']} h > {CAP_HOURS}")
    return c


# ------------------------------------------------------------------ progress file (rewritten after every job)
PROG_JSON = os.path.join(HERE, "progress.json")
PROG_MD = os.path.join(HERE, "progress.md")


def _load_prog():
    return json.load(open(PROG_JSON)) if os.path.exists(PROG_JSON) else {"jobs": [], "stop": None, "notes": []}


def _write_prog(P):
    json.dump(P, open(PROG_JSON, "w"), indent=1)
    L = ["# T15 progress (rewritten after every job)", "",
         f"wall clock since start: {wall_hours():.2f} h of {CAP_HOURS:.0f} h; CPU only, below-normal priority, threads <= 6 (BLAS 4, GBM 4); GPU not used.",
         f"stop condition: {P['stop'] or 'none hit'}", "",
         "| # | day | job | result file | wall s | cpu s | free RAM GB | free D: GB | status |", "|---|---|---|---|---|---|---|---|---|"]
    for i, j in enumerate(P["jobs"], 1):
        L.append(f"| {i} | {j['day']} | {j['job']} | {j['result']} | {j['wall_s']} | {j['cpu_s']} | {j['free_ram_gb']} | {j['free_disk_D_gb']} | {j['status']} |")
    tot_w = sum(j["wall_s"] for j in P["jobs"]); tot_c = sum(j["cpu_s"] for j in P["jobs"])
    L += ["", f"total job wall {tot_w:.0f} s, total job CPU (process time, all threads) {tot_c:.0f} s"]
    for n in P["notes"]:
        L.append(f"- {n}")
    open(PROG_MD, "w", encoding="utf-8").write("\n".join(L) + "\n")


def note(text):
    P = _load_prog(); P["notes"].append(text); _write_prog(P)


def run_job(day, job, fn, result=""):
    """guard, run fn(), record wall / CPU / checks in progress.md. A failing job is recorded and re-raised (the driver decides)."""
    P = _load_prog()
    try:
        c = guard(job)
    except StopRun as e:
        P["stop"] = str(e); _write_prog(P); raise
    t0, c0 = time.time(), time.process_time(); status = "ok"; out = None
    try:
        out = fn()
    except Exception as e:  # noqa: BLE001
        status = f"FAILED: {type(e).__name__}: {e}"
        raise
    finally:
        P = _load_prog()
        P["jobs"].append({"day": day, "job": job, "result": result, "wall_s": round(time.time() - t0, 1), "cpu_s": round(time.process_time() - c0, 1),
                          "free_ram_gb": c["free_ram_gb"], "free_disk_D_gb": c["free_disk_D_gb"], "status": status})
        _write_prog(P)
    return out


# ------------------------------------------------------------------ views
def harm_names(k):
    """column names of the harmonic profile with K harmonics: sin1..sinK, cos1..cosK, wkend, wk_sin.., wk_cos..; for K = 12 sin12 and wk_sin12 are
    identically 0 at integer hours and are omitted (23 columns of 12 + 11 = the span of the 23 hour dummies)."""
    s = [j for j in range(1, k + 1) if not (2 * j == 24)]
    return [f"sin{j}" for j in s] + [f"cos{j}" for j in range(1, k + 1)] + ["wkend"] + [f"wk_sin{j}" for j in s] + [f"wk_cos{j}" for j in range(1, k + 1)]


def harm_prof_str(k, hour, wk, keys):
    """[N, c] view matrix: harmonics of the UTC hour (`hour`, integer array), the weekend flag `wk` (0/1 float), weekend x harmonics, and the 4 stretch
    terms of the M6 keys (abs / square of str20, str60). Row-wise. NaN where a stretch key is NaN."""
    hour = np.asarray(hour, int); wk = np.asarray(wk, float); keys = np.asarray(keys, float)
    ix = {c: i for i, c in enumerate(KEYS)}
    s20, s60 = keys[:, ix["str20"]], keys[:, ix["str60"]]
    cols = {}
    for j in range(1, k + 1):
        a = 2 * np.pi * j * hour / 24
        cols[f"sin{j}"] = np.sin(a); cols[f"cos{j}"] = np.cos(a); cols[f"wk_sin{j}"] = wk * np.sin(a); cols[f"wk_cos{j}"] = wk * np.cos(a)
    cols["wkend"] = wk
    cols.update(abs_str20=np.abs(s20), abs_str60=np.abs(s60), sq_str20=s20 ** 2, sq_str60=s60 ** 2)
    names = harm_names(k) + STR_COLS
    return np.column_stack([cols[c] for c in names]).astype(float), names


def hour_wk(ta):
    ta = pd.DatetimeIndex(ta)
    return np.asarray(ta.hour), (np.asarray(ta.dayofweek) >= 5).astype(float)


def shuffle_within_month(ta, mask, seed):
    """Index array p with p[i] = a row of the SAME UTC calendar month (among `mask` rows) whose value row i takes; rows outside mask map to themselves.
    A permutation of the mask rows inside each month."""
    ta = pd.DatetimeIndex(ta); per = np.asarray(ta.year * 12 + ta.month); p = np.arange(len(ta)); rng = np.random.default_rng(seed)
    for m in np.unique(per[mask]):
        idx = np.where(mask & (per == m))[0]
        p[idx] = rng.permutation(idx)
    return p


# ------------------------------------------------------------------ statistics on monthly series
def tci(x):
    x = np.asarray(x, float); n = len(x)
    if n < 2:
        return {"mean": float(x.mean()) if n else None, "lo": None, "hi": None, "n": int(n)}
    m = x.mean(); se = x.std(ddof=1) / np.sqrt(n); t = stats.t.ppf(0.975, n - 1)
    return {"mean": float(m), "lo": float(m - t * se), "hi": float(m + t * se), "n": int(n)}


def series(res, tg, look=None):
    """{look: (months, array)} of the monthly gain series of a harness results dict (A: increment_series.view; B: monthly_rel.view)."""
    out = {}
    for lk, r in res["looks"].items():
        if look and lk != look:
            continue
        a = r[tg]
        out[lk] = (list(a["months"]), np.array(a["increment_series"]["view"] if tg == "A" else a["monthly_rel"]["view"], float))
    return out


def pooled(res, tg):
    s = series(res, tg); mo, v = [], []
    for lk in ("look1", "look2"):
        if lk in s:
            mo += s[lk][0]; v += list(s[lk][1])
    return mo, np.array(v)


def load_json(p):
    return json.load(open(p))


def stored(name):
    """stored T14 harness result."""
    return load_json(os.path.join(T14, name, "results.json"))


def dump(obj, path):
    json.dump(obj, open(path, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
