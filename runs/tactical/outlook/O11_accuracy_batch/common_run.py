"""Shared runner helpers of O11 parts a-c (log, nice, resumable jsonl)."""
import json, os, time


def make(here):
    res = os.path.join(here, "results.jsonl"); logp = os.path.join(here, "run.log")

    def log(*a):
        msg = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a); print(msg, flush=True)
        with open(logp, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    def done_keys(keys):
        if not os.path.exists(res):
            return set()
        return {tuple(r[k] for k in keys) for r in map(json.loads, open(res, encoding="utf-8"))}

    def write(rows):
        with open(res, "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    def rows_df():
        import pandas as pd
        return pd.DataFrame([json.loads(x) for x in open(res, encoding="utf-8")])
    return log, done_keys, write, rows_df, res


def nice():
    from threadpoolctl import threadpool_limits; threadpool_limits(limits=4)
    try:
        import psutil; psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass


def free_gb():
    import psutil; return psutil.virtual_memory().available / 2 ** 30
