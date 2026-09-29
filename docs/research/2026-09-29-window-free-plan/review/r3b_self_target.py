"""Review check (addendum to r3): share of runs that never reach the target built from ALL seeds of
their own condition (itself included), as arm A256's own seeds are judged in spec (0).
Run: PYTHONIOENCODING=utf-8 C:/Users/Step/miniforge3/envs/nt/python r3b_self_target.py"""
import json
import numpy as np
import pandas as pd
from r3_common_target import GRID, curve, smooth
df = pd.read_csv(GRID / "results.csv"); df = df[df.period == "P1"]
recs = [(r["condition"], smooth(curve(GRID / "runs" / r["run_id"] / "metrics.jsonl"))) for _, r in df.iterrows()]
out = {}
for frac in (0.05, 0.10):
    never = 0
    for cond in {c for c, _ in recs}:
        g = [J for c, J in recs if c == cond]
        R = np.mean([J.min() for J in g]) + frac * np.mean([J[0] - J.min() for J in g])
        never += sum(int(not np.any(J <= R)) for J in g)
    out[f"frac{frac}"] = {"runs": len(recs), "never_reach_self_included_target": never}
print(json.dumps(out))
open("r3b_self_target.json", "w").write(json.dumps(out, indent=1))
