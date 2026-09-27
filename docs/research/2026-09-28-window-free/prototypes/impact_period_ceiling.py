"""Read-only: do learned base periods of local runs press against the LOOKBACK ceiling (60)?"""
import glob
import json
import os

ROOT = "C:/Users/Step/Documents/neural_trade/runs"
for path in sorted(glob.glob(os.path.join(ROOT, "2026*", "metrics.jsonl"))):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    if not rows:
        continue
    run = os.path.basename(os.path.dirname(path))
    traj = [round(r.get("period/macd_1_slow", float("nan")), 1) for r in rows]
    last = {k.split("/", 1)[1]: v for k, v in rows[-1].items() if k.startswith("period/")}
    top = sorted(last.items(), key=lambda kv: -kv[1])[:2]
    print(f"{run}: {len(rows)} epochs; macd_1_slow by epoch {traj}; top final periods {[(k, round(v, 1)) for k, v in top]}")
