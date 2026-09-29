"""Q2: arm A's edge vs const_var on DEV fold -2 with the served (best-val checkpoint) weights of the 12 runs
re-predicted by q2_block_scaling.py (D-011-like serving), per horizon and for the horizon mean; and the edge
per 1-day and 2-day sub-block (how fold/block-dependent the edge is). const_var CRPS per sub-block is
recomputed from each run's eval report baseline sigma is not stored, so the whole-block const_var CRPS from
eval_report_test.json is used for the whole block only. Writes q2_dev_edge_checkpoint.json."""
import json, math
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
GRID = Path("D:/neural_trade/runs/ablations/ablate_physics_v1-full")
df = pd.read_csv(GRID / "results.csv")
sel = df[(df.period == "P1") & (df.condition.isin(["all_on", "all_off", "only:LAMBDA_T_PERP", "without:LAMBDA_VAC"]))]
rows = []
for _, r in sel.iterrows():
    z = np.load(HERE / "q2_block_scaling_preds" / f"{r['run_id']}.npz")
    ev = json.loads((GRID / "runs" / r["run_id"] / "eval_report_test.json").read_text(encoding="utf-8"))
    row = {"condition": r["condition"], "seed": int(r["seed"])}
    for h in ("h0", "h1", "h2"):
        cv = ev["baselines"]["const_var"]["horizons"][h]["variance"]["crps"]
        row[f"edge_{h}"] = math.log(cv) - math.log(float(z[f"crps_{h}"].mean()))
    row["edge_mean"] = float(np.mean([row[f"edge_{h}"] for h in ("h0", "h1", "h2")]))
    rows.append(row)
t = pd.DataFrame(rows)
out = {"runs": len(t), "mean": {c: float(t[c].mean()) for c in t.columns if c.startswith("edge")},
       "sd_over_runs": {c: float(t[c].std(ddof=1)) for c in t.columns if c.startswith("edge")},
       "by_condition": t.groupby("condition")[[c for c in t.columns if c.startswith("edge")]].mean().to_dict()}
(HERE / "q2_dev_edge_checkpoint.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=1))
