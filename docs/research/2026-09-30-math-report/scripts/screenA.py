import json, glob
import numpy as np
import pandas as pd

def load(name):
    rows = []
    for f in glob.glob(f"D:/neural_trade/runs/screens/{name}/results*.jsonl"):
        for line in open(f):
            r = json.loads(line)
            d = dict(r["config_diff"]); d["passed"] = r["passed"]; d["seed"] = r["seed"]; d["slice"] = r["data_end"][:4]
            h = r["health"]; d["clipped"] = h.get("clipped_share"); d["gmean"] = h.get("grad_global_norm_mean")
            d["gmax"] = h.get("grad_global_norm_max"); d["finite"] = h.get("finite"); d["nonfin"] = h.get("nonfinite_grad_steps")
            d["drop"] = h.get("train_loss_drop"); d["reasons"] = ";".join(x.split(" ")[0] for x in r["reasons"])
            d["auc0"] = r["direction_auc"]["h0"]["auc"] if r.get("direction_auc") else np.nan
            rows.append(d)
    return pd.DataFrame(rows)

pd.set_option("display.width", 220)
A = load("l1_A_hyper")
print("A trials", len(A), "passed", A.passed.sum())
print(A.reasons.value_counts())
print(A.groupby("GRAD_CLIP_NORM").agg(n=("passed", "size"), pass_rate=("passed", "mean"), clipped=("clipped", "mean"), gmean=("gmean", "median"), nonfin=("nonfin", "sum")))
lhs = A[A.LR.notna()] if "LR" in A else A
for col, bins in [("LR", [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]), ("INDICATOR_LR_MULT", [1, 2, 5, 10, 20]),
                  ("ADAM_BETA1", [0.8, 0.85, 0.9, 0.95]), ("ADAM_BETA2", [0.99, 0.995, 0.999, 0.9999])]:
    if col in A:
        g = A[A[col].notna()].groupby(pd.cut(A[col], bins, include_lowest=True))
        print(g.agg(n=("passed", "size"), pass_rate=("passed", "mean"), clipped=("clipped", "mean"), gmean=("gmean", "median"), nonfin=("nonfin", "sum"), drop=("drop", "median")))
print(A.groupby("slice").agg(n=("passed", "size"), pass_rate=("passed", "mean"), gmean=("gmean", "median")))
# default-like: clip 20 grid rows
print(A[A.GRAD_CLIP_NORM.notna()][["GRAD_CLIP_NORM", "slice", "seed", "clipped", "gmean", "gmax", "passed"]].sort_values(["GRAD_CLIP_NORM", "slice"]).to_string())
# logistic-ish: correlation of clipped_share with log LR, log mult within LHS
L = A[A.LR.notna()].copy()
L["lLR"] = np.log10(L.LR); L["lM"] = np.log10(L.INDICATOR_LR_MULT)
print(L[["lLR", "lM", "ADAM_BETA1", "ADAM_BETA2", "clipped", "gmean", "drop", "auc0"]].corr(method="spearman").round(2))
print(L.sort_values("LR")[["LR", "INDICATOR_LR_MULT", "ADAM_BETA1", "ADAM_BETA2", "slice", "seed", "clipped", "gmean", "nonfin", "drop", "passed", "reasons"]].to_string())
for n in ["l1_B_loss_weights", "l1_C_physics", "l1_D_loss_choice", "l1_E_maths"]:
    try:
        X = load(n); print(n, len(X), "passed", X.passed.sum(), X.reasons.value_counts().to_dict(), "median gmean", X.gmean.median())
    except Exception as e:
        print(n, e)
print(A.groupby("seed").agg(n=("passed","size"), pass_rate=("passed","mean"), gmean=("gmean","median"), clipped=("clipped","mean")))
H = A[A.LR.notna() & (A.LR > 3e-3)]
print(H[["LR","INDICATOR_LR_MULT","slice","seed","clipped","gmean","drop","auc0","passed"]].sort_values("LR").to_string())
