"""T09 Q9: Q1, Q3, Q4 (and the top-10% economics) on the Binance forward file, 2025-10..2026-09 (descriptive). The training span of every month
is the forward file only (2025-10-01 onward, up to 60 days), so the first scored month is 2025-11 (about 30 days of training).
usage: python run_forward.py -> results_forward.json"""
import json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import pandas as pd
import lib, analysis as A, guard
from run_minute import jsonable


def main():
    T0 = time.time(); D = lib.build_dataset("fwd"); print("dataset", len(D["t"]), flush=True)
    first_train = guard.FWD_FIRST; Wd = {}
    for e in lib.ENTRIES:
        Wd[e] = lib.walk(D, e, first="2025-11", last="2026-09", min_train=500, min_test=300, first_train=first_train, extras=e in ("close", "nopen"))
        print("walk", e, len(Wd[e]["months"]), flush=True)
    R = {"months": [r["month"] for r in Wd["close"]["months"]], "n_train_first": Wd["close"]["months"][0]["n_train"]}
    R["monthly"] = {e: [{k: r[k] for k in ("month", "auc", "n_train", "n_test", "up_rate")} for r in Wd[e]["months"]] for e in Wd}
    R["q1"] = {}
    for e in ("close", "nopen"):
        rows = Wd[e]["months"]; q = {"features": lib.FEAT_NAMES}
        for key in ("coef", "uni", "drop", "perm"):
            q[key] = {nm: lib.tint([r[key][j] for r in rows]) for j, nm in enumerate(lib.FEAT_NAMES)}
        for key in ("gperm", "gdrop", "gonly"):
            q[key] = {g: lib.tint([r[key][i] for r in rows]) for i, g in enumerate(lib.GROUPS)}
        q["auc"] = lib.tint([r["auc"] for r in rows]); R["q1"][e] = q
    IR = A.impulse(D, first_month="2025-10")
    R["q3"] = {}
    for e in lib.ENTRIES:
        X = IR[e]["auc"]; B = IR[e]["bps"]; from scipy.stats import t as Tt
        n = np.isfinite(X).sum(0); sd = np.nanstd(X, 0, ddof=1)
        R["q3"][e] = {"auc": np.nanmean(X, 0).tolist(), "auc_halfwidth": (Tt.ppf(0.975, np.maximum(n - 1, 1)) * sd / np.sqrt(np.maximum(n, 1))).tolist(),
                      "bps": np.nanmean(B, 0).tolist(), "months": IR[e]["months"]}
    R["q3"]["rule_share"] = {e: A.simple_rule_share(IR, Wd[e], e) for e in ("close", "nopen")}
    R["q4"] = {"auc": {e: lib.tint([r["auc"] for r in Wd[e]["months"]]) for e in lib.ENTRIES}}
    jt = lib.HS.index(lib.H0); rN = D["r_nopen"][:, jt]; yN = (rN > 0).astype(int); mN = np.isfinite(rN) & (np.abs(np.nan_to_num(rN)) > lib.DB0)
    t = pd.DatetimeIndex(D["t"].astype("datetime64[ns]")); mon = np.array([str(x) for x in t.to_period("M")]); pc = Wd["close"]["p"]
    R["q4"]["close_model_on_nopen_labels"] = lib.tint([lib.auc(yN[(mon == r["month"]) & mN & np.isfinite(pc)], pc[(mon == r["month"]) & mN & np.isfinite(pc)]) for r in Wd["close"]["months"]])
    MQ = A.mq_table(D); R["q4"]["market_quality"] = MQ
    cond = {e: A.conditional_table(D, Wd[e], e) for e in lib.ENTRIES}
    R["q4"]["by_spread_proxy"] = {e: A.cond_summary(cond[e][cond[e]["cond"].isin(["hl", "cs", "vol", "volume", "weekend", "move1sig"])], "all") for e in lib.ENTRIES}
    tr = A.top10_trades(D, Wd); R["q7_months"] = tr
    R["q7"] = {k: lib.tint([r[k] for r in tr if k in r]) for k in ("bps_close", "bps_nopen", "bps_mid", "bps_close_model_at_nopen", "hit_close", "hit_nopen", "cs_trades", "hl_trades", "cs_all", "hl_all")}
    R["corr"] = A.mq_correlations(Wd["close"], MQ)
    json.dump(jsonable(R), open(os.path.join(HERE, "results_forward.json"), "w"), indent=1)
    import pickle; pickle.dump({"R": R, "IR": IR}, open(os.path.join(HERE, "results_forward.pkl"), "wb"))
    print(f"done {time.time()-T0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
