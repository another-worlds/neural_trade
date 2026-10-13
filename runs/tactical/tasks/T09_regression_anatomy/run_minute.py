"""T09 main run on the cut file (2017-01..2025-07-20): Q1-Q7. usage: python run_minute.py -> results_minute.pkl (arrays for the figures)
and results_minute.json."""
import json, os, pickle, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np
import pandas as pd
import lib, analysis as A


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, pd.DataFrame):
        return jsonable(o.to_dict("records"))
    return o


def main():
    T0 = time.time(); log = lambda *a: print(f"[{time.time()-T0:5.0f}s]", *a, flush=True)
    D = lib.build_dataset("cut"); log("dataset", len(D["t"]))
    Wd = {}
    for e in lib.ENTRIES:
        Wd[e] = lib.walk(D, e, extras=e in ("close", "nopen")); log("walk", e, len(Wd[e]["months"]), "months")
    R = {"n_anchors": int(len(D["t"])), "months": [r["month"] for r in Wd["close"]["months"]]}
    R["q1"] = {e: A.q1_tables(Wd[e]) for e in ("close", "nopen")}
    drop_mean = {nm: np.mean([r["drop"][j] for r in Wd["close"]["months"]]) for j, nm in enumerate(lib.FEAT_NAMES)}
    rank = sorted(drop_mean, key=lambda k: -drop_mean[k]); R["q1"]["close"]["rank_all_months"] = rank; R["q1"]["close"]["drop_mean_all"] = drop_mean
    log("rank", rank[:5])
    R["q1_subsets"] = {e: A.subset_models(D, e, rank) for e in ("close",)}; log("subsets")
    pd_feats = list(dict.fromkeys(rank[:3] + ["ema20", "ret1"]))        # the top-3 by drop-one loss, plus ema20 (top by permutation in 2022-25) and ret1 (the one-minute move)
    R["q2"] = {e: A.q2_partial_dependence(D, Wd[e], e, pd_feats) for e in ("close", "nopen")}; log("q2")
    IR = A.impulse(D); log("impulse")
    IR_all = A.impulse(D, exclude_seam=False)
    R["seam"] = {"n_seam_anchors": int(D["seam"].sum()), "share": float(D["seam"].mean()),
                 "impulse_auc_max_abs_diff_incl_vs_excl": {e: float(np.nanmax(np.abs(IR[e]["auc"] - IR_all[e]["auc"]))) for e in lib.ENTRIES},
                 "impulse_auc_mean_abs_diff": {e: float(np.nanmean(np.abs(IR[e]["auc"] - IR_all[e]["auc"]))) for e in lib.ENTRIES}}
    # robustness: the yearly regression AUC with the seam anchors left out of the test set
    rob = {}
    for e in ("close", "nopen"):
        r_ = D["r_" + e][:, lib.HS.index(lib.H0)]; yy = (r_ > 0).astype(int); mk = np.isfinite(r_) & (np.abs(np.nan_to_num(r_)) > lib.DB0)
        tt = pd.DatetimeIndex(D["t"].astype("datetime64[ns]")); mo = np.array([str(x) for x in tt.to_period("M")]); rows_ = []
        for rec in Wd[e]["months"]:
            ii = np.flatnonzero((mo == rec["month"]) & mk & ~D["seam"] & np.isfinite(Wd[e]["p"]))
            rows_.append({"month": rec["month"], "auc": lib.auc(yy[ii], Wd[e]["p"][ii])})
        rob[e] = A.agg(rows_, "auc")
    R["seam"]["yearly_auc_without_seam_anchors"] = rob
    R["q3"] = {e: {"by_year_auc": {y: v.tolist() for y, v in A.impulse_by_year(IR, e, "auc").items()},
                   "by_year_bps": {y: v.tolist() for y, v in A.impulse_by_year(IR, e, "bps").items()},
                   "rule_share": A.simple_rule_share(IR, Wd[e], e)} for e in ("close", "nopen")}
    # monthly-CI of the impulse cells per year (close): t-interval over months for the AUC cells
    ci = {}
    for e in lib.ENTRIES:
        ms = IR[e]["months"]
        for y in sorted({m[:4] for m in ms}):
            ii = [i for i, m in enumerate(ms) if m[:4] == y]; X = IR[e]["auc"][ii]
            n = np.isfinite(X).sum(0); sd = np.nanstd(X, 0, ddof=1); from scipy.stats import t as Tt
            ci.setdefault(e, {})[y] = (Tt.ppf(0.975, np.maximum(n - 1, 1)) * sd / np.sqrt(np.maximum(n, 1))).tolist()
    R["q3"]["auc_halfwidth"] = ci
    cond = {e: A.conditional_table(D, Wd[e], e) for e in lib.ENTRIES}; log("conditions")
    R["q5_period"] = {e: A.cond_summary(cond[e], "period") for e in lib.ENTRIES}
    R["q5_year"] = {e: A.cond_summary(cond[e], "year") for e in lib.ENTRIES}
    # Q4: yearly AUC by entry, the close model scored on next-open labels, fade-1 univariate AUC by entry
    q4 = {"auc_by_year": {e: A.agg(Wd[e]["months"], "auc") for e in lib.ENTRIES}}
    jt = lib.HS.index(lib.H0); rN = D["r_nopen"][:, jt]; yN = (rN > 0).astype(int); mN = np.isfinite(rN) & (np.abs(np.nan_to_num(rN)) > lib.DB0)
    t = pd.DatetimeIndex(D["t"].astype("datetime64[ns]")); mon = np.array([str(x) for x in t.to_period("M")]); pc = Wd["close"]["p"]
    prac = []
    for rec in Wd["close"]["months"]:
        ii = np.flatnonzero((mon == rec["month"]) & mN & np.isfinite(pc)); prac.append({"month": rec["month"], "auc": lib.auc(yN[ii], pc[ii])})
    q4["close_model_on_nopen_labels"] = A.agg(prac, "auc")
    q4["roll_and_cs_by_year"] = {}
    MQ = A.mq_table(D)
    for y in sorted({m[:4] for m in MQ}):
        ms = [m for m in MQ if m[:4] == y]
        q4["roll_and_cs_by_year"][y] = {c: float(np.nanmean([MQ[m][c] for m in ms])) for c in lib.MQ_COLS}
    R["q4"] = q4
    # Q6
    fixed = {e: A.fixed_model(D, e) for e in ("close", "nopen")}
    dec = {e: A.decomposition(D, Wd[e], e, fixed[e]) for e in ("close", "nopen")}
    dec2 = {e: A.decomposition(D, Wd[e], e, fixed[e], cell_vars=("sig", "hl")) for e in ("close", "nopen")}
    R["q6"] = {e: {"waterfall": A.waterfall(dec[e]), "cell_edges": dec[e]["cell_edges"], "base_cell_share": dec[e]["base_cell_share"],
                   "fixed_coef": fixed[e]["coef"], "n_train_fixed": fixed[e]["n_train"]} for e in dec}
    for e in dec2:
        R["q6"][e]["waterfall_vol_x_hl"] = A.waterfall(dec2[e]); R["q6"][e]["cell_edges_vol_x_hl"] = dec2[e]["cell_edges"]
    R["q6"]["market_quality_by_year"] = q4["roll_and_cs_by_year"]
    R["q6"]["correlations_close"] = A.mq_correlations(Wd["close"], MQ); R["q6"]["correlations_nopen"] = A.mq_correlations(Wd["nopen"], MQ)
    # monthly series used by the figures
    R["monthly_auc"] = {e: {r["month"]: r["auc"] for r in Wd[e]["months"]} for e in lib.ENTRIES}
    R["monthly_mq"] = MQ
    # Q7
    tr = A.top10_trades(D, Wd); R["q7_months"] = tr
    q7 = {}
    for key in ("bps_close", "bps_nopen", "bps_mid", "bps_close_model_at_nopen", "hit_close", "hit_nopen", "cs_trades", "hl_trades", "cs_all", "hl_all", "n_close"):
        q7[key] = A.agg(tr, key)
    R["q7"] = q7; log("done analyses")
    pickle.dump({"R": R, "Wd": {e: {"months": Wd[e]["months"], "p": Wd[e]["p"]} for e in Wd}, "IR": IR, "cond": cond, "dec": dec, "fixed": fixed}, open(os.path.join(HERE, "results_minute.pkl"), "wb"))
    pickle.dump({e: Wd[e]["models"] for e in Wd}, open("D:/nt/nt_t09_cache/models.pkl", "wb"))
    json.dump(jsonable(R), open(os.path.join(HERE, "results_minute.json"), "w"), indent=1)
    log("saved")

if __name__ == "__main__":
    main()
