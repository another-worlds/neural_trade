"""O11 part d prospective scoring, for a LATER session. usage: python score_P_d.py [--csv path]

Loads frozen_P_d.json (refuses on a sha256 mismatch; exists only if the primary passed on J). The first 3 complete calendar months AFTER 2026-10 that the forward
file contains are scored ONCE with no refit: the monthly AUC of the chosen variant and of base on all valid decision rows (no dead band), their paired
difference, the gated-subset AUC / hit rate and the Brier score. The interval of the pooled AUC difference is a 5-day moving-block bootstrap over days
(10,000 draws, seed 0, paired rows). Pass iff the mean monthly difference >= +0.005 and the bootstrap lower bound > 0.
It refuses (reading only the last timestamp) with fewer than 3 complete months, and refuses a second scoring once results_P_d.json exists."""
import json
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import o11d_lib as D  # noqa: E402
sys.path.insert(0, os.path.join(D.OUTLOOK, "O9_ensemble_strategy"))
import score_P as SP9  # noqa: E402  (last_bar_time, complete_months: O9's month rule)
P2 = D.P2; P5 = D.P5; T = D.T
FROZEN = os.path.join(HERE, "frozen_P_d.json"); RESULTS = os.path.join(HERE, "results_P_d.json")
SCORE_FROM = pd.Timestamp("2026-09-17 00:00"); MIN_MONTHS = 3; BOOT_B = 10000


def fast_auc(y, p):
    from scipy.stats import rankdata
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((rankdata(p)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def block_auc_diff(day, y, pa, pb, B=10000, block=5, seed=0):
    """Moving-block bootstrap (blocks of days) of AUC(pa) - AUC(pb) on the pooled paired rows."""
    days = np.unique(day); nd = len(days); rows = [np.where(day == d)[0] for d in days]
    rng = np.random.default_rng(seed); nb = int(np.ceil(nd / block)); out = np.empty(B)
    starts = rng.integers(0, max(nd - block + 1, 1), size=(B, nb))
    for b in range(B):
        di = (starts[b][:, None] + np.arange(block)[None, :]).ravel()[:nd]; idx = np.concatenate([rows[i] for i in di])
        out[b] = fast_auc(y[idx], pa[idx]) - fast_auc(y[idx], pb[idx])
    return out


def score_core(art, bars, months):
    """The scoring on `bars` for `months` (strings). No month rule here (the caller enforces it)."""
    chosen = art["chosen"]; M = art["models"]
    DF, DX = P2.make_dframe_for(bars, P5.FREEZE_FIRST); DD = D.decision_data(bars, DF, DX)
    P = {v: np.full(len(DD["ta"]), np.nan) for v in ("base", chosen)}
    for mo in months:
        rows = D.month_idx(DD["ta"], mo); rows = rows[DD["valid"][rows]]
        for v in P:
            P[v][rows] = D.predict_j(M[v], DD["tb7"][rows], DD["G15"][rows])
    tab = D.table(P, DD, months)
    diff = D.paired(tab, chosen, "base", "auc", months)
    rows = np.where(np.isfinite(P["base"]))[0]; y = (DD["ret60"][rows] > 0).astype(int)
    day = pd.DatetimeIndex(DD["ta"][rows]).normalize().values.astype("datetime64[D]").astype(np.int64)
    bd = block_auc_diff(day, y, P[chosen][rows], P["base"][rows], B=BOOT_B)
    lo, hi = np.percentile(bd, [2.5, 97.5])
    res = {"months": list(months), "chosen": chosen, "monthly": tab, "monthly_auc_diff": diff.tolist(), "mean_monthly_auc_diff": float(diff.mean()),
           "pooled_auc": {v: fast_auc(y, P[v][rows]) for v in P}, "pooled_auc_diff_ci": {"mean": float(bd.mean()), "lo": float(lo), "hi": float(hi)}, "n_rows": int(len(rows))}
    for k in ("auc_gated", "hit_gated", "brier"):
        d = D.paired(tab, chosen, "base", k, months); res[f"mean_diff_{k}"] = float(d.mean()) if len(d) else None
    res["primary_pass"] = bool(res["mean_monthly_auc_diff"] >= 0.005 and lo > 0)
    return res


def main():
    csv = sys.argv[sys.argv.index("--csv") + 1] if "--csv" in sys.argv else T.FWD_CSV
    if os.path.exists(RESULTS):
        raise SystemExit("REFUSED: results_P_d.json exists; the prospective score is computed once.")
    if not os.path.exists(FROZEN):
        raise SystemExit("REFUSED: frozen_P_d.json does not exist (the primary did not pass on J, or the freeze was not made).")
    art = json.load(open(FROZEN))
    if P5.sha_of(art) != art["sha256"]:
        raise SystemExit("REFUSED: frozen_P_d.json does not match its sha256")
    last = SP9.last_bar_time(csv); cm = SP9.complete_months(last)
    if len(cm) < MIN_MONTHS:
        raise SystemExit(f"REFUSED: {len(cm)} complete months after 2026-10 (last bar {last}); need {MIN_MONTHS}. Nothing was read.")
    months = [str(m) for m in cm[:MIN_MONTHS]]
    bars = T.load_bars(csv, SCORE_FROM, (pd.Period(months[-1], freq="M") + 1).start_time + SP9.TAIL)
    res = score_core(art, bars, months); res["frozen_sha256"] = art["sha256"]
    json.dump(res, open(RESULTS, "w"), indent=1); print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
