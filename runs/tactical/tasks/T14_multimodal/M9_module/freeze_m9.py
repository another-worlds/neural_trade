"""Freeze the M9 profile module for the prospective period (SPEC_M9 "Freeze rule"). CPU. Run: D:/nt/nt_mm_env/python.exe freeze_m9.py [--view NAME]
Without --view: the pre-registered rule from results_m9.json (pre-registered views first, then the post hoc pool); if no view passes, writes frozen_P.json with
status "none" and freezes nothing. With --view NAME: writes frozen_P_<NAME>.json labelled as a post hoc candidate chosen by the lead / on the looks, NOT by the rule.
The artefact has no weights: the column recipe (sha256 of m9_lib.py and of the recipe), the pool window, the harness' fixed ridge / logistic settings, and the
final-fit profile (fitted means by UTC hour, weekday / weekend) used by the figure."""
import argparse
import hashlib
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); T14 = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import m9_lib as M9  # noqa: E402
import anchors as AN  # noqa: E402
import evaluate as EV  # noqa: E402
from _libs import T10, T11, T3  # noqa: E402
from m6_lib import KEYS  # noqa: E402

CUT = pd.Timestamp("2026-10-01")


def pool_info(base, valid, tr):
    ta = base.ta[tr]
    return {"cut": str(CUT), "n_pool": int(len(tr)), "first_pool_ta": str(ta.min()), "last_pool_ta": str(ta.max()),
            "pool_ta_sha256": hashlib.sha256(np.asarray(ta).astype("datetime64[ns]").astype(np.int64).tobytes()).hexdigest(),
            "rule": "ta >= max(cut - 365 d, 2025-10-02), ta + 60 min <= cut, harness ok, all 12 M6 keys finite (the harness train_mask)"}


def final_fit(view):
    """Final ridge (A) and logistic (B) on the pool of label-known Binance anchors to 2026-09-30; fitted means by UTC hour x weekday / weekend, with the V3 / H3
    fit alone and the realised means beside them. In-sample (fitted values on the pool): a picture of the estimated profile, not a score."""
    from sklearn.linear_model import LogisticRegression
    base = AN.load_base("fwd"); ta = base.ta
    desc = pd.read_parquet(os.path.join(AN.CACHE_DIR, "desc_fwd.parquet")).reindex(ta)[KEYS].to_numpy(float)
    Z, names = M9.view_matrix(view, ta, desc); finite = np.isfinite(Z).all(1) & np.isfinite(desc).all(1); valid = base.ok & finite
    tr = np.where(EV.train_mask(ta, valid, CUT, AN.LOOKS["look2"]["train_from"]))[0]
    hr = np.asarray(ta[tr].hour); wk = (np.asarray(ta[tr].dayofweek) >= 5)
    out = {"view": view, "columns": names, "pool": pool_info(base, valid, tr), "hours": list(range(24))}
    A = base.A; V3 = T11.X_A(A, "V3", np.arange(len(ta))); yA = A["T"]["A"]
    Xf = np.hstack([V3, Z]); pf = T11._lin(T11._ridge(Xf[tr], yA[tr]), Xf[tr]); p0 = T11._lin(T11._ridge(V3[tr], yA[tr]), V3[tr])
    B = base.B; yB = B["y"][EV.X_B].astype(float); H = EV.H_B
    har = T3.fit_har(B, H, tr, None, False); p1 = T10.fp_from_total_sd(T10.har_sd(har, B, H, tr), EV.X_B, H)
    Ztr = T10.h3_design(B, tr, p1)
    def lr(a):
        mu, sd = a.mean(0), a.std(0) + 1e-12
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = LogisticRegression(C=1.0, max_iter=2000).fit((a - mu) / sd, yB[tr])
        return m.predict_proba((a - mu) / sd)[:, 1]
    qf = lr(np.hstack([Ztr, Z[tr]])); q0 = lr(Ztr)
    def prof(v, grp):
        return [float(v[(hr == h) & (wk == grp)].mean()) if ((hr == h) & (wk == grp)).sum() else None for h in range(24)]
    for nm, series in (("A_view", pf), ("A_base", p0), ("A_real", yA[tr]), ("B_view", qf), ("B_base", q0), ("B_real", yB[tr])):
        out[nm] = {"weekday": prof(series, False), "weekend": prof(series, True)}
    out["n_by_hour"] = {"weekday": [int(((hr == h) & ~wk).sum()) for h in range(24)], "weekend": [int(((hr == h) & wk).sum()) for h in range(24)]}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--view"); a = ap.parse_args(argv); t0 = time.time()
    R = json.load(open(os.path.join(HERE, "results_m9.json"))); choice = R["freeze_choice"]
    rec = {"recipe_sha256": M9.recipe_sha256(), "m9_lib_sha256": M9.sha256_file(os.path.join(HERE, "m9_lib.py")),
           "harness": {"A": "ridge alpha 1, standardised, on V3 + view columns (T11 _ridge), monthly refit on the 365 days before the month",
                       "B": "logistic C 1, standardised, on H3 + view columns, same refit"},
           "criterion": {"A_inc_min": 0.005, "B_rel_min": 0.01, "pocock_alpha": 0.0294, "looks_months": [3, 6], "min_complete_months": 3,
                         "controls": "C1 permuted within month, C2a harness S3 duplicate (no C2b for the linear views)"}}
    if a.view:
        view = a.view; rec["status"] = "post hoc candidate"
        rec["label"] = "chosen by the lead / on the looks, NOT by the pre-registered freeze rule (no view passed the rule); only the prospective test is a valid judge"
    elif choice["view"]:
        view = choice["view"]; rec["status"] = "frozen by rule (" + choice["kind"] + ")"; rec["label"] = choice["kind"]
    else:
        rec.update(status="none", label="no pre-registered and no post hoc view passed the criterion on both looks; nothing is frozen", freeze_choice=choice,
                   seconds=round(time.time() - t0, 1))
        json.dump(rec, open(os.path.join(HERE, "frozen_P.json"), "w"), indent=1); print(json.dumps(rec, indent=1)); return
    rec.update(view=view, columns=M9.VIEW_COLS[view], n_cols=len(M9.VIEW_COLS[view]), keys=list(KEYS), freeze_choice=choice, final_fit=final_fit(view))
    json.dump(rec["final_fit"], open(os.path.join(HERE, f"profile_{view}.json"), "w"), indent=1)
    rec["profile_file"] = f"profile_{view}.json"; rec["profile_sha256"] = M9.sha256_file(os.path.join(HERE, rec["profile_file"])); rec["seconds"] = round(time.time() - t0, 1)
    p = os.path.join(HERE, f"frozen_P_{view}.json" if a.view else "frozen_P.json")
    json.dump({k: v for k, v in rec.items() if k != "final_fit"} | {"pool": rec["final_fit"]["pool"]}, open(p, "w"), indent=1)
    print(open(p).read())


if __name__ == "__main__":
    main()
