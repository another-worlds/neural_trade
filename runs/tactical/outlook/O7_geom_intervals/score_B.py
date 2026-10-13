"""O7 prospective scoring (SPEC.md), for a LATER session; NOT run in the O7 task. usage: python score_B.py [--csv path]

Loads O5's frozen artefact (O5_geom_bestbase/frozen_B.json, UNCHANGED; refuses on a sha256 mismatch against its own field and against
frozen_B.sha256), reads the Binance file from 2026-09-17 (bank warm-up), uses 2026-10 as the calibration month and scores the NEXT 3 complete
calendar months (2026-11, 2026-12, 2027-01; complete = the file reaches 240 minutes into the following month) once, with the frozen forecasters
(L1 = the candidate, L0 = the comparator, no refit). Intervals as the retrospective study: for each scored month M the 5% / 95% quantiles of
r_T / sigma_hat on month M-1 (labels ending by M's start) times sigma_hat; `--calib fixed` uses 2026-10 for all three months instead.
Criterion (T = 60, L1 against L0): mean interval-score difference below 0 with its 95% t interval below 0, L1 better in 3 of 3 months, and L1's
pooled coverage within [0.88, 0.92]. Refuses fewer than 3 complete months and any month before 2026-11. A later run writes
resultsB_followup.json instead of overwriting the verdict. T = 15 and 240 are reported without a verdict."""
import json, os, sys
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import o7_lib as P  # noqa: E402
O3 = P.O3; T = P.T
FROZEN = os.path.join(P.OUTLOOK, "O5_geom_bestbase", "frozen_B.json")
FROZEN_SHA = os.path.join(P.OUTLOOK, "O5_geom_bestbase", "frozen_B.sha256")
SCORE_FROM = pd.Timestamp("2026-09-17 00:00"); CAL_MONTH = pd.Period("2026-10", freq="M"); FIRST_SCORED = pd.Period("2026-11", freq="M"); MIN_MONTHS = 3
LABEL_TAIL = pd.Timedelta(minutes=max(P.HORIZONS))
PAIR = ("L1", "L0")                                          # candidate, comparator (frozen_B.json: candidate L1, base L0)


def load_frozen(path=FROZEN, sha_path=FROZEN_SHA):
    art = json.load(open(path))
    if P.sha_of(art) != art["sha256"]:
        raise ValueError("frozen_B.json does not match its sha256 field")
    if os.path.exists(sha_path) and open(sha_path).read().strip() != art["sha256"]:
        raise ValueError("frozen_B.json does not match frozen_B.sha256")
    if (art["base"], art["candidate"]) != ("L0", "L1"):
        raise ValueError("the O7 prospective test needs the ridge pair L0 / L1")
    return art


def complete_months(last_bar, first=FIRST_SCORED):
    out = []; m = first
    while (m + 1).start_time + LABEL_TAIL - pd.Timedelta(minutes=1) <= last_bar:
        out.append(m); m = m + 1
    return out


def rows_of(F, H, m, label_end_by=None):
    m0, m1 = P.month_bounds(m); ta = F["ta"]; ok = F[f"ok{H}"]
    sel = (ta >= np.datetime64(m0, "ns")) & (ta < np.datetime64(m1, "ns")) & ok
    if label_end_by is not None:
        sel &= ta + np.timedelta64(H, "m") <= np.datetime64(label_end_by, "ns")
    return np.where(sel)[0]


def score(art, bars, months=None, calib="rolling"):
    complete = complete_months(bars.index.max())
    if months is not None:
        months = [pd.Period(m, freq="M") for m in months]
        if any(m < FIRST_SCORED for m in months):
            raise ValueError("a month before 2026-11 is not allowed (2026-10 is the calibration month; the freeze trained up to 2026-09-30)")
        if any(m not in complete for m in months):
            raise ValueError("an incomplete month was requested")
        complete = months
    out = {"n_complete_months": len(complete), "complete_months": [str(m) for m in complete], "verdict": None}
    if len(complete) < MIN_MONTHS:
        return {**out, "rows": [], "status": f"withheld: {len(complete)} complete months, need {MIN_MONTHS}"}
    months = [str(m) for m in complete[:MIN_MONTHS]]
    F = P.build_frame_new(bars, O3.FIRST); rows = []
    assert pd.Timestamp(F["ta"][0]) < CAL_MONTH.start_time
    for H in P.HORIZONS:
        st = art["states"][str(H)]["m"]; mods = {"L1": P5_model(st["L1"]), "L0": P5_model(st["L0"])}
        for m in months:
            prev = str(pd.Period(m, freq="M") - 1) if calib == "rolling" else str(CAL_MONTH)
            cal = rows_of(F, H, prev, label_end_by=P.month_bounds(m)[0]); te = rows_of(F, H, m)
            assert len(cal) and len(te)
            for nm in ("L1", "L0"):
                sg_c = P.sigma_of(nm, O3.ridge_pred(mods[nm], P.getX(F, cal, nm))); sg_t = P.sigma_of(nm, O3.ridge_pred(mods[nm], P.getX(F, te, nm)))
                q = P.interval_bounds(F[f"r{H}"][cal] / sg_c); d = P.interval_stats(F[f"r{H}"][te], sg_t, q)
                d.update({"model": nm, "H": H, "month": m, "n_cal": int(len(cal)), "q_lo": q[0], "q_hi": q[1]}); rows.append(d)
    df = pd.DataFrame(rows); res = {}
    for H in P.HORIZONS:
        res[str(H)] = {"iscore": P.paired(df, *PAIR, "iscore_bps", months, H, min_n=MIN_MONTHS), "width": P.paired(df, *PAIR, "width_bps", months, H, min_n=MIN_MONTHS),
                       "cover_L1": P.pooled_cover(df, "L1", months, H), "cover_L0": P.pooled_cover(df, "L0", months, H)}
    verdict = P.criterion(res["60"]["iscore"], res["60"]["cover_L1"], need=MIN_MONTHS)
    out.update({"rows": rows, "months": months, "res": res, "status": "verdict", "calib": calib, "verdict": verdict})
    return out


def P5_model(j):
    return P.P5.model_from_json(j)


def main():
    csv = sys.argv[sys.argv.index("--csv") + 1] if "--csv" in sys.argv else T.FWD_CSV
    calib = sys.argv[sys.argv.index("--calib") + 1] if "--calib" in sys.argv else "rolling"
    art = load_frozen(); print("frozen sha256", art["sha256"], "base", art["base"], "candidate", art["candidate"])
    bars = T.load_bars(csv, SCORE_FROM, pd.Timestamp("2100-01-01")); print("bars", len(bars), bars.index.min(), bars.index.max())
    res = score(art, bars, calib=calib); res["frozen_sha256"] = art["sha256"]; print(res["status"])
    if res["verdict"] is None:
        return
    dst = os.path.join(HERE, "resultsB.json")
    if os.path.exists(dst):
        dst = os.path.join(HERE, "resultsB_followup.json")
    json.dump(res, open(dst, "w"), indent=1, default=float)
    print("criterion:", "pass" if res["verdict"]["pass"] else "fail", res["verdict"], "->", dst)
    for H, r in res["res"].items():
        print("T", H, r)


if __name__ == "__main__":
    main()
