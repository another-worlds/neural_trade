"""O6 prospective scoring (SPEC.md), for a LATER session. usage: python score_B.py [--csv path]

Loads frozen_B.json (refuses on a sha256 mismatch), reads the Binance file from 2026-09-17 (bank warm-up), finds the complete calendar months from
2026-10 (complete = the file reaches 240 minutes into the next month, for the longest label), scores the frozen L1 and L2 (= L1 + G60) on the first 3
complete months with NO refit and applies the O6 criterion at H = 240 (L2 - L1): mean Spearman gain >= +0.003, 95% t lower bound > 0, positive in
3 of 3 months, mean QLIKE difference <= 0. It refuses fewer than 3 complete months and any month before 2026-10. A later run writes
resultsB_followup.json instead of overwriting the verdict. Nothing here is run in the freeze task."""
import json, os, sys
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import o6_lib as P  # noqa: E402
O3 = P.O3; T = P.T

SCORE_FROM = pd.Timestamp("2026-09-17 00:00"); FIRST_MONTH = pd.Period("2026-10", freq="M"); MIN_MONTHS = 3
LABEL_TAIL = pd.Timedelta(minutes=max(P.HORIZONS))


def load_frozen(path=os.path.join(HERE, "frozen_B.json")):
    art = json.load(open(path))
    if P.sha_of(art) != art["sha256"]:
        raise ValueError("frozen_B.json does not match its sha256")
    return art


def complete_months(last_bar, first=FIRST_MONTH):
    out = []; m = first
    while (m + 1).start_time + LABEL_TAIL - pd.Timedelta(minutes=1) <= last_bar:
        out.append(m); m = m + 1
    return out


def score(art, bars, months=None):
    """Score the frozen models (no refit) on the first 3 complete months (or on `months`, all >= 2026-10 and complete). Fewer than 3: withheld."""
    complete = complete_months(bars.index.max())
    if months is not None:
        months = [pd.Period(m, freq="M") for m in months]
        if any(m < FIRST_MONTH for m in months):
            raise ValueError("a month before 2026-10 is not allowed (the freeze trained up to 2026-09-30)")
        if any(m not in complete for m in months):
            raise ValueError("an incomplete month was requested")
        complete = months
    out = {"n_complete_months": len(complete), "complete_months": [str(m) for m in complete], "verdict": None}
    if len(complete) < MIN_MONTHS:
        return {**out, "rows": [], "status": f"withheld: {len(complete)} complete months, need {MIN_MONTHS}"}
    months = [str(m) for m in complete[:MIN_MONTHS]]; base, cand = art["base"], art["candidate"]
    F = P.build_frame6(bars, P.FREEZE_FIRST); rows = []
    assert pd.Timestamp(F["ta"][0]) < FIRST_MONTH.start_time
    for m in months:
        m0, m1 = P.month_bounds(m)
        for H in P.HORIZONS:
            st = P.state_from_json(art["states"][str(H)]); te = O3.test_rows(F["ta"], F[f"ok{H}"], m0, m1)
            rows.extend(P.score(st, F, te, "B", m, models=(base, cand)))
    df = pd.DataFrame(rows); res = {}
    for H in P.HORIZONS:
        res[str(H)] = {"sp": P.paired(df, cand, base, "spearman", months, H, min_n=MIN_MONTHS), "ql": P.paired(df, cand, base, "qlike", months, H, min_n=MIN_MONTHS)}
    out.update({"rows": rows, "months": months, "res": res, "status": "verdict", "base": base, "candidate": cand,
                "verdict": P.criterion(res["240"]["sp"], res["240"]["ql"], MIN_MONTHS)})
    return out


def main():
    csv = sys.argv[sys.argv.index("--csv") + 1] if "--csv" in sys.argv else T.FWD_CSV
    art = load_frozen(); print("frozen sha256", art["sha256"], "base", art["base"], "candidate", art["candidate"])
    bars = T.load_bars(csv, SCORE_FROM, pd.Timestamp("2100-01-01")); print("bars", len(bars), bars.index.min(), bars.index.max())
    res = score(art, bars); res["frozen_sha256"] = art["sha256"]; print(res["status"])
    if res["verdict"] is None:
        return
    dst = os.path.join(HERE, "resultsB.json")
    if os.path.exists(dst):
        dst = os.path.join(HERE, "resultsB_followup.json")
    json.dump(res, open(dst, "w"), indent=1, default=float)
    print("criterion:", "hourly geometry adds" if res["verdict"]["pass"] else "does not add", res["verdict"], "->", dst)
    for H, r in res["res"].items():
        print("H", H, "dSpearman", r["sp"], "dQLIKE", r["ql"])


if __name__ == "__main__":
    main()
