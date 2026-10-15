"""T13 prospective scoring (SPEC.md), for a LATER session. usage: python score_B.py [--csv path]

Loads frozen_B.json (refuses on a sha256 mismatch, or when selection.json no longer matches the sha it froze), reads the Binance file from 2026-09-17
(feature history), finds the complete calendar months from 2026-10 (complete = the file holds 240 minutes into the next month; October is scored from
2026-10-02), scores A1 / H3 / AS, B1 / L1 and C1 / V3 on the FIRST 3 complete months with NO refit and applies the SPEC criteria (A: relative Brier gain of
A1 over H3 >= 1% with a day-block bootstrap CI above 0 and A1 beating AS; B: Spearman gain >= +0.003 with the monthly t-interval above 0; C: skill gain
>= +0.005 likewise), Holm over A, B, C. With fewer than 3 complete months it says so and writes nothing. A later run writes resultsB_followup.json
instead of overwriting the verdict. Nothing here is run in the T13 task (only on synthetic data in the tests)."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import pandas as pd

import t13_freeze as FZ
import t13_lib as L

SCORE_FROM = pd.Timestamp("2026-09-17 00:00"); FIRST_MONTH = pd.Period("2026-10", "M"); SCORE_START = pd.Timestamp("2026-10-02 00:00"); MIN_MONTHS = 3
TAIL = pd.Timedelta(minutes=240)


def load_frozen(path=os.path.join(HERE, "frozen_B.json"), selection_path=os.path.join(HERE, "selection.json")):
    art = json.load(open(path))
    if L.sha_of(art) != art["sha256"]:
        raise ValueError("frozen_B.json does not match its sha256")
    if os.path.exists(selection_path) and L.sha_file(selection_path) != art["selection_json_sha256"]:
        raise ValueError("selection.json differs from the file the freeze was made from")
    return art


def complete_months(last_bar, first=FIRST_MONTH):
    out = []; m = first
    while (m + 1).start_time + TAIL - pd.Timedelta(minutes=1) <= last_bar:
        out.append(m); m = m + 1
    return out


def holm_adj(ps):
    o = sorted(range(len(ps)), key=lambda i: ps[i]); adj = [0.0] * len(ps); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (len(ps) - r) * ps[i])); adj[i] = run
    return adj


def score(art, bars, months=None):
    months = complete_months(bars.index.max()) if months is None else months
    out = {"n_complete_months": len(months), "complete_months": [str(m) for m in months], "verdict": None, "frozen_sha256": art["sha256"]}
    if len(months) < MIN_MONTHS:
        return {**out, "status": f"withheld: {len(months)} complete months, need {MIN_MONTHS}"}
    months = months[:MIN_MONTHS]; assert all(m >= FIRST_MONTH for m in months)
    # ---- A
    P = L.L10.prep(bars, "fwd"); B = L.L10.build(P, L.T_MIN); mA = FZ.a_from_json(art["A"]); Bp = L.a_pair_B(P, B, tuple(mA["pair"])); recs = []
    for mo in months:
        idx, _ = L.T3.month_rows(Bp, L.T_MIN, mo)
        if mo == FIRST_MONTH:
            idx = idx[np.asarray(Bp["D"].ta[idx] >= SCORE_START)]
        pr = L.a_predict(mA, Bp, P, L.T_MIN, idx); recs.append((mo, idx, pr))
    cat = np.concatenate; ta = [Bp["D"].ta[i] for _, i, _ in recs]
    R = {"month": np.array([str(m) for (m, i, _) in recs for _ in range(len(i))]), "ta": cat([t.values for t in ta]),
         "day": cat([np.asarray(t.normalize().values.astype("datetime64[D]").astype(np.int64)) for t in ta]),
         "y": cat([Bp["y"][L.X_BPS][i] for _, i, _ in recs]).astype(float), "p": {m: cat([pr[m] for _, _, pr in recs]) for m in ("H3", "A1", "AS")}}
    sA = L.a_summary(R); cA = L.a_criterion(sA); del P, B, Bp
    # ---- B
    mB = FZ.b_from_json(art["B"]); pB = tuple(mB["pair"]); F = L.build_B_frame(bars, L.O5.FREEZE_FIRST, pairs=[pB]); ok = F["ok60"] & F[f"srok_{L.pair_key(pB)}"]; rows = []
    for mo in months:
        m0 = max(mo.start_time, SCORE_START); te = L.O3.test_rows(F["ta"], ok, m0, (mo + 1).start_time)
        rows.append(L.b_score(mB, F, te, mo))
    gB = list(L.b_gain(rows).values()); muB, loB, hiB, pBv = L.tcrit_ci(gB)
    cB = {"mean": muB, "ci": [loB, hiB], "p": pBv, "pos": int(sum(g > 0 for g in gB)), "n": len(gB), "pass": bool(muB >= 0.003 and loB > 0)}; del F
    # ---- C
    mC = FZ.c_from_json(art["C"]); pC = tuple(mC["pair"]); BC = L.build_C(bars, L.T01.WARM, "binance", pairs=[pC]); rc = [L.c_score(mC, BC, mo, m_start=SCORE_START) for mo in months]
    gC = list(L.c_gain(rc).values()); muC, loC, hiC, pCv = L.tcrit_ci(gC)
    cC = {"mean": muC, "ci": [loC, hiC], "p": pCv, "pos": int(sum(g > 0 for g in gC)), "n": len(gC), "pass": bool(muC >= 0.005 and loC > 0)}
    adj = holm_adj([cA["p"], pBv, pCv])
    out.update({"months": [str(m) for m in months], "A": {"summary": sA, "criterion": cA}, "B": {"rows": rows, "criterion": cB}, "C": {"rows": rc, "criterion": cC},
                "holm_p": dict(zip("ABC", adj)), "verdict": {"A": bool(cA["pass"] and adj[0] < 0.05), "B": bool(cB["pass"] and adj[1] < 0.05), "C": bool(cC["pass"] and adj[2] < 0.05)},
                "status": "verdict"})
    return out


def main():
    csv = sys.argv[sys.argv.index("--csv") + 1] if "--csv" in sys.argv else L.guard.FWD_CSV
    art = load_frozen(); print("frozen sha256", art["sha256"], art["pairs"])
    df = pd.read_csv(csv, usecols=["timestamp", "open", "high", "low", "close", "volume"]); df.index = pd.to_datetime(df["timestamp"]); df = df.drop(columns="timestamp").sort_index()
    bars = df[df.index >= SCORE_FROM]; L.guard.check_index(bars.index, "forward file"); print("bars", len(bars), bars.index.min(), bars.index.max())
    res = score(art, bars); print(res["status"])
    if res["verdict"] is None:
        return
    dst = os.path.join(HERE, "resultsB.json")
    if os.path.exists(dst):
        dst = os.path.join(HERE, "resultsB_followup.json")
    json.dump(res, open(dst, "w"), indent=1, default=float); print("verdict (criteria and Holm p < 0.05):", res["verdict"], "->", dst)


if __name__ == "__main__":
    main()
