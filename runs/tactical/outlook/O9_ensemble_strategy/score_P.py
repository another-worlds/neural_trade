"""O9 prospective scoring (SPEC.md + addendum), for a LATER session. usage: python score_P.py [--csv path]

Loads frozen_P.json (refuses on a sha256 mismatch). The first 3 complete calendar months AFTER 2026-10 (2026-11, 2026-12, 2027-01, ...) that the forward
file contains are scored ONCE, with no refit, for the candidates PRIMARY (choice.json) and ROBUST (choice_robust.json), their SHUF arms and O8's plain
BRK_k1.5_H60_s1.0, PESS fills, 2 bps. Daily net returns (by exit day); 5-day moving-block bootstrap, 10,000 draws, seed 0.
  criterion 1: the mean daily net of FULL > 0 with a 95% lower bound > 0 and a Holm-adjusted (over the two candidates) two-sided bootstrap p < 0.05
  criterion 2: FULL - SHUF (paired daily) lower bound > 0
  criterion 3: FULL - O8 baseline (paired daily) lower bound > 0
  "works prospectively" iff 1 and 2; "adds over the breakout" iff 3. Reported per candidate.
It refuses (before reading any bar, only the timestamp of the last row) when fewer than 3 complete months exist, and refuses a second scoring once
results_P.json exists. sigma_ref of the first scored month is the median sigma_60 of 2026-10, which is read for that only (no trading in 2026-10)."""
import json
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import o9_p2 as P  # noqa: E402
L = P.L; T = P.T; L8 = P.L8
FROZEN = os.path.join(HERE, "frozen_P.json"); RESULTS = os.path.join(HERE, "results_P.json")
SCORE_FROM = pd.Timestamp("2026-09-17 00:00"); MIN_MONTHS = 3
TAIL = pd.Timedelta(minutes=75)                                # orders valid 15 bars + the 60-bar time stop after the last decision of a month


def last_bar_time(csv):
    """Timestamp of the last row of the file (reads the final bytes only)."""
    with open(csv, "rb") as f:
        f.seek(0, 2); n = f.tell(); f.seek(max(0, n - 400)); last = f.read().decode().strip().splitlines()[-1]
    return pd.Timestamp(last.split(",")[0])


def complete_months(last_bar, first=P.FIRST_SCORED):
    out = []; m = first
    while (m + 1).start_time + TAIL <= last_bar:
        out.append(m); m = m + 1
    return out


def load_candidates():
    c = {}
    for nm, fn in (("PRIMARY", "choice.json"), ("ROBUST", "choice_robust.json")):
        j = json.load(open(os.path.join(HERE, fn), encoding="utf-8")); c[nm] = j["cfg"] if "cfg" in j else j["config"]
    return c


def score_core(art, bars, cands, months):
    """The scoring on `bars` for `months` (strings). No month rule here (the caller enforces it)."""
    models = P.load_artefact(art)
    DF, DX = P.make_dframe_for(bars, L.O3.FIRST + 1)
    prev = str(pd.Period(months[0], freq="M") - 1); allm = [prev] + list(months)
    S = P.signals_frozen(models, DF, DX, allm)
    mo_of = pd.DatetimeIndex(DF["ta"]).to_period("M").astype(str).values; rows = np.where(np.isin(mo_of, months) & np.isfinite(S["L1"]))[0]
    lo = pd.Period(months[0], freq="M").start_time; hi = (pd.Period(months[-1], freq="M") + 1).start_time
    days = pd.date_range(lo, hi - pd.Timedelta(days=1), freq="D")
    daily = {}; stats = {}
    inp = {a: P.arm_inputs(S, a, DF["ta"], allm) for a in ("FULL", "SHUF")}
    for nm, c in list(cands.items()) + [("BASE", P.BASE_CFG)]:
        for arm in (("FULL",) if nm == "BASE" else ("FULL", "SHUF")):
            sig, dv, p = inp[arm]; ref = P.ref_rows(sig, DF["ta"], allm)
            tf = P.trades9(bars, DF, rows, sig, dv, p, ref, c, True); k = f"{nm}/{arm}"
            daily[k] = P.daily_net(tf, days, 2.0)
            net = tf["w"].values * (tf["ret_gross"].values - 2e-4) if len(tf) else np.array([])
            cum = np.concatenate([[0.0], np.cumsum(net)]); stats[k] = {"n_trades": int(len(tf)), "hit": float((net > 0).mean()) if len(net) else None,
                                                                         "max_dd": float((np.maximum.accumulate(cum) - cum).max())}
    res = {"months": list(months), "per_candidate": {}, "stats": stats}
    c1 = {nm: P.boot_ci(daily[f"{nm}/FULL"].values) for nm in cands}
    adj = P.holm([c1[nm]["p_two_sided"] for nm in cands])
    for (nm, _), a in zip(cands.items(), adj):
        c2 = P.boot_ci((daily[f"{nm}/FULL"] - daily[f"{nm}/SHUF"]).values); c3 = P.boot_ci((daily[f"{nm}/FULL"] - daily["BASE/FULL"]).values)
        ok1 = c1[nm]["mean"] > 0 and c1[nm]["lo"] > 0 and a < 0.05; ok2 = c2["lo"] > 0; ok3 = c3["lo"] > 0
        st = stats[f"{nm}/FULL"]
        res["per_candidate"][nm] = {"c1": {**c1[nm], "p_holm": a, "pass": bool(ok1)}, "c2_FULL_minus_SHUF": {**c2, "pass": bool(ok2)}, "c3_FULL_minus_BASE": {**c3, "pass": bool(ok3)},
                                    "works_prospectively": bool(ok1 and ok2), "adds_over_breakout": bool(ok3),
                                    "owner_target_met": bool(st["hit"] is not None and st["hit"] > 0.6 and st["max_dd"] < 0.05)}
    return res


def main():
    csv = sys.argv[sys.argv.index("--csv") + 1] if "--csv" in sys.argv else T.FWD_CSV
    if os.path.exists(RESULTS):
        raise SystemExit("REFUSED: results_P.json exists; the prospective score is computed once.")
    art = json.load(open(FROZEN))
    if L.P5.sha_of(art) != art["sha256"]:
        raise SystemExit("REFUSED: frozen_P.json does not match its sha256")
    last = last_bar_time(csv); cm = complete_months(last)
    if len(cm) < MIN_MONTHS:
        raise SystemExit(f"REFUSED: {len(cm)} complete months after 2026-10 (last bar {last}); need {MIN_MONTHS}. Nothing was read.")
    months = [str(m) for m in cm[:MIN_MONTHS]]
    bars = T.load_bars(csv, SCORE_FROM, (pd.Period(months[-1], freq="M") + 1).start_time + TAIL)
    res = score_core(art, bars, load_candidates(), months); res["frozen_sha256"] = art["sha256"]
    json.dump(res, open(RESULTS, "w"), indent=1); print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
