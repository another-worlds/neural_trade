"""O10 prospective scoring (SPEC.md), for a LATER session, together with O9's. usage: python score_P.py [--csv path]

Loads frozen_P.json (refuses on a sha256 mismatch). The first 3 complete calendar months AFTER 2026-10 (2026-11, 2026-12, 2027-01) that the forward file contains are
scored ONCE, with no refit, for the choice (choice.json: ROBUST's entry with the chosen exit), E0 (ROBUST's own exit, T 60), E0 at the choice's T and the choice's SHUF
arm (its exit signal permuted within the month). PESS fills, 2 bps, 1x. Daily net returns (by exit day); 5-day moving-block bootstrap, 10,000 draws, seed 0.
  verdict "geometry/volatility exits add": CHOICE - E0 (paired daily) has a 95% lower bound > 0
  verdict "the strategy works": the CHOICE mean > 0 with its 95% lower bound > 0, and CHOICE - SHUF has a lower bound > 0
  also reported: the owner's target (hit > 60 %, drawdown < 5 %), CHOICE - E0T.
It refuses (before reading any bar, only the timestamp of the last row) when fewer than 3 complete months exist, and refuses a second scoring once results_P.json exists.
A month is complete when the file reaches its end plus 15 + T minutes (orders valid 15 bars + the time stop)."""
import json
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import o10_lib as O  # noqa: E402
P2 = O.P2; L9 = O.L9; T = O.T
FROZEN = os.path.join(HERE, "frozen_P.json"); RESULTS = os.path.join(HERE, "results_P.json"); CHOICE = os.path.join(HERE, "choice.json")
SCORE_FROM = pd.Timestamp("2026-09-17 00:00"); MIN_MONTHS = 3; FIRST_SCORED = P2.FIRST_SCORED


def last_bar_time(csv):
    """Timestamp of the last row of the file (reads the final bytes only)."""
    with open(csv, "rb") as f:
        f.seek(0, 2); n = f.tell(); f.seek(max(0, n - 400)); last = f.read().decode().strip().splitlines()[-1]
    return pd.Timestamp(last.split(",")[0])


def tail_of(t_max):
    return pd.Timedelta(minutes=O.VALID + int(t_max))


def complete_months(last_bar, t_max=240, first=FIRST_SCORED):
    out = []; m = first; tail = tail_of(t_max)
    while (m + 1).start_time + tail <= last_bar:
        out.append(m); m = m + 1
    return out


def score_core(art, bars, cfg, months):
    """The scoring on `bars` for `months` (strings). No month rule here (the caller enforces it)."""
    models = O.load_artefact10(art)
    DF, DX = P2.make_dframe_for(bars, O.O3.FIRST + 1)
    S = O.signals_frozen10(models, DF, DX, list(months)); ctx = O.make_ctx(bars, DF, S, O.gcols())
    rows = np.where(np.isin(ctx["mo_row"], months) & np.isfinite(S["sig"]))[0]
    lo = pd.Period(months[0], freq="M").start_time; hi = (pd.Period(months[-1], freq="M") + 1).start_time
    days = pd.date_range(lo, hi - pd.Timedelta(days=1), freq="D")
    aL, aS = L9.gate_masks(S["p"], S["dv"], O.ENTRY["delta"], O.ENTRY["g"])
    e4 = {"*": models["e4"][cfg["T"]]} if cfg["exit"] == "E4" and cfg["T"] in models["e4"] else {}
    daily = {}; stats = {}; kind = O.spec_of(cfg)["kind"]

    def run(key, c, ctx_, sx, **kw):
        tf = O.trades10(ctx_, rows, O.spec_of(c, **kw), S, sx, aL, aS, True)[0]
        daily[key] = P2.daily_net(tf, days, 2.0); net = tf["ret_gross"].values - 2e-4 if len(tf) else np.array([]); cum = np.concatenate([[0.0], np.cumsum(net)])
        stats[key] = {"n_trades": int(len(tf)), "hit": float((net > 0).mean()) if len(net) else None, "max_dd": float((np.maximum.accumulate(cum) - cum).max()),
                      "mean_hold": float(tf["hold"].mean()) if len(tf) else None}
        return tf
    plog = []
    run("CHOICE", cfg, ctx, S["sig"], models=e4, plog=plog)
    run("E0", {"exit": "E0", "T": 60}, ctx, S["sig"])
    if cfg["T"] != 60:
        run("E0T", {"exit": "E0", "T": cfg["T"]}, ctx, S["sig"])
    c2, sx = O.make_shuf(ctx, S["sig"], kind, list(months))
    shuf = None
    if kind == "E4":
        shuf = {}
        for mo, P in plog:
            shuf.setdefault(mo, []).append(P)
        shuf = {k: np.array(v) for k, v in shuf.items()}
    run("SHUF", cfg, c2, sx, models=e4, shuf=shuf)
    res = {"months": list(months), "choice": cfg, "stats": stats}
    c1 = P2.boot_ci(daily["CHOICE"].values); cadd = P2.boot_ci((daily["CHOICE"] - daily["E0"]).values); csh = P2.boot_ci((daily["CHOICE"] - daily["SHUF"]).values)
    st = stats["CHOICE"]
    res["c1_CHOICE"] = {**c1, "pass": bool(c1["mean"] > 0 and c1["lo"] > 0)}
    res["CHOICE_minus_E0"] = {**cadd, "pass": bool(cadd["lo"] > 0)}
    res["CHOICE_minus_SHUF"] = {**csh, "pass": bool(csh["lo"] > 0)}
    if "E0T" in daily:
        res["CHOICE_minus_E0T"] = P2.boot_ci((daily["CHOICE"] - daily["E0T"]).values)
    res["exits_add"] = bool(cadd["lo"] > 0)
    res["strategy_works"] = bool(res["c1_CHOICE"]["pass"] and csh["lo"] > 0)
    res["owner_target_met"] = bool(st["hit"] is not None and st["hit"] > 0.6 and st["max_dd"] < 0.05)
    return res


def main():
    csv = sys.argv[sys.argv.index("--csv") + 1] if "--csv" in sys.argv else T.FWD_CSV
    if os.path.exists(RESULTS):
        raise SystemExit("REFUSED: results_P.json exists; the prospective score is computed once.")
    art = json.load(open(FROZEN))
    if O.P5.sha_of(art) != art["sha256"]:
        raise SystemExit("REFUSED: frozen_P.json does not match its sha256")
    cfg = json.load(open(CHOICE, encoding="utf-8"))["cfg"]
    last = last_bar_time(csv); cm = complete_months(last, cfg["T"])
    if len(cm) < MIN_MONTHS:
        raise SystemExit(f"REFUSED: {len(cm)} complete months after 2026-10 (last bar {last}); need {MIN_MONTHS}. Nothing was read.")
    months = [str(m) for m in cm[:MIN_MONTHS]]
    bars = T.load_bars(csv, SCORE_FROM, (pd.Period(months[-1], freq="M") + 1).start_time + tail_of(cfg["T"]))
    res = score_core(art, bars, cfg, months); res["frozen_sha256"] = art["sha256"]
    json.dump(res, open(RESULTS, "w"), indent=1); print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
