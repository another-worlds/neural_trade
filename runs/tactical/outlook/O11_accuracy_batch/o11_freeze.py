"""O11 freeze and prospective scoring helpers for parts a, b, c (SPEC "Freeze"). The freeze fits on Binance bars 2025-10-01 .. 2026-09-30 23:59 only
(P5.load_freeze_bars reads exactly those rows) and writes `frozen_P_<part>.json` with its sha256. The prospective scoring is done LATER by `score_P_<part>.py`:
the first 3 complete calendar months after 2026-10 (2026-11, 2026-12, 2027-01; a month is complete when the file reaches 240 minutes into the next one), once,
no refit of the frozen arm. A scoring refuses (before reading a bar; only the last row's timestamp) when fewer than 3 months are complete, and refuses a
second scoring once results_P_<part>.json exists.

Prospective criteria (the lead may ratify or change them BEFORE the 3 months exist; they are the retrospective primary tests scaled to 3 months, as O5 did):
  a: mean monthly skill gain (candidate - base, target A) >= +0.010 and positive in 3 of 3 months.
  b: mean monthly Spearman gain refit - frozen (L1, H = 15) >= +0.003 and positive in 3 of 3 months.
  c: mean monthly pinball difference (candidate - base) < 0, candidate better in 3 of 3 months, pooled hit rates within [0.04, 0.06] and [0.94, 0.96]."""
import json, os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import o11_vol_lib as L  # noqa: E402
P5 = L.P5; P6 = L.P6; O3 = L.O3; T = L.T
FIRST_SCORED = pd.Period("2026-11", freq="M"); MIN_MONTHS = 3; TAIL = pd.Timedelta(minutes=max(L.HORIZONS))
SCORE_FROM = pd.Timestamp("2026-09-17 00:00")                    # bank warm-up for a scoring that needs no refit window


def last_bar_time(csv):
    with open(csv, "rb") as f:
        f.seek(0, 2); n = f.tell(); f.seek(max(0, n - 400)); last = f.read().decode().strip().splitlines()[-1]
    return pd.Timestamp(last.split(",")[0])


def complete_months(last_bar, first=FIRST_SCORED):
    out = []; m = first
    while (m + 1).start_time + TAIL - pd.Timedelta(minutes=1) <= last_bar:
        out.append(m); m = m + 1
    return out


def sha_check(art):
    if P5.sha_of(art) != art["sha256"]:
        raise SystemExit("REFUSED: the frozen artefact does not match its sha256")


def header(part, bars):
    return {"spec": f"O11 SPEC.md freeze, part {part}", "last_bar": str(bars.index.max()), "first_bar": str(bars.index.min()), "n_bars": int(len(bars)),
            "stride": T.STRIDE, "warm_bars": O3.FIRST, "m0": str(P5.B_M0), "ridge_alpha": 1.0,
            "features": "XB = S3 (16) + ln(EWMA_tau(channel)+eps_channel) (36); G = o1_lib G1 + G15 (328); L1 = XB + G",
            "lib_note": "o11_vol_lib.py with o1_lib / o3_lib / o5_lib / o6_lib / o7_lib / t01_lib as in the repo at the freeze commit"}


def ridge_json(m):
    return P6.model_to_json(m)


def ridge_load(j):
    return P6.model_from_json(j)


# ------------------------------------------------------------------ part a
def freeze_a(bars):
    assert bars.index.max() <= P5.B_HI
    F = L.build_fwd(bars, O3.FIRST); st = L.fit_a(F, "A", P5.B_M0)
    art = {**header("a", bars), "target": "A: dA = ln RV(a, a+60) - ln RV(a-60, a), T01's ok mask", "n_tr": st["n_tr"], "ar": [float(x) for x in st["ar"]],
           "m": {k: ridge_json(v) for k, v in st["m"].items()}}
    art["sha256"] = P5.sha_of(art); return art


def state_a(art):
    return {"tg": "A", "n_tr": art["n_tr"], "ar": np.array(art["ar"]), "m": {k: ridge_load(v) for k, v in art["m"].items()}}


def score_a_core(art, bars, months):
    F = L.build_fwd(bars, O3.FIRST); st = state_a(art); rows = []
    for mo in months:
        m0, m1 = L.month_bounds(mo); te = np.where((F["ta"] >= np.datetime64(m0, "ns")) & (F["ta"] < np.datetime64(m1, "ns")) & F["oka"])[0]
        rows += L.score_a(st, F, te, "P", mo)
    df = pd.DataFrame(rows)
    g = (df[df.model == "cand"].set_index("month").skill - df[df.model == "base"].set_index("month").skill).reindex(months)
    sp = (df[df.model == "cand"].set_index("month").spearman - df[df.model == "base"].set_index("month").spearman).reindex(months)
    ok = bool(g.mean() >= 0.010 and (g > 0).all())
    return {"months": list(months), "rows": rows, "skill_gain": g.tolist(), "spearman_gain": sp.tolist(), "mean_skill_gain": float(g.mean()), "pass": ok}


# ------------------------------------------------------------------ part b
def freeze_b(bars):
    assert bars.index.max() <= P5.B_HI
    F = L.build_fwd(bars, O3.FIRST); art = {**header("b", bars), "arm": "frozen on the Binance file to 2026-09-30; the refit arm is made at scoring", "states": {}}
    for H in (15, 60):
        tr = O3.train_rows(F["ta"], F[f"ok{H}"], H, P5.B_M0); L.check_rows_clean(F, tr, H, P5.B_M0)
        st = L.fit_ridge_h(F, H, tr, ("S3", "L1"))
        art["states"][str(H)] = {"H": H, "n_tr": st["n_tr"], "first_ta": st["first_ta"], "last_label_end": st["last_label_end"],
                                 "c": {k: float(v) for k, v in st["c"].items()}, "m": {k: ridge_json(v) for k, v in st["m"].items()}}
    art["sha256"] = P5.sha_of(art); return art


def state_b(j):
    return {"H": j["H"], "n_tr": j["n_tr"], "first_ta": j["first_ta"], "last_label_end": j["last_label_end"], "c": dict(j["c"]), "m": {k: ridge_load(v) for k, v in j["m"].items()}}


def score_b_core(art, bars_from_2025_10, months):
    """bars: Binance bars from 2025-10-01 to the end of the last scored month + tail (the refit arm needs the 365 days before each month)."""
    F = L.build_fwd(bars_from_2025_10, O3.FIRST); rows = []
    for mo in months:
        m0, m1 = L.month_bounds(mo)
        for H in (15, 60):
            te = O3.test_rows(F["ta"], F[f"ok{H}"], m0, m1)
            rows += L.score_b(state_b(art["states"][str(H)]), F, te, "frozen", mo)
            tr = L.train_rows_between(F["ta"], F[f"ok{H}"], H, m0 - pd.Timedelta(days=365), m0); L.check_rows_clean(F, tr, H, m0)
            rows += L.score_b(L.fit_ridge_h(F, H, tr, ("S3", "L1")), F, te, "refit_cal", mo)
    df = pd.DataFrame(rows); d = df[(df.H == 15) & (df.model == "L1")]
    g = (d[d.arm == "refit_cal"].set_index("month").spearman - d[d.arm == "frozen"].set_index("month").spearman).reindex(months)
    return {"months": list(months), "rows": rows, "spearman_gain_H15": g.tolist(), "mean": float(g.mean()), "pass": bool(g.mean() >= 0.003 and (g > 0).all())}


# ------------------------------------------------------------------ part c
def freeze_c(bars):
    assert bars.index.max() <= P5.B_HI
    F = L.build_fwd(bars, O3.FIRST); tr = O3.train_rows(F["ta"], F[f"ok{L.H_C}"], L.H_C, P5.B_M0); L.check_rows_clean(F, tr, L.H_C, P5.B_M0)
    m = L.l1_fit(F, tr); qm = L.fit_qr(F, tr, m)
    art = {**header("c", bars), "n_tr": int(len(tr)), "first_ta": str(pd.Timestamp(F["ta"][tr].min())), "last_label_end": str(pd.Timestamp(F["ta"][tr].max()) + pd.Timedelta(minutes=L.H_C)),
           "l1": ridge_json(m), "g1_idx": [int(i) for i in qm["g1_idx"]], "qr_mu": qm["mu"].tolist(), "qr_sd": qm["sd"].tolist(), "qr_alpha": L.QR_ALPHA, "qr_stride": L.QR_STRIDE,
           "qr": {str(tau): {"coef": [float(x) for x in qm["q"][tau].coef_], "intercept": float(qm["q"][tau].intercept_)} for tau in (L.Q_LO, L.Q_HI)}}
    art["sha256"] = P5.sha_of(art); return art


class _Lin:
    def __init__(self, j):
        self.coef_ = np.array(j["coef"]); self.intercept_ = j["intercept"]

    def predict(self, X):
        return X @ self.coef_ + self.intercept_


def state_c(art):
    m = ridge_load(art["l1"]); qm = {"g1_idx": np.array(art["g1_idx"]), "mu": np.array(art["qr_mu"]), "sd": np.array(art["qr_sd"]),
                                     "q": {tau: _Lin(art["qr"][str(tau)]) for tau in (L.Q_LO, L.Q_HI)}}
    return m, qm


def score_c_core(art, bars, months):
    F = L.build_fwd(bars, O3.FIRST); m, qm = state_c(art); rows = []
    for mo in months:
        f = str(pd.Period(mo, freq="M") - 1); cal, te = L.month_rows_c(F, f); assert len(cal) and len(te)
        bl, bh = L.base_quantiles(F, m, cal, te); cl, ch = L.qr_predict(qm, F, te, m); r = F[f"r{L.H_C}"][te]
        for nm, (lo, hi) in (("base", (bl, bh)), ("cand", (cl, ch))):
            rows.append({"span": "P", "month": mo, "model": nm, **L.tail_row(r, lo, hi)})
    df = pd.DataFrame(rows); d = (df[df.model == "cand"].set_index("month").pin_bps - df[df.model == "base"].set_index("month").pin_bps).reindex(months)
    c = df[df.model == "cand"]; n = c.n.sum(); h05 = float(c.n_lo.sum() / n); h95 = float(c.n_hi.sum() / n)
    ok = bool(d.mean() < 0 and (d < 0).all() and 0.04 <= h05 <= 0.06 and 0.94 <= h95 <= 0.96)
    return {"months": list(months), "rows": rows, "pinball_diff": d.tolist(), "mean": float(d.mean()), "hit05": h05, "hit95": h95, "pass": ok}


# ------------------------------------------------------------------ the scoring scripts' main
def main_score(part, frozen_path, results_path, core, load_from):
    csv = sys.argv[sys.argv.index("--csv") + 1] if "--csv" in sys.argv else T.FWD_CSV
    if os.path.exists(results_path):
        raise SystemExit(f"REFUSED: {os.path.basename(results_path)} exists; the prospective score is computed once.")
    art = json.load(open(frozen_path)); sha_check(art)
    last = last_bar_time(csv); cm = complete_months(last)
    if len(cm) < MIN_MONTHS:
        raise SystemExit(f"REFUSED: {len(cm)} complete months after 2026-10 (last bar {last}); need {MIN_MONTHS}. Nothing was read.")
    months = [str(m) for m in cm[:MIN_MONTHS]]
    bars = T.load_bars(csv, load_from, (pd.Period(months[-1], freq="M") + 1).start_time + TAIL)
    res = core(art, bars, months); res["frozen_sha256"] = art["sha256"]; res["part"] = part
    json.dump(res, open(results_path, "w"), indent=1, default=float); print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1, default=float))
