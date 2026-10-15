"""T13 (SPEC.md): pipelines of the three parts. CPU only.

  A  T10's H3 (t10_lib: HAR, logit(H1), S3 extras, UTC hour harmonics; logistic C = 1) for the barrier hit X = 50 bps, T = 60 min, and A1 = H3 + SR (same
     standardisation and the same logistic fit), AS = A1 with the SR rows permuted within the calendar month (training and test rows each). SR is
     divided by the sd of the 60-minute return of the month's fitted HAR (T03, times sqrt(pi/2), as H3's own p1).
  B  O5's L1 (ridge on [S3 | ln bank | G1 | G15], the code of o1/o3/o5_lib copied under ext/ from ed870691) and B1 = L1 + SR; Spearman with the next-60-minute RV.
  C  T11's V3 (S3 + 96 UTC 15-minute slots, ridge) and C1 = V3 + SR; skill against the AR baseline, target A (next-hour change in log RV).
The SR block is sr_lib.py. For B and C the sigma is the fixed blend sr_lib.sigma_blend (their bases have no HAR).
Anchor a sees bars < a only; the SR window of the pair (bar, nb) is part of the row filter: a row without a clean window is dropped for BOTH the base and the
candidate of that pair."""
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__)); TASKS = os.path.dirname(HERE)
for p in (HERE, os.path.join(HERE, "ext"), os.path.join(TASKS, "T10_barrier_hit"), os.path.join(TASKS, "T11_utc_seasonality"), os.path.join(TASKS, "T12_barrier_calibrated")):
    if p not in sys.path:
        sys.path.insert(0, p)
import sr_lib as S  # noqa: E402
import t10_lib as L10  # noqa: E402
import t11_lib as L11  # noqa: E402
import t12_lib as L12  # noqa: E402
import o1_lib as O1  # noqa: E402
import o3_lib as O3  # noqa: E402
import o5_lib as O5  # noqa: E402

T3 = L10.T3; T01 = L11.T; guard = L10.guard
PAIRS = S.PAIRS
X_BPS, T_MIN = 50, 60
SEL_MONTHS = [pd.Period(m, "M") for m in pd.period_range("2019-01", "2024-06", freq="M").astype(str)]
SEL_HI_BARS = pd.Timestamp("2024-07-01 03:59")             # the selection reads no bar after this (the 4-hour label tail of the last anchor)
J1A_MONTHS = list(pd.period_range("2017-10", "2018-12", freq="M"))
J1B_MONTHS = list(pd.period_range("2018-07", "2018-12", freq="M"))
J2_MONTHS = list(pd.period_range("2025-11", "2026-09", freq="M"))
BOOT = L10.BOOT
SCRATCH = "D:/nt/nt_scratch/t13"
F_LO, F_HI = pd.Timestamp("2025-10-01 00:00"), pd.Timestamp("2026-09-30 23:59")
F_M0 = F_HI + pd.Timedelta(minutes=1)
TRAIN_FROM_B = pd.Timestamp("2025-10-02 00:00")


def pair_key(p):
    return f"{p[0]}x{p[1]}"


def pair_from_key(k):
    a, b = k.split("x"); return int(a), int(b)


def sha_file(path):
    return hashlib.sha256(open(path, "rb").read().replace(b"\r\n", b"\n")).hexdigest()


def sha_of(obj):
    body = {k: v for k, v in obj.items() if k != "sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def tcrit_ci(x):
    x = np.asarray(x, float); n = len(x); mu = float(x.mean())
    if n < 2:
        return mu, float("nan"), float("nan"), float("nan")
    se = float(x.std(ddof=1) / np.sqrt(n)); h = stats.t.ppf(0.975, n - 1) * se
    p = float(stats.t.sf(mu / se, n - 1)) if se > 0 else (0.0 if mu > 0 else 1.0)
    return mu, float(mu - h), float(mu + h), p                       # p: one-sided, H1: mean > 0


# ================================================================== part A
def perm_within_month(M, months, seed):
    """rows of M permuted among the rows that share a calendar month (months: array of month labels, one per row)."""
    rng = np.random.default_rng(seed); out = M.copy(); months = np.asarray(months)
    for mo in np.unique(months):
        ix = np.where(months == mo)[0]; out[ix] = M[ix[rng.permutation(len(ix))]]
    return out


def a_sr(P, B, idx, sd, pair):
    return S.sr_block(P.h, P.l, P.c, B["a"][idx], sd, pair[0], pair[1])


def a_pair_B(P, B, pair):
    """B with the row filter of the pair (the SR window is clean)."""
    ok = S.clean_windows(P.tmin, pair[0], pair[1])(B["a"])
    return {**B, "ok": B["ok"] & ok}


def _logit_fit(Z, y):
    from sklearn.linear_model import LogisticRegression
    mu, sd = Z.mean(0), Z.std(0) + 1e-12
    if y.min() == y.max():
        return None
    lr = LogisticRegression(C=1.0, max_iter=2000).fit((Z - mu) / sd, y)
    return (mu, sd, lr.coef_[0].copy(), float(lr.intercept_[0]))


def _logit_pred(m, Z):
    mu, sd, coef, ic = m; return 1 / (1 + np.exp(-(((Z - mu) / sd) @ coef + ic)))


def a_fit(B, P, T, m0, days, pair, seed=0):
    """HAR, H3 (X = 50) as t10_lib.fit_models, A1 and AS, on the `days` before m0 (labels ended by m0). B is the pair-filtered dict."""
    rows = T3.train_rows(B, T, m0, days); X = X_BPS
    har = T3.fit_har(B, T, rows, None, False); p1, sd = L10.h1_prob(B, T, rows, har)
    Z0 = L10.h3_design(B, rows, p1[X]); y = B["y"][X][rows]
    SR = a_sr(P, B, rows, sd, pair); mo = np.asarray(B["D"].ta[rows].to_period("M").astype(str))
    out = {"har": har, "n_train": int(len(rows)), "pair": list(pair), "h3": _logit_fit(Z0, y), "a1": _logit_fit(np.hstack([Z0, SR]), y),
           "as": _logit_fit(np.hstack([Z0, perm_within_month(SR, mo, seed)]), y),
           "last_label_end": str(B["D"].ta[rows].max() + pd.Timedelta(minutes=T)) if len(rows) else None}
    return out


def a_predict(models, B, P, T, idx, seed=0):
    """H3, A1, AS probabilities (X = 50) on rows idx; the SR block is divided by the sd of the frozen / month HAR."""
    X = X_BPS; p1, sd = L10.h1_prob(B, T, idx, models["har"]); Z0 = L10.h3_design(B, idx, p1[X]); pair = tuple(models["pair"])
    SR = a_sr(P, B, idx, sd, pair); mo = np.asarray(B["D"].ta[idx].to_period("M").astype(str))
    out = {}
    for nm, Z in (("H3", Z0), ("A1", np.hstack([Z0, SR])), ("AS", np.hstack([Z0, perm_within_month(SR, mo, seed + 1)]))):
        m = models[{"H3": "h3", "A1": "a1", "AS": "as"}[nm]]
        out[nm] = p1[X] if m is None else _logit_pred(m, Z)
    return out


def a_walk(B, P, T, months, pair, days_fn=None, frozen=None, seed=0):
    """Monthly walk-forward (a refit at each month start) or one frozen fit. Returns the pooled rows and the per-month info."""
    Bp = a_pair_B(P, B, pair); recs = []; info = []
    for month in months:
        m0 = month.start_time; idx, n_all = T3.month_rows(Bp, T, month)
        if len(idx) == 0:
            info.append({"month": str(month), "skipped": True}); continue
        days = L10.TRAIN_DAYS if days_fn is None else days_fn(m0)
        models = frozen if frozen is not None else a_fit(Bp, P, T, m0, days, pair, seed)
        pr = a_predict(models, Bp, P, T, idx, seed)
        recs.append({"month": str(month), "ta": Bp["D"].ta[idx], "y": Bp["y"][X_BPS][idx], "p": pr})
        info.append({"month": str(month), "n": int(len(idx)), "n_train": models["n_train"], "last_label_end": models["last_label_end"]})
    cat = np.concatenate
    R = {"month": np.array([r["month"] for r in recs for _ in range(len(r["ta"]))]),
         "day": cat([np.asarray(r["ta"].normalize().values.astype("datetime64[D]").astype(np.int64)) for r in recs]),
         "ta": cat([r["ta"].values for r in recs]), "y": cat([r["y"] for r in recs]).astype(float),
         "p": {m: cat([r["p"][m] for r in recs]) for m in ("H3", "A1", "AS")}}
    return R, info


def a_monthly_gain(R):
    """per month relative Brier improvement of A1 over H3: 1 - brier(A1) / brier(H3)"""
    out = {}
    for mo in np.unique(R["month"]):
        k = R["month"] == mo; y = R["y"][k]
        out[str(mo)] = float(1 - np.mean((R["p"]["A1"][k] - y) ** 2) / np.mean((R["p"]["H3"][k] - y) ** 2))
    return out


def a_summary(R, seed=0):
    """Pooled Brier of H3 / A1 / AS; relative improvement of A1 over H3 and over AS with the day-block bootstrap 95% CI (10,000 draws) and the one-sided p."""
    y = R["y"]; idx = L10.boot_index(R["day"], BOOT, seed); se = {m: (R["p"][m] - y) ** 2 for m in R["p"]}
    out = {"n": int(len(y)), "base_rate": float(y.mean()), "n_months": int(len(np.unique(R["month"]))),
           "brier": {m: float(se[m].mean()) for m in se}, "auc": {m: L10.auc(y, R["p"][m]) for m in R["p"]}}
    for nm, base in (("A1_vs_H3", "H3"), ("A1_vs_AS", "AS"), ("AS_vs_H3", "H3")):
        top = nm.split("_vs_")[0]; bs = L10.day_boot_skill(se[top], se[base], R["day"], idx)
        out[nm] = {"rel": float(1 - se[top].sum() / se[base].sum()), "ci": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
                   "p": float((np.sum(bs <= 0) + 1) / (len(bs) + 1))}
    mg = a_monthly_gain(R); out["monthly_gain_A1_vs_H3"] = mg; out["mean_monthly_gain"] = float(np.mean(list(mg.values())))
    return out


def a_criterion(s):
    a = s["A1_vs_H3"]; b = s["A1_vs_AS"]
    c1 = bool(a["rel"] >= 0.01 and a["ci"][0] > 0); c2 = bool(b["rel"] > 0 and b["ci"][0] > 0)
    return {"c1_rel_ge_1pct_ci_gt_0": c1, "c2_beats_AS": c2, "pass": bool(c1 and c2), "p": float(max(a["p"], b["p"]))}


# ================================================================== part B
def build_B_frame(s, first, pairs=PAIRS):
    """O5's frame (build_frame: ta, XB [S3 | ln bank], G [G1 | G15], rv / ok per horizon) plus, per pair, the SR block (sigma = sigma_blend) and its clean-window flag."""
    F3 = O3.build_features(s, first)
    out = {"ta": F3["ta"], "XB": np.concatenate([F3["S3"], O3.log_bank(F3["bank"])], 1)}
    g1, g15 = O1.window_features(F3["A"], F3["anch"]); out["G"] = np.concatenate([g1, g15], 1).astype(np.float32); del g1, g15
    for H in O3.HORIZONS:
        out[f"rv{H}"] = F3[f"rv{H}"]; out[f"ok{H}"] = F3[f"ok{H}"]
    A = F3["A"]; a = F3["anch"]; sig = S.sigma_blend(F3["S3"]); tmin = s.index.values.astype("datetime64[m]").astype(np.int64)
    for p in pairs:
        out[f"SR_{pair_key(p)}"] = S.sr_block(A[:, 1], A[:, 2], A[:, 3], a, sig, p[0], p[1]).astype(np.float32)
        out[f"srok_{pair_key(p)}"] = S.clean_windows(tmin, p[0], p[1])(a)
    return out


def b_fit(F, m0, pair):
    """L1 and B1 (ridge alpha 1 on all 365-day rows, as o5_lib.fit_h's ridge) for a test starting m0, on the rows with a clean window of the pair."""
    k = pair_key(pair); ok = F["ok60"] & F[f"srok_{k}"]
    tr = O3.train_rows(F["ta"], ok, 60, m0); y = np.log(F["rv60"][tr])
    XL = np.concatenate([F["XB"][tr], F["G"][tr].astype(np.float64)], 1)
    st = {"pair": list(pair), "n_tr": int(len(tr)), "first_ta": str(pd.Timestamp(F["ta"][tr].min())) if len(tr) else None,
          "last_label_end": str(pd.Timestamp(F["ta"][tr].max()) + pd.Timedelta(minutes=60)) if len(tr) else None, "m": {}}
    st["m"]["L1"] = O3.ridge_fit(XL, y); st["m"]["B1"] = O3.ridge_fit(np.concatenate([XL, F[f"SR_{k}"][tr].astype(np.float64)], 1), y)
    return st


def b_score(st, F, te, month):
    k = pair_key(tuple(st["pair"])); v = F["rv60"][te]; XL = np.concatenate([F["XB"][te], F["G"][te].astype(np.float64)], 1)
    preds = {"L1": O3.ridge_pred(st["m"]["L1"], XL), "B1": O3.ridge_pred(st["m"]["B1"], np.concatenate([XL, F[f"SR_{k}"][te].astype(np.float64)], 1))}
    return {"month": str(month), "n_te": int(len(te)), "n_tr": st["n_tr"], "spearman": {m: float(spearmanr(mu, v)[0]) for m, mu in preds.items()}}


def b_walk(F, months, pair, frozen=None):
    """Monthly walk-forward on F (or, with `frozen`, one fit scored on every month of F). Returns the per-month rows."""
    rows = []; k = pair_key(pair); ok = F["ok60"] & F[f"srok_{k}"]
    for month in months:
        m0, m1 = month.start_time, (month + 1).start_time; te = O3.test_rows(F["ta"], ok, m0, m1)
        if len(te) == 0:
            continue
        st = frozen if frozen is not None else b_fit(F, m0, pair)
        if frozen is None:
            assert pd.Timestamp(st["last_label_end"]) <= m0
        rows.append(b_score(st, F, te, month))
    return rows


def b_gain(rows):
    """B1 - L1 Spearman per month"""
    return {r["month"]: r["spearman"]["B1"] - r["spearman"]["L1"] for r in rows}


# ================================================================== part C
def build_C(s, first, source, pairs=PAIRS):
    """T11's build (S3, V3 calendar, seam exclusion) plus the SR block per pair (sigma = sigma_blend) and the clean-window flags."""
    B = L11.build(s, first, source); D = B["D"]; A = D.A; sig = S.sigma_blend(B["F"]["S3"]); tmin = s.index.values.astype("datetime64[m]").astype(np.int64)
    B["SR"] = {}; B["srok"] = {}
    for p in pairs:
        k = pair_key(p); B["SR"][k] = S.sr_block(A[:, 1], A[:, 2], A[:, 3], D.anch, sig, p[0], p[1]).astype(np.float32)
        B["srok"][k] = S.clean_windows(tmin, p[0], p[1])(D.anch)
    return B


def c_pair_B(B, pair):
    return {**B, "ok": B["ok"] & B["srok"][pair_key(pair)]}


def c_fit(B, pair, m0, start=None):
    """AR baseline, V3 and C1 (T11's ridge fit) on the rows available before m0 whose 60-minute label ends by m0, from `start` (default: 365 days)."""
    k = pair_key(pair); Bp = c_pair_B(B, pair); tra, _ = L11.train_rows(Bp, m0, start=start); y = Bp["T"]["A"][tra]; X0 = L11.X_A(Bp, "V3", tra)
    return {"pair": list(pair), "ar": [float(v) for v in T01.ols1(Bp["AR"]["A"][tra], y)], "V3": L11._ridge(X0, y),
            "C1": L11._ridge(np.hstack([X0, B["SR"][k][tra].astype(np.float64)]), y), "n_tr": int(len(tra)),
            "last_label_end": str(Bp["D"].ta[tra].max() + pd.Timedelta(minutes=60)) if len(tra) else None}


def c_score(models, B, month, m_start=None):
    """One scored month for C with fitted `models`: mse of the AR baseline, V3 and C1 and skill = 1 - mse / mse_ar (T11's rows: its filter AND the clean SR window)."""
    pair = tuple(models["pair"]); k = pair_key(pair); Bp = c_pair_B(B, pair); ta = Bp["D"].ta
    m0 = month.start_time if m_start is None else max(month.start_time, m_start)
    te = np.asarray((ta >= m0) & (ta < (month + 1).start_time)); idx = np.where(te & Bp["ok"])[0]
    yt = Bp["T"]["A"][idx]; X0 = L11.X_A(Bp, "V3", idx); b = models["ar"]
    pred = {"ar": b[0] + b[1] * Bp["AR"]["A"][idx], "V3": L11._lin(models["V3"], X0), "C1": L11._lin(models["C1"], np.hstack([X0, B["SR"][k][idx].astype(np.float64)]))}
    mse = {m: float(np.mean((yt - v) ** 2)) for m, v in pred.items()}
    return {"month": str(month), "n_te": int(len(idx)), "n_tr": models["n_tr"], "mse": mse, "skill_ar": {m: 1 - mse[m] / mse["ar"] for m in mse}}


def c_month(B, pair, month, start=None):
    return c_score(c_fit(B, pair, month.start_time, start), B, month)


def c_walk(B, months, pair, start=None):
    rows = []
    for month in months:
        r = c_month(B, pair, month, start=start)
        if r["n_te"] > 0:
            rows.append(r)
    return rows


def c_gain(rows):
    return {r["month"]: r["skill_ar"]["C1"] - r["skill_ar"]["V3"] for r in rows}


# ================================================================== the selection-first rule (T10's)
def assert_selection_committed(path):
    return L10.assert_selection_committed(path)
