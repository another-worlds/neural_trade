"""T12 (SPEC.md): T10's H3 (unchanged: ../T10_barrier_hit/t10_lib.py) plus an isotonic map fitted on the PREVIOUS calendar month's H3 predictions and
their realised labels (labels ended before the month starts), per X and T. Look A = cut file 2017-09..2018-12 (reads nothing after 2018-12-31 23:59),
Look B = frozen on the Binance file 2025-10-02..2026-09-30 (reads no bar after 2026-09-30 23:59). CPU only."""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__)); TASKS = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(TASKS, "T10_barrier_hit"))
import numpy as np
import pandas as pd

import t10_lib as L  # noqa: E402  (also puts T03 and T09 on sys.path)

T3 = L.T3; guard = L.guard
XS, TS, PRIMARY = L.XS, L.TS, L.PRIMARY
COLS = ["timestamp", "open", "high", "low", "close", "volume"]
MODELS = ("B1", "H3", "H3c")
START_A = pd.Timestamp("2017-08-17")                                       # SPEC: "everything since 2017-08-17 where less exists"
A_FIRST, A_SCORED_FIRST, A_LAST = pd.Period("2017-09", "M"), pd.Period("2017-10", "M"), pd.Period("2018-12", "M")
A_END = pd.Timestamp("2018-12-31 23:59")
CUT_START = pd.Timestamp("2017-01-01 00:00")
F_LO, F_HI = pd.Timestamp("2025-10-01 00:00"), pd.Timestamp("2026-09-30 23:59")
F_M0 = F_HI + pd.Timedelta(minutes=1)
FIRST_B = pd.Period("2026-10", "M"); SCORE_FROM_B = pd.Timestamp("2026-10-02 00:00"); MIN_MONTHS = 3
BOOT = L.BOOT


# ------------------------------------------------------------------ loading (reads exactly the rows it may use)
def load_upto(path, first_ts, last_ts, chunk=100_000):
    """Bars first_ts..last_ts from a gap-free 1-minute CSV, parsing exactly the rows up to last_ts (nrows). If the file has gaps the rows up to
    last_ts are found with a chunked reader that stops at the first chunk holding a later bar (that chunk is parsed, then cut)."""
    n = int((last_ts - first_ts) / pd.Timedelta(minutes=1)) + 1
    df = pd.read_csv(path, usecols=COLS, nrows=n); ts = pd.to_datetime(df["timestamp"])
    assert ts.iloc[0] == first_ts, (ts.iloc[0], first_ts)
    if ts.iloc[-1] < last_ts:                                              # gaps: the file is shorter than the grid, read on in chunks
        parts = []
        for ch in pd.read_csv(path, usecols=COLS, chunksize=chunk):
            ct = pd.to_datetime(ch["timestamp"]); ch.index = ct; parts.append(ch[ct <= last_ts].drop(columns="timestamp"))
            if ct.iloc[-1] > last_ts:
                break
        df = pd.concat(parts)
    else:
        df.index = ts; df = df.drop(columns="timestamp")
    df = df[(df.index >= first_ts) & (df.index <= last_ts)].sort_index()
    assert df.index.max() <= last_ts and df.index.min() >= first_ts
    guard.check_index(df.index, os.path.basename(path))
    return df


def load_lookA(path=None):
    return load_upto(path or guard.CUT_CSV, CUT_START, A_END)


def load_freeze(path=None):
    return load_upto(path or guard.FWD_CSV, F_LO, F_HI)


# ------------------------------------------------------------------ isotonic map and H3c
def fit_iso(p, y):
    """Isotonic regression of the labels on the probabilities (T10's H2 settings); returns the thresholds so the map can be stored and re-applied."""
    from sklearn.isotonic import IsotonicRegression
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p, y.astype(float))
    return np.asarray(iso.X_thresholds_, float), np.asarray(iso.y_thresholds_, float)


def apply_iso(xt, yt, p):
    return np.interp(p, xt, yt)


def calib_mask(ta_prev, T, month):
    """rows of the previous month that calibrate `month`: the label ends by the month start (T3.cal_rows' rule, `<=`)."""
    return np.asarray(ta_prev + pd.Timedelta(minutes=T) <= month.start_time)


def add_h3c(recs, T):
    """recs: per-month dicts (month, ta, y{X}, p{model}{X}) in month order. Adds p['H3c'] from the previous month's H3 predictions (None if there is no
    previous month in recs or no ended label)."""
    by = {r["month"]: r for r in recs}
    for r in recs:
        prev = by.get(r["month"] - 1); r["p"]["H3c"] = None; r["iso"] = {}
        if prev is None:
            continue
        keep = calib_mask(prev["ta"], T, r["month"])
        if keep.sum() == 0:
            continue
        out = {}
        for X in XS:
            xt, yt = fit_iso(prev["p"]["H3"][X][keep], prev["y"][X][keep]); out[X] = apply_iso(xt, yt, r["p"]["H3"][X]); r["iso"][X] = {"n": int(keep.sum())}
        r["p"]["H3c"] = out


# ------------------------------------------------------------------ Look A walk-forward
def train_days(m0, floor=START_A):
    return float(min(L.TRAIN_DAYS, (m0 - floor) / pd.Timedelta(days=1)))


def walk(B, T, months, floor=START_A):
    """Monthly walk-forward of T10's H3 (and B1): fitted at each month start on the min(365 days, since `floor`) before it (labels ended by the start)."""
    recs, info = [], []
    for month in months:
        m0 = month.start_time; idx, n_all = T3.month_rows(B, T, month); cidx = T3.cal_rows(B, T, month)
        if len(idx) == 0 or len(cidx) == 0:
            info.append({"month": str(month), "skipped": True, "n": int(len(idx)), "n_cal": int(len(cidx))}); continue
        d = train_days(m0, floor); models = L.fit_models(B, T, m0, d); pr = L.predict_rows(models, B, T, idx, cidx)
        recs.append({"month": month, "ta": B["D"].ta[idx], "y": {X: B["y"][X][idx] for X in XS}, "p": {m: {X: pr[m][X] for X in XS} for m in ("B1", "H3")}})
        tr = T3.train_rows(B, T, m0, d)
        info.append({"month": str(month), "n": int(len(idx)), "n_dropped": int(n_all - len(idx)), "n_cal": int(len(cidx)), "train_days": d, "n_train": models["n_train"],
                     "train_first": str(B["D"].ta[tr].min()) if len(tr) else None, "last_label_end": models["last_label_end"]})
    add_h3c(recs, T)
    return recs, info


def pool(recs, first=A_SCORED_FIRST, models=MODELS):
    """Rows of the scored months (month >= first, H3c available) concatenated, in the shape of T10's walk() result."""
    rs = [r for r in recs if r["month"] >= first and r["p"].get("H3c") is not None]
    cat = np.concatenate
    return {"month": np.array([str(r["month"]) for r in rs for _ in range(len(r["ta"]))]),
            "day": cat([np.asarray(r["ta"].normalize().values.astype("datetime64[D]").astype(np.int64)) for r in rs]), "ta": cat([r["ta"].values for r in rs]),
            "y": {X: cat([r["y"][X] for r in rs]) for X in XS}, "p": {m: {X: cat([r["p"][m][X] for r in rs]) for X in XS} for m in models}}


# ------------------------------------------------------------------ metrics
def ece_boot(p, y, day, idx, bins=10):
    """ECE for each bootstrap draw (rows of idx index the unique days)."""
    _, inv = np.unique(day, return_inverse=True); nd = inv.max() + 1; b = np.minimum((p * bins).astype(int), bins - 1); key = inv * bins + b
    cnt = np.bincount(key, minlength=nd * bins).reshape(nd, bins); sp = np.bincount(key, p, minlength=nd * bins).reshape(nd, bins)
    sy = np.bincount(key, y.astype(float), minlength=nd * bins).reshape(nd, bins)
    num = np.zeros(len(idx)); den = cnt.sum(1)[idx].sum(1)
    for k in range(bins):
        num += np.abs(sy[:, k][idx].sum(1) - sp[:, k][idx].sum(1))
    return num / den


def assess(R, models=MODELS, seed=0):
    """T10's summarize (Brier, skill over B1 with the day-block bootstrap CI and p, log loss, AUC, ECE, monthly) plus the ECE CI, H3c against H3 (relative Brier
    with CI), Holm over the 9 cells per model, and the three criteria in the primary cell. R = {T: pooled rows}."""
    S = {}
    for T in TS:
        idx = L.boot_index(R[T]["day"], BOOT, seed=seed); sm = L.summarize(R[T], idx, models=models)
        for X in XS:
            cell = sm[X]; y = R[T]["y"][X].astype(float)
            for m in models:
                p = R[T]["p"][m][X]; e = ece_boot(p, y, R[T]["day"], idx); cell[m]["ece_ci"] = [float(np.quantile(e, 0.025)), float(np.quantile(e, 0.975))]
            if "H3c" in models and "H3" in models:
                se_c = (R[T]["p"]["H3c"][X] - y) ** 2; se_h = (R[T]["p"]["H3"][X] - y) ** 2
                _, inv = np.unique(R[T]["day"], return_inverse=True); sc = np.bincount(inv, se_c); sh = np.bincount(inv, se_h); rel = sc[idx].sum(1) / sh[idx].sum(1) - 1
                cell["H3c"]["rel_brier_vs_H3"] = float(se_c.sum() / se_h.sum() - 1); cell["H3c"]["rel_brier_vs_H3_ci"] = [float(np.quantile(rel, 0.025)), float(np.quantile(rel, 0.975))]
            S[f"{X}|{T}"] = cell
    holm = {}
    for m in models:
        if m == "B1":
            continue
        keys = [f"{X}|{T}" for T in TS for X in XS]; ps = [S[k][m]["skill_p"] for k in keys]; adj = T3.holm(ps)
        holm[m] = {k: {"p": float(p), "p_holm": float(a)} for k, p, a in zip(keys, ps, adj)}
    return {"cells": S, "holm": holm, "verdict": verdict(S) if "H3c" in models else None}


def verdict(S):
    c = S[f"{PRIMARY[0]}|{PRIMARY[1]}"]; h = c["H3c"]; h3 = c["H3"]
    c1 = bool(h["skill_vs_B1"] >= 0.02 and h["skill_ci"][0] > 0); c2 = bool(h["ece"] < 0.01); c3 = bool(h["brier"] <= 1.01 * h3["brier"])
    return {"skill": h["skill_vs_B1"], "skill_ci": h["skill_ci"], "ece": h["ece"], "ece_ci": h["ece_ci"], "brier_h3c": h["brier"], "brier_h3": h3["brier"],
            "rel_brier_vs_H3": h["rel_brier_vs_H3"], "c1_skill": c1, "c2_ece": c2, "c3_not_worse": c3, "pass": bool(c1 and c2 and c3)}


# ------------------------------------------------------------------ the frozen artefact (Look B)
def sha_of(obj):
    body = {k: v for k, v in obj.items() if k != "sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def h3_probs(models, B, T, idx):
    """T10's H3 on rows idx from fitted/restored `models` (same arithmetic as t10_lib.predict_rows)."""
    p1, _ = L.h1_prob(B, T, idx, models["har"]); out = {}
    for X in XS:
        h3 = models["h3"][X]
        if h3 is None:
            out[X] = p1[X]; continue
        mu, sd, coef, ic = h3; z = ((L.h3_design(B, idx, p1[X]) - mu) / sd) @ coef + ic; out[X] = 1 / (1 + np.exp(-z))
    return out


def _lst(a):
    return [float(v) for v in np.asarray(a).ravel()]


def models_from_art(art, T):
    h = art["har"][str(T)]; har = {"mu": np.array(h["mu"]), "sd": np.array(h["sd"]), "coef": np.array(h["coef"]), "ic": h["ic"], "c": h["c"], "plus": h["plus"]}
    h3 = {}
    for X in XS:
        d = art["h3"][str(T)][str(X)]
        h3[X] = None if d is None else (np.array(d["mu"]), np.array(d["sd"]), np.array(d["coef"]), d["ic"])
    return {"har": har, "h3": h3}


def freeze(bars, m0=F_M0, days=L.TRAIN_DAYS):
    """Fit T10's HAR and H3 ONCE per T on the `days` before m0 (labels ended by m0) and the isotonic map for the first scored month from the previous
    month's frozen-H3 predictions (labels ended by m0). Returns the artefact with its sha256. Reads no bar at or after m0."""
    assert bars.index.max() < m0, "a bar at or after the freeze cut"
    P = L.prep(bars, "fwd"); first = pd.Period(m0, "M")
    art = {"spec": "T12 SPEC.md Look B freeze", "m0": str(m0), "first_month": str(first), "last_bar": str(bars.index.max()), "n_bars": int(len(bars)),
           "train_days": days, "har": {}, "h3": {}, "iso_first": {}, "n_train": {}, "train_label_end_max": {}}
    for T in TS:
        B = L.build(P, T); models = L.fit_models(B, T, m0, days); h = models["har"]
        art["har"][str(T)] = {"mu": _lst(h["mu"]), "sd": _lst(h["sd"]), "coef": _lst(h["coef"]), "ic": float(h["ic"]), "c": float(h["c"]), "plus": bool(h["plus"])}
        art["h3"][str(T)] = {str(X): (None if models["h3"][X] is None else {"mu": _lst(models["h3"][X][0]), "sd": _lst(models["h3"][X][1]),
                                                                            "coef": _lst(models["h3"][X][2]), "ic": float(models["h3"][X][3])}) for X in XS}
        art["n_train"][str(T)] = models["n_train"]; art["train_label_end_max"][str(T)] = models["last_label_end"]
        cidx = T3.cal_rows(B, T, first); p = h3_probs(models_from_art(art, T), B, T, cidx); art["iso_first"][str(T)] = {}
        for X in XS:
            xt, yt = fit_iso(p[X], B["y"][X][cidx])
            art["iso_first"][str(T)][str(X)] = {"x": _lst(xt), "y": _lst(yt), "n": int(len(cidx)), "month": str(first - 1)}
    art["sha256"] = sha_of(art)
    return art


# ------------------------------------------------------------------ Look B scoring (frozen; no refit)
def complete_months(last_bar, first=FIRST_B):
    out = []; m = first
    while (m + 1).start_time + pd.Timedelta(minutes=max(TS) - 1) <= last_bar:     # the last 240-minute label of the month is in the file
        out.append(m); m = m + 1
    return out


def score(art, bars, months=None, score_from=SCORE_FROM_B, min_months=MIN_MONTHS):
    """Score the frozen H3 / H3c / B1 on the complete calendar months from art['first_month'] (anchors of the first month from `score_from`).
    H3c: the stored map for the first month, then each month's map from the previous month's frozen-H3 predictions (labels ended by the month start).
    Returns the result with the verdict (None with fewer than min_months months)."""
    first = pd.Period(art["first_month"], "M"); months = complete_months(bars.index.max(), first) if months is None else months
    out = {"n_months": len(months), "months": [str(m) for m in months]}
    if len(months) < min_months:
        return {**out, "verdict": None, "status": f"withheld: {len(months)} complete months, need {min_months}"}
    P = L.prep(bars, "fwd"); R = {}
    for T in TS:
        B = L.build(P, T); models = models_from_art(art, T); recs = []
        for month in months:
            idx, _ = T3.month_rows(B, T, month)
            if month == first:
                idx = idx[np.asarray(B["D"].ta[idx] >= score_from)]
            if len(idx) == 0:
                continue
            ph = h3_probs(models, B, T, idx); pc = {}
            if month == first:
                for X in XS:
                    m = art["iso_first"][str(T)][str(X)]; pc[X] = apply_iso(np.array(m["x"]), np.array(m["y"]), ph[X])
            else:
                cidx = T3.cal_rows(B, T, month); pcal = h3_probs(models, B, T, cidx)
                for X in XS:
                    xt, yt = fit_iso(pcal[X], B["y"][X][cidx]); pc[X] = apply_iso(xt, yt, ph[X])
            recs.append({"month": month, "ta": B["D"].ta[idx], "y": {X: B["y"][X][idx] for X in XS},
                         "p": {"B1": {X: L.fp_from_total_sd(B["trail"][T][idx], X, T) for X in XS}, "H3": ph, "H3c": pc}})
        R[T] = pool(recs, first=first)
    res = assess(R); res.update(out); res["status"] = "verdict"; res["frozen_sha256"] = art["sha256"]
    return res
