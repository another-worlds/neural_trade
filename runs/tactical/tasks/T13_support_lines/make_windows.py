"""T13 visual data: three real Binance windows (J2 period 2025-11..2026-09) with the S1 hull lines, S2 pivot levels and S3 regression channel of the pair chosen
for part C (selection.json: 1-minute bars, 240-bar lookback), next to H3's P(hit) and sigma (T10's H3 frozen at 2025-07-20, exactly the J2 model of part A).
Window rule, stated before the windows were looked at: of the J2 hourly anchors with a clean 240-minute window, the anchors at the 10th, 50th and 90th percentile
of H3's P(hit) (sorted by P, index round(q * (n - 1))). The drawn lines are computed here from the bars before the anchor only and are checked against sr_lib's
feature values for the same anchor. usage: python make_windows.py -> windows.json"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import pandas as pd

import sr_lib as S
import t13_lib as L

SEL = json.load(open(os.path.join(HERE, "selection.json")))
PAIR = L.pair_from_key(SEL["chosen"]["C"]); BAR, NB = PAIR
QUANTILES = (0.10, 0.50, 0.90)
LABELS = ("спокойный час", "средний час", "напряжённый час")


def upper_hull(y):
    """vertices (index, value) of the upper convex hull of the points (i, y_i)"""
    hull = []
    for i, v in enumerate(y):
        while len(hull) >= 2 and (hull[-1][0] - hull[-2][0]) * (v - hull[-2][1]) - (hull[-1][1] - hull[-2][1]) * (i - hull[-2][0]) >= 0:
            hull.pop()
        hull.append((i, float(v)))
    return hull


def geometry(h, l, c, a, sd):
    """lines of the window of NB one-minute bars ending at a-1 (BAR = 1): everything in price units except the bookkeeping, plus the sigma-unit features"""
    w = np.arange(a - NB, a); lh, ll, lc = np.log(h[w]), np.log(l[w]), np.log(c[w]); c0 = lc[-1]; m = NB - 1; x = np.arange(NB); tol = S.TOL * sd
    px = lambda v: float(np.exp(v))
    out = {}
    # S1
    uh = upper_hull(lh); lo_h = [(i, -v) for i, v in upper_hull(-ll)]
    (xj, yj), (xm, ym) = uh[-2], uh[-1]; sR = (ym - yj) / (xm - xj); (xk, yk), (xn, yn) = lo_h[-2], lo_h[-1]; sS = (yn - yk) / (xn - xk)
    Rn, Sn = ym + sR, yn + sS
    out["s1"] = {"upper": [[i, px(v)] for i, v in uh], "lower": [[i, px(v)] for i, v in lo_h], "R_now": px(Rn), "S_now": px(Sn), "xR": xj, "xS": xk}
    # S2
    k = S.K_PIV; piv_hi, piv_lo = [], []
    for i in range(k, NB - k):
        left, right = lh[i - k:i], lh[i + 1:i + k + 1]
        if lh[i] >= left.max() and lh[i] > right.max():
            piv_hi.append(i)
        left, right = ll[i - k:i], ll[i + 1:i + k + 1]
        if ll[i] <= left.min() and ll[i] < right.min():
            piv_lo.append(i)
    unb_hi = [i for i in piv_hi if lh[i + 1:].size == 0 or lh[i + 1:].max() <= lh[i]]; unb_lo = [i for i in piv_lo if ll[i + 1:].size == 0 or ll[i + 1:].min() >= ll[i]]
    out["s2"] = {"pivots_hi": [[i, px(lh[i])] for i in piv_hi], "pivots_lo": [[i, px(ll[i])] for i in piv_lo], "R": None, "S": None}
    if unb_hi:
        i = min(unb_hi, key=lambda j: lh[j] - c0); touch = [int(j) for j in range(i, NB) if abs(lh[j] - lh[i]) <= tol + 1e-12]
        out["s2"]["R"] = {"x": i, "level": px(lh[i]), "touches": touch}
    if unb_lo:
        i = min(unb_lo, key=lambda j: c0 - ll[j]); touch = [int(j) for j in range(i, NB) if abs(ll[j] - ll[i]) <= tol + 1e-12]
        out["s2"]["S"] = {"x": i, "level": px(ll[i]), "touches": touch}
    # S3
    b, a0 = np.polyfit(x, lc, 1); f = a0 + b * x; s = float(np.sqrt(((lc - f) ** 2).sum() / (NB - 2))); fn = a0 + b * NB
    out["s3"] = {"line": [px(a0), px(fn)], "upper": [px(a0 + 2 * s), px(fn + 2 * s)], "lower": [px(a0 - 2 * s), px(fn - 2 * s)], "sres_bps": float(s * 1e4)}
    # check against the library features of the same anchor
    X = dict(zip(S.SR_NAMES, S.sr_block(h, l, c, np.array([a]), np.array([sd]), BAR, NB)[0])); cl = lambda v: float(np.clip(v, -S.CAP, S.CAP))
    assert abs(X["s1_dR"] - cl((Rn - c0) / sd)) < 1e-9 and abs(X["s1_dS"] - cl((c0 - Sn) / sd)) < 1e-9
    assert abs(X["s3_dU"] - cl((fn + 2 * s - c0) / sd)) < 1e-9 and abs(X["s3_dL"] - cl((c0 - (fn - 2 * s)) / sd)) < 1e-9
    assert abs(X["s2_missR"] - (out["s2"]["R"] is None)) < 1e-9 and abs(X["s2_missS"] - (out["s2"]["S"] is None)) < 1e-9
    if out["s2"]["R"]:
        assert abs(X["s2_dR"] - cl((np.log(out["s2"]["R"]["level"]) - c0) / sd)) < 1e-9 and X["s2_touchR"] == len(out["s2"]["R"]["touches"])
    if out["s2"]["S"]:
        assert abs(X["s2_dS"] - cl((c0 - np.log(out["s2"]["S"]["level"])) / sd)) < 1e-9 and X["s2_touchS"] == len(out["s2"]["S"]["touches"])
    out["features"] = {n: float(X[n]) for n in ("s1_dR", "s1_dS", "s2_dR", "s2_dS", "s3_dU", "s3_dL", "s3_z", "s3_inside")}
    out["bars"] = {"h": [round(float(v), 2) for v in h[w]], "l": [round(float(v), 2) for v in l[w]], "c": [round(float(v), 2) for v in c[w]]}
    return out


def main():
    cut = L.guard.load_ohlcv("cut"); Pc = L.L10.prep(cut, "cut"); Bc = L.L10.build(Pc, L.T_MIN); prA = L.pair_from_key(SEL["chosen"]["A"]); Bcp = L.a_pair_B(Pc, Bc, prA)
    fr = L.a_fit(Bcp, Pc, L.T_MIN, L.guard.CUT_END + pd.Timedelta(minutes=1), L.L10.TRAIN_DAYS, prA); del cut, Pc, Bc, Bcp
    fwd = L.guard.load_ohlcv("fwd"); P = L.L10.prep(fwd, "fwd"); B = L.L10.build(P, L.T_MIN); Bp = L.a_pair_B(P, B, prA)
    clean = S.clean_windows(P.tmin, BAR, NB)(B["a"]); rows = []
    for mo in L.J2_MONTHS:
        idx, _ = L.T3.month_rows(Bp, L.T_MIN, mo); rows.append(idx[clean[idx]])
    idx = np.concatenate(rows)
    p1, sd = L.L10.h1_prob(Bp, L.T_MIN, idx, fr["har"]); pr = L.a_predict(fr, Bp, P, L.T_MIN, idx)["H3"]
    order = np.argsort(pr, kind="stable"); n = len(order); picks = [order[int(round(q * (n - 1)))] for q in QUANTILES]
    wins = []
    for q, lab, k in zip(QUANTILES, LABELS, picks):
        a = int(B["a"][idx[k]]); c0 = float(P.c[a - 1]); fut = np.maximum(np.log(P.h[a:a + 60] / c0), np.log(c0 / P.l[a:a + 60])); hit = bool(fut.max() >= 0.005)
        first = int(np.argmax(fut >= 0.005)) if hit else -1; side = 0
        if hit:
            side = 1 if np.log(P.h[a + first] / c0) >= 0.005 else -1
        g = geometry(P.h, P.l, P.c, a, float(sd[k]))
        g.update({"label": lab, "quantile": q, "anchor": str(P.idx[a]), "p_hit": float(pr[k]), "sigma_bps": float(sd[k] * 1e4), "sigma_price": float(c0 * sd[k]), "c0": c0,
                  "hit": hit, "first": first, "side": side, "fut": {"h": [round(float(v), 2) for v in P.h[a:a + 60]], "l": [round(float(v), 2) for v in P.l[a:a + 60]],
                                                                       "c": [round(float(v), 2) for v in P.c[a:a + 60]]}})
        wins.append(g)
    json.dump({"rule": "J2 hourly anchors (2025-11..2026-09) with a clean 240-minute window; anchors at the 10th / 50th / 90th percentile of H3's P(hit)",
               "pair": L.pair_key(PAIR), "n_anchors": int(n), "model": "T10 H3 (X = 50 bps, T = 60 min) fitted on the 365 days to 2025-07-20 (the J2 frozen model of part A)",
               "windows": wins}, open(os.path.join(HERE, "windows.json"), "w"))
    for w in wins:
        print(w["label"], w["anchor"], f"P(hit)={w['p_hit']:.3f} sigma={w['sigma_bps']:.1f} bps hit={w['hit']}", w["features"], "S2R", w["s2"]["R"] is not None, "S2S", w["s2"]["S"] is not None)


if __name__ == "__main__":
    main()
