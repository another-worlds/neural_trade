"""T14 harness, part 3: the identical test of one candidate view against V3 (target A) and H3 (target B), on Look 1 and Look 2.

    python evaluate.py --view M1_chronos2 [--looks look1,look2] [--c2-view NAME] [--out DIR] [--n-perm 5] [--n-boot 10000]
    python evaluate.py --baselines          # the reference numbers of V3 and H3 alone (no view)

Reads D:/nt/nt_t14_cache/view_{name}_{cut|fwd}.parquet (view_cache.py contract) and the anchors / base arrays (anchors.py). Writes into DIR
(default runs/tactical/tasks/T14_multimodal/<view>/): results.json, tables.txt, fig_increments.png, fig_increments.json (plotly figure JSON).

Target A (T01 definition, T11 V3 = S3 + 96 UTC 15-minute slots): ridge (alpha 1, standardised; T11's `_ridge` / `_lin`) on V3's columns, with the view's
columns appended or not; monthly refit on the 365 days before the month (look 2: also not before 2025-10-02) with labels ended before the month;
skill = 1 - MSE / MSE of the AR(1) of T01 fitted on the same rows; increment = skill(V3 + view) - skill(V3) per month; monthly t-interval.
Target B (T10 H3 at X = 50 bps, T = 60 minutes): logistic regression (C 1, standardised) on [logit(first passage of the HAR sd), the 10 extra S3
columns, 4 hour harmonics] (T10's `h3_design`), view columns appended or not; same refit rule; pooled Brier, relative improvement
1 - Brier(H3 + view) / Brier(H3), CI by the day-block bootstrap of T10 (10,000 draws, the same draws for every comparison).
Controls (same rows, same refit): C1 view columns permuted within the calendar month (n_perm seeds; the mean is the control), C2a information-equivalent
numeric control: the same window's S3 numeric columns (A: the 16 S3 columns; B: the 6 HAR + 10 extra columns) duplicated, plus a monotone nonlinear
transform sign(x) ln(1 + |x|) of them; C2b an optional second view supplied by the candidate (`--c2-view`, e.g. kNN on the same keys with shuffled
residuals). When the view has `point_log_rv`: the 50/50 average of V3's change forecast and `point_log_rv - LV0`.
Pre-registered flag per target and look (SPEC): A: increment >= +0.005, t-CI lower bound > 0, and the mean increment above C1's and C2's; B: relative
Brier improvement >= 1%, bootstrap CI lower bound > 0, above C1 and C2. "Above" is the point comparison of the means (the paired CI of the
difference is reported as well). The candidate passes a target when its flag holds on BOTH looks, and passes when it passes either target. Holm over
6 candidates x 2 targets is applied by the lead over the one-sided p-values written here (`holm_all`).
"""
import argparse
import json
import os
import subprocess
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anchors as AN  # noqa: E402
import view_cache as VC  # noqa: E402
from _libs import T01, T10, T11, T3, T14, ROOT  # noqa: E402

TRAIN_DAYS = 365
MIN_TRAIN, MIN_TEST = 300, 100
MIN_INC_A, MIN_REL_B = 0.005, 0.01
X_B, H_B = AN.X_B, AN.H_B


# ------------------------------------------------------------------ rows
def train_mask(ta, valid, m0, start=None, H=60, days=TRAIN_DAYS):
    """rows available before m0 (the T11 / T03 rule): not older than `days` (nor than `start`), label (H minutes) ended by m0, valid."""
    lo = m0 - pd.Timedelta(days=days)
    if start is not None:
        lo = max(lo, pd.Timestamp(start))
    return np.asarray((ta >= lo) & (ta + pd.Timedelta(minutes=H) <= m0)) & valid


def month_mask(ta, valid, month):
    m0, m1 = month.start_time, (month + 1).start_time
    return np.asarray((ta >= m0) & (ta < m1)) & valid


def mono(x):
    return np.sign(x) * np.log1p(np.abs(x))


def with_ok(base, ok):
    """Shallow copies of base.A / base.B whose `ok` is `ok` (T11 / T03 row filters read B['ok'])."""
    A = dict(base.A); A["ok"] = ok; B = dict(base.B); B["ok"] = ok
    return A, B


def permute_within_month(M, ta, valid, seed):
    """C1: each column's values shuffled among the valid rows of the same calendar month (independently per column)."""
    out = M.copy(); rng = np.random.default_rng(seed); per = np.asarray(ta.to_period("M").astype(str))
    for mo in np.unique(per):
        idx = np.where((per == mo) & valid)[0]
        for j in range(M.shape[1]):
            out[idx, j] = M[rng.permutation(idx), j]
    return out


# ------------------------------------------------------------------ target A
def run_A(base, valid, months, start, variants, point=None, keep_preds=False):
    """Monthly walk-forward of V3 (extra None) and V3 + each extra matrix in `variants` {name: [N, k] or None}. `point`: optional [N] point forecast
    of the next hour's log RV for the 50/50 average. Returns one dict per month: n_tr, n_te, mse, skill (all models incl. 'ar'), preds optional."""
    A = base.A; ta = base.ta; N = len(ta); V3 = T11.X_A(A, "V3", np.arange(N)); y = A["T"]["A"]; arx = A["AR"]["A"]; rows = []
    for mo in months:
        tr = np.where(train_mask(ta, valid, mo.start_time, start))[0]; te = np.where(month_mask(ta, valid, mo))[0]
        if len(tr) < MIN_TRAIN or len(te) < MIN_TEST:
            rows.append({"month": str(mo), "skipped": True, "n_tr": int(len(tr)), "n_te": int(len(te))}); continue
        ar = T01.ols1(arx[tr], y[tr]); preds = {"ar": ar[0] + ar[1] * arx[te]}
        for nm, ex in variants.items():
            X = V3 if ex is None else np.hstack([V3, ex])
            preds[nm] = T11._lin(T11._ridge(X[tr], y[tr]), X[te])
        if point is not None:
            preds["avg"] = 0.5 * preds["V3"] + 0.5 * (point[te] - A["LV0"][te])
        yt = y[te]; mse = {k: float(np.mean((yt - p) ** 2)) for k, p in preds.items()}
        rows.append({"month": str(mo), "skipped": False, "n_tr": int(len(tr)), "n_te": int(len(te)), "mse": mse,
                     "skill": {k: 1 - mse[k] / mse["ar"] for k in mse}, **({"preds": preds, "te": te, "tr": tr} if keep_preds else {})})
    return rows


def summarize_series(x):
    """mean, 95% t-interval, t, one-sided p (mean > 0), months positive of a monthly series."""
    x = np.asarray(x, float); n = len(x)
    if n < 2:
        return {"mean": float(x.mean()) if n else None, "ci": [None, None], "t": None, "p_one_sided": None, "positive": int((x > 0).sum()), "n_months": n}
    mu, lo, hi, t = T01.ci(x)
    return {"mean": mu, "ci": [lo, hi], "t": t, "p_one_sided": float(1 - stats.t.cdf(t, n - 1)), "positive": int((x > 0).sum()), "n_months": n}


def summarize_A(rows, names, ctrl_groups):
    """Increments over V3 per month for each name; controls with several seeds are averaged per month (`ctrl_groups` {group: [names]})."""
    ok = [r for r in rows if not r["skipped"]]; out = {"months": [r["month"] for r in ok]}
    skill = {k: [r["skill"][k] for r in ok] for k in ok[0]["skill"]}
    out["skill"] = {k: summarize_series(v) for k, v in skill.items()}
    inc = {k: np.array(skill[k]) - np.array(skill["V3"]) for k in skill if k not in ("V3", "ar")}
    series = {k: v for k, v in inc.items() if k in names}
    for g, members in ctrl_groups.items():
        series[g] = np.mean([inc[m] for m in members], axis=0); series[g + "_max"] = np.max([inc[m] for m in members], axis=0)
    out["increment_series"] = {k: [float(v) for v in vs] for k, vs in series.items()}
    out["increment"] = {k: summarize_series(v) for k, v in series.items() if not k.endswith("_max")}
    for k in series:
        if k.endswith("_max"):
            out["increment"][k] = {"mean": float(np.mean(series[k]))}
    return out


# ------------------------------------------------------------------ target B
def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def run_B(base, valid, months, start, variants):
    """Monthly walk-forward of H3 (extra None) and H3 + extra columns for the X = 50 bps, T = 60 min barrier. Returns pooled arrays."""
    B = base.B; ta = base.ta; y_all = B["y"][X_B]; ps = {k: [] for k in variants}; pH1 = []; ys = []; day = []; mon = []; info = []
    for mo in months:
        tr = np.where(train_mask(ta, valid, mo.start_time, start, H=H_B))[0]; te = np.where(month_mask(ta, valid, mo))[0]
        if len(tr) < MIN_TRAIN or len(te) < MIN_TEST:
            info.append({"month": str(mo), "skipped": True, "n_tr": int(len(tr)), "n_te": int(len(te))}); continue
        har = T3.fit_har(B, H_B, tr, None, False)
        p1_tr = T10.fp_from_total_sd(T10.har_sd(har, B, H_B, tr), X_B, H_B); p1_te = T10.fp_from_total_sd(T10.har_sd(har, B, H_B, te), X_B, H_B)
        Ztr = T10.h3_design(B, tr, p1_tr); Zte = T10.h3_design(B, te, p1_te); ytr = y_all[tr]
        for nm, ex in variants.items():
            if ytr.min() == ytr.max():
                p = p1_te
            else:
                a = Ztr if ex is None else np.hstack([Ztr, ex[tr]]); b = Zte if ex is None else np.hstack([Zte, ex[te]])
                mu, sd = a.mean(0), a.std(0) + 1e-12
                from sklearn.linear_model import LogisticRegression
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    lr = LogisticRegression(C=1.0, max_iter=2000).fit((a - mu) / sd, ytr)
                p = _sigmoid(((b - mu) / sd) @ lr.coef_[0] + float(lr.intercept_[0]))
            ps[nm].append(p)
        pH1.append(p1_te); ys.append(y_all[te].astype(float)); mon.append(np.full(len(te), str(mo)))
        day.append(np.asarray(ta[te].normalize().values.astype("datetime64[D]").astype(np.int64)))
        info.append({"month": str(mo), "skipped": False, "n_tr": int(len(tr)), "n_te": int(len(te))})
    cat = np.concatenate
    return {"p": {k: cat(v) for k, v in ps.items()}, "h1": cat(pH1), "y": cat(ys), "day": cat(day), "month": cat(mon), "info": info}


def summarize_B(R, names, ctrl_groups, n_boot, seed=0):
    y = R["y"]; se = {k: (p - y) ** 2 for k, p in R["p"].items()}; base = se["H3"]; idx = T10.boot_index(R["day"], n_boot, seed=seed)
    months = sorted(set(R["month"].tolist())); out = {"months": months, "n": int(len(y)), "base_rate": float(y.mean())}

    def rel(s):
        return float(1 - s.sum() / base.sum())

    def monthly(s):
        return [float(1 - s[R["month"] == m].sum() / base[R["month"] == m].sum()) for m in months]

    out["brier"] = {k: float(s.mean()) for k, s in se.items()}
    out["brier_h1"] = float(((R["h1"] - y) ** 2).mean())
    from sklearn.metrics import roc_auc_score
    out["auc"] = {k: (float(roc_auc_score(y, p)) if y.min() != y.max() else None) for k, p in R["p"].items() if k in ("H3",) + tuple(names)}
    out["rel_improvement"] = {}; out["monthly_rel"] = {}
    for k in names:
        bs = T10.day_boot_skill(se[k], base, R["day"], idx)
        out["rel_improvement"][k] = {"mean": rel(se[k]), "ci": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
                                     "p_one_sided": float((np.sum(bs <= 0) + 1) / (len(bs) + 1))}
        out["monthly_rel"][k] = monthly(se[k])
    for g, members in ctrl_groups.items():
        vals = [rel(se[m]) for m in members]
        out["rel_improvement"][g] = {"mean": float(np.mean(vals)), "max": float(np.max(vals)), "per_seed": [float(v) for v in vals]}
        out["monthly_rel"][g] = [float(v) for v in np.mean([monthly(se[m]) for m in members], axis=0)]
    # paired CI of the difference (candidate - control), same bootstrap draws
    out["diff_vs_control"] = {}
    for k in names:
        if k == "view":
            for g in list(ctrl_groups) + [n for n in names if n.startswith("c2")]:
                if g == "view":
                    continue
                ms = ctrl_groups[g] if g in ctrl_groups else [g]
                se_c = np.mean([se[m] for m in ms], axis=0)
                bv = T10.day_boot_skill(se["view"], base, R["day"], idx); bc = T10.day_boot_skill(se_c, base, R["day"], idx); d = bv - bc
                out["diff_vs_control"][g] = {"mean": float(rel(se["view"]) - (1 - se_c.sum() / base.sum())),
                                             "ci": [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]}
    return out


# ------------------------------------------------------------------ one look
def _flags_A(inc, cands):
    v = inc["view"]; f = {"inc_ge_0.005": bool(v["mean"] is not None and v["mean"] >= MIN_INC_A), "ci_lo_gt_0": bool(v["ci"][0] is not None and v["ci"][0] > 0)}
    for c in cands:
        f[f"above_{c}"] = bool(v["mean"] is not None and v["mean"] > inc[c]["mean"])
    f["pass"] = bool(all(f.values())); return f


def _flags_B(rel, cands):
    v = rel["view"]; f = {"rel_ge_1pct": bool(v["mean"] >= MIN_REL_B), "ci_lo_gt_0": bool(v["ci"][0] > 0)}
    for c in cands:
        f[f"above_{c}"] = bool(v["mean"] > rel[c]["mean"])
    f["pass"] = bool(all(f.values())); return f


def evaluate_look(base, view_df, months, start, c2_df=None, n_perm=5, n_boot=10000, seed=0, log=None):
    """Both targets on one look. `view_df`: the candidate's DataFrame (index = anchor time); `c2_df`: optional candidate-specific control view."""
    t0 = time.time(); ta = base.ta; M, cols = VC.align_view(view_df, ta); finite = np.isfinite(M).all(1)
    valid = base.ok & finite
    if valid.sum() == 0:
        raise ValueError("no anchor has a finite view row")
    cnt = {"anchors": int(len(ta)), "base_ok": int(base.ok.sum()), "view_rows_finite": int(finite.sum()), "valid_both": int(valid.sum()),
           "dropped_for_view": int((base.ok & ~finite).sum()), "columns": cols}
    c2b = None
    if c2_df is not None:
        M2, cols2 = VC.align_view(c2_df, ta); valid = valid & np.isfinite(M2).all(1); c2b = M2; cnt["c2b_columns"] = cols2; cnt["valid_both"] = int(valid.sum())
    c1 = {f"c1_s{k}": permute_within_month(M, ta, valid, 1000 + k) for k in range(n_perm)}
    SA = base.A["F"]["S3"]; SB = np.hstack([base.B["D"].har, base.B["ex10"]])
    pt = M[:, cols.index(VC.POINT_COL)] if VC.POINT_COL in cols else None
    out = {"counts": cnt, "months": [str(m) for m in months], "start": str(start)}
    # ---- target A
    vA = {"V3": None, "view": M, "c2a": np.hstack([SA, mono(SA)]), **c1}
    if c2b is not None:
        vA["c2b"] = c2b
    rowsA = run_A(base, valid, months, start, vA, point=pt)
    names = ["view", "c2a"] + (["c2b"] if c2b is not None else []) + (["avg"] if pt is not None else [])
    grp = {"c1": list(c1)}
    sA = summarize_A(rowsA, names, grp)
    cands = ["c1", "c2a"] + (["c2b"] if c2b is not None else [])
    sA["flags"] = _flags_A(sA["increment"], cands)
    sA["per_month"] = [{k: v for k, v in r.items()} for r in rowsA]
    d = {}
    for c in cands:
        a = np.array(sA["increment_series"]["view"]); b = np.array(sA["increment_series"][c])
        d[c] = summarize_series(a - b)
    sA["diff_vs_control"] = d
    out["A"] = sA
    if log:
        log(f"  A done {time.time() - t0:.0f}s")
    # ---- target B
    vB = {"H3": None, "view": M, "c2a": np.hstack([SB, mono(SB)]), **c1}
    if c2b is not None:
        vB["c2b"] = c2b
    R = run_B(base, valid, months, start, vB)
    namesB = ["view", "c2a"] + (["c2b"] if c2b is not None else [])
    sB = summarize_B(R, namesB, grp, n_boot, seed)
    sB["flags"] = _flags_B(sB["rel_improvement"], cands)
    sB["info"] = R["info"]
    out["B"] = sB
    out["_B_raw"] = R
    if log:
        log(f"  B done {time.time() - t0:.0f}s")
    return out


# ------------------------------------------------------------------ the baselines alone (reference numbers) and their equality with T11 / T10
def baselines_look(base, months, start, n_boot=10000):
    """V3 and H3 alone on every ok anchor: V3 skill vs AR per month (mean, CI), H3 / H1 / B1 / B0 Brier and skill; plus the maximum absolute
    difference to T11's own fit / predict_A (V3) and T10's own fit_models / predict_rows (H3) on the same rows."""
    ta = base.ta; ok = base.ok; A, B = with_ok(base, ok)
    rowsA = run_A(base, ok, months, start, {"V3": None})
    sk = [r["skill"]["V3"] for r in rowsA if not r["skipped"]]
    RB = run_B(base, ok, months, start, {"H3": None})
    y = RB["y"]; se_h3 = (RB["p"]["H3"] - y) ** 2; se_h1 = (RB["h1"] - y) ** 2
    # B1 (trailing sd) and B0 (previous month base rate) for the same rows; T10's definitions
    p_b1 = []; p_b0 = []
    for mo in months:
        tr = np.where(train_mask(ta, ok, mo.start_time, start, H=H_B))[0]; te = np.where(month_mask(ta, ok, mo))[0]
        if len(tr) < MIN_TRAIN or len(te) < MIN_TEST:
            continue
        p_b1.append(T10.fp_from_total_sd(B["trail"][H_B][te], X_B, H_B))
        cidx = T3.cal_rows(B, H_B, mo); p_b0.append(np.full(len(te), B["y"][X_B][cidx].mean()))
    p_b1 = np.concatenate(p_b1); p_b0 = np.concatenate(p_b0); se_b1 = (p_b1 - y) ** 2; se_b0 = (p_b0 - y) ** 2
    idx = T10.boot_index(RB["day"], n_boot, seed=0); bs = T10.day_boot_skill(se_h3, se_b1, RB["day"], idx)
    # equality checks on the first scored month
    mo = months[min(2, len(months) - 1)]; m0 = mo.start_time
    tra, trr = T11.train_rows(A, m0, start=None)
    mod = T11.fit(A, tra, trr, ["V3"]); te = np.where(month_mask(ta, ok, mo))[0]
    p11 = T11.predict_A(mod, A, te)["V3"]
    X = T11.X_A(A, "V3", np.arange(len(ta))); trm = np.where(train_mask(ta, ok, m0, start))[0]
    mine = T11._lin(T11._ridge(X[trm], A["T"]["A"][trm]), X[te])
    dA = float(np.max(np.abs(p11 - mine))) if np.array_equal(trm, tra) else float("nan")
    rows10 = T3.train_rows(B, H_B, m0, TRAIN_DAYS)
    ms = T10.fit_models(B, H_B, m0, TRAIN_DAYS); cidx = T3.cal_rows(B, H_B, mo)
    p10 = T10.predict_rows(ms, B, H_B, te, cidx)["H3"][X_B]
    one = run_B(base, ok, [mo], start, {"H3": None})
    dB = float(np.max(np.abs(one["p"]["H3"] - p10))) if np.array_equal(trm, rows10) else float("nan")
    from sklearn.metrics import roc_auc_score
    return {"n_ok_anchors": int(ok.sum()), "months": [r["month"] for r in rowsA if not r["skipped"]],
            "A": {"V3_skill_vs_AR": summarize_series(sk), "monthly": [float(v) for v in sk], "n_te": [r["n_te"] for r in rowsA if not r["skipped"]]},
            "B": {"n": int(len(y)), "base_rate": float(y.mean()),
                  "brier": {"H3": float(se_h3.mean()), "H1": float(se_h1.mean()), "B1": float(se_b1.mean()), "B0": float(se_b0.mean())},
                  "H3_skill_vs_B1": {"mean": float(1 - se_h3.sum() / se_b1.sum()), "ci": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]},
                  "H3_skill_vs_H1": float(1 - se_h3.sum() / se_h1.sum()), "H3_skill_vs_B0": float(1 - se_h3.sum() / se_b0.sum()),
                  "auc_H3": float(roc_auc_score(y, RB["p"]["H3"]))},
            "equality": {"month": str(mo), "V3_max_abs_diff_vs_T11": dA, "H3_max_abs_diff_vs_T10": dB}}


# ------------------------------------------------------------------ output
def _git_head():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=T14, capture_output=True, text=True).stdout.strip()
    except Exception:
        return ""


def theme_colors():
    """Colours from src/neural_trade/visualization/theme.py (loaded by file path so no TF / plotly import happens)."""
    import importlib.util
    p = os.path.join(ROOT, "src", "neural_trade", "visualization", "theme.py")
    spec = importlib.util.spec_from_file_location("nt_theme", p); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def _lines(res):
    """(label, key, colour index) for the figure."""
    return [("candidate view", "view", 0), ("C1 permuted (mean)", "c1", 1), ("C2a S3 duplicated", "c2a", 2), ("C2b candidate control", "c2b", 3),
            ("50/50 with V3", "avg", 4)]


def make_figure(res, out_dir, name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    th = theme_colors(); col = [th.OTHER_SERIES[0], th.NEUTRAL, th.OTHER_SERIES[1], th.OTHER_SERIES[3], th.OTHER_SERIES[4]]
    looks = [k for k in ("look1", "look2") if k in res["looks"]]
    fig, axs = plt.subplots(2, len(looks), figsize=(7.2 * len(looks), 8.2), facecolor=th.PAPER, squeeze=False)
    pl = {"data": [], "layout": {"paper_bgcolor": th.PAPER, "plot_bgcolor": th.SURFACE, "font": {"color": th.INK, "family": th.FONT},
                                 "title": {"text": f"T14 {name}: monthly increment over V3 (A) and relative Brier improvement over H3 (B)"}, "height": 820, "annotations": []}}
    nx = len(looks)
    for j, lk in enumerate(looks):
        for i, tg in enumerate(("A", "B")):
            ax = axs[i][j]; ax.set_facecolor(th.SURFACE)
            r = res["looks"][lk][tg]; ser = r["increment_series"] if tg == "A" else r["monthly_rel"]
            months = r["months"]; xs = pd.PeriodIndex(months, freq="M").to_timestamp()
            ax_id = i * nx + j + 1; sfx = "" if ax_id == 1 else str(ax_id)
            for lab, key, ci in _lines(res):
                if key not in ser:
                    continue
                y = ser[key]
                ax.plot(xs, y, marker="o", ms=4, lw=1.6, color=col[ci], label=lab)
                pl["data"].append({"type": "scatter", "mode": "lines+markers", "name": lab, "x": [str(x.date()) for x in xs], "y": [float(v) for v in y],
                                   "line": {"color": col[ci]}, "xaxis": "x" + sfx, "yaxis": "y" + sfx, "legendgroup": lab, "showlegend": bool(i == 0 and j == 0)})
            thr = MIN_INC_A if tg == "A" else MIN_REL_B
            ax.axhline(0, color=th.NEUTRAL, lw=0.8); ax.axhline(thr, color=th.GOOD, lw=0.8, ls="--")
            ttl = f"{lk}: target {'A (skill increment over V3)' if tg == 'A' else 'B (rel. Brier improvement over H3)'}   flag pass={r['flags']['pass']}"
            ax.set_title(ttl, color=th.INK, fontsize=9)
            ax.set_ylabel("skill vs AR: increment over V3" if tg == "A" else "relative Brier improvement over H3", color=th.INK_2, fontsize=8)
            ax.tick_params(colors=th.MUTED, labelsize=8); ax.grid(color=th.GRID, lw=0.5)
            for sp in ax.spines.values():
                sp.set_color(th.AXIS)
            if i == 0 and j == 0:
                ax.legend(fontsize=7, facecolor=th.SURFACE, edgecolor=th.AXIS, labelcolor=th.INK_2)
            dx = 1.0 / nx; dy = 0.5
            pl["layout"]["xaxis" + sfx] = {"domain": [j * dx + 0.04 * dx, (j + 1) * dx - 0.02 * dx], "anchor": "y" + sfx, "gridcolor": th.GRID}
            pl["layout"]["yaxis" + sfx] = {"domain": [(1 - i) * dy + 0.07, (2 - i) * dy - 0.07], "anchor": "x" + sfx, "gridcolor": th.GRID, "zerolinecolor": th.AXIS}
            pl["layout"]["annotations"].append({"text": ttl, "x": (j + 0.5) * dx, "y": (2 - i) * dy - 0.04, "xref": "paper", "yref": "paper", "showarrow": False,
                                                "font": {"size": 11, "color": th.INK_2}})
    fig.suptitle(f"T14 {name}: monthly increments over V3 (A) / H3 (B), with controls (dashed green = pre-registered threshold)", color=th.INK, fontsize=10)
    fig.tight_layout()
    png = os.path.join(out_dir, "fig_increments.png"); fig.savefig(png, dpi=130, facecolor=th.PAPER); plt.close(fig)
    json.dump(pl, open(os.path.join(out_dir, "fig_increments.json"), "w"))
    return png


def _fmt(v, p=4, pct=False):
    if v is None:
        return "n/a"
    return f"{100 * v:+.2f}%" if pct else f"{v:+.{p}f}"


def write_tables(res, path):
    L = [f"T14 harness: view {res['view']}  (harness commit {res['head']}, view sha256 {res['view_sha256']})", f"columns: {res['columns']}", ""]
    for lk, r in res["looks"].items():
        c = r["counts"]
        L += [f"=== {lk}: months {r['months'][0]}..{r['months'][-1]}, train start {r['start']} ===",
              f"anchors {c['anchors']}, base ok {c['base_ok']}, view rows finite {c['view_rows_finite']}, used {c['valid_both']}, dropped for the view {c['dropped_for_view']}"]
        a = r["A"]
        L.append("-- target A: skill vs AR(1) (T01), monthly; increment over V3")
        L.append(f"{'month':8s} {'n_te':>6s} {'V3':>9s} {'V3+view':>9s} {'inc':>9s} {'C1':>9s} {'C2a':>9s}" + (f" {'C2b':>9s}" if "c2b" in a["increment"] else "") + (f" {'avg':>9s}" if "avg" in a["increment"] else ""))
        ok = [m for m in a["per_month"] if not m["skipped"]]
        for k, m in enumerate(ok):
            s = a["increment_series"]
            row = f"{m['month']:8s} {m['n_te']:6d} {m['skill']['V3']:+9.4f} {m['skill']['view']:+9.4f} {s['view'][k]:+9.4f} {s['c1'][k]:+9.4f} {s['c2a'][k]:+9.4f}"
            if "c2b" in s:
                row += f" {s['c2b'][k]:+9.4f}"
            if "avg" in s:
                row += f" {s['avg'][k]:+9.4f}"
            L.append(row)
        for m in a["per_month"]:
            if m["skipped"]:
                L.append(f"{m['month']:8s} skipped (n_tr {m['n_tr']}, n_te {m['n_te']})")
        for k, v in a["increment"].items():
            if "ci" in v:
                L.append(f"  increment {k:8s} mean {_fmt(v['mean'])}  95% t-CI [{_fmt(v['ci'][0])}, {_fmt(v['ci'][1])}]  months positive {v['positive']}/{v['n_months']}  p1 {v['p_one_sided']:.4f}")
        L.append(f"  skill V3 {_fmt(a['skill']['V3']['mean'])}  skill V3+view {_fmt(a['skill']['view']['mean'])}  skill AR 0 by definition; skill 'avg' " + (_fmt(a['skill']['avg']['mean']) if 'avg' in a['skill'] else 'n/a'))
        for k, v in a["diff_vs_control"].items():
            L.append(f"  candidate minus {k}: mean {_fmt(v['mean'])} CI [{_fmt(v['ci'][0])}, {_fmt(v['ci'][1])}]")
        L.append(f"  FLAGS A: {a['flags']}")
        b = r["B"]
        L += ["", f"-- target B: P(max |move| >= {X_B} bps within {H_B} min); n {b['n']}, base rate {b['base_rate']:.4f}",
              f"  Brier: " + ", ".join(f"{k} {v:.5f}" for k, v in b["brier"].items() if not k.startswith("c1_s")) + f", H1 {b['brier_h1']:.5f}",
              f"  AUC: " + ", ".join(f"{k} {v:.4f}" for k, v in b["auc"].items() if v is not None)]
        for k, v in b["rel_improvement"].items():
            if "ci" in v:
                L.append(f"  rel. Brier improvement over H3, {k:6s}: {_fmt(v['mean'], pct=True)}  95% day-block CI [{_fmt(v['ci'][0], pct=True)}, {_fmt(v['ci'][1], pct=True)}]  p1 {v['p_one_sided']:.4f}")
            else:
                L.append(f"  rel. Brier improvement over H3, {k:6s}: mean {_fmt(v['mean'], pct=True)} (max over seeds {_fmt(v['max'], pct=True)})")
        for k, v in b["diff_vs_control"].items():
            L.append(f"  candidate minus {k}: {_fmt(v['mean'], pct=True)} CI [{_fmt(v['ci'][0], pct=True)}, {_fmt(v['ci'][1], pct=True)}]")
        L.append("  monthly rel. improvement: " + ", ".join(f"{m} {_fmt(x, pct=True)}" for m, x in zip(b["months"], b["monthly_rel"]["view"])))
        L.append(f"  FLAGS B: {b['flags']}")
        L.append("")
    p = res["pass"]
    L += [f"PRE-REGISTERED PASS: A both looks {p['A']}, B both looks {p['B']}, candidate {p['candidate']}",
          f"one-sided p (max over the two looks): A {res['p_one_sided']['A']}, B {res['p_one_sided']['B']}  (Holm over 6 x 2 is the lead's)"]
    open(path, "w", encoding="utf-8").write("\n".join(L) + "\n")


def evaluate_view(name, looks=("look1", "look2"), c2_name=None, out_dir=None, n_perm=5, n_boot=10000, seed=0, figure=True, log=print):
    t0 = time.time(); out_dir = out_dir or os.path.join(T14, name); os.makedirs(out_dir, exist_ok=True)
    res = {"view": name, "head": _git_head(), "c2_view": c2_name, "looks": {}, "n_perm": n_perm, "n_boot": n_boot}
    shas = {}
    for lk in looks:
        cfg = AN.LOOKS[lk]; src = cfg["source"]; base = AN.load_base(src); view = VC.read_view(name, src); shas[src] = VC.view_sha(name, src)
        c2 = VC.read_view(c2_name, src) if c2_name else None
        res["columns"] = list(view.columns)
        log(f"{lk}: {len(base)} anchors, {int(base.ok.sum())} ok, view rows {len(view)}")
        r = evaluate_look(base, view, AN.look_months(lk), cfg["train_from"], c2, n_perm, n_boot, seed, log)
        r.pop("_B_raw"); r["months"] = r["A"]["months"]
        res["looks"][lk] = r
    res["view_sha256"] = shas
    both = lambda tg: all(res["looks"][lk][tg]["flags"]["pass"] for lk in looks)
    res["pass"] = {"A": both("A"), "B": both("B")}; res["pass"]["candidate"] = res["pass"]["A"] or res["pass"]["B"]
    res["p_one_sided"] = {"A": max(res["looks"][lk]["A"]["increment"]["view"]["p_one_sided"] for lk in looks),
                          "B": max(res["looks"][lk]["B"]["rel_improvement"]["view"]["p_one_sided"] for lk in looks)}
    res["runtime_s"] = time.time() - t0
    json.dump(res, open(os.path.join(out_dir, "results.json"), "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    write_tables(res, os.path.join(out_dir, "tables.txt"))
    if figure:
        make_figure(res, out_dir, name)
    log(f"done {name} in {res['runtime_s']:.0f}s -> {out_dir}")
    return res


def holm_all(result_paths):
    """Holm over candidates x {A, B} from several results.json files: returns {(view, target): p_holm} on the stored one-sided p-values (each already
    the maximum over the two looks)."""
    items = []
    for p in result_paths:
        r = json.load(open(p))
        for tg in ("A", "B"):
            items.append(((r["view"], tg), r["p_one_sided"][tg]))
    adj = T3.holm([p for _, p in items])
    return {k: float(a) for (k, _), a in zip(items, adj)}


def run_baselines(out_dir, looks=("look1", "look2"), n_boot=10000, log=print):
    os.makedirs(out_dir, exist_ok=True); res = {"head": _git_head(), "looks": {}}
    for lk in looks:
        cfg = AN.LOOKS[lk]; base = AN.load_base(cfg["source"]); t0 = time.time()
        res["looks"][lk] = baselines_look(base, AN.look_months(lk), cfg["train_from"], n_boot)
        log(f"{lk} baselines {time.time() - t0:.0f}s")
    json.dump(res, open(os.path.join(out_dir, "baselines.json"), "w"), indent=1)
    L = [f"T14 harness baselines (V3 for A, H3 for B; no view), harness commit {res['head']}"]
    for lk, r in res["looks"].items():
        a, b, e = r["A"]["V3_skill_vs_AR"], r["B"], r["equality"]
        L += [f"{lk}: ok anchors {r['n_ok_anchors']}, months {r['months'][0]}..{r['months'][-1]} ({len(r['months'])})",
              f"  A: V3 skill vs AR mean {_fmt(a['mean'])} CI [{_fmt(a['ci'][0])}, {_fmt(a['ci'][1])}] months positive {a['positive']}/{a['n_months']}",
              f"  B: n {b['n']} base rate {b['base_rate']:.4f}; Brier H3 {b['brier']['H3']:.5f} H1 {b['brier']['H1']:.5f} B1 {b['brier']['B1']:.5f} B0 {b['brier']['B0']:.5f}; "
              f"H3 skill vs B1 {_fmt(b['H3_skill_vs_B1']['mean'], pct=True)} CI [{_fmt(b['H3_skill_vs_B1']['ci'][0], pct=True)}, {_fmt(b['H3_skill_vs_B1']['ci'][1], pct=True)}]; "
              f"vs H1 {_fmt(b['H3_skill_vs_H1'], pct=True)}; vs B0 {_fmt(b['H3_skill_vs_B0'], pct=True)}; AUC {b['auc_H3']:.4f}",
              f"  equality on {e['month']}: V3 max|diff| vs T11 {e['V3_max_abs_diff_vs_T11']:.2e}; H3 max|diff| vs T10 {e['H3_max_abs_diff_vs_T10']:.2e}"]
    open(os.path.join(out_dir, "baselines.txt"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    log("\n".join(L)); return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--view"); ap.add_argument("--looks", default="look1,look2"); ap.add_argument("--c2-view"); ap.add_argument("--out")
    ap.add_argument("--n-perm", type=int, default=5); ap.add_argument("--n-boot", type=int, default=10000); ap.add_argument("--baselines", action="store_true")
    ap.add_argument("--no-fig", action="store_true")
    a = ap.parse_args(argv); looks = tuple(a.looks.split(","))
    if a.baselines:
        run_baselines(a.out or os.path.join(T14, "core", "reference"), looks, a.n_boot)
    else:
        if not a.view:
            ap.error("--view or --baselines is required")
        evaluate_view(a.view, looks, a.c2_view, a.out, a.n_perm, a.n_boot, figure=not a.no_fig)


if __name__ == "__main__":
    main()
